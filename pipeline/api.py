"""FastAPI REST API for the KB pipeline.

Runs inside the kb-pipeline container via:
    uvicorn pipeline.api:app --host 0.0.0.0 --port 8000

IMPORTANT: always run with --workers 1. Embedded Qdrant holds a RocksDB
filesystem lock; the in-memory job store must not be split across processes.
"""

from __future__ import annotations

import asyncio
import hmac
import json as _json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from pipeline.acquisition import IngestOptions
from pipeline.config import Config, load_config, warn_legacy_env
from pipeline.document_store import (
    AmbiguousDocPrefixError,
    DocumentStore,
    InvalidDocPrefixError,
    make_document_store,
)
from pipeline.indexer import PipelineEmbedder, make_embedder, make_fts_db
from pipeline.prompt_config import (
    configure_prompt_session_factory,
)
from pipeline.runtime_config import get_runtime_config

# Re-exported: routers construct stores via core.QdrantVectorStore (patchable in tests).
from pipeline.vectorstore import QdrantVectorStore as QdrantVectorStore
from pipeline.vectorstore import make_vector_store

if TYPE_CHECKING:
    from pipeline.gateway import Gateway
    from pipeline.reranker import Reranker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
# Guards _jobs: background threads mutate entries while request handlers read them.
_jobs_lock = threading.RLock()
_ACTIVE_JOB_STATES = frozenset({"pending", "running"})
_mutation_lock = threading.Lock()
# A queued job waits this long for the mutation lock (held by the dropzone watcher or a
# project switch) before failing.
_JOB_LOCK_TIMEOUT_SECS: float = max(0.0, float(os.environ.get("JOB_LOCK_TIMEOUT_SECS", "3600")))
_job_pool: ThreadPoolExecutor | None = None
_job_pool_lock = threading.Lock()


def _job_executor() -> Executor:
    """The single job worker: mutating jobs run one after another, in submission order.

    A dedicated thread, not Starlette's BackgroundTasks: those share the threadpool that
    serves synchronous endpoints (/query), so waiting jobs would starve searches.
    """
    global _job_pool
    with _job_pool_lock:
        if _job_pool is None:
            _job_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kb-job")
        return _job_pool


def _enqueue(fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """Queue a job runner (FIFO); its job record stays "pending" until it starts."""
    _job_executor().submit(fn, *args, **kwargs)


def _shutdown_jobs() -> None:
    global _job_pool
    with _job_pool_lock:
        if _job_pool is not None:
            _job_pool.shutdown(wait=False, cancel_futures=True)
            _job_pool = None
_MAX_JOBS = 100
_job_session_factory: Any | None = None
_JOB_REPLAYS_DIRNAME = "job_replays"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _job_replays_root(config: Config) -> Path:
    return config.kb_root / _JOB_REPLAYS_DIRNAME


def _save_upload_replay(
    config: Config,
    *,
    job_id: str,
    filename: str,
    content: bytes,
) -> str:
    """Persist uploaded bytes so file-upload jobs can be restarted."""
    root = _job_replays_root(config)
    job_dir = root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = Path(filename).name or "upload.bin"
    artifact_path = job_dir / artifact_name
    artifact_path.write_bytes(content)
    return str(artifact_path.relative_to(config.kb_root))


def _resolve_replay_path(config: Config, replay_relpath: str) -> Path:
    root = config.kb_root.resolve()
    candidate = (config.kb_root / replay_relpath).resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(status_code=400, detail="invalid replay artifact path")
    return candidate


def _materialize_replay_to_tmp(config: Config, params: dict[str, Any]) -> str:
    relpath = str(params.get("replay_relpath") or "").strip()
    if not relpath:
        raise HTTPException(status_code=400, detail="job is missing replay artifact metadata")
    artifact_path = _resolve_replay_path(config, relpath)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="replay artifact not found")

    filename = Path(str(params.get("filename") or "upload")).name or "upload"
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / filename
    shutil.copy2(artifact_path, tmp_path)
    return str(tmp_path)


def _delete_job_replay_artifact(config: Config, params: dict[str, Any]) -> None:
    relpath = str(params.get("replay_relpath") or "").strip()
    if not relpath:
        return
    try:
        artifact_path = _resolve_replay_path(config, relpath)
    except HTTPException:
        return
    if artifact_path.parent.name:
        shutil.rmtree(artifact_path.parent, ignore_errors=True)


def _job_snapshot() -> list[dict[str, Any]]:
    """Copies of all in-memory jobs, safe to iterate without the lock."""
    with _jobs_lock:
        return [dict(j) for j in _jobs.values()]


@contextmanager
def _exclusive(detail: str = "a job is queued or running") -> Iterator[None]:
    """Hold the mutation lock for a synchronous destructive operation, else 409."""
    if _job_busy() or not _mutation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=detail)
    try:
        yield
    finally:
        _mutation_lock.release()


@contextmanager
def _job_slot(job_id: str) -> Iterator[bool]:
    """Run a background job under the mutation lock.

    Waits up to ``_JOB_LOCK_TIMEOUT_SECS`` for the lock (a dropzone scan or a project
    switch may hold it); past that it yields ``False`` after recording the job as
    failed. Otherwise marks the job running, records any exception escaping the body
    as the job's error, and always releases the lock. The body is responsible for
    calling ``_finish_job`` with its result.
    """
    if not _mutation_lock.acquire(timeout=_JOB_LOCK_TIMEOUT_SECS):
        _finish_job(job_id, error="another job is already running")
        yield False
        return
    try:
        _set_running(job_id)
        yield True
    except Exception as exc:
        _finish_job(job_id, error=str(exc))
    finally:
        _mutation_lock.release()


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _job_busy() -> bool:
    """True while any job is queued or running (mutating operations must wait)."""
    return any(j["status"] in _ACTIVE_JOB_STATES for j in _job_snapshot())


def _create_job(job_type: str, params: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    # Who asked: captured now, while the request's context is at hand (runners run later).
    # Set in the caller's dict: routes keep using it (the upload route adds to it and hands
    # it to its runner).
    actor = _current_actor.get()
    if actor:
        params.setdefault("actor", actor)
    job = {
        "job_id": job_id,
        "type": job_type,
        "status": "pending",
        "created_at": _now_iso(),
        "finished_at": None,
        "params": params,
        "result": None,
        "error": None,
        "progress": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job
        # Evict the oldest *finished* jobs only — evicting a live job would make
        # _finish_job a no-op and strand its DB row in "running" forever.
        finished = sorted(
            (k for k, j in _jobs.items() if j["status"] not in _ACTIVE_JOB_STATES),
            key=lambda k: str(_jobs[k]["created_at"]),
        )
        for k in finished[: max(0, len(_jobs) - _MAX_JOBS)]:
            del _jobs[k]
        snapshot = dict(job)
    _persist_job(snapshot)
    return job_id


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(fields)
        snapshot = dict(job)
    _persist_job(snapshot)


def _finish_job(
    job_id: str, *, result: dict[str, Any] | None = None, error: str | None = None
) -> None:
    _update_job(
        job_id,
        status="done" if error is None else "error",
        finished_at=_now_iso(),
        result=result,
        error=error,
    )


def _set_running(job_id: str) -> None:
    _update_job(job_id, status="running")


def _fail_stale_jobs() -> int:
    """Mark DB jobs left pending/running by a previous process as failed."""
    if _job_session_factory is None:
        return 0
    try:
        from sqlalchemy import update

        from pipeline.db_models import KbJob
        from pipeline.document_store import _session_scope

        with _session_scope(_job_session_factory) as s:
            result = s.execute(
                update(KbJob)
                .where(KbJob.status.in_(sorted(_ACTIVE_JOB_STATES)))
                .values(status="error", error="interrupted by restart", finished_at=_now_iso())
            )
            return int(getattr(result, "rowcount", 0) or 0)
    except Exception as exc:
        logger.warning("Could not mark stale jobs: %s", exc)
        return 0


def _update_job_progress(
    job_id: str,
    *,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        progress = job.get("progress")
        progress = dict(progress) if isinstance(progress, dict) else {}

        if current is not None:
            progress["current"] = max(0, int(current))
        if total is not None:
            progress["total"] = max(0, int(total))
        if message is not None:
            progress["message"] = message

        cur = progress.get("current")
        tot = progress.get("total")
        if isinstance(cur, int) and isinstance(tot, int) and tot > 0:
            progress["percent"] = min(100, max(0, int((cur * 100) / tot)))
        else:
            progress["percent"] = None

        job["progress"] = progress
        snapshot = dict(job)
    _persist_job(snapshot)


def _job_progress_cb(job_id: str) -> Callable[[int, int, str], None]:
    """Return a (current, total, message) callback that updates the job's progress."""

    def _cb(current: int, total: int, message: str) -> None:
        _update_job_progress(job_id, current=current, total=total, message=message)

    return _cb


def _persist_job(job: dict[str, Any]) -> None:
    if _job_session_factory is None:
        return
    try:
        from pipeline.db_models import KbJob
        from pipeline.document_store import _session_scope

        with _session_scope(_job_session_factory) as s:
            existing = s.get(KbJob, str(job.get("job_id", "")))
            params_json = _json.dumps(job.get("params") or {}, ensure_ascii=False)
            result_val = job.get("result")
            progress_val = job.get("progress")
            result_json = (
                None if result_val is None else _json.dumps(result_val, ensure_ascii=False)
            )
            progress_json = (
                None if progress_val is None else _json.dumps(progress_val, ensure_ascii=False)
            )

            if existing is None:
                s.add(
                    KbJob(
                        job_id=str(job.get("job_id", "")),
                        type=str(job.get("type", "")),
                        status=str(job.get("status", "pending")),
                        created_at=str(job.get("created_at", _now_iso())),
                        finished_at=(
                            None if job.get("finished_at") is None else str(job.get("finished_at"))
                        ),
                        params_json=params_json,
                        result_json=result_json,
                        progress_json=progress_json,
                        error=(None if job.get("error") is None else str(job.get("error"))),
                    )
                )
                return

            existing.type = str(job.get("type", existing.type))
            existing.status = str(job.get("status", existing.status))
            existing.created_at = str(job.get("created_at", existing.created_at))
            existing.finished_at = (
                None if job.get("finished_at") is None else str(job.get("finished_at"))
            )
            existing.params_json = params_json
            existing.result_json = result_json
            existing.progress_json = progress_json
            existing.error = None if job.get("error") is None else str(job.get("error"))
    except Exception as exc:
        logger.warning("Job persistence failed for %s: %s", str(job.get("job_id", ""))[:8], exc)


def _job_from_row(row: Any) -> dict[str, Any]:
    params_raw = row.params_json or "{}"
    result_raw = row.result_json
    progress_raw = row.progress_json
    return {
        "job_id": str(row.job_id),
        "type": str(row.type),
        "status": str(row.status),
        "created_at": str(row.created_at),
        "finished_at": row.finished_at,
        "params": _json.loads(params_raw),
        "result": None if result_raw is None else _json.loads(result_raw),
        "error": row.error,
        "progress": None if progress_raw is None else _json.loads(progress_raw),
    }


def _load_job_from_db(job_id: str) -> dict[str, Any] | None:
    if _job_session_factory is None:
        return None
    try:
        from pipeline.db_models import KbJob
        from pipeline.document_store import _session_scope

        with _session_scope(_job_session_factory) as s:
            row = s.get(KbJob, job_id)
            if row is None:
                return None
            return _job_from_row(row)
    except Exception as exc:
        logger.warning("Job lookup failed for %s: %s", job_id[:8], exc)
        return None


def _list_jobs_from_db(limit: int = 50) -> list[dict[str, Any]]:
    if _job_session_factory is None:
        return []
    try:
        from sqlalchemy import desc, select

        from pipeline.db_models import KbJob
        from pipeline.document_store import _session_scope

        with _session_scope(_job_session_factory) as s:
            rows = (
                s.execute(select(KbJob).order_by(desc(KbJob.created_at)).limit(limit))
                .scalars()
                .all()
            )
            return [_job_from_row(row) for row in rows]
    except Exception as exc:
        logger.warning("Job list query failed: %s", exc)
        return []


def _delete_job_from_db(job_id: str) -> bool:
    if _job_session_factory is None:
        return False
    try:
        from sqlalchemy import delete

        from pipeline.db_models import KbJob
        from pipeline.document_store import _session_scope

        with _session_scope(_job_session_factory) as s:
            res = s.execute(delete(KbJob).where(KbJob.job_id == job_id))
            return int(res.rowcount or 0) > 0
    except Exception as exc:
        logger.warning("Job delete failed for %s: %s", job_id[:8], exc)
        return False


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_PIPELINE_API_SECRET: str = os.environ.get("PIPELINE_API_SECRET", "")
_DROPZONE_INTERVAL: int = max(0, int(os.environ.get("DROPZONE_INTERVAL_SECS", "10")))


def require_api_key(
    api_key: Annotated[str | None, Depends(_API_KEY_HEADER)],
) -> None:
    # Fail closed: without a configured secret nothing but /health answers.
    if not _PIPELINE_API_SECRET:
        raise HTTPException(
            status_code=503,
            detail="PIPELINE_API_SECRET is not set; the pipeline API refuses requests",
        )
    if not hmac.compare_digest((api_key or "").encode(), _PIPELINE_API_SECRET.encode()):
        raise HTTPException(status_code=403, detail="Invalid API key")


# The person a request acts for: the UI's X-KB-Actor (pacod-id's usr_ id), trusted only
# from a caller that holds the API key. Kept for the request (and the jobs it queues).
_ACTOR_RE = re.compile(r"^[a-z]{2,4}_[A-Za-z0-9]{1,64}$")
_current_actor: ContextVar[str | None] = ContextVar("kb_actor", default=None)


async def _authenticate(
    api_key: Annotated[str | None, Depends(_API_KEY_HEADER)],
    x_kb_actor: Annotated[str | None, Header(include_in_schema=False)] = None,
) -> None:
    # Async on purpose: the context variable set here is the endpoint's (a sync
    # dependency would run in a worker thread and keep it to itself).
    require_api_key(api_key)
    if x_kb_actor is not None and not _ACTOR_RE.fullmatch(x_kb_actor):
        raise HTTPException(status_code=400, detail="malformed X-KB-Actor")
    _current_actor.set(x_kb_actor)


def request_actor() -> str | None:
    """Who the current request acts for (None: no person, e.g. the CLI or a script)."""
    return _current_actor.get()


AuthDep = Annotated[None, Depends(_authenticate)]

# ---------------------------------------------------------------------------
# Runtime settings (mutable at runtime; initialised from config in lifespan)
# ---------------------------------------------------------------------------

_runtime: dict[str, Any] = {}

# Runtime settings, prompts, usage and AI interactions (pipeline/ledger.py);
# set in the lifespan, in both storage backends.
_ledger_factory: Any | None = None


def _security_bypass() -> bool:
    """Return True when the global security switch is OFF (scan should be skipped)."""
    return not _runtime.get("content_security_enabled", True)


def _ingest_options(
    *,
    llm_convert: bool | None = None,
    validate_language: bool | None = None,
    trusted: bool = False,
) -> IngestOptions:
    """The dashboard's switches for one ingest, with a request's explicit overrides.

    Used by every API-side ingest path, the dropzone watcher and watch, so turning
    the converter or the security scan off applies everywhere.
    """
    return IngestOptions(
        llm_convert=(
            llm_convert
            if llm_convert is not None
            else bool(_runtime.get("llm_converter_enabled", False))
        ),
        trusted=trusted or _security_bypass(),
        validate_language=(
            validate_language
            if validate_language is not None
            else bool(_runtime.get("language_validation_enabled", True))
        ),
        llm_runtime_config=get_runtime_config(_ledger_factory),
    )


def _collection_owner(app: FastAPI, collection: str) -> str | None:
    """Slug of the project whose Qdrant collection this is; None when unowned."""
    registry = getattr(app.state, "project_registry", None)
    if registry is None:
        return None
    return next((p.slug for p in registry.list() if p.collection_name == collection), None)


def _write_target_error(app: FastAPI) -> str | None:
    """Why ingest must not write now, or None.

    Writes pair ``app.state.vector_store`` with ``_runtime["active_project"]``. While
    the active collection belongs to no project (selected directly), vectors and
    keyword rows would land in different places, so writes wait for a project.
    """
    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None or getattr(app.state, "project_registry", None) is None:
        return None
    if _collection_owner(app, vector_store.collection_name) is None:
        return (
            f"the active collection '{vector_store.collection_name}' belongs to no project; "
            "select a project before ingesting"
        )
    return None


def _require_write_target(request: Request) -> None:
    """409 while the active collection belongs to no project (see _write_target_error)."""
    error = _write_target_error(request.app)
    if error is not None:
        raise HTTPException(status_code=409, detail=error)


def _reranker_factory(config: Config) -> Callable[[], Reranker]:
    def build() -> Reranker:
        from pipeline.reranker import CrossEncoderReranker, RemoteReranker

        if config.reranker_url:
            return RemoteReranker(config.reranker_url, timeout_s=config.reranker_timeout_ms / 1000)

        return CrossEncoderReranker(
            config.reranker_model,
            max_length=config.rerank_max_length,
            batch_size=config.rerank_batch_size,
            device=None if config.inference_device == "auto" else config.inference_device,
            revision=config.reranker_revision or None,
            trust_remote_code=config.reranker_trust_remote_code,
            code_revision=config.reranker_code_revision or None,
            gate_timeout_s=config.rerank_gate_timeout_ms / 1000,
        )

    return build


def _ensure_reranker(app: FastAPI) -> None:
    """Start loading the reranker in the background (no-op while loading or ready).

    Called at startup (RERANKER_ENABLED) and by POST /settings only, never by a search:
    loading costs gigabytes of RAM, and a failed load is retried only on request.
    """
    holder = getattr(app.state, "reranker", None)
    if holder is not None and holder.state in ("disabled", "failed"):
        holder.start(_reranker_factory(app.state.config))


def _switch_active_project(slug: str, app: FastAPI) -> None:
    """Atomically swap the active project and its Qdrant collection."""
    from pipeline.projects import ProjectStore

    registry: ProjectStore = app.state.project_registry
    project = registry.get(slug)
    collection_name = project.collection_name if project is not None else slug
    current: QdrantVectorStore = app.state.vector_store
    app.state.vector_store = QdrantVectorStore(current.client, collection_name)
    _runtime["active_project"] = slug
    _runtime["active_collection"] = collection_name


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _job_session_factory, _ledger_factory

    config: Config = load_config()
    warn_legacy_env()
    if not _PIPELINE_API_SECRET:
        logger.error(
            "PIPELINE_API_SECRET is empty — the pipeline API refuses every request "
            "except /health. Set it in .env (make setup generates one)."
        )
    embedder: PipelineEmbedder = make_embedder(config)

    # In server mode, qdrant may not be ready yet — retry with backoff.
    _vs_exc: Exception | None = None
    vector_store: QdrantVectorStore | None = None
    for _attempt in range(10):
        try:
            vector_store = make_vector_store(config)
            break
        except Exception as exc:
            _vs_exc = exc
            if config.qdrant_mode != "server":
                raise
            logger.warning(
                "Qdrant not reachable (attempt %d/10), retrying in 5 s: %s",
                _attempt + 1,
                exc,
            )
            await asyncio.sleep(5)  # never block the event loop during startup
    if vector_store is None:
        raise RuntimeError("Could not connect to Qdrant after 10 attempts") from _vs_exc
    document_store = make_document_store(config)
    _job_session_factory = getattr(document_store, "_factory", None)
    stale = _fail_stale_jobs()
    if stale:
        logger.warning("Marked %d job(s) interrupted by the last shutdown as failed", stale)
    from pipeline.ledger import ledger_session_factory

    _ledger_factory = ledger_session_factory(config, document_store)
    app.state.ledger_factory = _ledger_factory
    configure_prompt_session_factory(_ledger_factory)
    from pipeline.projects import make_project_registry

    project_registry = make_project_registry(config, _job_session_factory)
    default_proj = project_registry.ensure_default(config.qdrant_collection)
    _runtime["content_security_enabled"] = config.content_security_enabled
    _runtime["llm_converter_enabled"] = config.llm_converter_enabled
    _runtime["language_validation_enabled"] = config.language_validation_enabled
    _runtime["storage_backend"] = config.storage_backend
    _runtime["reranker_enabled"] = config.reranker_enabled
    from pipeline.gateway import get_gateway
    from pipeline.reranker import RerankerHolder

    app.state.config = config
    app.state.reranker = RerankerHolder(
        f"{config.reranker_url}/rerank" if config.reranker_url else config.reranker_model
    )
    if config.reranker_enabled:
        _ensure_reranker(app)  # loads in the background; searches use RRF until ready
    app.state.gateway = get_gateway(config)
    if not app.state.gateway.configured():
        logger.info("AI gateway not set up (PACOD_GW_URL / PACOD_GW_API_KEY): AI features are off")
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.document_store = document_store
    app.state.project_registry = project_registry
    _switch_active_project(default_proj.slug, app)

    from pipeline.projects import retag_legacy_default_project

    _fts = make_fts_db(config)
    try:
        retag_legacy_default_project(
            project_registry, default_proj.slug, document_store, _fts, app.state.vector_store
        )
    except Exception as exc:
        logger.warning("Legacy project re-tag failed (continuing): %s", exc)
    finally:
        _fts.close()

    def _dropzone_record_results(results: list[dict[str, Any]]) -> None:
        counts: dict[str, int] = {"ingested": 0, "skipped": 0, "error": 0}
        for r in results:
            s = str(r.get("status", "error"))
            counts[s] = counts.get(s, 0) + 1
        jid = _create_job("dropzone_scan", {"source": "auto"})
        _finish_job(jid, result={"files": results, "source": "auto", **counts})

    def _dropzone_record_error(error: str) -> None:
        jid = _create_job("dropzone_scan", {"source": "auto"})
        _finish_job(jid, error=error)

    _dropzone_stop = threading.Event()
    if _DROPZONE_INTERVAL > 0:
        from pipeline.dropzone import watch_dropzone_loop

        threading.Thread(
            target=watch_dropzone_loop,
            args=(
                config,
                embedder,
                vector_store,
                _dropzone_stop,
                _mutation_lock,
                _DROPZONE_INTERVAL,
                _dropzone_record_results,
                _dropzone_record_error,
            ),
            kwargs={
                # Same rule as every other API path: the store object is only
                # passed in database mode (filesystem mode uses the file ops,
                # including git autocommit).
                "store": document_store if config.storage_backend == "database" else None,
                # Read the active project on every scan so a project switch
                # after startup redirects auto-ingest to the right collection.
                # None (skip) while the active collection belongs to no project.
                "target": lambda: (
                    None
                    if _write_target_error(app) is not None
                    else (app.state.vector_store, _runtime.get("active_project") or None)
                ),
                # The dashboard's converter/security switches, read on every scan.
                "options": _ingest_options,
            },
            name="dropzone-watcher",
            daemon=True,
        ).start()

    yield

    _dropzone_stop.set()
    _shutdown_jobs()
    configure_prompt_session_factory(None)
    _job_session_factory = None
    _ledger_factory = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="KB Pipeline API", lifespan=_lifespan)


@app.exception_handler(InvalidDocPrefixError)
async def _invalid_prefix_handler(_request: Request, exc: InvalidDocPrefixError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(AmbiguousDocPrefixError)
async def _ambiguous_prefix_handler(
    _request: Request, exc: AmbiguousDocPrefixError
) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    from pipeline.acquisition import SUPPORTED_LANGUAGES

    config: Config = request.app.state.config
    holder = getattr(request.app.state, "reranker", None)
    return {
        "status": "ok",
        "mode": config.qdrant_mode,
        "supported_languages": sorted(SUPPORTED_LANGUAGES - {"unknown"}),
        "reranker": holder.status() if holder is not None else {"state": "disabled"},
    }


# ---------------------------------------------------------------------------
# Shared background-task helpers
# ---------------------------------------------------------------------------


def _do_index(
    folder: Path | None,
    config: Config,
    embedder: PipelineEmbedder,
    vector_store: QdrantVectorStore,
    supersedes: str | None,
    *,
    store: DocumentStore | None = None,
    doc_id: str | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> int:
    """Embed and index a document, optionally superseding a prior version. Returns chunk count."""
    from pipeline.indexer import index_document

    fts_db = make_fts_db(config)
    try:
        if folder is None and store is not None and doc_id is not None:
            manifest = store.load_manifest(doc_id)
            body = store.load_body(doc_id)
            chunks = index_document(
                None,
                config,
                embedder,
                vector_store,
                fts_db,
                doc_data=(manifest, body),
                progress_cb=progress_cb,
            )
            store.update_manifest(doc_id, chunk_count=len(chunks))
        else:
            chunks = index_document(
                folder,
                config,
                embedder,
                vector_store,
                fts_db,
                progress_cb=progress_cb,
            )
        if supersedes and chunks:
            from pipeline.versioning import supersede_document

            today = datetime.now(UTC).date().isoformat()
            supersede_document(supersedes, today, config, vector_store, fts_db, store=store)
        elif supersedes:
            # Quarantined (status=review) or empty documents produce no chunks;
            # superseding would leave the source with no searchable version.
            logger.warning(
                "Not superseding %s: new version produced no indexed chunks", supersedes[:8]
            )
    finally:
        fts_db.close()
    return len(chunks)


def _gateway_from_app(request: Request) -> Gateway:
    """The process's AI gateway client (set in the lifespan)."""
    gateway = getattr(request.app.state, "gateway", None)
    if gateway is None:
        from pipeline.gateway import get_gateway

        gateway = get_gateway(request.app.state.config)
    return cast("Gateway", gateway)


def _ledger_factory_from_app(request: Request) -> Any | None:
    """The ledger's session factory (settings, prompts, usage, interactions)."""
    return getattr(request.app.state, "ledger_factory", None)


# ---------------------------------------------------------------------------
# Routers (included last: they import the helpers defined above)
# ---------------------------------------------------------------------------

from pipeline.api_routes import (  # noqa: E402
    ai_interactions,
    documents,
    ingest,
    jobs,
    maintenance,
    people,
    project_tasks,
    projects,
    query,
    settings,
    usage,
)

app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(maintenance.router)
app.include_router(documents.router)
app.include_router(projects.router)
app.include_router(jobs.router)
app.include_router(settings.router)
app.include_router(usage.router)
app.include_router(ai_interactions.router)
app.include_router(project_tasks.router)
app.include_router(people.router)

# Backwards-compatible re-exports (tests and external importers).
from pipeline.api_routes.ingest import (  # noqa: E402
    IngestUrlRequest as IngestUrlRequest,
)
from pipeline.api_routes.ingest import (
    IngestUrlsRequest as IngestUrlsRequest,
)
from pipeline.api_routes.jobs import (  # noqa: E402
    _restart_job as _restart_job,
)
from pipeline.api_routes.maintenance import (  # noqa: E402
    _queue_rebuild as _queue_rebuild,
)
from pipeline.api_routes.maintenance import (
    _rebuild_params as _rebuild_params,
)
from pipeline.api_routes.project_tasks import (  # noqa: E402
    _run_convert_without_index as _run_convert_without_index,
)
from pipeline.api_routes.project_tasks import (
    _run_regenerate_from_vector_db as _run_regenerate_from_vector_db,
)
from pipeline.api_routes.query import (  # noqa: E402
    QueryRequest as QueryRequest,
)
from pipeline.api_routes.settings import (  # noqa: E402
    SettingsRequest as SettingsRequest,
)
