"""Ingestion endpoints: single URL, URL batch, file upload."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, field_validator

from pipeline.api import (
    AuthDep,
    _create_job,
    _do_index,
    _enqueue,
    _finish_job,
    _ingest_options,
    _job_progress_cb,
    _job_slot,
    _require_write_target,
    _runtime,
    _save_upload_replay,
    _update_job,
    _update_job_progress,
)
from pipeline.config import Config
from pipeline.document_store import (
    DocumentStore,
)
from pipeline.gateway import Attribution
from pipeline.indexer import BgeM3Embedder
from pipeline.parsers.web import require_http_url
from pipeline.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

# ---------------------------------------------------------------------------
# Ingest URL
# ---------------------------------------------------------------------------


def _tri_state(value: str) -> bool | None:
    """A form's auto/true/false: ``auto`` (or blank) is None — the dashboard's switch decides."""
    v = value.strip().lower()
    if v in ("auto", ""):
        return None
    if v in ("true", "1", "on", "yes"):
        return True
    if v in ("false", "0", "off", "no"):
        return False
    raise HTTPException(status_code=422, detail=f"expected auto, true or false, got {value!r}")


def _attribution(job_id: str, params: dict[str, Any]) -> Attribution:
    """What the gateway records for this job's LLM calls: who asked, the job, the project."""
    return Attribution(
        task="document_conversion",
        user_id=params.get("actor"),
        trace=job_id,
        end_user=params.get("project") or _runtime.get("active_project") or None,
    )


class IngestUrlRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def _http_only(cls, v: str) -> str:
        return require_http_url(v)

    doc_type: str = "generic"
    source_version: str = ""
    supersedes: str | None = None
    skip_index: bool = False
    json_convert: bool = True
    llm_convert: bool | None = None
    validate_language: bool | None = None
    trusted: bool = False


def _run_ingest_url(
    job_id: str,
    params: dict[str, Any],
    config: Config,
    embedder: BgeM3Embedder,
    vector_store: QdrantVectorStore,
    store: DocumentStore,
) -> None:
    with _job_slot(job_id) as acquired:
        if not acquired:
            return
        from pipeline.acquisition import ingest_document

        _store = store if config.storage_backend == "database" else None
        options = _ingest_options(
            llm_convert=params.get("llm_convert"),
            validate_language=params.get("validate_language"),
            trusted=bool(params.get("trusted", False)),
        )
        _update_job_progress(job_id, current=0, total=0, message="fetching & parsing source")
        doc_id, folder = ingest_document(
            params["url"],
            config,
            url_only=True,
            doc_type=params["doc_type"],
            source_version=params["source_version"],
            supersedes=params["supersedes"],
            store=_store,
            json_convert=params.get("json_convert", True),
            **options.kwargs(),
            project=params.get("project") or _runtime.get("active_project") or None,
            attribution=_attribution(job_id, params),
        )
        chunks = 0
        if not params["skip_index"]:
            chunks = _do_index(
                folder,
                config,
                embedder,
                vector_store,
                params["supersedes"],
                store=_store,
                doc_id=doc_id,
                progress_cb=_job_progress_cb(job_id),
            )
        folder_name = folder.name if folder is not None else None
        _finish_job(job_id, result={"doc_id": doc_id, "folder": folder_name, "chunks": chunks})


@router.post("/ingest/url")
def ingest_url(
    req: IngestUrlRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
) -> dict[str, str]:
    _require_write_target(request)
    params = {**req.model_dump(), "project": _runtime.get("active_project") or None}
    job_id = _create_job("ingest_url", params)
    _enqueue(
        _run_ingest_url,
        job_id,
        params,
        request.app.state.config,
        request.app.state.embedder,
        request.app.state.vector_store,
        request.app.state.document_store,
    )
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Ingest URLs (batch)
# ---------------------------------------------------------------------------


class IngestUrlsRequest(BaseModel):
    urls: list[str]
    doc_type: str = "generic"
    source_version: str = ""
    skip_index: bool = False
    llm_convert: bool | None = None
    validate_language: bool | None = None
    trusted: bool = False


def _run_ingest_urls(
    job_id: str,
    params: dict[str, Any],
    config: Config,
    embedder: BgeM3Embedder,
    vector_store: QdrantVectorStore,
    store: DocumentStore,
) -> None:
    with _job_slot(job_id) as acquired:
        if not acquired:
            return
        counts: dict[str, int] = {"ingested": 0, "skipped": 0, "error": 0}
        url_results: list[dict[str, Any]] = []
        urls = [str(u).strip() for u in params["urls"] if str(u).strip()]
        total_urls = len(urls)
        _update_job_progress(job_id, current=0, total=total_urls, message="ingest started")
        _store = store if config.storage_backend == "database" else None
        options = _ingest_options(
            llm_convert=params.get("llm_convert"),
            validate_language=params.get("validate_language"),
            trusted=bool(params.get("trusted", False)),
        )
        from pipeline.acquisition import ingest_document

        for idx, url in enumerate(urls, start=1):
            try:
                doc_id, folder = ingest_document(
                    url,
                    config,
                    url_only=True,
                    doc_type=params["doc_type"],
                    source_version=params["source_version"],
                    store=_store,
                    **options.kwargs(),
                    project=params.get("project") or _runtime.get("active_project") or None,
                    attribution=_attribution(job_id, params),
                )
                chunks = 0
                if not params["skip_index"]:
                    chunks = _do_index(
                        folder,
                        config,
                        embedder,
                        vector_store,
                        None,
                        store=_store,
                        doc_id=doc_id,
                    )
                url_results.append(
                    {"url": url, "status": "ingested", "doc_id": doc_id[:8], "chunks": chunks}
                )
                counts["ingested"] += 1
            except Exception as exc:
                msg = str(exc)
                # Treat SHA-256 dedup ("already ingested") as skipped, not error
                if "already ingested" in msg:
                    url_results.append({"url": url, "status": "skipped"})
                    counts["skipped"] += 1
                else:
                    url_results.append({"url": url, "status": "error", "error": msg})
                    counts["error"] += 1
            _update_job_progress(
                job_id,
                current=idx,
                total=total_urls,
                message=f"processed {idx}/{total_urls} urls",
            )
        _finish_job(job_id, result={"urls": url_results, **counts})


@router.post("/ingest/urls")
def ingest_urls(
    req: IngestUrlsRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
) -> dict[str, str]:
    if not req.urls:
        raise HTTPException(status_code=422, detail="urls list is empty")
    _require_write_target(request)
    params = {**req.model_dump(), "project": _runtime.get("active_project") or None}
    job_id = _create_job("ingest_urls", params)
    _enqueue(
        _run_ingest_urls,
        job_id,
        params,
        request.app.state.config,
        request.app.state.embedder,
        request.app.state.vector_store,
        request.app.state.document_store,
    )
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Ingest file
# ---------------------------------------------------------------------------


def _run_ingest_file(
    job_id: str,
    tmp_path: str,
    params: dict[str, Any],
    config: Config,
    embedder: BgeM3Embedder,
    vector_store: QdrantVectorStore,
    store: DocumentStore,
) -> None:
    # The upload's temp directory is removed whether or not the job ran.
    try:
        with _job_slot(job_id) as acquired:
            if not acquired:
                return
            from pipeline.acquisition import ingest_document

            _store = store if config.storage_backend == "database" else None
            options = _ingest_options(
                llm_convert=params.get("llm_convert"),
                validate_language=params.get("validate_language"),
                trusted=bool(params.get("trusted", False)),
            )
            _update_job_progress(job_id, current=0, total=0, message="parsing document")
            doc_id, folder = ingest_document(
                tmp_path,
                config,
                doc_type=params["doc_type"],
                source_version=params["source_version"],
                supersedes=params["supersedes"],
                store=_store,
                json_convert=params.get("json_convert", True),
                **options.kwargs(),
                project=params.get("project") or _runtime.get("active_project") or None,
                attribution=_attribution(job_id, params),
            )
            chunks = 0
            if not params["skip_index"]:
                chunks = _do_index(
                    folder,
                    config,
                    embedder,
                    vector_store,
                    params["supersedes"],
                    store=_store,
                    doc_id=doc_id,
                    progress_cb=_job_progress_cb(job_id),
                )
            folder_name = folder.name if folder is not None else None
            _finish_job(job_id, result={"doc_id": doc_id, "folder": folder_name, "chunks": chunks})
    finally:
        shutil.rmtree(Path(tmp_path).parent, ignore_errors=True)


_UPLOAD_MAX_BYTES = int(os.environ.get("UPLOAD_MAX_BYTES", str(200 * 1024 * 1024)))


async def _read_upload_capped(file: UploadFile) -> bytes:
    """Read an upload in chunks, refusing anything over UPLOAD_MAX_BYTES (413)."""
    buf = bytearray()
    while chunk := await file.read(1024 * 1024):
        buf.extend(chunk)
        if len(buf) > _UPLOAD_MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"upload exceeds {_UPLOAD_MAX_BYTES} bytes")
    return bytes(buf)


@router.post("/ingest/file")
async def ingest_file(
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
    file: UploadFile = File(...),
    doc_type: str = Form("generic"),
    source_version: str = Form(""),
    supersedes: str | None = Form(None),
    skip_index: bool = Form(False),
    trusted: bool = Form(False),
    json_convert: bool = Form(True),
    llm_convert: str = Form("auto"),
    validate_language: str = Form("auto"),
) -> dict[str, str]:
    await asyncio.to_thread(_require_write_target, request)
    config: Config = request.app.state.config
    raw_name = file.filename or "upload"
    filename = unicodedata.normalize(
        "NFC", raw_name.encode("utf-8", errors="replace").decode("utf-8")
    )
    safe_name = Path(filename).name or "upload"
    content = await _read_upload_capped(file)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = str(Path(tmp_dir) / safe_name)
    await asyncio.to_thread(Path(tmp_path).write_bytes, content)
    _llm_convert = _tri_state(llm_convert)
    _validate_language = _tri_state(validate_language)
    params: dict[str, Any] = {
        "filename": safe_name,
        "doc_type": doc_type,
        "source_version": source_version,
        "supersedes": supersedes,
        "skip_index": skip_index,
        "trusted": trusted,
        "json_convert": json_convert,
        "llm_convert": _llm_convert,
        "validate_language": _validate_language,
        "project": _runtime.get("active_project") or None,
    }
    job_id = _create_job("ingest_file", params)
    replay_relpath = _save_upload_replay(
        config,
        job_id=job_id,
        filename=safe_name,
        content=content,
    )
    params["replay_relpath"] = replay_relpath
    _update_job(job_id, params=dict(params))

    _enqueue(
        _run_ingest_file,
        job_id,
        tmp_path,
        params,
        config,
        request.app.state.embedder,
        request.app.state.vector_store,
        request.app.state.document_store,
    )
    return {"job_id": job_id}
