"""Index maintenance jobs: reindex, rebuild, watch, dropzone scan."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
)

from pipeline.api import (
    AuthDep,
    _create_job,
    _enqueue,
    _finish_job,
    _ingest_options,
    _job_busy,
    _job_progress_cb,
    _job_slot,
    _require_write_target,
    _runtime,
    _update_job_progress,
)
from pipeline.config import Config
from pipeline.document_store import (
    DocumentStore,
)
from pipeline.indexer import BgeM3Embedder, make_fts_db
from pipeline.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

# ---------------------------------------------------------------------------
# Reindex
# ---------------------------------------------------------------------------


def _run_reindex(
    job_id: str,
    doc_id: str,
    config: Config,
    embedder: BgeM3Embedder,
    vector_store: QdrantVectorStore,
    store: DocumentStore,
) -> None:
    with _job_slot(job_id) as acquired:
        if not acquired:
            return
        _store = store if config.storage_backend == "database" else None
        from pipeline.indexer import index_document

        fts_db = make_fts_db(config)
        try:
            if _store is not None:
                full_id = _store.find_by_prefix(doc_id) or doc_id
                manifest = _store.load_manifest(full_id)
                body = _store.load_body(full_id)
                chunks = index_document(
                    None,
                    config,
                    embedder,
                    vector_store,
                    fts_db,
                    doc_data=(manifest, body),
                    progress_cb=_job_progress_cb(job_id),
                )
                _store.update_manifest(full_id, chunk_count=len(chunks))
                _finish_job(job_id, result={"doc_id": full_id[:8], "chunks": len(chunks)})
            else:
                from pipeline.versioning import find_doc_folder

                folder = find_doc_folder(doc_id, config, prefix=True)
                if folder is None:
                    _finish_job(job_id, error=f"document '{doc_id}' not found")
                    return
                chunks = index_document(
                    folder,
                    config,
                    embedder,
                    vector_store,
                    fts_db,
                    progress_cb=_job_progress_cb(job_id),
                )
                _finish_job(job_id, result={"folder": folder.name, "chunks": len(chunks)})
        finally:
            fts_db.close()


@router.post("/reindex/{doc_id}")
def reindex(
    doc_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
) -> dict[str, str]:
    _require_write_target(request)
    job_id = _create_job("reindex", {"doc_id": doc_id})
    _enqueue(
        _run_reindex,
        job_id,
        doc_id,
        request.app.state.config,
        request.app.state.embedder,
        request.app.state.vector_store,
        request.app.state.document_store,
    )
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------


def _run_rebuild(
    job_id: str,
    config: Config,
    embedder: BgeM3Embedder,
    vector_store: QdrantVectorStore,
    store: DocumentStore,
    project: str | None = None,
    all_projects: bool = False,
    registry: Any = None,
) -> None:
    with _job_slot(job_id) as acquired:
        if not acquired:
            return
        _update_job_progress(job_id, current=0, total=0, message="rebuild started")
        _store = store if config.storage_backend == "database" else None
        from pipeline.versioning import rebuild_all_projects, rebuild_index

        def _progress(cur: int, total: int, msg: str) -> None:
            _update_job_progress(job_id, current=cur, total=total, message=msg)

        projects = registry.list() if registry is not None else []
        default_slug = projects[0].slug if projects else None
        fts_db = make_fts_db(config)
        try:
            if all_projects and registry is not None:
                ok, err = rebuild_all_projects(
                    config,
                    embedder,
                    vector_store.client,
                    fts_db,
                    registry,
                    store=_store,
                    progress_cb=_progress,
                )
            else:
                ok, err = rebuild_index(
                    config,
                    embedder,
                    vector_store,
                    fts_db,
                    store=_store,
                    progress_cb=_progress,
                    project=project,
                    default_project=default_slug,
                )
        finally:
            fts_db.close()
        _finish_job(job_id, result={"ok": ok, "errors": err})


@router.post("/rebuild")
def rebuild(
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
    all_projects: bool = False,
) -> dict[str, str]:
    """Rebuild the active project's index, or every project with ``?all_projects=true``."""
    if _job_busy():
        raise HTTPException(status_code=409, detail="another job is already running")
    params = _rebuild_params({"all_projects": all_projects})
    job_id = _create_job("rebuild", params)
    _queue_rebuild(background_tasks, request, job_id, params)
    return {"job_id": job_id}


def _rebuild_params(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve rebuild scope; never allow an unscoped (all-into-one) rebuild.

    Jobs recorded before per-project rebuilds have no ``project`` param, so
    they fall back to the active project.
    """
    all_projects = bool(params.get("all_projects", False))
    project = params.get("project") or _runtime.get("active_project") or None
    if not all_projects and not project:
        raise HTTPException(
            status_code=409, detail="no active project — select a project before rebuilding"
        )
    return {"project": project, "all_projects": all_projects}


def _queue_rebuild(
    background_tasks: BackgroundTasks, request: Request, job_id: str, params: dict[str, Any]
) -> None:
    _enqueue(
        _run_rebuild,
        job_id,
        request.app.state.config,
        request.app.state.embedder,
        request.app.state.vector_store,
        request.app.state.document_store,
        params.get("project"),
        bool(params.get("all_projects", False)),
        request.app.state.project_registry,
    )


# ---------------------------------------------------------------------------
# Watch
# ---------------------------------------------------------------------------


def _run_watch(
    job_id: str,
    watch_file: str | None,
    config: Config,
    embedder: BgeM3Embedder,
    vector_store: QdrantVectorStore,
    store: DocumentStore,
    project: str | None = None,
) -> None:
    with _job_slot(job_id) as acquired:
        if not acquired:
            return
        _update_job_progress(job_id, current=0, total=0, message="watch started")
        _store = store if config.storage_backend == "database" else None
        from pipeline.watch import watch_sources

        wf = Path(watch_file) if watch_file else config.kb_root / "watch.json"
        if not wf.resolve().is_relative_to(config.kb_root.resolve()):
            raise ValueError("watch_file must be inside KB_ROOT")
        fts_db = make_fts_db(config)
        try:
            results = watch_sources(
                config,
                wf,
                embedder,
                vector_store,
                fts_db,
                store=_store,
                progress_cb=lambda cur, total, msg: _update_job_progress(
                    job_id,
                    current=cur,
                    total=total,
                    message=msg,
                ),
                project=project,
                options=_ingest_options,
            )
        finally:
            fts_db.close()
        _finish_job(job_id, result={"sources": results})


@router.post("/watch")
def watch(
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
    watch_file: str | None = None,
) -> dict[str, str]:
    _require_write_target(request)
    project = _runtime.get("active_project") or None
    job_id = _create_job("watch", {"watch_file": watch_file, "project": project})
    _enqueue(
        _run_watch,
        job_id,
        watch_file,
        request.app.state.config,
        request.app.state.embedder,
        request.app.state.vector_store,
        request.app.state.document_store,
        project,
    )
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Dropzone scan
# ---------------------------------------------------------------------------


def _run_dropzone_scan(
    job_id: str,
    config: Config,
    embedder: BgeM3Embedder,
    vector_store: QdrantVectorStore,
    store: DocumentStore,
    project: str | None = None,
) -> None:
    with _job_slot(job_id) as acquired:
        if not acquired:
            return
        _update_job_progress(job_id, current=0, total=0, message="dropzone scan started")
        _store = store if config.storage_backend == "database" else None
        from pipeline.dropzone import scan_dropzone

        fts_db = make_fts_db(config)
        try:
            results = scan_dropzone(
                config,
                embedder,
                vector_store,
                fts_db,
                store=_store,
                progress_cb=lambda cur, total, msg: _update_job_progress(
                    job_id,
                    current=cur,
                    total=total,
                    message=msg,
                ),
                project=project,
                options=_ingest_options,
            )
        finally:
            fts_db.close()
        counts: dict[str, int] = {"ingested": 0, "skipped": 0, "error": 0}
        for r in results:
            s = str(r.get("status", "error"))
            counts[s] = counts.get(s, 0) + 1
        _finish_job(job_id, result={"files": results, "source": "manual", **counts})


@router.post("/dropzone/scan")
def scan_dropzone_api(
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
) -> dict[str, str]:
    _require_write_target(request)
    project = _runtime.get("active_project") or None
    job_id = _create_job("dropzone_scan", {"source": "manual", "project": project})
    _enqueue(
        _run_dropzone_scan,
        job_id,
        request.app.state.config,
        request.app.state.embedder,
        request.app.state.vector_store,
        request.app.state.document_store,
        project,
    )
    return {"job_id": job_id}
