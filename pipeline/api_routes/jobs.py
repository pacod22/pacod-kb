"""Job history, restart and deletion."""

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
    _delete_job_from_db,
    _delete_job_replay_artifact,
    _enqueue,
    _get_job,
    _job_busy,
    _job_snapshot,
    _jobs,
    _jobs_lock,
    _list_jobs_from_db,
    _load_job_from_db,
    _materialize_replay_to_tmp,
    _save_upload_replay,
    _update_job,
)
from pipeline.api_routes.ingest import (
    _run_ingest_file,
    _run_ingest_url,
    _run_ingest_urls,
)
from pipeline.api_routes.maintenance import (
    _queue_rebuild,
    _rebuild_params,
    _run_dropzone_scan,
    _run_reindex,
    _run_watch,
)

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


_RESTARTABLE_JOB_TYPES = frozenset(
    {"ingest_file", "ingest_url", "ingest_urls", "reindex", "rebuild", "watch", "dropzone_scan"}
)


def _restart_job(
    job: dict[str, Any],
    *,
    background_tasks: BackgroundTasks,
    request: Request,
) -> str:
    job_type = str(job.get("type", ""))
    params = dict(job.get("params") or {})
    if job_type not in _RESTARTABLE_JOB_TYPES:
        raise HTTPException(status_code=400, detail=f"job type '{job_type}' cannot be restarted")
    if job_type == "reindex" and not str(params.get("doc_id", "")).strip():
        raise HTTPException(status_code=422, detail="job params missing doc_id")
    if job_type == "rebuild":
        params = _rebuild_params(params)
    # Validate/materialise inputs *before* creating the job so a failure here
    # cannot leave an orphaned "pending" job that blocks every later mutation.
    tmp_path = (
        _materialize_replay_to_tmp(request.app.state.config, params)
        if job_type == "ingest_file"
        else ""
    )
    job_id = _create_job(job_type, params)
    if job_type == "ingest_file":
        # Give the new job its own replay copy: deleting the original job
        # removes the original artifact.
        params["replay_relpath"] = _save_upload_replay(
            request.app.state.config,
            job_id=job_id,
            filename=str(params.get("filename") or "upload"),
            content=Path(tmp_path).read_bytes(),
        )
        _update_job(job_id, params=dict(params))
        _enqueue(
            _run_ingest_file,
            job_id,
            tmp_path,
            params,
            request.app.state.config,
            request.app.state.embedder,
            request.app.state.vector_store,
            request.app.state.document_store,
        )
        return job_id

    if job_type == "ingest_url":
        _enqueue(
            _run_ingest_url,
            job_id,
            params,
            request.app.state.config,
            request.app.state.embedder,
            request.app.state.vector_store,
            request.app.state.document_store,
        )
        return job_id
    if job_type == "ingest_urls":
        _enqueue(
            _run_ingest_urls,
            job_id,
            params,
            request.app.state.config,
            request.app.state.embedder,
            request.app.state.vector_store,
            request.app.state.document_store,
        )
        return job_id
    if job_type == "reindex":
        doc_id = str(params.get("doc_id", "")).strip()
        if not doc_id:
            raise HTTPException(status_code=422, detail="job params missing doc_id")
        _enqueue(
            _run_reindex,
            job_id,
            doc_id,
            request.app.state.config,
            request.app.state.embedder,
            request.app.state.vector_store,
            request.app.state.document_store,
        )
        return job_id
    if job_type == "rebuild":
        _queue_rebuild(background_tasks, request, job_id, params)
        return job_id
    if job_type == "watch":
        _enqueue(
            _run_watch,
            job_id,
            params.get("watch_file"),
            request.app.state.config,
            request.app.state.embedder,
            request.app.state.vector_store,
            request.app.state.document_store,
            params.get("project"),
        )
        return job_id
    if job_type == "dropzone_scan":
        _enqueue(
            _run_dropzone_scan,
            job_id,
            request.app.state.config,
            request.app.state.embedder,
            request.app.state.vector_store,
            request.app.state.document_store,
            params.get("project"),
        )
        return job_id

    raise HTTPException(status_code=400, detail=f"job type '{job_type}' cannot be restarted")


@router.get("/jobs/{job_id}")
def get_job(job_id: str, _auth: AuthDep) -> dict[str, Any]:
    job = _get_job(job_id)
    if job is None:
        job = _load_job_from_db(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs")
def list_jobs(_auth: AuthDep) -> dict[str, Any]:
    jobs = _list_jobs_from_db(50)
    if not jobs:
        jobs = sorted(_job_snapshot(), key=lambda j: str(j["created_at"]), reverse=True)[:50]
    return {"jobs": list(jobs)}


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str, request: Request, _auth: AuthDep) -> dict[str, Any]:
    in_memory = _get_job(job_id)
    from_db = _load_job_from_db(job_id) if in_memory is None else None
    existing = in_memory or from_db
    if existing is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if str(existing.get("status", "")) == "running":
        raise HTTPException(status_code=409, detail="cannot delete a running job")

    _delete_job_replay_artifact(request.app.state.config, dict(existing.get("params") or {}))
    with _jobs_lock:
        _jobs.pop(job_id, None)
    deleted_db = _delete_job_from_db(job_id)
    if in_memory is None and not deleted_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": job_id}


@router.post("/jobs/{job_id}/restart")
def restart_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
) -> dict[str, Any]:
    if _job_busy():
        raise HTTPException(status_code=409, detail="another job is already running")
    job = _get_job(job_id)
    if job is None:
        job = _load_job_from_db(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if str(job.get("status", "")) == "running":
        raise HTTPException(status_code=409, detail="cannot restart a running job")

    new_job_id = _restart_job(job, background_tasks=background_tasks, request=request)
    return {"job_id": new_job_id, "restarted_from": job_id}
