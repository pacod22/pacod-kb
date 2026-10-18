"""Qdrant collections and project registry endpoints."""

from __future__ import annotations

import json as _json
import logging
import shutil
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)
from pydantic import BaseModel

import pipeline.api as core
from pipeline.api import (
    AuthDep,
    _exclusive,
    _job_busy,
    _runtime,
    _switch_active_project,
)
from pipeline.config import Config
from pipeline.document_store import (
    DocumentStore,
)
from pipeline.indexer import make_fts_db
from pipeline.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


@router.get("/collections")
def list_collections(request: Request, _auth: AuthDep) -> dict[str, Any]:
    vector_store: QdrantVectorStore = request.app.state.vector_store
    cols = vector_store.client.get_collections().collections
    return {
        "collections": [
            {"name": c.name, "points_count": getattr(c, "points_count", None)} for c in cols
        ],
        "active": _runtime.get("active_collection"),
    }


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""
    collection_name: str = ""


class UpdateProjectRequest(BaseModel):
    collection_name: str


@router.get("/projects")
def list_projects(request: Request, _auth: AuthDep) -> dict[str, Any]:
    from dataclasses import asdict

    from pipeline.projects import ProjectStore

    registry: ProjectStore = request.app.state.project_registry
    store: DocumentStore = request.app.state.document_store
    registry_projects = registry.list()
    default_slug = registry_projects[0].slug if registry_projects else None
    all_docs = store.list_all()
    counts: dict[str, int] = {}
    for doc in all_docs:
        p = str(doc.get("project") or "") or (default_slug or "")
        if p:
            counts[p] = counts.get(p, 0) + 1
    projects = [{**asdict(p), "doc_count": counts.get(p.slug, 0)} for p in registry_projects]
    return {"projects": projects, "active": _runtime.get("active_project")}


@router.post("/projects")
def create_project(
    req: CreateProjectRequest,
    request: Request,
    _auth: AuthDep,
) -> dict[str, Any]:
    from dataclasses import asdict

    from pipeline.projects import ProjectStore

    registry: ProjectStore = request.app.state.project_registry
    try:
        project = registry.create(req.name, req.description, req.collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Auto-create the Qdrant collection for this project
    vector_store: QdrantVectorStore = request.app.state.vector_store
    try:
        core.QdrantVectorStore(vector_store.client, project.collection_name)
    except Exception:
        pass
    return {"project": asdict(project)}


@router.patch("/projects/{slug}")
def update_project(
    slug: str,
    req: UpdateProjectRequest,
    request: Request,
    _auth: AuthDep,
) -> dict[str, Any]:
    from dataclasses import asdict

    from pipeline.projects import ProjectStore

    if _job_busy():
        raise HTTPException(status_code=409, detail="a job is currently running")

    registry: ProjectStore = request.app.state.project_registry
    project = registry.get(slug)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{slug}' not found")
    if _runtime.get("active_project") == slug:
        raise HTTPException(
            status_code=409,
            detail="cannot change collection for the active project",
        )

    store: DocumentStore = request.app.state.document_store
    projects = registry.list()
    default_slug = projects[0].slug if projects else None
    doc_count = 0
    for doc in store.list_all():
        if (doc.get("project") or default_slug) == slug:
            doc_count += 1
    if doc_count > 0:
        raise HTTPException(
            status_code=409,
            detail="cannot change collection for a project that already has documents",
        )

    try:
        updated = registry.update_collection_name(slug, req.collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    vector_store: QdrantVectorStore = request.app.state.vector_store
    try:
        core.QdrantVectorStore(vector_store.client, updated.collection_name)
    except Exception:
        pass
    return {"project": asdict(updated)}


@router.delete("/projects/{slug}")
def delete_project_api(
    slug: str,
    request: Request,
    _auth: AuthDep,
) -> dict[str, Any]:
    from pipeline.projects import ProjectStore

    registry: ProjectStore = request.app.state.project_registry
    project = registry.get(slug)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{slug}' not found")
    with _exclusive():
        return _delete_project(slug, project, request)


def _delete_project(slug: str, project: Any, request: Request) -> dict[str, Any]:
    registry = request.app.state.project_registry

    config: Config = request.app.state.config
    vector_store: QdrantVectorStore = request.app.state.vector_store
    projects = registry.list()
    default_slug = projects[0].slug if projects else None

    # 1. Drop all FTS5 entries for this project
    fts_db = make_fts_db(config)
    try:
        fts_db.drop_by_project(slug)
    finally:
        fts_db.close()

    # 2. Delete the Qdrant collection (swallow errors — may not exist yet)
    try:
        vector_store.client.delete_collection(project.collection_name)
    except Exception:
        pass

    # 3. Delete all document records / folders for this project
    store: DocumentStore = request.app.state.document_store
    _store = store if config.storage_backend == "database" else None
    if _store is not None:
        all_docs = store.list_all()
        for doc in all_docs:
            if (doc.get("project") or default_slug) == slug:
                try:
                    _store.delete(doc["doc_id"])
                except Exception:
                    pass
    else:
        docs_dir = config.documents_dir
        if docs_dir.exists():
            for folder in list(docs_dir.iterdir()):
                mf = folder / "manifest.json"
                if not mf.exists():
                    continue
                try:
                    data = _json.loads(mf.read_text(encoding="utf-8"))
                    if (data.get("project") or default_slug) == slug:
                        shutil.rmtree(folder)
                except Exception:
                    pass

    # 4. Remove from registry
    registry.delete(slug)

    # 5. If deleted project was active, reset to first remaining or clear
    if _runtime.get("active_project") == slug:
        remaining = registry.list()
        if remaining:
            _switch_active_project(remaining[0].slug, request.app)
        else:
            _runtime["active_project"] = ""
            _runtime["active_collection"] = ""

    return {"deleted": slug}
