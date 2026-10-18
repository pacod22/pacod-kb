"""Document listing, detail, downloads, deletion and export."""

from __future__ import annotations

import json as _json
import logging
import mimetypes
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
)

from pipeline.api import (
    AuthDep,
    _exclusive,
    _now_iso,
    _runtime,
)
from pipeline.config import Config
from pipeline.document_store import (
    DocumentStore,
    _sources_to_csv,
)
from pipeline.indexer import make_fts_db
from pipeline.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

# ---------------------------------------------------------------------------
# Delete document
# ---------------------------------------------------------------------------


@router.delete("/documents/{doc_id}")
def delete_document_api(
    doc_id: str,
    request: Request,
    _auth: AuthDep,
) -> dict[str, Any]:
    with _exclusive():
        return _delete_document(doc_id, request)


def _delete_document(doc_id: str, request: Request) -> dict[str, Any]:
    config: Config = request.app.state.config
    vector_store: QdrantVectorStore = request.app.state.vector_store
    store: DocumentStore = request.app.state.document_store
    _store = store if config.storage_backend == "database" else None

    from pipeline.versioning import delete_document

    if _store is not None:
        found_id = _store.find_by_prefix(doc_id)
        if found_id is None:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        full_doc_id = found_id
        fts_db = make_fts_db(config)
        try:
            delete_document(full_doc_id, None, vector_store, fts_db, store=_store)
        finally:
            fts_db.close()
        return {"deleted": full_doc_id}

    from pipeline.versioning import find_doc_folder

    folder = find_doc_folder(doc_id, config, prefix=True)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    full_doc_id = _json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["doc_id"]

    fts_db = make_fts_db(config)
    try:
        delete_document(full_doc_id, folder, vector_store, fts_db)
    finally:
        fts_db.close()

    return {"deleted": full_doc_id, "folder": folder.name}


# ---------------------------------------------------------------------------
# Documents list + export
# ---------------------------------------------------------------------------


@router.get("/documents")
def list_documents_api(request: Request, _auth: AuthDep) -> dict[str, Any]:
    store: DocumentStore = request.app.state.document_store
    docs = store.list_all()
    proj = _runtime.get("active_project", "")
    if proj:
        from pipeline.projects import ProjectStore

        registry: ProjectStore = request.app.state.project_registry
        proj_list = registry.list()
        default_slug: str | None = proj_list[0].slug if proj_list else None
        docs = [d for d in docs if (d.get("project") or default_slug) == proj]
    return {"documents": docs}


@router.get("/documents/{doc_id}")
def get_document_api(
    doc_id: str,
    request: Request,
    _auth: AuthDep,
) -> dict[str, Any]:
    store: DocumentStore = request.app.state.document_store
    full_doc_id = store.find_by_prefix(doc_id) or doc_id

    try:
        manifest = store.load_manifest(full_doc_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found") from None

    try:
        body = store.load_body(full_doc_id)
    except Exception:
        body = ""

    superseded_by: dict[str, Any] | None = None
    successor_id = store.find_superseding(full_doc_id)
    if successor_id is not None:
        try:
            superseded_by = store.load_manifest(successor_id)
        except KeyError:
            superseded_by = None

    # Only filesystem documents have a folder on disk; DB mode returns None.
    source_size: int | None = None
    config: Config = request.app.state.config
    if config.storage_backend != "database":
        from pipeline.versioning import find_doc_folder

        folder = find_doc_folder(full_doc_id, config)
        source_file = str(manifest.get("source_file") or "")
        if folder is not None and source_file:
            sp = folder / Path(source_file).name
            if sp.exists():
                source_size = sp.stat().st_size

    return {
        "manifest": manifest,
        "body": body,
        "superseded_by": superseded_by,
        "source_size": source_size,
    }


@router.get("/documents/{doc_id}/download/original")
def download_document_original_api(
    doc_id: str,
    request: Request,
    _auth: AuthDep,
) -> Any:
    from fastapi.responses import Response as _Response

    store: DocumentStore = request.app.state.document_store
    full_doc_id = store.find_by_prefix(doc_id) or doc_id

    try:
        source_bytes, source_name = store.load_source_blob(full_doc_id)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Document source '{doc_id}' not found",
        ) from None

    safe_name = Path(source_name).name or f"{full_doc_id[:8]}.bin"
    media_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    quoted_name = urllib.parse.quote(safe_name, safe="")
    return _Response(
        content=source_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_name}"},
    )


@router.get("/documents/{doc_id}/download/markdown")
def download_document_markdown_api(
    doc_id: str,
    request: Request,
    _auth: AuthDep,
) -> Any:
    from fastapi.responses import Response as _Response

    from pipeline.normalize import assemble_document

    store: DocumentStore = request.app.state.document_store
    full_doc_id = store.find_by_prefix(doc_id) or doc_id

    try:
        manifest = store.load_manifest(full_doc_id)
        body = store.load_body(full_doc_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found") from None

    base_name = Path(str(manifest.get("source_file") or "")).stem
    md_name = f"{base_name or full_doc_id[:8]}.md"
    quoted_name = urllib.parse.quote(md_name, safe="")
    return _Response(
        content=assemble_document(manifest, body),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_name}"},
    )


@router.get("/export")
def export_documents(request: Request, _auth: AuthDep) -> Any:
    from fastapi.responses import Response as _Response

    store: DocumentStore = request.app.state.document_store
    docs = store.export_all()
    payload: dict[str, Any] = {
        "export_version": "1",
        "exported_at": _now_iso(),
        "storage_backend": _runtime.get("storage_backend", "filesystem"),
        "document_count": len(docs),
        "documents": docs,
    }
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return _Response(
        content=_json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="kb_export_{ts}.json"'},
    )


@router.get("/export/sources")
def export_sources(
    request: Request,
    _auth: AuthDep,
    fmt: str = Query("json", alias="format"),
    doc_ids: list[str] = Query(default=[]),
) -> Any:
    from fastapi.responses import Response as _Response

    store: DocumentStore = request.app.state.document_store
    rows = store.export_as_ingested_sources()
    if doc_ids:
        id_set = set(doc_ids)
        rows = [r for r in rows if r["source_id"] in id_set]
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    if fmt == "csv":
        csv_bytes = ("﻿" + _sources_to_csv(rows)).encode("utf-8")
        return _Response(
            content=csv_bytes,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="kb_sources_{ts}.csv"'},
        )

    payload: dict[str, Any] = {
        "export_version": "1",
        "exported_at": _now_iso(),
        "source_count": len(rows),
        "sources": rows,
    }
    return _Response(
        content=_json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="kb_sources_{ts}.json"'},
    )
