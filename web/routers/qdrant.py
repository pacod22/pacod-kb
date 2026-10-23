"""Qdrant collection management routes (server mode only)."""

from __future__ import annotations

import os
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from markupsafe import escape

from web.auth import Principal, require_role, require_user, require_write
from web.dependencies import PipelineClientDep, QdrantClientDep
from web.html import Tone, notice
from web.roles import role_satisfies
from web.templates_config import templates

router = APIRouter(prefix="/kb/qdrant", dependencies=[Depends(require_write)])

_PAGE_SIZE = 50
_MAX_BROWSE_ALL = 10_000

# Destructive/irreversible collection operations are admin-only.
_ADMIN = [Depends(require_role("admin"))]

_BACK_TO_COLLECTIONS = '<a href="/kb/qdrant/" class="btn btn-sm">Back to collections</a>'

_TONES: dict[str, Tone] = {"success": "good", "warning": "warn", "danger": "bad", "info": "info"}


def _alert(kind: str, html: str) -> str:
    """Wrap pre-escaped *html* in a notice (web/html.py); *kind* is success/warning/danger/info.

    The text goes in one <span> (the notice's text is a flex column), and the result is a plain
    str: a Markup would escape the links concatenated after it.
    """
    return str(notice(_TONES[kind], f"<span>{html}</span>"))


def _back_link(collection: str) -> str:
    href = f"/kb/qdrant/{quote(collection, safe='')}?tab=manage"
    return f'<a href="{href}" class="btn btn-sm">Back</a>'


def _qdrant_base_url() -> str:
    return os.environ.get("QDRANT_URL", "http://pacod-qdrant:6333").rstrip("/")


def _qdrant_api_key() -> str | None:
    return os.environ.get("QDRANT_API_KEY") or None


# ---------------------------------------------------------------------------
# Which collections are kb's
# ---------------------------------------------------------------------------
#
# Qdrant may be shared with other products (pacod-qdrant on pacod-net: pacod-ap keeps its
# pacod_* collections there). These pages show and touch only kb's: names that start with
# KB_QDRANT_COLLECTION_PREFIX (default "kb_"; kb's default collection is kb_chunks), and the
# collections kb's projects own, whose names may predate the prefix. Anything else is a 404.
# An empty prefix allows every collection (a Qdrant of kb's own).


def _collection_prefix() -> str:
    return os.environ.get("KB_QDRANT_COLLECTION_PREFIX", "kb_").strip()


async def _project_collections(pipeline_client: httpx.AsyncClient) -> set[str]:
    """The collections kb's projects own; empty when the pipeline cannot say."""
    try:
        resp = await pipeline_client.get("/projects")
        resp.raise_for_status()
        projects = resp.json().get("projects", [])
        return {str(p.get("collection_name") or "") for p in projects} - {""}
    except (httpx.HTTPError, ValueError, AttributeError):
        return set()


def _has_prefix(name: str) -> bool:
    prefix = _collection_prefix()
    return not prefix or name.startswith(prefix)


async def kb_collection(collection: str, pipeline_client: PipelineClientDep) -> str:
    """The path's collection, if it is kb's (else 404, as if it did not exist)."""
    if _has_prefix(collection) or collection in await _project_collections(pipeline_client):
        return collection
    raise HTTPException(status_code=404, detail="Collection not found")


KbCollection = Annotated[str, Depends(kb_collection)]


async def _owned_collections(pipeline_client: PipelineClientDep) -> set[str]:
    return await _project_collections(pipeline_client)


# Awaited in the event loop, so a sync route (Qdrant's client blocks) still runs in the threadpool.
OwnedCollections = Annotated[set[str], Depends(_owned_collections)]


def _prefix_error(name: str) -> HTMLResponse | None:
    """A 400 alert when a new collection name lacks kb's prefix (it would be hidden here)."""
    if _has_prefix(name):
        return None
    return HTMLResponse(
        _alert(
            "danger",
            f"Collection names start with <strong>{escape(_collection_prefix())}</strong>: "
            "the Qdrant server may be shared, and this page shows only kb's collections.",
        )
        + _BACK_TO_COLLECTIONS,
        status_code=400,
    )


# ---------------------------------------------------------------------------
# Collections list
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def list_collections(
    request: Request, client: QdrantClientDep, owned: OwnedCollections
) -> HTMLResponse:
    colls = client.get_collections().collections
    items: list[dict[str, Any]] = []
    for c in colls:
        if not (_has_prefix(c.name) or c.name in owned):
            continue
        try:
            info = client.get_collection(c.name)
            points = info.points_count or 0
            vectors = info.indexed_vectors_count or 0
        except Exception:
            points = vectors = 0
        items.append({"name": c.name, "points": points, "indexed_vectors": vectors})
    return templates.TemplateResponse(
        request,
        "qdrant/collections.html",
        {"collections": items},
    )


@router.post("/", response_class=HTMLResponse)
def create_collection(
    request: Request,
    client: QdrantClientDep,
    name: str = Form(...),
    vector_size: int = Form(1024),
    distance: str = Form("Cosine"),
) -> HTMLResponse:
    from qdrant_client.models import Distance, VectorParams

    if (refused := _prefix_error(name)) is not None:
        return refused
    dist_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=vector_size, distance=dist_map.get(distance, Distance.COSINE)
        ),
    )
    return HTMLResponse(
        _alert("success", f"Collection <strong>{escape(name)}</strong> created.")
        + _BACK_TO_COLLECTIONS
    )


# ---------------------------------------------------------------------------
# Collection detail — three tabs: Browse | Search | Manage
# ---------------------------------------------------------------------------


@router.get("/{collection}", response_class=HTMLResponse)
def collection_detail(
    request: Request,
    collection: KbCollection,
    client: QdrantClientDep,
    tab: str = "browse",
) -> HTMLResponse:
    info = client.get_collection(collection)
    aliases: list[str] = []
    snapshots: list[Any] = []
    if tab == "manage":
        try:
            aliases_resp = client.get_collection_aliases(collection)
            aliases = [a.alias_name for a in aliases_resp.aliases]
        except Exception:
            pass
        try:
            snapshots = client.list_snapshots(collection)
        except Exception:
            pass
    return templates.TemplateResponse(
        request,
        "qdrant/collection.html",
        {
            "collection": collection,
            "info": info,
            "tab": tab,
            "aliases": aliases,
            "snapshots": snapshots,
        },
    )


# ---------------------------------------------------------------------------
# Browse — paginated point list (HTMX partial)
# ---------------------------------------------------------------------------

_VALID_PAGE_SIZES = {"50", "100", "1000", "2000", "5000", "all"}


@router.get("/{collection}/browse", response_class=HTMLResponse)
def browse_points(
    request: Request,
    collection: KbCollection,
    client: QdrantClientDep,
    offset: str = "",
    active_only: str = "true",
    doc_title: str = "",
    page_size: str = "50",
) -> HTMLResponse:
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        IsNullCondition,
        MatchValue,
        PayloadField,
    )

    if page_size not in _VALID_PAGE_SIZES:
        page_size = "50"

    conditions: list[Any] = []
    if active_only == "true":
        conditions.append(IsNullCondition(is_null=PayloadField(key="valid_until")))
    if doc_title:
        conditions.append(FieldCondition(key="title", match=MatchValue(value=doc_title)))
    flt = Filter(must=conditions) if conditions else None

    next_off = ""
    if page_size == "all":
        # "All" is capped so a large collection cannot exhaust UI memory or
        # render an unbounded table; the Next link continues past the cap.
        raw_records: list[Any] = []
        cur_offset: Any = offset or None
        while True:
            batch, batch_next = client.scroll(
                collection_name=collection,
                limit=1000,
                offset=cur_offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=flt,
            )
            raw_records.extend(batch)
            if batch_next is None:
                break
            if len(raw_records) >= _MAX_BROWSE_ALL:
                next_off = str(batch_next)
                break
            cur_offset = batch_next
    else:
        raw_records, next_offset = client.scroll(
            collection_name=collection,
            limit=int(page_size),
            offset=offset or None,
            with_payload=True,
            with_vectors=False,
            scroll_filter=flt,
        )
        next_off = str(next_offset) if next_offset is not None else ""

    points: list[dict[str, Any]] = []
    for r in raw_records:
        p = dict(r.payload or {})
        points.append(
            {
                "id": str(r.id),
                "chunk_id": p.get("chunk_id", ""),
                "source_id": p.get("source_id", ""),
                "title": p.get("title", ""),
                "heading_path": p.get("heading_path", []),
                "heading_level": p.get("heading_level", 0),
                "sequence": p.get("sequence", 0),
                "valid_until": p.get("valid_until"),
                "extraction_method": p.get("extraction_method", ""),
                "confidence": p.get("confidence", 1.0),
            }
        )

    return templates.TemplateResponse(
        request,
        "qdrant/_browse.html",
        {
            "collection": collection,
            "points": points,
            "next_offset": next_off,
            "active_only": active_only,
            "has_more": bool(next_off),
            "doc_title": doc_title,
            "page_size": page_size,
        },
    )


# ---------------------------------------------------------------------------
# Point detail
# ---------------------------------------------------------------------------


@router.get("/{collection}/points/{point_id}", response_class=HTMLResponse)
def point_detail(
    request: Request,
    collection: KbCollection,
    point_id: str,
    client: QdrantClientDep,
) -> HTMLResponse:
    records = client.retrieve(
        collection_name=collection,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return HTMLResponse(_alert("danger", "Point not found."), status_code=404)
    r = records[0]
    return templates.TemplateResponse(
        request,
        "qdrant/point.html",
        {
            "collection": collection,
            "point_id": point_id,
            "payload": dict(r.payload or {}),
        },
    )


# ---------------------------------------------------------------------------
# Export points (selected UUIDs or all for a doc_id)
# ---------------------------------------------------------------------------


@router.get("/{collection}/export")
def export_points(
    request: Request,
    collection: KbCollection,
    client: QdrantClientDep,
    point_ids: list[str] = Query(default=[]),
    doc_id: str = "",
    doc_title: str = "",
) -> Any:
    import json as _json
    from datetime import UTC, datetime
    from urllib.parse import quote

    from fastapi.responses import Response as _Response

    export_all: bool = not point_ids and not doc_id and not doc_title

    points_out: list[dict[str, Any]] = []

    if point_ids:
        records = client.retrieve(
            collection_name=collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            p = dict(r.payload or {})
            p["_point_id"] = str(r.id)
            points_out.append(p)
    elif doc_id or doc_title:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if doc_id:
            flt = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=doc_id))])
        else:
            flt = Filter(must=[FieldCondition(key="title", match=MatchValue(value=doc_title))])
        offset: Any = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=flt,
            )
            for r in batch:
                p = dict(r.payload or {})
                p["_point_id"] = str(r.id)
                points_out.append(p)
            if next_offset is None:
                break
            offset = next_offset
    else:
        # export_all — scroll entire collection without filter
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in batch:
                p = dict(r.payload or {})
                p["_point_id"] = str(r.id)
                points_out.append(p)
            if next_offset is None:
                break
            offset = next_offset

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if doc_title:
        slug = doc_title[:20].replace(" ", "_").lower()
    elif doc_id:
        slug = doc_id[:8]
    elif export_all:
        slug = "all"
    else:
        slug = "selected"
    payload = {
        "collection": collection,
        "exported_at": datetime.now(UTC).isoformat(),
        "doc_id": doc_id or None,
        "point_count": len(points_out),
        "points": points_out,
    }
    fname = f"qdrant_{collection}_{slug}_{ts}.json"
    # Content-Disposition is a latin-1 header; non-ASCII titles (e.g. Cyrillic)
    # must not go into `filename=` verbatim or the ASGI server 500s. Provide an
    # ASCII-safe fallback plus an RFC 5987 filename* with the real UTF-8 name.
    ascii_fname = fname.encode("ascii", "replace").decode("ascii").replace("?", "_")
    quoted = quote(fname, safe="")
    disposition = f"attachment; filename=\"{ascii_fname}\"; filename*=UTF-8''{quoted}"
    return _Response(
        content=_json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": disposition},
    )


# ---------------------------------------------------------------------------
# Delete single point
# ---------------------------------------------------------------------------


@router.delete("/{collection}/points/{point_id}", response_class=HTMLResponse)
def delete_point(
    request: Request,
    collection: KbCollection,
    point_id: str,
    client: QdrantClientDep,
) -> HTMLResponse:
    from qdrant_client.models import PointIdsList

    client.delete(
        collection_name=collection,
        points_selector=PointIdsList(points=[point_id]),
    )
    return HTMLResponse("")  # HTMX removes the row via hx-swap="outerHTML"


# ---------------------------------------------------------------------------
# Bulk delete
# ---------------------------------------------------------------------------


@router.post("/{collection}/delete", response_class=HTMLResponse)
def bulk_delete(
    request: Request,
    collection: KbCollection,
    client: QdrantClientDep,
    mode: str = Form(...),  # "doc_id" | "superseded"
    doc_id: str = Form(""),
    user: Principal = Depends(require_user),
) -> HTMLResponse:
    if mode != "doc_id" and not role_satisfies(user.role, "admin"):
        return HTMLResponse(
            _alert("danger", "Deleting all superseded points requires the admin role."),
            status_code=403,
        )
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        FilterSelector,
        IsNullCondition,
        MatchValue,
        PayloadField,
    )

    try:
        if mode == "doc_id":
            if not doc_id.strip():
                return HTMLResponse(_alert("warning", "No doc_id provided."))
            flt = Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=doc_id.strip()))]
            )
        else:  # superseded
            flt = Filter(must_not=[IsNullCondition(is_null=PayloadField(key="valid_until"))])

        client.delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=flt),
        )
        label = f"doc_id={doc_id.strip()!r}" if mode == "doc_id" else "all superseded"
        return HTMLResponse(_alert("success", f"Deleted points matching {escape(label)}."))
    except Exception as exc:
        return HTMLResponse(_alert("danger", str(escape(str(exc)))))


# ---------------------------------------------------------------------------
# Semantic search (proxied to pipeline API)
# ---------------------------------------------------------------------------


@router.post("/{collection}/search", response_class=HTMLResponse)
async def search(
    request: Request,
    collection: KbCollection,
    pipeline_client: PipelineClientDep,
    query: str = Form(...),
    top_k: int = Form(10),
    include_historical: bool = Form(False),
) -> HTMLResponse:
    try:
        resp = await pipeline_client.post(
            "/query",
            json={
                "query": query,
                "top_k": top_k,
                "include_historical": include_historical,
                "collection": collection,
                "wrap_untrusted": False,
            },
        )
        resp.raise_for_status()
        results: list[dict[str, Any]] = resp.json().get("results", [])
    except httpx.HTTPStatusError as exc:
        return HTMLResponse(_alert("danger", f"Search failed: {escape(exc.response.text)}"))
    return templates.TemplateResponse(
        request,
        "qdrant/_search_results.html",
        {"results": results, "query": query},
    )


# ---------------------------------------------------------------------------
# Collection management (rename, delete)
# ---------------------------------------------------------------------------


@router.delete(
    "/{collection}",
    response_class=HTMLResponse,
    dependencies=_ADMIN,
)
def delete_collection(
    request: Request, collection: KbCollection, client: QdrantClientDep
) -> HTMLResponse:
    client.delete_collection(collection)
    return HTMLResponse(
        _alert("warning", f"Collection <strong>{escape(collection)}</strong> deleted.")
        + _BACK_TO_COLLECTIONS
    )


@router.post("/{collection}/rename", response_class=HTMLResponse, dependencies=_ADMIN)
def rename_collection(
    request: Request,
    collection: KbCollection,
    client: QdrantClientDep,
    new_name: str = Form(...),
) -> HTMLResponse:
    from qdrant_client.models import VectorParams

    if (refused := _prefix_error(new_name)) is not None:
        return refused
    info = client.get_collection(collection)
    vectors_cfg = info.config.params.vectors
    if isinstance(vectors_cfg, dict):
        first = next(iter(vectors_cfg.values()))
        size, distance = first.size, first.distance
    else:
        size = vectors_cfg.size  # type: ignore[union-attr]
        distance = vectors_cfg.distance  # type: ignore[union-attr]

    client.create_collection(
        collection_name=new_name,
        vectors_config=VectorParams(size=size, distance=distance),
    )

    from qdrant_client.models import (
        CreateAlias,
        CreateAliasOperation,
        DeleteAlias,
        DeleteAliasOperation,
        PointStruct,
    )

    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=500,
            with_vectors=True,
            with_payload=True,
        )
        if result:
            pts = [
                PointStruct(id=r.id, vector=r.vector or {}, payload=r.payload or {})  # type: ignore[arg-type]
                for r in result
            ]
            client.upsert(collection_name=new_name, points=pts)
        if next_offset is None:
            break
        offset = next_offset

    aliases_resp = client.get_collection_aliases(collection)
    for alias in aliases_resp.aliases:
        client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(collection_name=new_name, alias_name=alias.alias_name)
                )
            ]
        )
        client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias.alias_name))
            ]
        )

    client.delete_collection(collection)
    return HTMLResponse(
        _alert(
            "success",
            f"Renamed <strong>{escape(collection)}</strong> → <strong>{escape(new_name)}</strong>.",
        )
        + _BACK_TO_COLLECTIONS
    )


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@router.post("/{collection}/snapshots", response_class=HTMLResponse)
def create_snapshot(
    request: Request, collection: KbCollection, client: QdrantClientDep
) -> HTMLResponse:
    snap = client.create_snapshot(collection)
    if snap is None:
        return HTMLResponse(_alert("danger", "Snapshot creation failed."))
    return HTMLResponse(
        _alert("success", f"Snapshot <strong>{escape(snap.name)}</strong> created.")
        + _back_link(collection)
    )


@router.delete("/{collection}/snapshots/{name}", response_class=HTMLResponse, dependencies=_ADMIN)
def delete_snapshot(
    request: Request, collection: KbCollection, name: str, client: QdrantClientDep
) -> HTMLResponse:
    client.delete_snapshot(collection, name)
    return HTMLResponse(_alert("warning", f"Snapshot <strong>{escape(name)}</strong> deleted."))


@router.get("/{collection}/snapshots/{name}/download")
def download_snapshot(collection: KbCollection, name: str) -> StreamingResponse:
    import httpx as _httpx

    base_url = _qdrant_base_url()
    api_key = _qdrant_api_key()
    headers: dict[str, str] = {}
    if api_key:
        headers["api-key"] = api_key

    async def _stream() -> Any:
        async with _httpx.AsyncClient() as hc:
            async with hc.stream(
                "GET",
                f"{base_url}/collections/{quote(collection, safe='')}"
                f"/snapshots/{quote(name, safe='')}",
                headers=headers,
                timeout=300.0,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(65536):
                    yield chunk

    return StreamingResponse(
        _stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.post("/{collection}/snapshots/restore", response_class=HTMLResponse, dependencies=_ADMIN)
async def restore_snapshot(
    request: Request,
    collection: KbCollection,
    _client: QdrantClientDep,
    file: UploadFile = File(...),
) -> HTMLResponse:
    base_url = _qdrant_base_url()
    api_key = _qdrant_api_key()
    headers: dict[str, str] = {}
    if api_key:
        headers["api-key"] = api_key

    content = await file.read()
    async with httpx.AsyncClient(timeout=300.0) as hc:
        resp = await hc.post(
            f"{base_url}/collections/{quote(collection, safe='')}/snapshots/upload",
            headers=headers,
            files={"snapshot": (file.filename, content, "application/octet-stream")},
        )
    if resp.status_code >= 400:
        return HTMLResponse(_alert("danger", f"Restore failed: {escape(resp.text)}"))
    return HTMLResponse(
        _alert("success", f"Snapshot restored to <strong>{escape(collection)}</strong>.")
        + _back_link(collection)
    )
