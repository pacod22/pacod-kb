"""Project-wide regenerate-from-vector-DB and convert-without-index jobs."""

from __future__ import annotations

import json as _json
import logging
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
    _job_busy,
    _job_slot,
    _now_iso,
    _update_job_progress,
)
from pipeline.config import Config
from pipeline.document_store import (
    DocumentStore,
)
from pipeline.indexer import BgeM3Embedder
from pipeline.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

# ---------------------------------------------------------------------------
# Regenerate from Vector DB
# ---------------------------------------------------------------------------


def _payload_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _reconstruct_body_from_payloads(payloads: list[dict[str, Any]]) -> str:
    ordered = sorted(
        payloads,
        key=lambda item: (
            _payload_int(item.get("sequence")),
            _payload_int(item.get("byte_start")),
            str(item.get("chunk_id") or ""),
        ),
    )
    parts: list[str] = []
    seen: set[str] = set()
    for payload in ordered:
        text = str(payload.get("raw_text") or payload.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return "\n\n".join(parts)


def _run_regenerate_from_vector_db(
    job_id: str,
    project_slug: str | None,
    config: Config,
    vector_store: QdrantVectorStore,
    store: DocumentStore,
) -> None:
    """Regenerate documents list from Qdrant collection."""
    with _job_slot(job_id) as acquired:
        if not acquired:
            return
        _update_job_progress(
            job_id, current=0, total=0, message="regenerating from vector db started"
        )
        collection_name = vector_store.collection_name
        target_project_slug = project_slug
        registry_inst: Any | None = None

        # If project_slug is specified, use its collection; otherwise use active
        try:
            from pipeline.projects import make_project_registry

            registry_inst = make_project_registry(config, getattr(store, "_factory", None))
            if project_slug:
                project = registry_inst.get(project_slug)
                if project:
                    collection_name = project.collection_name
                    target_project_slug = project.slug
            else:
                for project in registry_inst.list():
                    if project.collection_name == collection_name:
                        target_project_slug = project.slug
                        break
        except Exception:
            registry_inst = None

        # Get all points from the collection
        _update_job_progress(job_id, current=0, total=0, message="fetching points from collection")
        collected_docs: dict[str, dict[str, Any]] = {}

        try:
            # Scroll through all points in the collection
            offset: Any = None
            page_size = 1000
            while True:
                points, next_offset = vector_store.client.scroll(
                    collection_name=collection_name,
                    limit=page_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not points:
                    break

                for point in points:
                    if not hasattr(point, "payload"):
                        continue
                    payload = point.payload or {}
                    doc_id = payload.get("source_id")
                    if not doc_id:
                        continue

                    # Collect unique document metadata
                    if doc_id not in collected_docs:
                        retrieved_at = payload.get("retrieved_at") or payload.get("created_at")
                        collected_docs[doc_id] = {
                            "doc_id": doc_id,
                            "title": payload.get("title", "Unknown"),
                            "chunk_count": 0,
                            "status": "ok",
                            "retrieved_at": retrieved_at,
                            "project": payload.get("project") or target_project_slug,
                            "source_uri": payload.get("url"),
                            "doc_type": payload.get("source_type", "generic"),
                            "language": payload.get("language", "unknown"),
                            "extraction_method": payload.get("extraction_method", "deterministic"),
                            "valid_until": payload.get("valid_until"),
                            "payloads": [],
                        }
                    doc = collected_docs[doc_id]
                    doc["chunk_count"] += 1
                    cast_payload = dict(payload)
                    doc["payloads"].append(cast_payload)
                    if not doc.get("source_uri") and payload.get("url"):
                        doc["source_uri"] = payload.get("url")
                    if not doc.get("project") and payload.get("project"):
                        doc["project"] = payload.get("project")
                    if not doc.get("retrieved_at") and (
                        payload.get("retrieved_at") or payload.get("created_at")
                    ):
                        doc["retrieved_at"] = payload.get("retrieved_at") or payload.get(
                            "created_at"
                        )

                if next_offset is None or len(points) < page_size:
                    break
                offset = next_offset
        except Exception as exc:
            _finish_job(job_id, error=f"Failed to fetch from Qdrant: {exc}")
            return

        doc_list = list(collected_docs.values())
        persisted = 0
        updated = 0
        if config.storage_backend == "database":
            total_docs = len(doc_list)
            _update_job_progress(
                job_id,
                current=0,
                total=total_docs,
                message="saving regenerated documents to database",
            )
            for idx, doc in enumerate(doc_list, start=1):
                doc_id = str(doc["doc_id"])
                project_value = str(doc.get("project") or target_project_slug or "default")
                title = str(doc.get("title") or f"Recovered {doc_id[:8]}")
                retrieved_at = str(doc.get("retrieved_at") or _now_iso())
                source_uri = str(doc.get("source_uri") or "")
                body = _reconstruct_body_from_payloads(doc.get("payloads", []))
                source_file = f"recovered_{doc_id[:8]}.md"
                source_bytes = body.encode("utf-8")
                front_matter = {
                    "doc_id": doc_id,
                    "title": title,
                    "source_file": source_file,
                    "source_uri": source_uri,
                    "source_version": None,
                    "source_sha256": doc_id,
                    "retrieved_at": retrieved_at,
                    "converted_at": retrieved_at,
                    "converter": "recovered/qdrant",
                    "extraction_method": str(doc.get("extraction_method") or "deterministic"),
                    "language": str(doc.get("language") or "unknown"),
                    "doc_type": str(doc.get("doc_type") or "generic"),
                    "supersedes": None,
                    "valid_until": doc.get("valid_until"),
                    "project": project_value,
                }
                try:
                    existing = store.load_manifest(doc_id)
                except Exception:
                    existing = None

                if existing is None:
                    store.save(
                        doc_id,
                        front_matter,
                        body,
                        source_bytes,
                        ".md",
                        status=str(doc.get("status") or "ok"),
                    )
                    persisted += 1
                else:
                    update_fields: dict[str, Any] = {
                        "title": title,
                        "converter": "recovered/qdrant",
                        "extraction_method": str(doc.get("extraction_method") or "deterministic"),
                        "language": str(doc.get("language") or "unknown"),
                        "doc_type": str(doc.get("doc_type") or "generic"),
                        "project": project_value,
                        "chunk_count": int(doc.get("chunk_count") or 0),
                        "valid_until": doc.get("valid_until"),
                    }
                    if source_uri:
                        update_fields["source_uri"] = source_uri
                    if retrieved_at:
                        update_fields["retrieved_at"] = retrieved_at
                        update_fields["converted_at"] = retrieved_at
                    if body:
                        update_fields["body"] = body
                        update_fields["source_bytes"] = source_bytes
                        update_fields["source_ext"] = ".md"
                        update_fields["source_file"] = source_file
                    store.update_manifest(doc_id, **update_fields)
                    updated += 1

                store.update_manifest(doc_id, chunk_count=int(doc.get("chunk_count") or 0))
                store.append_event(
                    doc_id,
                    "regenerated_from_vector_db",
                    f"collection={collection_name}",
                )
                doc["project"] = project_value
                _update_job_progress(
                    job_id,
                    current=idx,
                    total=total_docs,
                    message=f"saved {idx}/{total_docs} regenerated documents",
                )
        _update_job_progress(
            job_id,
            current=len(doc_list),
            total=len(doc_list),
            message=f"regenerated {len(doc_list)} documents from vector db",
        )
        _finish_job(
            job_id,
            result={
                "doc_count": len(doc_list),
                "documents": [
                    {k: v for k, v in doc.items() if k != "payloads"} for doc in doc_list[:100]
                ],
                "persisted": persisted,
                "updated": updated,
            },
        )


@router.post("/projects/{slug}/regenerate-from-vector-db")
def regenerate_from_vector_db(
    slug: str,
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
) -> dict[str, str]:
    """Regenerate documents list based on data from the Qdrant collection."""
    if _job_busy():
        raise HTTPException(status_code=409, detail="another job is already running")

    job_id = _create_job("regenerate_from_vector_db", {"project": slug})
    _enqueue(
        _run_regenerate_from_vector_db,
        job_id,
        slug,
        request.app.state.config,
        request.app.state.vector_store,
        request.app.state.document_store,
    )
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Convert without Index
# ---------------------------------------------------------------------------


def _run_convert_without_index(
    job_id: str,
    project_slug: str | None,
    config: Config,
    embedder: BgeM3Embedder,
    store: DocumentStore,
) -> None:
    """Convert documents without sending to vector database."""
    with _job_slot(job_id) as acquired:
        if not acquired:
            return
        _update_job_progress(job_id, current=0, total=0, message="convert without index started")

        # Get all files from dropzone
        _update_job_progress(
            job_id,
            current=0,
            total=0,
            message="scanning for documents to convert",
        )

        dropzone_dir = config.kb_dropzone_dir
        if not dropzone_dir.exists():
            _finish_job(job_id, error="Dropzone directory not found")
            return

        files = [f for f in dropzone_dir.iterdir() if f.is_file()]
        total_files = len(files)

        if total_files == 0:
            _finish_job(job_id, result={"converted": 0, "skipped": 0, "errors": 0})
            return

        converted_count = 0
        error_count = 0

        for idx, file_path in enumerate(files):
            _update_job_progress(
                job_id, current=idx, total=total_files, message=f"converting {file_path.name}"
            )

            try:
                # Parse but don't ingest (no store write, no vector index)
                from pipeline.parsers import get_parser

                parser = get_parser(file_path.suffix.lower())
                if parser is None:
                    error_count += 1
                    continue

                # Parse the document
                result = parser.parse(file_path)

                # Store converted result in a temporary location
                converted_dir = config.kb_root / "converted_staging"
                converted_dir.mkdir(parents=True, exist_ok=True)

                converted_file = converted_dir / f"{file_path.stem}_converted.json"
                converted_file.write_text(
                    _json.dumps(
                        {
                            "source_file": file_path.name,
                            "title": result.title or file_path.stem,
                            "body": result.body,
                            "language": result.language,
                            "extraction_method": result.extraction_method,
                            "converter": result.converter,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                converted_count += 1
            except Exception as exc:
                logger.exception(f"Error converting {file_path.name}: {exc}")
                error_count += 1

        _update_job_progress(
            job_id, current=total_files, total=total_files, message="conversion complete"
        )
        _finish_job(
            job_id,
            result={
                "converted": converted_count,
                "failed": error_count,
                "total": total_files,
                "staging_dir": "converted_staging",
            },
        )


@router.post("/projects/{slug}/convert-without-index")
def convert_without_index(
    slug: str,
    background_tasks: BackgroundTasks,
    request: Request,
    _auth: AuthDep,
) -> dict[str, str]:
    """Convert documents without sending to the vector database."""
    if _job_busy():
        raise HTTPException(status_code=409, detail="another job is already running")

    job_id = _create_job("convert_without_index", {"project": slug})
    _enqueue(
        _run_convert_without_index,
        job_id,
        slug,
        request.app.state.config,
        request.app.state.embedder,
        request.app.state.document_store,
    )
    return {"job_id": job_id}
