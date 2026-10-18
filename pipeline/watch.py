"""Watch mode: poll configured web sources for changes and re-ingest on sha256 change."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.acquisition import IngestOptions, ingest_document
from pipeline.indexer import index_document
from pipeline.versioning import reinstate_document, supersede_document

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.document_store import DocumentStore
    from pipeline.indexer import Embedder, FtsStore
    from pipeline.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def _find_current_doc_for_url(
    url: str,
    config: Config,
    *,
    store: DocumentStore | None = None,
) -> str | None:
    """Return the currently-valid doc_id for *url* (``valid_until`` is null)."""
    if store is not None:
        return store.find_by_source_uri(url)

    docs_dir = config.documents_dir
    if not docs_dir.exists():
        return None
    for folder in docs_dir.iterdir():
        manifest = folder / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("source_uri") == url and data.get("valid_until") is None:
                return str(data["doc_id"])
        except Exception:
            pass
    return None


def _load_manifest(doc_id: str, folder: Path | None, store: DocumentStore | None) -> dict[str, Any]:
    if store is not None:
        return store.load_manifest(doc_id)
    if folder is None or not (folder / "manifest.json").exists():
        return {}
    data: dict[str, Any] = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return data


def watch_sources(
    config: Config,
    watch_file: Path,
    embedder: Embedder,
    vector_store: VectorStore,
    fts_db: FtsStore,
    *,
    store: DocumentStore | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    project: str | None = None,
    options: Callable[[], IngestOptions] | None = None,
) -> list[dict[str, Any]]:
    """Single-pass check of all web sources in *watch_file*.

    New versions are tagged with *project* (``None`` = the default project).
    *options* supplies the runtime switches per source (env config without it).
    A prior version is only superseded once the new one is indexed with at
    least one chunk, so a quarantined or empty fetch never hides the source.
    If a source reverts to an earlier version, that version is reinstated.

    For each entry: fetches the URL (via :func:`~pipeline.acquisition.ingest_document`),
    compares the resulting sha256 against the currently-valid manifest, and on change
    indexes the new document and supersedes the prior version.

    Returns a list of result dicts, one per source:
    ``{"url": ..., "label": ..., "status": "unchanged"|"new"|"updated"|"error", ...}``
    """
    if not watch_file.exists():
        logger.warning("Watch file not found: %s", watch_file)
        return []

    sources: list[dict[str, Any]] = json.loads(watch_file.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    total = len(sources)
    if progress_cb is not None:
        progress_cb(0, total, "watch started")

    for idx, entry in enumerate(sources, start=1):
        url: str = entry["url"]
        doc_type: str = entry.get("doc_type", "web_page")
        label: str = entry.get("label", url)

        logger.info("Checking %s (%s)", label, url)
        prior_doc_id = _find_current_doc_for_url(url, config, store=store)

        try:
            extra = options().kwargs() if options is not None else {}
            new_doc_id, folder = ingest_document(
                url,
                config,
                doc_type=doc_type,
                store=store,
                url_only=True,
                project=project,
                **extra,
            )
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", url, exc)
            results.append({"url": url, "label": label, "status": "error", "error": str(exc)})
            if progress_cb is not None:
                progress_cb(idx, total, f"checked {idx}/{total} sources")
            continue

        if new_doc_id == prior_doc_id:
            logger.info("No change: %s", url)
            results.append({"url": url, "label": label, "status": "unchanged"})
            if progress_cb is not None:
                progress_cb(idx, total, f"checked {idx}/{total} sources")
            continue

        reinstated_from: str | None = None
        try:
            manifest = _load_manifest(new_doc_id, folder, store)
            if manifest.get("valid_until"):
                reinstated_from = str(manifest["valid_until"])
                # Content reverted to a version we already hold (sha256 dedup
                # returned it): make it current again before superseding.
                reinstate_document(new_doc_id, config, vector_store, fts_db, store=store)
                manifest = _load_manifest(new_doc_id, folder, store)
            if folder is None and store is not None:
                body = store.load_body(new_doc_id)
                chunks = index_document(
                    None,
                    config,
                    embedder,
                    vector_store,
                    fts_db,
                    doc_data=(manifest, body),
                )
                store.update_manifest(new_doc_id, chunk_count=len(chunks))
            else:
                chunks = index_document(folder, config, embedder, vector_store, fts_db)
        except Exception as exc:
            logger.error("Indexing failed for %s: %s", url, exc)
            if reinstated_from is not None:
                # Undo the reinstatement so only the prior version stays current.
                supersede_document(
                    new_doc_id, reinstated_from, config, vector_store, fts_db, store=store
                )
            results.append({"url": url, "label": label, "status": "error", "error": str(exc)})
            if progress_cb is not None:
                progress_cb(idx, total, f"checked {idx}/{total} sources")
            continue

        if not chunks:
            status = str(manifest.get("status", "ok"))
            reason = "quarantined for review" if status != "ok" else "produced no chunks"
            logger.warning("New version of %s %s — prior version kept current", url, reason)
            results.append(
                {
                    "url": url,
                    "label": label,
                    "status": "review" if status != "ok" else "error",
                    "doc_id": new_doc_id[:8],
                    "error": f"new version {reason}; prior version kept",
                }
            )
            if progress_cb is not None:
                progress_cb(idx, total, f"checked {idx}/{total} sources")
            continue

        if prior_doc_id is not None:
            today = datetime.now(UTC).date().isoformat()
            supersede_document(prior_doc_id, today, config, vector_store, fts_db, store=store)
            logger.info("Updated %s: %s → %s", label, prior_doc_id[:8], new_doc_id[:8])
            results.append(
                {
                    "url": url,
                    "label": label,
                    "status": "updated",
                    "new_doc_id": new_doc_id[:8],
                    "prior_doc_id": prior_doc_id[:8],
                }
            )
        else:
            logger.info("New source ingested: %s (%s)", label, new_doc_id[:8])
            results.append({"url": url, "label": label, "status": "new", "doc_id": new_doc_id[:8]})

        if progress_cb is not None:
            progress_cb(idx, total, f"checked {idx}/{total} sources")

    return results
