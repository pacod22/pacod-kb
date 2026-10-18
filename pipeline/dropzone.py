"""Dropzone: auto-ingest files placed in KB_ROOT/dropzone/.

Files are moved after processing:
  - dropzone/done/    on successful ingest or when already known (SHA-256 match)
  - dropzone/failed/  on parse / ingest error

Set DROPZONE_INTERVAL_SECS (env) to control the watcher poll interval (default 10).
Set DROPZONE_INTERVAL_SECS=0 to disable the watcher.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.acquisition import IngestOptions, ingest_document, sha256_of_bytes
from pipeline.indexer import index_document, make_fts_db

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.document_store import DocumentStore
    from pipeline.indexer import Embedder, FtsStore
    from pipeline.vectorstore import VectorStore

logger = logging.getLogger(__name__)

_STABILITY_SECS = 2


def _known_manifest(
    sha256: str, config: Config, store: DocumentStore | None
) -> dict[str, Any] | None:
    """Manifest of an already-ingested document with this sha256, or None."""
    if store is not None:
        if not store.sha256_exists(sha256):
            return None
        try:
            return store.load_manifest(sha256)
        except KeyError:
            return {}
    folder = _find_by_sha256(sha256, config)
    if folder is None:
        return None
    try:
        data: dict[str, Any] = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        return data
    except (OSError, ValueError):
        return {}


def _needs_index(manifest: dict[str, Any]) -> bool:
    return manifest.get("status") == "ok" and int(manifest.get("chunk_count") or 0) == 0


def _is_candidate(path: Path) -> bool:
    """Regular, non-hidden file (skips .DS_Store, editor temp files, ...)."""
    return path.is_file() and not path.name.startswith(".")


def _find_by_sha256(sha256: str, config: Config) -> Path | None:
    """Return the archive folder whose doc_id matches *sha256*, or None."""
    from pipeline.fs_layout import find_folder_by_doc_id

    return find_folder_by_doc_id(config.documents_dir, sha256)


def _move_safe(src: Path, dest_dir: Path) -> None:
    """Move *src* to *dest_dir*, appending _N if the name already exists."""
    dest = dest_dir / src.name
    if dest.exists():
        stem, suffix = src.stem, src.suffix
        n = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{n}{suffix}"
            n += 1
    shutil.move(str(src), str(dest))


def scan_dropzone(
    config: Config,
    embedder: Embedder,
    vector_store: VectorStore,
    fts_db: FtsStore,
    *,
    store: DocumentStore | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    project: str | None = None,
    options: Callable[[], IngestOptions] | None = None,
) -> list[dict[str, Any]]:
    """Single-pass scan of KB_ROOT/dropzone/.

    New documents are tagged with *project* (``None`` = the default project);
    *vector_store* must be that project's collection. *options*, when given,
    supplies the runtime switches (converter, security scan, language check)
    for each file; without it the env config applies (the CLI).

    For each regular file that has been stable for at least _STABILITY_SECS:
    - If SHA-256 matches an existing document: log skip, move to done/.
    - If new: call ingest_document + index_document, move to done/.
    - On error: move to failed/.

    Returns a list of result dicts, one per candidate file processed.
    """
    dropzone_dir = config.kb_dropzone_dir
    if not dropzone_dir.exists():
        return []

    done_dir = config.kb_done_dir
    failed_dir = config.kb_failed_dir
    done_dir.mkdir(exist_ok=True)
    failed_dir.mkdir(exist_ok=True)

    now = time.time()
    candidates = sorted(p for p in dropzone_dir.iterdir() if _is_candidate(p))
    stable_candidates: list[Path] = []
    for file_path in candidates:
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            continue
        if now - mtime < _STABILITY_SECS:
            logger.debug("Dropzone: %s is too new — deferring to next scan", file_path.name)
            continue
        stable_candidates.append(file_path)

    results: list[dict[str, Any]] = []
    total = len(stable_candidates)
    if progress_cb is not None:
        progress_cb(0, total, "dropzone scan started")

    for idx, file_path in enumerate(stable_candidates, start=1):
        try:
            source_bytes = file_path.read_bytes()
        except OSError as exc:
            logger.warning("Dropzone: cannot read %s: %s", file_path.name, exc)
            continue

        sha256 = sha256_of_bytes(source_bytes)
        known = _known_manifest(sha256, config, store)

        # A known document that was acquired but never indexed (e.g. indexing
        # failed earlier) falls through: ingest dedups and we index it now.
        if known is not None and not _needs_index(known):
            logger.info(
                "Dropzone: %s already ingested as doc_id=%s — skipping",
                file_path.name,
                sha256[:8],
            )
            _move_safe(file_path, done_dir)
            results.append({"file": file_path.name, "status": "skipped", "doc_id": sha256[:8]})
            if progress_cb is not None:
                progress_cb(idx, total, f"processed {idx}/{total} files")
            continue

        ingested = False
        try:
            extra = options().kwargs() if options is not None else {}
            doc_id, folder = ingest_document(
                str(file_path), config, store=store, project=project, **extra
            )
            ingested = True
            if folder is None and store is not None:
                manifest = store.load_manifest(doc_id)
                body = store.load_body(doc_id)
                chunks = index_document(
                    None,
                    config,
                    embedder,
                    vector_store,
                    fts_db,
                    doc_data=(manifest, body),
                )
                store.update_manifest(doc_id, chunk_count=len(chunks))
            else:
                chunks = index_document(folder, config, embedder, vector_store, fts_db)
            logger.info(
                "Dropzone: %s ingested → doc_id=%s (%d chunks)",
                file_path.name,
                doc_id[:8],
                len(chunks),
            )
            _move_safe(file_path, done_dir)
            results.append(
                {
                    "file": file_path.name,
                    "status": "ingested",
                    "doc_id": doc_id[:8],
                    "chunks": len(chunks),
                }
            )
        except Exception as exc:
            if ingested:
                logger.error(
                    "Dropzone: %s was acquired but indexing failed: %s", file_path.name, exc
                )
            else:
                logger.error("Dropzone: failed to ingest %s: %s", file_path.name, exc)
            _move_safe(file_path, failed_dir)
            results.append({"file": file_path.name, "status": "error", "error": str(exc)})

        if progress_cb is not None:
            progress_cb(idx, total, f"processed {idx}/{total} files")

    return results


def watch_dropzone_loop(
    config: Config,
    embedder: Embedder,
    vector_store: VectorStore,
    stop_event: threading.Event,
    mutation_lock: threading.Lock,
    interval_secs: int,
    on_scan_done: Callable[[list[dict[str, Any]]], None] | None = None,
    on_scan_error: Callable[[str], None] | None = None,
    *,
    store: DocumentStore | None = None,
    target: Callable[[], tuple[VectorStore, str | None] | None] | None = None,
    options: Callable[[], IngestOptions] | None = None,
) -> None:
    """Long-running daemon loop: scan dropzone every *interval_secs* seconds.

    *target*, when given, is called before each scan and returns the
    ``(vector_store, project_slug)`` to ingest into, so a project switch made
    after startup is honoured; ``None`` skips the scan (no valid write target).
    Without it, *vector_store* and the default project are used.  *options*
    supplies the dashboard's switches per scan.

    Skips the scan when *mutation_lock* is held by another operation (non-blocking
    acquire), so the watcher never races with a concurrent rebuild or reindex.

    *on_scan_done* is called with the result list after each scan that processes at
    least one file.  *on_scan_error* is called with the error message if the scan
    itself raises an unexpected exception.  Both are optional.
    """
    logger.info(
        "Dropzone watcher started (interval=%ds, dir=%s)",
        interval_secs,
        config.kb_dropzone_dir,
    )
    while not stop_event.wait(interval_secs):
        dz = config.kb_dropzone_dir
        if not dz.exists() or not any(_is_candidate(p) for p in dz.iterdir()):
            continue

        if not mutation_lock.acquire(blocking=False):
            logger.debug("Dropzone: skipping scan — another operation in progress")
            continue
        try:
            scan_target = target() if target is not None else (vector_store, None)
            if scan_target is None:
                logger.debug("Dropzone: skipping scan — no project to ingest into")
                continue
            scan_vs, scan_project = scan_target
            fts_db = make_fts_db(config)
            try:
                results = scan_dropzone(
                    config,
                    embedder,
                    scan_vs,
                    fts_db,
                    store=store,
                    project=scan_project,
                    options=options,
                )
            finally:
                fts_db.close()
            if results and on_scan_done is not None:
                on_scan_done(results)
        except Exception as exc:
            logger.error("Dropzone watcher error: %s", exc)
            if on_scan_error is not None:
                on_scan_error(str(exc))
        finally:
            mutation_lock.release()
    logger.info("Dropzone watcher stopped")
