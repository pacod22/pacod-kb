"""Versioning: supersession, change detection, and full index rebuild."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.document_store import DocumentStore
    from pipeline.indexer import Embedder, FtsStore
    from pipeline.projects import ProjectStore
    from pipeline.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def find_doc_folder(doc_id: str, config: Config, *, prefix: bool = False) -> Path | None:
    """Return the document folder for *doc_id* (looked up by its 8-char folder prefix).

    With prefix=True, matches any doc whose doc_id starts with *doc_id*
    (useful for the 8-char short prefixes shown in CLI output); raises
    ``InvalidDocPrefixError`` / ``AmbiguousDocPrefixError`` like ``find_by_prefix``.
    """
    docs_dir = config.documents_dir
    if prefix:
        from pipeline.document_store import scan_folders_by_prefix

        return scan_folders_by_prefix(docs_dir, doc_id)
    from pipeline.fs_layout import find_folder_by_doc_id

    return find_folder_by_doc_id(docs_dir, doc_id)


def supersede_document(
    prior_doc_id: str,
    valid_until: str,
    config: Config,
    vector_store: VectorStore,
    fts_db: FtsStore,
    *,
    store: DocumentStore | None = None,
) -> None:
    """Mark the prior document version as superseded.

    Sets *valid_until* on:
      - prior ``document.md`` front matter
      - prior ``manifest.json``
      - keyword-index rows for that doc_id
      - Qdrant point payloads for that doc_id

    The prior version's chunks **stay indexed** (hard rule 9 — never delete
    historical records; filter them with ``valid_until`` instead).
    """
    if _apply_valid_until(
        prior_doc_id,
        valid_until,
        config,
        vector_store,
        fts_db,
        store=store,
        event=("superseded", f"valid_until={valid_until}"),
    ):
        logger.info("Superseded %s  valid_until=%s", prior_doc_id[:8], valid_until)


def reinstate_document(
    doc_id: str,
    config: Config,
    vector_store: VectorStore,
    fts_db: FtsStore,
    *,
    store: DocumentStore | None = None,
) -> None:
    """Make a previously superseded version current again (clears ``valid_until``).

    Used when a watched source reverts to content that was already ingested:
    the sha256 dedup returns the old document, which must become current
    before the version it replaces is superseded.
    """
    if _apply_valid_until(
        doc_id,
        None,
        config,
        vector_store,
        fts_db,
        store=store,
        event=("reinstated", "valid_until cleared"),
    ):
        logger.info("Reinstated %s as current version", doc_id[:8])


def _apply_valid_until(
    doc_id: str,
    valid_until: str | None,
    config: Config,
    vector_store: VectorStore,
    fts_db: FtsStore,
    *,
    store: DocumentStore | None,
    event: tuple[str, str],
) -> bool:
    """Write *valid_until* to all four locations. Returns False if doc not found."""
    if store is not None:
        # Database mode: update via store
        store.update_manifest(doc_id, valid_until=valid_until)
        store.append_event(doc_id, *event)
    else:
        # Filesystem mode: update files
        from pipeline.normalize import assemble_document, strip_front_matter

        folder = find_doc_folder(doc_id, config)
        if folder is None:
            logger.warning("Doc %s not found in archive — skipping valid_until update", doc_id[:8])
            return False

        doc_md_path = folder / "document.md"
        original_text = doc_md_path.read_text(encoding="utf-8")
        front, body_before = strip_front_matter(original_text)
        front["valid_until"] = valid_until
        new_doc_text = assemble_document(front, body_before)

        _, body_after = strip_front_matter(new_doc_text)
        if body_before != body_after:
            raise RuntimeError(f"Body changed during valid_until update for {doc_id[:8]}")

        doc_md_path.write_text(new_doc_text, encoding="utf-8")

        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["valid_until"] = valid_until
        from pipeline.fs_layout import write_json_atomic

        write_json_atomic(manifest_path, manifest)

        from pipeline.events import append_event

        append_event(manifest_path, *event)

    # --- Update index records in both stores (no deletion, same in both modes) ---
    fts_db.set_valid_until(doc_id, valid_until)
    vector_store.set_valid_until(doc_id, valid_until)
    return True


def delete_document(
    doc_id: str,
    folder: Path | None,
    vector_store: VectorStore,
    fts_db: FtsStore,
    *,
    store: DocumentStore | None = None,
) -> None:
    """Permanently remove a document: wipes index entries and the archive folder."""
    vector_store.delete_by_doc(doc_id)
    fts_db.delete_by_doc(doc_id)
    if store is not None:
        store.delete(doc_id)
    elif folder is not None:
        import shutil

        shutil.rmtree(folder)
    logger.info("Deleted document %s", doc_id[:8])


_LEGACY_TAGS = (None, "", "default")


def _project_of(manifest: dict[str, Any], default_project: str | None) -> str | None:
    tag = manifest.get("project")
    return default_project if tag in _LEGACY_TAGS else str(tag)


def rebuild_index(
    config: Config,
    embedder: Embedder,
    vector_store: VectorStore,
    fts_db: FtsStore,
    *,
    store: DocumentStore | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    project: str | None = None,
    default_project: str | None = None,
    reuse_vectors: bool = True,
) -> tuple[int, int]:
    """Drop and regenerate the index from the documents archive.

    With *reuse_vectors* (and EMBED_REUSE), the collection's vectors are snapshotted
    before it is dropped, so chunks whose embedded text is unchanged are not embedded
    again (up to REBUILD_REUSE_MAX_POINTS of them).

    With *project*, only that project's documents are re-indexed into
    *vector_store* (which must be the project's collection) and only that
    project's keyword rows are dropped — other projects are untouched.
    Untagged / legacy ``"default"``-tagged documents belong to
    *default_project*.  Without *project*, every document is re-indexed into
    *vector_store* (only correct for single-project setups).

    Returns ``(success_count, error_count)``.  Documents are processed in
    deterministic order.
    """
    from pipeline.indexer import index_document

    def _wanted(manifest: dict[str, Any]) -> bool:
        return project is None or _project_of(manifest, default_project) == project

    cache: dict[Any, Any] = {}
    snapshot = getattr(vector_store, "snapshot_vectors", None)
    model = getattr(embedder, "model_name", None)
    if reuse_vectors and config.embed_reuse and snapshot is not None and model:
        try:
            cache = snapshot(
                model=model,
                allow_untagged=config.embed_reuse_untagged,
                limit=config.rebuild_reuse_max_points,
            )
            logger.info("Rebuild: %d vectors kept for reuse", len(cache))
        except Exception as exc:  # noqa: BLE001 — rebuild embeds everything instead
            logger.warning("Rebuild: vector snapshot failed (embedding all): %s", exc)

    vector_store.drop_all()
    if project is None:
        fts_db.drop_all()
    else:
        fts_db.drop_by_project(project)
        if project == default_project:
            fts_db.drop_by_project("default")

    ok = err = 0

    if store is not None:
        # Database mode: iterate manifests from the store
        manifests = sorted(
            (m for m in store.list_all() if _wanted(m)),
            key=lambda x: str(x.get("doc_id", "")),
        )
        total = len(manifests)
        if progress_cb is not None:
            progress_cb(0, total, "rebuild started")
        for idx, m in enumerate(manifests, start=1):
            doc_id = str(m.get("doc_id", ""))
            try:
                body = store.load_body(doc_id)
                chunks = index_document(
                    None,
                    config,
                    embedder,
                    vector_store,
                    fts_db,
                    doc_data=(m, body),
                    default_project=default_project,
                    vector_cache=cache,
                )
                store.update_manifest(doc_id, chunk_count=len(chunks))
                ok += 1
            except Exception as exc:
                logger.error("rebuild failed for %s: %s", doc_id[:8], exc)
                err += 1
            if progress_cb is not None:
                progress_cb(idx, total, f"indexed {idx}/{total} documents")
    else:
        # Filesystem mode
        docs_dir = config.documents_dir
        if not docs_dir.exists():
            logger.info("No documents directory — nothing to rebuild")
            if progress_cb is not None:
                progress_cb(0, 0, "no documents")
            return 0, 0

        folders: list[Path] = []
        for folder in sorted(docs_dir.iterdir()):
            if not (folder / "document.md").exists():
                continue
            if project is not None:
                try:
                    mf = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("rebuild: unreadable manifest in %s — skipped", folder.name)
                    continue
                if not _wanted(mf):
                    continue
            folders.append(folder)
        total = len(folders)
        if progress_cb is not None:
            progress_cb(0, total, "rebuild started")

        for idx, folder in enumerate(folders, start=1):
            try:
                index_document(
                    folder,
                    config,
                    embedder,
                    vector_store,
                    fts_db,
                    default_project=default_project,
                    vector_cache=cache,
                )
                ok += 1
            except Exception as exc:
                logger.error("rebuild failed for %s: %s", folder.name, exc)
                err += 1
            if progress_cb is not None:
                progress_cb(idx, total, f"indexed {idx}/{total} documents")

    logger.info("Rebuild complete (project=%s): %d ok, %d errors", project or "*", ok, err)
    return ok, err


def rebuild_all_projects(
    config: Config,
    embedder: Embedder,
    qdrant_client: Any,
    fts_db: FtsStore,
    registry: ProjectStore,
    *,
    store: DocumentStore | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    reuse_vectors: bool = True,
) -> tuple[int, int]:
    """Rebuild every registered project into its own Qdrant collection."""
    from pipeline.vectorstore import QdrantVectorStore

    projects = registry.list()
    if not projects:
        return 0, 0
    default_slug = projects[0].slug
    ok = err = 0
    for p in projects:
        if progress_cb is not None:
            progress_cb(0, 0, f"rebuilding project {p.slug}")
        p_ok, p_err = rebuild_index(
            config,
            embedder,
            QdrantVectorStore(qdrant_client, p.collection_name),
            fts_db,
            store=store,
            progress_cb=progress_cb,
            project=p.slug,
            default_project=default_slug,
            reuse_vectors=reuse_vectors,
        )
        ok += p_ok
        err += p_err
    return ok, err
