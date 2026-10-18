"""Filesystem layout helpers for ``KB_ROOT/documents/<doc_id[:8]>_<slug>/``.

Every document folder name starts with the first 8 hex characters of its
doc_id, so lookups glob on that prefix instead of parsing every manifest in
the archive (which made each dedup check, supersede or event append O(N)).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FOLDER_PREFIX = 8


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON via a temp file + ``os.replace`` so a crash never truncates *path*."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_manifest(folder: Path) -> dict[str, Any] | None:
    """Parsed ``manifest.json`` of *folder*, or None if missing/unreadable."""
    mf = folder / "manifest.json"
    if not mf.exists():
        return None
    try:
        data: Any = json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable manifest %s: %s", mf, exc)
        return None
    return data if isinstance(data, dict) else None


def candidate_folders(docs_dir: Path, id_or_prefix: str) -> list[Path]:
    """Folders whose name starts with the first 8 chars of *id_or_prefix*."""
    if not docs_dir.exists() or len(id_or_prefix) < _FOLDER_PREFIX:
        return []
    return sorted(p for p in docs_dir.glob(f"{id_or_prefix[:_FOLDER_PREFIX]}*") if p.is_dir())


def find_folder_by_doc_id(docs_dir: Path, doc_id: str) -> Path | None:
    """Return the folder holding exactly *doc_id*, or None."""
    for folder in candidate_folders(docs_dir, doc_id):
        manifest = read_manifest(folder)
        if manifest is not None and manifest.get("doc_id") == doc_id:
            return folder
    return None


def new_doc_folder(docs_dir: Path, doc_id: str, slug: str) -> Path:
    """Folder path for a new document, never reusing another document's folder.

    Two documents sharing the 8-char prefix *and* the slug used to land in the
    same folder, silently overwriting the first one.
    """
    folder = docs_dir / f"{doc_id[:_FOLDER_PREFIX]}_{slug}"
    existing = read_manifest(folder) if folder.exists() else None
    if existing is not None and existing.get("doc_id") != doc_id:
        folder = docs_dir / f"{doc_id[:16]}_{slug}"
    return folder
