"""Project registry.

Two backends, selected by ``KB_STORAGE_BACKEND`` via :func:`make_project_registry`:

* :class:`ProjectRegistry` — filesystem JSON (``KB_ROOT/projects.json``), the
  default for filesystem storage mode.
* :class:`DatabaseProjectRegistry` — PostgreSQL/SQLite ``kb_projects`` table,
  the single source of truth in database storage mode. Projects are fully
  isolated: each owns a Qdrant collection and documents are scoped by slug.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    """Convert a display name to a kebab-case slug suitable as a Qdrant collection name."""
    s = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    s = re.sub(r"\s+", "-", s).strip("-")
    return (s or "project")[:50]


@dataclass
class Project:
    slug: str
    name: str
    collection_name: str
    created_at: str
    description: str = field(default="")


class ProjectStore:
    """Common interface implemented by both project registry backends."""

    def list(self) -> list[Project]:
        raise NotImplementedError

    def get(self, slug: str) -> Project | None:
        raise NotImplementedError

    def create(self, name: str, description: str = "", collection_name: str = "") -> Project:
        raise NotImplementedError

    def delete(self, slug: str) -> None:
        raise NotImplementedError

    def update_collection_name(self, slug: str, collection_name: str) -> Project:
        raise NotImplementedError

    def ensure_default(self, collection_name: str) -> Project:
        raise NotImplementedError


class ProjectRegistry(ProjectStore):
    """Persistent project registry backed by KB_ROOT/projects.json."""

    def __init__(self, kb_root: Path) -> None:
        self._path = kb_root / "projects.json"
        self._projects: dict[str, Project] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
            self._projects = {}
            for key, value in raw.items():
                item = dict(value)
                item.setdefault("slug", key)
                # Backward-compat: legacy entries had no collection_name.
                if not item.get("collection_name"):
                    item["collection_name"] = str(item.get("slug") or key)
                self._projects[key] = Project(**item)
        except Exception as exc:
            # Never fall through to an empty registry: the next _save() would
            # overwrite the (recoverable) file. Keep a copy and fail loudly.
            backup = self._path.with_name(
                f"{self._path.name}.corrupt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
            )
            shutil.copy2(self._path, backup)
            raise ValueError(
                f"{self._path} is unreadable ({exc}); a copy was saved to {backup.name}. "
                "Fix or remove the file and restart."
            ) from exc

    def _save(self) -> None:
        data = {slug: asdict(p) for slug, p in self._projects.items()}
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def list(self) -> list[Project]:
        return list(self._projects.values())

    def get(self, slug: str) -> Project | None:
        return self._projects.get(slug)

    def create(self, name: str, description: str = "", collection_name: str = "") -> Project:
        slug = _slug(name)
        collection = _slug(collection_name) if collection_name.strip() else slug
        if slug in self._projects:
            raise ValueError(f"Project '{slug}' already exists")
        if any(p.collection_name == collection for p in self._projects.values()):
            raise ValueError(f"Collection '{collection}' is already assigned to another project")
        p = Project(
            slug=slug,
            name=name,
            collection_name=collection,
            created_at=datetime.now(UTC).isoformat(),
            description=description,
        )
        self._projects[slug] = p
        self._save()
        return p

    def delete(self, slug: str) -> None:
        if slug not in self._projects:
            raise KeyError(f"Project '{slug}' not found")
        del self._projects[slug]
        self._save()

    def update_collection_name(self, slug: str, collection_name: str) -> Project:
        project = self._projects.get(slug)
        if project is None:
            raise KeyError(f"Project '{slug}' not found")
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")
        normalized = _slug(collection_name)
        if normalized == project.collection_name:
            return project
        if any(p.slug != slug and p.collection_name == normalized for p in self._projects.values()):
            raise ValueError(f"Collection '{normalized}' is already assigned to another project")
        project.collection_name = normalized
        self._projects[slug] = project
        self._save()
        return project

    def ensure_default(self, collection_name: str) -> Project:
        """Return the first project, creating a default one from collection_name if empty."""
        if not self._projects:
            default_collection = _slug(collection_name) if collection_name else "kb-default"
            slug = default_collection
            p = Project(
                slug=slug,
                name=collection_name or "Default",
                collection_name=default_collection,
                created_at=datetime.now(UTC).isoformat(),
            )
            self._projects[slug] = p
            self._save()
        return next(iter(self._projects.values()))


class DatabaseProjectRegistry(ProjectStore):
    """Project registry backed by the ``kb_projects`` table (database mode)."""

    def __init__(self, session_factory: Any) -> None:
        self._factory = session_factory

    def _to_project(self, row: Any) -> Project:
        return Project(
            slug=row.slug,
            name=row.name,
            collection_name=row.collection_name,
            created_at=row.created_at,
            description=row.description or "",
        )

    def list(self) -> list[Project]:
        from sqlalchemy import select

        from pipeline.db_models import KbProject
        from pipeline.document_store import _session_scope

        with _session_scope(self._factory) as s:
            rows = s.execute(select(KbProject).order_by(KbProject.created_at)).scalars().all()
            return [self._to_project(r) for r in rows]

    def get(self, slug: str) -> Project | None:
        from pipeline.db_models import KbProject
        from pipeline.document_store import _session_scope

        with _session_scope(self._factory) as s:
            row = s.get(KbProject, slug)
            return self._to_project(row) if row is not None else None

    def create(self, name: str, description: str = "", collection_name: str = "") -> Project:
        from sqlalchemy import select

        from pipeline.db_models import KbProject
        from pipeline.document_store import _session_scope

        slug = _slug(name)
        collection = _slug(collection_name) if collection_name.strip() else slug
        with _session_scope(self._factory) as s:
            if s.get(KbProject, slug) is not None:
                raise ValueError(f"Project '{slug}' already exists")
            clash = s.execute(
                select(KbProject).where(KbProject.collection_name == collection)
            ).scalar_one_or_none()
            if clash is not None:
                raise ValueError(
                    f"Collection '{collection}' is already assigned to another project"
                )
            created_at = datetime.now(UTC).isoformat()
            s.add(
                KbProject(
                    slug=slug,
                    name=name,
                    collection_name=collection,
                    created_at=created_at,
                    description=description,
                )
            )
            return Project(
                slug=slug,
                name=name,
                collection_name=collection,
                created_at=created_at,
                description=description,
            )

    def delete(self, slug: str) -> None:
        from pipeline.db_models import KbProject
        from pipeline.document_store import _session_scope

        with _session_scope(self._factory) as s:
            row = s.get(KbProject, slug)
            if row is None:
                raise KeyError(f"Project '{slug}' not found")
            s.delete(row)

    def update_collection_name(self, slug: str, collection_name: str) -> Project:
        from sqlalchemy import select

        from pipeline.db_models import KbProject
        from pipeline.document_store import _session_scope

        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")
        normalized = _slug(collection_name)
        with _session_scope(self._factory) as s:
            row = s.get(KbProject, slug)
            if row is None:
                raise KeyError(f"Project '{slug}' not found")
            if normalized == row.collection_name:
                return self._to_project(row)
            clash = s.execute(
                select(KbProject).where(
                    KbProject.collection_name == normalized,
                    KbProject.slug != slug,
                )
            ).scalar_one_or_none()
            if clash is not None:
                raise ValueError(
                    f"Collection '{normalized}' is already assigned to another project"
                )
            row.collection_name = normalized
            return self._to_project(row)

    def ensure_default(self, collection_name: str) -> Project:
        from sqlalchemy import select

        from pipeline.db_models import KbProject
        from pipeline.document_store import _session_scope

        with _session_scope(self._factory) as s:
            first = s.execute(select(KbProject).order_by(KbProject.created_at)).scalars().first()
            if first is not None:
                return self._to_project(first)
            default_collection = _slug(collection_name) if collection_name else "kb-default"
            created_at = datetime.now(UTC).isoformat()
            s.add(
                KbProject(
                    slug=default_collection,
                    name=collection_name or "Default",
                    collection_name=default_collection,
                    created_at=created_at,
                    description="",
                )
            )
            return Project(
                slug=default_collection,
                name=collection_name or "Default",
                collection_name=default_collection,
                created_at=created_at,
            )

    def import_from_json(self, path: Path) -> int:
        """One-time migration of a legacy ``projects.json`` into the DB.

        Only runs when the table is empty. Returns the number of projects
        imported (0 when nothing to migrate).
        """
        if not path.exists() or self.list():
            return 0
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0

        from pipeline.db_models import KbProject
        from pipeline.document_store import _session_scope

        imported = 0
        with _session_scope(self._factory) as s:
            for key, value in raw.items():
                item = dict(value)
                slug = str(item.get("slug") or key)
                collection = str(item.get("collection_name") or slug)
                s.add(
                    KbProject(
                        slug=slug,
                        name=str(item.get("name") or slug),
                        collection_name=collection,
                        created_at=str(item.get("created_at") or datetime.now(UTC).isoformat()),
                        description=str(item.get("description") or ""),
                    )
                )
                imported += 1
        if imported:
            logger.info("Imported %d project(s) from %s into the database", imported, path)
        return imported


def make_project_registry(config: Any, session_factory: Any | None = None) -> ProjectStore:
    """Return the project registry for the active storage backend.

    In database mode the ``kb_projects`` table is authoritative and a legacy
    ``projects.json`` (if present) is migrated in once. In filesystem mode the
    JSON-backed :class:`ProjectRegistry` is used.
    """
    backend = getattr(config, "storage_backend", "filesystem")
    if backend == "database" and session_factory is not None:
        registry = DatabaseProjectRegistry(session_factory)
        try:
            registry.import_from_json(Path(config.kb_root) / "projects.json")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("projects.json migration skipped: %s", exc)
        return registry
    return ProjectRegistry(Path(config.kb_root))


# ---------------------------------------------------------------------------
# Default-project resolution + legacy "default" tag migration
# ---------------------------------------------------------------------------

LEGACY_DEFAULT_TAG = "default"


def default_project_slug(config: Any, session_factory: Any | None = None) -> str:
    """Return the default (first) project's slug, creating it if none exist.

    In database mode a *session_factory* is required — falling back to the
    JSON registry there would silently create a stray ``projects.json``.
    """
    if getattr(config, "storage_backend", "filesystem") == "database" and session_factory is None:
        raise ValueError("database mode needs a session factory to resolve the default project")
    registry = make_project_registry(config, session_factory)
    return registry.ensure_default(config.qdrant_collection).slug


def retag_legacy_default_project(
    registry: ProjectStore,
    default_slug: str,
    store: Any,
    fts_db: Any,
    vector_store: Any | None = None,
) -> dict[str, int]:
    """Re-tag documents/chunks carrying the literal ``"default"`` project tag.

    Older dropzone/watch/CLI ingests stamped ``project="default"`` although the
    real default project's slug is derived from ``QDRANT_COLLECTION``; those
    documents were invisible to project-scoped listings and keyword search.
    Documents with no project at all are claimed by the default project too.

    Idempotent, and a no-op when a real project named ``"default"`` exists.
    Returns counts of updated documents and FTS rows.
    """
    counts = {"documents": 0, "fts_rows": 0}
    if default_slug == LEGACY_DEFAULT_TAG or registry.get(LEGACY_DEFAULT_TAG) is not None:
        return counts
    for manifest in store.list_all():
        if manifest.get("project") in (None, "", LEGACY_DEFAULT_TAG):
            store.update_manifest(str(manifest["doc_id"]), project=default_slug)
            counts["documents"] += 1
    counts["fts_rows"] = int(fts_db.retag_project(LEGACY_DEFAULT_TAG, default_slug) or 0)
    if vector_store is not None:
        try:
            vector_store.retag_project(LEGACY_DEFAULT_TAG, default_slug)
        except Exception as exc:  # payload tag is informational only
            logger.warning("Qdrant project re-tag skipped: %s", exc)
    if counts["documents"] or counts["fts_rows"]:
        logger.info(
            "Re-tagged legacy 'default' project → %r: %d documents, %d FTS rows",
            default_slug,
            counts["documents"],
            counts["fts_rows"],
        )
    return counts
