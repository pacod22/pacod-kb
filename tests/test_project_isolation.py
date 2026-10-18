"""Isolation + database-backed project registry regression tests.

Covers the Postgres-mode fixes that make projects fully isolated and stored in
the database:

* ``DatabaseDocumentStore`` persists and returns the ``project`` field so
  documents can be filtered per project.
* ``DatabaseProjectRegistry`` stores projects in the ``kb_projects`` table with
  the same semantics as the filesystem registry.
* ``make_project_registry`` selects the backend and migrates a legacy
  ``projects.json`` into the database once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine

from pipeline.document_store import DatabaseDocumentStore
from pipeline.projects import (
    DatabaseProjectRegistry,
    ProjectRegistry,
    make_project_registry,
)

_FRONT_MATTER: dict[str, Any] = {
    "doc_id": "a" * 64,
    "title": "Alpha Doc",
    "source_file": "a.md",
    "source_uri": "https://example.com/a",
    "source_version": "v1",
    "source_sha256": "a" * 64,
    "retrieved_at": "2026-01-01T00:00:00+00:00",
    "converted_at": "2026-01-01T00:00:01+00:00",
    "converter": "passthrough/1.0",
    "extraction_method": "deterministic",
    "language": "en",
    "doc_type": "generic",
    "supersedes": None,
    "valid_until": None,
    "project": "alpha",
}

_BODY = "# Heading\n\nbody\n"
_BYTES = b"raw"
_EXT = ".md"


@pytest.fixture()
def engine(tmp_path: Path) -> Any:
    return create_engine(f"sqlite:///{tmp_path}/kb.db")


@pytest.fixture()
def db_store(engine: Any) -> DatabaseDocumentStore:
    return DatabaseDocumentStore(engine)


def _fm(**over: Any) -> dict[str, Any]:
    return {**_FRONT_MATTER, **over}


# ---------------------------------------------------------------------------
# DatabaseDocumentStore project persistence
# ---------------------------------------------------------------------------


class TestDatabaseDocumentStoreProject:
    def test_project_round_trips_in_manifest(self, db_store: DatabaseDocumentStore) -> None:
        db_store.save(_FRONT_MATTER["doc_id"], _fm(), _BODY, _BYTES, _EXT)
        m = db_store.load_manifest(_FRONT_MATTER["doc_id"])
        assert m["project"] == "alpha"

    def test_project_present_in_list_all(self, db_store: DatabaseDocumentStore) -> None:
        db_store.save(_FRONT_MATTER["doc_id"], _fm(), _BODY, _BYTES, _EXT)
        docs = db_store.list_all()
        assert docs[0]["project"] == "alpha"

    def test_documents_isolated_per_project(self, db_store: DatabaseDocumentStore) -> None:
        doc_a = "a" * 64
        doc_b = "b" * 64
        db_store.save(
            doc_a,
            _fm(doc_id=doc_a, source_sha256=doc_a, project="alpha"),
            _BODY,
            _BYTES,
            _EXT,
        )
        db_store.save(
            doc_b,
            _fm(
                doc_id=doc_b,
                source_sha256=doc_b,
                source_uri="https://example.com/b",
                project="beta",
            ),
            _BODY,
            _BYTES,
            _EXT,
        )
        docs = db_store.list_all()
        alpha = [d for d in docs if d["project"] == "alpha"]
        beta = [d for d in docs if d["project"] == "beta"]
        assert {d["doc_id"] for d in alpha} == {doc_a}
        assert {d["doc_id"] for d in beta} == {doc_b}

    def test_missing_project_is_none(self, db_store: DatabaseDocumentStore) -> None:
        fm = _fm()
        del fm["project"]
        db_store.save(_FRONT_MATTER["doc_id"], fm, _BODY, _BYTES, _EXT)
        assert db_store.load_manifest(_FRONT_MATTER["doc_id"])["project"] is None


# ---------------------------------------------------------------------------
# DatabaseProjectRegistry
# ---------------------------------------------------------------------------


class TestDatabaseProjectRegistry:
    @pytest.fixture()
    def registry(self, db_store: DatabaseDocumentStore) -> DatabaseProjectRegistry:
        # Sharing the store's factory guarantees the kb_projects table exists.
        return DatabaseProjectRegistry(db_store._factory)

    def test_create_and_get(self, registry: DatabaseProjectRegistry) -> None:
        created = registry.create("Legal Research", collection_name="eu-legal")
        assert created.slug == "legal-research"
        assert created.collection_name == "eu-legal"
        fetched = registry.get("legal-research")
        assert fetched is not None
        assert fetched.collection_name == "eu-legal"

    def test_create_defaults_collection_to_slug(self, registry: DatabaseProjectRegistry) -> None:
        created = registry.create("Policy Drafts")
        assert created.collection_name == "policy-drafts"

    def test_create_rejects_duplicate_collection(self, registry: DatabaseProjectRegistry) -> None:
        registry.create("Team One", collection_name="shared")
        with pytest.raises(ValueError, match="already assigned"):
            registry.create("Team Two", collection_name="shared")

    def test_delete(self, registry: DatabaseProjectRegistry) -> None:
        registry.create("Temp")
        registry.delete("temp")
        assert registry.get("temp") is None

    def test_update_collection_name(self, registry: DatabaseProjectRegistry) -> None:
        registry.create("Team One")
        updated = registry.update_collection_name("team-one", "team-one-v2")
        assert updated.collection_name == "team-one-v2"
        assert registry.get("team-one").collection_name == "team-one-v2"  # type: ignore[union-attr]

    def test_update_collection_rejects_duplicate(self, registry: DatabaseProjectRegistry) -> None:
        registry.create("Alpha", collection_name="alpha-col")
        registry.create("Beta", collection_name="beta-col")
        with pytest.raises(ValueError, match="already assigned"):
            registry.update_collection_name("alpha", "beta-col")

    def test_ensure_default_creates_once(self, registry: DatabaseProjectRegistry) -> None:
        first = registry.ensure_default("kb_chunks")
        again = registry.ensure_default("kb_chunks")
        assert first.slug == again.slug
        assert len(registry.list()) == 1

    def test_persistence_across_instances(self, db_store: DatabaseDocumentStore) -> None:
        DatabaseProjectRegistry(db_store._factory).create("Persisted")
        reloaded = DatabaseProjectRegistry(db_store._factory).get("persisted")
        assert reloaded is not None


# ---------------------------------------------------------------------------
# make_project_registry factory + migration
# ---------------------------------------------------------------------------


class TestMakeProjectRegistry:
    def test_filesystem_backend(self, kb_root: Path) -> None:
        class _Cfg:
            storage_backend = "filesystem"

        cfg = _Cfg()
        cfg.kb_root = kb_root  # type: ignore[attr-defined]
        registry = make_project_registry(cfg, None)
        assert isinstance(registry, ProjectRegistry)

    def test_database_backend(self, kb_root: Path, db_store: DatabaseDocumentStore) -> None:
        class _Cfg:
            storage_backend = "database"

        cfg = _Cfg()
        cfg.kb_root = kb_root  # type: ignore[attr-defined]
        registry = make_project_registry(cfg, db_store._factory)
        assert isinstance(registry, DatabaseProjectRegistry)

    def test_migrates_projects_json_once(
        self, kb_root: Path, db_store: DatabaseDocumentStore
    ) -> None:
        (kb_root / "projects.json").write_text(
            json.dumps(
                {
                    "legacy": {
                        "slug": "legacy",
                        "name": "Legacy",
                        "collection_name": "legacy-col",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "description": "old",
                    }
                }
            ),
            encoding="utf-8",
        )

        class _Cfg:
            storage_backend = "database"

        cfg = _Cfg()
        cfg.kb_root = kb_root  # type: ignore[attr-defined]

        registry = make_project_registry(cfg, db_store._factory)
        migrated = registry.get("legacy")
        assert migrated is not None
        assert migrated.collection_name == "legacy-col"

        # A second factory call must not duplicate the imported project.
        make_project_registry(cfg, db_store._factory)
        slugs = [p.slug for p in DatabaseProjectRegistry(db_store._factory).list()]
        assert slugs.count("legacy") == 1


