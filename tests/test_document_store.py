"""Round-trip tests for FilesystemDocumentStore and DatabaseDocumentStore."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pipeline.config import Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRONT_MATTER: dict[str, Any] = {
    "doc_id": "a" * 64,
    "title": "Test Document",
    "source_file": "test.md",
    "source_uri": "https://example.com/test",
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
}

_BODY = "# Heading\n\nParagraph text for testing.\n"
_SOURCE_BYTES = b"raw source bytes"
_SOURCE_EXT = ".md"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fs_store(config: Config) -> Any:
    from pipeline.document_store import FilesystemDocumentStore

    return FilesystemDocumentStore(config)


@pytest.fixture()
def db_store(tmp_path: Path) -> Any:
    from sqlalchemy import create_engine

    from pipeline.document_store import DatabaseDocumentStore

    engine = create_engine(f"sqlite:///{tmp_path}/test_kb.db")
    return DatabaseDocumentStore(engine)


# ---------------------------------------------------------------------------
# FilesystemDocumentStore
# ---------------------------------------------------------------------------


class TestFilesystemDocumentStore:
    def test_sha256_exists_false_when_empty(self, fs_store: Any) -> None:
        assert fs_store.sha256_exists("a" * 64) is False

    def test_save_and_sha256_exists(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        assert fs_store.sha256_exists(_FRONT_MATTER["doc_id"]) is True

    def test_load_body_round_trips(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        body = fs_store.load_body(_FRONT_MATTER["doc_id"])
        assert "Heading" in body

    def test_load_source_blob_round_trips(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        source_bytes, source_name = fs_store.load_source_blob(_FRONT_MATTER["doc_id"])
        assert source_bytes == _SOURCE_BYTES
        assert source_name == _FRONT_MATTER["source_file"]

    def test_load_manifest_has_title(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        m = fs_store.load_manifest(_FRONT_MATTER["doc_id"])
        assert m["title"] == "Test Document"

    def test_update_manifest_valid_until(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        fs_store.update_manifest(_FRONT_MATTER["doc_id"], valid_until="2026-06-01")
        m = fs_store.load_manifest(_FRONT_MATTER["doc_id"])
        assert m["valid_until"] == "2026-06-01"

    def test_append_event_and_list(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        fs_store.append_event(_FRONT_MATTER["doc_id"], "ingested", "https://example.com")
        m = fs_store.load_manifest(_FRONT_MATTER["doc_id"])
        events = m.get("events", [])
        assert any(e["event"] == "ingested" for e in events)

    def test_find_by_prefix(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        found = fs_store.find_by_prefix(_FRONT_MATTER["doc_id"][:8])
        assert found == _FRONT_MATTER["doc_id"]

    def test_find_by_source_uri(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        found = fs_store.find_by_source_uri("https://example.com/test")
        assert found == _FRONT_MATTER["doc_id"]

    def test_find_by_source_uri_ignores_superseded(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        fs_store.update_manifest(_FRONT_MATTER["doc_id"], valid_until="2026-06-01")
        found = fs_store.find_by_source_uri("https://example.com/test")
        assert found is None

    def test_list_all_returns_manifest(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        docs = fs_store.list_all()
        assert len(docs) == 1
        assert docs[0]["title"] == "Test Document"

    def test_delete_removes_folder(self, fs_store: Any, config: Config) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        assert len(list(config.documents_dir.iterdir())) == 1
        fs_store.delete(_FRONT_MATTER["doc_id"])
        assert len(list(config.documents_dir.iterdir())) == 0

    def test_export_all_includes_body(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        docs = fs_store.export_all()
        assert len(docs) == 1
        assert "Heading" in docs[0]["body"]

    def test_export_all_includes_events(self, fs_store: Any) -> None:
        fs_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        fs_store.append_event(_FRONT_MATTER["doc_id"], "ingested", "url")
        docs = fs_store.export_all()
        assert len(docs) == 1
        assert isinstance(docs[0].get("events"), list)
        assert docs[0]["events"][0]["event"] == "ingested"


# ---------------------------------------------------------------------------
# DatabaseDocumentStore
# ---------------------------------------------------------------------------


class TestDatabaseDocumentStore:
    def test_sha256_exists_false_when_empty(self, db_store: Any) -> None:
        assert db_store.sha256_exists("a" * 64) is False

    def test_save_and_sha256_exists(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        assert db_store.sha256_exists(_FRONT_MATTER["doc_id"]) is True

    def test_load_body_round_trips(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        body = db_store.load_body(_FRONT_MATTER["doc_id"])
        assert body == _BODY

    def test_load_source_blob_round_trips(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        source_bytes, source_name = db_store.load_source_blob(_FRONT_MATTER["doc_id"])
        assert source_bytes == _SOURCE_BYTES
        assert source_name == _FRONT_MATTER["source_file"]

    def test_load_manifest_has_title(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        m = db_store.load_manifest(_FRONT_MATTER["doc_id"])
        assert m["title"] == "Test Document"

    def test_update_manifest_valid_until(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        db_store.update_manifest(_FRONT_MATTER["doc_id"], valid_until="2026-06-01")
        m = db_store.load_manifest(_FRONT_MATTER["doc_id"])
        assert m["valid_until"] == "2026-06-01"

    def test_append_event_stored(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        db_store.append_event(_FRONT_MATTER["doc_id"], "ingested", "https://example.com")
        m = db_store.load_manifest(_FRONT_MATTER["doc_id"])
        events = m.get("events", [])
        assert any(e["event"] == "ingested" for e in events)

    def test_multiple_events_accumulate(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        db_store.append_event(_FRONT_MATTER["doc_id"], "ingested", "url1")
        db_store.append_event(_FRONT_MATTER["doc_id"], "reindexed", "")
        m = db_store.load_manifest(_FRONT_MATTER["doc_id"])
        events = m.get("events", [])
        assert len(events) == 2

    def test_find_by_prefix(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        found = db_store.find_by_prefix(_FRONT_MATTER["doc_id"][:8])
        assert found == _FRONT_MATTER["doc_id"]

    def test_find_by_prefix_returns_none_when_missing(self, db_store: Any) -> None:
        assert db_store.find_by_prefix("00000000") is None

    def test_find_by_source_uri(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        found = db_store.find_by_source_uri("https://example.com/test")
        assert found == _FRONT_MATTER["doc_id"]

    def test_find_by_source_uri_ignores_superseded(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        db_store.update_manifest(_FRONT_MATTER["doc_id"], valid_until="2026-06-01")
        found = db_store.find_by_source_uri("https://example.com/test")
        assert found is None

    def test_list_all_returns_manifest(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        docs = db_store.list_all()
        assert len(docs) == 1
        assert docs[0]["title"] == "Test Document"

    def test_delete_removes_row(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        db_store.delete(_FRONT_MATTER["doc_id"])
        assert db_store.list_all() == []

    def test_export_all_includes_body(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        docs = db_store.export_all()
        assert len(docs) == 1
        assert docs[0]["body"] == _BODY

    def test_export_includes_events(self, db_store: Any) -> None:
        db_store.save(
            _FRONT_MATTER["doc_id"],
            dict(_FRONT_MATTER),
            _BODY,
            _SOURCE_BYTES,
            _SOURCE_EXT,
        )
        db_store.append_event(_FRONT_MATTER["doc_id"], "ingested", "url")
        docs = db_store.export_all()
        assert len(docs[0]["events"]) == 1
        assert docs[0]["events"][0]["event"] == "ingested"


# ---------------------------------------------------------------------------
# make_document_store factory
# ---------------------------------------------------------------------------


class TestMakeDocumentStore:
    def test_filesystem_mode(self, config: Config) -> None:
        from pipeline.document_store import FilesystemDocumentStore, make_document_store

        store = make_document_store(config)
        assert isinstance(store, FilesystemDocumentStore)

    def test_database_mode(self, kb_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KB_ROOT", str(kb_root))
        monkeypatch.setenv("KB_STORAGE_BACKEND", "database")
        monkeypatch.setenv("KB_DATABASE_HOST", "db.example.com")
        monkeypatch.setenv("KB_DATABASE_USER", "kb_user")
        monkeypatch.setenv("KB_DATABASE_PASSWORD", "secret")
        monkeypatch.setenv("QDRANT_MODE", "embedded")
        monkeypatch.setenv("GIT_AUTOCOMMIT", "false")
        monkeypatch.delenv("QDRANT_URL", raising=False)

        from sqlalchemy import create_engine as sa_create_engine

        from pipeline.config import load_config
        from pipeline.document_store import DatabaseDocumentStore, make_document_store

        config = load_config()
        fake_engine = sa_create_engine(f"sqlite:///{kb_root}/index/test_factory.db")
        with patch("sqlalchemy.create_engine", return_value=fake_engine) as mocked_create_engine:
            store = make_document_store(config)
        assert isinstance(store, DatabaseDocumentStore)
        called_url = str(mocked_create_engine.call_args[0][0])
        assert called_url.startswith("postgresql+psycopg2://")

    def test_database_mode_requires_postgres_env(
        self,
        kb_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("KB_ROOT", str(kb_root))
        monkeypatch.setenv("KB_STORAGE_BACKEND", "database")
        monkeypatch.setenv("QDRANT_MODE", "embedded")
        monkeypatch.setenv("GIT_AUTOCOMMIT", "false")
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("KB_DATABASE_HOST", raising=False)
        monkeypatch.delenv("KB_DATABASE_USER", raising=False)
        monkeypatch.delenv("KB_DATABASE_PASSWORD", raising=False)

        from pipeline.config import load_config

        with pytest.raises(
            SystemExit, match="KB_STORAGE_BACKEND=database requires PostgreSQL settings"
        ):
            load_config()


class TestMakeFtsDb:
    def test_filesystem_mode_returns_sqlite_fts(self, config: Config) -> None:
        from pipeline.indexer import FtsDb, make_fts_db

        fts = make_fts_db(config)
        try:
            assert isinstance(fts, FtsDb)
        finally:
            fts.close()

    def test_database_mode_uses_postgres_backend_and_migration(
        self,
        kb_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("KB_ROOT", str(kb_root))
        monkeypatch.setenv("KB_STORAGE_BACKEND", "database")
        monkeypatch.setenv("KB_DATABASE_HOST", "db.example.com")
        monkeypatch.setenv("KB_DATABASE_USER", "kb_user")
        monkeypatch.setenv("KB_DATABASE_PASSWORD", "secret")
        monkeypatch.setenv("QDRANT_MODE", "embedded")
        monkeypatch.setenv("GIT_AUTOCOMMIT", "false")
        monkeypatch.delenv("QDRANT_URL", raising=False)

        from pipeline.config import load_config
        from pipeline.indexer import make_fts_db

        cfg = load_config()
        legacy = cfg.index_dir / "fts5.db"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"legacy")

        pg_mock = MagicMock()
        pg_mock.migrate_from_sqlite.return_value = 3

        with patch("sqlalchemy.create_engine") as create_engine_mock:
            create_engine_mock.return_value = MagicMock()
            with patch("pipeline.indexer.PostgresFtsDb", return_value=pg_mock) as pg_cls:
                fts = make_fts_db(cfg)

        assert fts is pg_mock
        pg_cls.assert_called_once()
        pg_mock.migrate_from_sqlite.assert_called_once_with(legacy)


class TestBackendParityRegressions:
    def test_rebuild_index_db_updates_real_chunk_count(self, db_store: Any, config: Config) -> None:
        from pipeline.versioning import rebuild_index

        doc_a = "c" * 64
        doc_b = "d" * 64

        fm_a = {
            **_FRONT_MATTER,
            "doc_id": doc_a,
            "source_sha256": doc_a,
            "source_uri": "https://example.com/a",
        }
        fm_b = {
            **_FRONT_MATTER,
            "doc_id": doc_b,
            "source_sha256": doc_b,
            "source_uri": "https://example.com/b",
        }
        db_store.save(doc_a, fm_a, _BODY, _SOURCE_BYTES, _SOURCE_EXT)
        db_store.save(doc_b, fm_b, _BODY, _SOURCE_BYTES, _SOURCE_EXT)

        vector_store = MagicMock()
        fts_db = MagicMock()

        def _fake_index_document(
            _doc_folder: Any,
            _config: Any,
            _embedder: Any,
            _vector_store: Any,
            _fts: Any,
            *,
            doc_data: tuple[dict[str, Any], str] | None = None,
            **_kwargs: Any,
        ) -> list[object]:
            assert doc_data is not None
            doc_id = doc_data[0]["doc_id"]
            return [object()] * (2 if doc_id == doc_a else 5)

        with patch("pipeline.indexer.index_document", side_effect=_fake_index_document):
            ok, err = rebuild_index(
                config,
                embedder=MagicMock(),
                vector_store=vector_store,
                fts_db=fts_db,
                store=db_store,
            )

        assert ok == 2
        assert err == 0
        assert db_store.load_manifest(doc_a)["chunk_count"] == 2
        assert db_store.load_manifest(doc_b)["chunk_count"] == 5


# ---------------------------------------------------------------------------
# Prefix lookup safety (both backends)
# ---------------------------------------------------------------------------


def _save_ids(store: Any, *ids: str) -> None:
    for i, doc_id in enumerate(ids):
        fm = {**_FRONT_MATTER, "doc_id": doc_id, "source_sha256": doc_id, "title": f"Doc {i}"}
        store.save(doc_id, fm, _BODY, _SOURCE_BYTES, _SOURCE_EXT)


@pytest.mark.parametrize("backend", ["fs_store", "db_store"])
class TestFindByPrefix:
    def test_unique_prefix_resolves(self, backend: str, request: Any) -> None:
        store = request.getfixturevalue(backend)
        _save_ids(store, "ab" * 32, "cd" * 32)
        assert store.find_by_prefix("abababab") == "ab" * 32
        assert store.find_by_prefix("cd" * 32) == "cd" * 32
        assert store.find_by_prefix("ffffffff") is None

    @pytest.mark.parametrize("bad", ["%", "_", "abc", "%%%%%%%%", "abababa_", "../../etc", ""])
    def test_invalid_prefix_rejected(self, backend: str, request: Any, bad: str) -> None:
        from pipeline.document_store import InvalidDocPrefixError

        store = request.getfixturevalue(backend)
        _save_ids(store, "ab" * 32)
        with pytest.raises(InvalidDocPrefixError):
            store.find_by_prefix(bad)

    def test_ambiguous_prefix_rejected(self, backend: str, request: Any) -> None:
        from pipeline.document_store import AmbiguousDocPrefixError

        store = request.getfixturevalue(backend)
        _save_ids(store, "12345678" + "a" * 56, "12345678" + "b" * 56)
        with pytest.raises(AmbiguousDocPrefixError):
            store.find_by_prefix("12345678")
        assert store.find_by_prefix("12345678a") == "12345678" + "a" * 56
