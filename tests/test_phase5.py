"""Phase 5 acceptance tests: structured logging, enhanced status, watch mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(
    tmp_path: Path,
    doc_id: str,
    source_uri: str = "file.txt",
    valid_until: str | None = None,
    status: str = "ok",
    chunk_count: int = 0,
) -> Path:
    """Write a minimal manifest.json and return its path."""
    folder = tmp_path / doc_id[:8]
    folder.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "doc_id": doc_id,
        "title": "Test",
        "source_file": "file.txt",
        "source_uri": source_uri,
        "source_version": "",
        "source_sha256": doc_id,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "converted_at": "2026-01-01T00:00:00+00:00",
        "converter": "passthrough/1.0",
        "extraction_method": "deterministic",
        "language": "en",
        "doc_type": "generic",
        "supersedes": None,
        "valid_until": valid_until,
        "status": status,
        "error": None,
        "chunk_count": chunk_count,
    }
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


_DOC_A = "a" * 64
_DOC_B = "b" * 64


# ---------------------------------------------------------------------------
# TestEventLog
# ---------------------------------------------------------------------------


class TestEventLog:
    def test_append_event_adds_entry(self, tmp_path: Path) -> None:
        from pipeline.events import append_event

        mp = _make_manifest(tmp_path, _DOC_A)
        append_event(mp, "ingested", "source.txt")

        data = json.loads(mp.read_text())
        assert "events" in data
        assert len(data["events"]) == 1
        ev = data["events"][0]
        assert ev["event"] == "ingested"
        assert ev["detail"] == "source.txt"
        assert "at" in ev

    def test_append_event_accumulates(self, tmp_path: Path) -> None:
        from pipeline.events import append_event

        mp = _make_manifest(tmp_path, _DOC_A)
        append_event(mp, "ingested", "source.txt")
        append_event(mp, "superseded", "valid_until=2026-06-01")

        data = json.loads(mp.read_text())
        assert len(data["events"]) == 2
        assert data["events"][0]["event"] == "ingested"
        assert data["events"][1]["event"] == "superseded"

    def test_append_event_on_manifest_without_events_field(self, tmp_path: Path) -> None:
        from pipeline.events import append_event

        mp = _make_manifest(tmp_path, _DOC_A)
        # confirm no events field yet
        data = json.loads(mp.read_text())
        assert "events" not in data

        append_event(mp, "ingested", "")
        data = json.loads(mp.read_text())
        assert data["events"][0]["event"] == "ingested"

    def test_ingest_writes_ingested_event(self, config: Any) -> None:
        from pipeline.acquisition import ingest_document

        _, folder = ingest_document(str(FIXTURES / "sample.txt"), config)
        data = json.loads((folder / "manifest.json").read_text())
        events = data.get("events", [])
        assert any(e["event"] == "ingested" for e in events)

    def test_supersede_writes_superseded_event(self, config: Any) -> None:
        from pipeline.acquisition import ingest_document
        from pipeline.versioning import supersede_document

        # Ingest a document to act as the prior version
        prior_id, prior_folder = ingest_document(str(FIXTURES / "sample.txt"), config)

        # We need stores for supersede_document — use in-memory stubs
        from qdrant_client import QdrantClient

        from pipeline.indexer import FtsDb
        from pipeline.vectorstore import QdrantVectorStore

        fts = FtsDb(config.index_dir / "fts5.db")
        vstore = QdrantVectorStore(QdrantClient(":memory:"), config.qdrant_collection)

        try:
            supersede_document(prior_id, "2026-12-31", config, vstore, fts)
        finally:
            fts.close()

        data = json.loads((prior_folder / "manifest.json").read_text())
        events = data.get("events", [])
        assert any(e["event"] == "superseded" for e in events)

    def test_superseded_event_contains_valid_until(self, config: Any) -> None:
        from pipeline.acquisition import ingest_document
        from pipeline.versioning import supersede_document

        prior_id, prior_folder = ingest_document(str(FIXTURES / "sample.txt"), config)

        from qdrant_client import QdrantClient

        from pipeline.indexer import FtsDb
        from pipeline.vectorstore import QdrantVectorStore

        fts = FtsDb(config.index_dir / "fts5.db")
        vstore = QdrantVectorStore(QdrantClient(":memory:"), config.qdrant_collection)

        try:
            supersede_document(prior_id, "2026-12-31", config, vstore, fts)
        finally:
            fts.close()

        data = json.loads((prior_folder / "manifest.json").read_text())
        superseded_events = [e for e in data.get("events", []) if e["event"] == "superseded"]
        assert superseded_events
        assert "2026-12-31" in superseded_events[0]["detail"]


# ---------------------------------------------------------------------------
# TestStatus
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_counts_superseded(self, kb_root: Path, monkeypatch: Any) -> None:
        docs = kb_root / "documents"
        docs.mkdir(exist_ok=True)

        _make_manifest(docs, _DOC_A)  # active
        _make_manifest(docs, _DOC_B, valid_until="2026-01-01")  # superseded

        monkeypatch.setenv("KB_ROOT", str(kb_root))
        monkeypatch.setenv("QDRANT_MODE", "embedded")
        monkeypatch.delenv("QDRANT_URL", raising=False)
        from pipeline.config import load_config

        cfg = load_config()

        # Collect stats the same way the CLI does
        total = active = superseded_count = errors = chunk_total = 0
        for folder in sorted(cfg.documents_dir.iterdir()):
            mp = folder / "manifest.json"
            if not mp.exists():
                continue
            data = json.loads(mp.read_text())
            total += 1
            chunk_total += int(data.get("chunk_count", 0))
            if data.get("valid_until") is not None:
                superseded_count += 1
            elif data.get("status") == "ok":
                active += 1
            elif data.get("status") == "error":
                errors += 1

        assert total == 2
        assert active == 1
        assert superseded_count == 1
        assert errors == 0

    def test_status_sums_chunk_count(self, kb_root: Path, monkeypatch: Any) -> None:
        docs = kb_root / "documents"
        docs.mkdir(exist_ok=True)

        _make_manifest(docs, _DOC_A, chunk_count=5)
        _make_manifest(docs, _DOC_B, chunk_count=3)

        monkeypatch.setenv("KB_ROOT", str(kb_root))
        monkeypatch.setenv("QDRANT_MODE", "embedded")
        monkeypatch.delenv("QDRANT_URL", raising=False)
        from pipeline.config import load_config

        cfg = load_config()

        chunk_total = 0
        for folder in sorted(cfg.documents_dir.iterdir()):
            mp = folder / "manifest.json"
            if not mp.exists():
                continue
            data = json.loads(mp.read_text())
            chunk_total += int(data.get("chunk_count", 0))

        assert chunk_total == 8

    def test_status_review_queue_count(self, kb_root: Path, monkeypatch: Any) -> None:
        (kb_root / "review" / "item1.txt").write_text("x")
        (kb_root / "review" / "item2.txt").write_text("x")

        monkeypatch.setenv("KB_ROOT", str(kb_root))
        monkeypatch.setenv("QDRANT_MODE", "embedded")
        monkeypatch.delenv("QDRANT_URL", raising=False)
        from pipeline.config import load_config

        cfg = load_config()
        review_count = sum(1 for _ in cfg.review_dir.iterdir())
        assert review_count == 2


# ---------------------------------------------------------------------------
# TestWatch
# ---------------------------------------------------------------------------


_URL = "https://example.com/doc"
_PRIOR_ID = "1" * 64
_NEW_ID = "2" * 64


@pytest.fixture()
def watch_file(tmp_path: Path) -> Path:
    wf = tmp_path / "watch.json"
    wf.write_text(
        json.dumps([{"url": _URL, "doc_type": "web_page", "label": "Example Doc"}]),
        encoding="utf-8",
    )
    return wf


@pytest.fixture()
def watch_stores() -> tuple[Any, Any]:
    import tempfile

    from qdrant_client import QdrantClient

    from pipeline.indexer import FtsDb
    from pipeline.vectorstore import QdrantVectorStore

    tmp = tempfile.mkdtemp()
    fts = FtsDb(Path(tmp) / "fts5.db")
    vstore = QdrantVectorStore(QdrantClient(":memory:"), "kb_chunks")
    return vstore, fts


class TestWatch:
    def test_missing_watch_file_returns_empty(self, config: Any) -> None:
        from qdrant_client import QdrantClient

        from pipeline.indexer import FtsDb
        from pipeline.vectorstore import QdrantVectorStore
        from pipeline.watch import watch_sources

        fts = FtsDb(config.index_dir / "fts5.db")
        vstore = QdrantVectorStore(QdrantClient(":memory:"), config.qdrant_collection)

        try:
            results = watch_sources(
                config,
                config.kb_root / "nonexistent.json",
                MagicMock(),
                vstore,
                fts,
            )
        finally:
            fts.close()

        assert results == []

    def test_unchanged_source_returns_unchanged(
        self, config: Any, watch_file: Path
    ) -> None:
        from pipeline.watch import watch_sources

        # Seed the documents dir with an existing manifest for the same URL
        docs = config.documents_dir
        mp = _make_manifest(docs, _PRIOR_ID, source_uri=_URL)
        # Ensure document.md exists so the folder looks valid
        (mp.parent / "document.md").write_text("---\ndoc_id: " + _PRIOR_ID + "\n---\nbody\n")

        fts_mock = MagicMock()
        vstore_mock = MagicMock()

        with patch("pipeline.watch.ingest_document", return_value=(_PRIOR_ID, mp.parent)):
            results = watch_sources(config, watch_file, MagicMock(), vstore_mock, fts_mock)

        assert len(results) == 1
        assert results[0]["status"] == "unchanged"
        assert results[0]["url"] == _URL

    def test_new_source_returns_new(self, config: Any, watch_file: Path) -> None:
        from pipeline.watch import watch_sources

        # No prior manifest — this is a brand-new URL
        docs = config.documents_dir
        # mock folder for new doc
        new_folder = docs / _NEW_ID[:8]
        new_folder.mkdir(parents=True, exist_ok=True)

        fts_mock = MagicMock()
        vstore_mock = MagicMock()
        embedder_mock = MagicMock()

        with patch("pipeline.watch.ingest_document", return_value=(_NEW_ID, new_folder)):
            with patch("pipeline.watch.index_document"):
                results = watch_sources(config, watch_file, embedder_mock, vstore_mock, fts_mock)

        assert len(results) == 1
        assert results[0]["status"] == "new"
        assert results[0]["doc_id"] == _NEW_ID[:8]

    def test_changed_source_returns_updated(self, config: Any, watch_file: Path) -> None:
        from pipeline.watch import watch_sources

        # Seed docs dir with prior manifest
        docs = config.documents_dir
        prior_mp = _make_manifest(docs, _PRIOR_ID, source_uri=_URL)
        (prior_mp.parent / "document.md").write_text(
            "---\ndoc_id: " + _PRIOR_ID + "\n---\nbody\n"
        )

        # New folder for the changed doc
        new_folder = docs / (_NEW_ID[:8] + "_page")
        new_folder.mkdir(parents=True, exist_ok=True)
        _make_manifest(new_folder.parent if new_folder.parent != docs else docs, _NEW_ID)
        new_manifest = new_folder / "manifest.json"
        new_manifest.write_text(
            json.dumps(
                {
                    "doc_id": _NEW_ID,
                    "title": "New",
                    "source_file": "page.html",
                    "source_uri": _URL,
                    "source_version": "",
                    "source_sha256": _NEW_ID,
                    "retrieved_at": "2026-01-01T00:00:00+00:00",
                    "converted_at": "2026-01-01T00:00:00+00:00",
                    "converter": "web/1.0",
                    "extraction_method": "deterministic",
                    "language": "en",
                    "doc_type": "web_page",
                    "supersedes": None,
                    "valid_until": None,
                    "status": "ok",
                    "error": None,
                    "chunk_count": 0,
                }
            ),
            encoding="utf-8",
        )

        fts_mock = MagicMock()
        vstore_mock = MagicMock()
        embedder_mock = MagicMock()

        with patch("pipeline.watch.ingest_document", return_value=(_NEW_ID, new_folder)):
            with patch("pipeline.watch.index_document"):
                with patch("pipeline.watch.supersede_document") as mock_sup:
                    results = watch_sources(
                        config, watch_file, embedder_mock, vstore_mock, fts_mock
                    )

        assert len(results) == 1
        assert results[0]["status"] == "updated"
        assert results[0]["prior_doc_id"] == _PRIOR_ID[:8]
        assert results[0]["new_doc_id"] == _NEW_ID[:8]
        # supersede_document must have been called on the prior doc
        mock_sup.assert_called_once()
        call_args = mock_sup.call_args[0]
        assert call_args[0] == _PRIOR_ID

    def test_fetch_error_returns_error_status(self, config: Any, watch_file: Path) -> None:
        from pipeline.watch import watch_sources

        fts_mock = MagicMock()
        vstore_mock = MagicMock()

        with patch("pipeline.watch.ingest_document", side_effect=ValueError("timeout")):
            results = watch_sources(config, watch_file, MagicMock(), vstore_mock, fts_mock)

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "timeout" in results[0]["error"]

    def test_watch_sources_reports_progress(self, config: Any, watch_file: Path) -> None:
        from pipeline.watch import watch_sources

        fts_mock = MagicMock()
        vstore_mock = MagicMock()
        embedder_mock = MagicMock()
        progress_events: list[tuple[int, int, str]] = []

        with patch("pipeline.watch.ingest_document", return_value=(_NEW_ID, Path("/tmp/new"))):
            with patch("pipeline.watch.index_document"):
                watch_sources(
                    config,
                    watch_file,
                    embedder_mock,
                    vstore_mock,
                    fts_mock,
                    progress_cb=lambda cur, total, msg: progress_events.append((cur, total, msg)),
                )

        assert progress_events[0][:2] == (0, 1)
        assert progress_events[-1][:2] == (1, 1)

    def test_find_current_doc_for_url_ignores_superseded(self, config: Any) -> None:
        from pipeline.watch import _find_current_doc_for_url

        docs = config.documents_dir
        # superseded doc for the URL
        _make_manifest(docs, _PRIOR_ID, source_uri=_URL, valid_until="2026-01-01")
        # active doc for a different URL
        _make_manifest(docs, _DOC_A, source_uri="https://other.example.com/")

        result = _find_current_doc_for_url(_URL, config)
        assert result is None  # prior is superseded, so no current doc found

    def test_find_current_doc_for_url_finds_active(self, config: Any) -> None:
        from pipeline.watch import _find_current_doc_for_url

        docs = config.documents_dir
        _make_manifest(docs, _PRIOR_ID, source_uri=_URL)

        result = _find_current_doc_for_url(_URL, config)
        assert result == _PRIOR_ID

    def test_find_current_doc_for_url_uses_store_when_provided(self, config: Any) -> None:
        from pipeline.watch import _find_current_doc_for_url

        store_mock = MagicMock()
        store_mock.find_by_source_uri.return_value = _PRIOR_ID

        result = _find_current_doc_for_url(_URL, config, store=store_mock)

        assert result == _PRIOR_ID
        store_mock.find_by_source_uri.assert_called_once_with(_URL)

