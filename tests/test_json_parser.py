"""Tests for JSON ingestion: JsonParser rendering, dispatch, ingest + dropzone."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from pipeline.acquisition import ingest_document
from pipeline.dropzone import scan_dropzone
from pipeline.indexer import FtsDb
from pipeline.parsers import SUPPORTED_EXTENSIONS, get_parser
from pipeline.parsers.json_parser import JsonParser
from pipeline.vectorstore import QdrantVectorStore


class MockEmbedder:
    VECTOR_DIM = 1024

    def encode(
        self, texts: list[str], *, progress_cb: Callable[[int, int], None] | None = None
    ) -> list[list[float]]:
        import random

        vecs = [
            [random.Random(hash(t) ^ i).gauss(0, 1) for i in range(self.VECTOR_DIM)]
            for t in texts
        ]
        if progress_cb is not None:
            progress_cb(len(texts), len(texts))
        return vecs

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


# ─── dispatch ────────────────────────────────────────────────────────────────


def test_json_in_supported_extensions() -> None:
    assert "json" in SUPPORTED_EXTENSIONS


def test_get_parser_returns_json_parser() -> None:
    assert isinstance(get_parser("json"), JsonParser)
    assert isinstance(get_parser(".JSON"), JsonParser)


# ─── rendering ───────────────────────────────────────────────────────────────


def test_parse_renders_headings_and_scalars(tmp_path: Path) -> None:
    src = _write_json(
        tmp_path / "doc.json",
        {
            "title": "Tax Regulation",
            "year": 2024,
            "active": True,
            "section": {"name": "Scope", "text": "Applies to all residents."},
        },
    )
    result = JsonParser().parse(src)
    assert result.title == "Tax Regulation"
    assert "**year:** 2024" in result.body
    assert "**active:** true" in result.body
    assert "## section" in result.body
    assert "Applies to all residents." in result.body


def test_parse_array_of_objects_uses_item_name_as_heading(tmp_path: Path) -> None:
    src = _write_json(
        tmp_path / "list.json",
        {"articles": [{"name": "Article 1", "text": "First."}, {"id": "A2", "text": "Second."}]},
    )
    body = JsonParser().parse(src).body
    assert "Article 1" in body
    assert "A2" in body
    assert "Item 1" not in body  # named items, not positional fallback


def test_parse_array_of_scalars_renders_bullets(tmp_path: Path) -> None:
    src = _write_json(tmp_path / "tags.json", {"tags": ["alpha", "beta"]})
    body = JsonParser().parse(src).body
    assert "- alpha" in body
    assert "- beta" in body


def test_parse_title_falls_back_to_stem(tmp_path: Path) -> None:
    src = _write_json(tmp_path / "no-title.json", {"foo": "bar"})
    assert JsonParser().parse(src).title == "no-title"


def test_parse_invalid_json_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        JsonParser().parse(bad)


# ─── ingestion path ──────────────────────────────────────────────────────────


def test_ingest_document_accepts_json(tmp_path: Path, config: Any) -> None:
    src = _write_json(
        tmp_path / "policy.json",
        {"title": "Privacy Policy", "body": "We process personal data lawfully and fairly."},
    )
    doc_id, folder = ingest_document(str(src), config)
    assert (folder / "document.md").exists()
    md = (folder / "document.md").read_text(encoding="utf-8")
    assert "Privacy Policy" in md
    assert (folder / "source.json").exists()


# ─── dropzone path ───────────────────────────────────────────────────────────


@pytest.fixture()
def stores(config: Any):  # noqa: ANN201
    qclient = QdrantClient(":memory:")
    vstore = QdrantVectorStore(qclient, config.qdrant_collection)
    fts = FtsDb(config.index_dir / "fts5.db")
    yield vstore, fts
    fts.close()


def test_dropzone_ingests_json(config: Any, stores: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pipeline.dropzone._STABILITY_SECS", 0)
    vstore, fts = stores
    config.kb_dropzone_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config.kb_dropzone_dir / "drop.json",
        {"title": "Dropped Doc", "text": "Content ingested via the dropzone."},
    )

    results = scan_dropzone(config, MockEmbedder(), vstore, fts)

    assert len(results) == 1
    assert results[0]["status"] == "ingested"
    assert (config.kb_done_dir / "drop.json").exists()
    assert not (config.kb_dropzone_dir / "drop.json").exists()


def test_dropzone_invalid_json_goes_to_failed(
    config: Any, stores: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pipeline.dropzone._STABILITY_SECS", 0)
    vstore, fts = stores
    config.kb_dropzone_dir.mkdir(parents=True, exist_ok=True)
    (config.kb_dropzone_dir / "broken.json").write_text("{nope", encoding="utf-8")

    results = scan_dropzone(config, MockEmbedder(), vstore, fts)

    assert results[0]["status"] == "error"
    assert (config.kb_failed_dir / "broken.json").exists()


def test_dropzone_reports_progress(
    config: Any,
    stores: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pipeline.dropzone._STABILITY_SECS", 0)
    vstore, fts = stores
    config.kb_dropzone_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config.kb_dropzone_dir / "progress.json",
        {"title": "Progress", "text": "Dropzone progress callback test."},
    )

    progress_events: list[tuple[int, int, str]] = []
    scan_dropzone(
        config,
        MockEmbedder(),
        vstore,
        fts,
        progress_cb=lambda cur, total, msg: progress_events.append((cur, total, msg)),
    )

    assert progress_events[0][:2] == (0, 1)
    assert progress_events[-1][:2] == (1, 1)

