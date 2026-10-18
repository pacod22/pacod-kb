"""Regression tests for P2 performance changes (behaviour must be unchanged)."""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from pipeline.chunker import Chunk, _ByteOffsets, chunk_document


def _tc(text: str) -> int:
    return max(1, len(text) // 4)


def _chunk(cid: str, doc: str, text: str, project_valid: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        heading_path=["H"],
        heading_level=1,
        parent_chunk=None,
        sequence=0,
        byte_start=0,
        byte_end=len(text.encode()),
        embedded_text=text,
        valid_until=project_valid,
        extraction_method="deterministic",
        confidence=1.0,
        text=text,
    )


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def test_byte_offsets_match_naive_encoding_for_cyrillic() -> None:
    rng = random.Random(7)
    alphabet = "абвгдежзийклмнопрстуфхцчшщъьюя ABC €😀\n"
    body = "".join(rng.choice(alphabet) for _ in range(20_000))
    to_byte = _ByteOffsets(body, stride=257)
    for pos in [0, 1, 256, 257, 258, 9_999, len(body)] + rng.sample(range(len(body)), 200):
        assert to_byte(pos) == len(body[:pos].encode("utf-8"))


def test_chunk_text_equals_byte_slice() -> None:
    body = "# Закон\n\n" + "\n\n".join(f"Член {i}. Текст на кирилица." for i in range(300))
    raw = body.encode("utf-8")
    for c in chunk_document(body, "f" * 64, "Doc", _tc, max_tokens=60):
        assert raw[c.byte_start : c.byte_end].decode("utf-8") == c.text


def test_headings_inside_code_fences_are_ignored() -> None:
    body = "# Real\n\nIntro.\n\n```bash\n# not a heading\necho hi\n```\n\nAfter code.\n"
    chunks = chunk_document(body, "a" * 64, "Doc", _tc)
    assert {tuple(c.heading_path) for c in chunks} == {("Real",)}


def test_oversized_paragraph_is_split_but_tables_are_not() -> None:
    long_para = " ".join(f"Sentence number {i} is here." for i in range(400))
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i * 2} |" for i in range(300))
    body = f"# S\n\n{long_para}\n\n{table}\n"

    def words(text: str) -> int:  # additive across sentence splits, unlike len // 4
        return max(1, len(text.split()))

    chunks = chunk_document(body, "b" * 64, "Doc", words, max_tokens=100)
    text_chunks = [c for c in chunks if not c.text.lstrip().startswith("|")]
    assert len(text_chunks) > 5
    assert all(words(c.text) <= 100 for c in text_chunks[1:])  # first binds the heading
    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|")]
    assert len(table_chunks) == 1 and table_chunks[0].text.count("\n") >= 300


def test_chunk_ids_use_16_char_prefix() -> None:
    chunks = chunk_document("# A\n\nbody\n", "0123456789abcdef" + "9" * 48, "Doc", _tc)
    assert chunks[0].chunk_id.startswith("0123456789abcdef#")


# ---------------------------------------------------------------------------
# SQLite FTS store
# ---------------------------------------------------------------------------


def _legacy_fts(path: Path) -> None:
    """Create an FTS database with the pre-P2 schema and two rows."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE chunks_meta (
            chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, heading_path TEXT NOT NULL,
            heading_level INTEGER NOT NULL, parent_chunk TEXT, sequence INTEGER NOT NULL,
            byte_start INTEGER NOT NULL, byte_end INTEGER NOT NULL, valid_until TEXT,
            extraction_method TEXT NOT NULL, confidence REAL NOT NULL,
            project TEXT DEFAULT 'default');
        CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, embedded_text,
            tokenize='unicode61');
    """)
    for cid, doc, text in (("c1", "d1", "alpha legacy words"), ("c2", "d2", "beta legacy")):
        con.execute(
            "INSERT INTO chunks_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, doc, '["H"]', 1, None, 0, 0, 5, None, "deterministic", 1.0, "p"),
        )
        con.execute("INSERT INTO chunks_fts VALUES (?, ?)", (cid, text))
    con.commit()
    con.close()


def test_fts_migration_preserves_rows(tmp_path: Path) -> None:
    from pipeline.indexer import FtsDb

    path = tmp_path / "fts5.db"
    _legacy_fts(path)
    db = FtsDb(path)
    try:
        assert [c for c, _ in db.search("legacy", 10, project="p")] in (["c1", "c2"], ["c2", "c1"])
        assert db.get_embedded_text("c1") == "alpha legacy words"
        db.delete_by_doc("d1")
        assert [c for c, _ in db.search("legacy", 10)] == ["c2"]
        fts_rows = db._con.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        assert fts_rows == 1  # FTS row removed with its meta row
    finally:
        db.close()
    FtsDb(path).close()  # reopening is a no-op migration


def test_fts_bulk_lookup_and_batch_upsert(tmp_path: Path) -> None:
    from pipeline.indexer import FtsDb

    db = FtsDb(tmp_path / "fts.db")
    try:
        chunks = [_chunk(f"c{i}", "d", f"word{i} shared") for i in range(1200)]
        db.upsert_batch(chunks, "p", retrieved_at="2026-01-01T00:00:00")
        db.upsert_batch(chunks[:10], "p")  # re-upsert replaces, no duplicates
        assert db._con.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 1200
        got = db.get_chunks_bulk([c.chunk_id for c in chunks] + ["missing"])
        assert len(got) == 1200 and "missing" not in got
        meta, text = got["c5"]
        assert text == "word5 shared" and meta["project"] == "p"
        assert got["c500"][0]["retrieved_at"] == "2026-01-01T00:00:00"
        db.drop_by_project("p")
        assert db._con.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 0
    finally:
        db.close()


def test_search_results_identical_through_bulk_path(tmp_path: Path) -> None:
    """retrieval.search output equals a per-hit lookup of the same fused list."""
    from unittest.mock import MagicMock

    from pipeline.indexer import FtsDb
    from pipeline.retrieval import _rrf, search

    db = FtsDb(tmp_path / "fts.db")
    try:
        db.upsert_batch([_chunk(f"c{i}", f"d{i % 3}", f"topic {i} keyword") for i in range(30)])
        vs = MagicMock()
        vs.query.return_value = [("c3", 1), ("c7", 2), ("c11", 3)]
        emb = MagicMock()
        emb.encode.return_value = [[0.0]]
        results = search("keyword", emb, vs, db, top_k=8)
        fused = _rrf(vs.query.return_value, db.search("keyword", 20))
        expected = [cid for cid, _ in fused][:8]
        assert [r.chunk_id for r in results] == expected
        assert all(r.text == db.get_embedded_text(r.chunk_id) for r in results)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------


def test_new_doc_folder_never_overwrites_other_document(tmp_path: Path) -> None:
    from pipeline.fs_layout import find_folder_by_doc_id, new_doc_folder, write_json_atomic

    a, b = "12345678" + "a" * 56, "12345678" + "b" * 56
    fa = new_doc_folder(tmp_path, a, "same-title")
    fa.mkdir()
    write_json_atomic(fa / "manifest.json", {"doc_id": a})
    fb = new_doc_folder(tmp_path, b, "same-title")
    assert fb != fa and fb.name.startswith(b[:16])
    fb.mkdir()
    write_json_atomic(fb / "manifest.json", {"doc_id": b})
    assert find_folder_by_doc_id(tmp_path, a) == fa
    assert find_folder_by_doc_id(tmp_path, b) == fb
    assert new_doc_folder(tmp_path, a, "same-title") == fa  # same doc → same folder


def test_write_json_atomic_leaves_no_temp_files(tmp_path: Path) -> None:
    from pipeline.fs_layout import write_json_atomic

    target = tmp_path / "manifest.json"
    write_json_atomic(target, {"title": "Кирилица"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"title": "Кирилица"}
    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"title": "Кирилица"}
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]


# ---------------------------------------------------------------------------
# Qdrant batching
# ---------------------------------------------------------------------------


def test_large_upsert_is_batched(monkeypatch: pytest.MonkeyPatch) -> None:
    from qdrant_client import QdrantClient

    from pipeline import vectorstore

    client = QdrantClient(":memory:")
    calls: list[int] = []
    real = client.upsert

    def spy(collection_name: str, points: list[Any], **kw: Any) -> Any:
        calls.append(len(points))
        return real(collection_name=collection_name, points=points, **kw)

    monkeypatch.setattr(client, "upsert", spy)
    vs = vectorstore.QdrantVectorStore(client, "kb")
    chunks = [_chunk(f"c{i}", "d", f"t{i}") for i in range(600)]
    vs.upsert(chunks, [[0.1] * 1024 for _ in chunks])
    assert calls == [256, 256, 88]
    assert client.count("kb").count == 600
    client.close()
