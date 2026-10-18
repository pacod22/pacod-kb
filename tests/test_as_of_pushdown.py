"""as_of is pushed into both searches, so superseded chunks cannot crowd out valid ones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from pipeline.chunker import Chunk
from pipeline.indexer import FtsDb
from pipeline.retrieval import search
from pipeline.vectorstore import QdrantVectorStore


class _AxisEmbedder:
    """Deterministic: texts containing "old" (and the query) share one axis, others another."""

    VECTOR_DIM = 1024

    def encode(self, texts: list[str], *, progress_cb: Any = None) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.VECTOR_DIM
            v[0 if ("old" in t or t == "alpha") else 1] = 1.0
            out.append(v)
        return out

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _chunk(cid: str, doc: str, text: str, valid_until: str | None) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=doc, heading_path=["Doc"], heading_level=1, parent_chunk=None,
        sequence=0, byte_start=0, byte_end=len(text), embedded_text=text,
        valid_until=valid_until, extraction_method="deterministic", confidence=1.0, text=text,
    )


@pytest.fixture()
def crowded(tmp_path: Path) -> Any:
    """100 chunks superseded on 2020-01-01 that outrank the one current chunk on both legs."""
    emb = _AxisEmbedder()
    vstore = QdrantVectorStore(QdrantClient(":memory:"), "c")
    fts = FtsDb(tmp_path / "fts5.db")
    old = [
        _chunk(f"old{i:03d}#s#0", f"old{i:03d}", f"old alpha {i}", "2020-01-01")
        for i in range(100)
    ]
    # Long text: lower BM25 than the short superseded chunks for "alpha".
    current = _chunk("cur#s#0", "cur", "alpha " + "filler words " * 80, None)
    later = _chunk("later#s#0", "later", "old alpha valid until 2030", "2030-06-30")
    chunks = [*old, current, later]
    vstore.upsert(chunks, emb.encode([c.embedded_text for c in chunks]))
    fts.upsert_batch(chunks, "p")
    yield emb, vstore, fts
    fts.close()


def test_pushdown_finds_valid_chunks_behind_superseded_ones(crowded: Any) -> None:
    emb, vstore, fts = crowded
    hits = search("alpha", emb, vstore, fts, top_k=5, as_of="2024-01-01", project="p")
    ids = {h.chunk_id for h in hits}
    assert ids == {"cur#s#0", "later#s#0"}


def test_post_filter_alone_loses_them(crowded: Any) -> None:
    """Why the pushdown exists: the old path drops every candidate after fusion."""
    emb, vstore, fts = crowded
    hits = search(
        "alpha", emb, vstore, fts, top_k=5, as_of="2024-01-01", project="p", as_of_pushdown=False
    )
    assert "cur#s#0" not in {h.chunk_id for h in hits}


def test_fts_as_of_keeps_null_and_later_dates(tmp_path: Path) -> None:
    fts = FtsDb(tmp_path / "fts5.db")
    try:
        fts.upsert_batch(
            [
                _chunk("a#s#0", "a", "kiwi one", None),
                _chunk("b#s#0", "b", "kiwi two", "2024-06-01"),
                _chunk("c#s#0", "c", "kiwi three", "2022-01-01"),
            ],
            "p",
        )
        ids = {cid for cid, _ in fts.search("kiwi", 10, as_of="2024-01-01")}
        assert ids == {"a#s#0", "b#s#0"}
        # valid_until is exclusive: a chunk superseded on the day is not valid that day.
        assert {cid for cid, _ in fts.search("kiwi", 10, as_of="2024-06-01")} == {"a#s#0"}
        assert set(fts.valid_until_values_after("2021-12-31")) == {"2022-01-01", "2024-06-01"}
        assert fts.valid_until_values_after("2024-06-01") == []
    finally:
        fts.close()


def test_vector_valid_until_in_filter_in_embedded_qdrant() -> None:
    emb = _AxisEmbedder()
    vstore = QdrantVectorStore(QdrantClient(":memory:"), "c")
    chunks = [
        _chunk("now#s#0", "now", "old a", None),
        _chunk("v1#s#0", "v1", "old b", "2024-06-01"),
        _chunk("v0#s#0", "v0", "old c", "2020-01-01"),
    ]
    vstore.upsert(chunks, emb.encode([c.embedded_text for c in chunks]))
    q = emb.encode(["alpha"])[0]
    assert {c for c, _ in vstore.query(q, 10, valid_until_in=["2024-06-01"])} == {
        "now#s#0",
        "v1#s#0",
    }
    assert {c for c, _ in vstore.query(q, 10, valid_until_in=[])} == {"now#s#0"}
    assert {c for c, _ in vstore.query(q, 10)} == {"now#s#0"}
