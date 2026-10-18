"""Phase 2 tests: chunker, indexer, FTS5, vector store, retrieval.

All tests use:
  - QdrantClient(":memory:")  — no disk I/O, no BGE-M3 download
  - MockEmbedder              — deterministic random 1024-dim vectors
  - Fixture markdown files from tests/fixtures/
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared mock embedder
# ---------------------------------------------------------------------------


class MockEmbedder:
    VECTOR_DIM: int = 1024

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


@pytest.fixture()
def embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture()
def in_memory_store():  # type: ignore[no-untyped-def]
    from qdrant_client import QdrantClient

    from pipeline.vectorstore import QdrantVectorStore

    client = QdrantClient(":memory:")
    return QdrantVectorStore(client, "test_chunks")


@pytest.fixture()
def fts_db(tmp_path: Path):  # type: ignore[no-untyped-def]
    from pipeline.indexer import FtsDb

    db = FtsDb(tmp_path / "fts5.db")
    yield db
    db.close()


# ---------------------------------------------------------------------------
# chunk.json schema
# ---------------------------------------------------------------------------


class TestChunkSchema:
    def test_schema_loads(self) -> None:
        schema_path = Path("pipeline/schemas/chunk.json")
        assert schema_path.exists()
        data = json.loads(schema_path.read_text())
        assert data["type"] == "object"
        required = data["required"]
        for field in ("chunk_id", "source_id", "byte_start", "byte_end", "text"):
            assert field in required

    def test_valid_chunk_passes(self) -> None:

        from pipeline.chunker import Chunk
        from pipeline.indexer import _validate_chunk

        c = Chunk(
            chunk_id="abcd1234#preamble#0",
            doc_id="a" * 64,
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=10,
            embedded_text="[Doc]\nhello world",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        _validate_chunk(c)  # must not raise

    def test_invalid_doc_id_rejected(self) -> None:
        import jsonschema

        from pipeline.chunker import Chunk
        from pipeline.indexer import _validate_chunk

        c = Chunk(
            chunk_id="abcd1234#preamble#0",
            doc_id="short",  # not 64 hex chars
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=10,
            embedded_text="[Doc]\nhello world",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        with pytest.raises(jsonschema.ValidationError):
            _validate_chunk(c)


# ---------------------------------------------------------------------------
# Chunker: structural properties
# ---------------------------------------------------------------------------


class TestChunker:
    def _tok(self, text: str) -> int:
        return max(1, len(text) // 4)

    def test_no_chunks_for_empty_body(self) -> None:
        from pipeline.chunker import chunk_document

        chunks = chunk_document("", "a" * 64, "Doc", self._tok)
        assert chunks == []

    def test_single_chunk_for_small_body(self) -> None:
        from pipeline.chunker import chunk_document

        body = "# Title\n\nShort content here."
        chunks = chunk_document(body, "a" * 64, "Doc", self._tok)
        assert len(chunks) >= 1

    def test_every_chunk_has_breadcrumb(self) -> None:
        from pipeline.chunker import chunk_document

        body = (FIXTURES / "sample.md").read_text()
        chunks = chunk_document(
            body, "a" * 64, "Technical Documentation: Vector Databases", self._tok
        )
        for c in chunks:
            assert c.embedded_text.startswith("[")
            assert "]\n" in c.embedded_text

    def test_char_span_accuracy(self) -> None:
        """body_bytes[byte_start:byte_end].decode() must equal chunk body text."""
        from pipeline.chunker import chunk_document

        body = (FIXTURES / "sample.md").read_text()
        body_bytes = body.encode("utf-8")
        chunks = chunk_document(body, "a" * 64, "Doc", self._tok)

        for c in chunks:
            sliced = body_bytes[c.byte_start : c.byte_end].decode("utf-8")
            # The embedded_text is breadcrumb + sliced (plus possible caption for tables)
            # At minimum the sliced text must appear in embedded_text
            assert sliced in c.embedded_text, (
                f"chunk {c.chunk_id}: body slice not found in embedded_text"
            )

    def test_byte_span_no_overlap(self) -> None:
        """Chunks from same section must not overlap in byte space."""
        from pipeline.chunker import chunk_document

        body = (FIXTURES / "sample.md").read_text()
        chunks = chunk_document(body, "a" * 64, "Doc", self._tok, max_tokens=50)

        # Check no overlapping spans (sort by byte_start)
        sorted_chunks = sorted(chunks, key=lambda c: c.byte_start)
        for i in range(len(sorted_chunks) - 1):
            a, b = sorted_chunks[i], sorted_chunks[i + 1]
            # Spans may have gaps (blank lines between sections) but must not overlap
            assert a.byte_end <= b.byte_start, (
                f"Overlap: {a.chunk_id} end={a.byte_end} > {b.chunk_id} start={b.byte_start}"
            )

    def test_no_heading_without_content_on_oversized_split(self) -> None:
        """Heading line must not become a standalone tiny chunk when section is split."""
        from pipeline.chunker import chunk_document

        # Build a body where one section is very large
        para = "word " * 200  # ~1000 tokens at 1 tok/4 chars
        body = "# BigSection\n\n" + para + "\n\n" + para
        chunks = chunk_document(body, "a" * 64, "Doc", self._tok, max_tokens=100)

        for c in chunks:
            # No chunk should be ONLY a heading line
            text = c.embedded_text.split("]\n", 1)[-1].strip()
            assert not (text.startswith("#") and "\n" not in text and len(text) < 60), (
                f"Heading-only chunk found: {text!r}"
            )

    def test_table_is_single_chunk(self) -> None:
        """A markdown table must never be split across chunks."""
        from pipeline.chunker import chunk_document

        table = (
            "| Name | Value |\n"
            "| --- | --- |\n"
            "| Alpha | 1 |\n"
            "| Beta | 2 |\n"
            "| Gamma | 3 |\n"
        )
        body = "# Section\n\nIntro paragraph.\n\n" + table + "\nMore content here.\n"
        chunks = chunk_document(body, "a" * 64, "Doc", self._tok)

        # Find chunks that contain any | character
        table_chunks = [c for c in chunks if "| --- |" in c.embedded_text]
        assert len(table_chunks) == 1, f"Table split into {len(table_chunks)} chunks"

        # The full table header must be in the single table chunk
        assert "| Name | Value |" in table_chunks[0].embedded_text
        assert "| Gamma | 3 |" in table_chunks[0].embedded_text

    def test_sequence_is_monotone(self) -> None:
        from pipeline.chunker import chunk_document

        body = (FIXTURES / "sample.md").read_text()
        chunks = chunk_document(body, "a" * 64, "Doc", self._tok)
        seqs = [c.sequence for c in chunks]
        assert seqs == list(range(len(chunks)))

    def test_heading_path_populated(self) -> None:
        from pipeline.chunker import chunk_document

        body = (FIXTURES / "sample.md").read_text()
        chunks = chunk_document(body, "a" * 64, "Doc", self._tok)
        # At least some chunks should have non-empty heading_path
        assert any(len(c.heading_path) > 0 for c in chunks)

    def test_legal_body_chunks(self) -> None:
        """Bulgarian legal fixture should chunk after promote_legal promotion."""
        from pipeline.chunker import chunk_document
        from pipeline.promote_legal import promote_legal

        raw = (FIXTURES / "legal_bg.md").read_text()
        body = promote_legal(raw)
        chunks = chunk_document(body, "b" * 64, "Закон", self._tok)
        assert len(chunks) >= 3

        # All chunks reference the doc
        for c in chunks:
            assert c.doc_id == "b" * 64

    def test_byte_offsets_are_utf8(self) -> None:
        """Byte offsets must correctly handle multi-byte Cyrillic characters."""
        from pipeline.chunker import chunk_document
        from pipeline.promote_legal import promote_legal

        raw = (FIXTURES / "legal_bg.md").read_text()
        body = promote_legal(raw)
        body_bytes = body.encode("utf-8")
        chunks = chunk_document(body, "c" * 64, "Закон", self._tok)

        for c in chunks:
            # Byte slice must decode cleanly
            sliced = body_bytes[c.byte_start : c.byte_end].decode("utf-8")
            assert sliced in c.embedded_text

    def test_chunk_id_unique_within_document(self) -> None:
        from pipeline.chunker import chunk_document

        body = (FIXTURES / "sample.md").read_text()
        chunks = chunk_document(body, "a" * 64, "Doc", self._tok, max_tokens=50)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_parent_chunk_set_for_subsections(self) -> None:
        from pipeline.chunker import chunk_document

        body = "# H1\n\nIntro\n\n## H2\n\nContent\n\n### H3\n\nDeep content\n"
        chunks = chunk_document(body, "a" * 64, "Doc", self._tok)

        h3_chunk = next((c for c in chunks if c.heading_level == 3), None)
        h2_chunk = next((c for c in chunks if c.heading_level == 2), None)

        assert h3_chunk is not None, "Expected H3 chunk"
        assert h2_chunk is not None, "Expected H2 chunk"
        assert h3_chunk.parent_chunk == h2_chunk.chunk_id, (
            f"H3 parent={h3_chunk.parent_chunk!r}, expected {h2_chunk.chunk_id!r}"
        )


# ---------------------------------------------------------------------------
# FTS5 keyword store
# ---------------------------------------------------------------------------


class TestFtsDb:
    def test_upsert_and_search(self, fts_db, embedder) -> None:  # type: ignore[no-untyped-def]
        from pipeline.chunker import Chunk

        c = Chunk(
            chunk_id="abc#sec#0",
            doc_id="a" * 64,
            heading_path=["Introduction"],
            heading_level=1,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=50,
            embedded_text="[Doc > Introduction]\nvector database stores embeddings",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        fts_db.upsert(c)
        hits = fts_db.search("vector database", top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == "abc#sec#0"
        assert hits[0][1] == 1  # rank position

    def test_search_excludes_historical(self, fts_db) -> None:  # type: ignore[no-untyped-def]
        from pipeline.chunker import Chunk

        current = Chunk(
            chunk_id="curr#sec#0",
            doc_id="b" * 64,
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=30,
            embedded_text="[Doc]\ncurrent version text",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        superseded = Chunk(
            chunk_id="old#sec#0",
            doc_id="c" * 64,
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=30,
            embedded_text="[Doc]\ncurrent version text old",
            valid_until="2024-01-01",
            extraction_method="deterministic",
            confidence=1.0,
        )
        fts_db.upsert(current)
        fts_db.upsert(superseded)

        # Default: only current
        hits = fts_db.search("current version", top_k=10)
        hit_ids = [h[0] for h in hits]
        assert "curr#sec#0" in hit_ids
        assert "old#sec#0" not in hit_ids

        # Historical: both
        hits_all = fts_db.search("current version", top_k=10, include_historical=True)
        all_ids = [h[0] for h in hits_all]
        assert "curr#sec#0" in all_ids
        assert "old#sec#0" in all_ids

    def test_search_with_punctuation_does_not_raise(self, fts_db) -> None:  # type: ignore[no-untyped-def]
        """Queries containing FTS5 syntax chars (comma, parens, quotes) must not crash."""
        from pipeline.chunker import Chunk

        c = Chunk(
            chunk_id="bg#sec#0",
            doc_id="d" * 64,
            heading_path=["Услуги"],
            heading_level=1,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=60,
            embedded_text="[Doc > Услуги]\nУслуги извършени по електронен път",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        fts_db.upsert(c)
        # Must not raise sqlite3.OperationalError
        hits = fts_db.search("Услуги, извършени по електронен път", top_k=5)
        assert any(h[0] == "bg#sec#0" for h in hits)

    def test_sanitize_fts5(self) -> None:
        from pipeline.indexer import _sanitize_fts5

        assert _sanitize_fts5("hello, world") == '"hello" "world"'
        assert _sanitize_fts5("AND OR NOT") == '"AND" "OR" "NOT"'
        assert _sanitize_fts5('say "quoted"') == '"say" "quoted"'
        assert _sanitize_fts5("word* (prefix)") == '"word" "prefix"'
        assert _sanitize_fts5("   ") == '""'
        assert _sanitize_fts5(",,,") == '""'
        # Unicode / Cyrillic preserved
        assert _sanitize_fts5("Услуги, извършени") == '"Услуги" "извършени"'

    def test_idempotent_upsert(self, fts_db) -> None:  # type: ignore[no-untyped-def]
        from pipeline.chunker import Chunk

        c = Chunk(
            chunk_id="dup#sec#0",
            doc_id="d" * 64,
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=20,
            embedded_text="[Doc]\nsome content",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        fts_db.upsert(c)
        fts_db.upsert(c)  # second upsert must not duplicate

        hits = fts_db.search("some content", top_k=10)
        assert len(hits) == 1

    def test_delete_by_doc(self, fts_db) -> None:  # type: ignore[no-untyped-def]
        from pipeline.chunker import Chunk

        c = Chunk(
            chunk_id="del#sec#0",
            doc_id="e" * 64,
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=20,
            embedded_text="[Doc]\ndelete me content",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        fts_db.upsert(c)
        assert len(fts_db.search("delete me", top_k=5)) == 1

        fts_db.delete_by_doc("e" * 64)
        assert len(fts_db.search("delete me", top_k=5)) == 0


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class TestVectorStore:
    def test_upsert_and_query(self, in_memory_store, embedder) -> None:  # type: ignore[no-untyped-def]
        from pipeline.chunker import Chunk

        chunks = [
            Chunk(
                chunk_id=f"doc#sec#{i}",
                doc_id="a" * 64,
                heading_path=["Section"],
                heading_level=1,
                parent_chunk=None,
                sequence=i,
                byte_start=i * 10,
                byte_end=(i + 1) * 10,
                embedded_text=f"[Doc > Section]\ncontent {i}",
                valid_until=None,
                extraction_method="deterministic",
                confidence=1.0,
            )
            for i in range(3)
        ]
        vectors = embedder.encode([c.embedded_text for c in chunks])
        in_memory_store.upsert(chunks, vectors)

        query_vec = vectors[0]
        hits = in_memory_store.query(query_vec, top_k=3)
        assert len(hits) >= 1
        assert hits[0][0] == "doc#sec#0"  # closest to itself

    def test_query_filters_historical(self, in_memory_store, embedder) -> None:  # type: ignore[no-untyped-def]
        from pipeline.chunker import Chunk

        current = Chunk(
            chunk_id="curr#0",
            doc_id="b" * 64,
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=20,
            embedded_text="[Doc]\ncurrent",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        old = Chunk(
            chunk_id="old#0",
            doc_id="c" * 64,
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=20,
            embedded_text="[Doc]\ncurrent",
            valid_until="2024-01-01",
            extraction_method="deterministic",
            confidence=1.0,
        )
        vecs = embedder.encode([current.embedded_text, old.embedded_text])
        in_memory_store.upsert([current, old], vecs)

        # Default: only current version
        hits = in_memory_store.query(vecs[0], top_k=10)
        ids = [h[0] for h in hits]
        assert "curr#0" in ids
        assert "old#0" not in ids

        # With historical: both
        hits_all = in_memory_store.query(vecs[0], top_k=10, include_historical=True)
        all_ids = [h[0] for h in hits_all]
        assert "curr#0" in all_ids
        assert "old#0" in all_ids

    def test_delete_by_doc(self, in_memory_store, embedder) -> None:  # type: ignore[no-untyped-def]
        from pipeline.chunker import Chunk

        c = Chunk(
            chunk_id="del#0",
            doc_id="d" * 64,
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=10,
            embedded_text="[Doc]\nhello",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        vecs = embedder.encode([c.embedded_text])
        in_memory_store.upsert([c], vecs)

        hits_before = in_memory_store.query(vecs[0], top_k=5)
        assert any(h[0] == "del#0" for h in hits_before)

        in_memory_store.delete_by_doc("d" * 64)
        hits_after = in_memory_store.query(vecs[0], top_k=5)
        assert not any(h[0] == "del#0" for h in hits_after)


# ---------------------------------------------------------------------------
# index_document integration
# ---------------------------------------------------------------------------


class TestIndexDocument:
    def test_index_updates_manifest(
        self,
        config,  # type: ignore[no-untyped-def]
        in_memory_store,
        fts_db,
        embedder,
    ) -> None:
        from pipeline.acquisition import ingest_document
        from pipeline.indexer import index_document

        fixture = FIXTURES / "sample.md"
        doc_id, folder = ingest_document(str(fixture), config)

        chunks = index_document(folder, config, embedder, in_memory_store, fts_db)
        assert len(chunks) > 0

        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["chunk_count"] == len(chunks)

    def test_index_idempotent(
        self,
        config,  # type: ignore[no-untyped-def]
        in_memory_store,
        fts_db,
        embedder,
    ) -> None:
        from pipeline.acquisition import ingest_document
        from pipeline.indexer import index_document

        fixture = FIXTURES / "sample.txt"
        _, folder = ingest_document(str(fixture), config)

        chunks1 = index_document(folder, config, embedder, in_memory_store, fts_db)
        chunks2 = index_document(folder, config, embedder, in_memory_store, fts_db)

        assert len(chunks1) == len(chunks2)

        # FTS5 should not have duplicates
        fts_hits = fts_db.search("artificial intelligence", top_k=20)
        chunk_ids = [h[0] for h in fts_hits]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_all_chunks_schema_valid(
        self,
        config,  # type: ignore[no-untyped-def]
        in_memory_store,
        fts_db,
        embedder,
    ) -> None:
        from pipeline.acquisition import ingest_document
        from pipeline.indexer import _validate_chunk, index_document

        fixture = FIXTURES / "sample.md"
        _, folder = ingest_document(str(fixture), config)
        chunks = index_document(folder, config, embedder, in_memory_store, fts_db)

        for c in chunks:
            _validate_chunk(c)  # must not raise

    def test_byte_span_slice_matches_embedded_body(
        self,
        config,  # type: ignore[no-untyped-def]
        in_memory_store,
        fts_db,
        embedder,
    ) -> None:
        from pipeline.acquisition import ingest_document
        from pipeline.indexer import index_document
        from pipeline.normalize import strip_front_matter

        fixture = FIXTURES / "sample.md"
        _, folder = ingest_document(str(fixture), config)
        chunks = index_document(folder, config, embedder, in_memory_store, fts_db)

        doc_md = (folder / "document.md").read_text(encoding="utf-8")
        _, body = strip_front_matter(doc_md)
        body_bytes = body.encode("utf-8")

        for c in chunks:
            sliced = body_bytes[c.byte_start : c.byte_end].decode("utf-8")
            assert sliced in c.embedded_text, (
                f"chunk {c.chunk_id}: body_bytes slice not in embedded_text"
            )

    def test_legal_fixture_indexed(
        self,
        config,  # type: ignore[no-untyped-def]
        in_memory_store,
        fts_db,
        embedder,
    ) -> None:
        from pipeline.acquisition import ingest_document
        from pipeline.indexer import index_document

        fixture = FIXTURES / "legal_bg.md"
        _, folder = ingest_document(str(fixture), config, doc_type="legal_code")
        chunks = index_document(folder, config, embedder, in_memory_store, fts_db)

        assert len(chunks) >= 3
        # Legal content should be retrievable
        hits = fts_db.search("задълженията", top_k=5)
        assert len(hits) >= 1


# ---------------------------------------------------------------------------
# Retrieval (RRF fusion)
# ---------------------------------------------------------------------------


class TestRetrieval:
    def test_rrf_fusion_deduplicates(self) -> None:
        from pipeline.retrieval import _rrf

        vector_hits = [("a", 1), ("b", 2), ("c", 3)]
        fts_hits = [("b", 1), ("a", 2), ("d", 3)]
        fused = _rrf(vector_hits, fts_hits)

        ids = [cid for cid, _ in fused]
        # No duplicate ids
        assert len(ids) == len(set(ids))
        # "a" and "b" appear in both lists so they should score highest
        assert ids[0] in ("a", "b")
        assert ids[1] in ("a", "b")

    def test_search_returns_results(
        self,
        config,  # type: ignore[no-untyped-def]
        in_memory_store,
        fts_db,
        embedder,
    ) -> None:
        from pipeline.acquisition import ingest_document
        from pipeline.indexer import index_document
        from pipeline.retrieval import search

        fixture = FIXTURES / "sample.md"
        _, folder = ingest_document(str(fixture), config)
        index_document(folder, config, embedder, in_memory_store, fts_db)

        results = search("vector database embeddings", embedder, in_memory_store, fts_db, top_k=3)
        assert len(results) >= 1

        for r in results:
            assert r.chunk_id
            assert r.source_id
            assert r.score > 0

    def test_search_html_fixture(
        self,
        config,  # type: ignore[no-untyped-def]
        in_memory_store,
        fts_db,
        embedder,
    ) -> None:
        from pipeline.acquisition import ingest_document
        from pipeline.indexer import index_document
        from pipeline.retrieval import search

        fixture = FIXTURES / "sample.html"
        _, folder = ingest_document(str(fixture), config)
        index_document(folder, config, embedder, in_memory_store, fts_db)

        results = search("knowledge graph nodes edges", embedder, in_memory_store, fts_db, top_k=3)
        assert len(results) >= 1

    def test_search_excludes_superseded_by_default(
        self,
        config,  # type: ignore[no-untyped-def]
        in_memory_store,
        fts_db,
        embedder,
    ) -> None:
        """Chunks with valid_until set must not appear in default search."""
        from pipeline.chunker import Chunk
        from pipeline.retrieval import search

        # Directly inject a superseded chunk
        c = Chunk(
            chunk_id="superseded#0",
            doc_id="f" * 64,
            heading_path=[],
            heading_level=0,
            parent_chunk=None,
            sequence=0,
            byte_start=0,
            byte_end=30,
            embedded_text="[Old Law]\nsuperseded legal text",
            valid_until="2023-12-31",
            extraction_method="deterministic",
            confidence=1.0,
        )
        vecs = embedder.encode([c.embedded_text])
        in_memory_store.upsert([c], vecs)
        fts_db.upsert(c)

        results = search("superseded legal text", embedder, in_memory_store, fts_db, top_k=5)
        assert not any(r.chunk_id == "superseded#0" for r in results)

        results_hist = search(
            "superseded legal text",
            embedder,
            in_memory_store,
            fts_db,
            top_k=5,
            include_historical=True,
        )
        assert any(r.chunk_id == "superseded#0" for r in results_hist)
