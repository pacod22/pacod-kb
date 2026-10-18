"""Phase 3 tests: versioning, supersession, as-of queries, and rebuild."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from pipeline.acquisition import ingest_document
from pipeline.config import load_config
from pipeline.indexer import FtsDb, index_document
from pipeline.retrieval import search
from pipeline.vectorstore import QdrantVectorStore
from pipeline.versioning import find_doc_folder, rebuild_index, supersede_document

# ─── shared helpers ──────────────────────────────────────────────────────────


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


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def kb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "kb"
    (root / "documents").mkdir(parents=True)
    (root / "index").mkdir()
    (root / "review").mkdir()
    monkeypatch.setenv("KB_ROOT", str(root))
    monkeypatch.setenv("GIT_AUTOCOMMIT", "false")
    monkeypatch.setenv("LLM_CONVERTER_ENABLED", "false")
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    return root


@pytest.fixture()
def config(kb_root: Path):  # noqa: ANN201
    return load_config()


@pytest.fixture()
def embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture()
def stores(config):  # noqa: ANN201
    """In-memory (vector_store, fts_db) pair."""
    qclient = QdrantClient(":memory:")
    vstore = QdrantVectorStore(qclient, config.qdrant_collection)
    fts = FtsDb(config.index_dir / "fts5.db")
    yield vstore, fts
    fts.close()


# ─── 1. supersession ─────────────────────────────────────────────────────────

_V1 = (
    "# Policy Document\n\n"
    "SHARED_KEYWORD version one content. "
    "This document describes the initial policy.\n\n"
    "It contains multiple paragraphs so the chunker produces at least one chunk.\n"
)
_V2 = (
    "# Policy Document\n\n"
    "SHARED_KEYWORD version two content. "
    "This document describes the updated policy.\n\n"
    "It contains multiple paragraphs so the chunker produces at least one chunk.\n"
)
_VALID_UNTIL = "2025-12-31"


@pytest.fixture()
def versioned(tmp_path: Path, config, embedder, stores):
    """Ingest v1, ingest v2 superseding it, call supersede_document."""
    vstore, fts = stores

    src1 = _write(tmp_path, "policy_v1.txt", _V1)
    doc_id_v1, folder_v1 = ingest_document(str(src1), config)
    index_document(folder_v1, config, embedder, vstore, fts)

    src2 = _write(tmp_path, "policy_v2.txt", _V2)
    doc_id_v2, folder_v2 = ingest_document(str(src2), config, supersedes=doc_id_v1)
    index_document(folder_v2, config, embedder, vstore, fts)

    supersede_document(doc_id_v1, _VALID_UNTIL, config, vstore, fts)

    return {
        "doc_id_v1": doc_id_v1,
        "folder_v1": folder_v1,
        "doc_id_v2": doc_id_v2,
        "folder_v2": folder_v2,
    }


class TestSupersession:
    # ── existence (inspect stores directly, not via search) ───────────────────

    def test_v1_chunks_remain_in_fts(self, versioned, stores):
        """Hard rule 9: prior version chunks must not be deleted from FTS."""
        _, fts = stores
        rows = fts._con.execute(
            "SELECT chunk_id FROM chunks_meta WHERE doc_id = ?",
            (versioned["doc_id_v1"],),
        ).fetchall()
        assert len(rows) > 0

    def test_v2_chunks_in_fts(self, versioned, stores):
        _, fts = stores
        rows = fts._con.execute(
            "SELECT chunk_id FROM chunks_meta WHERE doc_id = ?",
            (versioned["doc_id_v2"],),
        ).fetchall()
        assert len(rows) > 0

    # ── valid_until propagation ───────────────────────────────────────────────

    def test_v1_fts_chunks_have_valid_until(self, versioned, stores):
        _, fts = stores
        rows = fts._con.execute(
            "SELECT valid_until FROM chunks_meta WHERE doc_id = ?",
            (versioned["doc_id_v1"],),
        ).fetchall()
        assert all(row[0] == _VALID_UNTIL for row in rows)

    def test_v2_fts_chunks_valid_until_null(self, versioned, stores):
        _, fts = stores
        rows = fts._con.execute(
            "SELECT valid_until FROM chunks_meta WHERE doc_id = ?",
            (versioned["doc_id_v2"],),
        ).fetchall()
        assert all(row[0] is None for row in rows)

    def test_v1_front_matter_valid_until(self, versioned):
        from pipeline.normalize import strip_front_matter

        doc_md = (versioned["folder_v1"] / "document.md").read_text(encoding="utf-8")
        front, _ = strip_front_matter(doc_md)
        assert front["valid_until"] == _VALID_UNTIL

    def test_v1_manifest_valid_until(self, versioned):
        manifest = json.loads((versioned["folder_v1"] / "manifest.json").read_text())
        assert manifest["valid_until"] == _VALID_UNTIL

    def test_v2_front_matter_supersedes(self, versioned):
        from pipeline.normalize import strip_front_matter

        doc_md = (versioned["folder_v2"] / "document.md").read_text(encoding="utf-8")
        front, _ = strip_front_matter(doc_md)
        assert front["supersedes"] == versioned["doc_id_v1"]

    def test_v1_body_unchanged_after_supersede(self, versioned):
        """supersede_document must not alter the stored body text."""
        from pipeline.normalize import strip_front_matter

        doc_md = (versioned["folder_v1"] / "document.md").read_text(encoding="utf-8")
        _, body = strip_front_matter(doc_md)
        assert "SHARED_KEYWORD" in body
        assert "version one" in body

    # ── filtering behaviour ───────────────────────────────────────────────────

    def test_default_search_excludes_v1(self, versioned, embedder, stores):
        vstore, fts = stores
        results = search("SHARED_KEYWORD", embedder, vstore, fts, top_k=20)
        doc_ids = {r.source_id for r in results}
        assert versioned["doc_id_v1"] not in doc_ids

    def test_historical_search_includes_v1(self, versioned, embedder, stores):
        vstore, fts = stores
        results = search(
            "SHARED_KEYWORD", embedder, vstore, fts, top_k=20, include_historical=True
        )
        doc_ids = {r.source_id for r in results}
        assert versioned["doc_id_v1"] in doc_ids

    def test_as_of_before_valid_until_includes_v1(self, versioned, embedder, stores):
        """as_of=2025-12-30 < valid_until=2025-12-31 → v1 still valid."""
        vstore, fts = stores
        results = search(
            "SHARED_KEYWORD", embedder, vstore, fts, top_k=20, as_of="2025-12-30"
        )
        doc_ids = {r.source_id for r in results}
        assert versioned["doc_id_v1"] in doc_ids

    def test_as_of_equal_to_valid_until_excludes_v1(self, versioned, embedder, stores):
        """as_of=2025-12-31 == valid_until → v1 expired (valid_until <= as_of)."""
        vstore, fts = stores
        results = search(
            "SHARED_KEYWORD", embedder, vstore, fts, top_k=20, as_of="2025-12-31"
        )
        doc_ids = {r.source_id for r in results}
        assert versioned["doc_id_v1"] not in doc_ids

    def test_as_of_after_valid_until_excludes_v1(self, versioned, embedder, stores):
        """as_of=2026-01-01 > valid_until → v1 expired."""
        vstore, fts = stores
        results = search(
            "SHARED_KEYWORD", embedder, vstore, fts, top_k=20, as_of="2026-01-01"
        )
        doc_ids = {r.source_id for r in results}
        assert versioned["doc_id_v1"] not in doc_ids

    def test_as_of_v2_always_appears(self, versioned, embedder, stores):
        """v2 (valid_until=None) appears regardless of as_of."""
        vstore, fts = stores
        results = search(
            "SHARED_KEYWORD", embedder, vstore, fts, top_k=20, as_of="2030-01-01"
        )
        doc_ids = {r.source_id for r in results}
        assert versioned["doc_id_v2"] in doc_ids

    def test_as_of_results_carry_valid_until(self, versioned, embedder, stores):
        """SearchResult.valid_until reflects the chunk's stored value."""
        vstore, fts = stores
        results = search(
            "SHARED_KEYWORD", embedder, vstore, fts, top_k=20, include_historical=True
        )
        for r in results:
            if r.source_id == versioned["doc_id_v1"]:
                assert r.valid_until == _VALID_UNTIL
            else:
                assert r.valid_until is None

    def test_find_doc_folder_by_exact_id(self, versioned, config):
        folder = find_doc_folder(versioned["doc_id_v1"], config)
        assert folder is not None
        assert folder == versioned["folder_v1"]

    def test_find_doc_folder_by_prefix(self, versioned, config):
        prefix = versioned["doc_id_v1"][:8]
        folder = find_doc_folder(prefix, config, prefix=True)
        assert folder == versioned["folder_v1"]

    def test_find_doc_folder_missing_returns_none(self, config):
        assert find_doc_folder("0" * 64, config) is None


# ─── 2. same-version re-conversion ───────────────────────────────────────────

_STABLE = "# Re-Conversion Test\n\nSome stable content for re-indexing purposes.\n"


class TestReConversion:
    """index_document is idempotent: old chunks gone, count unchanged."""

    def test_stale_chunks_removed_on_reindex(self, tmp_path: Path, config, embedder, stores):
        """Inject a stale chunk, call index_document again — stale must vanish."""
        from pipeline.chunker import Chunk

        vstore, fts = stores
        src = _write(tmp_path, "reconv.txt", _STABLE)
        doc_id, folder = ingest_document(str(src), config)
        chunks = index_document(folder, config, embedder, vstore, fts)
        assert len(chunks) > 0

        # Inject stale chunk into both stores
        stale = Chunk(
            chunk_id=f"{doc_id[:8]}#stale#0",
            doc_id=doc_id,
            heading_path=["Stale"],
            heading_level=1,
            parent_chunk=None,
            sequence=999,
            byte_start=0,
            byte_end=4,
            embedded_text="stale placeholder",
            valid_until=None,
            extraction_method="deterministic",
            confidence=1.0,
        )
        vstore.upsert([stale], embedder.encode(["stale placeholder"]))
        fts.upsert(stale)

        rows_before = fts._con.execute(
            "SELECT chunk_id FROM chunks_meta WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        assert any("stale" in row[0] for row in rows_before)

        # Re-index — must delete stale chunk via delete_by_doc before re-inserting
        chunks_after = index_document(folder, config, embedder, vstore, fts)

        rows_after = fts._con.execute(
            "SELECT chunk_id FROM chunks_meta WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        assert all("stale" not in row[0] for row in rows_after)
        assert len(rows_after) == len(chunks_after)

    def test_no_duplicate_chunks_on_repeated_reindex(
        self, tmp_path: Path, config, embedder, stores
    ):
        vstore, fts = stores
        src = _write(tmp_path, "nodup.txt", _STABLE)
        doc_id, folder = ingest_document(str(src), config)

        index_document(folder, config, embedder, vstore, fts)
        index_document(folder, config, embedder, vstore, fts)

        rows = fts._con.execute(
            "SELECT chunk_id FROM chunks_meta WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        chunk_ids = [row[0] for row in rows]
        assert len(chunk_ids) == len(set(chunk_ids)), "duplicate chunk_ids found"


# ─── 3. rebuild_index ────────────────────────────────────────────────────────


class TestRebuild:
    def test_rebuild_same_chunk_metadata(self, tmp_path: Path, config, embedder, stores):
        """After rebuild, chunk metadata snapshot (id, spans, valid_until) matches."""
        vstore, fts = stores

        src_a = _write(tmp_path, "rebuild_a.txt", "# Doc A\n\nContent of document A.\n")
        src_b = _write(tmp_path, "rebuild_b.txt", "# Doc B\n\nContent of document B.\n")

        _, folder_a = ingest_document(str(src_a), config)
        _, folder_b = ingest_document(str(src_b), config)
        index_document(folder_a, config, embedder, vstore, fts)
        index_document(folder_b, config, embedder, vstore, fts)

        def snapshot() -> dict[str, tuple]:
            rows = fts._con.execute(
                "SELECT chunk_id, byte_start, byte_end, valid_until, parent_chunk, sequence "
                "FROM chunks_meta ORDER BY chunk_id"
            ).fetchall()
            return {row[0]: row[1:] for row in rows}

        before = snapshot()
        ok, err = rebuild_index(config, embedder, vstore, fts)
        assert err == 0
        assert ok == 2
        assert snapshot() == before

    def test_rebuild_preserves_valid_until_from_front_matter(
        self, tmp_path: Path, config, embedder, stores
    ):
        """rebuild reads valid_until from document.md front matter → correct after rebuild."""
        vstore, fts = stores

        src1 = _write(tmp_path, "ver1.txt", "# Rebuild Policy\n\nVersion one details.\n")
        src2 = _write(tmp_path, "ver2.txt", "# Rebuild Policy\n\nVersion two details.\n")

        doc_id_v1, folder_v1 = ingest_document(str(src1), config)
        doc_id_v2, folder_v2 = ingest_document(str(src2), config, supersedes=doc_id_v1)

        index_document(folder_v1, config, embedder, vstore, fts)
        index_document(folder_v2, config, embedder, vstore, fts)

        valid_until = "2025-06-01"
        supersede_document(doc_id_v1, valid_until, config, vstore, fts)

        # Rebuild from scratch — valid_until comes from document.md front matter
        ok, err = rebuild_index(config, embedder, vstore, fts)
        assert err == 0
        assert ok == 2

        v1_rows = fts._con.execute(
            "SELECT valid_until FROM chunks_meta WHERE doc_id = ?", (doc_id_v1,)
        ).fetchall()
        assert len(v1_rows) > 0
        assert all(row[0] == valid_until for row in v1_rows)

        v2_rows = fts._con.execute(
            "SELECT valid_until FROM chunks_meta WHERE doc_id = ?", (doc_id_v2,)
        ).fetchall()
        assert all(row[0] is None for row in v2_rows)

    def test_rebuild_empty_docs_dir(self, config, embedder, stores):
        vstore, fts = stores
        ok, err = rebuild_index(config, embedder, vstore, fts)
        assert ok == 0
        assert err == 0

    def test_rebuild_reports_progress(self, tmp_path: Path, config, embedder, stores):
        vstore, fts = stores
        src = _write(
            tmp_path,
            "rebuild_progress.txt",
            "# Rebuild Progress\n\n"
            "This is an English test document used for rebuild progress checks.\n",
        )
        _, folder = ingest_document(str(src), config)
        index_document(folder, config, embedder, vstore, fts)

        progress_events: list[tuple[int, int, str]] = []
        ok, err = rebuild_index(
            config,
            embedder,
            vstore,
            fts,
            progress_cb=lambda cur, total, msg: progress_events.append((cur, total, msg)),
        )

        assert ok == 1
        assert err == 0
        assert progress_events[0][:2] == (0, 1)
        assert progress_events[-1][:2] == (1, 1)

    def test_drop_all_clears_fts(self, tmp_path: Path, config, embedder, stores):
        vstore, fts = stores
        src = _write(tmp_path, "clear_test.txt", "# Clear\n\nSome content.\n")
        _doc_id, folder = ingest_document(str(src), config)
        index_document(folder, config, embedder, vstore, fts)

        fts.drop_all()
        rows = fts._con.execute("SELECT count(*) FROM chunks_meta").fetchone()
        assert rows[0] == 0

    def test_drop_all_clears_vector_store(self, tmp_path: Path, config, embedder, stores):
        vstore, fts = stores
        src = _write(tmp_path, "clear_vtest.txt", "# Clear\n\nSome vector content.\n")
        _doc_id, folder = ingest_document(str(src), config)
        index_document(folder, config, embedder, vstore, fts)

        vstore.drop_all()
        results = vstore.query([0.0] * 1024, top_k=10, include_historical=True)
        assert results == []


# ─── 4. set_valid_until helpers ──────────────────────────────────────────────


class TestSetValidUntil:
    def test_fts_set_valid_until(self, tmp_path: Path, config, embedder, stores):
        vstore, fts = stores
        src = _write(tmp_path, "sv_fts.txt", "# FTS Valid Until\n\nContent here.\n")
        doc_id, folder = ingest_document(str(src), config)
        index_document(folder, config, embedder, vstore, fts)

        fts.set_valid_until(doc_id, "2024-01-01")
        rows = fts._con.execute(
            "SELECT valid_until FROM chunks_meta WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        assert all(row[0] == "2024-01-01" for row in rows)

    def test_vector_set_valid_until(self, tmp_path: Path, config, embedder, stores):
        vstore, fts = stores
        src = _write(tmp_path, "sv_vec.txt", "# Vector Valid Until\n\nContent here.\n")
        doc_id, folder = ingest_document(str(src), config)
        index_document(folder, config, embedder, vstore, fts)

        vstore.set_valid_until(doc_id, "2024-06-15")

        # Only historical query sees the point (valid_until is now set)
        with_hist = vstore.query([0.0] * 1024, top_k=10, include_historical=True)
        without_hist = vstore.query([0.0] * 1024, top_k=10, include_historical=False)
        assert len(with_hist) > 0
        assert len(without_hist) == 0
