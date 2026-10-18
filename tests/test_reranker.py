"""Second-stage reranking: ordering, fail-open, min_score, background loading, the API."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient

from pipeline.chunker import Chunk
from pipeline.indexer import FtsDb
from pipeline.inference import InferenceGate
from pipeline.reranker import CrossEncoderReranker, RerankerHolder, RerankUnavailableError
from pipeline.retrieval import SearchOptions, SearchTrace, search
from pipeline.vectorstore import QdrantVectorStore
from tests.test_phase3 import MockEmbedder


class _KeywordReranker:
    """Relevance = 0.9 when the passage mentions "zebra", else 0.1."""

    model_name = "stub"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append(list(passages))
        return [0.9 if "zebra" in p else 0.1 for p in passages]


class _FailingReranker:
    model_name = "broken"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        raise self._exc


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=cid.split("#")[0], heading_path=["Doc"], heading_level=1,
        parent_chunk=None, sequence=0, byte_start=0, byte_end=len(text.encode()),
        embedded_text=text, valid_until=None, extraction_method="deterministic",
        confidence=1.0, text=text,
    )


@pytest.fixture()
def stores(tmp_path: Path) -> Any:
    emb = MockEmbedder()
    vstore = QdrantVectorStore(QdrantClient(":memory:"), "c")
    fts = FtsDb(tmp_path / "fts5.db")
    chunks = [_chunk(f"d{i}#s#0", f"lion facts number {i}") for i in range(12)]
    chunks.append(_chunk("z#s#0", "zebra facts"))
    chunks.append(_chunk("dup#s#0", "lion facts number 3"))  # same text as d3
    vstore.upsert(chunks, emb.encode([c.embedded_text for c in chunks]))
    fts.upsert_batch(chunks, "p")
    yield emb, vstore, fts
    fts.close()


def _run(stores: Any, reranker: Any = None, **opts: Any) -> tuple[list[Any], SearchTrace]:
    emb, vstore, fts = stores
    trace = SearchTrace()
    hits = search(
        "facts", emb, vstore, fts, top_k=5, project="p",
        options=SearchOptions(**opts), reranker=reranker, trace=trace,
    )
    return hits, trace


def test_reranker_reorders_by_relevance(stores: Any) -> None:
    hits, trace = _run(stores, _KeywordReranker())
    assert trace.reranked
    assert hits[0].chunk_id == "z#s#0"
    assert hits[0].score == hits[0].rerank_score == 0.9
    assert hits[0].rrf_score > 0
    assert "rerank" in trace.timings_ms


def test_identical_texts_are_scored_once(stores: Any) -> None:
    reranker = _KeywordReranker()
    _run(stores, reranker)
    passages = reranker.calls[0]
    assert len(passages) == len(set(passages))


@pytest.mark.parametrize("exc", [RuntimeError("boom"), RerankUnavailableError("busy")])
def test_failure_keeps_the_fused_order(stores: Any, exc: Exception) -> None:
    plain, _ = _run(stores)
    hits, trace = _run(stores, _FailingReranker(exc))
    assert not trace.reranked
    assert [h.chunk_id for h in hits] == [h.chunk_id for h in plain]
    assert all(h.rerank_score is None for h in hits)


def test_min_score_applies_only_when_reranked(stores: Any) -> None:
    hits, _ = _run(stores, _KeywordReranker(), min_score=0.5)
    assert [h.chunk_id for h in hits] == ["z#s#0"]
    plain, _ = _run(stores, None, min_score=0.5)
    assert len(plain) == 5


def test_without_reranker_the_candidate_depth_is_unchanged(stores: Any) -> None:
    _, plain = _run(stores)
    _, deep = _run(stores, _KeywordReranker(), candidate_k=50)
    assert len(plain.candidates) <= 14
    assert len(deep.candidates) >= len(plain.candidates)


# ── CrossEncoderReranker with a stub model ──────────────────────────────────────────────


class _StubCrossEncoder:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.max_seq_length = kwargs.get("max_length")

    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> Any:
        return np.array([min(1.0, len(p) / 100) for _, p in pairs])


def test_cross_encoder_scores_and_respects_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sentence_transformers.CrossEncoder", _StubCrossEncoder)
    gate = InferenceGate()
    rr = CrossEncoderReranker("stub", gate=gate, gate_timeout_s=0.05, code_revision="abc")
    assert rr._model.kwargs["config_kwargs"] == {"code_revision": "abc"}
    assert rr.score("q", ["x" * 10, "x" * 50]) == pytest.approx([0.1, 0.5])
    with gate.batch(), pytest.raises(RerankUnavailableError):
        rr.score("q", ["busy"])


def test_holder_loads_in_the_background() -> None:
    holder = RerankerHolder("stub")
    assert holder.status()["state"] == "disabled"
    holder.start(_KeywordReranker)
    for _ in range(100):
        if holder.state == "ready":
            break
        time.sleep(0.01)
    assert holder.state == "ready"
    assert holder.get() is not None


def test_holder_reports_a_failed_load() -> None:
    holder = RerankerHolder("bad")

    def broken() -> Any:
        raise OSError("model not found")

    holder.start(broken)
    for _ in range(100):
        if holder.state == "failed":
            break
        time.sleep(0.01)
    assert holder.status() == {
        "state": "failed", "model": "bad", "error": "OSError: model not found"
    }
    assert holder.get() is None


# ── the /query response and the runtime switch ──────────────────────────────────────────


def test_query_response_reports_reranking(
    monkeypatch: pytest.MonkeyPatch, config: Any, tmp_path: Path
) -> None:
    from contextlib import asynccontextmanager

    from fastapi.testclient import TestClient

    from pipeline import api
    from pipeline.gateway import Gateway
    from pipeline.indexer import make_fts_db
    from pipeline.ledger import ledger_session_factory

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    emb = MockEmbedder()
    vstore = QdrantVectorStore(QdrantClient(":memory:"), "c")
    chunks = [_chunk("z#s#0", "zebra facts"), _chunk("l#s#0", "lion facts")]
    vstore.upsert(chunks, emb.encode([c.embedded_text for c in chunks]))
    fts = make_fts_db(config)
    fts.upsert_batch(chunks, "p")
    fts.close()

    holder = RerankerHolder("stub")
    started: list[str] = []
    def factory(_cfg: Any) -> Any:
        started.append("x")
        return _KeywordReranker

    monkeypatch.setattr(api, "_reranker_factory", factory)
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "k")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    keys = ("config", "embedder", "vector_store", "reranker", "ledger_factory", "gateway")
    saved = {k: getattr(api.app.state, k, None) for k in keys}
    saved_runtime = dict(api._runtime)
    api.app.state.config = config
    api.app.state.embedder = emb
    api.app.state.vector_store = vstore
    api.app.state.reranker = holder
    api.app.state.ledger_factory = ledger_session_factory(config)
    api.app.state.gateway = Gateway("", "")
    api._runtime.update(active_project="p", reranker_enabled=False)
    try:
        with TestClient(api.app) as client:
            headers = {"X-API-Key": "k"}
            body = {"query": "facts", "wrap_untrusted": False}
            r = client.post("/query", json=body, headers=headers).json()
            assert (r["reranked"], r["score_kind"]) == (False, "rrf")
            assert all(hit["rerank_score"] is None and hit["rrf_score"] > 0 for hit in r["results"])

            s = client.post("/settings", json={"reranker_enabled": True}, headers=headers).json()
            assert started and s["reranker_enabled"] is True
            for _ in range(100):
                if holder.state == "ready":
                    break
                time.sleep(0.01)

            r = client.post("/query", json=body, headers=headers).json()
            assert (r["reranked"], r["score_kind"]) == (True, "rerank")
            assert r["results"][0]["chunk_id"] == "z#s#0"
            assert r["results"][0]["score"] == r["results"][0]["rerank_score"] == 0.9

            r = client.post("/query", json={**body, "rerank": False}, headers=headers).json()
            assert r["reranked"] is False

            r = client.post("/query", json={**body, "min_score": 0.5}, headers=headers).json()
            assert [hit["chunk_id"] for hit in r["results"]] == ["z#s#0"]

            assert client.get("/health").json()["reranker"]["state"] == "ready"
    finally:
        api._runtime.clear()
        api._runtime.update(saved_runtime)
        for k, v in saved.items():
            setattr(api.app.state, k, v)


def test_a_request_never_loads_the_reranker(
    monkeypatch: pytest.MonkeyPatch, config: Any
) -> None:
    """rerank: true with the setting off must not load a multi-GB model."""
    from contextlib import asynccontextmanager

    from fastapi.testclient import TestClient

    from pipeline import api
    from pipeline.gateway import Gateway
    from pipeline.ledger import ledger_session_factory

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    emb = MockEmbedder()
    vstore = QdrantVectorStore(QdrantClient(":memory:"), "c")
    vstore.upsert([_chunk("z#s#0", "zebra facts")], emb.encode(["zebra facts"]))
    holder = RerankerHolder("stub")
    monkeypatch.setattr(api, "_reranker_factory", lambda cfg: _KeywordReranker)
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "k")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    keys = ("config", "embedder", "vector_store", "reranker", "ledger_factory", "gateway")
    saved = {k: getattr(api.app.state, k, None) for k in keys}
    saved_runtime = dict(api._runtime)
    api.app.state.config = config
    api.app.state.embedder = emb
    api.app.state.vector_store = vstore
    api.app.state.reranker = holder
    api.app.state.ledger_factory = ledger_session_factory(config)
    api.app.state.gateway = Gateway("", "")
    api._runtime.update(active_project="p", reranker_enabled=False)
    try:
        with TestClient(api.app) as client:
            r = client.post(
                "/query",
                json={"query": "zebra", "rerank": True, "wrap_untrusted": False},
                headers={"X-API-Key": "k"},
            ).json()
        assert r["reranked"] is False
        assert holder.state == "disabled"
    finally:
        api._runtime.clear()
        api._runtime.update(saved_runtime)
        for k, v in saved.items():
            setattr(api.app.state, k, v)
