"""RemoteEmbedder and RemoteReranker: the shared embeddings service (pacod-embeddings) over HTTP.

The service is pacod-ap's embeddings_service: POST /embed {"texts": [...]} -> {"embeddings":
[[...]]} (BGE-M3 dense, normalized, 1024-d) and POST /rerank {"query", "documents"} ->
{"scores": [...]} in [0, 1]. A fake server behind httpx.MockTransport stands in for it.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from pipeline.indexer import EmbeddingServiceError, RemoteEmbedder

_DIM = 1024


class _FakeTokenizer:
    """One token per word, plus two special tokens (like the real one's <s> and </s>)."""

    model_max_length = 8192

    def __call__(self, texts: list[str], **_: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [list(range(len(t.split()) + 2)) for t in texts]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


class _FakeService:
    """POST /embed: vector[0] = the text's word count, so tests can check order and batching."""

    def __init__(self, *, dim: int = _DIM, failures: list[int] | None = None) -> None:
        self.dim = dim
        self.failures = list(failures or [])  # status codes to answer first, in order
        self.batches: list[list[str]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        if self.failures:
            return httpx.Response(self.failures.pop(0), json={"detail": "busy"})
        texts = json.loads(request.content)["texts"]
        self.batches.append(texts)
        vectors = [[float(len(t.split()))] + [0.0] * (self.dim - 1) for t in texts]
        return httpx.Response(200, json={"embeddings": vectors})


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pipeline.indexer.time.sleep", lambda _s: None)


def _embedder(service: Any, **kwargs: Any) -> RemoteEmbedder:
    handler = service.handle if hasattr(service, "handle") else service
    return RemoteEmbedder(
        "http://pacod-embeddings:8080/",
        tokenizer=_FakeTokenizer(),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_encode_restores_order_and_respects_the_token_budget() -> None:
    service = _FakeService()
    emb = _embedder(service, batch_tokens=12)
    texts = ["a b c d e f g h i j", "one", "x y", "p q r s"]
    vectors = emb.encode(texts)
    assert [v[0] for v in vectors] == [10.0, 1.0, 2.0, 4.0]
    assert all(len(v) == _DIM for v in vectors)
    # Ascending length, at most 12 padded tokens per request: [one, x y] then [p q r s], [a..j].
    assert service.batches == [["one", "x y"], ["p q r s"], ["a b c d e f g h i j"]]


def test_progress_ends_at_total() -> None:
    seen: list[tuple[int, int]] = []
    _embedder(_FakeService(), batch_tokens=4).encode(
        ["a", "b", "c"], progress_cb=lambda d, t: seen.append((d, t))
    )
    assert seen[-1] == (3, 3)


def test_no_texts_no_request() -> None:
    service = _FakeService()
    assert _embedder(service).encode([]) == []
    assert service.batches == []


def test_query_embeddings_are_cached() -> None:
    service = _FakeService()
    emb = _embedder(service, query_cache_size=2)
    assert emb.encode_query("what now")[0] == 2.0
    emb.encode_query("what now")
    assert len(service.batches) == 1


def test_count_tokens_is_local() -> None:
    service = _FakeService()
    assert _embedder(service).count_tokens("three words here") == 3
    assert service.batches == []


def test_a_vector_of_the_wrong_dimension_is_refused() -> None:
    with pytest.raises(EmbeddingServiceError, match="384-dimensional"):
        _embedder(_FakeService(dim=384)).encode(["a"])


def test_a_missing_vector_is_refused() -> None:
    def short(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.0] * _DIM]})

    with pytest.raises(EmbeddingServiceError, match="1 vectors for 2 texts"):
        _embedder(short).encode(["a", "b"])


def test_5xx_is_retried_then_succeeds() -> None:
    service = _FakeService(failures=[503, 502])
    assert _embedder(service, attempts=3).encode(["a"])[0][0] == 1.0


def test_5xx_gives_up_after_the_last_attempt() -> None:
    service = _FakeService(failures=[503, 503, 503])
    with pytest.raises(EmbeddingServiceError, match="HTTP 503"):
        _embedder(service, attempts=3).encode(["a"])


def test_4xx_is_not_retried() -> None:
    service = _FakeService(failures=[422, 503])
    with pytest.raises(EmbeddingServiceError, match="HTTP 422"):
        _embedder(service, attempts=3).encode(["a"])
    assert service.failures == [503]


def test_connection_errors_are_retried() -> None:
    calls = {"n": 0}
    service = _FakeService()

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return service.handle(request)

    assert _embedder(flaky).encode(["a b"])[0][0] == 2.0

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(EmbeddingServiceError, match="unreachable"):
        _embedder(down, attempts=2).encode(["a"])


def test_the_dimension_is_the_collections() -> None:
    from pipeline.vectorstore import _VECTOR_SIZE

    assert RemoteEmbedder.VECTOR_DIM == _VECTOR_SIZE


def test_make_embedder_uses_the_service_when_embeddings_url_is_set(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    import transformers

    from pipeline import indexer

    loaded: list[str] = []

    def from_pretrained(name: str) -> _FakeTokenizer:
        loaded.append(name)
        return _FakeTokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *a, **k: pytest.fail("no model weights when the service embeds"),
    )
    cfg = dataclasses.replace(config, embeddings_url="http://pacod-embeddings:8080")
    emb = indexer.make_embedder(cfg)
    assert isinstance(emb, RemoteEmbedder)
    assert emb.model_name == cfg.embedding_model
    assert loaded == [cfg.embedding_model]


def test_config_reads_the_service_urls(config: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.config import load_config

    monkeypatch.setenv("EMBEDDINGS_URL", "http://pacod-embeddings:8080/")
    monkeypatch.setenv("RERANKER_URL", "http://pacod-embeddings:8080")
    cfg = load_config()
    assert cfg.embeddings_url == "http://pacod-embeddings:8080"
    assert cfg.reranker_url == "http://pacod-embeddings:8080"
    monkeypatch.delenv("EMBEDDINGS_URL")
    monkeypatch.delenv("RERANKER_URL")
    assert load_config().embeddings_url == ""


# ---------------------------------------------------------------------------
# RemoteReranker
# ---------------------------------------------------------------------------


def _reranker(handler: Any) -> Any:
    from pipeline.reranker import RemoteReranker

    return RemoteReranker("http://pacod-embeddings:8080", transport=httpx.MockTransport(handler))


def test_remote_reranker_scores_through_the_service() -> None:
    seen: list[dict[str, Any]] = []

    def rerank(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"scores": [0.9, 0.1]})

    assert _reranker(rerank).score("q", ["a", "b"]) == [0.9, 0.1]
    assert seen == [{"query": "q", "documents": ["a", "b"]}]
    assert _reranker(rerank).score("q", []) == []


def test_remote_reranker_fails_loudly_so_search_keeps_the_fused_order() -> None:
    from pipeline.reranker import RerankUnavailableError

    with pytest.raises(httpx.HTTPStatusError):
        _reranker(lambda r: httpx.Response(503)).score("q", ["a"])
    with pytest.raises(RerankUnavailableError):
        _reranker(lambda r: httpx.Response(200, json={"scores": [0.5]})).score("q", ["a", "b"])


def test_the_reranker_factory_picks_the_service(config: Any) -> None:
    import dataclasses

    from pipeline.api import _reranker_factory
    from pipeline.reranker import RemoteReranker

    cfg = dataclasses.replace(config, reranker_url="http://pacod-embeddings:8080")
    assert isinstance(_reranker_factory(cfg)(), RemoteReranker)
