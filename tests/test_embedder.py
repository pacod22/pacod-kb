"""BgeM3Embedder scheduling: token-budgeted micro-batches, query priority, LRU, threads."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from pipeline.inference import InferenceGate, available_cpus


class _FakeTokenizer:
    def __call__(self, texts: list[str], **_: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [list(range(len(t.split()) + 2)) for t in texts]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


class _FakeModel:
    """Vector = [word count, position marker]: lets tests check order and batching."""

    def __init__(self, name: str, device: str | None = None) -> None:
        self.tokenizer = _FakeTokenizer()
        self.max_seq_length = 8192
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **_: Any) -> Any:
        self.calls.append(list(texts))
        return np.array([[float(len(t.split())), float(ord(t[0]))] for t in texts])


@pytest.fixture()
def make(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeModel)
    from pipeline.indexer import BgeM3Embedder

    def build(**kwargs: Any) -> Any:
        return BgeM3Embedder("fake", gate=InferenceGate(), **kwargs)

    return build


def test_encode_restores_order_and_respects_the_token_budget(make: Any) -> None:
    emb = make(batch_tokens=20)
    texts = [f"{chr(97 + i)} " + "w " * n for i, n in enumerate([9, 1, 15, 3, 3, 30, 2])]
    vectors = emb.encode(texts)
    assert [v[0] for v in vectors] == [float(len(t.split())) for t in texts]
    assert [v[1] for v in vectors] == [float(ord(t[0])) for t in texts]
    for call in emb._model.calls:
        padded = max(len(t.split()) + 2 for t in call) * len(call)
        assert len(call) == 1 or padded <= 20
    # Length-sorted: no batch mixes the 30-word text with short ones.
    assert any(len(call) == 1 and len(call[0].split()) == 31 for call in emb._model.calls)


def test_progress_is_throttled_and_ends_at_total(make: Any) -> None:
    emb = make(batch_tokens=8)
    seen: list[tuple[int, int]] = []
    emb.encode([f"t{i} x" for i in range(100)], progress_cb=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (100, 100)
    assert len(seen) <= 5


def test_count_tokens_does_not_wait_for_model_calls(make: Any) -> None:
    emb = make()
    with emb._gate.batch():  # a model call in progress
        start = time.monotonic()
        assert emb.count_tokens("one two three") == 3
        assert time.monotonic() - start < 1.0


def test_query_embeddings_are_cached(make: Any) -> None:
    emb = make(query_cache_size=2)
    first = emb.encode_query("alpha beta")
    assert emb.encode_query("alpha beta") == first
    assert len(emb._model.calls) == 1
    emb.encode_query("gamma")
    emb.encode_query("delta")  # evicts "alpha beta"
    emb.encode_query("alpha beta")
    assert len(emb._model.calls) == 4


def test_cache_size_zero_disables_the_cache(make: Any) -> None:
    emb = make(query_cache_size=0)
    emb.encode_query("same")
    emb.encode_query("same")
    assert len(emb._model.calls) == 2


def test_interactive_callers_overtake_waiting_batches() -> None:
    gate = InferenceGate()
    order: list[str] = []
    holding = threading.Event()
    release = threading.Event()

    def first_batch() -> None:
        with gate.batch():
            holding.set()
            release.wait(5)

    def later(kind: str) -> None:
        ctx = gate.batch() if kind == "batch" else gate.interactive()
        with ctx:
            order.append(kind)

    t0 = threading.Thread(target=first_batch)
    t0.start()
    holding.wait(5)
    waiting_batch = threading.Thread(target=later, args=("batch",))
    waiting_batch.start()
    time.sleep(0.05)
    query = threading.Thread(target=later, args=("interactive",))
    query.start()
    time.sleep(0.05)
    release.set()
    for t in (t0, waiting_batch, query):
        t.join(5)
    assert order == ["interactive", "batch"]


def test_interactive_timeout_yields_false_and_frees_batches() -> None:
    gate = InferenceGate()
    with gate.batch():
        with gate.interactive(timeout_s=0.05) as acquired:
            assert acquired is False
    with gate.batch():  # nothing left waiting or held
        pass


def test_available_cpus_is_positive() -> None:
    cpus = available_cpus()
    assert cpus is None or cpus >= 1
