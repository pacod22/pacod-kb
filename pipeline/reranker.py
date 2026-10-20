"""Second-stage reranking: a cross-encoder scores the fused candidates (CPU).

Hybrid retrieval finds the right section among its candidates but orders them by
rank fusion only. A cross-encoder reads query and passage together and returns a
relevance probability in [0, 1], which also makes a minimum score meaningful.
Reranking is optional (RERANKER_ENABLED, per request ``rerank``), loads in the
background at startup, shares the inference gate with the embedder, and fails open:
on any error or timeout the fused order is kept. With RERANKER_URL the shared embeddings
service scores instead (``RemoteReranker``) and no model is loaded here.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from pipeline.inference import GATE, InferenceGate

logger = logging.getLogger(__name__)


class RerankUnavailableError(Exception):
    """The reranker could not run in time (the inference gate stayed busy)."""


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to *query*, in [0, 1]."""
        ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder with sigmoid scores, behind the inference gate."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 384,
        batch_size: int = 16,
        device: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        code_revision: str | None = None,
        gate: InferenceGate | None = None,
        gate_timeout_s: float = 1.0,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        # Remote code (e.g. gte) comes from another repository: pin it like the weights.
        extra: dict[str, Any] = {"code_revision": code_revision} if code_revision else {}
        self.model_name = model_name
        self._model: Any = CrossEncoder(
            model_name,
            device=device,
            revision=revision or None,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
            activation_fn=torch.nn.Sigmoid(),
            model_kwargs=extra or None,
            config_kwargs=extra or None,
        )
        self._batch_size = max(1, batch_size)
        self._gate = gate or GATE
        self._gate_timeout_s = gate_timeout_s
        self.score("warm-up", ["warm-up"])  # first call allocates; keep it off a query

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        with self._gate.interactive(self._gate_timeout_s) as acquired:
            if not acquired:
                raise RerankUnavailableError("inference gate busy")
            raw: Any = self._model.predict(
                [(query, p) for p in passages],
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        return [float(x) for x in raw]


class RemoteReranker:
    """The shared embeddings service's cross-encoder (pacod-embeddings, ``POST /rerank``).

    pacod-ap's service scores with bge-reranker-v2-m3 and returns sigmoid scores in [0, 1], the
    same scale as ``CrossEncoderReranker``; nothing is loaded here and the inference gate is not
    involved. One attempt within *timeout_s*: a slow or failed call raises, and the search keeps
    its fused order (fail open).
    """

    model_name = "BAAI/bge-reranker-v2-m3"

    def __init__(self, url: str, *, timeout_s: float = 5.0, transport: Any = None) -> None:
        import httpx

        self._endpoint = f"{url.rstrip('/')}/rerank"
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 2.0)),
            transport=transport,
            follow_redirects=False,
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        resp = self._client.post(self._endpoint, json={"query": query, "documents": list(passages)})
        resp.raise_for_status()
        scores = resp.json()["scores"]
        if not isinstance(scores, list) or len(scores) != len(passages):
            raise RerankUnavailableError("rerank service returned a score list of the wrong length")
        return [float(x) for x in scores]

    def close(self) -> None:
        self._client.close()


RerankerState = Literal["disabled", "loading", "ready", "failed"]


class RerankerHolder:
    """Loads a reranker in a daemon thread so startup and /health never wait for it."""

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name
        self.state: RerankerState = "disabled"
        self.error: str | None = None
        self._reranker: Reranker | None = None
        self._lock = threading.Lock()

    def start(self, factory: Callable[[], Reranker]) -> None:
        with self._lock:
            if self.state in ("loading", "ready"):
                return
            self.state = "loading"
            self.error = None

        def load() -> None:
            try:
                reranker = factory()
            except Exception as exc:  # noqa: BLE001 — reported by /health, search fails open
                logger.error("Reranker %s failed to load: %s", self.model_name, exc)
                with self._lock:
                    self.state = "failed"
                    self.error = f"{type(exc).__name__}: {exc}"
                return
            with self._lock:
                self._reranker = reranker
                self.state = "ready"
            logger.info("Reranker ready: %s", self.model_name)

        threading.Thread(target=load, name="reranker-load", daemon=True).start()

    def get(self) -> Reranker | None:
        return self._reranker if self.state == "ready" else None

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"state": self.state, "model": self.model_name or None}
        if self.error:
            out["error"] = self.error
        return out
