"""Hybrid retrieval: Qdrant vector search + keyword search fused via RRF (k=60)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.indexer import Embedder, FtsStore
    from pipeline.reranker import Reranker
    from pipeline.vectorstore import VectorStore

logger = logging.getLogger(__name__)


_RRF_K = 60
# as_of is pushed into both searches as "valid_until IS NULL OR IN (dates after as_of)"
# while there are at most this many distinct later dates; else post-filter (over-fetch).
_AS_OF_PUSHDOWN_MAX = 1000


@dataclass
class SearchResult:
    chunk_id: str
    source_id: str
    heading_path: list[str]
    text: str
    score: float
    byte_start: int
    byte_end: int
    valid_until: str | None
    # score is the ordering score: rerank_score when reranked, else rrf_score.
    rrf_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class SearchTrace:
    """Per-stage timings (ms) and the fused candidate order of one search.

    Filled only when passed to search(); the eval harness and latency probes read it.
    """

    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)
    reranked: bool = False


@dataclass(frozen=True)
class SearchOptions:
    """How a search runs.

    keyword_mode / keyword_weight: KEYWORD_QUERY_MODE, RRF_KEYWORD_WEIGHT.
    candidate_k: per-leg depth when reranking (RETRIEVAL_CANDIDATE_K).
    rerank_top_n: fused candidates the reranker scores (RERANK_TOP_N).
    min_score: drop reranked results below this relevance (only when reranked).
    """

    keyword_mode: str = "legacy"
    keyword_weight: float = 1.0
    candidate_k: int = 50
    rerank_top_n: int = 10
    min_score: float | None = None


def _rrf(
    vector_hits: list[tuple[str, int]],
    fts_hits: list[tuple[str, int]],
    k: int = _RRF_K,
    *,
    keyword_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over two ranked lists (keyword list weighted)."""
    scores: dict[str, float] = {}
    for chunk_id, rank in vector_hits:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for chunk_id, rank in fts_hits:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + keyword_weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search(
    query: str,
    embedder: Embedder,
    vector_store: VectorStore,
    fts_db: FtsStore,
    top_k: int = 5,
    *,
    include_historical: bool = False,
    as_of: str | None = None,
    project: str | None = None,
    as_of_pushdown: bool = True,
    options: SearchOptions | None = None,
    reranker: Reranker | None = None,
    trace: SearchTrace | None = None,
) -> list[SearchResult]:
    """Hybrid semantic + keyword search with RRF fusion, optionally reranked.

    By default only returns chunks whose valid_until is null (current versions).
    Set include_historical=True to include all superseded document chunks.
    Set as_of to an ISO date string (YYYY-MM-DD) to return chunks valid at that
    date: a chunk is valid at D iff valid_until is null OR valid_until > as_of.
    as_of is pushed down into both searches (as_of_pushdown, the default) so
    superseded chunks never crowd out valid ones; valid_until <= as_of is also
    post-filtered as a safety net. Pass as_of_pushdown=False when the keyword store
    does not cover the vector collection (its dates would be incomplete).
    Set project to a project slug to scope FTS5 results to that project (vector
    results are already scoped by the active Qdrant collection).
    With a *reranker*, each leg fetches options.candidate_k candidates and the first
    options.rerank_top_n fused ones are rescored; a failing reranker keeps the fused
    order. Without one the candidate depth stays max(top_k*3, 20): deeper lists
    change RRF's order even before any reranking.
    """
    import json

    opts = options or SearchOptions()
    # Older keyword stores (and test doubles) take no mode argument.
    mode_kw: dict[str, str] = (
        {"mode": opts.keyword_mode} if opts.keyword_mode != "legacy" else {}
    )
    use_historical = include_historical or (as_of is not None)
    fetch_k = max(top_k * 3, 20)  # over-fetch before fusion
    if reranker is not None:
        fetch_k = max(opts.candidate_k, top_k * 3)

    later_dates: list[str] | None = None
    if as_of is not None and as_of_pushdown:
        dates = fts_db.valid_until_values_after(as_of, limit=_AS_OF_PUSHDOWN_MAX + 1)
        if len(dates) <= _AS_OF_PUSHDOWN_MAX:
            later_dates = dates
    if as_of is not None and later_dates is None:
        # Post-filter only: superseded chunks are discarded after fusion, so fetch
        # deeper to keep them from starving the result list below top_k.
        fetch_k *= 3

    t0 = time.perf_counter()
    # encode_query (BgeM3Embedder) overtakes waiting ingest batches and is cached.
    encode_query = getattr(embedder, "encode_query", None)
    query_vector = encode_query(query) if encode_query is not None else embedder.encode([query])[0]
    t1 = time.perf_counter()
    if later_dates is not None:
        vector_hits = vector_store.query(query_vector, fetch_k, valid_until_in=later_dates)
    else:
        vector_hits = vector_store.query(query_vector, fetch_k, include_historical=use_historical)
    t2 = time.perf_counter()
    if later_dates is not None:
        fts_hits = fts_db.search(
            query, fetch_k, include_historical=True, project=project, as_of=as_of, **mode_kw
        )
    else:
        fts_hits = fts_db.search(
            query, fetch_k, include_historical=use_historical, project=project, **mode_kw
        )
    t3 = time.perf_counter()

    fused = _rrf(vector_hits, fts_hits, keyword_weight=opts.keyword_weight)
    # One round-trip for all candidates (was two queries per fused hit).
    rows = fts_db.get_chunks_bulk([chunk_id for chunk_id, _ in fused])

    # Hydrated candidates in fused order: top_k, or the reranker's head.
    limit = max(opts.rerank_top_n, top_k) if reranker is not None else top_k
    candidates: list[SearchResult] = []
    for chunk_id, score in fused:
        hit = rows.get(chunk_id)
        if hit is None:
            continue
        meta, et = hit

        valid_until: str | None = meta["valid_until"]

        # as_of post-filter: chunk valid at D iff valid_until is None OR valid_until > as_of.
        # (retrieved_at is deliberately ignored: it is the fetch date, not the date
        # the content took effect — an old version imported today is still valid at D.)
        if as_of is not None and valid_until is not None and valid_until <= as_of:
            continue

        candidates.append(
            SearchResult(
                chunk_id=chunk_id,
                source_id=meta["doc_id"],
                heading_path=json.loads(meta["heading_path"]),
                text=et,
                score=score,
                byte_start=meta["byte_start"],
                byte_end=meta["byte_end"],
                valid_until=valid_until,
                rrf_score=score,
            )
        )
        if len(candidates) == limit:
            break
    t5 = time.perf_counter()

    reranked = False
    if reranker is not None and candidates:
        try:
            # Versions of a document often repeat a chunk verbatim: score each text once.
            texts = list(dict.fromkeys(c.text for c in candidates))
            by_text = dict(zip(texts, reranker.score(query, texts), strict=True))
            for c in candidates:
                c.rerank_score = c.score = by_text[c.text]
            candidates.sort(key=lambda c: (c.score, c.rrf_score), reverse=True)
            reranked = True
        except Exception as exc:  # noqa: BLE001 — fail open: keep the fused order
            logger.warning("Reranking skipped, fused order kept: %s", type(exc).__name__)
    t6 = time.perf_counter()

    results = candidates
    if reranked and opts.min_score is not None:
        results = [c for c in results if c.score >= opts.min_score]
    results = results[:top_k]

    if trace is not None:
        trace.candidates = [chunk_id for chunk_id, _ in fused]
        trace.reranked = reranked
        trace.timings_ms.update(
            embed=(t1 - t0) * 1000,
            vector=(t2 - t1) * 1000,
            keyword=(t3 - t2) * 1000,
            fuse_hydrate=(t5 - t3) * 1000,
            total=(time.perf_counter() - t0) * 1000,
        )
        if reranker is not None:
            trace.timings_ms["rerank"] = (t6 - t5) * 1000
    return results
