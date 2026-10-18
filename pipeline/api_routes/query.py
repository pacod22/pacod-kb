"""Hybrid search endpoint."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)
from pydantic import BaseModel, Field, field_validator

import pipeline.api as core
from pipeline.api import (
    AuthDep,
    _ledger_factory_from_app,
    _runtime,
)
from pipeline.config import Config
from pipeline.indexer import BgeM3Embedder, make_fts_db
from pipeline.retrieval import SearchOptions, SearchTrace
from pipeline.retrieval import search as _retrieval_search
from pipeline.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

# ---------------------------------------------------------------------------
# Query (synchronous — embedder is thread-safe)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    include_historical: bool = False
    # ISO date: chunks valid on that day (valid_until null or later).
    as_of: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    # Wrap chunk text as untrusted data for downstream LLMs. Human-facing
    # consumers (the web search UI) pass False to get clean, unfenced text.
    wrap_untrusted: bool = True
    # Search a specific Qdrant collection instead of the active project's.
    collection: str | None = None
    # Rerank the fused candidates (default: the reranker_enabled setting). Only takes
    # effect when the reranker is loaded; a request never loads it.
    rerank: bool | None = None
    # Drop results whose reranker relevance (0-1) is lower; ignored when not reranked.
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("as_of")
    @classmethod
    def _real_date(cls, v: str | None) -> str | None:
        if v is not None:
            date.fromisoformat(v)  # ValueError -> 422 for e.g. 2024-02-30
        return v


_NO_PROJECT = "__no_project__"


def _query_target(
    request: Request, collection: str | None
) -> tuple[QdrantVectorStore, str | None]:
    """Vector store + keyword (FTS) project scope for a query.

    The keyword scope is the project that owns the searched collection, so vectors
    and keyword hits always come from the same project. A collection owned by no
    project has no keyword rows: FTS is scoped to a slug that matches nothing, so
    results are vector-only rather than cross-project.
    """
    active: QdrantVectorStore = request.app.state.vector_store
    name = collection or active.collection_name
    if name == active.collection_name:
        store = active
    else:
        if not active.client.collection_exists(name):
            raise HTTPException(status_code=404, detail=f"Collection '{name}' does not exist")
        store = core.QdrantVectorStore(active.client, name, ensure=False)
    if getattr(request.app.state, "project_registry", None) is None:
        return store, _runtime.get("active_project") or None
    return store, core._collection_owner(request.app, name) or _NO_PROJECT


@router.post("/query")
def query(req: QueryRequest, request: Request, _auth: AuthDep) -> dict[str, Any]:
    config: Config = request.app.state.config
    embedder: BgeM3Embedder = request.app.state.embedder
    vector_store, project = _query_target(request, req.collection)
    use_rerank = req.rerank if req.rerank is not None else bool(_runtime.get("reranker_enabled"))
    holder = getattr(request.app.state, "reranker", None)
    # Only a reranker that is already loaded: a request never loads one (that is the
    # operator's call: RERANKER_ENABLED or POST /settings; the model costs GBs of RAM).
    reranker = holder.get() if use_rerank and holder is not None else None
    trace = SearchTrace()
    fts_db = make_fts_db(config)
    try:
        results = _retrieval_search(
            req.query,
            embedder,
            vector_store,
            fts_db,
            top_k=req.top_k,
            include_historical=req.include_historical,
            project=project,
            as_of=req.as_of,
            # Pushdown reads later valid_until dates from the keyword store, which
            # holds no rows for an unowned collection.
            as_of_pushdown=project != _NO_PROJECT,
            options=SearchOptions(
                keyword_mode=config.keyword_query_mode,
                keyword_weight=config.rrf_keyword_weight,
                candidate_k=config.retrieval_candidate_k,
                rerank_top_n=config.rerank_top_n,
                min_score=req.min_score,
            ),
            reranker=reranker,
            trace=trace,
        )
    finally:
        fts_db.close()

    # Defense in depth: delimit retrieved chunk text as untrusted data so a
    # downstream LLM treats it as data, not instructions (OWASP structured-prompt).
    # Gated globally by config and per-request (the human web UI opts out).
    wrap = config.content_security_wrap_output and req.wrap_untrusted
    preamble = ""
    if wrap:
        from pipeline.content_security import untrusted_preamble, wrap_untrusted

        # One prompt lookup per response, not one per result.
        preamble = untrusted_preamble(_ledger_factory_from_app(request))

    return {
        "results": [
            {
                "chunk_id": r.chunk_id,
                "source_id": r.source_id,
                "heading_path": r.heading_path,
                "text": wrap_untrusted(r.text, preamble=preamble) if wrap else r.text,
                # The ordering score: rerank_score when reranked, else rrf_score.
                "score": r.score,
                "rrf_score": r.rrf_score,
                "rerank_score": r.rerank_score,
                "byte_start": r.byte_start,
                "byte_end": r.byte_end,
                "valid_until": r.valid_until,
            }
            for r in results
        ],
        "reranked": trace.reranked,
        "score_kind": "rerank" if trace.reranked else "rrf",
    }
