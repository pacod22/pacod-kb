"""The search result partials render both score kinds (reranker relevance, RRF)."""

from __future__ import annotations

from typing import Any

import pytest


def _hit(score: float, rrf: float, rerank: float | None, cid: str = "abc#s#0") -> dict[str, Any]:
    return {
        "chunk_id": cid, "source_id": "d" * 64, "heading_path": ["Act", "Article 17"],
        "text": "[Act > Article 17]\nThe data subject shall have the right...", "score": score,
        "rrf_score": rrf, "rerank_score": rerank, "byte_start": 0, "byte_end": 10,
        "valid_until": None,
    }


@pytest.fixture()
def env() -> Any:
    from web.templates_config import templates

    return templates.env


def test_qdrant_results_show_reranker_relevance(env: Any) -> None:
    html = env.get_template("qdrant/_search_results.html").render(
        query="erasure", results=[_hit(0.97, 0.03, 0.97), _hit(0.20, 0.02, 0.20, "x#s#1")]
    )
    assert "ranked by reranker relevance" in html
    assert "97%" in html and "20%" in html  # absolute, not relative to the best
    assert "reranker relevance: 0.970" in html


def test_qdrant_results_show_rrf_relative(env: Any) -> None:
    html = env.get_template("qdrant/_search_results.html").render(
        query="erasure", results=[_hit(0.032, 0.032, None), _hit(0.016, 0.016, None, "x#s#1")]
    )
    assert "ranked by rank fusion (RRF)" in html
    assert "100%" in html and "50%" in html


def test_qdrant_results_tolerate_an_older_pipeline(env: Any) -> None:
    hit = _hit(0.032, 0.032, None)
    del hit["rerank_score"], hit["rrf_score"]
    html = env.get_template("qdrant/_search_results.html").render(query="q", results=[hit])
    assert "ranked by rank fusion (RRF)" in html


def test_query_results_label_the_score(env: Any) -> None:
    tpl = env.get_template("pipeline/_query_results.html")
    assert "relevance 0.97" in tpl.render(results=[_hit(0.97, 0.03, 0.97)], error=None)
    assert "RRF 0.0320" in tpl.render(results=[_hit(0.032, 0.032, None)], error=None)
