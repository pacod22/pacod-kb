"""Tests for eval/metrics.py — section-based relevance and ranking metrics."""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    RelevantUnit,
    RetrievedChunk,
    judge,
    mrr_at_k,
    n_relevant,
    ndcg_at_k,
    paired_deltas,
    percentile,
    recall_at_k,
    section_matches,
)


@pytest.mark.parametrize(
    ("path", "section", "expected"),
    [
        (["GDPR", "CHAPTER III", "Article 17 — Right to erasure"], "Article 17", True),
        (["GDPR", "CHAPTER III", "Article 17 — Right to erasure"], "Article 1", False),
        (["GDPR", "CHAPTER I", "Article 1 — Subject-matter"], "Article 1", True),
        (["Конституция", "Глава втора", "Чл. 17"], "Чл. 1", False),
        (["Конституция", "Глава втора", "Чл. 17"], "Чл. 17", True),
        (["Ustav", "III.", "Članak 37."], "Članak 37.", True),
        (["Ustava", "II.", "138. člen (x)"], "38. člen", False),
        (["Photosynthesis", "Light-independent reactions", "Calvin cycle"], "calvin CYCLE", True),
    ],
)
def test_section_matches_word_bounded(path: list[str], section: str, expected: bool) -> None:
    assert section_matches(path, section) is expected


def _chunk(doc: str, *path: str) -> RetrievedChunk:
    return RetrievedChunk(doc=doc, heading_path=tuple(path))


def test_judge_counts_a_group_once() -> None:
    rel = [
        RelevantUnit("gdpr_en.md", "Article 17", "g17"),
        RelevantUnit("gdpr_de.md", "Artikel 17", "g17"),
    ]
    retrieved = [
        _chunk("gdpr_en.md", "GDPR", "Article 16 — Rectification"),
        _chunk("gdpr_de.md", "DSGVO", "Artikel 17 — Recht auf Löschung"),
        _chunk("gdpr_en.md", "GDPR", "Article 17 — Right to erasure"),
        _chunk("gdpr_de.md", "DSGVO", "Artikel 17 — Recht auf Löschung"),
    ]
    assert judge(retrieved, rel) == [None, "g17", None, None]
    assert n_relevant(rel) == 1


def test_judge_requires_matching_doc() -> None:
    rel = [RelevantUnit("constitution_bg.md", "Чл. 32")]
    retrieved = [_chunk("other.md", "X", "Чл. 32")]
    assert judge(retrieved, rel) == [None]


def test_rank_metrics_known_values() -> None:
    judged: list[str | None] = [None, "a", None, "b"]
    assert recall_at_k(judged, 2, 1) == 0.0
    assert recall_at_k(judged, 2, 4) == 1.0
    assert mrr_at_k(judged, 10) == 0.5
    assert mrr_at_k(judged, 1) == 0.0
    expected_ndcg = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(judged, 2, 10) == pytest.approx(expected_ndcg)
    assert ndcg_at_k(["a"], 1, 10) == 1.0
    assert ndcg_at_k([], 0, 10) == 0.0


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 100) == 5.0
    assert math.isnan(percentile([], 50))


def test_paired_deltas_counts_and_ci() -> None:
    stats = paired_deltas([0.0, 1.0, 0.5, 0.5], [1.0, 1.0, 0.0, 0.5], n_boot=500)
    assert (stats.wins, stats.losses, stats.ties) == (1, 1, 2)
    assert stats.mean_delta == pytest.approx(0.125)
    assert stats.ci_low <= stats.mean_delta <= stats.ci_high
    with pytest.raises(ValueError):
        paired_deltas([1.0], [1.0, 2.0])
