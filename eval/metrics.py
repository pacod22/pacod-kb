"""Retrieval metrics for the v2 evaluation suite (pure functions, no pipeline imports).

Relevance is judged by section, not chunk id: a retrieved chunk is relevant when it
comes from a labelled document and its heading path contains the labelled section
(word-bounded, so "Article 1" never matches "Article 17"). Labels survive chunker
changes that rename chunk ids.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RelevantUnit:
    doc: str
    section: str
    group: str | None = None

    @property
    def key(self) -> str:
        """Units sharing a group (parallel translations) count as one."""
        return self.group or f"{self.doc}::{self.section}"


@dataclass(frozen=True)
class RetrievedChunk:
    doc: str
    heading_path: tuple[str, ...]


def section_matches(heading_path: Sequence[str], section: str) -> bool:
    """True when *section* occurs word-bounded in the joined heading path."""
    joined = " > ".join(heading_path).casefold()
    pattern = r"(?<!\w)" + re.escape(section.casefold()) + r"(?!\w)"
    return re.search(pattern, joined) is not None


def judge(
    retrieved: Sequence[RetrievedChunk], relevant: Sequence[RelevantUnit]
) -> list[str | None]:
    """Per rank: the relevance key first satisfied there, else None.

    A key counts only at its first occurrence, so two chunks of the same article
    (or of two translations in one group) are one hit, not two.
    """
    seen: set[str] = set()
    out: list[str | None] = []
    for chunk in retrieved:
        hit: str | None = None
        for unit in relevant:
            if unit.key in seen or unit.doc != chunk.doc:
                continue
            if section_matches(chunk.heading_path, unit.section):
                hit = unit.key
                break
        if hit is not None:
            seen.add(hit)
        out.append(hit)
    return out


def n_relevant(relevant: Sequence[RelevantUnit]) -> int:
    return len({u.key for u in relevant})


def recall_at_k(judged: Sequence[str | None], n_rel: int, k: int) -> float:
    if n_rel == 0:
        return 0.0
    return len({j for j in judged[:k] if j is not None}) / n_rel


def mrr_at_k(judged: Sequence[str | None], k: int) -> float:
    for rank, j in enumerate(judged[:k], start=1):
        if j is not None:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(judged: Sequence[str | None], n_rel: int, k: int) -> float:
    if n_rel == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, j in enumerate(judged[:k], 1) if j is not None)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_rel, k) + 1))
    return dcg / idcg


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * p / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass(frozen=True)
class PairedStats:
    mean_delta: float
    wins: int
    losses: int
    ties: int
    ci_low: float
    ci_high: float


def paired_deltas(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000, seed: int = 0, eps: float = 1e-9
) -> PairedStats:
    """Per-query deltas b - a with a bootstrap 95 % CI of the mean delta."""
    if len(a) != len(b):
        raise ValueError("paired samples differ in length")
    deltas = [y - x for x, y in zip(a, b, strict=True)]
    if not deltas:
        return PairedStats(0.0, 0, 0, 0, 0.0, 0.0)
    wins = sum(1 for d in deltas if d > eps)
    losses = sum(1 for d in deltas if d < -eps)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(n_boot))
    return PairedStats(
        mean_delta=sum(deltas) / n,
        wins=wins,
        losses=losses,
        ties=n - wins - losses,
        ci_low=means[int(0.025 * (n_boot - 1))],
        ci_high=means[int(0.975 * (n_boot - 1))],
    )
