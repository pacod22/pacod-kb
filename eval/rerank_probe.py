"""Cross-encoder latency probe on CPU over real corpus chunks.

    uv run python -m eval.rerank_probe --device cpu --threads 4

For each model and every (N candidates × max_length × batch size) cell it scores N
real chunk texts against golden queries and reports p50/p95 latency and resident
memory. Quality is measured separately by the v2 suite (--rerank). Run it on a
production-class CPU before choosing RERANKER_MODEL / RERANK_TOP_N.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.metrics import percentile
from eval.suite_v2 import CACHE_ROOT, RESULTS_DIR, _cpu_model, load_golden

DEFAULT_MODELS = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "Alibaba-NLP/gte-multilingual-reranker-base",
    "BAAI/bge-reranker-v2-m3",
)
REMOTE_CODE = {"Alibaba-NLP/gte-multilingual-reranker-base"}


def _chunk_texts(cache: Path | None) -> list[str]:
    if cache is None:
        caches = sorted(CACHE_ROOT.glob("*/kb/index/fts5.db"), key=lambda p: p.stat().st_mtime)
        if not caches:
            sys.exit("no eval index cache found: run `python -m eval.run_eval --suite v2` first")
        db = caches[-1]
    else:
        db = cache / "kb" / "index" / "fts5.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT embedded_text FROM chunks_meta WHERE project = 'eval'"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows if r[0]]


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="cross-encoder CPU latency probe")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--n", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--max-length", nargs="+", type=int, default=[256, 384, 512])
    p.add_argument("--batch", nargs="+", type=int, default=[8, 16])
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--queries", type=int, default=12, help="golden queries per cell")
    p.add_argument("--index-cache", type=Path, default=None)
    p.add_argument("--trust-remote-code", action="store_true", help="allow gte's remote code")
    p.add_argument("--output", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)

    import torch
    from sentence_transformers import CrossEncoder

    if args.threads:
        torch.set_num_threads(args.threads)
    texts = _chunk_texts(args.index_cache)
    queries = [q.query for q in load_golden()][: args.queries]
    rng = random.Random(0)
    cells: list[dict[str, Any]] = []

    for model_name in args.models:
        remote = model_name in REMOTE_CODE
        if remote and not args.trust_remote_code:
            print(f"skip {model_name}: needs --trust-remote-code", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        model = CrossEncoder(
            model_name, device=args.device, trust_remote_code=remote,
            activation_fn=torch.nn.Sigmoid(),
        )
        load_s = time.perf_counter() - t0
        model.predict([(queries[0], texts[0])])  # warm-up
        for max_len in args.max_length:
            model.max_seq_length = max_len
            for n in args.n:
                for batch in args.batch:
                    samples: list[float] = []
                    for q in queries:
                        pairs = [(q, t) for t in rng.sample(texts, n)]
                        t1 = time.perf_counter()
                        model.predict(pairs, batch_size=batch, show_progress_bar=False)
                        samples.append((time.perf_counter() - t1) * 1000)
                    cell = {
                        "model": model_name, "n": n, "max_length": max_len, "batch": batch,
                        "p50_ms": round(percentile(samples, 50), 1),
                        "p95_ms": round(percentile(samples, 95), 1),
                        "rss_mb": round(_rss_mb()), "load_s": round(load_s, 1),
                    }
                    cells.append(cell)
                    print(
                        f"{model_name:46} N={n:2} len={max_len} b={batch:2} "
                        f"p50={cell['p50_ms']:8.1f}ms p95={cell['p95_ms']:8.1f}ms "
                        f"rss={cell['rss_mb']}MB",
                        file=sys.stderr,
                    )
        del model

    result = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu": _cpu_model(),
        "device": args.device,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cells": cells,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = args.output / f"{stamp}_rerank-probe-{args.device}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"results: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
