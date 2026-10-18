"""Ingest embedding throughput: fixed 32-text slices vs length-sorted micro-batches.

    uv run python -m eval.embed_bench --n 400

Embeds the same real corpus chunks both ways, reports chunks/s, and checks the vectors
agree (minimum cosine). Run after `python -m eval.run_eval --suite v2` built an index.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.rerank_probe import _chunk_texts
from eval.suite_v2 import RESULTS_DIR, _cpu_model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="embedding throughput: old vs new batching")
    p.add_argument("--n", type=int, default=400, help="chunks to embed")
    p.add_argument("--batch-tokens", nargs="+", type=int, default=[1024, 2048, 4096])
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--index-cache", type=Path, default=None)
    p.add_argument("--output", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)

    import numpy as np

    from pipeline.indexer import BgeM3Embedder
    from pipeline.inference import configure_threads

    threads = configure_threads(args.threads)
    texts = _chunk_texts(args.index_cache)
    random.Random(0).shuffle(texts)
    texts = texts[: args.n]
    emb = BgeM3Embedder(device=args.device, query_cache_size=0)
    emb._model.encode(texts[:2], normalize_embeddings=True)  # warm-up

    t0 = time.perf_counter()
    old: list[Any] = []
    for i in range(0, len(texts), 32):  # the previous BgeM3Embedder.encode
        old.extend(emb._model.encode(texts[i : i + 32], normalize_embeddings=True))
    old_s = time.perf_counter() - t0
    old_m = np.array(old)
    rows: dict[str, Any] = {
        "fixed_32": {"seconds": round(old_s, 2), "chunks_per_s": round(len(texts) / old_s, 2)}
    }
    print(f"fixed 32 slices      {len(texts) / old_s:7.2f} chunks/s", file=sys.stderr)

    for budget in args.batch_tokens:
        emb._batch_tokens = budget
        t0 = time.perf_counter()
        new_m = np.array(emb.encode(texts))
        new_s = time.perf_counter() - t0
        cos = float(np.min(np.sum(old_m * new_m, axis=1)))
        rows[f"sorted_{budget}"] = {
            "seconds": round(new_s, 2),
            "chunks_per_s": round(len(texts) / new_s, 2),
            "min_cosine_vs_fixed": round(cos, 7),
            "batches": len(emb._plan(texts)),
        }
        print(
            f"sorted, {budget:5} tokens {len(texts) / new_s:7.2f} chunks/s  "
            f"min cos {cos:.7f}  batches {rows[f'sorted_{budget}']['batches']}",
            file=sys.stderr,
        )

    result = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu": _cpu_model(), "device": args.device, "torch_threads": threads, "n": len(texts),
        "runs": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}_embed-bench.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"results: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
