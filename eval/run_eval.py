"""Retrieval evaluation harness.

Measures Recall@5 and MRR against golden_set.jsonl using the structured
(heading-aware) chunker versus a naive fixed-size baseline.

Usage:
    KB_ROOT=/tmp/eval_kb python -m eval.run_eval [--fixtures-dir tests/fixtures]

``--suite v2`` runs the multilingual evaluation instead (eval/suite_v2.py):
    python -m eval.run_eval --suite v2 --embedder bge-m3 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

GOLDEN = Path(__file__).parent / "golden_set.jsonl"
FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Mock embedder (avoids downloading BGE-M3 during eval harness)
# ---------------------------------------------------------------------------


class _MockEmbedder:
    """Deterministic random embedder for eval — not semantically meaningful."""

    VECTOR_DIM: int = 1024

    def __init__(self, seed: int = 42) -> None:
        import random

        self._rng = random.Random(seed)

    def encode(
        self, texts: list[str], *, progress_cb: Callable[[int, int], None] | None = None
    ) -> list[list[float]]:
        import random

        vecs = [
            [random.Random(hash(t) ^ i).gauss(0, 1) for i in range(self.VECTOR_DIM)]
            for t in texts
        ]
        if progress_cb is not None:
            progress_cb(len(texts), len(texts))
        return vecs

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Naive fixed-size chunker baseline
# ---------------------------------------------------------------------------


def _naive_chunks(body: str, doc_id: str, chunk_size: int = 1000) -> list[str]:
    """Split body into fixed-size character chunks — baseline comparison."""
    body_bytes = body.encode("utf-8")
    pieces = []
    pos = 0
    while pos < len(body_bytes):
        pieces.append(body_bytes[pos : pos + chunk_size].decode("utf-8", errors="replace"))
        pos += chunk_size
    return pieces


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _recall_at_k(hits: list[str], expected: str, k: int = 5) -> float:
    return 1.0 if expected in hits[:k] else 0.0


def _reciprocal_rank(hits: list[str], expected: str) -> float:
    for i, h in enumerate(hits, start=1):
        if h == expected:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


def run(fixtures_dir: Path, kb_root: Path) -> dict[str, float]:
    """Run structured vs naive eval. Returns metrics dict."""

    from qdrant_client import QdrantClient

    from pipeline.acquisition import ingest_document
    from pipeline.config import load_config
    from pipeline.indexer import FtsDb, index_document
    from pipeline.normalize import strip_front_matter
    from pipeline.retrieval import search
    from pipeline.vectorstore import QdrantVectorStore

    os.environ["KB_ROOT"] = str(kb_root)
    os.environ["GIT_AUTOCOMMIT"] = "false"
    os.environ["QDRANT_MODE"] = "embedded"
    config = load_config()

    embedder = _MockEmbedder()

    # In-memory Qdrant for eval
    qclient = QdrantClient(":memory:")
    vstore = QdrantVectorStore(qclient, config.qdrant_collection)
    fts = FtsDb(kb_root / "index" / "fts5.db")

    # Ingest and index all fixtures
    fixture_files = list(fixtures_dir.glob("*"))
    doc_source_map: dict[str, str] = {}  # doc_id → source_file

    for fpath in fixture_files:
        try:
            doc_id, folder = ingest_document(str(fpath), config)
            index_document(folder, config, embedder, vstore, fts)
            doc_source_map[doc_id] = fpath.name
        except Exception as exc:
            print(f"  WARN: skipped {fpath.name}: {exc}", file=sys.stderr)

    golden = [json.loads(ln) for ln in GOLDEN.read_text().splitlines() if ln.strip()]

    # --- Structured chunker metrics ---
    recall_vals: list[float] = []
    mrr_vals: list[float] = []

    for entry in golden:
        question: str = entry["question"]
        expected_sf: str = entry["expected_source_file"]

        results = search(question, embedder, vstore, fts, top_k=5)
        hit_source_files = [doc_source_map.get(r.source_id, "") for r in results]

        recall_vals.append(_recall_at_k(hit_source_files, expected_sf))
        mrr_vals.append(_reciprocal_rank(hit_source_files, expected_sf))

    structured_recall = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0
    structured_mrr = sum(mrr_vals) / len(mrr_vals) if mrr_vals else 0.0

    # --- Naive fixed-size baseline metrics ---
    naive_recall_vals: list[float] = []
    naive_mrr_vals: list[float] = []

    naive_fts = FtsDb(kb_root / "index" / "naive_fts.db")
    naive_vstore = QdrantVectorStore(QdrantClient(":memory:"), "naive_chunks")

    docs_dir = config.documents_dir
    if docs_dir.exists():
        for folder in sorted(docs_dir.iterdir()):
            manifest_path = folder / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            doc_id = manifest["doc_id"]
            doc_md = (folder / "document.md").read_text(encoding="utf-8")
            _, body = strip_front_matter(doc_md)

            # Build naive chunk objects compatible with our stores
            from pipeline.chunker import Chunk

            body_bytes = body.encode("utf-8")
            chunk_size_bytes = 1000
            pos = 0
            seq = 0
            naive_chunks_list: list[Chunk] = []
            while pos < len(body_bytes):
                end = min(pos + chunk_size_bytes, len(body_bytes))
                text = body_bytes[pos:end].decode("utf-8", errors="replace")
                cid = f"{doc_id[:8]}#naive#{seq}"
                naive_chunks_list.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        heading_path=[],
                        heading_level=0,
                        parent_chunk=None,
                        sequence=seq,
                        byte_start=pos,
                        byte_end=end,
                        embedded_text=text,
                        valid_until=None,
                        extraction_method=manifest.get("extraction_method", "deterministic"),
                        confidence=1.0,
                    )
                )
                pos = end
                seq += 1

            if naive_chunks_list:
                vectors = embedder.encode([c.embedded_text for c in naive_chunks_list])
                naive_vstore.upsert(naive_chunks_list, vectors)
                naive_fts.upsert_batch(naive_chunks_list)

    for entry in golden:
        question = entry["question"]
        expected_sf = entry["expected_source_file"]

        results = search(question, embedder, naive_vstore, naive_fts, top_k=5)
        # Map chunk source_id → source_file via the original doc_source_map
        hit_source_files = [doc_source_map.get(r.source_id, "") for r in results]

        naive_recall_vals.append(_recall_at_k(hit_source_files, expected_sf))
        naive_mrr_vals.append(_reciprocal_rank(hit_source_files, expected_sf))

    naive_recall = sum(naive_recall_vals) / len(naive_recall_vals) if naive_recall_vals else 0.0
    naive_mrr = sum(naive_mrr_vals) / len(naive_mrr_vals) if naive_mrr_vals else 0.0

    fts.close()
    naive_fts.close()

    return {
        "structured_recall_5": structured_recall,
        "structured_mrr": structured_mrr,
        "naive_recall_5": naive_recall,
        "naive_mrr": naive_mrr,
        "n_questions": len(golden),
    }


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--suite", choices=["legacy", "v2"], default="legacy")
    known, rest = pre.parse_known_args()
    if known.suite == "v2":
        from eval.suite_v2 import main as v2_main

        sys.exit(v2_main(rest))

    parser = argparse.ArgumentParser(description="Run retrieval eval harness")
    parser.add_argument(
        "--suite",
        choices=["legacy", "v2"],
        default="legacy",
        help="legacy: fixtures smoke test (default); v2: multilingual suite (see --suite v2 -h)",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=FIXTURES,
        help="Directory containing fixture files to ingest",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Exit 1 if structured Recall@5 is below this threshold (regression gate)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="kb_eval_") as tmp:
        kb_root = Path(tmp)
        (kb_root / "documents").mkdir()
        (kb_root / "index").mkdir()
        (kb_root / "review").mkdir()

        print(f"Running eval with fixtures from: {args.fixtures_dir}")
        metrics = run(args.fixtures_dir, kb_root)

    print("\n=== Retrieval Eval Results ===")
    print(f"Questions : {metrics['n_questions']}")
    print(
        f"Structured chunker  — Recall@5: {metrics['structured_recall_5']:.3f}  "
        f"MRR: {metrics['structured_mrr']:.3f}"
    )
    print(
        f"Naive (fixed-size)  — Recall@5: {metrics['naive_recall_5']:.3f}  "
        f"MRR: {metrics['naive_mrr']:.3f}"
    )

    if args.min_recall is not None:
        actual = metrics["structured_recall_5"]
        if actual < args.min_recall:
            print(
                f"\nRECALL GATE FAILED: {actual:.3f} < {args.min_recall:.3f}",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
