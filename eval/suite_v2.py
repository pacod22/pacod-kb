"""v2 retrieval evaluation: multilingual corpus, section-level labels, latency.

    uv run python -m eval.run_eval --suite v2 --embedder bge-m3 --device cpu

Ingests eval/corpus through the real ingest_document + index_document path into a
cached index (eval/.cache/<key>), checks the corpus and the golden set against it
(exit 1 on any mismatch), runs the golden queries through pipeline.retrieval.search,
and writes eval/results/<date>_<label>.json plus a Markdown summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

EVAL_DIR = Path(__file__).parent
REPO = EVAL_DIR.parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_V2 = EVAL_DIR / "golden_v2.jsonl"
CACHE_ROOT = EVAL_DIR / ".cache"
RESULTS_DIR = EVAL_DIR / "results"
PROJECT = "eval"
COLLECTION = "eval_chunks"
PAD_PROJECT = "__pad__"

# Pinned before any pipeline import: the corpus must never reach the LLM converter
# (cost, and it would change the text) and runs must not depend on .env.
_PINNED_ENV = {
    "LLM_CONVERTER_ENABLED": "false",
    "CONTENT_SECURITY_LLM_ENABLED": "false",
    "KB_STORAGE_BACKEND": "filesystem",
    "GIT_AUTOCOMMIT": "false",
    "QDRANT_MODE": "embedded",
    "RERANKER_ENABLED": "false",
    "QUERY_EMBED_CACHE_SIZE": "0",
}

# Code that shapes chunk ids and text: part of the index cache key.
_INDEX_CODE = (
    "pipeline/acquisition.py",
    "pipeline/chunker.py",
    "pipeline/content_filter.py",
    "pipeline/content_security.py",
    "pipeline/normalize.py",
    "pipeline/parsers/passthrough.py",
    "pipeline/promote_legal.py",
)

_STAGES = ("total", "embed", "vector", "keyword", "fuse_hydrate", "rerank")


class EvalError(Exception):
    """The corpus or golden set does not match the index: results would be meaningless."""


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    lang: str
    bucket: str
    split: str
    relevant: tuple[RelevantUnit, ...]
    must_contain: str | None


@dataclass(frozen=True)
class IndexedChunk:
    doc: str
    heading_path: tuple[str, ...]
    text: str


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.run_eval --suite v2", description="v2 retrieval evaluation"
    )
    p.add_argument("--embedder", choices=["mock", "bge-m3"], default="bge-m3")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    p.add_argument("--embed-batch-tokens", type=int, default=1024, help="ingest micro-batch budget")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--keyword-mode", choices=["legacy", "or", "and_or"], default="legacy")
    p.add_argument("--keyword-weight", type=float, default=1.0, help="RRF weight of keyword hits")
    p.add_argument("--rerank", default="off", help="off, or a cross-encoder model name")
    p.add_argument("--rerank-top-n", type=int, default=10)
    p.add_argument("--rerank-max-length", type=int, default=384)
    p.add_argument("--rerank-batch", type=int, default=16)
    p.add_argument("--candidate-k", type=int, default=50, help="per-leg depth when reranking")
    p.add_argument("--trust-remote-code", action="store_true", help="for models that need it")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--index-cache", type=Path, default=None, help="index directory to (re)use")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--pad-to", type=int, default=0, help="pad the index with distractor points")
    p.add_argument("--latency-repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no-latency", action="store_true")
    p.add_argument("--under-ingest", action="store_true", help="measure latency under ingest load")
    p.add_argument("--ingest-repeats", type=int, default=1, help="query passes under ingest load")
    p.add_argument(
        "--ingest-load",
        choices=["sequential", "concurrent"],
        default="sequential",
        help="sequential: parse a PDF, then embed (one ingest job after another); "
        "concurrent: parsing and embedding at the same time (harsher)",
    )
    p.add_argument("--label", default="", help="results file label (default: derived)")
    p.add_argument("--output", type=Path, default=RESULTS_DIR)
    p.add_argument("--compare", type=Path, default=None, help="baseline results JSON")
    p.add_argument("--min-ndcg", type=float, default=None, help="exit 1 below this nDCG@10")
    return p


# ---------------------------------------------------------------------------
# Corpus, golden set, index
# ---------------------------------------------------------------------------


def load_manifest() -> list[dict[str, str]]:
    path = CORPUS_DIR / "manifest.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_golden(path: Path = GOLDEN_V2) -> list[GoldenQuery]:
    out: list[GoldenQuery] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        rel = tuple(
            RelevantUnit(doc=r["doc"], section=r["section"], group=r.get("group"))
            for r in row["relevant"]
        )
        # Cross-lingual labels must not credit a same-language translation (group off).
        if row["bucket"] == "cross_lingual":
            rel = tuple(RelevantUnit(u.doc, u.section) for u in rel)
        out.append(
            GoldenQuery(
                id=row["id"], query=row["query"], lang=row["lang"], bucket=row["bucket"],
                split=row["split"], relevant=rel, must_contain=row.get("must_contain"),
            )
        )
    return out


def _sha(paths: Iterable[Path], extra: Sequence[str] = ()) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    for item in extra:
        h.update(item.encode())
    return h.hexdigest()[:16]


def corpus_hash() -> str:
    return _sha(sorted(CORPUS_DIR.glob("*.md")) + [CORPUS_DIR / "manifest.jsonl"])


def index_key(embedder_name: str, pad_to: int) -> str:
    files = sorted(CORPUS_DIR.glob("*.md")) + [CORPUS_DIR / "manifest.jsonl"]
    files += [REPO / rel for rel in _INDEX_CODE]
    # Host and container indexes differ numerically: never share a cache directory.
    return _sha(files, [embedder_name, str(pad_to), sys.platform, platform.machine()])


def _make_embedder(args: argparse.Namespace) -> Any:
    if args.embedder == "mock":
        from eval.run_eval import _MockEmbedder

        return _MockEmbedder()
    from pipeline.indexer import BgeM3Embedder
    from pipeline.inference import configure_threads

    configure_threads(args.threads)  # as the API does at startup (0 = derived)
    # The query cache stays off: repeated latency samples must embed every time.
    return BgeM3Embedder(
        device=None if args.device == "auto" else args.device,
        batch_tokens=args.embed_batch_tokens,
        query_cache_size=0,
    )


def _open_stores(kb: Path) -> tuple[Any, Any]:
    from qdrant_client import QdrantClient

    from pipeline.indexer import FtsDb
    from pipeline.vectorstore import QdrantVectorStore

    (kb / "index").mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(kb / "index" / "qdrant"))
    return QdrantVectorStore(client, COLLECTION), FtsDb(kb / "index" / "fts5.db")


def build_index(cache: Path, embedder: Any, pad_to: int) -> dict[str, Any]:
    """Ingest and index the corpus into *cache*; returns the cache metadata."""
    from pipeline.acquisition import ingest_document
    from pipeline.config import load_config
    from pipeline.indexer import index_document

    if cache.exists():
        shutil.rmtree(cache)
    kb = cache / "kb"
    for sub in ("documents", "index", "review"):
        (kb / sub).mkdir(parents=True)
    os.environ["KB_ROOT"] = str(kb)
    config = load_config()
    vstore, fts = _open_stores(kb)

    doc_map: dict[str, str] = {}
    n_chunks = 0
    ingest_s = index_s = 0.0
    errors: list[str] = []
    try:
        for row in load_manifest():
            t0 = time.perf_counter()
            try:
                doc_id, folder = ingest_document(
                    str(CORPUS_DIR / row["file"]), config, doc_type=row["doc_type"],
                    llm_convert=False, project=PROJECT,
                )
            except Exception as exc:  # noqa: BLE001 — reported, then the run fails
                errors.append(f"{row['file']}: ingest failed: {exc}")
                continue
            t1 = time.perf_counter()
            assert folder is not None
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "ok":
                errors.append(f"{row['file']}: status {manifest.get('status')!r}, expected 'ok'")
            if manifest.get("language") != row["lang"]:
                errors.append(
                    f"{row['file']}: language {manifest.get('language')!r} != {row['lang']!r}"
                )
            chunks = index_document(folder, config, embedder, vstore, fts, default_project=PROJECT)
            t2 = time.perf_counter()
            if not chunks:
                errors.append(f"{row['file']}: 0 chunks")
            doc_map[row["file"]] = doc_id
            n_chunks += len(chunks)
            ingest_s += t1 - t0
            index_s += t2 - t1
            print(f"  indexed {row['file']:30} {len(chunks):4} chunks", file=sys.stderr)
        if errors:
            raise EvalError("corpus did not ingest cleanly:\n  " + "\n  ".join(errors))
        if pad_to > n_chunks:
            _pad(vstore, fts, pad_to - n_chunks)
    finally:
        fts.close()
        vstore.client.close()

    meta = {
        "doc_map": doc_map,
        "n_chunks": n_chunks,
        "pad_to": pad_to,
        "ingest_seconds": round(ingest_s, 3),
        "index_seconds": round(index_s, 3),
        "index_chunks_per_s": round(n_chunks / index_s, 2) if index_s else None,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (cache / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _pad(vstore: Any, fts: Any, n: int) -> None:
    """Add *n* distractor points (random vectors, unmatched text) in another project.

    Random 1024-d vectors are near-orthogonal to every query and the text shares no
    words with the corpus, so quality is unaffected while store sizes (and the
    embedded store's brute-force scan) grow to the padded size.
    """
    import numpy as np

    from pipeline.chunker import Chunk

    rng = np.random.default_rng(0)
    batch = 1000
    for start in range(0, n, batch):
        size = min(batch, n - start)
        vecs = rng.standard_normal((size, 1024)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        chunks = [
            Chunk(
                chunk_id=f"pad{start + i:08d}#pad#0", doc_id=f"pad{(start + i) // 50:06d}",
                heading_path=["pad"], heading_level=1, parent_chunk=None, sequence=i,
                byte_start=0, byte_end=0,
                embedded_text=" ".join(f"zzpad{rng.integers(1 << 30):x}" for _ in range(40)),
                valid_until=None, extraction_method="deterministic", confidence=1.0,
            )
            for i in range(size)
        ]
        vstore.upsert(chunks, vecs.tolist(), {"project": PAD_PROJECT})
        fts.upsert_batch(chunks, PAD_PROJECT)
    print(f"  padded index with {n} distractor points", file=sys.stderr)


def load_chunks(kb: Path, doc_map: dict[str, str]) -> dict[str, IndexedChunk]:
    rev = {v: k for k, v in doc_map.items()}
    con = sqlite3.connect(f"file:{kb / 'index' / 'fts5.db'}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT chunk_id, doc_id, heading_path, embedded_text FROM chunks_meta "
            "WHERE project = ?",
            (PROJECT,),
        ).fetchall()
    finally:
        con.close()
    return {
        cid: IndexedChunk(rev[doc], tuple(json.loads(hp)), text or "")
        for cid, doc, hp, text in rows
        if doc in rev
    }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_golden(golden: Sequence[GoldenQuery], chunks: dict[str, IndexedChunk]) -> None:
    """Every label must resolve to chunks, and must_contain must be in the primary section."""
    by_doc: dict[str, list[IndexedChunk]] = defaultdict(list)
    for c in chunks.values():
        by_doc[c.doc].append(c)
    errors: list[str] = []
    for q in golden:
        if q.bucket == "negative":
            if q.relevant:
                errors.append(f"{q.id}: negative query has relevant units")
            continue
        if not q.relevant or not q.must_contain:
            errors.append(f"{q.id}: needs relevant units and must_contain")
            continue
        for unit in q.relevant:
            if unit.doc not in by_doc:
                errors.append(f"{q.id}: unknown doc {unit.doc}")
                continue
            if not any(section_matches(c.heading_path, unit.section) for c in by_doc[unit.doc]):
                errors.append(f"{q.id}: section {unit.section!r} not found in {unit.doc}")
        primary = q.relevant[0]
        texts = " ".join(
            c.text
            for c in by_doc.get(primary.doc, [])
            if section_matches(c.heading_path, primary.section)
        )
        if _norm(q.must_contain) not in _norm(texts):
            errors.append(f"{q.id}: must_contain not in {primary.doc} / {primary.section}")
    if errors:
        raise EvalError("golden set does not match the corpus:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Running queries
# ---------------------------------------------------------------------------


def _search(query: str, ctx: dict[str, Any], top_k: int) -> tuple[list[Any], Any]:
    from pipeline.retrieval import SearchTrace, search

    trace = SearchTrace()
    results = search(
        query, ctx["embedder"], ctx["vstore"], ctx["fts"], top_k=top_k, project=PROJECT,
        options=ctx["options"], reranker=ctx.get("reranker"), trace=trace,
    )
    return results, trace


def evaluate(
    golden: Sequence[GoldenQuery], ctx: dict[str, Any], top_k: int
) -> list[dict[str, Any]]:
    chunks: dict[str, IndexedChunk] = ctx["chunks"]
    rows: list[dict[str, Any]] = []
    for q in golden:
        results, trace = _search(q.query, ctx, top_k)
        retrieved = [
            RetrievedChunk(chunks[r.chunk_id].doc, chunks[r.chunk_id].heading_path)
            if r.chunk_id in chunks
            else RetrievedChunk("", ())
            for r in results
        ]
        judged = judge(retrieved, q.relevant)
        n_rel = n_relevant(q.relevant)
        pool = [
            RetrievedChunk(chunks[c].doc, chunks[c].heading_path)
            for c in trace.candidates
            if c in chunks
        ]
        pool_judged = judge(pool, q.relevant)
        rows.append(
            {
                "id": q.id, "bucket": q.bucket, "lang": q.lang, "split": q.split,
                "chunk_ids": [r.chunk_id for r in results],
                "scores": [round(float(r.score), 6) for r in results],
                "judged": judged,
                "r1": recall_at_k(judged, n_rel, 1),
                "r5": recall_at_k(judged, n_rel, 5),
                "r10": recall_at_k(judged, n_rel, 10),
                "mrr10": mrr_at_k(judged, 10),
                "ndcg10": ndcg_at_k(judged, n_rel, 10),
                "pool_recall": {
                    str(k): recall_at_k(pool_judged, n_rel, k) for k in (10, 20, 30, 50)
                },
                "pool_size": len(trace.candidates),
                "top1_score": float(results[0].score) if results else None,
                "reranked": bool(trace.reranked),
            }
        )
    return rows


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in rows if r["bucket"] != "negative"]

    def summary(items: Sequence[dict[str, Any]]) -> dict[str, float]:
        if not items:
            return {"n": 0}
        n = len(items)
        out: dict[str, float] = {"n": n}
        for key in ("r1", "r5", "r10", "mrr10", "ndcg10"):
            out[key] = round(sum(r[key] for r in items) / n, 4)
        for k in ("10", "20", "30", "50"):
            out[f"pool_r{k}"] = round(sum(r["pool_recall"][k] for r in items) / n, 4)
        return out

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in pos:
        by_bucket[r["bucket"]].append(r)
        by_lang[r["lang"]].append(r)

    def scores(items: Iterable[dict[str, Any]]) -> list[float]:
        return [r["top1_score"] for r in items if r["top1_score"] is not None]

    neg = scores(r for r in rows if r["bucket"] == "negative")
    hit = scores(r for r in pos if r["judged"] and r["judged"][0] is not None)
    return {
        "overall": summary(pos),
        "by_bucket": {b: summary(v) for b, v in sorted(by_bucket.items())},
        "by_lang": {lang: summary(v) for lang, v in sorted(by_lang.items())},
        "top1_score": {
            "negative": _dist(neg),
            "positive_hit": _dist(hit),
        },
    }


def _dist(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "p50": round(percentile(values, 50), 4),
        "p90": round(percentile(values, 90), 4),
        "max": round(max(values), 4),
    }


# ---------------------------------------------------------------------------
# Latency (idle or under ingest load)
# ---------------------------------------------------------------------------


def _minimal_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """A text-only PDF (Helvetica) — enough to drive docling's layout models."""

    def esc(s: str) -> str:
        s = s.encode("latin-1", "replace").decode("latin-1")
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",  # pages, filled below
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for lines in pages:
        page_no = len(objs) + 1
        kids.append(f"{page_no} 0 R")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_no + 1} 0 R >>".encode()
        )
        shown = " ".join(f"({esc(ln)}) '" for ln in lines)
        stream = f"BT /F1 10 Tf 40 800 Td 13 TL {shown} ET"
        data = stream.encode("latin-1")
        objs.append(b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class IngestLoad:
    """Background load like a running ingest: embedding batches plus docling PDF parses."""

    def __init__(
        self, embedder: Any, texts: Sequence[str], workdir: Path, mode: str = "sequential"
    ) -> None:
        self._mode = mode
        self._embedder = embedder
        self._texts = list(texts)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.counts = {"embed_batches": 0, "pdf_parses": 0}
        self.errors: list[str] = []
        words = " ".join(self._texts[:200]).split()
        lines = [" ".join(words[i : i + 14]) for i in range(0, min(len(words), 14 * 150), 14)]
        if not lines:
            raise EvalError("no chunk text to build the ingest-load PDF from")
        self._pdf = workdir / "load.pdf"
        self._pdf.write_bytes(_minimal_pdf([lines[i : i + 55] for i in range(0, len(lines), 55)]))

    def _run(self, loop: Callable[[], None]) -> None:
        try:
            loop()
        except Exception as exc:  # noqa: BLE001 — a dead load thread must be visible
            self.errors.append(f"{loop.__name__}: {exc}")
            print(f"  WARN: ingest load {loop.__name__} stopped: {exc}", file=sys.stderr)

    def _embed_loop(self) -> None:
        i = 0
        while not self._stop.is_set():
            batch = self._texts[i : i + 32] or self._texts[:32]
            i = (i + 32) % max(len(self._texts), 1)
            self._embedder.encode(batch)
            self.counts["embed_batches"] += 1

    def _pdf_loop(self) -> None:
        from pipeline.parsers.pdf import PdfParser

        parser = PdfParser()
        while not self._stop.is_set():
            parser.parse(self._pdf)
            self.counts["pdf_parses"] += 1

    def _job_loop(self) -> None:
        """What one ingest job after another does: parse a PDF, then embed chunks."""
        from pipeline.parsers.pdf import PdfParser

        parser = PdfParser()
        i = 0
        while not self._stop.is_set():
            parser.parse(self._pdf)
            self.counts["pdf_parses"] += 1
            for _ in range(2):
                batch = self._texts[i : i + 32] or self._texts[:32]
                i = (i + 32) % max(len(self._texts), 1)
                self._embedder.encode(batch)
                self.counts["embed_batches"] += 1

    def __enter__(self) -> IngestLoad:
        try:
            import docling.document_converter  # noqa: F401

            has_docling = True
        except ImportError:
            print("  WARN: docling missing; load is embedding only", file=sys.stderr)
            has_docling = False
        targets: list[Callable[[], None]]
        if not has_docling:
            targets = [self._embed_loop]
        elif self._mode == "sequential":
            targets = [self._job_loop]
        else:
            targets = [self._embed_loop, self._pdf_loop]
        for target in targets:
            t = threading.Thread(target=self._run, args=(target,), daemon=True)
            t.start()
            self._threads.append(t)
        time.sleep(3.0)  # let the load reach steady state (docling loads its models)
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=600)


def measure_latency(
    golden: Sequence[GoldenQuery], ctx: dict[str, Any], top_k: int, repeats: int, warmup: int
) -> dict[str, Any]:
    queries = [q.query for q in golden]
    for q in queries[:warmup]:
        _search(q, ctx, top_k)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        for q in queries:
            _, trace = _search(q, ctx, top_k)
            for stage, ms in trace.timings_ms.items():
                samples[stage].append(ms)
    return {
        stage: {
            "n": len(v),
            "p50": round(percentile(v, 50), 2),
            "p95": round(percentile(v, 95), 2),
            "p99": round(percentile(v, 99), 2),
            "mean": round(sum(v) / len(v), 2),
        }
        for stage, v in samples.items()
    }


# ---------------------------------------------------------------------------
# Context, comparison, output
# ---------------------------------------------------------------------------


def _git(*cmd: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *cmd], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_context(args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "corpus_hash": corpus_hash(),
        "golden_hash": _sha([GOLDEN_V2]),
        "n_chunks": meta["n_chunks"],
        "pad_to": meta.get("pad_to", 0),
        "qdrant_mode": "embedded",
        "embedder": args.embedder,
        "device": args.device,
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "options": {
            k: v for k, v in vars(args).items() if not isinstance(v, Path) and k != "compare"
        },
    }
    try:
        import torch

        ctx["torch"] = torch.__version__
        # At the end: docling may have changed it during an ingest-load run.
        ctx["torch_threads_end"] = torch.get_num_threads()
    except ImportError:
        pass
    return ctx


def _torch_threads() -> int | None:
    try:
        import torch

        return int(torch.get_num_threads())
    except ImportError:
        return None


def compare(rows: Sequence[dict[str, Any]], baseline_path: Path) -> dict[str, Any]:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_rows = {r["id"]: r for r in base["queries"]}
    common = [r for r in rows if r["id"] in base_rows]
    identical = [r["id"] for r in common if r["chunk_ids"] == base_rows[r["id"]]["chunk_ids"]]
    out: dict[str, Any] = {
        "baseline": baseline_path.name,
        "n_common": len(common),
        "identical_rankings": len(identical),
        "changed": [r["id"] for r in common if r["id"] not in set(identical)],
        "paired": {},
    }
    pos = [r for r in common if r["bucket"] != "negative"]
    groups: dict[str, list[dict[str, Any]]] = {"overall": pos}
    for r in pos:
        groups.setdefault(r["bucket"], []).append(r)
    for name, items in groups.items():
        out["paired"][name] = {}
        for metric in ("ndcg10", "mrr10", "r5"):
            stats = paired_deltas(
                [base_rows[r["id"]][metric] for r in items], [r[metric] for r in items]
            )
            out["paired"][name][metric] = {
                "delta": round(stats.mean_delta, 4), "wins": stats.wins,
                "losses": stats.losses, "ties": stats.ties,
                "ci95": [round(stats.ci_low, 4), round(stats.ci_high, 4)],
            }
    return out


def render_markdown(result: dict[str, Any]) -> str:
    agg = result["metrics"]
    lines = [
        f"### {result['label']} — {result['context']['n_chunks']} chunks, "
        f"{agg['overall'].get('n', 0)} queries, device {result['context']['device']}",
        "",
        "| slice | n | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | pool R@30 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def row(name: str, s: dict[str, Any]) -> str:
        if not s.get("n"):
            return f"| {name} | 0 | | | | | | |"
        return (
            f"| {name} | {s['n']} | {s['r1']:.3f} | {s['r5']:.3f} | {s['r10']:.3f} | "
            f"{s['mrr10']:.3f} | {s['ndcg10']:.3f} | {s['pool_r30']:.3f} |"
        )

    lines.append(row("**overall**", agg["overall"]))
    for name, s in agg["by_bucket"].items():
        lines.append(row(name, s))
    for name, s in agg["by_lang"].items():
        lines.append(row(f"lang={name}", s))
    t1 = agg["top1_score"]
    lines += [
        "",
        f"Top-1 score — negatives: {t1['negative']}; positive hits: {t1['positive_hit']}",
    ]
    for cond in ("latency_idle", "latency_under_ingest"):
        lat = result.get(cond)
        if not lat:
            continue
        lines += ["", f"**{cond}** (ms)", ""]
        lines += ["| stage | n | p50 | p95 | p99 |", "|---|---|---|---|---|"]
        for stage in _STAGES:
            if stage in lat:
                s = lat[stage]
                lines.append(f"| {stage} | {s['n']} | {s['p50']} | {s['p95']} | {s['p99']} |")
    ing = result.get("ingest")
    if ing:
        lines += ["", f"Ingest: {ing}"]
    cmp_ = result.get("comparison")
    if cmp_:
        lines += [
            "",
            f"vs {cmp_['baseline']}: "
            f"{cmp_['identical_rankings']}/{cmp_['n_common']} identical rankings",
        ]
        for name, metrics in cmp_["paired"].items():
            n = metrics["ndcg10"]
            lines.append(
                f"- {name}: ΔnDCG@10 {n['delta']:+.4f} (CI {n['ci95']}, "
                f"W/L/T {n['wins']}/{n['losses']}/{n['ties']}), "
                f"ΔMRR@10 {metrics['mrr10']['delta']:+.4f}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.update(_PINNED_ENV)
    import logging

    logging.basicConfig(level=logging.WARNING)

    golden = load_golden()
    if args.split != "all":
        golden = [q for q in golden if q.split == args.split]

    embedder = _make_embedder(args)
    threads_at_start = _torch_threads()
    model_name = "mock" if args.embedder == "mock" else "BAAI/bge-m3"
    cache = args.index_cache or CACHE_ROOT / index_key(model_name, args.pad_to)
    try:
        if args.rebuild_index or not (cache / "meta.json").exists():
            print(f"Building index in {cache} ...", file=sys.stderr)
            meta = build_index(cache, embedder, args.pad_to)
            built_now = True
        else:
            meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
            built_now = False
        kb = cache / "kb"
        os.environ["KB_ROOT"] = str(kb)
        chunks = load_chunks(kb, meta["doc_map"])
        validate_golden(load_golden(), chunks)
    except EvalError as exc:
        print(f"EVAL ERROR: {exc}", file=sys.stderr)
        return 1

    from pipeline.retrieval import SearchOptions

    vstore, fts = _open_stores(kb)
    options = SearchOptions(
        keyword_mode=args.keyword_mode,
        keyword_weight=args.keyword_weight,
        candidate_k=args.candidate_k,
        rerank_top_n=args.rerank_top_n,
    )
    ctx: dict[str, Any] = {
        "embedder": embedder, "vstore": vstore, "fts": fts, "chunks": chunks, "options": options,
    }
    if args.rerank != "off":
        from pipeline.reranker import CrossEncoderReranker

        print(f"Loading reranker {args.rerank} ...", file=sys.stderr)
        ctx["reranker"] = CrossEncoderReranker(
            args.rerank,
            max_length=args.rerank_max_length,
            batch_size=args.rerank_batch,
            device=None if args.device == "auto" else args.device,
            trust_remote_code=args.trust_remote_code,
            gate_timeout_s=600.0,  # measure reranking, never skip it
        )
    try:
        rows = evaluate(golden, ctx, args.top_k)
        result: dict[str, Any] = {
            "label": args.label
            or f"{args.embedder}-{args.device}-kw_{args.keyword_mode}"
            + ("" if args.rerank == "off" else f"-rr_{args.rerank.split('/')[-1]}"),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "context": {**run_context(args, meta), "torch_threads": threads_at_start},
            "metrics": aggregate(rows),
            "ingest": {
                "built_now": built_now,
                "index_chunks_per_s": meta.get("index_chunks_per_s"),
                "index_seconds": meta.get("index_seconds"),
                "ingest_seconds": meta.get("ingest_seconds"),
            },
            "queries": rows,
        }
        if not args.no_latency:
            result["latency_idle"] = measure_latency(
                golden, ctx, args.top_k, args.latency_repeats, args.warmup
            )
            if args.under_ingest:
                texts = [c.text for c in chunks.values()]
                with IngestLoad(embedder, texts, cache, args.ingest_load) as load:
                    result["latency_under_ingest"] = measure_latency(
                        golden, ctx, args.top_k, args.ingest_repeats, 0
                    )
                result["latency_under_ingest"]["load"] = {
                    "mode": args.ingest_load, **load.counts, "errors": load.errors,
                }
    finally:
        fts.close()
        vstore.client.close()

    if args.compare:
        result["comparison"] = compare(rows, args.compare)

    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = args.output / f"{stamp}_{result['label']}.json"
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(render_markdown(result))
    print(f"\nresults: {out_path}", file=sys.stderr)

    overall_ndcg = result["metrics"]["overall"].get("ndcg10", 0.0)
    if args.min_ndcg is not None and overall_ndcg < args.min_ndcg:
        print(f"nDCG@10 below {args.min_ndcg}", file=sys.stderr)
        return 1
    return 0
