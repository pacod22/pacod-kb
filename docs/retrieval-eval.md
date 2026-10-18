# Retrieval evaluation

How kb measures search quality and latency, the current numbers, and the decisions they back.
The harness is `eval/run_eval.py --suite v2` (code in `eval/suite_v2.py`, metrics in
`eval/metrics.py`). Every retrieval change is judged against a recorded baseline before it
ships.

## What is measured

**Corpus** (`eval/corpus/`, sources and licences in `eval/corpus/SOURCES.md`). 14 documents,
about 1.9 MB, ingested through the real `ingest_document` + `index_document` path with the
LLM converter forced off:

- GDPR and the Law Enforcement Directive in EN/DE/BG. The two share wording and article
  numbers, so they compete for every data-protection query.
- The constitutions of Bulgaria, Russia (articles 1–137), Croatia and Slovenia.
- Four Wikipedia articles, two in English and two in German.

**Golden set** (`eval/golden_v2.jsonl`). 92 hand-written queries in five buckets:

- `paraphrase` and `exact_term` (article numbers, identifiers).
- `cross_lingual`: the answer exists only in another language.
- `native`: bg/ru/de/hr/sl queries with inflected forms.
- `negative`: off-topic and in-domain questions with no answer.

Queries are split into `dev` and `test` halves: tune on dev, confirm on test.

**Relevance is judged by section, not by chunk id.** A result is relevant when it comes from a
labelled document and its heading path contains the labelled section. The match is
word-bounded, so "Article 1" never matches "Article 17". Labels therefore survive chunker
changes. Parallel translations of one article form a group that counts once. Each query's
`must_contain` snippet must appear in its labelled section, so a wrong article number fails the
run instead of passing silently.

**The run fails** (exit 1) if any corpus file:

- does not ingest with status `ok`,
- is detected in a language other than the one it declares,
- produces no chunks,
- or leaves a golden label unresolved.

CI runs the suite with the mock embedder on every push, which catches corpus or chunker drift.

**Reported metrics:**

- Quality: R@1/5/10, MRR@10 and nDCG@10, overall, per bucket and per language.
- Candidate-pool recall: is the right section among the fused candidates at all?
- Top-1 score distributions for negative queries versus correct hits: can a threshold
  separate them?
- Latency p50/p95/p99 per stage (embed, vector, keyword, fuse, rerank), idle and under
  ingest load.
- Ingest throughput in chunks/s.

## Running it

| Command | What |
|---|---|
| `make eval` | BGE-M3 on CPU, full golden set. Builds the index once into `eval/.cache/<key>`, then reuses it. |
| `make eval EVAL_ARGS="--split dev --no-latency"` | Quality only, dev half. |
| `make eval EVAL_ARGS="--under-ingest"` | Also measures latency while an ingest runs (embedding batches plus docling PDF parses). |
| `make eval EVAL_ARGS="--compare eval/results/<baseline>.json"` | Checks rankings are identical, plus paired deltas with bootstrap CIs. |
| `make eval-docker CPUS=4` | The same run inside the pipeline image, CPU-capped: the production-like measurement. |
| `make rerank-probe` | Cross-encoder latency grid on CPU (models × candidates × max length × batch). |
| `make eval-smoke` | The old fixtures smoke test, which CI runs. |

The cache key covers the corpus, the code that shapes chunks, the embedder and the platform.
Results go to `eval/results/<timestamp>_<label>.json`, and a Markdown summary is printed.
Latency runs disable the query-embedding cache.

## Results

All runs: Apple M5 CPU (no MPS), 4 torch threads, BGE-M3 fp32, embedded Qdrant, SQLite FTS5,
`top_k=10`. The JSON results are in `eval/results/`. They are not production numbers: an x86
cloud vCPU is likely several times slower per core. Repeat with `make eval-docker` on the
production machine before choosing reranker settings.

### Baseline (before this work, harder corpus): `20261018-112756_baseline2-bge-m3-cpu.json`

1,719 chunks, 77 labelled queries plus 15 negatives.

| slice | n | R@1 | R@5 | MRR@10 | nDCG@10 | pool R@30 |
|---|---|---|---|---|---|---|
| overall | 77 | 0.831 | 0.987 | 0.898 | 0.920 | 1.000 |
| paraphrase | 23 | 0.826 | 0.957 | 0.880 | 0.900 | 1.000 |
| exact_term | 18 | 0.778 | 1.000 | 0.863 | 0.897 | 1.000 |
| cross_lingual | 16 | 0.938 | 1.000 | 0.969 | 0.977 | 1.000 |
| native | 20 | 0.800 | 1.000 | 0.892 | 0.920 | 1.000 |

What the baseline shows:

- **Dense retrieval already finds the right section.** It is among the fused candidates for
  every query (pool R@30 = 1.0) and in the top 5 for 98.7 %. The headroom is in the order,
  R@1 0.83, which is what a reranker addresses.
- **The keyword half contributes nothing.** The top-1 RRF score is 0.0164 (= 1/61, rank 1 in
  the vector list alone) for every negative query and for most correct hits. Requiring every
  word means natural questions get no keyword hits. A threshold on RRF scores cannot separate
  answers from non-answers.
- **Long documents were quarantined.** Before the spam-heuristic fix, the GDPR in all three
  languages and the Bulgarian constitution were quarantined as spam (whole-document vocabulary
  diversity < 0.2) and never indexed. The baseline above already includes the fix
  (MIGRATIONS.md).
- **Ingest is slow on CPU:** 3.7 chunks/s.

### Latency: search while an ingest runs

| | load | idle p50 | under ingest p50 | under ingest p95 |
|---|---|---|---|---|
| baseline (one lock, fixed batches of 32) | concurrent | 49 ms | 24.3 s | 29.2 s |
| inference gate, 2048-token micro-batches | concurrent | 47 ms | 1.43 s | 2.91 s |
| inference gate, 2048-token micro-batches | sequential | — | 1.03 s | 1.34 s |
| **inference gate, 1024-token micro-batches (default)** | **sequential** | **47 ms** | **0.41 s** | **0.57 s** |

"Concurrent" parses PDFs and embeds at the same time, which is harsher than production.
"Sequential" is one ingest job after another: parse, then embed. That is what the pipeline
does, since jobs are serialised.

The baseline's 24 s is a query waiting for a whole 32-chunk embedding batch while docling's PDF
models compete for the same cores. With the gate, a query waits for at most one micro-batch.
The remaining wait under the concurrent load is docling running outside the gate. Rankings are
identical (92/92) and idle latency is unchanged.

### Ingest embedding throughput (`eval/embed_bench.py`, 240 real chunks)

| batching | chunks/s | min cosine vs before |
|---|---|---|
| fixed slices of 32 (before) | 3.17 | — |
| length-sorted, ≤ 1024 tokens | 6.98 | 0.9999998 |
| length-sorted, ≤ 2048 tokens | 6.76 | 0.9999997 |
| length-sorted, ≤ 4096 tokens | 6.89 | 0.9999997 |

Sorting by length removes the padding waste, so ingest is about 2.2× faster with unchanged
vectors. 1024 tokens is as fast as larger budgets and halves a query's worst-case wait: it is
the default (`EMBED_BATCH_TOKENS`).

### Keyword half: `KEYWORD_QUERY_MODE=or` (analysed terms) — not adopted

ΔMRR@10 against the baseline, paired by query (`or` and `and_or` gave identical rankings):

| variant | split | overall | exact_term | native | paraphrase | cross_lingual |
|---|---|---|---|---|---|---|
| `or`, weight 1.0 | dev | +0.004 | +0.130 | +0.050 | −0.014 | −0.167 |
| `or`, weight 0.5 | dev | +0.034 | +0.074 | +0.150 | +0.028 | −0.146 |
| `or`, weight 0.25 | dev | +0.039 | +0.074 | +0.150 | −0.014 | −0.063 |
| `or`, weight 0.25 | test | **−0.041** | −0.013 | +0.067 | −0.073 | −0.163 |
| `or`, weight 0.5 | test | **−0.051** | +0.006 | +0.017 | −0.077 | −0.163 |

The dev-split gain does not hold on the held-out half (2 wins against 5 losses).

- **Where it helps:** native-language and exact-term queries, where the keyword leg finally
  matches inflected words and article numbers.
- **Where it hurts:** cross-lingual and paraphrase queries. An English question about the
  Bulgarian constitution shares no words with its answer, and keyword hits on common words
  ("members", "national") in English GDPR text push the right section down.

**Decision:** `legacy` stays the default. The analysed modes remain available for corpora that
are mostly native-language and exact-term lookups, and are re-tested with the reranker below.

### Reranker (cross-encoder on CPU)

**Latency** (`make rerank-probe`, real chunks, batch 16, M5 CPU with 4 threads; p50, nearly
equal to p95). The reranker roughly doubles the pipeline's RAM.

| model | 10 candidates | 20 | 30 | RSS |
|---|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` (568M), max length 384 | 1.78 s | 3.05 s | 3.88 s | ~2.5–3.0 GB |
| `BAAI/bge-reranker-v2-m3`, max length 512 | 2.54 s | 4.25 s | 5.21 s | ~3.0–3.4 GB |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (118M), 512 | 0.35 s | 0.59 s | 0.72 s | ~1.4 GB |

`Alibaba-NLP/gte-multilingual-reranker-base` was not measured. It runs remote code from
another repository (`trust_remote_code`), which needs an explicit decision first.

**Quality** (all 77 labelled queries, against the baseline; end-to-end latency includes
retrieval):

| variant | R@1 | MRR@10 | Δ MRR | Δ nDCG@10 (95 % CI) | W/L | p50 / p95 |
|---|---|---|---|---|---|---|
| baseline (no reranker) | 0.831 | 0.898 | | | | 0.05 s |
| MiniLM, 20 candidates | 0.766 | 0.868 | −0.030 | −0.019 [−0.064, +0.026] | 8/12 | 0.66 / 0.70 s |
| MiniLM, 30 candidates | 0.766 | 0.864 | −0.034 | −0.022 [−0.066, +0.021] | 8/12 | 0.97 / 1.23 s |
| **bge-reranker-v2-m3, 10 candidates, length 384** | **0.909** | **0.944** | **+0.046** | **+0.034 [+0.003, +0.068]** | **9/3** | **2.07 / 2.18 s** |
| bge-reranker-v2-m3, 20 candidates, length 512 | 0.935 | 0.954 | +0.056 | +0.045 [+0.011, +0.080] | 11/3 | ~4.3 s |

By split, `bge-reranker-v2-m3` at 10 candidates gives dev +0.068 MRR (6 wins, 1 loss) and test
+0.023 (3/2). Nothing about the reranker was tuned on dev, so both halves are held-out data. The
gains are largest on exact-term and native-language queries (+0.08 MRR each). Cross-lingual
queries are unchanged.

**Its score is a usable threshold.** Top-1 relevance is p90 0.06 (max 0.35) on queries with no
answer, against a median of 0.98 on correct hits. `min_score` = 0.1 drops 14 of 15 unanswerable
queries and 2 of 70 correct top answers. RRF scores could not separate the two at all.

MiniLM is faster but worse than no reranker: it loses 0.15 MRR on cross-lingual queries, and its
scores are not calibrated (one unanswerable query scored 0.97).

**Decisions:**

- `RERANKER_MODEL` defaults to `BAAI/bge-reranker-v2-m3`, with `RERANK_TOP_N=10` and
  `RERANK_MAX_LENGTH=384`: the setting that fits a 3 s p95 on this CPU.
- `RERANKER_ENABLED` stays **false**. The gain falls just short of the +0.05 bar on the test
  half alone, and the latency must be confirmed on the production CPU.
- To switch it on, run `make eval-docker CPUS=<n> EVAL_ARGS="--rerank BAAI/bge-reranker-v2-m3"`
  on the production host. Keep the setting if p95 stays within budget; lower
  `RERANK_TOP_N`/`RERANK_MAX_LENGTH` if not.
- Consumers that need "no answer" detection can pass `min_score` (e.g. 0.1) on reranked
  requests.
- Licensing: the reranker models are published under Apache-2.0. MS MARCO-derived training
  data carries non-commercial terms (relevant for MiniLM, which was not adopted). Check your
  policy before production use.

With the reranker on, `KEYWORD_QUERY_MODE=or` (weight 0.25) gives identical metrics: the
reranker sees the same ten candidates and orders them the same way. `legacy` stays the default.

### Correctness fixes (no ranking change on the golden set)

- **`as_of`** is applied inside both searches. A test where 100 superseded chunks outrank the
  one valid chunk finds it only with the pushdown.
- **Keyword hits come from the project that owns the searched collection.**
- **Selecting an unowned collection is now vector-only search, and writes are refused.**

Identical rankings after these changes: 80/80 on the first golden set, 92/92 on the second.

## Not adopted, and what needs the next reindex

The work behind 1.0.0 deliberately avoided re-embedding existing collections. These
improvements need a reindex and are best bundled into one migration:

- **BGE-M3 learned sparse vectors** as a second named vector in Qdrant. One forward pass gives
  dense and sparse; the `sparse_linear.pt` head must be fetched into the model cache. This would
  replace the keyword leg's lexical ranking with one that works the same on both backends and
  handles inflection without stemming. Judge it with this harness against the exact_term and
  native buckets.
- **Chunker cleanup:**
  - count the breadcrumb in the 500-token budget;
  - merge heading-only chunks;
  - give a table directly under a heading its caption;
  - split oversized tables with the header repeated;
  - remove the `must_bind_next` doubling.
- **HNSW / int8 quantization** for large server-mode collections.
- **Embedder comparison** (e.g. Qwen3-Embedding-0.6B) through the harness: `--embedder` takes a
  model at the next reindex.
- **Contextual chunk headers** (LLM-written, through the gateway): costs money per chunk.
- **Rebuild into a shadow collection, then swap,** so search stays full during a rebuild. Vector
  reuse already makes rebuilds fast.
- **Serbian detection:** langdetect has no `sr` profile, so Serbian text is detected as
  hr/mk/bg/ru or rejected.

Not adopted at all:

- ColBERT multi-vectors (about 100× the vector storage, slow MaxSim on CPU).
- GraphRAG (cost; the corpus already has explicit structure).
- Late chunking (8k-token re-embeds on CPU).
- LLM query rewriting / HyDE (latency and gateway cost on every query).
