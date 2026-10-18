# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] — 2026-10-18

First public release.

### Ingest

- Accepts PDF (docling), DOCX, HTML, Markdown, plain text and JSON.
- Sources: upload, a watched dropzone folder, URL fetch, and watched web sources with automatic
  supersession.
- Canonical Markdown with YAML front matter. Heading promotion for legal texts in 8 languages
  (en, de, bg, ru, sr, mk, hr, sl), papers and articles. Boilerplate and noise filtering.
- Optional LLM clean-up through the pacod-gw AI gateway. A failed conversion keeps the
  deterministic parse.
- Content security: a tiered prompt-injection and spam scan blocks or quarantines documents
  before they are indexed, and search results are fenced as untrusted data.
- OCR for scanned PDFs (`OCR_ENABLED`), with two-engine confidence scoring when tesseract is
  installed.

### Index and search

- BGE-M3 dense embeddings in Qdrant (embedded or server) plus BM25 keyword search (SQLite FTS5
  or PostgreSQL), fused by reciprocal rank fusion.
- Isolated projects, each with its own collection.
- Every document version is kept. Point-in-time (`as_of`) and historical queries are filtered
  inside both searches.
- Optional cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, off by default) with a calibrated
  0–1 relevance score and `min_score`.
- Optional analysed keyword queries (`KEYWORD_QUERY_MODE`): stopwords, light stemming, article
  references.
- A priority gate puts searches ahead of ingest work. Ingest embeds length-sorted micro-batches
  and reuses the vectors of unchanged chunks on supersession and rebuild.

### Operate

- Pipeline REST API, CLI (`kb`), and a web UI (FastAPI + HTMX). Sign-in and roles (viewer,
  editor, admin) come from pacod-id.
- Jobs are queued and persisted. A ledger records AI usage and cost as priced by the gateway.
- Filesystem or PostgreSQL document storage. Docker Compose profiles for every service.
- Retrieval evaluation harness: a multilingual corpus, section-labelled golden queries, quality
  and latency metrics (`make eval`, docs/retrieval-eval.md).

[Unreleased]: https://github.com/pacod22/pacod-kb/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/pacod22/pacod-kb/releases/tag/v1.0.0
