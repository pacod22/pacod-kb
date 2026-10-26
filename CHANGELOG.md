# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- pacod-id's internal listener is on `pacod-admin-net` only, an internal network shared by the pacod
  stacks, bound to its alias there, `pacod-id-internal:8071`; kb-ui joins it to call that
  address. `make network` creates the network (MIGRATIONS.md).
- The shared network pacod-net (pacod-demo/docs/NETWORK.md): every container joins it alone, under
  the convention's names (`kb-pipeline`, `kb-ui`; `pacod-qdrant`, `pacod-postgres`, `pacod-id`,
  `pacod-gw`, `pacod-mailpit`, `gw-mock`). `docker-compose.yml` holds kb's own services, the new
  `docker-compose.standalone.yml` its copies of the shared components. The bundled pacod-id keeps
  its database on `pacod-postgres` (one-shot `pacod-db-init`, `platform/postgres/kb-db.sh`). See
  MIGRATIONS.md.
- The `/data/...` bind mounts are variables (`KB_DATA`/`KB_ROOT`, `KB_MODEL_CACHE`,
  `KB_QDRANT_DATA`, `KB_POSTGRES_DATA`); a name is a Docker volume (macOS).
- `QDRANT_IMAGE_TAG` defaults to `v1.19.2`.
- Logging follows the pacod convention (pacod-demo/docs/LOGGING.md): kb-pipeline and kb-ui (uvicorn's
  lines included) write one JSON event per line by default (`LOG_FORMAT=text` for a terminal;
  `LOG_LEVEL` takes `debug|info|warn|error`); request lines without the query string, health
  checks at debug; every compose service rotates its log (json-file, 5 × 10 MB). The UI now
  honours `LOG_LEVEL`. `kb.caddy`'s access log: `KB_CADDY_LOG_FORMAT`, more redaction.

### Added

- `EMBEDDINGS_URL`: embed through the shared embeddings service (tokenizer only in the pipeline).
- `RERANKER_URL`: rerank through the same service's `/rerank`.
- `KB_QDRANT_COLLECTION_PREFIX`: the UI's Qdrant pages show and touch only kb's collections.
- `platform/caddy/kb.caddy`: the UI behind the shared edge (`KB_SITE`).

### Fixed

- `02-migrate-kb-jobs-index.sql` no longer fails on an empty database; the pipeline creates the
  index with `kb_jobs`.

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
