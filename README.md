# pacod-kb

[![CI](https://github.com/pacod22/pacod-kb/actions/workflows/ci.yml/badge.svg)](https://github.com/pacod22/pacod-kb/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A document knowledge base for retrieval-augmented generation: it ingests files and web pages,
converts them to canonical Markdown, indexes them for hybrid semantic and keyword search, keeps
every version, and hands downstream models chunks with a stable metadata contract. People sign
in through **[pacod-id](https://github.com/pacod22/pacod-id)**; every LLM call goes through the
AI gateway **[pacod-gw](https://github.com/pacod22/pacod-gw)**.

- [docs/runbook.md](docs/runbook.md): running it — a development stack, production, operations,
  troubleshooting.
- [docs/architecture.md](docs/architecture.md): how the parts fit — containers, sign-in, the
  gateway, who acted.
- [docs/metadata-contract.md](docs/metadata-contract.md): what a RAG consumer gets — chunk and
  manifest fields, versioning, byte offsets.
- [docs/retrieval-eval.md](docs/retrieval-eval.md): how search quality and latency are measured,
  the current numbers, and the decisions behind the defaults.
- [MIGRATIONS.md](MIGRATIONS.md): upgrading an existing installation; [CHANGELOG.md](CHANGELOG.md).
- [CLAUDE.md](CLAUDE.md): notes for working on the code, module by module.

## What it does

- **Ingest** PDF, DOCX, HTML, Markdown, text and JSON — uploaded, dropped into a folder, fetched
  from a URL, or watched for changes. Legal texts, papers and articles get structure-aware
  heading promotion; boilerplate is filtered; an optional LLM pass (through the gateway) cleans
  up the Markdown, and a failed one never loses the document.
- **Defend** against indirect prompt injection and spam: a tiered content-security scan blocks
  or quarantines suspicious documents before they are written or indexed, and search results
  are fenced as untrusted data.
- **Search** with BGE-M3 embeddings in Qdrant and BM25 keywords, fused by RRF, per project
  (isolated collections), with point-in-time queries over superseded versions. An optional
  cross-encoder reranker gives a calibrated 0–1 relevance score. Searches stay responsive while
  documents are being ingested, on CPU alone.
- **Operate** from a web UI (FastAPI + HTMX) or a CLI: documents, projects, jobs, AI usage and
  cost as the gateway priced it, members and roles (viewer, editor, admin, granted in pacod-id).

## Layout

```text
pacod-kb/
├── pipeline/            # the pipeline API (:8000): ingest, index, search, jobs, the gateway client
├── web/                 # the UI (:8080): sign-in through pacod-id, sessions, members, the pages
├── eval/                # retrieval evaluation: multilingual corpus, golden queries, latency
├── tests/               # pytest; a fake pacod-id (tests/fake_id.py) for the UI
├── docker/              # the bundled pacod-id's realm, clients and database script, the UI entrypoint
├── platform/            # kb's share of the shared configuration: postgres/ (kb-db.sh + SQL), caddy/
├── vendor/wheels/       # pacod-id's Python package, vendored (scripts/vendor-pacod-id.sh)
├── scripts/             # make setup's secret generator, the vendoring script
└── docs/
```

## Getting started

Prerequisites: Docker, and checkouts of pacod-id and pacod-gw next to this one.

```sh
make setup     # .env with generated secrets (none printed); on macOS, data in Docker volumes
make up        # everything, with pacod-id and a development gateway; slow the first time
open http://localhost:8080/kb    # sign in as ID_ADMIN_EMAIL / ID_ADMIN_PASSWORD (grep ID_ADMIN_ .env)
```

Production — a real gateway, pacod-id behind TLS — is [docs/runbook.md](docs/runbook.md), section 2.

## Shared network

kb follows the pacod network and naming convention,
[pacod-demo/docs/NETWORK.md](../pacod-demo/docs/NETWORK.md): every container joins the external
Docker network `pacod-net`, kb's own services are `kb-pipeline` and `kb-ui`, and the components
other products share keep their shared names (`pacod-qdrant`, `pacod-postgres`, `pacod-id`,
`pacod-gw`, `pacod-mailpit`).

- `docker-compose.yml` is the app file: kb-pipeline and kb-ui only.
- `docker-compose.standalone.yml` holds kb's own copies of the shared components, switched by
  `COMPOSE_PROFILES`. `make up` loads both files.
- The full stack (pacod-demo) loads only the app file and runs every shared component once. kb
  then keeps its `kb_*` collections on the shared Qdrant (the UI's Qdrant pages show nothing
  else), its database `kb` on the shared PostgreSQL (`platform/postgres/kb-db.sh`), and can embed
  through the shared embeddings service (`EMBEDDINGS_URL`). Its edge serves the UI through
  `platform/caddy/kb.caddy` (`KB_SITE`).

## Development

Python 3.13 with uv.

```sh
uv sync                                   # dependencies, dev tools included
make check                                # ruff, mypy --strict (pipeline/, web/), pytest
uv run pytest tests/test_gateway.py -v    # one file
KB_UPDATE_ROUTE_SNAPSHOTS=1 uv run pytest tests/test_route_contract.py   # after an intended API change
make eval                                 # retrieval eval with BGE-M3 on CPU (downloads the model)
```

Tests need no network, no model download and no running pacod-id or gateway.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md).

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The evaluation corpus in
`eval/corpus/` is third-party material under its own terms ([SOURCES.md](eval/corpus/SOURCES.md)).

