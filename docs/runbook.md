# pacod-kb runbook

Running pacod-kb: a development stack, a production installation, everyday operations, and what
to do when something is wrong. How the parts fit: [architecture.md](architecture.md). Upgrading
an existing installation: [MIGRATIONS.md](../MIGRATIONS.md).

## 1. A development stack on one machine

Prerequisites: Docker, `make`, `openssl`, and checkouts of pacod-id and pacod-gw next to this
one (`../pacod-id`, `../pacod-gw`; or set `PACOD_ID_SRC` / `PACOD_GW_SRC`).

```sh
make setup     # .env from .env.example, every secret generated (none printed); host dirs under /data
make up        # builds pacod-id's and the dev gateway's images the first time (Rust: slow once)
```

Every container joins the shared Docker network `pacod-net` (`make network`, run by `make setup`
and `make up`, creates it and `pacod-admin-net`, where kb-ui reaches pacod-id's internal listener) under the names of the pacod convention (pacod-demo/docs/NETWORK.md). kb itself,
`kb-pipeline` and `kb-ui`, always runs (`docker-compose.yml`). The default profiles
(`COMPOSE_PROFILES` in `.env`) add kb's copies of the shared components
(`docker-compose.standalone.yml`): `pacod-qdrant` (server), `pacod-postgres` with the one-shot
`pacod-db-init` (db), `pacod-id` (id; its database is on pacod-postgres), `pacod-mailpit`
(mailpit), and the development gateway `pacod-gw` (gw-lite in front of pacod-gw's mock
providers, `gw-mock` — nothing leaves the machine) (dev-gateway). Only one copy of each shared
component may run on a machine: stop another pacod stack before `make up`.

**Data.** On a server the data stays in host directories: `KB_ROOT` (`/data/kb`),
`KB_MODEL_CACHE` (`/data/model-cache`), `KB_QDRANT_DATA` (`/data/qdrant/storage`) and
`KB_POSTGRES_DATA` (`/data/postgres`); `make setup` creates them. A value without a slash names
a Docker volume instead (`KB_DATA` overrides `KB_ROOT` as the pipeline's mount). macOS cannot
share `/data`, so there `make setup` writes a new `.env` with the volumes `kb-data`,
`kb-model-cache`, `pacod-qdrant-data` and `pacod-postgres-data`.

| What | Address |
|---|---|
| The UI | http://localhost:8080/kb |
| pacod-id's sign-in pages (the issuer) | http://127.0.0.1:18070 |
| Mail pacod-id sends (invitations, resets) | http://127.0.0.1:18025 |
| PostgreSQL (db profile), for psql and backups | 127.0.0.1:`${KB_POSTGRES_PORT:-5432}` |
| Qdrant (server profile) | http://127.0.0.1:6333 |

When a PostgreSQL installed on the host already holds 5432, `make up` stops with "port is already
allocated": set `KB_POSTGRES_PORT` (e.g. `25432`) in `.env`. Containers always reach it on
`pacod-postgres:5432`.

Sign in with `ID_ADMIN_EMAIL` and `ID_ADMIN_PASSWORD` from `.env` (`grep ID_ADMIN_ .env`): that
account is an identity admin and a kb admin. Change its password at pacod-id
(`<issuer>/account/security`), then invite others from **Administration → Members**.

The development gateway's mock echoes the prompt, so `.env.example` leaves conversion off
(`LLM_CONVERTER_ENABLED=false`). **AI Management → Test connection / Test model** still exercises
the whole path, cost included.

Useful targets: `make ps`, `make logs S=kb-ui|kb-pipeline|pacod-id|pacod-gw`,
`make shell S=kb-pipeline`, `make db-shell`, `make check-env`, `make down` (stops every profile).

## 2. Production

### 2.1 The AI gateway

Point kb at a real gateway and drop the `dev-gateway` profile:

```sh
PACOD_GW_URL=https://gw.example.com      # or http://pacod-gw:8080, the shared one on pacod-net
PACOD_GW_API_KEY=pgw_kb_…                # a key for kb, created in the gateway
LLM_CONVERTER_ENABLED=true
LLM_CONVERTER_MODEL=anthropic/claude-opus-5-5   # a model (or @pool) the key may use
```

- **Full pacod-gw**: create an organisation and workspace for kb in its console, a key in that
  workspace, and allow the models kb uses.
- **pacod-gw-lite** (`pacod-gw/deploy/lite`): put the providers' keys there
  (`PGW_PROVIDER_ANTHROPIC_KEY`, …), declare the models (`PGW_MODEL_OPUS=anthropic/claude-opus-5-5`,
  with `_PRICE` so calls are priced) and kb's key (`PGW_CLIENT_KB_KEY=pgw_kb_…`, generated with
  `printf 'pgw_kb_%s\n' "$(openssl rand -hex 24)"`). Its body limit (`PGW_MAX_BODY_BYTES`,
  32 MiB) fits kb's inline PDFs (at most 20 MiB raw).

Without `PACOD_GW_URL` or `PACOD_GW_API_KEY` AI is off and documents keep their deterministic
parse. AI Management shows the gateway's state and tests it.

### 2.2 pacod-id

Either keep the bundled pacod-id (profile `id`) behind TLS, or use an existing one.

**Bundled.** pacod-id's sign-in pages answer only on the issuer's host, and in production that is
an https address behind your proxy:

```sh
KB_ID_ISSUER=https://id.example.com   # the proxy forwards it to pacod-id:8070 (Host unchanged)
KB_UI_ORIGIN=https://kb.example.com   # the proxy forwards it to kb-ui:8080
UI_SESSION_HTTPS_ONLY=true
UI_BIND=127.0.0.1                     # the proxy runs on this host; or an internal address
UI_FORWARDED_ALLOW_IPS=<proxy address>
PID_TRUSTED_PROXIES=<proxy address>   # so pacod-id's sign-in throttle sees people's addresses
PID_SMTP_HOST=…  PID_SMTP_PORT=587  PID_SMTP_TLS=starttls  PID_SMTP_USERNAME=…  PID_SMTP_PASSWORD=…
ID_MAIL_FROM=id@example.com
```

Drop `mailpit` from the profiles. After the first sign-in, remove `ID_ADMIN_PASSWORD` from
`.env` (pacod-id creates the bootstrap admin once; see pacod-id's runbook, section 9). **Back up
`PID_KEK`** separately from the database backups: it seals what pacod-id stores. pacod-id's
database (`pacod_id`, role `pacod_id`) is on `pacod-postgres`, next to kb's; `pacod-db-init`
creates it (`docker/pacod-id/id-db.sh`).

**An existing pacod-id** (for example one shared with pacod-gw): add kb's realm and clients from
[`docker/pacod-id/pacod-id.toml`](../docker/pacod-id/pacod-id.toml) to its configuration (the `kb`
brand needs a pacod-id with that brand), drop the `id` and `mailpit` profiles, and set
`KB_ID_ISSUER` (its public address for the realm), `PACOD_ID_INTERNAL_URL` (its internal listener
as the UI reaches it), `KB_ID_UI_SECRET` and `KB_ID_SERVICE_SECRET` (the clients' secrets).

### 2.3 The rest

- `WEB_FETCH_ALLOW_PRIVATE=false` refuses URL ingests that resolve to private addresses (an
  SSRF guard); recommended wherever the pipeline can reach internal services.
- Pin `QDRANT_IMAGE_TAG` (default `v1.19.2`): Qdrant storage cannot be opened by an older
  server.
- **A shared Qdrant.** On a Qdrant other products use too (`pacod-qdrant` in the full stack),
  kb's collections are those named `KB_QDRANT_COLLECTION_PREFIX…` (default `kb_`; the default
  collection is `kb_chunks`) plus its projects' own. The UI's Qdrant pages list, browse, export,
  search and delete only those (anything else is a 404) and create only prefixed ones. Name new
  projects' collections with the prefix too. The pipeline's `GET /collections` still lists every
  collection on the server.
- **A shared embeddings service.** `EMBEDDINGS_URL=http://pacod-embeddings:8080` makes the
  pipeline embed through pacod-ap's embeddings service (`POST /embed`, the same BGE-M3 dense
  vectors, normalized) instead of loading the model: only the tokenizer is loaded, which saves
  about 2 GB of RAM. Existing vectors stay valid. Requests are retried on connection errors and
  5xx (`EMBEDDINGS_TIMEOUT_SECS`, 120 s each); a vector that is not 1024-dimensional is refused.
  `RERANKER_URL` (the same service's `/rerank`, bge-reranker-v2-m3) does the same for the
  reranker, with `RERANKER_TIMEOUT_MS` (5 s) per call. If that service's bi-encoder is ever
  replaced (pacod-ap's fine-tuning and `/reload`), kb's index must be rebuilt
  (`kb rebuild --all-projects`).
- `make upgrade` pulls, rebuilds kb's images and restarts; `make images FORCE=1` rebuilds
  pacod-id's and the dev gateway's.

### 2.4 Search speed and quality

Measurements and the reasoning behind the defaults are in [retrieval-eval.md](retrieval-eval.md).

- **CPU threads.** Leave `TORCH_NUM_THREADS=0`. On Linux the pipeline derives the count from
  the container's CPU quota (`--cpus` / `cpus:`), and docling uses the same count. Set it only
  to reserve cores for something else on the host.
- **Search during ingest.** A query waits for at most one embedding micro-batch, at most
  `EMBED_BATCH_TOKENS` padded tokens (1024 by default). Lower it if queries during large
  ingests are too slow. That costs some ingest throughput.
- **The reranker** (`RERANKER_ENABLED`) reorders the best fused candidates with a
  cross-encoder. It gives better ordering and a meaningful 0–1 score (`min_score`), but costs
  CPU time per query and RAM: about 2.3 GB for `BAAI/bge-reranker-v2-m3`. Before switching it
  on, measure on the production machine:
  1. `make rerank-probe` gives the latency per model, number of candidates and length.
  2. `make eval-docker CPUS=<the container's CPUs> EVAL_ARGS="--rerank <model> --rerank-top-n <n>"`
     gives quality plus end-to-end latency.
  3. Pick the largest `RERANK_TOP_N` whose p95 fits your budget.

  It can also be switched at runtime (`POST /settings {"reranker_enabled": true}`, loaded in
  the background; `/health` shows `reranker.state`). A request's `"rerank": true` or `false`
  overrides the setting for that request only while the model is loaded; a request never loads
  it. Latency grows with `max(top_k, RERANK_TOP_N)`: every returned result is reranked.
- **Models offline.** Run the pipeline once with network access, or run the probe, so the
  embedder and reranker land in the model cache (`KB_MODEL_CACHE`). Then set
  `HF_HUB_OFFLINE=1`. A model that needs remote code (`RERANKER_TRUST_REMOTE_CODE=true`)
  should also pin `RERANKER_REVISION` and `RERANKER_CODE_REVISION`.
- **Keyword half of hybrid search.** `KEYWORD_QUERY_MODE=legacy` requires every query word to
  match, so the keyword list is empty for most natural-language questions. `or` / `and_or`
  search analysed terms instead (stopwords dropped, inflections as prefixes, article numbers
  as phrases). Change the mode only after an eval run shows it helps on your documents.

## 3. Operations

### 3.1 Ingesting

From the UI (**Ingest**), from the dropzone, or with the CLI inside the pipeline container
(`make ingest URL=…`, `make ingest-file FILE=…`, or `docker exec kb-pipeline kb …`):

```sh
kb ingest path/to/file.pdf                    # pdf / docx / md / txt / html / json
kb ingest https://example.com/page --doc-type web_page
kb ingest new_version.txt --supersedes <prior-doc-id>
kb ingest security-research-notes.md --trusted   # bypass the content-security scan
kb fetch https://example.com/page --output-dir ./out   # Markdown files only, no KB storage
kb query "what is the maximum penalty?" --top-k 5 [--historical]
kb status
kb reindex <doc-id-prefix>     ·   kb rebuild [--all-projects]
kb export --output kb_export.json
```

The CLI follows the env configuration (`LLM_CONVERTER_ENABLED`, `CONTENT_SECURITY_ENABLED`);
the dashboard's switches apply to everything that runs inside the pipeline API (UI ingests, the
dropzone, `/watch`).

**Dropzone.** Files placed in `KB_DROPZONE` (default `KB_ROOT/dropzone`) are ingested every
`DROPZONE_INTERVAL_SECS` (0 = off): known ones are moved to `KB_DONE` as skipped, new ones are
ingested and indexed and moved there, failures go to `KB_FAILED`. Files changed in the last two
seconds wait for the next scan. Each scan that touches a file appears as a job.

**Watched web sources** (`KB_ROOT/watch.json`):

```json
[
  {"url": "https://legislation.example.gov/act-42", "doc_type": "legal_code", "label": "Act 42"},
  {"url": "https://example.com/policy", "doc_type": "web_page", "label": "Policy"}
]
```

`kb watch --once` (or **Dashboard → Run watch**) re-fetches each source; a changed page is ingested
and supersedes the prior version, an unchanged one is skipped, and a source that reverted
reinstates the earlier version. `kb watch --interval 1800` polls.

Supported languages: `en`, `de`, `bg`, `ru`, `sr`, `mk`, `hr`, `sl` (others are refused; an
undetectable language is accepted as `unknown` with a warning).

### 3.2 Runtime switches

The dashboard's switches (content-security scan, AI conversion, language validation) and the
active project live in the pipeline process (not persisted; env vars are authoritative at
start). The same through the API:

```sh
curl -H "X-API-Key: $PIPELINE_API_SECRET" http://localhost:8000/settings
curl -X POST -H "X-API-Key: $PIPELINE_API_SECRET" -H 'Content-Type: application/json' \
     -d '{"llm_converter_enabled": false}' http://localhost:8000/settings
curl -X POST -H "X-API-Key: $PIPELINE_API_SECRET" -H 'Content-Type: application/json' \
     -d '{"reranker_enabled": true}' http://localhost:8000/settings    # loads in the background
```

Models, output limits and prompts are changed in **AI Management** (admins); they are kept in
the ledger and apply on the next call.

### 3.3 People

Roles are granted in **Administration → Members**: invite by email (pacod-id mails the
invitation, or gives a link to pass on), change a role, remove. Roles: **viewer** (read, search),
**editor** (ingest, re-index, delete documents, manage projects), **admin** (AI Management,
prompts, switches, members). The last admin cannot be removed. Passwords, sessions and
sign-in methods are each person's own, at pacod-id (**My account**).

### 3.4 Logs

kb logs like every pacod product (pacod-demo/docs/LOGGING.md): one event per line on stdout,
kept by Docker (`json-file`, 5 files of 10 MB per container, every service of both compose
files). `make logs S=<service>` follows one.

| Variable | Values | Default |
|---|---|---|
| `LOG_LEVEL` | `debug`, `info`, `warn`, `error` (any case; `warning` = `warn`; anything else = `info`) | `info` |
| `LOG_FORMAT` | `json` (one object per line) or `text` (for reading in a terminal) | `json` |
| `SERVICE_NAME` | the `service` of every event; compose sets `kb-pipeline` and `kb-ui` | |
| `KB_CADDY_LOG_FORMAT` | the edge's access log (`platform/caddy/kb.caddy`, full stack): `console` or `json` | `console` |

Both set kb-pipeline and kb-ui (and the `kb` CLI, which logs to stderr: its output is on
stdout). kb's own stack passes `PID_LOG`/`PID_LOG_FORMAT` to pacod-id and
`RUST_LOG`/`PGW_LOG_FORMAT` to the development gateway (both `text` by default). A JSON event
starts with `timestamp` (UTC, milliseconds), `level`, `service`, `logger`, `message`, then the
event's fields; an exception adds `error` and `stack_trace`:

```text
{"timestamp":"2026-10-23T06:41:07.695Z","level":"info","service":"kb-ui","logger":"web.identity","message":"kb catalogue pushed to pacod-id (version 3, changed False)"}
{"timestamp":"2026-10-23T06:41:09.120Z","level":"info","service":"kb-ui","logger":"uvicorn.access","message":"GET /kb/login 302","method":"GET","path":"/kb/login","status":302,"client_ip":"172.18.0.9"}
2026-10-23T06:41:09.120Z INFO  kb-ui uvicorn.access: GET /kb/login 302 method=GET path=/kb/login status=302 client_ip=172.18.0.9
```

uvicorn's own lines are events too; its request lines never carry the query string (it can
hold sign-in codes), and the health checks (`/health`, `/kb/healthz`) appear only at `debug`, like
the outgoing requests' lines of the HTTP client (httpx: full URLs, pacod-id's feed polls).
Read JSON with `docker logs kb-ui | jq -r '.message'`, or set `LOG_FORMAT=text` and
`make up`.

## 4. Backups

| What | Where |
|---|---|
| Documents (filesystem mode), keyword index, ledger | `KB_DATA` / `KB_ROOT` (`/data/kb`) |
| Documents, keyword index, ledger, jobs (database mode) | pacod-postgres, database `kb` (`KB_POSTGRES_DATA`, `/data/postgres`) |
| Vectors | Qdrant storage (`KB_QDRANT_DATA`, `/data/qdrant/storage`), or rebuild with `kb rebuild --all-projects` |
| UI sessions | volume `kb-ui-data` (losing it signs everyone out, nothing more) |
| Accounts, roles, sessions | pacod-postgres, database `pacod_id` — and `PID_KEK` |
| Secrets | `.env` |

## 5. When something is wrong

| Symptom | Cause | What to do |
|---|---|---|
| Every page says "pacod-id cannot be reached" (503) | The UI's copy of pacod-id's decisions is older than 60 s | `make logs S=pacod-id`; `curl localhost:8080/kb/healthz` shows `identity: stale`. Pages recover by themselves once pacod-id answers |
| "Sign-in is not set up on this server" | `PACOD_ID_ISSUER`, `KB_UI_ORIGIN` or a client secret is blank in the UI | `make check-env`, then `make up` |
| pacod-id shows an error about the redirect URI | `KB_UI_ORIGIN` differs from the address used in the browser | Use exactly `KB_UI_ORIGIN`; the UI moves `/kb/login` there, but bookmarks to another host do not match |
| "No role in the knowledge base" | The account has no kb role | An admin invites the address or gives it a role in Members |
| Signed out everywhere unexpectedly | pacod-id saw a refresh token used twice (reuse) | Rare; check pacod-id's log for the account. Run the UI with one worker only |
| Documents are not converted although AI is on | No gateway (`PACOD_GW_URL`/`PACOD_GW_API_KEY`), or the call failed | The pipeline log says "LLM conversion skipped" or why it failed; AI Management → Test model; the interaction log has the gateway's request id |
| "The AI gateway refused the key" | Wrong or revoked `PACOD_GW_API_KEY` | Fix it in `.env`, restart the pipeline |
| "The AI budget is used up" | The gateway's budget for kb's key | Raise it in the gateway |
| The pipeline answers 503 to everything but `/health` | `PIPELINE_API_SECRET` is blank | `make setup` (it generates one), then `make up` |
| `make up` stops with "missing for profiles …" | A secret the profiles need is blank | `make setup` |
| `make up` stops with "hosts from before the shared network" | `.env` still names `postgres`, `kb-qdrant`, `gw`, `id` or `mailpit` | Use the `pacod-*` names ([MIGRATIONS.md](../MIGRATIONS.md)) |
| `make up` stops about `kb-id-pg-data` | pacod-id's data is still in its old volume | Move it to pacod-postgres ([MIGRATIONS.md](../MIGRATIONS.md)), then remove the volume |
| "container name … is already in use" | Another pacod stack runs its copy of a shared component (`pacod-postgres`, `pacod-id`, …) | Stop that stack, or run kb inside the full stack |
| `pacod-db-init` exits 1 | Wrong `KB_DATABASE_PASSWORD` for the existing PostgreSQL, or tables owned by a role other than `kb` | `make logs S=pacod-db-init` |
