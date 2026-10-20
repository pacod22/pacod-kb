# pacod-kb architecture

pacod-kb is a document knowledge base: it ingests files and web pages, converts them to
canonical Markdown, indexes them for hybrid search, and keeps every version. It is a product
of the pacod family: people sign in through **pacod-id**, and every LLM call goes through
**pacod-gw**. Deep implementation notes, module by module, are in [CLAUDE.md](../CLAUDE.md).

## Containers and the network

```
Browser ─── sign-in pages, authorize, logout ───▶ pacod-id public listener (KB_ID_ISSUER)
   │ kb_session cookie: a server-side session's random secret, nothing else
   ▼
kb-ui :8080  (web/)                       ── token, JWKS, feed, admin API ──▶ pacod-id :8071
   │ X-API-Key + X-KB-Actor
   ▼
kb-pipeline :8000  (pipeline/)  ── Bearer pgw_kb_… ──▶ pacod-gw /v1/chat/completions ──▶ providers
   │
   ├── Qdrant (embedded, or pacod-qdrant :6333)    vectors
   ├── BGE-M3 in process, or pacod-embeddings      embeddings (EMBEDDINGS_URL)
   ├── SQLite FTS5 or PostgreSQL                   keyword index
   └── filesystem archive or PostgreSQL            documents, the ledger (pacod-postgres, db kb)
```

Every container joins one external Docker network, `pacod-net`, under a name that is unique
across the pacod repositories (pacod-demo/docs/NETWORK.md): kb's own services are `kb-pipeline`
and `kb-ui` (`docker-compose.yml`, the app file); the components other products share are
`pacod-qdrant`, `pacod-postgres` (+ the one-shot `pacod-db-init`), `pacod-id`, `pacod-gw`
(gw-lite here) with `gw-mock`, and `pacod-mailpit`. kb's own stack runs its copies of those
(`docker-compose.standalone.yml`); the full stack (pacod-demo) runs one of each for every product
and loads only kb's app file. In the full stack kb also shares Qdrant (its collections are
`kb_*`; the UI's Qdrant pages show nothing else) and, with `EMBEDDINGS_URL`, pacod-ap's
embeddings service.

One flat network gives up the separation kb had before (the pipeline shared no network with
pacod-id). The pipeline still receives none of pacod-id's secrets, and the UI never receives
the gateway key: each container gets an explicit environment (`tests/test_compose_env.py`
checks this).

The pipeline runs one worker (embedded Qdrant holds a file lock; the job store is in memory). The
UI runs one worker too (session refresh locks, the pacod-id feed follower).

## Ingest, index, search

```
source (file / URL)
  → acquisition.ingest_document   SHA-256 dedup · parse · language check · content-security scan
                                  · optional LLM conversion (gateway) · noise filter · heading
                                  promotion · write document.md + manifest + source snapshot
  → indexer.index_document        heading-aware chunks · BGE-M3 embeddings · Qdrant + keyword index
  → retrieval.search              RRF fusion (k=60) of vector and BM25 ranks · valid_until filter
```

Projects are fully isolated collections (their own Qdrant collection and keyword rows). The
metadata a consumer sees is described in [metadata-contract.md](metadata-contract.md).

## Sign-in (pacod-id)

The UI is a confidential OIDC client (`pacod-kb-ui`) and a service (`pacod-kb-service`, product
`kb`), both with audience `pacod-kb` (token exchange requires the same audience).

1. `/kb/login` stores a `state` and a PKCE verifier in the signed cookie and redirects to
   `{issuer}/oidc/authorize` with the redirect URI built from `KB_UI_ORIGIN`.
2. pacod-id signs the person in (its hosted pages, in the `kb` brand) and returns to
   `/kb/auth/callback` with `code`, `state` and `iss`.
3. The UI checks `iss` and `state`, exchanges the code at the **internal** token endpoint, and
   verifies the access token (signature from the internal JWKS, issuer, audience, a session id).
   Discovery is never used: inside compose the issuer's host is not reachable, and pacod-id
   serves browser endpoints only on the issuer's own host.
4. The tokens go into a server-side session (`ui_sessions` in the UI's SQLite file, Fernet under
   an HKDF of `UI_SESSION_SECRET` and the cookie secret). The cookie carries only that secret.

**Roles.** Access tokens carry no roles. The UI pushes kb's manifest (`web/authz/kb.yaml`:
viewer, editor, admin, at the platform, none implicit) to pacod-id at every start and follows
pacod-id's authorization feed; a person's role is what the manifest allows them at the platform.
Without a kb role they see a "no access" page. If the feed copy is older than 60 seconds, pages
answer 503: who may do what is not known.

**Per request**: the session row → a fresh feed → the pacod-id session and account still live
→ refresh when the access token has under 30 s left → the role. pacod-id rotates refresh tokens
(a replaced token presented more than 30 s later signs the person out everywhere), so a refresh
takes a per-session lock, re-reads the row, persists the new pair by compare-and-swap before
using it, and retries a lost answer at once (inside the grace) only.

**Revocation** reaches the UI through the feed: `session.revoked`, `principal.disabled` and
`principal.deleted` delete the matching session rows (the feed copy remembers ended sessions
only 15 minutes). **Sign-out** ends the pacod-id session (`revoke_session`, server to server),
then sends the browser through `/oidc/logout`, which then only clears pacod-id's cookie.

**Members** (`/kb/users`, admins): the UI exchanges the admin's access token for a 60-second
admin-API token and calls pacod-id's `/admin/v1` through the kb product, so only kb's roles are
listed and changed, with pacod-id checking the admin's right to grant each one. Invitations
create accounts even though the realm is closed to sign-up.

## AI (pacod-gw)

`pipeline/gateway.py` is the only way kb calls a model: OpenAI-shaped
`POST {PACOD_GW_URL}/v1/chat/completions` with the `pgw_kb_…` key. The gateway holds the
provider keys, routes by model name (`provider/model`, or a pool `@name`), translates a PDF
`file` part for models that read documents, fails over between models, and prices each call.

- **Conversion** streams (`stream_options.include_usage`: the last chunk carries the cost), with
  the read timeout as idle timeout and a total deadline; a PDF over 20 MiB, or for a model that
  takes no documents, goes as extracted text. A cut-off, refused or empty answer keeps the
  deterministic parse.
- **Content security**: an optional second opinion (JSON-schema answer, 15 s, fails open).
- **Attribution**: `x-gw-task` (the feature), `x-gw-trace` (the job), `x-gw-end-user` (the
  project), `x-gw-user-id` (the person) — opaque ids only.
- **Retries**: once, only for a 429 after its `Retry-After`; never for 402 or 504.
- **Accounting**: kb records each call's tokens with the gateway's cost, currency and request id
  (`usage_events`, `ai_interactions`), in both storage backends (the ledger).
- **Not configured** (URL or key blank): AI is off, nothing else breaks.

## Who acted

The UI adds `X-KB-Actor` (the person's `usr_` id) to every pipeline call. The pipeline trusts it
only with the API key, captures it into each job's parameters when the job is queued, records it
on settings changes and usage rows, and passes it to the gateway. When pacod-id deletes an
account, the UI ends its sessions, asks the pipeline to replace the id by `deleted-user`
everywhere (`POST /people/{sub}/erase`), and only then confirms the deletion to pacod-id, which
keeps announcing it until confirmed.
