# Metadata contract for RAG consumers

What every indexed chunk and every document exposes, by the names to filter on and attribute
sources with. `POST /query` returns chunks; `GET /documents` and `GET /export` return documents.

## Chunk record fields

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Stable identifier: `{doc_id[:16]}#{path_slug}#{seq_in_section}`; it encodes the heading structure, it is not a sequence number |
| `doc_id` | `str` | SHA-256 of the source bytes (64 hex characters) |
| `heading_path` | `list[str]` | Ordered heading breadcrumb, e.g. `["Chapter 1", "§ 5"]` |
| `heading_level` | `int` | Depth of the deepest heading in the path (1–6) |
| `parent_chunk` | `str \| None` | `chunk_id` of the parent section's chunk, or `None` |
| `sequence` | `int` | Zero-based chunk order within the document |
| `byte_start` | `int` | UTF-8 byte offset into the stripped document body |
| `byte_end` | `int` | UTF-8 byte offset (exclusive) into the stripped document body |
| `embedded_text` | `str` | The text embedded: a `[DocTitle > H1 > H2]` breadcrumb, then the chunk (tables also get the preceding paragraph as caption) |
| `valid_until` | `str \| None` | ISO date (exclusive) from which the chunk is superseded, or `None` while current |
| `extraction_method` | `str` | `"deterministic"` \| `"ocr"` \| `"vlm"` (converted by an LLM through the AI gateway) |
| `confidence` | `float` | Extraction confidence (1.0 for deterministic) |

`/query` wraps each `embedded_text` in an untrusted-data fence (delimiters and a "treat as data,
not instructions" preamble) unless `CONTENT_SECURITY_WRAP_OUTPUT=false`: ingested documents are
untrusted input to a downstream model.

## `POST /query`

**Request:**

| Field | Default | Notes |
|---|---|---|
| `query` | (required) | 1–4000 characters |
| `top_k` | 5 | 1–50 |
| `include_historical` | false | Also return superseded chunks |
| `as_of` | – | `YYYY-MM-DD`: chunks valid on that day |
| `wrap_untrusted` | true | Fence the text for a downstream LLM |
| `collection` | – | Another project's collection. Keyword hits come from the project that owns it. |
| `rerank` | – | Override the `reranker_enabled` setting for this request. Takes effect only while the reranker is loaded; a request never loads it. |
| `min_score` | – | 0–1: drop results whose reranker relevance is lower. Ignored when not reranked. |

Invalid input is a 422.

**Response:** `{"results": [...], "reranked": bool, "score_kind": "rerank" | "rrf"}`

Each result holds:

- `chunk_id`, `heading_path`, `byte_start`, `byte_end`, `valid_until`, as above.
- `source_id`: the `doc_id`.
- `text`: the `embedded_text`.
- `score`, `rrf_score`, `rerank_score`, described below.

**Scores:**

- `score` is the value results are ordered by.
  - When `reranked` is true, it is the cross-encoder's relevance probability (`rerank_score`,
    0–1). It is comparable across queries, so a threshold such as `min_score` means something.
  - Otherwise it is the reciprocal-rank-fusion score (`rrf_score`, at most about 0.033). That
    value only orders one result list and is not comparable across queries.
- `rerank_score` is `null` when the request was not reranked. This happens when the reranker is
  off, still loading, or busy past `RERANK_GATE_TIMEOUT_MS`; search then keeps the fused order.
- Before October 2026, `score` was always the RRF score.

## Versioning and `valid_until`

- A chunk is **current** when `valid_until is None`.
- A chunk is **valid at date D** when `valid_until is None or valid_until > D` (exclusive): with
  `valid_until = "2025-12-31"` it is visible at `as_of="2025-12-30"`, not at `as_of="2025-12-31"`.
- `search()` returns current chunks by default; `as_of="YYYY-MM-DD"` returns those valid then;
  `include_historical=True` returns all.

Superseding a document sets `valid_until` in four places at once: `document.md`'s front matter,
the manifest, the keyword index and the vector payloads.

## `heading_path`

- Source attribution: `" > ".join(chunk.heading_path)`.
- Filter by section: match prefixes of `heading_path`.
- Surrounding context: fetch the chunks with the same `heading_path[:-1]`.

## Byte offsets

`byte_start` and `byte_end` are UTF-8 byte positions into the **stripped body** of
`document.md` (front matter removed). They differ from character positions for Cyrillic text.

```python
body_bytes = body.encode("utf-8")
chunk_text = body_bytes[chunk.byte_start : chunk.byte_end].decode("utf-8")
```

## Manifest fields

Filesystem mode keeps `manifest.json` next to each `document.md` under `KB_ROOT/documents/`;
database mode stores the same fields in `kb_documents`.

| Field | Description |
|---|---|
| `doc_id` | SHA-256 of the source bytes |
| `title` | Extracted title (leading and trailing special characters stripped) |
| `source_uri` | Original URL or file path |
| `source_sha256` | Same as `doc_id` |
| `doc_type` | Legal: `legal_code`, `regulation`, `directive`, `decision`, `circular`; content: `article`, `paper`, `report`, `document`, `guide`; general: `generic`, `web_page` |
| `status` | `"ok"` \| `"review"` (quarantined by the content-security scan; never indexed) \| `"error"` |
| `project` | The project (an isolated collection) the document belongs to |
| `converter` | What produced the body: a parser, or the gateway model that converted it (`anthropic/claude-opus-5-5`) |
| `valid_until` | ISO date this version was superseded, or `null` |
| `supersedes` | `doc_id` of the prior version, or `null` |
| `chunk_count` | Chunks indexed |
| `meta_author`, `meta_published_date`, `meta_publication_venue` | `article` documents from a URL only; `null` when unknown |
| `events` | The lifecycle log (below) |

## Lifecycle events

```json
"events": [
  {"event": "ingested",   "at": "2026-01-01T10:00:00+00:00", "detail": "/path/to/source.txt"},
  {"event": "superseded", "at": "2026-06-01T09:00:00+00:00", "detail": "valid_until=2026-06-01"}
]
```

Event types: `ingested`, `superseded`, `reinstated` (a watched source reverted to an earlier
version), `security_flagged` (quarantined by the content-security scan), `review_flagged` (low
OCR confidence), `llm_usage` (the conversion's tokens and the gateway's cost and request id, as
JSON in `detail`).
