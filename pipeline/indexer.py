"""Embedder protocol, BGE-M3 implementation, keyword stores, and index_document."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pipeline.inference import GATE, InferenceGate
from pipeline.schema_validation import validate as validate_schema

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from pipeline.chunker import Chunk
    from pipeline.config import Config
    from pipeline.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# Progress callback: (current, total, message). Used by index_document to report
# indexing phases (chunking / embedding / writing) to the job store.
ProgressCb = Callable[[int, int, str], None]
# Embedder progress callback: (done, total) chunks embedded so far.
EmbedProgressCb = Callable[[int, int], None]

_FTS5_SPECIAL = re.compile(r"[^\w\s]", re.UNICODE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _sanitize_fts5(query: str) -> str:
    """Convert a raw user query to a safe FTS5 MATCH expression.

    FTS5 treats characters like commas, parentheses, quotes, and keywords
    (AND/OR/NOT) as syntax.  This function strips all non-word, non-space
    characters and returns each token quoted as a phrase so FTS5 treats
    them literally.  Returns '""' (matches nothing) for an empty input.
    """
    tokens = _FTS5_SPECIAL.sub(" ", query).split()
    return " ".join(f'"{t}"' for t in tokens) if tokens else '""'


# Keyword query modes (KEYWORD_QUERY_MODE). legacy: every word, as typed, must match.
# or: any analysed term (stopwords dropped, inflections as prefixes, article phrases);
# and_or: chunks with every analysed term first, then the rest of the "or" matches.
KEYWORD_MODES = ("legacy", "or", "and_or")


def build_fts5_match(query: str, mode: str) -> str | None:
    """FTS5 MATCH expression for *mode* ("legacy", "or" or "and"); None if nothing is left.

    Terms come from pipeline.text_lexicon (word characters only, so quoting is safe).
    """
    if mode == "legacy":
        return _sanitize_fts5(query)
    from pipeline.text_lexicon import analyze_query

    kq = analyze_query(query)
    terms = [f'"{t.text}"*' if t.prefix else f'"{t.text}"' for t in kq.terms]
    if mode == "and":
        return " AND ".join(terms) if len(terms) > 1 else None
    parts = terms + [f'"{a} {b}"' for a, b in kq.phrases]
    return " OR ".join(parts) if parts else None


def build_pg_tsquery(query: str) -> tuple[str, list[str]] | None:
    """(any-term tsquery, one tsquery per term) for PostgreSQL's 'simple' config."""
    from pipeline.text_lexicon import analyze_query

    kq = analyze_query(query)
    parts = [f"'{t.text}':*" if t.prefix else f"'{t.text}'" for t in kq.terms]
    parts += [f"'{a}' <-> '{b}'" for a, b in kq.phrases]
    if not parts:
        return None
    return " | ".join(f"({p})" for p in parts), parts




# ---------------------------------------------------------------------------
# Embedder protocol
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    """Embedding model interface."""

    VECTOR_DIM: int

    def encode(
        self, texts: list[str], *, progress_cb: EmbedProgressCb | None = None
    ) -> list[list[float]]: ...

    def count_tokens(self, text: str) -> int: ...


class FtsStore(Protocol):
    def upsert_batch(
        self, chunks: list[Chunk], project: str = "default", *, retrieved_at: str | None = None
    ) -> None: ...

    def search(
        self,
        query: str,
        top_k: int,
        *,
        include_historical: bool = False,
        project: str | None = None,
        as_of: str | None = None,
        mode: str = "legacy",
    ) -> list[tuple[str, int]]: ...

    def valid_until_values_after(self, as_of: str, limit: int = 1001) -> list[str]: ...

    def get_chunk_meta(self, chunk_id: str) -> dict[str, Any] | None: ...

    def get_embedded_text(self, chunk_id: str) -> str | None: ...

    def get_chunks_bulk(self, chunk_ids: list[str]) -> dict[str, tuple[dict[str, Any], str]]: ...

    def delete_by_doc(self, doc_id: str) -> None: ...

    def set_valid_until(self, doc_id: str, valid_until: str | None) -> None: ...

    def retag_project(self, old: str, new: str) -> int: ...

    def drop_by_project(self, project: str) -> None: ...

    def drop_all(self) -> None: ...

    def close(self) -> None: ...


_MAX_EMBED_BATCH = 32  # sequences per model call, whatever the token budget


class BgeM3Embedder:
    """BGE-M3 via sentence-transformers (multilingual, 1024-dim).

    One instance is shared by /query handlers (threadpool), background jobs and the
    dropzone thread. Model calls pass the shared inference gate (pipeline.inference):
    ingest embeds length-sorted micro-batches of at most *batch_tokens* padded tokens,
    and a query's embedding overtakes waiting batches, so search stays responsive while
    a document is being indexed.
    """

    VECTOR_DIM: int = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str | None = None,
        batch_tokens: int = 1024,
        query_cache_size: int = 256,
        gate: InferenceGate | None = None,
    ) -> None:
        import copy

        from sentence_transformers import SentenceTransformer

        # device=None lets sentence-transformers pick (cuda → mps → cpu).
        self._model: Any = SentenceTransformer(model_name, device=device)
        self.model_name = model_name  # tagged on every vector (embedding reuse)
        self._gate = gate or GATE
        self._batch_tokens = max(1, batch_tokens)
        self._max_len = int(getattr(self._model, "max_seq_length", 0) or 8192)
        # A private tokenizer for count_tokens and batch planning: HF fast tokenizers
        # raise "Already borrowed" when one instance is used from two threads, and the
        # chunker's many counts must not queue behind model calls.
        self._tokenizer: Any = copy.deepcopy(self._model.tokenizer)
        self._tokenizer_lock = threading.Lock()
        self._cache_size = max(0, query_cache_size)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _plan(self, texts: list[str]) -> list[list[int]]:
        """Indices in micro-batches: ascending length, padded size within the budget."""
        with self._tokenizer_lock:
            ids = self._tokenizer(texts, truncation=True, max_length=self._max_len)["input_ids"]
        lengths = [len(x) for x in ids]
        batches: list[list[int]] = []
        current: list[int] = []
        for i in sorted(range(len(texts)), key=lengths.__getitem__):
            # Ascending order: the newcomer is the longest, so it sets the padded size.
            full = len(current) >= _MAX_EMBED_BATCH
            if current and (full or lengths[i] * (len(current) + 1) > self._batch_tokens):
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)
        return batches

    def encode(
        self, texts: list[str], *, progress_cb: EmbedProgressCb | None = None
    ) -> list[list[float]]:
        total = len(texts)
        out: list[list[float]] = [[] for _ in range(total)]
        done = reported = 0
        for batch in self._plan(texts) if texts else []:
            with self._gate.batch():
                part: Any = self._model.encode(
                    [texts[i] for i in batch],
                    batch_size=len(batch),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            for i, vector in zip(batch, part, strict=True):
                out[i] = vector.tolist()
            done += len(batch)
            # Throttled: every report is a job-store write.
            if progress_cb is not None and (done - reported >= _MAX_EMBED_BATCH or done == total):
                progress_cb(done, total)
                reported = done
        return out

    def encode_query(self, text: str) -> list[float]:
        """Embed one search query ahead of any waiting ingest batch (LRU-cached)."""
        if self._cache_size:
            with self._cache_lock:
                hit = self._cache.get(text)
                if hit is not None:
                    self._cache.move_to_end(text)
                    return hit
        with self._gate.interactive():
            part: Any = self._model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )
        vector: list[float] = part[0].tolist()
        if self._cache_size:
            with self._cache_lock:
                self._cache[text] = vector
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return vector

    def count_tokens(self, text: str) -> int:
        with self._tokenizer_lock:
            ids: list[int] = self._tokenizer.encode(text, add_special_tokens=False)
        return len(ids)


def make_embedder(config: Config) -> BgeM3Embedder:
    """The embedder configured by *config*, after setting the inference thread count."""
    from pipeline.inference import configure_threads

    configure_threads(config.torch_num_threads)
    return BgeM3Embedder(
        config.embedding_model,
        device=None if config.inference_device == "auto" else config.inference_device,
        batch_tokens=config.embed_batch_tokens,
        query_cache_size=config.query_embed_cache_size,
    )


# ---------------------------------------------------------------------------
# Keyword stores (SQLite FTS5 + PostgreSQL full-text search)
# ---------------------------------------------------------------------------


_META_COLUMNS = (
    "chunk_id",
    "doc_id",
    "heading_path",
    "heading_level",
    "parent_chunk",
    "sequence",
    "byte_start",
    "byte_end",
    "valid_until",
    "extraction_method",
    "confidence",
    "project",
    "retrieved_at",
)


# Database files whose schema this process has already created or migrated: /query and
# every index job open a connection, and re-running the DDL each time is wasted work.
_FTS_READY: set[str] = set()
_FTS_READY_LOCK = threading.Lock()


class FtsDb:
    """SQLite FTS5 store for BM25 keyword search over chunk embedded_text.

    ``chunks_meta`` holds one row per chunk, including the ``rowid`` of its
    ``chunks_fts`` row (``fts_rowid``) and a copy of ``embedded_text``.  FTS5
    tables have no secondary indexes, so every lookup, join and delete goes
    through ``chunks_meta`` (indexed) and touches ``chunks_fts`` by rowid only.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path), timeout=30.0)
        # WAL lets /query readers proceed while the dropzone thread or a job
        # writes; busy_timeout absorbs short writer overlaps instead of failing
        # with "database is locked".
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA busy_timeout=30000")
        self._con.execute("PRAGMA synchronous=NORMAL")
        key = str(path.resolve())
        with _FTS_READY_LOCK:
            ready = key in _FTS_READY and self._con.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'chunks_meta'"
            ).fetchone()
            if not ready:
                self._setup()
                _FTS_READY.add(key)

    def _setup(self) -> None:
        self._con.executescript("""
            CREATE TABLE IF NOT EXISTS chunks_meta (
                chunk_id        TEXT PRIMARY KEY,
                doc_id          TEXT NOT NULL,
                heading_path    TEXT NOT NULL,
                heading_level   INTEGER NOT NULL,
                parent_chunk    TEXT,
                sequence        INTEGER NOT NULL,
                byte_start      INTEGER NOT NULL,
                byte_end        INTEGER NOT NULL,
                valid_until     TEXT,
                extraction_method TEXT NOT NULL,
                confidence      REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                embedded_text,
                tokenize='unicode61'
            );
        """)
        cols = {r[1] for r in self._con.execute("PRAGMA table_info(chunks_meta)").fetchall()}
        for name, ddl in (
            ("project", "TEXT DEFAULT 'default'"),
            ("fts_rowid", "INTEGER"),
            ("embedded_text", "TEXT"),
            ("retrieved_at", "TEXT"),
        ):
            if name not in cols:
                self._con.execute(f"ALTER TABLE chunks_meta ADD COLUMN {name} {ddl}")
        if "fts_rowid" not in cols:
            self._backfill_rowids()
        self._con.executescript("""
            CREATE INDEX IF NOT EXISTS idx_chunks_meta_doc_id ON chunks_meta (doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_meta_project ON chunks_meta (project);
            CREATE INDEX IF NOT EXISTS idx_chunks_meta_valid_until ON chunks_meta (valid_until);
            CREATE INDEX IF NOT EXISTS idx_chunks_meta_fts_rowid ON chunks_meta (fts_rowid);
        """)
        self._con.commit()

    def _backfill_rowids(self) -> None:
        """One-time migration: link existing meta rows to their FTS rows."""
        self._con.executescript("""
            CREATE TEMP TABLE _fts_map AS
                SELECT rowid AS fts_rowid, chunk_id, embedded_text FROM chunks_fts;
            CREATE INDEX _fts_map_cid ON _fts_map (chunk_id);
            UPDATE chunks_meta SET
                fts_rowid = (SELECT fts_rowid FROM _fts_map m
                             WHERE m.chunk_id = chunks_meta.chunk_id),
                embedded_text = (SELECT embedded_text FROM _fts_map m
                                 WHERE m.chunk_id = chunks_meta.chunk_id);
            DROP TABLE _fts_map;
        """)

    def _delete_where(self, where: str, params: tuple[Any, ...]) -> None:
        self._con.execute(
            f"DELETE FROM chunks_fts WHERE rowid IN "
            f"(SELECT fts_rowid FROM chunks_meta WHERE {where})",
            params,
        )
        self._con.execute(f"DELETE FROM chunks_meta WHERE {where}", params)

    def upsert(self, chunk: Chunk, project: str = "default") -> None:
        self.upsert_batch([chunk], project)

    def upsert_batch(
        self, chunks: list[Chunk], project: str = "default", *, retrieved_at: str | None = None
    ) -> None:
        """Insert/replace *chunks* in a single transaction."""
        if not chunks:
            return
        with self._con:
            ids = [c.chunk_id for c in chunks]
            for start in range(0, len(ids), 500):
                part = ids[start : start + 500]
                marks = ",".join("?" * len(part))
                self._delete_where(f"chunk_id IN ({marks})", tuple(part))
            # Explicit rowids allow two executemany calls instead of two statements per
            # chunk; the DELETE above already holds the write lock, so MAX is stable.
            base = int(self._con.execute("SELECT COALESCE(MAX(rowid), 0) FROM chunks_fts")
                       .fetchone()[0])
            self._con.executemany(
                "INSERT INTO chunks_fts (rowid, chunk_id, embedded_text) VALUES (?, ?, ?)",
                [(base + i, c.chunk_id, c.embedded_text) for i, c in enumerate(chunks, 1)],
            )
            self._con.executemany(
                """INSERT INTO chunks_meta (
                       chunk_id, doc_id, heading_path, heading_level, parent_chunk,
                       sequence, byte_start, byte_end, valid_until, extraction_method,
                       confidence, project, fts_rowid, embedded_text, retrieved_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        json.dumps(chunk.heading_path, ensure_ascii=False),
                        chunk.heading_level,
                        chunk.parent_chunk,
                        chunk.sequence,
                        chunk.byte_start,
                        chunk.byte_end,
                        chunk.valid_until,
                        chunk.extraction_method,
                        chunk.confidence,
                        project or "default",
                        base + i,
                        chunk.embedded_text,
                        retrieved_at,
                    )
                    for i, chunk in enumerate(chunks, 1)
                ],
            )

    def search(
        self,
        query: str,
        top_k: int,
        *,
        include_historical: bool = False,
        project: str | None = None,
        as_of: str | None = None,
        mode: str = "legacy",
    ) -> list[tuple[str, int]]:
        """Return [(chunk_id, rank_position)] sorted best-first.

        *as_of* (ISO date) keeps chunks valid on that date: valid_until null or later.
        *mode* is one of KEYWORD_MODES.
        """
        scope = (include_historical, project, as_of)
        if mode == "and_or":
            first = self._match(build_fts5_match(query, "and"), top_k, *scope)
            rest = self._match(build_fts5_match(query, "or"), top_k, *scope)
            known = set(first)
            ids = first + [c for c in rest if c not in known]
        else:
            ids = self._match(build_fts5_match(query, mode), top_k, *scope)
        return [(cid, i + 1) for i, cid in enumerate(ids[:top_k])]

    def _match(
        self,
        expression: str | None,
        top_k: int,
        include_historical: bool,
        project: str | None,
        as_of: str | None,
    ) -> list[str]:
        if expression is None:
            return []
        where = ["chunks_fts MATCH ?"]
        params: list[Any] = [expression]
        if as_of is not None:
            where.append("(m.valid_until IS NULL OR m.valid_until > ?)")
            params.append(as_of)
        elif not include_historical:
            where.append("m.valid_until IS NULL")
        if project:
            where.append("m.project = ?")
            params.append(project)
        params.append(top_k)
        rows = self._con.execute(
            f"""SELECT m.chunk_id
                FROM chunks_fts
                JOIN chunks_meta m ON m.fts_rowid = chunks_fts.rowid
                WHERE {" AND ".join(where)}
                ORDER BY chunks_fts.rank
                LIMIT ?""",
            params,
        ).fetchall()
        return [str(row[0]) for row in rows]

    def valid_until_values_after(self, as_of: str, limit: int = 1001) -> list[str]:
        """Distinct valid_until dates later than *as_of* (indexed; for as_of pushdown)."""
        rows = self._con.execute(
            "SELECT DISTINCT valid_until FROM chunks_meta WHERE valid_until > ? LIMIT ?",
            (as_of, limit),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def get_chunks_bulk(self, chunk_ids: list[str]) -> dict[str, tuple[dict[str, Any], str]]:
        """Return {chunk_id: (meta, embedded_text)} for the given ids (one query per 500)."""
        out: dict[str, tuple[dict[str, Any], str]] = {}
        cols = ", ".join(_META_COLUMNS)
        for start in range(0, len(chunk_ids), 500):
            part = chunk_ids[start : start + 500]
            marks = ",".join("?" * len(part))
            for row in self._con.execute(
                f"SELECT {cols}, embedded_text FROM chunks_meta WHERE chunk_id IN ({marks})",
                part,
            ):
                meta = dict(zip(_META_COLUMNS, row[:-1], strict=True))
                out[str(meta["chunk_id"])] = (meta, row[-1] or "")
        return out

    def get_chunk_meta(self, chunk_id: str) -> dict[str, Any] | None:
        hit = self.get_chunks_bulk([chunk_id]).get(chunk_id)
        return hit[0] if hit else None

    def get_embedded_text(self, chunk_id: str) -> str | None:
        hit = self.get_chunks_bulk([chunk_id]).get(chunk_id)
        return hit[1] if hit else None

    def delete_by_doc(self, doc_id: str) -> None:
        with self._con:
            self._delete_where("doc_id = ?", (doc_id,))

    def set_valid_until(self, doc_id: str, valid_until: str | None) -> None:
        with self._con:
            self._con.execute(
                "UPDATE chunks_meta SET valid_until = ? WHERE doc_id = ?",
                (valid_until, doc_id),
            )

    def retag_project(self, old: str, new: str) -> int:
        """Move every chunk tagged with project *old* to *new*; return rows changed."""
        with self._con:
            cur = self._con.execute(
                "UPDATE chunks_meta SET project = ? WHERE project = ?", (new, old)
            )
        return cur.rowcount

    def drop_by_project(self, project: str) -> None:
        """Delete all FTS5 and metadata entries for a given project slug."""
        with self._con:
            self._delete_where("project = ?", (project,))

    def drop_all(self) -> None:
        with self._con:
            self._con.execute("DELETE FROM chunks_fts")
            self._con.execute("DELETE FROM chunks_meta")

    def close(self) -> None:
        self._con.close()


class PostgresFtsDb:
    """PostgreSQL keyword store replacing SQLite FTS5 for DB-backed mode."""

    def __init__(self, engine: Engine, *, shared: bool = False, timeout_ms: int = 1500) -> None:
        self._engine = engine
        # A shared instance is cached by make_fts_db and outlives callers.
        self._shared = shared
        # Bound for analysed (OR) keyword queries; past it the query is vector-only.
        self._timeout_ms = max(1, timeout_ms)
        self._setup()

    def _setup(self) -> None:
        from sqlalchemy import text

        with self._engine.begin() as con:
            con.execute(
                text("""
                CREATE TABLE IF NOT EXISTS kb_chunks_fts (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    heading_level INTEGER NOT NULL,
                    parent_chunk TEXT,
                    sequence INTEGER NOT NULL,
                    byte_start INTEGER NOT NULL,
                    byte_end INTEGER NOT NULL,
                    valid_until TEXT,
                    extraction_method TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default',
                    embedded_text TEXT NOT NULL,
                    search_vector tsvector NOT NULL
                )
            """)
            )
            con.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_fts_search
                ON kb_chunks_fts USING GIN (search_vector)
            """)
            )
            con.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_fts_doc_id
                ON kb_chunks_fts (doc_id)
            """)
            )
            con.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_fts_project
                ON kb_chunks_fts (project)
            """)
            )
            con.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_fts_valid_until
                ON kb_chunks_fts (valid_until)
            """)
            )
            con.execute(
                text("ALTER TABLE kb_chunks_fts ADD COLUMN IF NOT EXISTS retrieved_at TEXT")
            )

    def _row_count(self) -> int:
        from sqlalchemy import text

        with self._engine.begin() as con:
            row = con.execute(text("SELECT count(*) FROM kb_chunks_fts")).scalar_one()
        return int(row)

    def migrate_from_sqlite(self, sqlite_path: Path) -> int:
        """Copy legacy SQLite FTS rows into PostgreSQL when table is empty."""
        if not sqlite_path.exists() or self._row_count() > 0:
            return 0

        con = sqlite3.connect(str(sqlite_path))
        try:
            rows = con.execute(
                """
                SELECT
                    m.chunk_id,
                    m.doc_id,
                    m.heading_path,
                    m.heading_level,
                    m.parent_chunk,
                    m.sequence,
                    m.byte_start,
                    m.byte_end,
                    m.valid_until,
                    m.extraction_method,
                    m.confidence,
                    COALESCE(m.project, 'default') AS project,
                    f.embedded_text
                FROM chunks_meta m
                JOIN chunks_fts f ON m.chunk_id = f.chunk_id
                """
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return 0

        from sqlalchemy import text

        with self._engine.begin() as pg:
            stmt = text("""
                INSERT INTO kb_chunks_fts (
                    chunk_id,
                    doc_id,
                    heading_path,
                    heading_level,
                    parent_chunk,
                    sequence,
                    byte_start,
                    byte_end,
                    valid_until,
                    extraction_method,
                    confidence,
                    project,
                    embedded_text,
                    search_vector
                ) VALUES (
                    :chunk_id,
                    :doc_id,
                    :heading_path,
                    :heading_level,
                    :parent_chunk,
                    :sequence,
                    :byte_start,
                    :byte_end,
                    :valid_until,
                    :extraction_method,
                    :confidence,
                    :project,
                    :embedded_text,
                    to_tsvector('simple', :embedded_text)
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    heading_path = EXCLUDED.heading_path,
                    heading_level = EXCLUDED.heading_level,
                    parent_chunk = EXCLUDED.parent_chunk,
                    sequence = EXCLUDED.sequence,
                    byte_start = EXCLUDED.byte_start,
                    byte_end = EXCLUDED.byte_end,
                    valid_until = EXCLUDED.valid_until,
                    extraction_method = EXCLUDED.extraction_method,
                    confidence = EXCLUDED.confidence,
                    project = EXCLUDED.project,
                    embedded_text = EXCLUDED.embedded_text,
                    search_vector = EXCLUDED.search_vector
            """)
            for r in rows:
                pg.execute(
                    stmt,
                    {
                        "chunk_id": r[0],
                        "doc_id": r[1],
                        "heading_path": r[2],
                        "heading_level": r[3],
                        "parent_chunk": r[4],
                        "sequence": r[5],
                        "byte_start": r[6],
                        "byte_end": r[7],
                        "valid_until": r[8],
                        "extraction_method": r[9],
                        "confidence": r[10],
                        "project": r[11],
                        "embedded_text": r[12],
                    },
                )
        return len(rows)

    def upsert(self, chunk: Chunk, project: str = "default") -> None:
        self.upsert_batch([chunk], project)

    def upsert_batch(
        self, chunks: list[Chunk], project: str = "default", *, retrieved_at: str | None = None
    ) -> None:
        """Insert/replace *chunks* in one transaction (executemany)."""
        from sqlalchemy import text

        if not chunks:
            return
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "heading_path": json.dumps(c.heading_path, ensure_ascii=False),
                "heading_level": c.heading_level,
                "parent_chunk": c.parent_chunk,
                "sequence": c.sequence,
                "byte_start": c.byte_start,
                "byte_end": c.byte_end,
                "valid_until": c.valid_until,
                "extraction_method": c.extraction_method,
                "confidence": c.confidence,
                "project": project or "default",
                "embedded_text": c.embedded_text,
                "retrieved_at": retrieved_at,
            }
            for c in chunks
        ]
        with self._engine.begin() as con:
            con.execute(
                text("""
                INSERT INTO kb_chunks_fts (
                    chunk_id, doc_id, heading_path, heading_level, parent_chunk,
                    sequence, byte_start, byte_end, valid_until, extraction_method,
                    confidence, project, embedded_text, retrieved_at, search_vector
                ) VALUES (
                    :chunk_id, :doc_id, :heading_path, :heading_level, :parent_chunk,
                    :sequence, :byte_start, :byte_end, :valid_until, :extraction_method,
                    :confidence, :project, :embedded_text, :retrieved_at,
                    to_tsvector('simple', :embedded_text)
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    heading_path = EXCLUDED.heading_path,
                    heading_level = EXCLUDED.heading_level,
                    parent_chunk = EXCLUDED.parent_chunk,
                    sequence = EXCLUDED.sequence,
                    byte_start = EXCLUDED.byte_start,
                    byte_end = EXCLUDED.byte_end,
                    valid_until = EXCLUDED.valid_until,
                    extraction_method = EXCLUDED.extraction_method,
                    confidence = EXCLUDED.confidence,
                    project = EXCLUDED.project,
                    embedded_text = EXCLUDED.embedded_text,
                    retrieved_at = EXCLUDED.retrieved_at,
                    search_vector = EXCLUDED.search_vector
            """),
                rows,
            )

    def search(
        self,
        query: str,
        top_k: int,
        *,
        include_historical: bool = False,
        project: str | None = None,
        as_of: str | None = None,
        mode: str = "legacy",
    ) -> list[tuple[str, int]]:
        from sqlalchemy import text

        q = query.strip()
        if not q:
            return []
        if mode != "legacy":
            return self._search_analysed(q, top_k, include_historical, project, as_of)

        where: list[str] = ["search_vector @@ websearch_to_tsquery('simple', :q)"]
        params: dict[str, Any] = {"q": q, "limit": top_k}
        if as_of is not None:
            where.append("(valid_until IS NULL OR valid_until > :as_of)")
            params["as_of"] = as_of
        elif not include_historical:
            where.append("valid_until IS NULL")
        if project:
            where.append("project = :project")
            params["project"] = project

        sql = (
            "SELECT chunk_id "
            "FROM kb_chunks_fts "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('simple', :q)) DESC "
            "LIMIT :limit"
        )
        with self._engine.begin() as con:
            rows = con.execute(text(sql), params).fetchall()
        return [(str(row[0]), i + 1) for i, row in enumerate(rows)]

    def _search_analysed(
        self,
        query: str,
        top_k: int,
        include_historical: bool,
        project: str | None,
        as_of: str | None,
    ) -> list[tuple[str, int]]:
        """Any analysed term matches; more distinct matched terms rank first.

        ts_rank_cd has no IDF, so under OR a chunk matching one common word could
        outrank one matching every term; counting matched terms first prevents that
        ("and_or" therefore behaves like "or" here).
        """
        from sqlalchemy import text
        from sqlalchemy.exc import OperationalError

        built = build_pg_tsquery(query)
        if built is None:
            return []
        any_q, parts = built
        params: dict[str, Any] = {"q": any_q, "limit": top_k}
        params.update({f"t{i}": p for i, p in enumerate(parts)})
        where = ["search_vector @@ to_tsquery('simple', :q)"]
        if as_of is not None:
            where.append("(valid_until IS NULL OR valid_until > :as_of)")
            params["as_of"] = as_of
        elif not include_historical:
            where.append("valid_until IS NULL")
        if project:
            where.append("project = :project")
            params["project"] = project
        matched = " + ".join(
            f"(search_vector @@ to_tsquery('simple', :t{i}))::int" for i in range(len(parts))
        )
        sql = (
            f"SELECT chunk_id FROM kb_chunks_fts WHERE {' AND '.join(where)} "
            f"ORDER BY ({matched}) DESC, "
            "ts_rank_cd(search_vector, to_tsquery('simple', :q), 1) DESC, chunk_id "
            "LIMIT :limit"
        )
        try:
            with self._engine.begin() as con:
                con.execute(text(f"SET LOCAL statement_timeout = {int(self._timeout_ms)}"))
                rows = con.execute(text(sql), params).fetchall()
        except OperationalError as exc:
            logger.warning("Keyword search skipped (timeout or error): %s", type(exc).__name__)
            return []
        return [(str(row[0]), i + 1) for i, row in enumerate(rows)]

    def valid_until_values_after(self, as_of: str, limit: int = 1001) -> list[str]:
        from sqlalchemy import text

        with self._engine.begin() as con:
            rows = con.execute(
                text(
                    "SELECT DISTINCT valid_until FROM kb_chunks_fts "
                    "WHERE valid_until > :as_of LIMIT :limit"
                ),
                {"as_of": as_of, "limit": limit},
            ).fetchall()
        return [str(r[0]) for r in rows]

    def get_chunks_bulk(self, chunk_ids: list[str]) -> dict[str, tuple[dict[str, Any], str]]:
        from sqlalchemy import bindparam, text

        if not chunk_ids:
            return {}
        stmt = text(
            f"SELECT {', '.join(_META_COLUMNS)}, embedded_text "
            "FROM kb_chunks_fts WHERE chunk_id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))
        with self._engine.begin() as con:
            rows = con.execute(stmt, {"ids": list(chunk_ids)}).mappings().all()
        out: dict[str, tuple[dict[str, Any], str]] = {}
        for r in rows:
            meta = {c: r[c] for c in _META_COLUMNS}
            out[str(r["chunk_id"])] = (meta, str(r["embedded_text"] or ""))
        return out

    def get_chunk_meta(self, chunk_id: str) -> dict[str, Any] | None:
        from sqlalchemy import text

        with self._engine.begin() as con:
            row = (
                con.execute(
                    text("""
                    SELECT
                        chunk_id,
                        doc_id,
                        heading_path,
                        heading_level,
                        parent_chunk,
                        sequence,
                        byte_start,
                        byte_end,
                        valid_until,
                        extraction_method,
                        confidence,
                        project
                    FROM kb_chunks_fts
                    WHERE chunk_id = :chunk_id
                """),
                    {"chunk_id": chunk_id},
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def get_embedded_text(self, chunk_id: str) -> str | None:
        from sqlalchemy import text

        with self._engine.begin() as con:
            row = con.execute(
                text("SELECT embedded_text FROM kb_chunks_fts WHERE chunk_id = :chunk_id"),
                {"chunk_id": chunk_id},
            ).first()
        return str(row[0]) if row else None

    def delete_by_doc(self, doc_id: str) -> None:
        from sqlalchemy import text

        with self._engine.begin() as con:
            con.execute(
                text("DELETE FROM kb_chunks_fts WHERE doc_id = :doc_id"), {"doc_id": doc_id}
            )

    def set_valid_until(self, doc_id: str, valid_until: str | None) -> None:
        from sqlalchemy import text

        with self._engine.begin() as con:
            con.execute(
                text("UPDATE kb_chunks_fts SET valid_until = :valid_until WHERE doc_id = :doc_id"),
                {"doc_id": doc_id, "valid_until": valid_until},
            )

    def retag_project(self, old: str, new: str) -> int:
        from sqlalchemy import text

        with self._engine.begin() as con:
            result = con.execute(
                text("UPDATE kb_chunks_fts SET project = :new WHERE project = :old"),
                {"new": new, "old": old},
            )
            return int(result.rowcount or 0)

    def drop_by_project(self, project: str) -> None:
        from sqlalchemy import text

        with self._engine.begin() as con:
            con.execute(
                text("DELETE FROM kb_chunks_fts WHERE project = :project"), {"project": project}
            )

    def drop_all(self) -> None:
        from sqlalchemy import text

        with self._engine.begin() as con:
            con.execute(text("DELETE FROM kb_chunks_fts"))

    def close(self) -> None:
        if not self._shared:
            self._engine.dispose()


_PG_FTS_CACHE: dict[str, PostgresFtsDb] = {}
_PG_FTS_LOCK = threading.Lock()


def make_fts_db(config: Config) -> FtsStore:
    """Create the keyword store backend for the current storage mode.

    In database mode, use PostgreSQL FTS and migrate once from legacy SQLite data.
    """
    sqlite_path = config.index_dir / "fts5.db"
    if config.storage_backend != "database":
        return FtsDb(sqlite_path)

    from sqlalchemy import create_engine

    from pipeline.document_store import _build_database_url

    # One engine (connection pool) per database, created once: building it per
    # call cost a pool, a TCP/auth handshake and 5 DDL statements per /query.
    url = _build_database_url(config)
    with _PG_FTS_LOCK:
        pg = _PG_FTS_CACHE.get(url)
        if pg is None:
            engine = create_engine(url, echo=False, pool_pre_ping=True)
            pg = PostgresFtsDb(engine, shared=True, timeout_ms=config.keyword_pg_timeout_ms)
            migrated = pg.migrate_from_sqlite(sqlite_path)
            if migrated:
                logger.info("Migrated %d FTS rows from SQLite to PostgreSQL", migrated)
            _PG_FTS_CACHE[url] = pg
    return pg


# ---------------------------------------------------------------------------
# index_document — the single indexing entry point
# ---------------------------------------------------------------------------


def _validate_chunk(chunk: Chunk) -> None:
    data: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.doc_id,
        "heading_path": chunk.heading_path,
        "heading_level": chunk.heading_level,
        "parent_id": chunk.parent_chunk,
        "sequence": chunk.sequence,
        "byte_start": chunk.byte_start,
        "byte_end": chunk.byte_end,
        "text": chunk.embedded_text,
        "raw_text": chunk.text,
        "chunk_type": "child",
        "content_hash": chunk.content_hash,
        "valid_until": chunk.valid_until,
        "extraction_method": chunk.extraction_method,
        "confidence": chunk.confidence,
    }
    validate_schema(data, "chunk.json")


def _purge_doc(doc_id: object, vector_store: VectorStore, fts_db: FtsStore) -> None:
    """Remove every index entry for *doc_id* (no-op when the id is unknown)."""
    if doc_id:
        vector_store.delete_by_doc(str(doc_id))
        fts_db.delete_by_doc(str(doc_id))


def _reusable_vectors(
    chunks: list[Chunk],
    config: Config,
    embedder: Embedder,
    vector_store: VectorStore,
    cache: Mapping[tuple[str, str], Sequence[float]] | None,
) -> dict[int, list[float]]:
    """{chunk index: stored vector} for chunks whose exact embedded text was embedded before.

    Never fails the indexing: a lookup error just means embedding everything.
    """
    model = getattr(embedder, "model_name", None)
    if not config.embed_reuse or not model:
        return {}
    from pipeline.vectorstore import reuse_key

    keys = [reuse_key(c.content_hash, c.embedded_text) for c in chunks]
    found: dict[tuple[str, str], Sequence[float]] = {}
    if cache:
        found.update({k: cache[k] for k in keys if k in cache})
    missing = [(c.content_hash, c.embedded_text) for c, k in zip(chunks, keys, strict=True)
               if k not in found and c.content_hash]
    lookup = getattr(vector_store, "fetch_reusable_vectors", None)
    if missing and lookup is not None:
        try:
            found.update(
                lookup(missing, model=model, allow_untagged=config.embed_reuse_untagged)
            )
        except Exception as exc:  # noqa: BLE001 — embed instead
            logger.warning("Vector reuse lookup failed (embedding all chunks): %s", exc)
    return {i: [float(x) for x in found[k]] for i, k in enumerate(keys) if k in found}


def index_document(
    doc_folder: Path | None,
    config: Config,
    embedder: Embedder,
    vector_store: VectorStore,
    fts_db: FtsStore,
    *,
    doc_data: tuple[dict[str, Any], str] | None = None,
    progress_cb: ProgressCb | None = None,
    default_project: str | None = None,
    vector_cache: Mapping[tuple[str, str], Sequence[float]] | None = None,
) -> list[Chunk]:
    """Chunk, embed, and index one document.  Updates manifest chunk_count.

    Chunks whose exact embedded text already has a vector from the same model (in
    *vector_cache*, a rebuild's snapshot, or among the collection's points: a previous
    version) reuse it instead of being embedded again (EMBED_REUSE).

    Documents with no project, or the legacy ``"default"`` tag, are indexed
    under *default_project* when given.

    Pass *doc_data* = (manifest_dict, body) when *doc_folder* is None (database
    mode).  In filesystem mode (*doc_folder* is a real Path), *doc_data* is
    ignored and all reads come from the folder.

    Does NOT modify acquisition outputs — only reads document.md and writes
    to the index stores (Qdrant + FTS5) and updates chunk_count in manifest.
    """
    from pipeline.chunker import chunk_document
    from pipeline.normalize import strip_front_matter

    # Resolve front matter and body from the appropriate source.
    if doc_folder is not None:
        # Filesystem mode — read from document.md.
        # Quarantine enforcement: never index a document that isn't cleared ("ok").
        manifest_path = doc_folder / "manifest.json"
        if manifest_path.exists():
            try:
                fs_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                fs_manifest = {}
            status = fs_manifest.get("status", "ok")
            if status != "ok":
                logger.warning(
                    "Skipping indexing for %s — manifest status=%s (not 'ok')",
                    doc_folder.name,
                    status,
                )
                _purge_doc(fs_manifest.get("doc_id"), vector_store, fts_db)
                return []

        doc_md = (doc_folder / "document.md").read_text(encoding="utf-8")
        front, body = strip_front_matter(doc_md)
        name_for_log = doc_folder.name
    else:
        # Database mode — use caller-supplied data.
        if doc_data is None:
            raise ValueError("doc_data is required when doc_folder is None")
        front, body = doc_data
        doc_status = str(front.get("status", "ok"))
        if doc_status != "ok":
            logger.warning(
                "Skipping indexing for %s — manifest status=%s (not 'ok')",
                str(front.get("doc_id", ""))[:8],
                doc_status,
            )
            _purge_doc(front.get("doc_id"), vector_store, fts_db)
            return []
        name_for_log = str(front.get("doc_id", ""))[:8]

    doc_id: str = front["doc_id"]
    doc_title: str = front.get("title", name_for_log)
    valid_until: str | None = front.get("valid_until")
    extraction_method: str = front.get("extraction_method", "deterministic")
    raw_project = front.get("project")
    doc_project: str = (
        str(raw_project)
        if raw_project and raw_project != "default"
        else (default_project or "default")
    )

    source_uri: str = str(front.get("source_uri") or "")
    year_match = _YEAR_RE.search(doc_title) or _YEAR_RE.search(source_uri)
    doc_meta: dict[str, Any] = {
        "source_id": doc_id,
        "source_type": str(front.get("doc_type") or "generic"),
        "title": doc_title,
        "legal_year": int(year_match.group()) if year_match else 0,
        "language": str(front.get("language") or "unknown"),
        "url": source_uri if source_uri.startswith(("http://", "https://")) else None,
        "project": doc_project,
        "retrieved_at": str(front.get("retrieved_at") or ""),
        "converted_at": str(front.get("converted_at") or ""),
        "embedding_model": getattr(embedder, "model_name", None),
    }

    if progress_cb is not None:
        progress_cb(0, 0, "chunking document")

    chunks = chunk_document(
        body,
        doc_id,
        doc_title,
        embedder.count_tokens,
        valid_until=valid_until,
        extraction_method=extraction_method,
    )

    if not chunks:
        logger.warning("No chunks produced for %s", name_for_log)
        _purge_doc(doc_id, vector_store, fts_db)
        return []

    # Validate all chunks before writing
    for chunk in chunks:
        _validate_chunk(chunk)

    # Embed (reusing unchanged chunks' vectors) and write to stores
    n_chunks = len(chunks)
    known = _reusable_vectors(chunks, config, embedder, vector_store, vector_cache)
    vectors: list[list[float] | None] = [known.get(i) for i in range(n_chunks)]
    todo = [i for i in range(n_chunks) if vectors[i] is None]
    embed_cb: EmbedProgressCb | None = None
    if progress_cb is not None:
        msg = f"embedding {len(todo)} chunks" + (f" ({len(known)} reused)" if known else "")
        progress_cb(0, len(todo), msg)
        embed_cb = lambda done, total: progress_cb(done, total, msg)  # noqa: E731
    if todo:
        fresh = embedder.encode([chunks[i].embedded_text for i in todo], progress_cb=embed_cb)
        for i, vector in zip(todo, fresh, strict=True):
            vectors[i] = vector
    if known:
        logger.info("Reused %d of %d vectors for %s", len(known), n_chunks, name_for_log)

    # Replace stale entries only now that chunking, validation and embedding
    # succeeded — a failure above leaves the previous index intact.
    _purge_doc(doc_id, vector_store, fts_db)

    if progress_cb is not None:
        progress_cb(n_chunks, n_chunks, "writing to index")
    vector_store.upsert(chunks, [v for v in vectors if v is not None], doc_meta)
    fts_db.upsert_batch(
        chunks, doc_project, retrieved_at=str(front.get("retrieved_at") or "") or None
    )

    # Update manifest chunk_count (filesystem mode only; DB mode caller updates via store)
    if doc_folder is not None:
        manifest_path = doc_folder / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["chunk_count"] = len(chunks)
            from pipeline.fs_layout import write_json_atomic

            write_json_atomic(manifest_path, manifest)

    logger.info("Indexed %s: %d chunks", name_for_log, len(chunks))
    return chunks
