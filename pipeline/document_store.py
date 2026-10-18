"""DocumentStore abstraction: filesystem (default) and SQLAlchemy database backends.

Usage
-----
Call ``make_document_store(config)`` to get the right implementation.  Both
backends satisfy the ``DocumentStore`` protocol so the rest of the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.config import Config

from pipeline.fs_layout import (
    candidate_folders,
    find_folder_by_doc_id,
    new_doc_folder,
    read_manifest,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# doc_id prefix lookup
# ---------------------------------------------------------------------------

_DOC_PREFIX_RE = re.compile(r"[0-9a-f]{8,64}")


class InvalidDocPrefixError(ValueError):
    """The supplied doc_id / prefix is not 8–64 lowercase hex characters."""


class AmbiguousDocPrefixError(LookupError):
    """The supplied prefix matches more than one document."""


def normalize_doc_prefix(prefix: str) -> str:
    """Return *prefix* lower-cased, or raise ``InvalidDocPrefixError``.

    Requiring hex keeps SQL ``LIKE`` wildcards and path tricks out, and the
    8-char minimum stops a short prefix from silently matching many documents.
    """
    candidate = prefix.strip().lower()
    if not _DOC_PREFIX_RE.fullmatch(candidate):
        raise InvalidDocPrefixError(
            f"Invalid document id {prefix!r}: expected 8-64 hexadecimal characters"
        )
    return candidate


def scan_folders_by_prefix(docs_dir: Path, prefix: str) -> Path | None:
    """Return the single document folder whose doc_id starts with *prefix*.

    Raises ``AmbiguousDocPrefixError`` if more than one document matches.
    """
    p = normalize_doc_prefix(prefix)
    matches: list[Path] = []
    for folder in candidate_folders(docs_dir, p):
        manifest = read_manifest(folder)
        if manifest is None:
            continue
        full_id = str(manifest.get("doc_id", ""))
        if full_id == p:
            return folder
        if full_id.startswith(p):
            matches.append(folder)
    if len(matches) > 1:
        raise AmbiguousDocPrefixError(f"Prefix {prefix!r} matches {len(matches)} documents")
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# ingested_sources mapping helper
# ---------------------------------------------------------------------------


def _to_ingested_source(manifest: dict[str, Any]) -> dict[str, Any]:
    """Map a KB manifest dict to an ingested_sources-compatible record.

    Field mapping to accounting-agent's ingested_sources table:
      source_id   ← doc_id
      title       ← title
      source_type ← doc_type
      legal_year  ← extracted from title / source_uri (20xx year), else ingest year
      language    ← language
      url         ← source_uri when it is an http(s) URL, else None
      file_path   ← source_file / source_uri when it is a file path, else None
      chunk_count ← chunk_count
      ingested_at ← retrieved_at
      uploaded_at ← retrieved_at for file sources, None for URL sources
      status      ← "NEW" (ok), "SUPERSEDED" (valid_until set),
                    "QUARANTINED" (review), "ERROR"
    """
    source_uri = str(manifest.get("source_uri") or "")
    source_file = str(manifest.get("source_file") or "")
    is_url = source_uri.startswith(("http://", "https://"))
    title = str(manifest.get("title") or "")
    retrieved_at = str(manifest.get("retrieved_at") or "")

    # Extract legal year: title first, then URI, then ingest year
    legal_year = 0
    for text in (title, source_uri):
        m = re.search(r"\b(20\d{2})\b", text)
        if m:
            legal_year = int(m.group(1))
            break
    if not legal_year:
        try:
            legal_year = int(retrieved_at[:4])
        except (ValueError, IndexError):
            legal_year = datetime.now(UTC).year

    kb_status = str(manifest.get("status") or "ok")
    valid_until = manifest.get("valid_until")
    if valid_until:
        status = "SUPERSEDED"
    elif kb_status == "ok":
        status = "NEW"
    elif kb_status == "review":
        status = "QUARANTINED"
    else:
        status = "ERROR"

    url: str | None = source_uri[:1024] if is_url else None
    file_path: str | None = None
    if not is_url:
        fp = source_file or source_uri
        file_path = fp[:512] if fp else None

    return {
        "source_id": str(manifest.get("doc_id") or ""),
        "title": title[:512],
        "source_type": str(manifest.get("doc_type") or "generic"),
        "legal_year": legal_year,
        "language": str(manifest.get("language") or "unknown"),
        "url": url,
        "file_path": file_path,
        "chunk_count": int(manifest.get("chunk_count") or 0),
        "ingested_at": retrieved_at,
        "uploaded_at": retrieved_at if not is_url else None,
        "status": status,
    }


_INGESTED_SOURCES_FIELDS = [
    "source_id",
    "title",
    "source_type",
    "legal_year",
    "language",
    "url",
    "file_path",
    "chunk_count",
    "ingested_at",
    "uploaded_at",
    "status",
]


def _sources_to_csv(rows: list[dict[str, Any]]) -> str:
    """Serialise a list of ingested_sources records to CSV (Excel-compatible UTF-8 BOM)."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=_INGESTED_SOURCES_FIELDS,
        extrasaction="ignore",
        lineterminator="\r\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class DocumentStore:
    """Interface that both storage backends must implement.

    Using a regular class instead of typing.Protocol so concrete subclasses
    can be detected with isinstance() and so mypy --strict is satisfied without
    runtime_checkable overhead.  Each method raises NotImplementedError.
    """

    def sha256_exists(self, sha256: str) -> bool:
        raise NotImplementedError

    def save(
        self,
        doc_id: str,
        front_matter: dict[str, Any],
        body: str,
        source_bytes: bytes,
        source_ext: str,
        *,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        raise NotImplementedError

    def load_manifest(self, doc_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def load_body(self, doc_id: str) -> str:
        raise NotImplementedError

    def load_source_blob(self, doc_id: str) -> tuple[bytes, str]:
        """Return (source_bytes, original_filename) for a document."""
        raise NotImplementedError

    def update_manifest(self, doc_id: str, **fields: Any) -> None:
        raise NotImplementedError

    def append_event(self, doc_id: str, event: str, detail: str = "") -> None:
        raise NotImplementedError

    def find_by_prefix(self, prefix: str) -> str | None:
        """Return the full doc_id whose SHA-256 starts with *prefix*, or None.

        Raises ``InvalidDocPrefixError`` for non-hex/short input and
        ``AmbiguousDocPrefixError`` when several documents match.
        """
        raise NotImplementedError

    def find_by_source_uri(self, uri: str) -> str | None:
        """Return the currently-valid doc_id for *uri* (valid_until is null), or None."""
        raise NotImplementedError

    def list_all(self) -> list[dict[str, Any]]:
        """Return all manifest dicts (including events list) sorted newest-first."""
        raise NotImplementedError

    def find_superseding(self, doc_id: str) -> str | None:
        """Return the doc_id of the version that supersedes *doc_id*, if any."""
        for m in self.list_all():
            if m.get("supersedes") == doc_id:
                return str(m["doc_id"])
        return None

    def delete(self, doc_id: str) -> None:
        raise NotImplementedError

    def export_all(self) -> list[dict[str, Any]]:
        """Return all documents with manifest fields + body + events, newest-first."""
        raise NotImplementedError

    def export_as_ingested_sources(self) -> list[dict[str, Any]]:
        """Return all documents mapped to ingested_sources table format, newest-first."""
        return [_to_ingested_source(m) for m in self.list_all()]


# ---------------------------------------------------------------------------
# Filesystem implementation
# ---------------------------------------------------------------------------


_FRONT_MATTER_MIRRORED = frozenset({"valid_until", "project"})


class FilesystemDocumentStore(DocumentStore):
    """Stores documents as document.md + manifest.json + source.* in KB_ROOT/documents/."""

    def __init__(self, config: Config) -> None:
        self._config = config

    # --- dedup ---

    def sha256_exists(self, sha256: str) -> bool:
        return find_folder_by_doc_id(self._config.documents_dir, sha256) is not None

    # --- write ---

    def save(
        self,
        doc_id: str,
        front_matter: dict[str, Any],
        body: str,
        source_bytes: bytes,
        source_ext: str,
        *,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        from pipeline.acquisition import slugify
        from pipeline.normalize import assemble_document

        slug = slugify(str(front_matter["title"]))
        folder = new_doc_folder(self._config.documents_dir, doc_id, slug)
        folder.mkdir(parents=True, exist_ok=True)

        (folder / "document.md").write_text(assemble_document(front_matter, body), encoding="utf-8")
        (folder / f"source{source_ext}").write_bytes(source_bytes)

        manifest: dict[str, Any] = {
            **front_matter,
            "status": status,
            "error": error,
            "chunk_count": 0,
        }
        write_json_atomic(folder / "manifest.json", manifest)

    # --- read ---

    def _find_folder(self, doc_id: str, *, prefix: bool = False) -> Path | None:
        if prefix:
            return scan_folders_by_prefix(self._config.documents_dir, doc_id)
        return find_folder_by_doc_id(self._config.documents_dir, doc_id)

    def load_manifest(self, doc_id: str) -> dict[str, Any]:
        folder = self._find_folder(doc_id)
        if folder is None:
            raise KeyError(f"Document not found: {doc_id[:8]}")
        return dict(json.loads((folder / "manifest.json").read_text(encoding="utf-8")))

    def load_body(self, doc_id: str) -> str:
        from pipeline.normalize import strip_front_matter

        folder = self._find_folder(doc_id)
        if folder is None:
            raise KeyError(f"Document not found: {doc_id[:8]}")
        _, body = strip_front_matter((folder / "document.md").read_text(encoding="utf-8"))
        return body

    def load_source_blob(self, doc_id: str) -> tuple[bytes, str]:
        folder = self._find_folder(doc_id)
        if folder is None:
            raise KeyError(f"Document not found: {doc_id[:8]}")

        manifest = dict(json.loads((folder / "manifest.json").read_text(encoding="utf-8")))
        source_candidates = sorted(
            p for p in folder.iterdir() if p.is_file() and p.name.startswith("source")
        )
        if not source_candidates:
            raise KeyError(f"Source snapshot not found: {doc_id[:8]}")

        source_path = source_candidates[0]
        source_name = Path(str(manifest.get("source_file") or "")).name
        if not source_name:
            suffix = source_path.suffix or ".bin"
            source_name = f"{doc_id[:8]}{suffix}"
        return source_path.read_bytes(), source_name

    def update_manifest(self, doc_id: str, **fields: Any) -> None:
        folder = self._find_folder(doc_id)
        if folder is None:
            return
        mf = folder / "manifest.json"
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        manifest.update(fields)
        write_json_atomic(mf, manifest)

        # Mirror front-matter fields into document.md (the indexer reads them there)
        mirrored = {k: v for k, v in fields.items() if k in _FRONT_MATTER_MIRRORED}
        if mirrored:
            from pipeline.normalize import assemble_document, strip_front_matter

            doc_md_path = folder / "document.md"
            front, body = strip_front_matter(doc_md_path.read_text(encoding="utf-8"))
            front.update(mirrored)
            doc_md_path.write_text(assemble_document(front, body), encoding="utf-8")

    def append_event(self, doc_id: str, event: str, detail: str = "") -> None:
        folder = self._find_folder(doc_id)
        if folder is None:
            return
        from pipeline.events import append_event as _append

        _append(folder / "manifest.json", event, detail)

    def find_by_prefix(self, prefix: str) -> str | None:
        folder = scan_folders_by_prefix(self._config.documents_dir, prefix)
        if folder is None:
            return None
        try:
            return str(json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["doc_id"])
        except Exception:
            return None

    def find_by_source_uri(self, uri: str) -> str | None:
        docs_dir = self._config.documents_dir
        if not docs_dir.exists():
            return None
        for folder in docs_dir.iterdir():
            mf = folder / "manifest.json"
            if not mf.exists():
                continue
            try:
                data: dict[str, Any] = json.loads(mf.read_text(encoding="utf-8"))
                if data.get("source_uri") == uri and data.get("valid_until") is None:
                    return str(data["doc_id"])
            except Exception:
                pass
        return None

    def list_all(self) -> list[dict[str, Any]]:
        docs_dir = self._config.documents_dir
        if not docs_dir.exists():
            return []
        manifests = []
        for folder in docs_dir.iterdir():
            mf = folder / "manifest.json"
            if mf.exists():
                try:
                    data = dict(json.loads(mf.read_text(encoding="utf-8")))
                    data["_folder"] = str(folder)
                    manifests.append(data)
                except Exception:
                    pass
        manifests.sort(key=lambda m: str(m.get("retrieved_at", "")), reverse=True)
        return manifests

    def delete(self, doc_id: str) -> None:
        import shutil

        folder = self._find_folder(doc_id)
        if folder is not None:
            shutil.rmtree(folder)

    def export_all(self) -> list[dict[str, Any]]:
        result = []
        for m in self.list_all():
            doc_id = str(m.get("doc_id", ""))
            export_m = {k: v for k, v in m.items() if k != "_folder"}
            try:
                export_m["body"] = self.load_body(doc_id)
            except Exception:
                export_m["body"] = ""
            events = export_m.get("events")
            if not isinstance(events, list):
                events = []
            export_m["events"] = events
            result.append(export_m)
        return result


# ---------------------------------------------------------------------------
# Database implementation
# ---------------------------------------------------------------------------


@contextmanager
def _session_scope(session_factory: Any) -> Iterator[Any]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Optional manifest fields stored in KbDocument.extra (JSON) in database mode.
_EXTRA_FIELDS = frozenset({"meta_author", "meta_published_date", "meta_publication_venue"})


def _fit(value: Any, limit: int) -> str:
    """Stringify and truncate to a bounded column width (PostgreSQL rejects overflow)."""
    text = "" if value is None else str(value)
    if len(text) > limit:
        logger.warning("Truncating %d-char value to %d chars for storage", len(text), limit)
    return text[:limit]


def _load_extra(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return {k: v for k, v in data.items() if k in _EXTRA_FIELDS} if isinstance(data, dict) else {}


class DatabaseDocumentStore(DocumentStore):
    """Stores documents in kb_documents + kb_document_events (SQLAlchemy sync)."""

    def __init__(self, engine: Any) -> None:
        from sqlalchemy.orm import sessionmaker

        from pipeline.db_models import Base

        Base.metadata.create_all(engine)
        self._engine = engine
        self._factory = sessionmaker(engine)
        self._ensure_columns()

    @property
    def session_factory(self) -> Any:
        """The sessionmaker behind this store (also the ledger's in database mode)."""
        return self._factory

    def _ensure_columns(self) -> None:
        """Add columns introduced after a database was created (idempotent).

        ``create_all`` never ALTERs an existing table, so older databases lack
        ``project`` (per-project isolation) and ``extra`` (optional fields).
        """
        from sqlalchemy import inspect, text

        from pipeline.ledger import ensure_ledger_columns

        ensure_ledger_columns(self._engine)
        try:
            cols = {c["name"] for c in inspect(self._engine).get_columns("kb_documents")}
        except Exception:
            return
        for name, ddl in (("project", "VARCHAR(64)"), ("extra", "TEXT")):
            if name in cols:
                continue
            try:
                with self._engine.begin() as con:
                    con.execute(text(f"ALTER TABLE kb_documents ADD COLUMN {name} {ddl}"))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not add kb_documents.%s column: %s", name, exc)

    # --- dedup ---

    def sha256_exists(self, sha256: str) -> bool:
        from sqlalchemy import select

        from pipeline.db_models import KbDocument

        with _session_scope(self._factory) as s:
            row = s.execute(
                select(KbDocument.doc_id).where(KbDocument.source_sha256 == sha256)
            ).first()
            return row is not None

    # --- write ---

    def save(
        self,
        doc_id: str,
        front_matter: dict[str, Any],
        body: str,
        source_bytes: bytes,
        source_ext: str,
        *,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        from pipeline.db_models import KbDocument

        fm = front_matter
        extra = {k: fm[k] for k in _EXTRA_FIELDS if fm.get(k) is not None}
        with _session_scope(self._factory) as s:
            doc = KbDocument(
                doc_id=doc_id,
                title=_fit(fm.get("title", ""), 512),
                source_file=_fit(fm.get("source_file", ""), 512),
                source_uri=str(fm.get("source_uri", "")),
                source_version=_fit(fm.get("source_version"), 255) or None,
                source_sha256=str(fm.get("source_sha256", doc_id)),
                retrieved_at=_fit(fm.get("retrieved_at", ""), 40),
                converted_at=_fit(fm.get("converted_at", ""), 40),
                converter=_fit(fm.get("converter", ""), 255),
                extraction_method=_fit(fm.get("extraction_method", "deterministic"), 32),
                language=_fit(fm.get("language", "unknown"), 8),
                doc_type=_fit(fm.get("doc_type", "generic"), 64),
                supersedes=fm.get("supersedes") or None,
                valid_until=fm.get("valid_until") or None,
                status=_fit(status, 16),
                error=error,
                chunk_count=0,
                project=_fit(fm.get("project"), 64) or None,
                extra=json.dumps(extra, ensure_ascii=False) if extra else None,
                body=body,
                source_bytes=source_bytes,
                source_ext=_fit(source_ext, 16) or None,
            )
            s.add(doc)

    # --- read ---

    def load_manifest(self, doc_id: str) -> dict[str, Any]:
        from sqlalchemy import select

        from pipeline.db_models import KbDocument, KbDocumentEvent

        with _session_scope(self._factory) as s:
            doc = s.execute(
                select(KbDocument).where(KbDocument.doc_id == doc_id)
            ).scalar_one_or_none()
            if doc is None:
                raise KeyError(doc_id)
            m = self._to_dict(doc)
            events = (
                s.execute(
                    select(KbDocumentEvent)
                    .where(KbDocumentEvent.doc_id == doc_id)
                    .order_by(KbDocumentEvent.at)
                )
                .scalars()
                .all()
            )
            m["events"] = [{"event": e.event, "at": e.at, "detail": e.detail} for e in events]
            return m

    def load_body(self, doc_id: str) -> str:
        from sqlalchemy import select

        from pipeline.db_models import KbDocument

        with _session_scope(self._factory) as s:
            body = s.execute(
                select(KbDocument.body).where(KbDocument.doc_id == doc_id)
            ).scalar_one_or_none()
            if body is None:
                raise KeyError(doc_id)
            return str(body)

    def load_source_blob(self, doc_id: str) -> tuple[bytes, str]:
        from sqlalchemy import select

        from pipeline.db_models import KbDocument

        with _session_scope(self._factory) as s:
            row = s.execute(
                select(
                    KbDocument.source_bytes,
                    KbDocument.source_file,
                    KbDocument.source_ext,
                ).where(KbDocument.doc_id == doc_id)
            ).first()
            if row is None:
                raise KeyError(f"Document not found: {doc_id[:8]}")
            source_bytes, source_file, source_ext = row
            if source_bytes is None:
                raise KeyError(f"Source snapshot not found: {doc_id[:8]}")
            source_name = Path(str(source_file or "")).name
            if not source_name:
                source_name = f"{doc_id[:8]}{str(source_ext or '.bin')}"
            return bytes(source_bytes), source_name

    def update_manifest(self, doc_id: str, **fields: Any) -> None:
        from sqlalchemy import select, update

        from pipeline.db_models import KbDocument

        columns = {k: v for k, v in fields.items() if k not in _EXTRA_FIELDS}
        extra_updates = {k: v for k, v in fields.items() if k in _EXTRA_FIELDS}
        with _session_scope(self._factory) as s:
            if extra_updates:
                current = s.execute(
                    select(KbDocument.extra).where(KbDocument.doc_id == doc_id)
                ).scalar_one_or_none()
                merged = {**_load_extra(current), **extra_updates}
                columns["extra"] = json.dumps(merged, ensure_ascii=False)
            if columns:
                s.execute(update(KbDocument).where(KbDocument.doc_id == doc_id).values(**columns))

    def append_event(self, doc_id: str, event: str, detail: str = "") -> None:
        from pipeline.db_models import KbDocumentEvent

        with _session_scope(self._factory) as s:
            s.add(
                KbDocumentEvent(
                    doc_id=doc_id,
                    event=event,
                    at=datetime.now(UTC).isoformat(),
                    detail=detail,
                )
            )

    def find_by_prefix(self, prefix: str) -> str | None:
        from sqlalchemy import select

        from pipeline.db_models import KbDocument

        p = normalize_doc_prefix(prefix)
        with _session_scope(self._factory) as s:
            exact = s.execute(select(KbDocument.doc_id).where(KbDocument.doc_id == p)).first()
            if exact:
                return str(exact[0])
            rows = s.execute(
                select(KbDocument.doc_id)
                .where(KbDocument.doc_id.startswith(p, autoescape=True))
                .limit(2)
            ).all()
            if len(rows) > 1:
                raise AmbiguousDocPrefixError(f"Prefix {prefix!r} matches more than one document")
            return str(rows[0][0]) if rows else None

    def find_by_source_uri(self, uri: str) -> str | None:
        from sqlalchemy import select

        from pipeline.db_models import KbDocument

        with _session_scope(self._factory) as s:
            row = s.execute(
                select(KbDocument.doc_id).where(
                    KbDocument.source_uri == uri,
                    KbDocument.valid_until.is_(None),
                )
            ).first()
            return str(row[0]) if row else None

    @staticmethod
    def _events_by_doc(s: Any, doc_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Load events for many documents with one query per 500 ids (was N+1)."""
        from sqlalchemy import select

        from pipeline.db_models import KbDocumentEvent

        out: dict[str, list[dict[str, Any]]] = {d: [] for d in doc_ids}
        for start in range(0, len(doc_ids), 500):
            part = doc_ids[start : start + 500]
            rows = s.execute(
                select(KbDocumentEvent)
                .where(KbDocumentEvent.doc_id.in_(part))
                .order_by(KbDocumentEvent.at, KbDocumentEvent.id)
            ).scalars()
            for e in rows:
                out[e.doc_id].append({"event": e.event, "at": e.at, "detail": e.detail})
        return out

    def list_all(self) -> list[dict[str, Any]]:
        from sqlalchemy import select
        from sqlalchemy.orm import defer

        from pipeline.db_models import KbDocument

        with _session_scope(self._factory) as s:
            # Never load bodies/source blobs for a listing.
            docs = (
                s.execute(
                    select(KbDocument)
                    .options(defer(KbDocument.body), defer(KbDocument.source_bytes))
                    .order_by(KbDocument.retrieved_at.desc())
                )
                .scalars()
                .all()
            )
            events = self._events_by_doc(s, [d.doc_id for d in docs])
            result = []
            for doc in docs:
                m = self._to_dict(doc)
                m["events"] = events[doc.doc_id]
                result.append(m)
            return result

    def find_superseding(self, doc_id: str) -> str | None:
        from sqlalchemy import select

        from pipeline.db_models import KbDocument

        with _session_scope(self._factory) as s:
            row = s.execute(
                select(KbDocument.doc_id).where(KbDocument.supersedes == doc_id).limit(1)
            ).first()
            return str(row[0]) if row else None

    def delete(self, doc_id: str) -> None:
        from sqlalchemy import delete

        from pipeline.db_models import KbDocument

        with _session_scope(self._factory) as s:
            s.execute(delete(KbDocument).where(KbDocument.doc_id == doc_id))

    def export_all(self) -> list[dict[str, Any]]:
        from sqlalchemy import select
        from sqlalchemy.orm import defer

        from pipeline.db_models import KbDocument

        with _session_scope(self._factory) as s:
            docs = (
                s.execute(
                    select(KbDocument)
                    .options(defer(KbDocument.source_bytes))
                    .order_by(KbDocument.retrieved_at.desc())
                )
                .scalars()
                .all()
            )
            events = self._events_by_doc(s, [d.doc_id for d in docs])
            result = []
            for doc in docs:
                m = self._to_dict(doc)
                m["body"] = doc.body
                m["events"] = events[doc.doc_id]
                result.append(m)
            return result

    @staticmethod
    def _to_dict(doc: Any) -> dict[str, Any]:
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source_file": doc.source_file,
            "source_uri": doc.source_uri,
            "source_version": doc.source_version,
            "source_sha256": doc.source_sha256,
            "retrieved_at": doc.retrieved_at,
            "converted_at": doc.converted_at,
            "converter": doc.converter,
            "extraction_method": doc.extraction_method,
            "language": doc.language,
            "doc_type": doc.doc_type,
            "supersedes": doc.supersedes,
            "valid_until": doc.valid_until,
            "status": doc.status,
            "error": doc.error,
            "chunk_count": doc.chunk_count,
            "project": doc.project,
            **_load_extra(getattr(doc, "extra", None)),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_database_url(config: Config) -> str:
    """Build a SQLAlchemy database URL from individual config fields."""
    import urllib.parse

    host = config.database_host
    user = config.database_user
    password = config.database_password
    if not host or not user or not password:
        raise RuntimeError(
            "KB_STORAGE_BACKEND=database requires PostgreSQL settings "
            "(KB_DATABASE_HOST, KB_DATABASE_USER, KB_DATABASE_PASSWORD)."
        )

    port = config.database_port
    name = config.database_name
    creds = f"{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(password, safe='')}@"
    return f"postgresql+psycopg2://{creds}{host}:{port}/{name}"


def make_document_store(config: Config) -> DocumentStore:
    """Return the configured DocumentStore implementation."""
    if config.storage_backend == "database":
        from sqlalchemy import create_engine
        from sqlalchemy.exc import NoSuchModuleError

        url = _build_database_url(config)
        config.index_dir.mkdir(parents=True, exist_ok=True)
        try:
            engine = create_engine(url, echo=False)
        except ModuleNotFoundError as exc:
            if url.startswith("postgresql+") and exc.name == "psycopg2":
                raise RuntimeError(
                    "PostgreSQL backend selected but psycopg2 is not installed. "
                    "Install the postgres extra (e.g. `uv sync --extra postgres`) or "
                    "rebuild the pipeline image with INSTALL_POSTGRES_EXTRA=1 (the default)."
                ) from exc
            raise
        except NoSuchModuleError as exc:
            if url.startswith("postgresql+"):
                raise RuntimeError(
                    "PostgreSQL backend selected but the SQLAlchemy PostgreSQL driver "
                    "is unavailable. "
                    "Install the postgres extra or rebuild the Docker image with postgres support."
                ) from exc
            raise
        logger.info("DocumentStore: database backend (%s)", url.split("://")[0])
        return DatabaseDocumentStore(engine)

    logger.debug("DocumentStore: filesystem backend")
    return FilesystemDocumentStore(config)
