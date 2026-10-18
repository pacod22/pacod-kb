"""The ledger: runtime settings, prompt overrides, LLM usage and AI interactions.

Both storage backends keep it, so a setting saved in the dashboard or a usage row
is never silently dropped:

* database mode — the document database itself (the store's session factory);
* filesystem mode — ``{index_dir}/ledger.db``, a small SQLite file (WAL) holding only
  the ``app_settings``, ``usage_events`` and ``ai_interactions`` tables.

Columns added to these tables after a database was created are back-filled by
:func:`ensure_ledger_columns` (``create_all`` never ALTERs an existing table).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pipeline.config import Config
from pipeline.document_store import DatabaseDocumentStore, DocumentStore

logger = logging.getLogger(__name__)

LEDGER_TABLES = ("app_settings", "usage_events", "ai_interactions")

# (table, column, DDL type) added after the first release of each table.
_LATE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("usage_events", "currency", "VARCHAR(8)"),
    ("usage_events", "cost_source", "VARCHAR(16)"),
    ("usage_events", "request_id", "VARCHAR(64)"),
    ("ai_interactions", "currency", "VARCHAR(8)"),
    ("ai_interactions", "cost_source", "VARCHAR(16)"),
    ("ai_interactions", "request_id", "VARCHAR(64)"),
    ("ai_interactions", "model_served", "VARCHAR(255)"),
)

_lock = threading.Lock()
_factories: dict[str, Any] = {}


def ledger_session_factory(config: Config, store: DocumentStore | None = None) -> Any:
    """The session factory the ledger lives behind, for this configuration."""
    if isinstance(store, DatabaseDocumentStore):
        return store.session_factory
    if config.storage_backend == "database":
        return _database_ledger(config)
    return _filesystem_ledger(config)


def ensure_ledger_columns(engine: Any) -> None:
    """Add the ledger columns introduced after a database was created (idempotent)."""
    from sqlalchemy import inspect, text

    try:
        inspector = inspect(engine)
        existing = {t: {c["name"] for c in inspector.get_columns(t)} for t in LEDGER_TABLES}
    except Exception:
        return
    for table, column, ddl in _LATE_COLUMNS:
        if column in existing.get(table, set()):
            continue
        try:
            with engine.begin() as con:
                con.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not add %s.%s column: %s", table, column, exc)


def _database_ledger(config: Config) -> Any:
    from pipeline.document_store import make_document_store

    key = f"db:{config.database_host}:{config.database_port}/{config.database_name}"
    with _lock:
        factory = _factories.get(key)
        if factory is None:
            store = make_document_store(config)
            assert isinstance(store, DatabaseDocumentStore)
            factory = store.session_factory
            _factories[key] = factory
        return factory


def _filesystem_ledger(config: Config) -> Any:
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    from pipeline.db_models import Base

    path = config.index_dir / "ledger.db"
    key = f"fs:{path.resolve()}"
    with _lock:
        factory = _factories.get(key)
        if factory is not None:
            return factory
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path}", echo=False)

        @event.listens_for(engine, "connect")
        def _wal(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

        tables = [Base.metadata.tables[name] for name in LEDGER_TABLES]
        Base.metadata.create_all(engine, tables=tables)
        ensure_ledger_columns(engine)
        factory = sessionmaker(engine)
        _factories[key] = factory
        return factory
