"""SQLAlchemy ORM models for the optional database storage backend.

Tables:
  kb_documents       — one row per ingested document (mirrors manifest.json + body)
  kb_document_events — append-only lifecycle events (mirrors the events[] array)
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KbDocument(Base):
    __tablename__ = "kb_documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    source_file: Mapped[str] = mapped_column(String(512))
    source_uri: Mapped[str] = mapped_column(Text)
    source_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_sha256: Mapped[str] = mapped_column(String(64), index=True)
    retrieved_at: Mapped[str] = mapped_column(String(40))
    converted_at: Mapped[str] = mapped_column(String(40))
    converter: Mapped[str] = mapped_column(String(255))
    extraction_method: Mapped[str] = mapped_column(String(32))
    language: Mapped[str] = mapped_column(String(8))
    doc_type: Mapped[str] = mapped_column(String(64))
    supersedes: Mapped[str | None] = mapped_column(String(64), nullable=True)
    valid_until: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    # Project (Qdrant collection) that owns this document. Enforces per-project
    # isolation: list/search/delete/export are all scoped by this slug.
    project: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # JSON object of optional manifest fields without a dedicated column
    # (e.g. article meta_author / meta_published_date / meta_publication_venue).
    extra: Mapped[str | None] = mapped_column(Text, nullable=True)
    body: Mapped[str] = mapped_column(Text)
    source_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    source_ext: Mapped[str | None] = mapped_column(String(16), nullable=True)


class KbProject(Base):
    """A project — an isolated document collection backed by a Qdrant collection.

    In database mode this table is the single source of truth for projects,
    replacing the legacy ``KB_ROOT/projects.json`` registry.
    """

    __tablename__ = "kb_projects"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    # Unique for new databases; create()/update_collection_name() also check in code.
    collection_name: Mapped[str] = mapped_column(String(64), index=True, unique=True)
    created_at: Mapped[str] = mapped_column(String(40))
    description: Mapped[str] = mapped_column(Text, default="")


class KbDocumentEvent(Base):
    __tablename__ = "kb_document_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("kb_documents.doc_id", ondelete="CASCADE"),
        index=True,
    )
    event: Mapped[str] = mapped_column(String(64))
    at: Mapped[str] = mapped_column(String(40))
    detail: Mapped[str] = mapped_column(Text, default="")


class KbJob(Base):
    __tablename__ = "kb_jobs"

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    finished_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    params_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


# The Jobs page lists the newest jobs first. platform/postgres/kb/02-migrate-kb-jobs-index.sql
# adds the same index to databases whose kb_jobs predates it (create_all skips existing tables).
Index("ix_kb_jobs_created_at_status", KbJob.created_at.desc(), KbJob.status)


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[str] = mapped_column(String(40), index=True)


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    turn_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    occurred_at: Mapped[str] = mapped_column(String(40), index=True)
    model: Mapped[str] = mapped_column(String(255), index=True)
    purpose: Mapped[str] = mapped_column(String(64), index=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    # Added Oct 2026 (back-filled by pipeline.ledger.ensure_ledger_columns): the gateway's
    # currency and request id; cost_source 'gateway' | 'unpriced' (NULL: older kb estimate).
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    cost_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)



class AiInteraction(Base):
    """Full per-call record of a single LLM interaction.

    Complements the aggregated ``usage_events`` ledger by storing the request
    prompts, response text, timing, finish reason, and error details for each
    individual model call. Writes are best-effort (never block user flows).
    """

    __tablename__ = "ai_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(255), index=True)
    purpose: Mapped[str] = mapped_column(String(64), index=True)
    model_role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="ok", index=True)
    # Correlation
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    turn_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    doc_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Request / response payloads (may be truncated before storage)
    request_system: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_user: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Request parameters
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Accounting
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    # Diagnostics
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Added Oct 2026 (see UsageEvent): what the gateway reported for the call.
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    cost_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_served: Mapped[str | None] = mapped_column(String(255), nullable=True)

