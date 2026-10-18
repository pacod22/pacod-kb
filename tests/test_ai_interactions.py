from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.llm_management import log_ai_interaction


@pytest.fixture()
def session_factory(tmp_path: Path) -> Any:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from pipeline.db_models import Base

    engine = create_engine(f"sqlite:///{tmp_path}/ai_interactions.db")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _all_interactions(session_factory: Any) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from pipeline.db_models import AiInteraction
    from pipeline.document_store import _session_scope

    with _session_scope(session_factory) as s:
        rows = s.execute(select(AiInteraction).order_by(AiInteraction.id)).scalars().all()
        return [
            {
                "provider": r.provider,
                "purpose": r.purpose,
                "status": r.status,
                "cost_source": r.cost_source,
                "request_system": r.request_system,
                "request_user": r.request_user,
                "response_text": r.response_text,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cached_tokens": r.cached_tokens,
                "duration_ms": r.duration_ms,
                "cost_usd": r.cost_usd,
                "error": r.error,
            }
            for r in rows
        ]


def test_log_ai_interaction_is_noop_without_session_factory() -> None:
    # Must not raise when DB is unavailable (filesystem mode).
    log_ai_interaction(
        session_factory=None,
        provider="anthropic",
        model="anthropic/claude-haiku-4-5",
        purpose="content_security",
    )


def test_log_ai_interaction_persists_full_record(session_factory: Any) -> None:
    log_ai_interaction(
        session_factory=session_factory,
        provider="anthropic",
        model="anthropic/claude-opus-4-8",
        purpose="document_conversion",
        status="ok",
        model_role="primary",
        turn_id="doc-123",
        doc_id="doc-123",
        request_system="You are a converter.",
        request_user="Convert this.",
        response_text="# Converted",
        temperature=0.5,
        max_tokens=8192,
        input_tokens=100,
        output_tokens=50,
        cached_tokens=10,
        duration_ms=1234,
    )

    rows = _all_interactions(session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "anthropic"
    assert row["purpose"] == "document_conversion"
    assert row["status"] == "ok"
    assert row["request_system"] == "You are a converter."
    assert row["response_text"] == "# Converted"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50
    assert row["cached_tokens"] == 10
    assert row["duration_ms"] == 1234
    # No gateway price given: stored as 0, marked unpriced (kb no longer estimates).
    assert row["cost_usd"] == 0.0
    assert row["cost_source"] == "unpriced"


def test_log_ai_interaction_truncates_long_payloads(session_factory: Any) -> None:
    huge = "x" * 60_000
    log_ai_interaction(
        session_factory=session_factory,
        provider="openai",
        model="openai/gpt-5.5",
        purpose="document_conversion",
        request_user=huge,
        response_text=huge,
    )
    row = _all_interactions(session_factory)[0]
    assert row["request_user"] is not None
    assert len(row["request_user"]) < len(huge)
    assert row["request_user"].endswith("…[truncated]")


def test_log_ai_interaction_records_errors(session_factory: Any) -> None:
    log_ai_interaction(
        session_factory=session_factory,
        provider="gemini",
        model="gemini/gemini-3.5-flash",
        purpose="price_fetch",
        status="error",
        error="boom",
    )
    row = _all_interactions(session_factory)[0]
    assert row["status"] == "error"
    assert row["error"] == "boom"
    assert row["provider"] == "gemini"



