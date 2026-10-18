"""Usage accounting: rows carry what the gateway reported."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.llm_management import (
    log_ai_interaction,
    provider_of,
    record_usage_event,
    to_usage_payload,
)


@pytest.fixture()
def session_factory(tmp_path: Path) -> Any:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from pipeline.db_models import Base

    engine = create_engine(f"sqlite:///{tmp_path}/llm_management.db")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _rows(session_factory: Any, model: Any) -> list[Any]:
    from sqlalchemy import select

    from pipeline.document_store import _session_scope

    with _session_scope(session_factory) as s:
        rows = list(s.execute(select(model)).scalars().all())
        s.expunge_all()
        return rows


def test_provider_of_prefixed_bare_and_pool_ids() -> None:
    assert provider_of("anthropic/claude-opus-4-8") == "anthropic"
    assert provider_of("gpt-4o") == "openai"
    assert provider_of("gemini-1.5-pro") == "gemini"
    assert provider_of("deepseek-chat") == "deepseek"
    assert provider_of("glm-4.5") == "glm"
    assert provider_of("@fast") == "pool"


def test_usage_event_keeps_the_gateway_price_and_request(session_factory: Any) -> None:
    from pipeline.db_models import UsageEvent

    common: dict[str, Any] = {
        "session_factory": session_factory,
        "user_id": "usr_1",
        "session_id": None,
        "turn_id": "doc",
        "model": "anthropic/claude-test",
        "purpose": "document_conversion",
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_tokens": 0,
    }
    record_usage_event(**common, cost_usd=0.25, currency="EUR", request_id="req_1")
    record_usage_event(**common, cost_usd=None)
    priced, unpriced = _rows(session_factory, UsageEvent)
    assert (priced.cost_usd, priced.currency, priced.cost_source, priced.request_id) == (
        0.25,
        "EUR",
        "gateway",
        "req_1",
    )
    assert (unpriced.cost_usd, unpriced.currency, unpriced.cost_source) == (0.0, None, "unpriced")


def test_interaction_keeps_the_gateway_fields(session_factory: Any) -> None:
    from pipeline.db_models import AiInteraction

    log_ai_interaction(
        session_factory=session_factory,
        provider="anthropic/claude-test",
        model="anthropic/claude-test",
        purpose="content_security",
        cost_usd=0.001,
        currency="USD",
        request_id="req_2",
        model_served="anthropic/claude-haiku-4-5",
    )
    (row,) = _rows(session_factory, AiInteraction)
    assert (row.request_id, row.model_served, row.cost_source) == (
        "req_2",
        "anthropic/claude-haiku-4-5",
        "gateway",
    )


def test_usage_payload_keeps_cost_and_ids() -> None:
    payload = to_usage_payload(
        {
            "input_tokens": 3,
            "output_tokens": 4,
            "cost": 0.5,
            "currency": None,
            "request_id": "req_3",
            "model_served": None,
        }
    )
    assert payload == {
        "input_tokens": 3,
        "output_tokens": 4,
        "cached_tokens": 0,
        "cost": 0.5,
        "currency": "USD",
        "request_id": "req_3",
    }
    assert "cost" not in to_usage_payload({"input_tokens": 1, "cost": None})
