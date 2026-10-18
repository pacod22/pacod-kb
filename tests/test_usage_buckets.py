"""Usage time buckets are portable SQL (PostgreSQL has no strftime)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from pipeline import api  # routers import pipeline.api; load it first
from pipeline.api_routes.usage import _bucket_label, _usage_bucket_expr
from pipeline.db_models import UsageEvent
from pipeline.ledger import ledger_session_factory
from pipeline.llm_management import record_usage_event


@pytest.mark.parametrize("bucket", ["hour", "day", "week", "month"])
def test_bucket_sql_is_substr_on_postgresql(bucket: str) -> None:
    sql = str(_usage_bucket_expr(UsageEvent, bucket).compile(dialect=postgresql.dialect()))
    assert "substr" in sql.lower()
    assert "strftime" not in sql.lower()


@pytest.mark.parametrize(
    ("raw", "bucket", "label"),
    [
        ("2026-10-14T09", "hour", "2026-10-14T09:00:00Z"),
        ("2026-10-14", "day", "2026-10-14"),
        ("2026-10-14", "week", "2026-W42"),
        ("2026-01-01", "week", "2026-W01"),
        ("2026-10", "month", "2026-10"),
        ("", "week", ""),
    ],
)
def test_bucket_labels(raw: str, bucket: str, label: str) -> None:
    assert _bucket_label(raw, bucket) == label


@pytest.fixture()
def client(config: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    factory = ledger_session_factory(config)
    for at, tokens in (
        ("2026-10-12T08:15:00+00:00", 10),  # Monday, ISO week 42
        ("2026-10-14T09:30:00+00:00", 20),  # Wednesday, week 42
        ("2026-10-14T09:45:00+00:00", 30),
        ("2026-10-19T10:00:00+00:00", 40),  # Monday, week 43
    ):
        record_usage_event(
            session_factory=factory,
            user_id="usr_a",
            session_id=None,
            turn_id="doc",
            model="anthropic/claude-test",
            purpose="document_conversion",
            input_tokens=tokens,
            output_tokens=0,
            cached_tokens=0,
            occurred_at=at,
            cost_usd=tokens / 100,
        )
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    api.app.state.ledger_factory = factory
    try:
        with TestClient(api.app, headers={"X-API-Key": "test-secret"}) as c:
            yield c
    finally:
        del api.app.state.ledger_factory


def _series(client: TestClient, bucket: str) -> dict[str, int]:
    resp = client.get(
        "/usage-events/summary",
        params={
            "bucket": bucket,
            "range_start": "2026-10-01T00:00:00Z",
            "range_end": "2026-10-31T00:00:00Z",
        },
    )
    assert resp.status_code == 200, resp.text
    return {t["bucket"]: t["input_tokens"] for t in resp.json()["timeseries"]}


def test_summary_buckets_by_day_week_month_and_hour(client: TestClient) -> None:
    assert _series(client, "day") == {"2026-10-12": 10, "2026-10-14": 50, "2026-10-19": 40}
    assert _series(client, "week") == {"2026-W42": 60, "2026-W43": 40}
    assert _series(client, "month") == {"2026-10": 100}
    assert _series(client, "hour") == {
        "2026-10-12T08:00:00Z": 10,
        "2026-10-14T09:00:00Z": 50,
        "2026-10-19T10:00:00Z": 40,
    }
