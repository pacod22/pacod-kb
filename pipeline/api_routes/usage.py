"""LLM usage and cost reporting."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
)

from pipeline.api import (
    AuthDep,
    _ledger_factory_from_app,
)
from pipeline.llm_management import (
    provider_of,
)

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()


def _usage_empty() -> dict[str, Any]:
    return {
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "event_count": 0,
    }


# Length of the ``occurred_at`` prefix (an ISO-8601 UTC string) each bucket groups by.
_BUCKET_PREFIX = {"hour": 13, "day": 10, "week": 10, "month": 7}


def _usage_bucket_expr(usage_event_cls: Any, bucket: str) -> Any:
    """A prefix of ``occurred_at``: portable across SQLite and PostgreSQL.

    ``strftime`` exists only in SQLite. Weeks are grouped by day here and
    folded into ISO weeks by :func:`_bucket_label`.
    """
    from sqlalchemy import func

    return func.substr(usage_event_cls.occurred_at, 1, _BUCKET_PREFIX.get(bucket, 10))


def _bucket_label(raw: str, bucket: str) -> str:
    """The display label of a bucket prefix: ``…T13:00:00Z``, ``2026-W42``, as is."""
    if not raw:
        return raw
    if bucket == "hour":
        return f"{raw}:00:00Z"
    if bucket == "week":
        try:
            year, week, _ = date.fromisoformat(raw).isocalendar()
        except ValueError:
            return raw
        return f"{year}-W{week:02d}"
    return raw


@router.get("/usage-events/turn/{turn_id}")
def usage_events_turn(
    request: Request,
    turn_id: str,
    _auth: AuthDep,
) -> dict[str, Any]:
    session_factory = _ledger_factory_from_app(request)
    if session_factory is None:
        return {"turn_id": turn_id, "totals": _usage_empty(), "rows": []}
    try:
        from sqlalchemy import func, select

        from pipeline.db_models import UsageEvent
        from pipeline.document_store import _session_scope

        with _session_scope(session_factory) as s:
            rows = s.execute(
                select(
                    UsageEvent.purpose,
                    UsageEvent.model,
                    func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
                    func.coalesce(func.sum(UsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.cached_tokens), 0),
                    func.count(UsageEvent.id),
                )
                .where(UsageEvent.turn_id == turn_id)
                .group_by(UsageEvent.purpose, UsageEvent.model)
                .order_by(func.sum(UsageEvent.cost_usd).desc())
            ).all()
        out_rows: list[dict[str, Any]] = [
            {
                "purpose": str(r[0]),
                "model": str(r[1]),
                "cost_usd": float(r[2] or 0.0),
                "input_tokens": int(r[3] or 0),
                "output_tokens": int(r[4] or 0),
                "cached_tokens": int(r[5] or 0),
                "event_count": int(r[6] or 0),
            }
            for r in rows
        ]
        totals = _usage_empty()
        for row in out_rows:
            totals["cost_usd"] += float(row["cost_usd"])
            totals["input_tokens"] += int(row["input_tokens"])
            totals["output_tokens"] += int(row["output_tokens"])
            totals["cached_tokens"] += int(row["cached_tokens"])
            totals["event_count"] += int(row["event_count"])
        return {"turn_id": turn_id, "totals": totals, "rows": out_rows}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"failed to aggregate turn usage: {exc}"
        ) from exc


@router.get("/usage-events/overview")
def usage_events_overview(request: Request, _auth: AuthDep) -> dict[str, Any]:
    session_factory = _ledger_factory_from_app(request)
    if session_factory is None:
        empty = _usage_empty()
        return {"hour": empty, "day": empty, "week": empty, "month": empty}
    try:
        from sqlalchemy import case, func, select

        from pipeline.db_models import UsageEvent
        from pipeline.document_store import _session_scope

        now = datetime.now(UTC)
        start_hour = now.replace(minute=0, second=0, microsecond=0)
        start_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_week = start_day - timedelta(days=start_day.weekday())
        start_month = start_day.replace(day=1)

        with _session_scope(session_factory) as s:
            row = s.execute(
                select(
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_hour.isoformat(),
                                    UsageEvent.cost_usd,
                                ),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_hour.isoformat(),
                                    UsageEvent.input_tokens,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_hour.isoformat(),
                                    UsageEvent.output_tokens,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_hour.isoformat(),
                                    UsageEvent.cached_tokens,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case((UsageEvent.occurred_at >= start_hour.isoformat(), 1), else_=0)
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_day.isoformat(),
                                    UsageEvent.cost_usd,
                                ),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_day.isoformat(),
                                    UsageEvent.input_tokens,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_day.isoformat(),
                                    UsageEvent.output_tokens,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_day.isoformat(),
                                    UsageEvent.cached_tokens,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case((UsageEvent.occurred_at >= start_day.isoformat(), 1), else_=0)
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_week.isoformat(),
                                    UsageEvent.cost_usd,
                                ),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_week.isoformat(),
                                    UsageEvent.input_tokens,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_week.isoformat(),
                                    UsageEvent.output_tokens,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    UsageEvent.occurred_at >= start_week.isoformat(),
                                    UsageEvent.cached_tokens,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case((UsageEvent.occurred_at >= start_week.isoformat(), 1), else_=0)
                        ),
                        0,
                    ),
                    func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
                    func.coalesce(func.sum(UsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.cached_tokens), 0),
                    func.count(UsageEvent.id),
                ).where(UsageEvent.occurred_at >= start_month.isoformat())
            ).one()

        return {
            "hour": {
                "cost_usd": float(row[0] or 0.0),
                "input_tokens": int(row[1] or 0),
                "output_tokens": int(row[2] or 0),
                "cached_tokens": int(row[3] or 0),
                "event_count": int(row[4] or 0),
            },
            "day": {
                "cost_usd": float(row[5] or 0.0),
                "input_tokens": int(row[6] or 0),
                "output_tokens": int(row[7] or 0),
                "cached_tokens": int(row[8] or 0),
                "event_count": int(row[9] or 0),
            },
            "week": {
                "cost_usd": float(row[10] or 0.0),
                "input_tokens": int(row[11] or 0),
                "output_tokens": int(row[12] or 0),
                "cached_tokens": int(row[13] or 0),
                "event_count": int(row[14] or 0),
            },
            "month": {
                "cost_usd": float(row[15] or 0.0),
                "input_tokens": int(row[16] or 0),
                "output_tokens": int(row[17] or 0),
                "cached_tokens": int(row[18] or 0),
                "event_count": int(row[19] or 0),
            },
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"failed to aggregate usage overview: {exc}"
        ) from exc


@router.get("/usage-events/summary")
def usage_events_summary(
    request: Request,
    _auth: AuthDep,
    bucket: str = Query("day"),
    provider: str = Query(""),
    model: str = Query(""),
    range_start: str = Query(""),
    range_end: str = Query(""),
) -> dict[str, Any]:
    session_factory = _ledger_factory_from_app(request)
    if session_factory is None:
        return {
            "bucket": bucket,
            "range_start": "",
            "range_end": "",
            "totals": _usage_empty(),
            "timeseries": [],
            "model_series": [],
            "by_model": [],
            "by_provider": [],
            "by_purpose": [],
            "by_user": [],
            "provider_options": [],
            "model_options": [],
        }
    try:
        from sqlalchemy import func, select

        from pipeline.db_models import UsageEvent
        from pipeline.document_store import _session_scope

        bucket_key = bucket if bucket in {"hour", "day", "week", "month"} else "day"
        now = datetime.now(UTC)
        default_days = {"hour": 2, "day": 30, "week": 84, "month": 365}
        start_dt = now - timedelta(days=default_days[bucket_key])
        end_dt = now
        if range_start.strip():
            try:
                start_dt = datetime.fromisoformat(range_start.replace("Z", "+00:00"))
            except Exception:
                pass
        if range_end.strip():
            try:
                end_dt = datetime.fromisoformat(range_end.replace("Z", "+00:00"))
            except Exception:
                pass

        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)

        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()
        provider_filter = provider.strip().lower()
        model_filter = model.strip()

        with _session_scope(session_factory) as s:
            base_conds: list[Any] = [
                UsageEvent.occurred_at >= start_iso,
                UsageEvent.occurred_at <= end_iso,
            ]

            option_rows = s.execute(
                select(
                    UsageEvent.model,
                    func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
                )
                .where(*base_conds)
                .group_by(UsageEvent.model)
                .order_by(func.sum(UsageEvent.cost_usd).desc())
            ).all()
            model_options = [str(r[0]) for r in option_rows]
            provider_options = sorted({provider_of(m) for m in model_options})

            conds = list(base_conds)
            if model_filter:
                conds.append(UsageEvent.model == model_filter)
            elif provider_filter:
                provider_models = [m for m in model_options if provider_of(m) == provider_filter]
                if provider_models:
                    conds.append(UsageEvent.model.in_(provider_models))
                else:
                    conds.append(UsageEvent.model == "__NO_MATCH__")

            totals_row = s.execute(
                select(
                    func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
                    func.coalesce(func.sum(UsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.cached_tokens), 0),
                    func.count(UsageEvent.id),
                ).where(*conds)
            ).one()

            by_model_rows = s.execute(
                select(
                    UsageEvent.model,
                    func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
                    func.coalesce(func.sum(UsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.cached_tokens), 0),
                    func.count(UsageEvent.id),
                )
                .where(*conds)
                .group_by(UsageEvent.model)
                .order_by(func.sum(UsageEvent.cost_usd).desc())
            ).all()

            by_purpose_rows = s.execute(
                select(
                    UsageEvent.purpose,
                    func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
                    func.coalesce(func.sum(UsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.cached_tokens), 0),
                    func.count(UsageEvent.id),
                )
                .where(*conds)
                .group_by(UsageEvent.purpose)
                .order_by(func.sum(UsageEvent.cost_usd).desc())
            ).all()

            by_user_rows = s.execute(
                select(
                    UsageEvent.user_id,
                    func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
                    func.coalesce(func.sum(UsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.cached_tokens), 0),
                    func.count(UsageEvent.id),
                )
                .where(*conds)
                .group_by(UsageEvent.user_id)
                .order_by(func.sum(UsageEvent.cost_usd).desc())
                .limit(20)
            ).all()

            bucket_expr = _usage_bucket_expr(UsageEvent, bucket_key)
            timeseries_rows = s.execute(
                select(
                    bucket_expr.label("b"),
                    func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
                    func.coalesce(func.sum(UsageEvent.input_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.output_tokens), 0),
                    func.coalesce(func.sum(UsageEvent.cached_tokens), 0),
                    func.count(UsageEvent.id),
                )
                .where(*conds)
                .group_by(bucket_expr)
                .order_by(bucket_expr)
            ).all()

            model_bucket_rows = s.execute(
                select(
                    bucket_expr.label("b"),
                    UsageEvent.model,
                    func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
                    (
                        func.coalesce(func.sum(UsageEvent.input_tokens), 0)
                        + func.coalesce(func.sum(UsageEvent.output_tokens), 0)
                        + func.coalesce(func.sum(UsageEvent.cached_tokens), 0)
                    ),
                )
                .where(*conds)
                .group_by(bucket_expr, UsageEvent.model)
                .order_by(bucket_expr)
            ).all()

        by_model: list[dict[str, Any]] = [
            {
                "model": str(r[0]),
                "cost_usd": float(r[1] or 0.0),
                "input_tokens": int(r[2] or 0),
                "output_tokens": int(r[3] or 0),
                "cached_tokens": int(r[4] or 0),
                "event_count": int(r[5] or 0),
            }
            for r in by_model_rows
        ]

        by_provider_map: dict[str, dict[str, Any]] = {}
        for row in by_model:
            p = provider_of(str(row.get("model", "")))
            agg = by_provider_map.setdefault(
                p,
                {
                    "provider": p,
                    "cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "event_count": 0,
                },
            )
            agg["cost_usd"] += float(row["cost_usd"])
            agg["input_tokens"] += int(row["input_tokens"])
            agg["output_tokens"] += int(row["output_tokens"])
            agg["cached_tokens"] += int(row["cached_tokens"])
            agg["event_count"] += int(row["event_count"])

        by_provider = sorted(
            by_provider_map.values(),
            key=lambda x: float(x["cost_usd"]),
            reverse=True,
        )

        # Rows come in bucket order; days of one ISO week fold into one entry.
        folded: dict[str, list[float]] = {}
        for r in timeseries_rows:
            acc = folded.setdefault(_bucket_label(str(r[0] or ""), bucket_key), [0.0] * 5)
            for i in range(5):
                acc[i] += float(r[i + 1] or 0)
        timeseries = [
            {
                "bucket": label,
                "cost_usd": acc[0],
                "input_tokens": int(acc[1]),
                "output_tokens": int(acc[2]),
                "cached_tokens": int(acc[3]),
                "tokens": int(acc[1] + acc[2] + acc[3]),
                "event_count": int(acc[4]),
            }
            for label, acc in folded.items()
        ]
        buckets = [str(t["bucket"]) for t in timeseries]

        top_model_ids = [str(x["model"]) for x in by_model[:8]]
        series_map: dict[str, dict[str, Any]] = {}
        other_cost: dict[str, float] = dict.fromkeys(buckets, 0.0)
        other_tokens: dict[str, int] = dict.fromkeys(buckets, 0)
        for r in model_bucket_rows:
            b = _bucket_label(str(r[0] or ""), bucket_key)
            m = str(r[1] or "")
            c = float(r[2] or 0.0)
            t = int(r[3] or 0)
            if m in top_model_ids:
                data = series_map.setdefault(
                    m,
                    {
                        "model": m,
                        "cost_usd": [0.0 for _ in buckets],
                        "tokens": [0 for _ in buckets],
                    },
                )
                if b in buckets:
                    idx = buckets.index(b)
                    data["cost_usd"][idx] += c
                    data["tokens"][idx] += t
            else:
                if b in other_cost:
                    other_cost[b] += c
                    other_tokens[b] += t

        model_series = [series_map[m] for m in top_model_ids if m in series_map]
        if any(v > 0.0 for v in other_cost.values()) or any(v > 0 for v in other_tokens.values()):
            model_series.append(
                {
                    "model": "Other",
                    "cost_usd": [float(other_cost[b]) for b in buckets],
                    "tokens": [int(other_tokens[b]) for b in buckets],
                }
            )

        return {
            "bucket": bucket_key,
            "range_start": start_iso,
            "range_end": end_iso,
            "totals": {
                "cost_usd": float(totals_row[0] or 0.0),
                "input_tokens": int(totals_row[1] or 0),
                "output_tokens": int(totals_row[2] or 0),
                "cached_tokens": int(totals_row[3] or 0),
                "event_count": int(totals_row[4] or 0),
            },
            "timeseries": timeseries,
            "model_series": model_series,
            "by_model": by_model,
            "by_provider": by_provider,
            "by_purpose": [
                {
                    "purpose": str(r[0]),
                    "cost_usd": float(r[1] or 0.0),
                    "input_tokens": int(r[2] or 0),
                    "output_tokens": int(r[3] or 0),
                    "cached_tokens": int(r[4] or 0),
                    "event_count": int(r[5] or 0),
                }
                for r in by_purpose_rows
            ],
            "by_user": [
                {
                    "user_id": str(r[0] or "unknown"),
                    "cost_usd": float(r[1] or 0.0),
                    "input_tokens": int(r[2] or 0),
                    "output_tokens": int(r[3] or 0),
                    "cached_tokens": int(r[4] or 0),
                    "event_count": int(r[5] or 0),
                }
                for r in by_user_rows
            ],
            "provider_options": provider_options,
            "model_options": model_options,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"failed to aggregate usage summary: {exc}"
        ) from exc
