"""LLM usage accounting: the ledger's ``usage_events`` and ``ai_interactions`` rows.

The gateway (pipeline/gateway.py) prices every call; these helpers record what it
reported — tokens, cost and currency, and its request id — next to kb's own context
(document, purpose, the acting person). Writes are best-effort and never block a
user-facing flow.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def provider_of(model_or_provider: str) -> str:
    """Return a coarse provider label from model/provider identifiers."""
    raw = (model_or_provider or "").strip().lower()
    if not raw:
        return "unknown"
    if raw.startswith("@"):
        return "pool"
    if raw in {"openai", "anthropic", "gemini", "deepseek", "glm"}:
        return raw
    if "/" in raw:
        return raw.split("/", 1)[0]
    if raw.startswith("claude"):
        return "anthropic"
    if raw.startswith("gpt") or raw.startswith("o1") or raw.startswith("o3"):
        return "openai"
    if raw.startswith("gemini"):
        return "gemini"
    if raw.startswith("deepseek"):
        return "deepseek"
    if raw.startswith("glm"):
        return "glm"
    return raw


def cost_source(cost: float | None) -> str:
    """``gateway`` when the gateway priced the call, ``unpriced`` when it did not."""
    return "gateway" if cost is not None else "unpriced"


def record_usage_event(
    *,
    session_factory: Any | None,
    user_id: str | None,
    session_id: str | None,
    turn_id: str | None,
    model: str,
    purpose: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    cost_usd: float | None,
    currency: str | None = None,
    request_id: str | None = None,
    occurred_at: str | None = None,
) -> None:
    """Best-effort append-only usage ledger write.

    *cost_usd* is the gateway's price for the call (in *currency*, USD unless the
    gateway says otherwise); ``None`` stores 0 marked ``unpriced``.
    """
    if session_factory is None:
        return
    try:
        from pipeline.db_models import UsageEvent
        from pipeline.document_store import _session_scope

        with _session_scope(session_factory) as s:
            s.add(
                UsageEvent(
                    user_id=user_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    occurred_at=occurred_at or datetime.now(UTC).isoformat(),
                    model=model,
                    purpose=purpose,
                    input_tokens=max(input_tokens, 0),
                    output_tokens=max(output_tokens, 0),
                    cached_tokens=max(cached_tokens, 0),
                    cost_usd=float(cost_usd or 0.0),
                    currency=currency if cost_usd is not None else None,
                    cost_source=cost_source(cost_usd),
                    request_id=request_id,
                )
            )
    except Exception as exc:
        # Accounting never blocks user-facing flows.
        logger.debug("usage event not recorded: %s", exc)


_MAX_INTERACTION_TEXT_CHARS: int = 40_000


def _truncate_text(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) > _MAX_INTERACTION_TEXT_CHARS:
        return value[:_MAX_INTERACTION_TEXT_CHARS] + "\n…[truncated]"
    return value


def log_ai_interaction(
    *,
    session_factory: Any | None,
    provider: str,
    model: str,
    purpose: str,
    status: str = "ok",
    model_role: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    doc_id: str | None = None,
    request_system: str | None = None,
    request_user: str | None = None,
    response_text: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    finish_reason: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    cost_usd: float | None = None,
    currency: str | None = None,
    request_id: str | None = None,
    model_served: str | None = None,
    duration_ms: int = 0,
    error: str | None = None,
    meta: dict[str, Any] | None = None,
    occurred_at: str | None = None,
) -> None:
    """Best-effort append-only write of a single detailed LLM interaction.

    Never raises; logging detail must not block user-facing flows. Large request
    and response payloads are truncated before storage.
    """
    if session_factory is None:
        return
    try:
        from pipeline.db_models import AiInteraction
        from pipeline.document_store import _session_scope

        meta_json: str | None = None
        if meta:
            try:
                meta_json = json.dumps(meta, ensure_ascii=False)
            except Exception:
                meta_json = None

        with _session_scope(session_factory) as s:
            s.add(
                AiInteraction(
                    created_at=occurred_at or datetime.now(UTC).isoformat(),
                    provider=provider_of(provider or model),
                    model=model or "unknown",
                    purpose=purpose,
                    model_role=model_role,
                    status=status,
                    user_id=user_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    doc_id=doc_id,
                    request_system=_truncate_text(request_system),
                    request_user=_truncate_text(request_user),
                    response_text=_truncate_text(response_text),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    finish_reason=finish_reason,
                    input_tokens=max(int(input_tokens), 0),
                    output_tokens=max(int(output_tokens), 0),
                    cached_tokens=max(int(cached_tokens), 0),
                    cost_usd=float(cost_usd or 0.0),
                    currency=currency if cost_usd is not None else None,
                    cost_source=cost_source(cost_usd),
                    request_id=request_id,
                    model_served=model_served,
                    duration_ms=max(int(duration_ms), 0),
                    error=_truncate_text(error),
                    meta_json=meta_json,
                )
            )
    except Exception as exc:
        # Detailed logging never blocks user-facing flows.
        logger.debug("AI interaction not recorded: %s", exc)


def to_usage_payload(data: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a conversion's usage blob (tokens, plus the gateway's cost and ids)."""
    if not isinstance(data, dict):
        return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    payload: dict[str, Any] = {
        "input_tokens": int(data.get("input_tokens") or 0),
        "output_tokens": int(data.get("output_tokens") or 0),
        "cached_tokens": int(data.get("cached_tokens") or 0),
    }
    cost = data.get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        payload["cost"] = float(cost)
        payload["currency"] = str(data.get("currency") or "USD")
    for key in ("request_id", "model_served"):
        if data.get(key):
            payload[key] = str(data[key])
    return payload
