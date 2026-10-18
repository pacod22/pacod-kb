"""Per-call AI interaction log."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

# ---------------------------------------------------------------------------
# AI interactions (detailed per-call logging)
# ---------------------------------------------------------------------------


def _ai_interaction_summary(row: Any) -> dict[str, Any]:
    return {
        "id": int(row.id),
        "created_at": str(row.created_at),
        "provider": str(row.provider),
        "model": str(row.model),
        "purpose": str(row.purpose),
        "model_role": row.model_role,
        "status": str(row.status),
        "user_id": row.user_id,
        "session_id": row.session_id,
        "turn_id": row.turn_id,
        "doc_id": row.doc_id,
        "input_tokens": int(row.input_tokens or 0),
        "output_tokens": int(row.output_tokens or 0),
        "cached_tokens": int(row.cached_tokens or 0),
        "cost_usd": float(row.cost_usd or 0.0),
        "currency": row.currency,
        "cost_source": row.cost_source,
        "request_id": row.request_id,
        "model_served": row.model_served,
        "duration_ms": int(row.duration_ms or 0),
        "finish_reason": row.finish_reason,
        "error": row.error,
    }


def _ai_interaction_detail(row: Any) -> dict[str, Any]:
    detail = _ai_interaction_summary(row)
    detail.update(
        {
            "request_system": row.request_system,
            "request_user": row.request_user,
            "response_text": row.response_text,
            "temperature": row.temperature,
            "max_tokens": row.max_tokens,
            "meta_json": row.meta_json,
        }
    )
    return detail


@router.get("/ai-interactions")
def list_ai_interactions(
    request: Request,
    _auth: AuthDep,
    provider: str = Query(""),
    model: str = Query(""),
    purpose: str = Query(""),
    status: str = Query(""),
    turn_id: str = Query(""),
    doc_id: str = Query(""),
    limit: int = Query(50),
    offset: int = Query(0),
) -> dict[str, Any]:
    session_factory = _ledger_factory_from_app(request)
    if session_factory is None:
        return {"interactions": [], "total": 0, "limit": limit, "offset": offset}
    try:
        from sqlalchemy import func, select

        from pipeline.db_models import AiInteraction
        from pipeline.document_store import _session_scope

        capped_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))
        conds: list[Any] = []
        if provider.strip():
            conds.append(AiInteraction.provider == provider.strip().lower())
        if model.strip():
            conds.append(AiInteraction.model == model.strip())
        if purpose.strip():
            conds.append(AiInteraction.purpose == purpose.strip())
        if status.strip():
            conds.append(AiInteraction.status == status.strip())
        if turn_id.strip():
            conds.append(AiInteraction.turn_id == turn_id.strip())
        if doc_id.strip():
            conds.append(AiInteraction.doc_id == doc_id.strip())

        with _session_scope(session_factory) as s:
            total = int(s.execute(select(func.count(AiInteraction.id)).where(*conds)).scalar_one())
            rows = (
                s.execute(
                    select(AiInteraction)
                    .where(*conds)
                    .order_by(AiInteraction.id.desc())
                    .limit(capped_limit)
                    .offset(safe_offset)
                )
                .scalars()
                .all()
            )
            interactions = [_ai_interaction_summary(r) for r in rows]
            purpose_options = [
                str(r[0])
                for r in s.execute(
                    select(AiInteraction.purpose).distinct().order_by(AiInteraction.purpose)
                ).all()
            ]
            provider_options = [
                str(r[0])
                for r in s.execute(
                    select(AiInteraction.provider).distinct().order_by(AiInteraction.provider)
                ).all()
            ]
        return {
            "interactions": interactions,
            "total": total,
            "limit": capped_limit,
            "offset": safe_offset,
            "purpose_options": purpose_options,
            "provider_options": provider_options,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"failed to list ai interactions: {exc}"
        ) from exc


@router.get("/ai-interactions/turn/{turn_id}")
def ai_interactions_turn(
    request: Request,
    turn_id: str,
    _auth: AuthDep,
) -> dict[str, Any]:
    session_factory = _ledger_factory_from_app(request)
    if session_factory is None:
        return {"turn_id": turn_id, "interactions": []}
    try:
        from sqlalchemy import select

        from pipeline.db_models import AiInteraction
        from pipeline.document_store import _session_scope

        with _session_scope(session_factory) as s:
            rows = (
                s.execute(
                    select(AiInteraction)
                    .where(AiInteraction.turn_id == turn_id)
                    .order_by(AiInteraction.id.asc())
                )
                .scalars()
                .all()
            )
            interactions = [_ai_interaction_detail(r) for r in rows]
        return {"turn_id": turn_id, "interactions": interactions}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"failed to load turn interactions: {exc}"
        ) from exc


@router.get("/ai-interactions/{interaction_id}")
def get_ai_interaction(
    request: Request,
    interaction_id: int,
    _auth: AuthDep,
) -> dict[str, Any]:
    session_factory = _ledger_factory_from_app(request)
    if session_factory is None:
        raise HTTPException(status_code=404, detail="ai interaction logging requires database mode")
    try:
        from sqlalchemy import select

        from pipeline.db_models import AiInteraction
        from pipeline.document_store import _session_scope

        with _session_scope(session_factory) as s:
            row = s.execute(
                select(AiInteraction).where(AiInteraction.id == interaction_id)
            ).scalar_one_or_none()
            if row is None:
                raise HTTPException(status_code=404, detail="interaction not found")
            return _ai_interaction_detail(row)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"failed to load ai interaction: {exc}"
        ) from exc
