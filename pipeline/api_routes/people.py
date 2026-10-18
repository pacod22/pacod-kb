"""A person's traces in kb, for pacod-id's account deletion."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

import pipeline.api as core
from pipeline.api import _ACTOR_RE, AuthDep, _jobs, _jobs_lock, _ledger_factory_from_app

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

DELETED = "deleted-user"


@router.post("/people/{sub}/erase")
def erase_person(request: Request, sub: str, _auth: AuthDep) -> dict[str, int]:
    """Replace *sub* by ``deleted-user`` wherever kb recorded who acted: usage rows, AI
    interactions, who changed a setting, and job parameters. Idempotent; the UI calls it when
    pacod-id deletes an account, then confirms the deletion to pacod-id."""
    if not _ACTOR_RE.fullmatch(sub):
        raise HTTPException(status_code=400, detail="not a pacod-id id")
    from sqlalchemy import func, update

    from pipeline.db_models import AiInteraction, AppSetting, KbJob, UsageEvent
    from pipeline.document_store import _session_scope

    counts: dict[str, int] = {}
    factory = _ledger_factory_from_app(request)
    if factory is not None:
        with _session_scope(factory) as s:
            for name, column in (
                ("usage_events", UsageEvent.user_id),
                ("ai_interactions", AiInteraction.user_id),
                ("app_settings", AppSetting.updated_by),
            ):
                table: Any = column.class_
                result = s.execute(update(table).where(column == sub).values({column: DELETED}))
                counts[name] = int(result.rowcount or 0)
    if core._job_session_factory is not None:
        with _session_scope(core._job_session_factory) as s:
            result = s.execute(
                update(KbJob)
                .where(KbJob.params_json.contains(sub))
                .values(params_json=func.replace(KbJob.params_json, sub, DELETED))
            )
            counts["jobs"] = int(result.rowcount or 0)
    with _jobs_lock:
        for job in _jobs.values():
            if job.get("params", {}).get("actor") == sub:
                job["params"] = {**job["params"], "actor": DELETED}
    logger.info("Erased the traces of a deleted account: %s", counts)
    return counts
