"""Runtime settings, LLM configuration, prompts, and the AI gateway's status."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)
from pydantic import BaseModel, Field, field_validator

import pipeline.api as core
from pipeline.api import (
    AuthDep,
    _collection_owner,
    _gateway_from_app,
    _job_busy,
    _ledger_factory_from_app,
    _mutation_lock,
    _runtime,
    _switch_active_project,
    request_actor,
)
from pipeline.gateway import Gateway, GatewayError
from pipeline.prompt_config import (
    list_prompt_state,
    set_prompt_text,
)
from pipeline.runtime_config import (
    get_runtime_config,
    get_runtime_config_state,
    set_runtime_config,
    valid_model_name,
)
from pipeline.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()

# ---------------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------------


class SettingsRequest(BaseModel):
    content_security_enabled: bool | None = None
    llm_converter_enabled: bool | None = None
    language_validation_enabled: bool | None = None
    active_collection: str | None = None
    active_project: str | None = None
    reranker_enabled: bool | None = None


class LlmConfigUpdate(BaseModel):
    """AI Management overrides; model names are the gateway's (provider/model, @pool)."""

    converter_model: str | None = None
    converter_max_tokens: int | None = Field(default=None, ge=1, le=200_000)
    converter_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    converter_temperature_enabled: bool | None = None
    security_model: str | None = None

    @field_validator("converter_model", "security_model")
    @classmethod
    def _gateway_model(cls, v: str | None) -> str | None:
        return valid_model_name(v) if v is not None else None


class PromptPatchRequest(BaseModel):
    value: str


class GatewayPingRequest(BaseModel):
    # "models": GET /v1/models (free; checks address and key). "completion": one
    # tiny real call through the converter model (or *model*).
    mode: Literal["models", "completion"] = "models"
    model: str | None = None

    @field_validator("model")
    @classmethod
    def _gateway_model(cls, v: str | None) -> str | None:
        return valid_model_name(v) if v else None


def _settings_view(request: Request) -> dict[str, Any]:
    holder = getattr(request.app.state, "reranker", None)
    return {
        **_runtime,
        "gateway_configured": _gateway_from_app(request).configured(),
        "reranker": holder.status() if holder is not None else {"state": "disabled"},
    }


@router.get("/settings")
def get_settings(request: Request, _auth: AuthDep) -> dict[str, Any]:
    return _settings_view(request)


@router.post("/settings")
def update_settings(request: Request, req: SettingsRequest, _auth: AuthDep) -> dict[str, Any]:
    if req.content_security_enabled is not None:
        _runtime["content_security_enabled"] = req.content_security_enabled
    if req.llm_converter_enabled is not None:
        _runtime["llm_converter_enabled"] = req.llm_converter_enabled
    if req.language_validation_enabled is not None:
        _runtime["language_validation_enabled"] = req.language_validation_enabled
    if req.reranker_enabled is not None:
        _runtime["reranker_enabled"] = req.reranker_enabled
        if req.reranker_enabled:
            core._ensure_reranker(request.app)  # loads in the background
    if req.active_collection is not None or req.active_project is not None:
        _switch_target(request, req.active_collection, req.active_project)
    return _settings_view(request)


def _switch_target(request: Request, collection: str | None, project: str | None) -> None:
    """Swap the active collection/project while nothing is queued or running.

    Holding the mutation lock means the dropzone watcher and job runners
    cannot be mid-write to the old collection while it is replaced.

    Selecting a project's collection selects that project, so searches and writes
    keep its vectors and keyword rows paired. A collection no project owns can be
    selected for vector-only search; writes are refused until a project is chosen.
    """
    if _job_busy() or not _mutation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="a job is queued or running; cannot switch project/collection"
        )
    try:
        current: QdrantVectorStore = request.app.state.vector_store
        if collection is not None:
            # Never auto-create: a typo would otherwise silently start a new,
            # empty collection and redirect all writes into it.
            if not current.client.collection_exists(collection):
                raise HTTPException(
                    status_code=404, detail=f"Collection '{collection}' does not exist"
                )
            owner = _collection_owner(request.app, collection)
            if owner is not None and project is None:
                _switch_active_project(owner, request.app)
            else:
                request.app.state.vector_store = core.QdrantVectorStore(
                    current.client, collection
                )
                _runtime["active_collection"] = collection
        if project is not None:
            from pipeline.projects import ProjectStore

            registry: ProjectStore = request.app.state.project_registry
            if registry.get(project) is None:
                raise HTTPException(status_code=404, detail=f"Project '{project}' not found")
            _switch_active_project(project, request.app)
    finally:
        _mutation_lock.release()


def _gateway_models(gateway: Gateway) -> list[str]:
    """The gateway's model names for the AI Management picker; empty when unreachable."""
    if not gateway.configured():
        return []
    try:
        return gateway.models()
    except GatewayError as exc:
        logger.info("Gateway model list unavailable: %s", exc)
        return []


@router.get("/llm-config")
def get_llm_config(request: Request, _auth: AuthDep) -> dict[str, Any]:
    return get_runtime_config_state(
        _ledger_factory_from_app(request), _gateway_models(_gateway_from_app(request))
    )


@router.patch("/llm-config")
def patch_llm_config(
    request: Request,
    req: LlmConfigUpdate,
    _auth: AuthDep,
) -> dict[str, Any]:
    updates = req.model_dump(exclude_unset=True)
    state = set_runtime_config(
        _ledger_factory_from_app(request), updates, updated_by=request_actor() or "api"
    )
    state["available_models"] = sorted(
        set(state["available_models"]) | set(_gateway_models(_gateway_from_app(request)))
    )
    return state


@router.get("/llm-prompts")
def get_llm_prompts(request: Request, _auth: AuthDep) -> dict[str, Any]:
    return list_prompt_state(_ledger_factory_from_app(request))


@router.patch("/llm-prompts/{prompt_id}")
def patch_llm_prompt(
    request: Request,
    prompt_id: str,
    req: PromptPatchRequest,
    _auth: AuthDep,
) -> dict[str, Any]:
    try:
        return set_prompt_text(
            _ledger_factory_from_app(request),
            prompt_id,
            req.value,
            updated_by=request_actor() or "api",
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown prompt: {prompt_id}") from None
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"failed to save prompt '{prompt_id}'") from exc


# ---------------------------------------------------------------------------
# The AI gateway (pacod-gw)
# ---------------------------------------------------------------------------


@router.get("/ai/gateway")
def gateway_status(request: Request, _auth: AuthDep) -> dict[str, Any]:
    """Whether AI is set up, where the gateway is, and the models in use (no key)."""
    gateway = _gateway_from_app(request)
    effective = get_runtime_config(_ledger_factory_from_app(request))
    return {
        "configured": gateway.configured(),
        "url": gateway.origin,
        "converter_model": effective["converter_model"],
        "security_model": effective["security_model"],
        "converter_enabled": bool(_runtime.get("llm_converter_enabled", False)),
        "security_llm_enabled": bool(request.app.state.config.content_security_llm_enabled),
    }


@router.post("/ai/gateway/ping")
def gateway_ping(request: Request, req: GatewayPingRequest, _auth: AuthDep) -> dict[str, Any]:
    """Test the connection: the model list, or one tiny completion (recorded as usage)."""
    gateway = _gateway_from_app(request)
    started = time.monotonic()
    try:
        if req.mode == "models":
            models = gateway.models()
            return {
                "ok": True,
                "mode": "models",
                "models": models,
                "latency_ms": int((time.monotonic() - started) * 1000),
            }
        factory = _ledger_factory_from_app(request)
        model = req.model or str(get_runtime_config(factory)["converter_model"])
        done = gateway.ping(model)
    except GatewayError as exc:
        return {
            "ok": False,
            "mode": req.mode,
            "error": str(exc),
            "code": exc.code,
            "request_id": exc.request_id,
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
    from pipeline.llm_management import record_usage_event

    record_usage_event(
        session_factory=factory,
        user_id=None,
        session_id=None,
        turn_id=None,
        model=done.model_served or model,
        purpose="gateway_ping",
        input_tokens=done.usage.input_tokens,
        output_tokens=done.usage.output_tokens,
        cached_tokens=done.usage.cached_tokens,
        cost_usd=done.usage.cost,
        currency=done.usage.currency,
        request_id=done.request_id,
    )
    return {
        "ok": True,
        "mode": "completion",
        "model": model,
        "model_served": done.model_served,
        "request_id": done.request_id,
        "latency_ms": done.duration_ms,
        "cost": done.usage.cost,
        "currency": done.usage.currency,
        "reply": done.text[:200],
    }
