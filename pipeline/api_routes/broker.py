"""pacod-broker, read side: the Broker page's data and kb's connection to the broker.

Every read answers 200 and reports a broker (or gateway) failure in ``broker_error``
(``gateway_error``): a broker that is down costs the page its facts, never the page.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

import pipeline.api as core
from pipeline import broker, broker_config
from pipeline.api import AuthDep, _gateway_from_app, _ledger_factory_from_app, request_actor
from pipeline.gateway import Gateway, GatewayError, GatewayModel
from pipeline.runtime_config import get_runtime_config

logger = logging.getLogger(__name__)

router: APIRouter = APIRouter()


def _connection(request: Request) -> broker_config.BrokerConnection:
    return broker_config.get_connection(
        request.app.state.config, _ledger_factory_from_app(request), core._PIPELINE_API_SECRET
    )


def _transport(request: Request) -> httpx.BaseTransport | None:
    """The tests' seam: a MockTransport on ``app.state.broker_transport``."""
    return getattr(request.app.state, "broker_transport", None)


def _gateway_rows(gateway: Gateway) -> tuple[list[GatewayModel], str | None]:
    if not gateway.configured():
        return [], "the AI gateway is not set up"
    try:
        return gateway.model_rows(), None
    except GatewayError as exc:
        return [], str(exc)


@router.get("/broker")
def broker_overview(request: Request, _auth: AuthDep) -> dict[str, Any]:
    """The models kb uses and the gateway's linked models, with the broker's facts."""
    conn = _connection(request)
    rows, gateway_error = _gateway_rows(_gateway_from_app(request))
    effective = get_runtime_config(_ledger_factory_from_app(request))
    in_use = {
        "converter": str(effective.get("converter_model") or ""),
        "security": str(effective.get("security_model") or ""),
    }
    enabled = {
        "converter": bool(core._runtime.get("llm_converter_enabled", False)),
        "security": bool(request.app.state.config.content_security_llm_enabled),
    }
    view = broker.overview(
        conn,
        rows,
        {k: v for k, v in in_use.items() if v},
        enabled=enabled,
        transport=_transport(request),
    )
    return {
        "configured": conn.configured,
        "url": conn.url,
        "token_source": conn.sources.get("token"),
        **view,
        "gateway_error": gateway_error,
    }


@router.get("/broker/catalogue")
def broker_catalogue(request: Request, _auth: AuthDep) -> dict[str, Any]:
    """Every model and embedding the broker lists, marked by how the gateway offers it."""
    conn = _connection(request)
    rows, gateway_error = _gateway_rows(_gateway_from_app(request))
    out: dict[str, Any] = {
        "items": [],
        "truncated": False,
        "gateway_error": gateway_error,
        "broker_error": None,
    }
    try:
        out.update(broker.catalogue(conn, rows, transport=_transport(request)))
    except broker.BrokerUnavailableError as exc:
        out["broker_error"] = str(exc)
    return out


@router.get("/broker/search")
def broker_search(
    request: Request,
    _auth: AuthDep,
    q: str = Query(default="", max_length=200),
    kind: str = Query(default="", max_length=32),
) -> dict[str, Any]:
    conn = _connection(request)
    try:
        hits = broker.search(conn, q, kind or None, transport=_transport(request))
    except broker.BrokerUnavailableError as exc:
        return {"hits": [], "broker_error": str(exc)}
    return {"hits": hits, "broker_error": None}


# ---- the connection ---------------------------------------------------------------------


class BrokerConfigUpdate(BaseModel):
    """A field left out is kept; ``null`` returns it to the deployment's value.

    ``token``: left out = kept; ``null`` = the deployment's pairing; ``""`` = open mode (no
    token) for the resulting address; text = the token for the resulting address.
    """

    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    timeout_seconds: float | None = None
    token: str | None = None


def _describe(request: Request) -> dict[str, Any]:
    return broker_config.describe(
        request.app.state.config, _connection(request), core._PIPELINE_API_SECRET
    )


@router.get("/broker/config")
def get_broker_config(request: Request, _auth: AuthDep) -> dict[str, Any]:
    return _describe(request)


@router.put("/broker/config")
def put_broker_config(
    request: Request, req: BrokerConfigUpdate, _auth: AuthDep
) -> dict[str, Any]:
    fields = req.model_fields_set
    changes: dict[str, Any] = {
        name: getattr(req, name) for name in ("url", "timeout_seconds") if name in fields
    }
    # Left out: the token is kept (save's default).
    token: dict[str, str | None] = {"token": req.token} if "token" in fields else {}
    try:
        broker_config.save(
            request.app.state.config,
            _ledger_factory_from_app(request),
            core._PIPELINE_API_SECRET,
            changes,
            updated_by=request_actor() or "api",
            **token,
        )
    except broker_config.BrokerConfigError as exc:
        raise HTTPException(
            status_code=exc.status, detail={"code": exc.code, "message": exc.message}
        ) from None
    broker.reset_cache()
    logger.info(
        "Broker connection changed",
        extra={"fields": sorted(fields), "actor": request_actor() or "api"},
    )
    return _describe(request)


@router.delete("/broker/config")
def reset_broker_config(request: Request, _auth: AuthDep) -> dict[str, Any]:
    broker_config.reset(_ledger_factory_from_app(request))
    broker.reset_cache()
    logger.info("Broker connection reset", extra={"actor": request_actor() or "api"})
    return _describe(request)


@router.post("/broker/config/test")
def test_broker_config(request: Request, _auth: AuthDep) -> dict[str, Any]:
    """Whether the broker in force answers, is ready and takes kb's credentials."""
    return broker.probe(_connection(request), transport=_transport(request))

