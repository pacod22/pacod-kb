"""The Broker page (admin): what pacod-broker says about the models kb can use.

kb only reads the broker. The page shows the models kb uses and the gateway's models that
name a broker resource, with reputation, price and status; the broker's catalogue of models
and embeddings, marked by how the gateway offers each; a search; and kb's connection to the
broker, which an admin may override (pipeline/broker_config.py).

Listings are written by other providers: Jinja escapes them, nothing is marked safe, and
interface addresses are shown as text, never as links.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse

from web.auth import require_role
from web.dependencies import PipelineClientDep
from web.routers.pipeline import _get_json, _pipeline_error
from web.templates_config import templates

router = APIRouter(prefix="/kb/broker", dependencies=[Depends(require_role("admin"))])

KINDS = ("model", "embedding", "mcp_server", "agent", "tool", "compute", "dataset")
_TOKEN_ACTIONS = ("keep", "set", "open", "deployment")
# Pages of the catalogue the pipeline may read before it answers (5 per kind, each bounded
# by the broker timeout); the first failure ends it sooner.
_CATALOGUE_TIMEOUT = 75.0
# Answers the connection partial and every section that follows the connection.
_CHANGED = {"HX-Trigger": "broker-changed"}


def _empty_overview() -> dict[str, Any]:
    return {
        "configured": False,
        "in_use": [],
        "linked": [],
        "broker_error": "the pipeline did not answer",
        "gateway_error": None,
    }


def _empty_config() -> dict[str, Any]:
    return {}


@router.get("", response_class=HTMLResponse)
async def broker_page(request: Request, client: PipelineClientDep) -> HTMLResponse:
    overview, config = await asyncio.gather(
        _get_json(client, "/broker", _empty_overview()),
        _get_json(client, "/broker/config", _empty_config()),
    )
    return templates.TemplateResponse(
        request,
        "broker/index.html",
        {"overview": overview, "config": config, "kinds": KINDS},
    )


@router.get("/overview", response_class=HTMLResponse)
async def broker_overview(request: Request, client: PipelineClientDep) -> HTMLResponse:
    overview = await _get_json(client, "/broker", _empty_overview())
    return templates.TemplateResponse(request, "broker/_overview.html", {"overview": overview})


def _matches(item: dict[str, Any], words: list[str]) -> bool:
    haystack = " ".join(
        str(item.get(k) or "") for k in ("id", "name", "provider", "summary", "upstream_id")
    ).lower() + " " + " ".join(item.get("tags") or []).lower()
    return all(w in haystack for w in words)


@router.get("/catalogue", response_class=HTMLResponse)
async def broker_catalogue(
    request: Request,
    client: PipelineClientDep,
    q: str = Query(default="", max_length=200),
    kind: str = Query(default="", max_length=32),
    usable: str = Query(default="", max_length=8),
) -> HTMLResponse:
    """The broker's models and embeddings (cached by the pipeline), filtered here."""
    data: dict[str, Any]
    try:
        resp = await client.get("/broker/catalogue", timeout=_CATALOGUE_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        data = {"items": [], "truncated": False, "broker_error": _pipeline_error(exc, client)}
    items = [i for i in data.get("items") or [] if isinstance(i, dict)]
    total = len(items)
    words = q.lower().split()
    if words:
        items = [i for i in items if _matches(i, words)]
    if kind in ("model", "embedding"):
        items = [i for i in items if i.get("kind") == kind]
    if usable == "1":
        items = [i for i in items if (i.get("usage") or {}).get("state") == "usable"]
    return templates.TemplateResponse(
        request,
        "broker/_catalogue.html",
        {"catalogue": data, "items": items, "total": total, "q": q, "kind": kind,
         "usable": usable == "1"},  # fmt: skip
    )


@router.get("/search", response_class=HTMLResponse)
async def broker_search(
    request: Request,
    client: PipelineClientDep,
    q: str = Query(default="", max_length=200),
    kind: str = Query(default="", max_length=32),
) -> HTMLResponse:
    result: dict[str, Any] = {"hits": [], "broker_error": None}
    # Words, a kind, or both: a kind alone lists what the broker has of it.
    searched = bool(q.strip()) or kind in KINDS
    if searched:
        params = {"q": q, **({"kind": kind} if kind in KINDS else {})}
        try:
            resp = await client.get("/broker/search", params=params, timeout=30.0)
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:
            result = {"hits": [], "broker_error": _pipeline_error(exc, client)}
    return templates.TemplateResponse(
        request, "broker/_search_results.html", {"result": result, "searched": searched}
    )


# ---- the connection ---------------------------------------------------------------------


async def _connection_panel(
    request: Request,
    client: PipelineClientDep,
    *,
    config: dict[str, Any] | None = None,
    probe: dict[str, Any] | None = None,
    error: str | None = None,
    saved: bool = False,
    changed: bool = False,
) -> HTMLResponse:
    if config is None:
        config = await _get_json(client, "/broker/config", _empty_config())
    return templates.TemplateResponse(
        request,
        "broker/_connection.html",
        {"config": config, "probe": probe, "error": error, "saved": saved},
        headers=_CHANGED if changed else None,
    )


def _error_detail(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except Exception:
        detail = None
    if isinstance(detail, dict) and detail.get("message"):
        return str(detail["message"])
    return f"The pipeline refused the change (HTTP {resp.status_code})."


def config_body(form: dict[str, str]) -> tuple[dict[str, Any], str | None]:
    """The connection form as the pipeline's PUT body, and an error when it is incomplete.

    A field is sent only when it changed (or goes back to the deployment's): an unchanged
    value would otherwise become an override of the same value.
    """
    body: dict[str, Any] = {}
    if form.get("url_deployment"):
        body["url"] = None
    else:
        url = form.get("url", "").strip()
        if url.rstrip("/") != form.get("url_current", "").strip().rstrip("/"):
            body["url"] = url
    if form.get("timeout_deployment"):
        body["timeout_seconds"] = None
    else:
        raw = form.get("timeout_seconds", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                return body, "The timeout must be a number of seconds."
            try:
                current: float | None = float(form.get("timeout_current", ""))
            except ValueError:
                current = None
            if timeout != current:
                body["timeout_seconds"] = timeout
    action = form.get("token_action", "keep")
    if action not in _TOKEN_ACTIONS:
        action = "keep"
    if action == "set":
        token = form.get("token", "").strip()
        if not token:
            return body, "Enter the token, or choose another option."
        body["token"] = token
    elif action == "open":
        body["token"] = ""
    elif action == "deployment":
        body["token"] = None
    return body, None


@router.post("/config", response_class=HTMLResponse)
async def save_config(request: Request, client: PipelineClientDep) -> HTMLResponse:
    form = {k: str(v) for k, v in (await request.form()).items()}
    body, error = config_body(form)
    if error is not None:
        return await _connection_panel(request, client, error=error)
    if not body:
        return await _connection_panel(request, client, error="Nothing changed.")
    try:
        resp = await client.put("/broker/config", json=body)
    except Exception as exc:
        return await _connection_panel(request, client, error=_pipeline_error(exc, client))
    if resp.status_code != 200:
        return await _connection_panel(request, client, error=_error_detail(resp))
    return await _connection_panel(request, client, config=resp.json(), saved=True, changed=True)


@router.post("/config/reset", response_class=HTMLResponse)
async def reset_config(request: Request, client: PipelineClientDep) -> HTMLResponse:
    try:
        resp = await client.delete("/broker/config")
        resp.raise_for_status()
    except Exception as exc:
        return await _connection_panel(request, client, error=_pipeline_error(exc, client))
    return await _connection_panel(request, client, config=resp.json(), saved=True, changed=True)


@router.post("/test", response_class=HTMLResponse)
async def test_connection(request: Request, client: PipelineClientDep) -> HTMLResponse:
    try:
        resp = await client.post("/broker/config/test", timeout=30.0)
        resp.raise_for_status()
        probe: dict[str, Any] = resp.json()
    except Exception as exc:
        probe = {"reachable": False, "error": _pipeline_error(exc, client)}
    return await _connection_panel(request, client, probe=probe)
