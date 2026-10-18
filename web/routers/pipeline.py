"""Pipeline management routes."""

from __future__ import annotations

import asyncio
import unicodedata
import urllib.parse
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from markupsafe import escape

from web.auth import require_role, require_write
from web.dependencies import PipelineClientDep
from web.templates_config import templates

# Settings that affect every user (AI configuration, prompts, security) are admin-only.
_ADMIN = [Depends(require_role("admin"))]


_TRUTHY = frozenset({"on", "true", "1", "yes"})


def _truthy(value: object) -> bool:
    """HTML form boolean: checkbox "on" and the usual true spellings."""
    return str(value).strip().lower() in _TRUTHY


def _tri_state(value: str) -> bool | None:
    """'auto' → None (use the server default), otherwise a boolean."""
    return None if value == "auto" else _truthy(value)


def _seg(value: object) -> str:
    """Percent-encode one URL path segment for calls to the pipeline API."""
    return urllib.parse.quote(str(value), safe="")


router = APIRouter(prefix="/kb/pipeline", dependencies=[Depends(require_write)])


async def _fetch_documents(client: PipelineClientDep) -> list[dict[str, Any]]:
    """Load document manifests from the pipeline API (store-backed)."""
    try:
        resp = await client.get("/documents")
        if resp.status_code != 200:
            return []
        payload: dict[str, Any] = resp.json()
        docs = payload.get("documents", [])
        if not isinstance(docs, list):
            return []
        return [d for d in docs if isinstance(d, dict)]
    except Exception:
        return []


def _empty_usage_summary() -> dict[str, Any]:
    return {
        "totals": {},
        "timeseries": [],
        "model_series": [],
        "by_model": [],
        "by_provider": [],
        "by_purpose": [],
        "by_user": [],
        "provider_options": [],
        "model_options": [],
    }


async def _get_json(
    client: PipelineClientDep,
    path: str,
    default: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GET *path* from the pipeline API; *default* on any error or non-200."""
    try:
        resp = await client.get(path, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return default


def _empty_gateway_state() -> dict[str, Any]:
    return {"configured": False, "url": "", "converter_model": "", "security_model": ""}


async def _fetch_ai_management_state(client: PipelineClientDep) -> dict[str, Any]:
    """Load the AI Management page's data (six independent pipeline calls, concurrently)."""
    (
        gateway_state,
        llm_state,
        usage_overview,
        usage_summary,
        prompts_state,
        interactions_state,
    ) = await asyncio.gather(
        _get_json(client, "/ai/gateway", _empty_gateway_state()),
        _get_json(
            client, "/llm-config", {"effective": {}, "env_default": {}, "available_models": []}
        ),
        _get_json(client, "/usage-events/overview", {}),
        _get_json(client, "/usage-events/summary", _empty_usage_summary()),
        _get_json(client, "/llm-prompts", {"prompts": []}),
        _get_json(client, "/ai-interactions", {"interactions": [], "total": 0}, {"limit": 50}),
    )
    return {
        "gateway_state": gateway_state,
        "llm_state": llm_state,
        "usage_overview": usage_overview,
        "usage_summary": usage_summary,
        "prompts_state": prompts_state,
        "interactions_state": interactions_state,
        "provider_filter": "",
        "model_filter": "",
    }


def _pipeline_error(exc: Exception, client: httpx.AsyncClient) -> str:
    """Turn a pipeline-API request failure into a clear, actionable message.

    Connection/transport failures (the httpx "All connection attempts failed"
    case) become an explanation that the pipeline container is unreachable;
    HTTP error responses surface the API's JSON ``detail`` when present.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            payload = exc.response.json()
            if isinstance(payload, dict) and "detail" in payload:
                return str(payload["detail"])
        except Exception:
            pass
        return exc.response.text or f"Pipeline API returned HTTP {exc.response.status_code}."
    if isinstance(exc, httpx.TransportError):
        return (
            f"Cannot reach the pipeline API at {client.base_url}. "
            "Make sure the kb-pipeline container is running and healthy "
            "(startup takes 30–60s while BGE-M3 loads)."
        )
    return str(exc)


async def _fetch_projects_data(client: PipelineClientDep) -> dict[str, Any]:
    try:
        resp = await client.get("/projects")
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return data
        return {
            "projects": [],
            "active": None,
            "error": f"Pipeline API returned HTTP {resp.status_code} for /projects.",
        }
    except Exception as exc:
        return {"projects": [], "active": None, "error": _pipeline_error(exc, client)}


def _find_project(projects_data: dict[str, Any], slug: str) -> dict[str, Any] | None:
    projects = projects_data.get("projects", [])
    if not isinstance(projects, list):
        return None
    for project in projects:
        if isinstance(project, dict) and project.get("slug") == slug:
            return project
    return None


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, client: PipelineClientDep) -> HTMLResponse:
    try:
        settings_resp = await client.get("/settings")
        settings: dict[str, Any] = settings_resp.json() if settings_resp.status_code == 200 else {}
    except Exception:
        settings = {}
    try:
        cols_resp = await client.get("/collections")
        cols: dict[str, Any] = (
            cols_resp.json()
            if cols_resp.status_code == 200
            else {"collections": [], "active": None}
        )
    except Exception:
        cols = {"collections": [], "active": None}
    try:
        projects_resp = await client.get("/projects")
        projects_data: dict[str, Any] = (
            projects_resp.json()
            if projects_resp.status_code == 200
            else {"projects": [], "active": None}
        )
    except Exception:
        projects_data = {"projects": [], "active": None}
    active_project: str = settings.get("active_project", "")
    manifests = await _fetch_documents(client)
    active = [m for m in manifests if m.get("status") == "ok" and not m.get("valid_until")]
    superseded = [m for m in manifests if m.get("valid_until")]
    errored = [m for m in manifests if m.get("status") != "ok"]
    return templates.TemplateResponse(
        request,
        "pipeline/index.html",
        {
            "total": len(manifests),
            "active": len(active),
            "superseded": len(superseded),
            "errored": len(errored),
            "settings": settings,
            "cols": cols,
            "projects_data": projects_data,
            "active_project": active_project,
        },
    )


# AI Management shows full prompts and document text: admins only, reads included.
@router.get("/ai-management", response_class=HTMLResponse, dependencies=_ADMIN)
async def ai_management(request: Request, client: PipelineClientDep) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pipeline/ai_management.html",
        await _fetch_ai_management_state(client),
    )


@router.post("/settings/llm-config", response_class=HTMLResponse, dependencies=_ADMIN)
async def update_llm_config(
    request: Request,
    client: PipelineClientDep,
) -> HTMLResponse:
    form = await request.form()
    payload: dict[str, Any] = {}
    for key in ("converter_model", "security_model"):
        value = str(form.get(key, "")).strip()
        if value:
            payload[key] = value
    for key, cast in (("converter_max_tokens", int), ("converter_temperature", float)):
        raw = str(form.get(key, "")).strip()
        if raw:
            try:
                payload[key] = cast(raw)
            except ValueError:
                pass
    payload["converter_temperature_enabled"] = _truthy(
        form.get("converter_temperature_enabled", "")
    )

    error: str | None = None
    try:
        resp = await client.patch("/llm-config", json=payload)
        if resp.status_code == 422:
            error = "Invalid value: model names are provider/model or @pool."
            llm_state: dict[str, Any] = {"effective": payload, "env_default": {}}
        else:
            resp.raise_for_status()
            llm_state = resp.json()
    except Exception as exc:
        error = str(exc)
        llm_state = {"effective": payload, "env_default": {}, "available_models": []}

    return templates.TemplateResponse(
        request,
        "pipeline/_llm_config_panel.html",
        {"llm_state": llm_state, "error": error, "saved": error is None},
    )


@router.post("/ai/ping", response_class=HTMLResponse, dependencies=_ADMIN)
async def ping_gateway(
    request: Request,
    client: PipelineClientDep,
    mode: str = Form("models"),
) -> HTMLResponse:
    """Test the gateway: its model list, or one tiny completion through the converter."""
    ping: dict[str, Any]
    try:
        resp = await client.post(
            "/ai/gateway/ping",
            json={"mode": "completion" if mode == "completion" else "models"},
            timeout=60.0,
        )
        resp.raise_for_status()
        ping = resp.json()
    except Exception as exc:
        ping = {"ok": False, "mode": mode, "error": _pipeline_error(exc, client)}
    gateway_state = await _get_json(client, "/ai/gateway", _empty_gateway_state())
    return templates.TemplateResponse(
        request,
        "pipeline/_gateway_status_panel.html",
        {"gateway_state": gateway_state, "ping": ping},
    )


@router.post("/settings/llm-prompts/{prompt_id}", response_class=HTMLResponse, dependencies=_ADMIN)
async def update_llm_prompt(
    request: Request,
    client: PipelineClientDep,
    prompt_id: str,
    value: str = Form(""),
) -> HTMLResponse:
    error: str | None = None
    try:
        resp = await client.patch(f"/llm-prompts/{_seg(prompt_id)}", json={"value": value})
        resp.raise_for_status()
    except Exception as exc:
        error = str(exc)

    try:
        prompt_resp = await client.get("/llm-prompts")
        prompts_state = prompt_resp.json() if prompt_resp.status_code == 200 else {"prompts": []}
    except Exception:
        prompts_state = {"prompts": []}

    return templates.TemplateResponse(
        request,
        "pipeline/_llm_prompts_panel.html",
        {"prompts_state": prompts_state, "error": error, "saved": error is None},
    )


@router.get("/usage/summary", response_class=HTMLResponse, dependencies=_ADMIN)
async def usage_summary_panel(
    request: Request,
    client: PipelineClientDep,
    bucket: str = Query("day"),
    provider: str = Query(""),
    model: str = Query(""),
    range_start: str = Query(""),
    range_end: str = Query(""),
    metric: str = Query("cost"),
) -> HTMLResponse:
    try:
        overview_resp = await client.get("/usage-events/overview")
        usage_overview: dict[str, Any] = (
            overview_resp.json() if overview_resp.status_code == 200 else {}
        )
    except Exception:
        usage_overview = {}
    try:
        summary_resp = await client.get(
            "/usage-events/summary",
            params={
                "bucket": bucket,
                "provider": provider,
                "model": model,
                "range_start": range_start,
                "range_end": range_end,
            },
        )
        usage_summary: dict[str, Any] = (
            summary_resp.json() if summary_resp.status_code == 200 else _empty_usage_summary()
        )
    except Exception:
        usage_summary = _empty_usage_summary()

    return templates.TemplateResponse(
        request,
        "pipeline/_usage_costs_panel.html",
        {
            "usage_overview": usage_overview,
            "usage_summary": usage_summary,
            "provider_filter": provider,
            "model_filter": model,
            "bucket_filter": bucket,
            "range_start_filter": range_start,
            "range_end_filter": range_end,
            "metric_filter": metric,
        },
    )


@router.get("/ai-interactions", response_class=HTMLResponse, dependencies=_ADMIN)
async def ai_interactions_panel(
    request: Request,
    client: PipelineClientDep,
    provider: str = Query(""),
    purpose: str = Query(""),
    status: str = Query(""),
    turn_id: str = Query(""),
    doc_id: str = Query(""),
    limit: int = Query(50),
    offset: int = Query(0),
) -> HTMLResponse:
    params: dict[str, Any] = {
        "provider": provider,
        "purpose": purpose,
        "status": status,
        "turn_id": turn_id,
        "doc_id": doc_id,
        "limit": limit,
        "offset": offset,
    }
    try:
        resp = await client.get("/ai-interactions", params=params)
        interactions_state: dict[str, Any] = (
            resp.json() if resp.status_code == 200 else {"interactions": [], "total": 0}
        )
    except Exception:
        interactions_state = {"interactions": [], "total": 0}
    return templates.TemplateResponse(
        request,
        "pipeline/_ai_interactions_panel.html",
        {
            "interactions_state": interactions_state,
            "provider_filter": provider,
            "purpose_filter": purpose,
            "status_filter": status,
            "turn_id_filter": turn_id,
            "doc_id_filter": doc_id,
            "limit_filter": limit,
            "offset_filter": offset,
        },
    )


@router.get(
    "/ai-interactions/detail/{interaction_id}",
    response_class=HTMLResponse,
    dependencies=_ADMIN,
)
async def ai_interaction_detail(
    request: Request,
    client: PipelineClientDep,
    interaction_id: int,
) -> HTMLResponse:
    interaction: dict[str, Any] = {}
    error: str | None = None
    try:
        resp = await client.get(f"/ai-interactions/{_seg(interaction_id)}")
        if resp.status_code == 200:
            interaction = resp.json()
        else:
            error = f"Interaction not found (HTTP {resp.status_code})."
    except Exception as exc:
        error = str(exc)
    return templates.TemplateResponse(
        request,
        "pipeline/_ai_interaction_detail.html",
        {"interaction": interaction, "error": error},
    )


@router.post("/settings/security", response_class=HTMLResponse, dependencies=_ADMIN)
async def toggle_security(
    request: Request,
    client: PipelineClientDep,
    enabled: str = Form(""),
) -> HTMLResponse:
    value = _truthy(enabled)
    try:
        resp = await client.post("/settings", json={"content_security_enabled": value})
        resp.raise_for_status()
        settings: dict[str, Any] = resp.json()
    except Exception:
        settings = {"content_security_enabled": value}
    return templates.TemplateResponse(
        request, "pipeline/_security_toggle.html", {"settings": settings}
    )


# Turning the converter on spends money at the gateway: admins only.
@router.post("/settings/converter", response_class=HTMLResponse, dependencies=_ADMIN)
async def toggle_converter(
    request: Request,
    client: PipelineClientDep,
    enabled: str = Form(""),
) -> HTMLResponse:
    value = _truthy(enabled)
    try:
        resp = await client.post("/settings", json={"llm_converter_enabled": value})
        resp.raise_for_status()
        settings: dict[str, Any] = resp.json()
    except Exception:
        settings = {"llm_converter_enabled": value}
    return templates.TemplateResponse(
        request, "pipeline/_converter_toggle.html", {"settings": settings}
    )


@router.post("/settings/language", response_class=HTMLResponse)
async def toggle_language_validation(
    request: Request,
    client: PipelineClientDep,
    enabled: str = Form(""),
) -> HTMLResponse:
    value = _truthy(enabled)
    try:
        resp = await client.post("/settings", json={"language_validation_enabled": value})
        resp.raise_for_status()
        settings: dict[str, Any] = resp.json()
    except Exception:
        settings = {"language_validation_enabled": value}
    return templates.TemplateResponse(
        request, "pipeline/_language_toggle.html", {"settings": settings}
    )


@router.post("/settings/collection", response_class=HTMLResponse)
async def switch_collection(
    request: Request,
    client: PipelineClientDep,
    collection: str = Form(...),
) -> HTMLResponse:
    error: str | None = None
    settings: dict[str, Any] = {}
    try:
        resp = await client.post("/settings", json={"active_collection": collection})
        resp.raise_for_status()
        settings = resp.json()
    except httpx.HTTPStatusError as exc:
        error = (
            "Cannot switch while a job is running."
            if exc.response.status_code == 409
            else exc.response.text
        )
    except Exception as exc:
        error = str(exc)
    try:
        cols_resp = await client.get("/collections")
        cols: dict[str, Any] = (
            cols_resp.json()
            if cols_resp.status_code == 200
            else {"collections": [], "active": settings.get("active_collection")}
        )
    except Exception:
        cols = {"collections": [], "active": settings.get("active_collection")}
    return templates.TemplateResponse(
        request,
        "pipeline/_collection_selector.html",
        {"cols": cols, "settings": settings, "error": error},
    )


@router.post("/settings/project", response_class=HTMLResponse)
async def switch_project(
    request: Request,
    client: PipelineClientDep,
    project: str = Form(...),
    sidebar: str = Form(""),
) -> HTMLResponse:
    error: str | None = None
    settings: dict[str, Any] = {}
    try:
        resp = await client.post("/settings", json={"active_project": project})
        resp.raise_for_status()
        settings = resp.json()
    except httpx.HTTPStatusError as exc:
        error = (
            "Cannot switch while a job is running."
            if exc.response.status_code == 409
            else exc.response.text
        )
    except Exception as exc:
        error = str(exc)
    try:
        projects_resp = await client.get("/projects")
        projects_data: dict[str, Any] = (
            projects_resp.json()
            if projects_resp.status_code == 200
            else {"projects": [], "active": None}
        )
    except Exception:
        projects_data = {"projects": [], "active": None}
    active_project: str = settings.get("active_project", project)
    is_sidebar = _truthy(sidebar)
    if is_sidebar:
        return templates.TemplateResponse(
            request,
            "pipeline/_sidebar_project_switcher.html",
            {
                "sidebar_projects": projects_data.get("projects", []),
                "active_project": active_project,
                "error": error,
            },
        )
    return templates.TemplateResponse(
        request,
        "pipeline/_project_selector.html",
        {
            "projects_data": projects_data,
            "settings": settings,
            "active_project": active_project,
            "error": error,
        },
    )


@router.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request, client: PipelineClientDep) -> HTMLResponse:
    projects_data = await _fetch_projects_data(client)
    error = request.query_params.get("error") or projects_data.get("error")
    return templates.TemplateResponse(
        request, "pipeline/projects.html", {"projects_data": projects_data, "error": error}
    )


@router.get("/projects/new", response_class=HTMLResponse)
async def project_create_view(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pipeline/project_form.html",
        {
            "mode": "create",
            "title": "Create project",
            "submit_label": "Create project",
            "action_url": "/kb/pipeline/projects",
            "form_values": {"name": "", "description": "", "collection_name": ""},
            "blocked_reasons": [],
        },
    )


@router.get("/projects/{slug}/edit", response_class=HTMLResponse)
async def project_edit_view(
    slug: str,
    request: Request,
    client: PipelineClientDep,
) -> HTMLResponse:
    projects_data = await _fetch_projects_data(client)
    project = _find_project(projects_data, slug)
    if project is None:
        return templates.TemplateResponse(
            request,
            "pipeline/project_form.html",
            {
                "mode": "edit",
                "title": "Edit project",
                "submit_label": "Save",
                "action_url": f"/kb/pipeline/projects/{slug}/edit",
                "error": f"Project '{slug}' not found.",
                "project": {"slug": slug, "name": slug, "collection_name": "", "doc_count": 0},
                "form_values": {"collection_name": ""},
                "edit_blocked": True,
                "blocked_reasons": ["Project not found"],
            },
            status_code=404,
        )

    active_slug = str(projects_data.get("active") or "")
    try:
        doc_count = int(project.get("doc_count", 0) or 0)
    except (TypeError, ValueError):
        doc_count = 0
    edit_blocked = (slug == active_slug) or (doc_count > 0)
    blocked_reasons = []
    if slug == active_slug:
        blocked_reasons.append("This is the active project")
    if doc_count > 0:
        blocked_reasons.append(f"It has {doc_count} document{'s' if doc_count != 1 else ''}")
    return templates.TemplateResponse(
        request,
        "pipeline/project_form.html",
        {
            "mode": "edit",
            "title": "Edit project",
            "submit_label": "Save",
            "action_url": f"/kb/pipeline/projects/{slug}/edit",
            "project": project,
            "form_values": {"collection_name": project.get("collection_name") or slug},
            "edit_blocked": edit_blocked,
            "blocked_reasons": blocked_reasons,
        },
    )


@router.post("/projects")
async def create_project(
    request: Request,
    client: PipelineClientDep,
    name: str = Form(...),
    description: str = Form(""),
    collection_name: str = Form(""),
) -> Any:
    error: str | None = None
    try:
        resp = await client.post(
            "/projects",
            json={
                "name": name,
                "description": description,
                "collection_name": collection_name,
            },
        )
        resp.raise_for_status()
        return RedirectResponse("/kb/pipeline/projects", status_code=303)
    except Exception as exc:
        error = _pipeline_error(exc, client)
    return templates.TemplateResponse(
        request,
        "pipeline/project_form.html",
        {
            "mode": "create",
            "title": "Create project",
            "submit_label": "Create project",
            "action_url": "/kb/pipeline/projects",
            "error": error,
            "form_values": {
                "name": name,
                "description": description,
                "collection_name": collection_name,
            },
            "blocked_reasons": [],
        },
    )


@router.post("/projects/{slug}/activate")
async def activate_project(
    slug: str,
    request: Request,
    client: PipelineClientDep,
) -> Any:
    try:
        resp = await client.post("/settings", json={"active_project": slug})
        resp.raise_for_status()
    except Exception as exc:
        qs = urllib.parse.urlencode({"error": _pipeline_error(exc, client)})
        return RedirectResponse(f"/kb/pipeline/projects?{qs}", status_code=303)
    return RedirectResponse("/kb/pipeline/projects", status_code=303)


@router.post("/projects/{slug}/edit")
async def edit_project(
    slug: str,
    request: Request,
    client: PipelineClientDep,
    collection_name: str = Form(""),
) -> Any:
    error: str | None = None
    if not collection_name.strip():
        error = "Collection name is required."
    else:
        try:
            resp = await client.patch(
                f"/projects/{_seg(slug)}",
                json={"collection_name": collection_name},
            )
            resp.raise_for_status()
            return RedirectResponse("/kb/pipeline/projects", status_code=303)
        except Exception as exc:
            error = _pipeline_error(exc, client)
    projects_data = await _fetch_projects_data(client)
    project = _find_project(projects_data, slug) or {
        "slug": slug,
        "name": slug,
        "collection_name": collection_name,
        "doc_count": 0,
    }
    active_slug = str(projects_data.get("active") or "")
    try:
        doc_count = int(project.get("doc_count", 0) or 0)
    except (TypeError, ValueError):
        doc_count = 0
    edit_blocked = (slug == active_slug) or (doc_count > 0)
    blocked_reasons = []
    if slug == active_slug:
        blocked_reasons.append("This is the active project")
    if doc_count > 0:
        blocked_reasons.append(f"It has {doc_count} document{'s' if doc_count != 1 else ''}")
    return templates.TemplateResponse(
        request,
        "pipeline/project_form.html",
        {
            "mode": "edit",
            "title": "Edit project",
            "submit_label": "Save",
            "action_url": f"/kb/pipeline/projects/{slug}/edit",
            "project": project,
            "error": error,
            "form_values": {
                "collection_name": collection_name or project.get("collection_name") or slug
            },
            "edit_blocked": edit_blocked,
            "blocked_reasons": blocked_reasons,
        },
    )


@router.post("/projects/{slug}/delete", dependencies=_ADMIN)
async def delete_project(
    slug: str,
    request: Request,
    client: PipelineClientDep,
) -> Any:
    try:
        resp = await client.delete(f"/projects/{_seg(slug)}")
        resp.raise_for_status()
    except Exception as exc:
        qs = urllib.parse.urlencode({"error": _pipeline_error(exc, client)})
        return RedirectResponse(f"/kb/pipeline/projects?{qs}", status_code=303)
    return RedirectResponse("/kb/pipeline/projects", status_code=303)


@router.get("/documents", response_class=HTMLResponse)
async def documents(request: Request, client: PipelineClientDep, status: str = "") -> HTMLResponse:
    try:
        settings_resp = await client.get("/settings")
        settings: dict[str, Any] = settings_resp.json() if settings_resp.status_code == 200 else {}
    except Exception:
        settings = {}
    active_project: str = settings.get("active_project", "")
    manifests = await _fetch_documents(client)
    if status == "active":
        manifests = [m for m in manifests if m.get("status") == "ok" and not m.get("valid_until")]
    elif status == "superseded":
        manifests = [m for m in manifests if m.get("valid_until")]
    elif status == "error":
        manifests = [m for m in manifests if m.get("status") != "ok"]
    return templates.TemplateResponse(
        request,
        "pipeline/documents.html",
        {"manifests": manifests, "status": status, "active_project": active_project},
    )


@router.get("/documents/{doc_id}", response_class=HTMLResponse)
async def document_detail(
    request: Request,
    doc_id: str,
    client: PipelineClientDep,
) -> HTMLResponse:
    try:
        resp = await client.get(f"/documents/{_seg(doc_id)}")
    except Exception:
        resp = None
    if resp is None or resp.status_code == 404:
        return HTMLResponse(
            "<div class='alert alert-danger'>Document not found.</div>",
            status_code=404,
        )
    if resp.status_code >= 400:
        return HTMLResponse(
            f"<div class='alert alert-danger'>Failed to load document: {escape(resp.text)}</div>",
            status_code=resp.status_code,
        )

    data: dict[str, Any] = resp.json()
    manifest = data.get("manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    body = str(data.get("body", "") or "")
    superseded_by_raw = data.get("superseded_by")
    superseded_by: dict[str, Any] | None = (
        superseded_by_raw if isinstance(superseded_by_raw, dict) else None
    )
    source_size_raw = data.get("source_size")
    source_size: int | None = int(source_size_raw) if isinstance(source_size_raw, int) else None

    return templates.TemplateResponse(
        request,
        "pipeline/document.html",
        {
            "manifest": manifest,
            "body": body,
            "superseded_by": superseded_by,
            "source_size": source_size,
        },
    )


@router.get("/documents/{doc_id}/download/original")
async def download_document_original(doc_id: str, client: PipelineClientDep) -> Any:
    resp = await client.get(f"/documents/{_seg(doc_id)}/download/original")
    resp.raise_for_status()
    cd = resp.headers.get(
        "content-disposition",
        f'attachment; filename="{doc_id[:8]}.bin"',
    )
    media_type = resp.headers.get("content-type", "application/octet-stream")
    return Response(
        content=resp.content,
        media_type=media_type,
        headers={"Content-Disposition": cd},
    )


@router.get("/documents/{doc_id}/download/markdown")
async def download_document_markdown(doc_id: str, client: PipelineClientDep) -> Any:
    resp = await client.get(f"/documents/{_seg(doc_id)}/download/markdown")
    resp.raise_for_status()
    cd = resp.headers.get(
        "content-disposition",
        f'attachment; filename="{doc_id[:8]}.md"',
    )
    media_type = resp.headers.get("content-type", "text/markdown; charset=utf-8")
    return Response(
        content=resp.content,
        media_type=media_type,
        headers={"Content-Disposition": cd},
    )


@router.delete("/documents/{doc_id}", response_class=HTMLResponse)
async def delete_document_route(
    request: Request, doc_id: str, client: PipelineClientDep
) -> HTMLResponse:
    try:
        resp = await client.delete(f"/documents/{_seg(doc_id)}")
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return HTMLResponse(
            "<div class='alert alert-danger mt-3'>"
            f"Delete failed: {escape(exc.response.text)}</div>",
            status_code=exc.response.status_code,
        )
    return HTMLResponse("", headers={"HX-Redirect": "/kb/pipeline/documents"})


@router.post("/documents/delete-bulk", response_class=HTMLResponse)
async def delete_bulk(request: Request, client: PipelineClientDep) -> HTMLResponse:
    form = await request.form()
    doc_ids = [str(v) for v in form.getlist("doc_ids")]
    if not doc_ids:
        return HTMLResponse("<div class='alert alert-warning'>No documents selected.</div>")
    errors: list[str] = []
    for did in doc_ids:
        try:
            resp = await client.delete(f"/documents/{_seg(did)}")
            resp.raise_for_status()
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        deleted = len(doc_ids) - len(errors)
        return HTMLResponse(
            f"<div class='alert alert-danger'>"
            f"Deleted {deleted} document(s); {len(errors)} failed: {escape('; '.join(errors))}"
            f"</div>"
        )
    return HTMLResponse("", headers={"HX-Redirect": "/kb/pipeline/documents"})


@router.get("/ingest", response_class=HTMLResponse)
async def ingest_form(
    request: Request, client: PipelineClientDep, supersedes: str = ""
) -> HTMLResponse:
    from datetime import date

    manifests = await _fetch_documents(client)
    documents = [
        {"doc_id": m.get("doc_id", ""), "title": m.get("title", "Untitled")}
        for m in manifests
        if m.get("doc_id") and m.get("status") == "ok" and not m.get("valid_until")
    ]
    try:
        settings_resp = await client.get("/settings")
        settings: dict[str, Any] = settings_resp.json() if settings_resp.status_code == 200 else {}
    except Exception:
        settings = {}
    return templates.TemplateResponse(
        request,
        "pipeline/ingest.html",
        {
            "supersedes_prefill": supersedes,
            "documents": documents,
            "today": date.today().isoformat(),
            "settings": settings,
        },
    )


@router.post("/ingest", response_class=HTMLResponse)
async def ingest_submit(
    request: Request,
    client: PipelineClientDep,
    ingest_type: str = Form("url"),
    url: str = Form(""),
    doc_type: str = Form("generic"),
    source_version: str = Form(""),
    supersedes: str = Form(""),
    file: UploadFile | None = File(None),
    json_convert: str = Form("true"),
    llm_convert: str = Form("auto"),
    validate_language: str = Form("auto"),
    trusted: str = Form("auto"),
) -> HTMLResponse:
    _json_convert = json_convert.lower() not in ("false", "0", "off", "no")
    # "auto" → None (use global runtime setting); "true"/"false" → explicit override
    _llm_convert: bool | None = _tri_state(llm_convert)
    _validate_language: bool | None = _tri_state(validate_language)
    # "auto" → False (API applies global bypass on top); "true" → always bypass
    _trusted: bool = trusted.lower() in ("true", "1", "bypass", "yes")
    try:
        if ingest_type == "file" and file is not None:
            data: dict[str, Any] = {
                "doc_type": (None, doc_type),
                "source_version": (None, source_version),
                "supersedes": (None, supersedes or ""),
                "skip_index": (None, "false"),
                "json_convert": (None, "true" if _json_convert else "false"),
                "llm_convert": (None, llm_convert),
                "validate_language": (None, validate_language),
                "trusted": (None, "true" if _trusted else "false"),
            }
            content = await file.read()
            safe_name = unicodedata.normalize("NFC", (file.filename or "upload"))
            # Percent-encode the filename so non-ASCII characters survive the HTTP header
            encoded_name = urllib.parse.quote(safe_name, safe="._-")
            files = {
                "file": (encoded_name, content, file.content_type or "application/octet-stream")
            }
            resp = await client.post("/ingest/file", data=data, files=files)
        else:
            payload: dict[str, Any] = {
                "url": url,
                "doc_type": doc_type,
                "source_version": source_version,
                "supersedes": supersedes or None,
                "skip_index": False,
                "json_convert": _json_convert,
                "llm_convert": _llm_convert,
                "validate_language": _validate_language,
                "trusted": _trusted,
            }
            resp = await client.post("/ingest/url", json=payload)
        resp.raise_for_status()
        job_id: str = resp.json()["job_id"]
    except httpx.HTTPStatusError as exc:
        return templates.TemplateResponse(
            request,
            "pipeline/job_status.html",
            {"error": exc.response.text, "job_id": None},
        )
    return templates.TemplateResponse(
        request,
        "pipeline/job_status.html",
        {"job_id": job_id, "error": None, "polling": True},
    )


@router.post("/ingest/urls", response_class=HTMLResponse)
async def ingest_urls_submit(
    request: Request,
    client: PipelineClientDep,
    urls_text: str = Form(""),
    doc_type: str = Form("generic"),
    source_version: str = Form(""),
    llm_convert: str = Form("auto"),
    validate_language: str = Form("auto"),
    trusted: str = Form("auto"),
) -> HTMLResponse:
    _trusted_bulk: bool = trusted.lower() in ("true", "1", "bypass", "yes")
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if not urls:
        return templates.TemplateResponse(
            request,
            "pipeline/job_status.html",
            {"error": "No URLs provided.", "job_id": None},
        )
    _llm_convert: bool | None = _tri_state(llm_convert)
    _validate_language: bool | None = _tri_state(validate_language)
    payload: dict[str, Any] = {
        "urls": urls,
        "doc_type": doc_type,
        "source_version": source_version,
        "skip_index": False,
        "llm_convert": _llm_convert,
        "validate_language": _validate_language,
        "trusted": _trusted_bulk,
    }
    try:
        resp = await client.post("/ingest/urls", json=payload)
        resp.raise_for_status()
        job_id: str = resp.json()["job_id"]
    except httpx.HTTPStatusError as exc:
        return templates.TemplateResponse(
            request,
            "pipeline/job_status.html",
            {"error": exc.response.text, "job_id": None},
        )
    return templates.TemplateResponse(
        request,
        "pipeline/job_status.html",
        {"job_id": job_id, "error": None, "polling": True},
    )


@router.get("/query", response_class=HTMLResponse)
async def query_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "pipeline/query.html", {"results": None})


@router.post("/query", response_class=HTMLResponse)
async def query_submit(
    request: Request,
    client: PipelineClientDep,
    query: str = Form(...),
    top_k: int = Form(5),
) -> HTMLResponse:
    try:
        # Human-facing search view: request unwrapped text (the untrusted-data
        # fence is for downstream LLM consumers, not the browser).
        resp = await client.post(
            "/query", json={"query": query, "top_k": top_k, "wrap_untrusted": False}
        )
        resp.raise_for_status()
        results: list[dict[str, Any]] = resp.json().get("results", [])
    except httpx.HTTPStatusError as exc:
        return templates.TemplateResponse(
            request,
            "pipeline/_query_results.html",
            {"results": [], "error": exc.response.text, "query": query},
        )
    return templates.TemplateResponse(
        request,
        "pipeline/_query_results.html",
        {"results": results, "query": query, "error": None},
    )


@router.post("/reindex/{doc_id}", response_class=HTMLResponse)
async def reindex(request: Request, doc_id: str, client: PipelineClientDep) -> HTMLResponse:
    resp = await client.post(f"/reindex/{_seg(doc_id)}")
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    return templates.TemplateResponse(
        request,
        "pipeline/job_status.html",
        {"job_id": job_id, "error": None, "polling": True},
    )


@router.post("/rebuild", response_class=HTMLResponse)
async def rebuild(request: Request, client: PipelineClientDep) -> HTMLResponse:
    resp = await client.post("/rebuild")
    if resp.status_code == 409:
        return templates.TemplateResponse(
            request,
            "pipeline/job_status.html",
            {"job_id": None, "error": "Another job is already running", "polling": False},
        )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    return templates.TemplateResponse(
        request,
        "pipeline/job_status.html",
        {"job_id": job_id, "error": None, "polling": True},
    )


@router.post("/dropzone/scan", response_class=HTMLResponse)
async def dropzone_scan(request: Request, client: PipelineClientDep) -> HTMLResponse:
    resp = await client.post("/dropzone/scan")
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    return templates.TemplateResponse(
        request,
        "pipeline/job_status.html",
        {"job_id": job_id, "error": None, "polling": True},
    )


@router.post("/watch", response_class=HTMLResponse)
async def watch(request: Request, client: PipelineClientDep) -> HTMLResponse:
    resp = await client.post("/watch")
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    return templates.TemplateResponse(
        request,
        "pipeline/job_status.html",
        {"job_id": job_id, "error": None, "polling": True},
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_status(request: Request, job_id: str, client: PipelineClientDep) -> HTMLResponse:
    resp = await client.get(f"/jobs/{_seg(job_id)}")
    if resp.status_code == 404:
        return templates.TemplateResponse(
            request,
            "pipeline/job_status.html",
            {"job_id": job_id, "error": "Job not found", "polling": False},
        )
    resp.raise_for_status()
    job = resp.json()
    done = job["status"] in ("done", "error")
    response = templates.TemplateResponse(
        request,
        "pipeline/job_status.html",
        {"job_id": job_id, "job": job, "polling": not done, "error": None},
    )
    if done:
        response.headers["HX-Trigger"] = "stop-poll"
    return response


@router.get("/jobs/{job_id}/view", response_class=HTMLResponse)
async def job_detail_page(request: Request, job_id: str, client: PipelineClientDep) -> HTMLResponse:
    resp = await client.get(f"/jobs/{_seg(job_id)}")
    if resp.status_code == 404:
        return templates.TemplateResponse(
            request,
            "pipeline/job_detail.html",
            {"job_id": job_id, "job": None, "error": "Job not found", "polling": False},
            status_code=404,
        )
    resp.raise_for_status()
    job = resp.json()
    done = job["status"] in ("done", "error")
    return templates.TemplateResponse(
        request,
        "pipeline/job_detail.html",
        {"job_id": job_id, "job": job, "polling": not done, "error": None},
    )


@router.get("/projects/{slug}/view", response_class=HTMLResponse)
async def project_detail_view(
    slug: str,
    request: Request,
    client: PipelineClientDep,
) -> HTMLResponse:
    """Display project detail page with all maintenance tasks."""
    projects_data = await _fetch_projects_data(client)
    project = _find_project(projects_data, slug)
    active_slug = str(projects_data.get("active") or "")

    if project is None:
        return templates.TemplateResponse(
            request,
            "pipeline/projects.html",
            {"projects_data": projects_data, "error": f"Project '{slug}' not found"},
            status_code=404,
        )

    return templates.TemplateResponse(
        request,
        "pipeline/project_detail.html",
        {
            "project": project,
            "active_slug": active_slug,
        },
    )


async def _render_jobs_table(
    request: Request,
    client: PipelineClientDep,
    *,
    error: str | None = None,
) -> HTMLResponse:
    ctx = await _jobs_context(client, error=error)
    return templates.TemplateResponse(
        request,
        "pipeline/_jobs_table.html",
        ctx,
    )


async def _jobs_context(
    client: PipelineClientDep,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    resp = await client.get("/jobs")
    resp.raise_for_status()
    jobs: list[dict[str, Any]] = resp.json().get("jobs", [])
    has_running = any(str(j.get("status", "")) == "running" for j in jobs)
    return {"jobs": jobs, "has_running": has_running, "jobs_error": error}


@router.get("/jobs", response_class=HTMLResponse)
async def list_jobs(request: Request, client: PipelineClientDep) -> HTMLResponse:
    ctx = await _jobs_context(client)
    if request.headers.get("hx-request") == "true":
        return templates.TemplateResponse(request, "pipeline/_jobs_table.html", ctx)
    return templates.TemplateResponse(request, "pipeline/jobs.html", ctx)


@router.post("/jobs/{job_id}/restart", response_class=HTMLResponse)
async def restart_job(request: Request, job_id: str, client: PipelineClientDep) -> Any:
    error: str | None = None
    restarted_job_id: str | None = None
    try:
        resp = await client.post(f"/jobs/{_seg(job_id)}/restart")
        if resp.status_code < 400:
            restarted_job_id = str(resp.json().get("job_id") or "").strip() or None
        if resp.status_code >= 400:
            detail = (
                resp.json().get("detail")
                if resp.headers.get("content-type", "").startswith("application/json")
                else None
            )
            error = str(detail or f"failed to restart job {job_id[:8]}")
    except Exception as exc:
        error = str(exc)
    if request.headers.get("hx-request") != "true" and error is None and restarted_job_id:
        return RedirectResponse(
            url=f"/kb/pipeline/jobs/{restarted_job_id}/view",
            status_code=303,
        )
    return await _render_jobs_table(request, client, error=error)


@router.post("/jobs/{job_id}/delete", response_class=HTMLResponse)
async def delete_job(request: Request, job_id: str, client: PipelineClientDep) -> Any:
    error: str | None = None
    try:
        resp = await client.delete(f"/jobs/{_seg(job_id)}")
        if resp.status_code >= 400:
            detail = (
                resp.json().get("detail")
                if resp.headers.get("content-type", "").startswith("application/json")
                else None
            )
            error = str(detail or f"failed to delete job {job_id[:8]}")
    except Exception as exc:
        error = str(exc)
    if request.headers.get("hx-request") != "true" and error is None:
        return RedirectResponse(url="/kb/pipeline/jobs", status_code=303)
    return await _render_jobs_table(request, client, error=error)


@router.get("/export")
async def export_documents(client: PipelineClientDep) -> Any:
    resp = await client.get("/export")
    resp.raise_for_status()
    cd = resp.headers.get("content-disposition", 'attachment; filename="kb_export.json"')
    return Response(
        content=resp.content,
        media_type="application/json",
        headers={"Content-Disposition": cd},
    )


@router.get("/export/sources")
async def export_sources(
    client: PipelineClientDep,
    doc_ids: list[str] = Query(default=[]),
) -> Any:
    params: dict[str, Any] = {}
    if doc_ids:
        params["doc_ids"] = doc_ids
    resp = await client.get("/export/sources", params=params)
    resp.raise_for_status()
    cd = resp.headers.get("content-disposition", 'attachment; filename="kb_sources.json"')
    return Response(
        content=resp.content,
        media_type="application/json",
        headers={"Content-Disposition": cd},
    )


@router.get("/export/sources/csv")
async def export_sources_csv(
    client: PipelineClientDep,
    doc_ids: list[str] = Query(default=[]),
) -> Any:
    params: dict[str, Any] = {"format": "csv"}
    if doc_ids:
        params["doc_ids"] = doc_ids
    resp = await client.get("/export/sources", params=params)
    resp.raise_for_status()
    cd = resp.headers.get("content-disposition", 'attachment; filename="kb_sources.csv"')
    return Response(
        content=resp.content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": cd},
    )


@router.post("/projects/{slug}/regenerate-from-vector-db", response_class=HTMLResponse)
async def regenerate_from_vector_db(
    request: Request, slug: str, client: PipelineClientDep
) -> HTMLResponse:
    """Regenerate documents list based on data from Qdrant collection."""
    try:
        resp = await client.post(f"/projects/{_seg(slug)}/regenerate-from-vector-db")
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        return templates.TemplateResponse(
            request,
            "pipeline/job_status.html",
            {"job_id": job_id, "error": None, "polling": True},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "pipeline/job_status.html",
            {"job_id": "", "error": f"Failed to start regeneration: {exc}", "polling": False},
        )


@router.post("/projects/{slug}/convert-without-index", response_class=HTMLResponse)
async def convert_without_index(
    request: Request, slug: str, client: PipelineClientDep
) -> HTMLResponse:
    """Convert documents without sending to the vector database."""
    try:
        resp = await client.post(f"/projects/{_seg(slug)}/convert-without-index")
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        return templates.TemplateResponse(
            request,
            "pipeline/job_status.html",
            {"job_id": job_id, "error": None, "polling": True},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "pipeline/job_status.html",
            {"job_id": "", "error": f"Failed to start conversion: {exc}", "polling": False},
        )
