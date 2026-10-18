"""KB UI FastAPI application."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pacod_id.client import ApiError
from starlette.middleware.sessions import SessionMiddleware

from web.actor import add_actor_header
from web.auth import IdentityUnavailable, NoKbAccess, require_user
from web.identity import Identity, IdSettings
from web.routers import auth_routes as auth_router
from web.routers import members as members_router
from web.routers import pipeline as pipeline_router
from web.routers import qdrant as qdrant_router
from web.sessions import SessionStore, db_url
from web.templates_config import templates

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent


# A UI session lasts at most this long (and ends earlier with pacod-id's session).
SESSION_MAX_AGE = int(os.environ.get("UI_SESSION_MAX_AGE_SECS", "43200") or 43200)
_PURGE_EVERY_SECS = 900.0


# A deletion's erase is retried this many times, waiting this long first (doubling).
ERASE_ATTEMPTS = 5
ERASE_FIRST_WAIT_SECS = 2.0


async def _erase_and_confirm(
    pipeline: httpx.AsyncClient, identity: Identity, sub: str, erasing: set[str]
) -> None:
    """Anonymise a deleted account's traces in the pipeline, then tell pacod-id. Without the
    confirmation pacod-id keeps the tombstone and announces it again on every snapshot load,
    so a failure here is retried then, too."""
    wait = ERASE_FIRST_WAIT_SECS
    try:
        for _attempt in range(ERASE_ATTEMPTS):
            try:
                r = await pipeline.post(f"/people/{quote(sub, safe='')}/erase")
                if r.status_code == 200:
                    await identity.service.deletion_done(sub)
                    logger.info("Erased and confirmed the deletion of %s", sub)
                    return
                logger.warning("Erasing %s: the pipeline answered %d", sub, r.status_code)
            except (httpx.HTTPError, ApiError) as exc:
                logger.warning("Erasing %s failed: %s", sub, exc)
            await asyncio.sleep(wait)
            wait *= 2
    finally:
        erasing.discard(sub)


def _on_identity_event(
    store: SessionStore, identity: Identity, pipeline: httpx.AsyncClient
) -> Callable[[dict[str, Any]], None]:
    """Feed events that end UI sessions (the cache remembers ended sids only 15 minutes), and
    account deletions (also re-announced from each snapshot until confirmed)."""
    erasing: set[str] = set()

    def listener(event: dict[str, Any]) -> None:
        kind = event.get("kind")
        loop = asyncio.get_running_loop()
        if kind == "session.revoked" and event.get("sid"):
            loop.run_in_executor(None, store.delete_by_sid, str(event["sid"]))
        elif kind in ("principal.disabled", "principal.deleted") and event.get("sub"):
            sub = str(event["sub"])
            loop.run_in_executor(None, store.delete_by_sub, sub)
            if kind == "principal.deleted" and sub not in erasing:
                erasing.add(sub)
                identity.spawn(_erase_and_confirm(pipeline, identity, sub, erasing))

    return listener


async def _purge_sessions(app: FastAPI, store: SessionStore) -> None:
    while True:
        await asyncio.sleep(_PURGE_EVERY_SECS)
        try:
            await asyncio.get_running_loop().run_in_executor(None, store.purge_expired)
            live = await asyncio.get_running_loop().run_in_executor(None, store.session_ids)
            locks: dict[str, asyncio.Lock] = app.state.session_locks
            for key in [k for k, lock in locks.items() if k not in live and not lock.locked()]:
                locks.pop(key, None)
        except Exception as exc:  # pragma: no cover - housekeeping only
            logger.warning("UI session purge failed: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Server-side sessions (web/sessions.py) in the UI's SQLite file.
    store = SessionStore(db_url(), _SESSION_SECRET, SESSION_MAX_AGE)
    store.init()
    app.state.session_store = store
    app.state.session_locks = {}

    qdrant_mode = os.environ.get("QDRANT_MODE", "embedded")
    pipeline_api_url = os.environ.get("PIPELINE_API_URL", "http://kb-pipeline:8000")
    pipeline_api_secret = os.environ.get("PIPELINE_API_SECRET", "")
    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")

    headers: dict[str, str] = {}
    if pipeline_api_secret:
        headers["X-API-Key"] = pipeline_api_secret

    pipeline_client = httpx.AsyncClient(
        base_url=pipeline_api_url,
        headers=headers,
        timeout=120.0,
        # Every call says who acts (web/actor.py). Tests install app.state.pipeline_transport.
        event_hooks={"request": [add_actor_header]},
        transport=getattr(app.state, "pipeline_transport", None),
    )

    qdrant_client = None
    if qdrant_mode == "server" and qdrant_url:
        from qdrant_client import QdrantClient

        try:
            qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key or None,
            )
        except Exception as exc:
            logger.warning(
                "Qdrant not reachable at UI startup (%s): %s — "
                "Qdrant management will be unavailable until the service is up.",
                qdrant_url,
                exc,
            )

    app.state.pipeline_client = pipeline_client
    app.state.qdrant_client = qdrant_client
    app.state.qdrant_enabled = qdrant_client is not None

    templates.env.globals["qdrant_enabled"] = app.state.qdrant_enabled

    # pacod-id: built here, never at import (each lifespan has its own event loop). Tests
    # install a fake pacod-id as app.state.id_transport.
    settings = IdSettings.from_env()
    identity = (
        Identity(settings, transport=getattr(app.state, "id_transport", None))
        if settings is not None
        else None
    )
    app.state.identity = identity
    if identity is not None:
        identity.on_event(_on_identity_event(store, identity, pipeline_client))
        identity.start()
    purge = asyncio.create_task(_purge_sessions(app, store))

    try:
        yield
    finally:
        purge.cancel()
        if identity is not None:
            await identity.stop()
        store.close()
        await pipeline_client.aclose()
        if qdrant_client is not None:
            qdrant_client.close()


app = FastAPI(title="KB UI", lifespan=_lifespan)

_INSECURE_SESSION_SECRETS = {"", "change-me-session-secret"}


def _session_secret() -> str:
    """Return UI_SESSION_SECRET, or a random per-process key if unset/default.

    A known signing key lets anyone forge a session cookie for any user, so the
    shipped placeholder is never used.  The random fallback means sessions do
    not survive a restart until a real secret is configured.
    """
    configured = os.environ.get("UI_SESSION_SECRET", "").strip()
    if configured not in _INSECURE_SESSION_SECRETS:
        return configured
    logger.warning(
        "UI_SESSION_SECRET is unset or the default placeholder — using a random "
        "per-process key; users will be logged out on every restart."
    )
    return secrets.token_urlsafe(48)


_SESSION_SECRET = _session_secret()

# The signed cookie carries only the server-side session's secret (and a sign-in in progress).
app.add_middleware(
    SessionMiddleware,
    secret_key=_SESSION_SECRET,
    session_cookie="kb_session",
    max_age=SESSION_MAX_AGE,
    same_site="lax",
    https_only=os.environ.get("UI_SESSION_HTTPS_ONLY", "false").lower() == "true",
)


@app.exception_handler(IdentityUnavailable)
async def _identity_unavailable(request: Request, exc: IdentityUnavailable) -> Response:
    if request.headers.get("hx-request") == "true":
        return PlainTextResponse(f"{exc}.", status_code=503, headers={"Retry-After": "10"})
    return templates.TemplateResponse(
        request,
        "identity_unavailable.html",
        {"reason": str(exc), "retry": request.url.path},
        status_code=503,
        headers={"Retry-After": "10"},
    )


@app.exception_handler(NoKbAccess)
async def _no_kb_access(request: Request, exc: NoKbAccess) -> Response:
    if request.headers.get("hx-request") == "true":
        return PlainTextResponse("No role in the knowledge base.", status_code=403)
    return templates.TemplateResponse(
        request, "no_access.html", {"principal": exc.principal}, status_code=403
    )


_SAFE = ("GET", "HEAD", "OPTIONS")


@app.middleware("http")
async def same_origin_writes(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Refuse a cross-site write: SameSite=Lax keeps the cookie off cross-site POSTs in
    current browsers; this also catches the ones that send it, and every form post."""
    origin = request.headers.get("origin")
    if request.method not in _SAFE and origin and origin != "null":
        identity: Identity | None = getattr(request.app.state, "identity", None)
        expected = (
            identity.settings.ui_origin
            if identity is not None
            else f"{request.url.scheme}://{request.url.netloc}"
        )
        if origin.rstrip("/") != expected:
            return PlainTextResponse("Cross-origin request refused.", status_code=403)
    return await call_next(request)

app.mount("/kb/static", StaticFiles(directory=str(_HERE / "static")), name="static")


_SIDEBAR_TTL_SECS = 5.0
_SIDEBAR_SKIP_PREFIXES = ("/kb/static", "/kb/login", "/kb/logout", "/kb/auth", "/kb/healthz")
_sidebar_cache: dict[str, Any] = {"at": 0.0, "active": "", "projects": []}


async def _fetch_sidebar(client: httpx.AsyncClient) -> tuple[str, list[dict[str, Any]]]:
    async def _get(path: str) -> Any:
        try:
            resp = await client.get(path)
            return resp.json() if resp.status_code == 200 else None
        except Exception:
            return None

    settings, payload = await asyncio.gather(_get("/settings"), _get("/projects"))
    active = str(settings.get("active_project", "")) if isinstance(settings, dict) else ""
    projects = payload.get("projects", []) if isinstance(payload, dict) else []
    return active, [p for p in projects if isinstance(p, dict)] if isinstance(
        projects, list
    ) else []


@app.middleware("http")
async def inject_sidebar_project_context(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Expose the active project + project list to full-page templates.

    Two pipeline calls per request used to run for every asset, HTMX poll and
    login page (and stalled every request when the pipeline was unreachable);
    now they run only for authenticated full-page loads, concurrently, and are
    cached briefly. Any mutating request invalidates the cache.
    """
    request.state.active_project = ""
    request.state.sidebar_projects = []

    path = request.url.path
    wants_sidebar = (
        request.method == "GET"
        and not path.startswith(_SIDEBAR_SKIP_PREFIXES)
        and request.headers.get("hx-request") != "true"
        and "kb_session" in request.cookies
    )
    client: httpx.AsyncClient | None = getattr(request.app.state, "pipeline_client", None)
    if wants_sidebar and client is not None:
        now = time.monotonic()
        if now - float(_sidebar_cache["at"]) > _SIDEBAR_TTL_SECS:
            active, projects = await _fetch_sidebar(client)
            _sidebar_cache.update(at=now, active=active, projects=projects)
        request.state.active_project = _sidebar_cache["active"]
        request.state.sidebar_projects = list(_sidebar_cache["projects"])

    response = await call_next(request)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        _sidebar_cache["at"] = 0.0
    return response


app.include_router(auth_router.router)
app.include_router(members_router.router)
app.include_router(pipeline_router.router)
app.include_router(qdrant_router.router)


@app.get("/kb", response_class=HTMLResponse)
async def index(request: Request, user: Any = Depends(require_user)) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/kb")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_redirect() -> RedirectResponse:
    return RedirectResponse(url="/kb/static/favicon.ico")
