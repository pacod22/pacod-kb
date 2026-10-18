"""Sign-in through pacod-id: /kb/login, the callback, sign-out, and the UI's health check."""

from __future__ import annotations

import hmac
import logging
import re
import time
import urllib.parse
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pacod_id.client import ApiError
from pacod_id.verify import InvalidToken, KeysUnavailable
from starlette.concurrency import run_in_threadpool

from web import oidc
from web.auth import LOGIN_KEY, SESSION_KEY, IdentityUnavailable, current_principal
from web.identity import Identity
from web.templates_config import templates

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb")

# What the login page says for each reason it is shown.
_MESSAGES = {
    "signed_out": "You are signed out.",
    "ended": "Your session ended. Sign in again.",
    "expired": "The sign-in took too long. Try again.",
    "state": "The sign-in could not be matched to this browser. Try again.",
    "issuer": "The sign-in came from an unexpected issuer.",
    "exchange": "pacod-id did not complete the sign-in. Try again.",
    "token": "pacod-id's answer could not be verified.",
    "access_denied": "The sign-in was cancelled.",
    "login_required": "Sign in to continue.",
}
_CODE = re.compile(r"[a-z_]{1,40}")


def _safe_next(raw: str | None) -> str:
    """Only allow same-site relative redirects."""
    if not raw or not raw.startswith("/") or raw.startswith("//"):
        return "/kb"
    # Browsers treat a backslash like "/", so "/\evil.com" means "//evil.com".
    if "\\" in raw or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in raw):
        return "/kb"
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme or parts.netloc:
        return "/kb"
    return raw


def _identity(request: Request) -> Identity | None:
    return getattr(request.app.state, "identity", None)


def _page(request: Request, message: str, next_path: str, status_code: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "login.html",
        {"message": message, "next": next_path},
        status_code=status_code,
    )


@router.get("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    next: str = "/kb",
    signed_out: str | None = None,
    ended: str | None = None,
    error: str | None = None,
) -> Any:
    """Sign in at pacod-id (also pacod-id's initiate_login_uri, after an invitation)."""
    next_path = _safe_next(next)
    identity = _identity(request)
    if identity is None:
        return _page(request, "Sign-in is not set up on this server.", next_path, 503)
    # The sign-in's secrets live in this origin's cookie: start on the configured origin.
    origin = f"{request.url.scheme}://{request.url.netloc}"
    if origin != identity.settings.ui_origin:
        return RedirectResponse(
            f"{identity.settings.ui_origin}{request.url.path}?{request.url.query}", 303
        )
    if signed_out is not None or ended is not None or error is not None:
        code = "signed_out" if signed_out is not None else "ended" if ended is not None else error
        message = _MESSAGES.get(code or "", "")
        if not message:
            shown = code if code and _CODE.fullmatch(code) else "error"
            message = f"The sign-in failed ({shown})."
        return _page(request, message, next_path)
    try:
        if await current_principal(request) is not None:
            return RedirectResponse(next_path, 303)
    except IdentityUnavailable as exc:
        return _page(request, f"{exc}. Try again in a moment.", next_path, 503)
    blob, challenge = oidc.new_login(next_path)
    request.session[LOGIN_KEY] = blob
    url = oidc.authorize_url(identity.settings, state=blob["state"], code_challenge=challenge)
    return RedirectResponse(url, 303)


@router.get("/auth/callback")
async def callback(
    request: Request,
    code: str = "",
    state: str = "",
    iss: str = "",
    error: str = "",
) -> Any:
    identity = _identity(request)
    if identity is None:
        return _page(request, "Sign-in is not set up on this server.", "/kb", 503)

    def fail(reason: str) -> RedirectResponse:
        query = urllib.parse.urlencode({"error": reason, "next": next_path})
        return RedirectResponse(f"/kb/login?{query}", 303)

    blob = request.session.pop(LOGIN_KEY, None)
    next_path = _safe_next(blob.get("next") if isinstance(blob, dict) else None)
    if not isinstance(blob, dict) or time.time() - float(blob.get("at", 0)) > oidc.LOGIN_TTL_SECS:
        return fail("expired")
    if iss != identity.settings.issuer:  # RFC 9207: the answer names its issuer
        return fail("issuer")
    if error:
        return fail(error if _CODE.fullmatch(error) else "exchange")
    if not state or not hmac.compare_digest(state, str(blob.get("state", ""))):
        return fail("state")
    try:
        tokens = await oidc.exchange_code(identity, code, str(blob["verifier"]))
    except ApiError as exc:
        logger.warning("Sign-in code exchange failed: %s", exc)
        return fail("exchange")
    try:
        claims = await oidc.verify(identity, tokens.access_token)
    except KeysUnavailable:
        message = "pacod-id's keys cannot be read. Try again in a moment."
        return _page(request, message, next_path, 503)
    except InvalidToken as exc:
        logger.warning("Sign-in token refused: %s", exc)
        return fail("token")

    store = request.app.state.session_store
    cookie_secret = await run_in_threadpool(
        lambda: store.create(
            sub=claims.sub,
            sid=claims.sid,
            email=claims.email,
            name=claims.name,
            access=tokens.access_token,
            refresh=tokens.refresh_token or "",
            access_expires_at=claims.expires_at,
        )
    )
    # A fresh session: never carry pre-authentication state forward.
    request.session.clear()
    request.session[SESSION_KEY] = cookie_secret
    logger.info("Signed in %s (session %s)", claims.sub, claims.sid)
    return RedirectResponse(next_path, 303)


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    """End the pacod-id session too: revoking the refresh token alone would leave it live."""
    identity = _identity(request)
    cookie_secret = request.session.get(SESSION_KEY)
    store = getattr(request.app.state, "session_store", None)
    if identity is not None and store is not None and isinstance(cookie_secret, str):
        rec = await run_in_threadpool(store.load, cookie_secret)
        if rec is not None:
            try:
                await identity.service.revoke_session(rec.sid)
            except ApiError as exc:
                logger.warning("Ending session %s at pacod-id failed: %s", rec.sid, exc)
            await run_in_threadpool(store.delete, cookie_secret)
    request.session.clear()
    if identity is None:
        return RedirectResponse("/kb/login?signed_out=1", 303)
    return RedirectResponse(oidc.logout_url(identity.settings), 303)


@router.get("/healthz", include_in_schema=False)
async def healthz(request: Request) -> JSONResponse:
    """Liveness for Docker and CI: always 200; says whether sign-in can decide right now."""
    identity = _identity(request)
    state = "off" if identity is None else "ready" if identity.feed_fresh else "stale"
    return JSONResponse({"status": "ok", "identity": state})
