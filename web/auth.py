"""Who is asking: sign-in through pacod-id, server-side sessions, and kb's role checks.

A person signs in at pacod-id (``web/oidc.py``); the UI keeps their tokens in a server-side
session (``web/sessions.py``) and only that session's random secret in the signed cookie.
Each request resolves the person once (:func:`current_principal`, kept on ``request.state``):

1. the session row — missing or expired: sign in;
2. pacod-id's feed — sign-in not set up, or the copy older than 60 s: 503 page
   (who may do what is not known);
3. the session ended or the account disabled in pacod-id: the row goes, sign in again;
4. an access token about to expire is refreshed first — one refresh per session at a time,
   persisted before it is used (pacod-id rotates refresh tokens; replaying a replaced one
   later than 30 s signs the person out everywhere);
5. the kb role from the feed's bindings; none: the "no access" page.

Role model (least -> most privileged): ``viewer`` < ``editor`` < ``admin`` (``web.roles``).
Dependencies: ``require_user``, ``require_role(r)``, ``require_write`` (reads for any role,
mutations for editors).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from fastapi import Depends, HTTPException, Request, status
from pacod_id.client import ApiError
from pacod_id.verify import InvalidToken, KeysUnavailable
from starlette.concurrency import run_in_threadpool

from web.actor import current_actor
from web.identity import Identity
from web.roles import role_satisfies
from web.sessions import SessionRecord, SessionStore

logger = logging.getLogger(__name__)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# POST routes that only read data (search forms) and stay open to viewers.
_READ_ONLY_POST_ROUTES = frozenset({"/kb/pipeline/query", "/kb/qdrant/{collection}/search"})

# The signed cookie holds only this: the server-side session's random secret.
SESSION_KEY = "s"
# A sign-in in progress (state, PKCE verifier, where to go next).
LOGIN_KEY = "login"
# Refresh an access token this long before it expires.
REFRESH_MARGIN_SECS = 30


@dataclass(frozen=True)
class Principal:
    """The signed-in person, as kb sees them."""

    sub: str  # pacod-id's usr_ id
    sid: str  # their pacod-id session
    email: str
    name: str
    role: str  # "admin" | "editor" | "viewer" | "" (no kb role)

    @property
    def id(self) -> str:
        return self.sub


class IdentityUnavailable(Exception):  # noqa: N818 - a state, not an error to fix
    """Who may do what cannot be decided now: 503 page."""


class NoKbAccess(Exception):  # noqa: N818 - a state, not an error to fix
    """Signed in, but without a kb role: 403 page."""

    def __init__(self, principal: Principal) -> None:
        super().__init__(principal.email)
        self.principal = principal


def _identity(request: Request) -> Identity | None:
    return getattr(request.app.state, "identity", None)


def _store(request: Request) -> SessionStore:
    store: SessionStore = request.app.state.session_store
    return store


def _locks(request: Request) -> dict[str, asyncio.Lock]:
    locks: dict[str, asyncio.Lock] = request.app.state.session_locks
    return locks


async def refresh_session(
    identity: Identity,
    store: SessionStore,
    locks: dict[str, asyncio.Lock],
    cookie_secret: str,
    rec: SessionRecord,
    *,
    margin: int = REFRESH_MARGIN_SECS,
) -> SessionRecord | None:
    """The session with an access token valid for *margin* seconds, or None when pacod-id
    ended it. Raises IdentityUnavailable when pacod-id cannot be reached (the row stays)."""
    lock = locks.setdefault(rec.id_hash, asyncio.Lock())
    async with lock:
        # Another request may have refreshed while this one waited: use its tokens.
        cur = await run_in_threadpool(store.load, cookie_secret)
        if cur is None:
            return None
        if cur.access_expires_at - time.time() >= margin:
            return cur
        tokens = None
        failure: ApiError | None = None
        for _attempt in (1, 2):
            try:
                tokens = await identity.ui.refresh(cur.refresh_token)
                break
            except ApiError as exc:
                failure = exc
                if 0 < exc.status < 500:
                    break  # refused: retrying cannot help
                # Unreachable or failing: once more at once. A lost answer retried within
                # pacod-id's 30-second grace returns the same successor; later it is reuse.
        if tokens is None:
            assert failure is not None
            if not 0 < failure.status < 500:
                raise IdentityUnavailable("pacod-id cannot be reached") from failure
            # invalid_grant: the grant ended (sign-out, session end, password change).
            logger.info("Session refresh refused (%s): signing out", failure.code)
            await run_in_threadpool(store.delete, cookie_secret)
            return None
        from web import oidc

        try:
            claims = await oidc.verify(identity, tokens.access_token)
        except KeysUnavailable as exc:
            raise IdentityUnavailable("pacod-id's keys cannot be read") from exc
        except InvalidToken:
            await run_in_threadpool(store.delete, cookie_secret)
            return None
        # Persist before use: the old refresh token is spent now.
        saved = await run_in_threadpool(
            lambda: store.update_tokens(
                cookie_secret,
                expected_version=cur.version,
                access=tokens.access_token,
                refresh=tokens.refresh_token or cur.refresh_token,
                access_expires_at=claims.expires_at,
                email=claims.email or cur.email,
                name=claims.name or cur.name,
            )
        )
        # Saved, or another writer won the compare-and-swap: either way the row is current.
        if not saved:
            logger.debug("Session %s was updated concurrently", rec.id_hash[:8])
        return await run_in_threadpool(store.load, cookie_secret)


async def current_principal(request: Request) -> Principal | None:
    """The signed-in person (resolved once per request), or None when nobody is."""
    if hasattr(request.state, "principal"):
        cached: Principal | None = request.state.principal
        return cached
    identity = _identity(request)
    if identity is None:
        raise IdentityUnavailable("Sign-in is not set up")
    session = request.session if "session" in request.scope else {}
    cookie_secret = session.get(SESSION_KEY)
    principal: Principal | None = None
    if isinstance(cookie_secret, str) and cookie_secret:
        store = _store(request)
        rec = await run_in_threadpool(store.load, cookie_secret)
        if rec is None:
            # The row is gone: expired, or ended by pacod-id (the feed deletes those rows).
            session.pop(SESSION_KEY, None)
            request.state.session_ended = True
        else:
            if not identity.feed_fresh:
                raise IdentityUnavailable("pacod-id cannot be reached")
            if identity.cache.sid_revoked(rec.sid) or identity.cache.is_disabled(rec.sub):
                await run_in_threadpool(store.delete, cookie_secret)
                session.pop(SESSION_KEY, None)
                request.state.session_ended = True
                rec = None
            elif rec.access_expires_at - time.time() < REFRESH_MARGIN_SECS:
                rec = await refresh_session(identity, store, _locks(request), cookie_secret, rec)
                if rec is None:
                    session.pop(SESSION_KEY, None)
                    request.state.session_ended = True
            if rec is not None:
                principal = Principal(
                    sub=rec.sub,
                    sid=rec.sid,
                    email=rec.email,
                    name=rec.name,
                    role=identity.role_of(rec.sub) or "",
                )
                request.state.session_secret = cookie_secret
                current_actor.set(rec.sub)
    request.state.principal = principal
    return principal


def _login_redirect(request: Request) -> HTTPException:
    next_path = request.url.path
    if request.url.query:
        next_path = f"{next_path}?{request.url.query}"
    ended = "&ended=1" if getattr(request.state, "session_ended", False) else ""
    location = f"/kb/login?next={quote(next_path, safe='')}{ended}"
    if request.headers.get("hx-request") == "true":
        # HTMX follows a redirect inside the swapped fragment; tell it to load the page.
        return HTTPException(status_code=401, headers={"HX-Redirect": location})
    # 303 so the browser issues a GET to the login page.
    return HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": location})


async def require_user(request: Request) -> Principal:
    """Any signed-in person with a kb role (sign-in otherwise; no role: the 403 page)."""
    principal = await current_principal(request)
    if principal is None:
        raise _login_redirect(request)
    if not principal.role:
        raise NoKbAccess(principal)
    return principal


def require_role(required: str) -> Callable[..., Awaitable[Principal]]:
    """Dependency factory enforcing a minimum role."""

    async def _dep(user: Principal = Depends(require_user)) -> Principal:
        if not role_satisfies(user.role, required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires the '{required}' role.",
            )
        return user

    return _dep


async def require_write(request: Request, user: Principal = Depends(require_user)) -> Principal:
    """Allow reads for any role; require ``editor`` for mutating methods.

    Applied at router level so every GET stays viewer-accessible while every
    POST/PUT/PATCH/DELETE requires at least editor privileges.
    """
    if request.method.upper() in _SAFE_METHODS:
        return user
    route_path = getattr(request.scope.get("route"), "path", "")
    if route_path in _READ_ONLY_POST_ROUTES:
        return user
    if not role_satisfies(user.role, "editor"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires the 'editor' role.",
        )
    return user


async def fresh_access_token(request: Request, *, margin: int = 60) -> str | None:
    """The signed-in person's access token, valid for *margin* seconds (refreshed if needed);
    None when the session is gone. For calls that act for the person (the members page)."""
    identity = _identity(request)
    cookie_secret = getattr(request.state, "session_secret", None)
    if identity is None or not cookie_secret:
        return None
    store = _store(request)
    rec = await run_in_threadpool(store.load, cookie_secret)
    if rec is None:
        return None
    if rec.access_expires_at - time.time() < margin:
        rec = await refresh_session(
            identity, store, _locks(request), cookie_secret, rec, margin=margin
        )
    return rec.access_token if rec is not None else None


def principal_of(request: Request) -> Any:
    """What templates see as ``current_user`` (set by the dependencies above)."""
    return getattr(request.state, "principal", None)
