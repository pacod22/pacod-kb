"""Members: who holds which kb role, and invitations — pacod-id's admin API, as the admin.

Every call acts for the signed-in admin: the UI exchanges their access token for a
60-second admin-API token (``Identity.admin``) through the kb product, so pacod-id lists
and changes kb's roles only, and checks the admin's own right to grant each one. After a
change the page waits for the feed to carry it (read your own writes), so the list shown is
the one roles are decided on.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from pacod_id.client import ApiError
from starlette.concurrency import run_in_threadpool

from web.auth import Principal, fresh_access_token, require_role
from web.identity import Identity
from web.legacy_users import former_accounts
from web.roles import ROLES
from web.sessions import db_url
from web.templates_config import templates

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/users", dependencies=[Depends(require_role("admin"))])

# pacod-id's ids: usr_…, svc_…, inv_… (letters and digits after the prefix).
_ID = re.compile(r"^[a-z]{2,4}_[A-Za-z0-9]{1,64}$")
_MEMBERS = "/scopes/platform/kb/members"

_ERRORS = {
    "last_holder": "The knowledge base needs another admin first.",
    "not_allowed": "You may not make this change.",
    "binding_exists": "They already hold that role.",
    "invitation_exists": "An invitation to this address is already pending.",
    "bad_email": "That is not an email address.",
    "principal_unknown": "pacod-id has no such account.",
    "not_a_member": "They hold no kb role.",
    "email_unavailable": "This installation sends no mail: choose “give me a link”.",
    "unreachable": "pacod-id cannot be reached. Try again in a moment.",
    "invalid_token": "Your sign-in has expired. Sign in again.",
    "not_found": "That invitation is gone.",
}
_LABELS = {"viewer": "Viewer", "editor": "Editor", "admin": "Admin"}


def _identity(request: Request) -> Identity:
    identity: Identity = request.app.state.identity
    return identity


def _message(exc: ApiError) -> str:
    return _ERRORS.get(exc.code, f"pacod-id refused: {exc.message or exc.code}")


async def _token(request: Request) -> str:
    token = await fresh_access_token(request)
    if token is None:
        raise ApiError(401, "invalid_token", "")
    return token


async def _render(
    request: Request,
    me: Principal,
    *,
    notice: str | None = None,
    error: str | None = None,
    link: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    identity = _identity(request)
    members: list[dict[str, Any]] = []
    invitations: list[dict[str, Any]] = []
    try:
        token = await _token(request)
        members = (await identity.admin("GET", _MEMBERS, token) or {}).get("members", [])
        invitations = (
            await identity.admin("GET", "/invitations", token, params={"scope": "platform"}) or {}
        ).get("invitations", [])
    except ApiError as exc:
        error = error or _message(exc)
    for m in members:
        roles = [r.get("role", "") for r in m.get("roles", []) if not r.get("inherited")]
        m["kb_role"] = next((r.split(".", 1)[1] for r in roles if r.startswith("kb.")), "")
    known = {str(m.get("email", "")).lower() for m in members}
    known |= {str(i.get("email", "")).lower() for i in invitations}
    former = [
        (email, role)
        for email, role in await run_in_threadpool(former_accounts, db_url())
        if email not in known
    ]
    return templates.TemplateResponse(
        request,
        "users/members.html",
        {
            "members": members,
            "invitations": invitations,
            "former": former,
            "grantable": identity.grantable(me.sub),
            "me": me,
            "notice": notice,
            "error": error,
            "link": link,
        },
        status_code=status_code,
    )


@router.get("", response_class=HTMLResponse)
async def members_page(
    request: Request, me: Principal = Depends(require_role("admin"))
) -> HTMLResponse:
    return await _render(request, me)


def _bad_input(role: str, me: Principal, identity: Identity) -> str | None:
    if role not in ROLES:
        return "Choose a role."
    if role not in identity.grantable(me.sub):
        return _ERRORS["not_allowed"]
    return None


@router.post("/invite", response_class=HTMLResponse)
async def invite(
    request: Request,
    email: str = Form(""),
    role: str = Form("viewer"),
    as_link: str = Form(""),
    me: Principal = Depends(require_role("admin")),
) -> HTMLResponse:
    identity = _identity(request)
    email = email.strip()
    problem = _bad_input(role, me, identity)
    if problem:
        return await _render(request, me, error=problem, status_code=422)
    body = {
        "email": email,
        "role": f"kb.{role}",
        "scope": "platform",
        "deliver": "link" if as_link else "email",
        "labels": {"role": _LABELS[role], "scope": "pacod-kb"},
    }
    try:
        answer = await identity.admin("POST", "/invitations", await _token(request), json=body)
    except ApiError as exc:
        code = 409 if exc.status == 409 else 400
        return await _render(request, me, error=_message(exc), status_code=code)
    link = (answer or {}).get("link")
    mailed = "" if link else " pacod-id mailed the invitation."
    notice = f"Invited {email} as {role}.{mailed}"
    return await _render(request, me, notice=notice, link=link)


@router.post("/{principal}/role", response_class=HTMLResponse)
async def change_role(
    request: Request,
    principal: str,
    role: str = Form(...),
    me: Principal = Depends(require_role("admin")),
) -> HTMLResponse:
    identity = _identity(request)
    if not _ID.fullmatch(principal):
        return await _render(request, me, error="Unknown member.", status_code=404)
    if principal == me.sub:
        return await _render(request, me, error="You cannot change your own role.", status_code=409)
    problem = _bad_input(role, me, identity)
    if problem:
        return await _render(request, me, error=problem, status_code=422)
    try:
        answer = await identity.admin(
            "PUT", f"{_MEMBERS}/{principal}", await _token(request), json={"role": f"kb.{role}"}
        )
    except ApiError as exc:
        return await _render(request, me, error=_message(exc), status_code=exc.status or 503)
    await identity.wait_for((answer or {}).get("seq"))
    return await _render(request, me, notice=f"Role changed to {role}.")


@router.post("/{principal}/remove", response_class=HTMLResponse)
async def remove_member(
    request: Request,
    principal: str,
    me: Principal = Depends(require_role("admin")),
) -> HTMLResponse:
    identity = _identity(request)
    if not _ID.fullmatch(principal):
        return await _render(request, me, error="Unknown member.", status_code=404)
    if principal == me.sub:
        return await _render(request, me, error="You cannot remove yourself.", status_code=409)
    try:
        answer = await identity.admin(
            "DELETE", f"{_MEMBERS}/{principal}", await _token(request), params={"below": "false"}
        )
    except ApiError as exc:
        return await _render(request, me, error=_message(exc), status_code=exc.status or 503)
    await identity.wait_for((answer or {}).get("seq"))
    return await _render(request, me, notice="Removed from the knowledge base.")


@router.post("/invitations/{invitation}/revoke", response_class=HTMLResponse)
async def revoke_invitation(
    request: Request,
    invitation: str,
    me: Principal = Depends(require_role("admin")),
) -> HTMLResponse:
    if not _ID.fullmatch(invitation):
        return await _render(request, me, error="Unknown invitation.", status_code=404)
    try:
        await _identity(request).admin(
            "DELETE", f"/invitations/{invitation}", await _token(request)
        )
    except ApiError as exc:
        return await _render(request, me, error=_message(exc), status_code=exc.status or 503)
    return await _render(request, me, notice="Invitation revoked.")


@router.post("/invitations/{invitation}/resend", response_class=HTMLResponse)
async def resend_invitation(
    request: Request,
    invitation: str,
    me: Principal = Depends(require_role("admin")),
) -> HTMLResponse:
    if not _ID.fullmatch(invitation):
        return await _render(request, me, error="Unknown invitation.", status_code=404)
    try:
        await _identity(request).admin(
            "POST", f"/invitations/{invitation}/resend", await _token(request), json={}
        )
    except ApiError as exc:
        return await _render(request, me, error=_message(exc), status_code=exc.status or 503)
    return await _render(request, me, notice="Invitation sent again.")
