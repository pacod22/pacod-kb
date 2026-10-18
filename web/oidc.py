"""Signing in at pacod-id: the authorization-code flow with PKCE, built by hand.

No discovery and no OAuth library: the browser endpoints (authorize, logout) are on the
public issuer, the token endpoint on pacod-id's internal listener — inside compose the issuer
host is not reachable, and pacod-id answers browser endpoints only on the issuer's own host.

* PKCE S256 and ``state`` protect the code flow, and the authorization response's ``iss``
  is checked (RFC 9207). No nonce: kb does not use the id_token (``JwtVerifier`` checks
  access tokens, and the code is exchanged server to server).
* The redirect URI comes from ``KB_UI_ORIGIN`` (``IdSettings``), never from the request.
* The person comes from the verified access token: ``sub``, ``sid``, and ``email``/``name``
  (scopes ``email``, ``profile``), which ``AccessToken`` drops and are read again from the
  same, now verified, token — as pacod-api does.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

import httpx
import jwt
from pacod_id.client import ApiError, Tokens
from pacod_id.verify import InvalidToken

from web.identity import Identity, IdSettings

SCOPE = "openid email profile"
LOGIN_TTL_SECS = 600  # a sign-in started longer ago is refused at the callback


@dataclass(frozen=True)
class Claims:
    sub: str
    sid: str
    email: str
    name: str
    expires_at: int


def new_login(next_path: str) -> tuple[dict[str, Any], str]:
    """A sign-in's secrets for the session (state, PKCE verifier, where to go), and the
    PKCE challenge for the authorize URL."""
    verifier = secrets.token_urlsafe(48)  # 64 characters (RFC 7636: 43 to 128)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    blob = {
        "state": secrets.token_urlsafe(24),
        "verifier": verifier,
        "next": next_path,
        "at": int(time.time()),
    }
    return blob, challenge.rstrip(b"=").decode()


def authorize_url(settings: IdSettings, *, state: str, code_challenge: str) -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": settings.ui_client_id,
            "redirect_uri": settings.redirect_uri,
            "scope": SCOPE,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return f"{settings.issuer}/oidc/authorize?{query}"


def logout_url(settings: IdSettings) -> str:
    """pacod-id's RP-initiated logout; after ``revoke_session`` it only clears its cookie."""
    query = urlencode(
        {"client_id": settings.ui_client_id, "post_logout_redirect_uri": settings.post_logout_uri}
    )
    return f"{settings.issuer}/oidc/logout?{query}"


def _basic(client_id: str, secret: str) -> str:
    # RFC 6749 2.3.1: form-encoded before Base64.
    pair = f"{quote(client_id, safe='')}:{quote(secret, safe='')}"
    return "Basic " + base64.b64encode(pair.encode()).decode()


async def exchange_code(identity: Identity, code: str, verifier: str) -> Tokens:
    """The code for tokens, at the internal token endpoint, as the UI client."""
    s = identity.settings
    try:
        r = await identity.ui.http.post(
            identity.ui.url("/oidc/token"),
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": s.redirect_uri,
                "code_verifier": verifier,
            },
            headers={"authorization": _basic(s.ui_client_id, s.ui_client_secret)},
        )
    except httpx.HTTPError as exc:
        raise ApiError(0, "unreachable", str(exc)) from exc
    try:
        body = r.json()
    except ValueError:
        body = {}
    if not r.is_success or not isinstance(body, dict) or "access_token" not in body:
        code_ = body.get("error", "error") if isinstance(body, dict) else "error"
        raise ApiError(r.status_code, str(code_), "")
    return Tokens(body["access_token"], body.get("refresh_token"), body.get("expires_in"))


async def verify(identity: Identity, access_token: str) -> Claims:
    """A person's access token, checked (raises InvalidToken, or KeysUnavailable)."""
    at = await identity.verifier.verify(access_token)
    if at.kind != "user" or not at.sid:
        raise InvalidToken("not a person's token")
    extra = jwt.decode(access_token, options={"verify_signature": False})
    return Claims(
        sub=at.sub,
        sid=at.sid,
        email=str(extra.get("email") or ""),
        name=str(extra.get("name") or ""),
        expires_at=at.expires_at,
    )
