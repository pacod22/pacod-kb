"""A fake pacod-id for the UI's tests, behind httpx.MockTransport, with real RS256 keys.

Adapted from pacod-id's own ``python/tests/test_client.py`` (``FakeId``). The UI's real
``Identity`` — its verifier, feed follower and clients — runs against it unchanged: the tests
install ``httpx.MockTransport(fake.handle)`` as ``app.state.id_transport``.

What it does, as pacod-id does (docs/api.md):

* ``/oidc/token``: ``authorization_code`` (PKCE S256, exact redirect URI, one use),
  ``refresh_token`` (rotation; the replaced token returns the same successor for 30 s, an
  older one is reuse: every grant of the account ends), ``client_credentials`` and token
  exchange for the admin API (``audience=pacod-id``, a 60-second ``adm_`` token);
* ``/oidc/jwks``, ``/oidc/revoke``;
* the internal API: catalogue PUT, snapshot, long-polled changes, session revocation,
  deletion confirmations;
* the admin API through a product: platform members, invitations.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import json
import time
from collections import Counter
from typing import Any
from urllib.parse import parse_qs, unquote

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "http://id.test"
INTERNAL_URL = "http://id-internal.test"
UI_ORIGIN = "http://testserver"
UI_CLIENT = ("pacod-kb-ui", "ui-client-secret-0123456789abcdef0123456789")
SERVICE_CLIENT = ("pacod-kb-service", "service-client-secret-0123456789abcdef0123")
REDIRECT_URI = f"{UI_ORIGIN}/kb/auth/callback"
GRACE_SECS = 30.0


def _error(status: int, code: str, message: str = "", *, oauth: bool = False) -> httpx.Response:
    if oauth:
        return httpx.Response(status, json={"error": code, "error_description": message})
    return httpx.Response(
        status, json={"error": {"type": code, "message": message, "details": {"code": code}}}
    )


class FakeId:
    def __init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "k1"
        self.seq = 1
        self.bindings: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.disabled: set[str] = set()
        self.deleted: set[str] = set()
        self.revoked_sids: set[str] = set()
        self.accounts: dict[str, dict[str, str]] = {}
        self.codes: dict[str, dict[str, Any]] = {}
        self.grants: dict[str, dict[str, Any]] = {}  # live refresh token -> grant
        self.spent: dict[str, tuple[str, float, dict[str, Any]]] = {}  # replaced -> successor
        self.invitations: list[dict[str, Any]] = []
        self.catalog: dict[str, Any] | None = None
        self.catalog_failures = 0  # refuse this many catalogue pushes first
        self.token_failures = 0  # answer this many token-endpoint calls with 503 first
        self.reuse_detected = 0
        self.sessions_revoked: list[str] = []
        self.deletions_done: list[str] = []
        self.calls: Counter[str] = Counter()  # "POST /oidc/token:refresh_token" -> count
        self.access_ttl = 300
        self.feed_wait = 0.05
        self._ids = itertools.count(1)

    # ---- for tests ------------------------------------------------------------------------

    def account(self, sub: str, email: str, name: str = "") -> None:
        self.accounts[sub] = {"email": email, "name": name or email.split("@")[0]}

    def _event(self, **event: Any) -> int:
        self.seq += 1
        self.events.append({**event, "seq": self.seq})
        return self.seq

    def bind(self, sub: str, role: str, path: tuple[str, ...] = ("platform",)) -> int:
        binding = {
            "id": f"bnd_{next(self._ids)}",
            "principal": sub,
            "role": role,
            "path": list(path),
        }
        self.bindings.append(binding)
        return self._event(kind="binding.added", principal_kind="user", source="manual", **binding)

    def unbind(self, sub: str) -> int:
        seq = self.seq
        for b in [b for b in self.bindings if b["principal"] == sub]:
            self.bindings.remove(b)
            seq = self._event(kind="binding.removed", principal_kind="user", source="manual", **b)
        return seq

    def disable(self, sub: str) -> int:
        self.disabled.add(sub)
        return self._event(kind="principal.disabled", sub=sub)

    def delete(self, sub: str) -> int:
        self.deleted.add(sub)
        self.bindings = [b for b in self.bindings if b["principal"] != sub]
        return self._event(kind="principal.deleted", sub=sub, principal_kind="user")

    def end_session(self, sid: str) -> int:
        self.revoked_sids.add(sid)
        for token, grant in list(self.grants.items()):
            if grant["sid"] == sid:
                del self.grants[token]
        return self._event(kind="session.revoked", sid=sid)

    def issue_code(
        self, sub: str, challenge: str, redirect_uri: str = REDIRECT_URI, sid: str | None = None
    ) -> str:
        code = f"code_{next(self._ids)}"
        self.codes[code] = {
            "sub": sub,
            "sid": sid or f"sid_{next(self._ids)}",
            "challenge": challenge,
            "redirect_uri": redirect_uri,
        }
        return code

    def access_token(
        self,
        sub: str,
        sid: str,
        *,
        aud: str = "pacod-kb",
        iss: str = ISSUER,
        typ: str = "at+jwt",
        expires_in: int | None = None,
        client_id: str = UI_CLIENT[0],
    ) -> str:
        now = int(time.time())
        acct = self.accounts.get(sub, {})
        claims: dict[str, Any] = {
            "iss": iss,
            "sub": sub,
            "aud": aud,
            "client_id": client_id,
            "iat": now,
            "exp": now + (self.access_ttl if expires_in is None else expires_in),
            "jti": f"jti_{next(self._ids)}",
            "sid": sid,
            "pid_kind": "user",
            "scope": "openid email profile",
            "email": acct.get("email", f"{sub}@example.test"),
            "name": acct.get("name", sub),
        }
        headers = {"typ": typ, "kid": self.kid}
        return jwt.encode(claims, self.key, algorithm="RS256", headers=headers)

    def jwks(self) -> dict[str, Any]:
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        jwk.update({"kid": self.kid, "alg": "RS256", "use": "sig"})
        return {"keys": [jwk]}

    # ---- the transport ----------------------------------------------------------------------

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/oidc/jwks":
            return httpx.Response(200, json=self.jwks())
        if path == "/oidc/token":
            return await self._token(request)
        if path == "/oidc/revoke":
            form = self._form(request)
            self.grants.pop(form.get("token", ""), None)
            return httpx.Response(200)
        if path.startswith("/internal/"):
            if not request.headers.get("authorization", "").startswith("Bearer svc_"):
                return _error(401, "invalid_token")
            return await self._internal(request, path)
        if path.startswith("/admin/v1/"):
            return self._admin(request, path)
        return _error(404, "not_found")

    @staticmethod
    def _form(request: httpx.Request) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(request.content.decode()).items()}

    @staticmethod
    def _client(request: httpx.Request) -> tuple[str, str]:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Basic "):
            return "", ""
        cid, _, secret = base64.b64decode(auth[6:]).decode().partition(":")
        return unquote(cid), unquote(secret)

    def _tokens(self, grant: dict[str, Any]) -> dict[str, Any]:
        refresh = f"pidr_{next(self._ids)}"
        self.grants[refresh] = grant
        return {
            "access_token": self.access_token(grant["sub"], grant["sid"]),
            "token_type": "Bearer",
            "expires_in": self.access_ttl,
            "refresh_token": refresh,
            "id_token": "not-checked-by-kb",
            "scope": "openid email profile",
        }

    async def _token(self, request: httpx.Request) -> httpx.Response:
        form = self._form(request)
        grant_type = form.get("grant_type", "")
        self.calls[f"token:{grant_type}"] += 1
        if self.token_failures > 0:
            self.token_failures -= 1
            return httpx.Response(503, json={"error": "temporarily_unavailable"})
        client = self._client(request)
        if client not in (UI_CLIENT, SERVICE_CLIENT):
            return _error(401, "invalid_client", oauth=True)
        if grant_type == "authorization_code":
            code = self.codes.pop(form.get("code", ""), None)
            if code is None or client != UI_CLIENT:
                return _error(400, "invalid_grant", oauth=True)
            digest = hashlib.sha256(form.get("code_verifier", "").encode()).digest()
            challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            if challenge != code["challenge"] or form.get("redirect_uri") != code["redirect_uri"]:
                return _error(400, "invalid_grant", oauth=True)
            return httpx.Response(200, json=self._tokens({"sub": code["sub"], "sid": code["sid"]}))
        if grant_type == "refresh_token":
            await asyncio.sleep(0.02)  # long enough for concurrent refreshes to overlap
            token = form.get("refresh_token", "")
            grant = self.grants.pop(token, None)
            if grant is not None:
                answer = self._tokens(grant)
                self.spent[token] = (answer["refresh_token"], time.monotonic(), answer)
                return httpx.Response(200, json=answer)
            if token in self.spent:
                successor, rotated, answer = self.spent[token]
                if time.monotonic() - rotated <= GRACE_SECS and successor in self.grants:
                    return httpx.Response(200, json=answer)
                # Reuse: every grant of the account ends.
                self.reuse_detected += 1
                sub = self.grants.get(successor, {}).get("sub")
                for g in list(self.grants.values()):
                    if g["sub"] == sub:
                        self.end_session(g["sid"])
                return _error(400, "invalid_grant", oauth=True)
            return _error(400, "invalid_grant", oauth=True)
        if grant_type == "client_credentials" and client == SERVICE_CLIENT:
            return httpx.Response(
                200, json={"access_token": f"svc_{next(self._ids)}", "expires_in": 300}
            )
        if grant_type.endswith("token-exchange") and client == SERVICE_CLIENT:
            try:
                claims = jwt.decode(
                    form.get("subject_token", ""),
                    self.key.public_key(),
                    algorithms=["RS256"],
                    audience="pacod-kb",
                    issuer=ISSUER,
                )
            except jwt.PyJWTError:
                return _error(400, "invalid_grant", oauth=True)
            if claims["sid"] in self.revoked_sids:
                return _error(400, "invalid_grant", oauth=True)
            return httpx.Response(
                200,
                json={
                    "access_token": f"adm_{claims['sub']}_{next(self._ids)}",
                    "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    "token_type": "Bearer",
                },
            )
        return _error(400, "unsupported_grant_type", oauth=True)

    async def _internal(self, request: httpx.Request, path: str) -> httpx.Response:
        if path == "/internal/v1/catalog/kb" and request.method == "PUT":
            if self.catalog_failures > 0:
                self.catalog_failures -= 1
                return _error(503, "unavailable")
            self.catalog = json.loads(request.content)["manifest"]
            return httpx.Response(200, json={"version": 1, "changed": True})
        if path == "/internal/v1/authz/snapshot":
            return httpx.Response(
                200,
                json={
                    "seq": self.seq,
                    "bindings": [
                        dict(b, principal_kind="user", source="manual") for b in self.bindings
                    ],
                    "disabled": sorted(self.disabled),
                    "deleted": sorted(self.deleted - set(self.deletions_done)),
                    "revoked_sids": [{"sid": s} for s in sorted(self.revoked_sids)],
                },
            )
        if path == "/internal/v1/authz/changes":
            await asyncio.sleep(self.feed_wait)
            after = int(request.url.params.get("after", "0"))
            return httpx.Response(
                200, json={"seq": self.seq, "events": [e for e in self.events if e["seq"] > after]}
            )
        if path.startswith("/internal/v1/sessions/") and path.endswith("/revoke"):
            sid = unquote(path.split("/")[4])
            self.sessions_revoked.append(sid)
            self.end_session(sid)
            return httpx.Response(204)
        if path.startswith("/internal/v1/deletions/") and path.endswith("/done"):
            self.deletions_done.append(unquote(path.split("/")[4]))
            return httpx.Response(204)
        return _error(404, "not_found")

    # ---- the admin API, through the kb product (members of the platform) --------------------

    def _admin(self, request: httpx.Request, path: str) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer adm_"):
            return _error(401, "invalid_token")
        actor = auth[len("Bearer adm_") :].rsplit("_", 1)[0]
        self.calls[f"admin:{request.method} {path}"] += 1
        is_admin = any(
            b["principal"] == actor and b["role"] == "kb.admin" for b in self.bindings
        )
        if request.method != "GET" and not is_admin:
            return _error(403, "not_allowed")
        parts = path.split("/")  # ['', 'admin', 'v1', ...]
        if parts[3:5] == ["scopes", "platform"] and parts[6:7] == ["members"]:
            if len(parts) == 7 and request.method == "GET":
                return httpx.Response(200, json={"scope": "platform", "members": self._members()})
            principal = unquote(parts[7]) if len(parts) > 7 else ""
            if request.method == "PUT":
                role = json.loads(request.content).get("role")
                if principal not in self.accounts:
                    return _error(404, "principal_unknown")
                if self._last_admin_would_go(principal, role):
                    return _error(409, "last_holder")
                self.unbind(principal)
                seq = self.bind(principal, role) if role else self.seq
                return httpx.Response(200, json={"seq": seq})
            if request.method == "DELETE":
                if not any(b["principal"] == principal for b in self.bindings):
                    return _error(404, "not_a_member")
                if self._last_admin_would_go(principal, None):
                    return _error(409, "last_holder")
                return httpx.Response(200, json={"seq": self.unbind(principal)})
        if parts[3] == "invitations":
            if len(parts) == 4 and request.method == "GET":
                return httpx.Response(200, json={"invitations": list(self.invitations)})
            if len(parts) == 4 and request.method == "POST":
                body = json.loads(request.content)
                if "@" not in body.get("email", ""):
                    return _error(422, "bad_email")
                if any(i["email"] == body["email"] for i in self.invitations):
                    return _error(409, "invitation_exists")
                inv = {
                    "id": f"inv_{next(self._ids)}",
                    "email": body["email"],
                    "role": body["role"],
                    "scope": body["scope"],
                    "emailed": body.get("deliver", "email") == "email",
                    "expired": False,
                }
                self.invitations.append(inv)
                answer = dict(inv)
                if not inv["emailed"]:
                    answer["link"] = f"{ISSUER}/invite?token=tok_{inv['id']}"
                return httpx.Response(201, json=answer)
            inv_id = parts[4]
            match = [i for i in self.invitations if i["id"] == inv_id]
            if not match:
                return _error(404, "not_found")
            if request.method == "DELETE":
                self.invitations.remove(match[0])
                return httpx.Response(204)
            if parts[5:6] == ["resend"]:
                return httpx.Response(200, json=match[0])
        return _error(404, "not_found")

    def _members(self) -> list[dict[str, Any]]:
        out = []
        for sub in sorted({b["principal"] for b in self.bindings if b["role"].startswith("kb.")}):
            acct = self.accounts.get(sub, {})
            out.append(
                {
                    "principal": sub,
                    "principal_kind": "user",
                    "email": acct.get("email", ""),
                    "name": acct.get("name", ""),
                    "status": "disabled" if sub in self.disabled else "active",
                    "roles": [
                        {
                            "binding": b["id"],
                            "role": b["role"],
                            "scope": "platform",
                            "inherited": False,
                        }
                        for b in self.bindings
                        if b["principal"] == sub and b["role"].startswith("kb.")
                    ],
                }
            )
        return out

    def _last_admin_would_go(self, principal: str, new_role: str | None) -> bool:
        admins = {b["principal"] for b in self.bindings if b["role"] == "kb.admin"}
        return admins == {principal} and new_role != "kb.admin"
