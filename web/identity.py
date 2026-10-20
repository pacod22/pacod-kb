"""pacod-id for the UI: its settings, the authorization feed, and kb roles.

One :class:`Identity` per process, built and started in the UI's lifespan (``web/app.py``),
never at import:

* ``service`` — pacod-id's internal API as ``pacod-kb-service``: the feed, the catalogue push,
  token exchange for the admin API, ending sessions, deletion confirmations;
* ``ui`` — the same API as ``pacod-kb-ui``: the sign-in code exchange and refresh tokens
  (a refresh token belongs to the client that obtained it);
* ``verifier`` — checks the access tokens sign-in yields (issuer, audience ``pacod-kb``,
  a person's session);
* ``cache`` — the copy of kb's bindings, disabled accounts and ended sessions;
* ``catalog`` — kb's rules (``web/authz/kb.yaml``): a person's kb role comes from what it
  allows, not from a binding's name.

Not configured (``PACOD_ID_ISSUER`` or a secret blank): :meth:`IdSettings.from_env` is
``None``, the UI still starts (its health check answers), and protected pages answer 503.

Must not import ``pipeline``: Dockerfile.ui ships only ``web/``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pacod_id.authz import Catalog, Manifest, Subject
from pacod_id.cache import AuthzCache
from pacod_id.client import ApiError, IdClient, IdConfig
from pacod_id.verify import JwtVerifier

logger = logging.getLogger(__name__)

PRODUCT = "kb"
AUDIENCE = "pacod-kb"
PLATFORM = ("platform",)
MANIFEST_PATH = Path(__file__).parent / "authz" / "kb.yaml"
# The catalogue push retries with backoff: first wait, longest wait (seconds).
CATALOG_RETRY = (2.0, 60.0)
# A person's kb role: the first whose permission the catalogue allows at the platform.
ROLE_PERMISSIONS = (
    ("admin", "kb.settings.manage"),
    ("editor", "kb.docs.write"),
    ("viewer", "kb.docs.read"),
)


@dataclass(frozen=True)
class IdSettings:
    """Where pacod-id is and who the UI is to it (env ``PACOD_ID_*``, ``KB_UI_ORIGIN``)."""

    issuer: str  # the public issuer browsers use, as tokens carry it in ``iss``
    internal_url: str  # pacod-id's internal listener as the UI reaches it
    ui_origin: str  # the UI's own origin: redirect URIs are built from it, never from a request
    ui_client_id: str
    ui_client_secret: str
    service_client_id: str
    service_client_secret: str
    max_staleness: float = 60.0

    @classmethod
    def from_env(cls) -> IdSettings | None:
        def env(name: str, default: str = "") -> str:
            return os.environ.get(name, default).strip()

        settings = cls(
            issuer=env("PACOD_ID_ISSUER").rstrip("/"),
            internal_url=env("PACOD_ID_INTERNAL_URL", "http://pacod-id:8071").rstrip("/"),
            ui_origin=env("KB_UI_ORIGIN").rstrip("/"),
            ui_client_id=env("PACOD_ID_UI_CLIENT_ID", "pacod-kb-ui"),
            ui_client_secret=env("PACOD_ID_UI_SECRET"),
            service_client_id=env("PACOD_ID_SERVICE_CLIENT_ID", "pacod-kb-service"),
            service_client_secret=env("PACOD_ID_SERVICE_SECRET"),
            max_staleness=float(env("PACOD_ID_MAX_STALENESS", "60") or 60),
        )
        missing = [
            name
            for name, value in (
                ("PACOD_ID_ISSUER", settings.issuer),
                ("KB_UI_ORIGIN", settings.ui_origin),
                ("PACOD_ID_UI_SECRET", settings.ui_client_secret),
                ("PACOD_ID_SERVICE_SECRET", settings.service_client_secret),
            )
            if not value
        ]
        if missing:
            logger.warning("Sign-in is not set up: %s blank", ", ".join(missing))
            return None
        return settings

    @property
    def redirect_uri(self) -> str:
        return f"{self.ui_origin}/kb/auth/callback"

    @property
    def post_logout_uri(self) -> str:
        return f"{self.ui_origin}/kb/login?signed_out=1"

    @property
    def account_url(self) -> str:
        """pacod-id's own account pages: password, sessions, connections."""
        return f"{self.issuer}/account"


class Identity:
    def __init__(
        self, settings: IdSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        def http(timeout: float) -> httpx.AsyncClient:
            return httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=transport)

        def config(client_id: str, secret: str) -> IdConfig:
            return IdConfig(
                issuers=(settings.issuer,),
                internal_url=settings.internal_url,
                client_id=client_id,
                client_secret=secret,
                product=PRODUCT,
                audience=AUDIENCE,
            )

        self.settings = settings
        # Longer than the feed's 25-second long poll.
        self.service = IdClient(
            config(settings.service_client_id, settings.service_client_secret), http=http(40)
        )
        self.ui = IdClient(config(settings.ui_client_id, settings.ui_client_secret), http=http(15))
        self.verifier = JwtVerifier(self.service)
        self.cache = AuthzCache(self.service)
        self.manifest = Manifest.from_yaml(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.catalog = Catalog([self.manifest])
        self._tasks: list[asyncio.Task[None]] = []
        self.catalog_pushed = False
        # Exchanged admin-API tokens (they live 60 s), by the hash of the person's access token.
        self._admin_tokens: dict[str, tuple[float, str]] = {}

    # ---- life cycle ----------------------------------------------------------------------

    def start(self) -> None:
        """Follow the feed and push the catalogue (in the background: start never waits)."""
        self.cache.start()
        self._tasks.append(asyncio.create_task(self._push_catalog()))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        await self.cache.stop()
        await self.service.aclose()
        await self.ui.aclose()

    def spawn(self, coro: Any) -> None:
        """Run *coro* in the background, cancelled at :meth:`stop`."""
        self._tasks = [t for t in self._tasks if not t.done()]
        self._tasks.append(asyncio.create_task(coro))

    async def _push_catalog(self) -> None:
        """PUT /internal/v1/catalog/kb until pacod-id takes it (the same manifest changes
        nothing). Until it lands, pacod-id keeps bindings to kb's roles but counts none —
        the bootstrap admin's kb.admin included."""
        backoff, longest = CATALOG_RETRY
        while True:
            try:
                answer = await self.service.call(
                    "PUT", f"/internal/v1/catalog/{PRODUCT}", {"manifest": self.manifest.source}
                )
                self.catalog_pushed = True
                logger.info(
                    "kb catalogue pushed to pacod-id (version %s, changed %s)",
                    (answer or {}).get("version"),
                    (answer or {}).get("changed"),
                )
                return
            except Exception as exc:
                logger.warning("kb catalogue push failed (%s); retrying in %.0f s", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, longest)

    def on_event(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Called with each feed change (``session.revoked``, ``principal.deleted``, …)."""
        self.cache.on_event(listener)

    # ---- checks --------------------------------------------------------------------------

    @property
    def feed_fresh(self) -> bool:
        """Whether the copy of who may do what is recent enough to decide on."""
        age = self.cache.feed_age
        return age is not None and age <= self.settings.max_staleness

    def subject(self, sub: str) -> Subject:
        return Subject("user", self.cache.bindings_of(sub))

    def role_of(self, sub: str) -> str | None:
        """``admin``, ``editor`` or ``viewer`` by what the person may do in kb; None for none."""
        subject = self.subject(sub)
        for role, permission in ROLE_PERMISSIONS:
            if self.catalog.allowed(subject, permission, PLATFORM):
                return role
        return None

    # ---- the admin API, acting for a person (the members page) -------------------------

    async def admin_token(self, access_token: str) -> str:
        """A 60-second admin-API token for the person holding *access_token* (cached 50 s)."""
        key = hashlib.sha256(access_token.encode()).hexdigest()
        now = time.monotonic()
        hit = self._admin_tokens.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
        tokens = await self.service.exchange_for_admin(access_token)
        self._admin_tokens = {k: v for k, v in self._admin_tokens.items() if v[0] > now}
        self._admin_tokens[key] = (now + 50.0, tokens.access_token)
        return tokens.access_token

    def forget_admin_token(self, access_token: str) -> None:
        self._admin_tokens.pop(hashlib.sha256(access_token.encode()).hexdigest(), None)

    async def admin(
        self,
        method: str,
        path: str,
        access_token: str,
        *,
        json: Any = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        """``/admin/v1`` + *path* as the person, through the kb product (narrowed to kb's
        roles). Raises ``ApiError`` with pacod-id's error code; a refused admin token is
        exchanged again once."""
        for attempt in (1, 2):
            token = await self.admin_token(access_token)
            try:
                r = await self.service.http.request(
                    method,
                    self.service.url("/admin/v1" + path),
                    headers={"authorization": f"Bearer {token}"},
                    json=json,
                    params=params,
                )
            except httpx.HTTPError as exc:
                raise ApiError(0, "unreachable", str(exc)) from exc
            if r.status_code == 401 and attempt == 1:
                self.forget_admin_token(access_token)
                continue
            return _answer(r)
        raise AssertionError("unreachable")

    async def wait_for(self, seq: int | None) -> None:
        """Until the feed copy has change *seq* (read your own writes; at most 5 s)."""
        if seq is not None:
            await self.cache.wait_for(int(seq), 5.0)

    def grantable(self, sub: str) -> list[str]:
        """The kb roles *sub* may give at the platform, least privileged first."""
        subject = self.subject(sub)
        return [
            role
            for role, _permission in reversed(ROLE_PERMISSIONS)
            if self.catalog.can_grant(subject, f"{PRODUCT}.{role}", PLATFORM) is None
        ]


def _answer(r: httpx.Response) -> Any:
    """pacod-id's answer, or ApiError with its ``details.code`` (as pacod_id.client reads it)."""
    try:
        body = r.json() if r.content else None
    except ValueError:
        body = None
    if r.is_success:
        return body
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        code = (err.get("details") or {}).get("code") or err.get("type") or "error"
        raise ApiError(r.status_code, str(code), str(err.get("message", "")))
    raise ApiError(r.status_code, str(err or "error"), "")
