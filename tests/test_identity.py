"""web/identity.py against the fake pacod-id: settings, catalogue, feed, roles."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from tests.fake_id import INTERNAL_URL, ISSUER, SERVICE_CLIENT, UI_CLIENT, UI_ORIGIN, FakeId
from web import identity as identity_module
from web.identity import Identity, IdSettings

SETTINGS = IdSettings(
    issuer=ISSUER,
    internal_url=INTERNAL_URL,
    ui_origin=UI_ORIGIN,
    ui_client_id=UI_CLIENT[0],
    ui_client_secret=UI_CLIENT[1],
    service_client_id=SERVICE_CLIENT[0],
    service_client_secret=SERVICE_CLIENT[1],
)


def _run(fake: FakeId, body: Callable[[Identity], Awaitable[Any]]) -> Any:
    async def main() -> Any:
        ident = Identity(SETTINGS, transport=httpx.MockTransport(fake.handle))
        ident.start()
        try:
            return await body(ident)
        finally:
            await ident.stop()

    return asyncio.run(main())


async def _until(cond: Callable[[], bool], within: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + within
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in {
        "PACOD_ID_ISSUER": "http://127.0.0.1:18070/",
        "KB_UI_ORIGIN": "http://localhost:8080",
        "PACOD_ID_UI_SECRET": "u" * 40,
        "PACOD_ID_SERVICE_SECRET": "s" * 40,
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("PACOD_ID_INTERNAL_URL", raising=False)
    s = IdSettings.from_env()
    assert s is not None
    assert s.issuer == "http://127.0.0.1:18070"
    assert s.internal_url == "http://id:8071"
    assert s.redirect_uri == "http://localhost:8080/kb/auth/callback"
    assert s.post_logout_uri == "http://localhost:8080/kb/login?signed_out=1"
    assert s.account_url == "http://127.0.0.1:18070/account"
    monkeypatch.setenv("PACOD_ID_SERVICE_SECRET", "")
    assert IdSettings.from_env() is None


def test_catalogue_is_pushed_and_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity_module, "CATALOG_RETRY", (0.01, 0.02))
    fake = FakeId()
    fake.catalog_failures = 2

    async def body(ident: Identity) -> None:
        await _until(lambda: ident.catalog_pushed)

    _run(fake, body)
    assert fake.catalog is not None
    assert fake.catalog["product"] == "kb"
    assert set(fake.catalog["roles"]) == {"viewer", "editor", "admin"}


def test_roles_follow_the_feed() -> None:
    fake = FakeId()
    fake.bind("usr_admin", "kb.admin")

    async def body(ident: Identity) -> list[str | None]:
        await _until(lambda: ident.cache.ready)
        seen = [ident.role_of("usr_admin"), ident.role_of("usr_nobody")]
        seq = fake.bind("usr_ed", "kb.editor")
        assert await ident.cache.wait_for(seq, 3.0)
        seen.append(ident.role_of("usr_ed"))
        seq = fake.unbind("usr_ed")
        seq = fake.bind("usr_ed", "kb.viewer")
        assert await ident.cache.wait_for(seq, 3.0)
        seen.append(ident.role_of("usr_ed"))
        seq = fake.bind("usr_other", "id.admin")  # another product's role
        assert await ident.cache.wait_for(seq, 3.0)
        seen.append(ident.role_of("usr_other"))
        return seen

    assert _run(fake, body) == ["admin", None, "editor", "viewer", None]


def test_feed_freshness(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeId()

    async def body(ident: Identity) -> tuple[bool, bool, bool]:
        before = ident.feed_fresh
        await _until(lambda: ident.cache.ready)
        fresh = ident.feed_fresh
        monkeypatch.setattr(type(ident.cache), "feed_age", property(lambda _self: 61.0))
        return before, fresh, ident.feed_fresh

    assert _run(fake, body) == (False, True, False)


def test_feed_events_reach_listeners() -> None:
    fake = FakeId()
    seen: list[dict[str, Any]] = []

    async def body(ident: Identity) -> None:
        ident.on_event(seen.append)
        await _until(lambda: ident.cache.ready)
        seq = fake.end_session("sid_9")
        assert await ident.cache.wait_for(seq, 3.0)
        assert ident.cache.sid_revoked("sid_9")

    _run(fake, body)
    assert any(e.get("kind") == "session.revoked" and e.get("sid") == "sid_9" for e in seen)
