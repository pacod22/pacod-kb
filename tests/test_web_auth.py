"""Sign-in through pacod-id, server-side sessions, refresh, revocation and role gates."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.fake_id import ISSUER, REDIRECT_URI, UI_CLIENT, FakeId
from tests.web_support import login_as, start_login, wait_for_seq, web_client
from web.roles import role_satisfies


@pytest.fixture()
def fake() -> FakeId:
    return FakeId()


@pytest.fixture()
def client(fake: FakeId) -> Iterator[TestClient]:
    with web_client(fake) as c:
        yield c


def _sid(fake: FakeId, sub: str) -> str:
    return next(g["sid"] for g in fake.grants.values() if g["sub"] == sub)


# ---- the redirect chain -----------------------------------------------------------------


def test_unauthenticated_pages_go_to_sign_in_with_an_encoded_next(client: TestClient) -> None:
    r = client.get("/kb/pipeline/documents?status=active", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == (
        "/kb/login?next=" + quote("/kb/pipeline/documents?status=active", safe="")
    )


def test_login_redirects_to_pacod_id_with_pkce(client: TestClient) -> None:
    q = start_login(client, "/kb/pipeline/")
    assert q["client_id"] == UI_CLIENT[0]
    assert q["redirect_uri"] == REDIRECT_URI
    assert q["response_type"] == "code"
    assert q["code_challenge_method"] == "S256"
    assert len(q["code_challenge"]) == 43
    assert q["scope"] == "openid email profile"
    assert q["state"]


def test_sign_in_lands_on_next_and_shows_the_person(client: TestClient, fake: FakeId) -> None:
    login_as(client, fake, "editor", email="ed@example.test", next_path="/kb")
    r = client.get("/kb")
    assert r.status_code == 200
    assert "ed@example.test" in r.text
    assert "editor" in r.text
    assert "Sign out" in r.text


def test_the_cookie_carries_only_the_session_secret(client: TestClient, fake: FakeId) -> None:
    import base64
    import json

    login_as(client, fake, "viewer")
    cookie = client.cookies.get("kb_session") or ""
    payload = cookie.split(".", 1)[0]  # itsdangerous: base64(json).timestamp.signature
    data = json.loads(base64.b64decode(payload + "=" * (-len(payload) % 4)))
    assert set(data) == {"s"}
    assert "pidr_" not in cookie and "eyJ" not in data["s"]


@pytest.mark.parametrize(
    ("tamper", "reason"),
    [("state", "state"), ("iss", "issuer"), ("error", "access_denied")],
)
def test_callback_refuses_a_bad_answer(
    client: TestClient, fake: FakeId, tamper: str, reason: str
) -> None:
    fake.account("usr_x", "x@example.test")
    q = start_login(client)
    params = {
        "code": fake.issue_code("usr_x", q["code_challenge"]),
        "state": q["state"],
        "iss": ISSUER,
    }
    if tamper == "state":
        params["state"] = "forged"
    elif tamper == "iss":
        params["iss"] = "http://evil.test"
    else:
        params = {"error": "access_denied", "state": q["state"], "iss": ISSUER}
    r = client.get("/kb/auth/callback", params=params, follow_redirects=False)
    assert r.status_code == 303
    assert parse_qs(urlsplit(r.headers["location"]).query)["error"] == [reason]
    assert client.get("/kb", follow_redirects=False).status_code == 303  # not signed in


def test_callback_without_a_sign_in_in_progress_is_expired(client: TestClient) -> None:
    r = client.get(
        "/kb/auth/callback",
        params={"code": "c", "state": "s", "iss": ISSUER},
        follow_redirects=False,
    )
    assert parse_qs(urlsplit(r.headers["location"]).query)["error"] == ["expired"]
    page = client.get(r.headers["location"])
    assert "took too long" in page.text


def test_a_code_works_once_and_needs_the_verifier(client: TestClient, fake: FakeId) -> None:
    fake.account("usr_y", "y@example.test")
    q = start_login(client)
    code = fake.issue_code("usr_y", "x" * 43)  # a challenge the UI's verifier does not match
    r = client.get(
        "/kb/auth/callback",
        params={"code": code, "state": q["state"], "iss": ISSUER},
        follow_redirects=False,
    )
    assert parse_qs(urlsplit(r.headers["location"]).query)["error"] == ["exchange"]


# ---- roles ------------------------------------------------------------------------------


def test_no_kb_role_sees_the_no_access_page(client: TestClient, fake: FakeId) -> None:
    login_as(client, fake, None, email="stranger@example.test")
    r = client.get("/kb")
    assert r.status_code == 403
    assert "stranger@example.test" in r.text
    assert "role in the knowledge base" in r.text


def test_a_role_change_applies_at_once(client: TestClient, fake: FakeId) -> None:
    sub = login_as(client, fake, "editor")
    wait_for_seq(client, fake.unbind(sub))
    wait_for_seq(client, fake.bind(sub, "kb.viewer"))
    assert client.post("/kb/pipeline/rebuild").status_code == 403
    wait_for_seq(client, fake.unbind(sub))
    assert client.get("/kb").status_code == 403  # no role left


def test_role_order() -> None:
    assert role_satisfies("admin", "editor") and role_satisfies("editor", "viewer")
    assert not role_satisfies("viewer", "editor") and not role_satisfies("", "viewer")
    assert not role_satisfies("admin", "superuser")


# ---- revocation and freshness ---------------------------------------------------------------


def test_a_session_ended_at_pacod_id_signs_out(client: TestClient, fake: FakeId) -> None:
    sub = login_as(client, fake, "viewer")
    assert client.get("/kb").status_code == 200
    wait_for_seq(client, fake.end_session(_sid(fake, sub)))
    r = client.get("/kb", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].endswith("&ended=1")


def test_a_disabled_account_signs_out(client: TestClient, fake: FakeId) -> None:
    sub = login_as(client, fake, "admin")
    wait_for_seq(client, fake.disable(sub))
    assert client.get("/kb", follow_redirects=False).status_code == 303


def test_a_stale_feed_answers_503_but_health_stays_up(
    client: TestClient, fake: FakeId, monkeypatch: pytest.MonkeyPatch
) -> None:
    login_as(client, fake, "viewer")
    identity = client.app.state.identity  # type: ignore[attr-defined]
    monkeypatch.setattr(type(identity.cache), "feed_age", property(lambda _self: 120.0))
    r = client.get("/kb")
    assert r.status_code == 503
    assert r.headers["retry-after"] == "10"
    health = client.get("/kb/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "identity": "stale"}


def test_health_and_login_without_sign_in_set_up(client: TestClient) -> None:
    app = client.app
    saved = app.state.identity  # type: ignore[attr-defined]
    app.state.identity = None  # type: ignore[attr-defined]
    try:
        assert client.get("/kb/healthz").json()["identity"] == "off"
        assert client.get("/kb/login").status_code == 503
        assert client.get("/kb").status_code == 503
    finally:
        app.state.identity = saved  # type: ignore[attr-defined]


# ---- refresh ----------------------------------------------------------------------------------


def test_an_expiring_access_token_is_refreshed(client: TestClient, fake: FakeId) -> None:
    fake.access_ttl = 10  # under the 30-second margin: the next request refreshes
    login_as(client, fake, "viewer")
    fake.access_ttl = 300
    assert client.get("/kb").status_code == 200
    assert fake.calls["token:refresh_token"] == 1
    assert client.get("/kb").status_code == 200
    assert fake.calls["token:refresh_token"] == 1  # fresh now


def test_a_refused_refresh_signs_out(client: TestClient, fake: FakeId) -> None:
    fake.access_ttl = 10
    login_as(client, fake, "viewer")
    fake.grants.clear()  # pacod-id ended the grant
    r = client.get("/kb", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].endswith("&ended=1")


def test_an_unreachable_refresh_is_retried_once_then_503_and_kept(
    client: TestClient, fake: FakeId
) -> None:
    fake.access_ttl = 10
    login_as(client, fake, "viewer")
    fake.access_ttl = 300
    fake.token_failures = 1  # one 503, then fine: the retry succeeds
    assert client.get("/kb").status_code == 200
    assert fake.calls["token:refresh_token"] == 2

    fake.access_ttl = 10
    login_as(client, fake, "viewer")
    fake.token_failures = 2
    assert client.get("/kb").status_code == 503  # the session stays
    fake.access_ttl = 300
    assert client.get("/kb").status_code == 200


def test_concurrent_refreshes_make_one_token_call(fake: FakeId, tmp_path: Path) -> None:
    from tests.test_identity import SETTINGS
    from web.auth import refresh_session
    from web.identity import Identity
    from web.sessions import SessionStore

    async def main() -> tuple[int, ...]:
        identity = Identity(SETTINGS, transport=httpx.MockTransport(fake.handle))
        store = SessionStore(f"sqlite:///{tmp_path}/s.db", "server-secret", 3600)
        store.init()
        fake.account("usr_c", "c@example.test")
        grant = {"sub": "usr_c", "sid": "sid_c"}
        fake.grants["pidr_c"] = grant
        secret = store.create(
            sub="usr_c",
            sid="sid_c",
            email="c@example.test",
            name="c",
            access=fake.access_token("usr_c", "sid_c", expires_in=5),
            refresh="pidr_c",
            access_expires_at=0,
        )
        rec = store.load(secret)
        assert rec is not None
        locks: dict[str, asyncio.Lock] = {}
        results = await asyncio.gather(
            refresh_session(identity, store, locks, secret, rec),
            refresh_session(identity, store, locks, secret, rec),
        )
        await identity.stop()
        return tuple(r.version for r in results if r is not None)

    assert asyncio.run(main()) == (2, 2)
    assert fake.calls["token:refresh_token"] == 1
    assert fake.reuse_detected == 0


# ---- sign-out, origins, HTMX ------------------------------------------------------------------


def test_logout_ends_the_pacod_id_session_and_redirects_there(
    client: TestClient, fake: FakeId
) -> None:
    sub = login_as(client, fake, "admin")
    sid = _sid(fake, sub)
    r = client.post("/kb/logout", follow_redirects=False)
    assert r.status_code == 303
    location = urlsplit(r.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == f"{ISSUER}/oidc/logout"
    q = parse_qs(location.query)
    assert q["client_id"] == [UI_CLIENT[0]]
    assert q["post_logout_redirect_uri"] == ["http://testserver/kb/login?signed_out=1"]
    assert fake.sessions_revoked == [sid]
    assert client.get("/kb", follow_redirects=False).status_code == 303
    assert "signed out" in client.get("/kb/login?signed_out=1").text


def test_logout_is_post_only(client: TestClient) -> None:
    assert client.get("/kb/logout").status_code == 405


def test_cross_origin_writes_are_refused(client: TestClient, fake: FakeId) -> None:
    login_as(client, fake, "admin")
    r = client.post("/kb/logout", headers={"Origin": "http://evil.test"}, follow_redirects=False)
    assert r.status_code == 403
    same = client.post(
        "/kb/logout", headers={"Origin": "http://testserver"}, follow_redirects=False
    )
    assert same.status_code == 303


def test_htmx_requests_get_hx_redirect(client: TestClient) -> None:
    r = client.get("/kb/pipeline/jobs", headers={"HX-Request": "true"}, follow_redirects=False)
    assert r.status_code == 401
    assert r.headers["hx-redirect"].startswith("/kb/login?next=")


def test_login_when_already_signed_in_goes_to_next(client: TestClient, fake: FakeId) -> None:
    login_as(client, fake, "viewer")
    r = client.get("/kb/login", params={"next": "/kb/pipeline/"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/kb/pipeline/"
