"""The members page: pacod-id's admin API through token exchange, acting for the admin."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.fake_id import FakeId
from tests.web_support import login_as, wait_for_seq, web_client

EXCHANGE = "token:urn:ietf:params:oauth:grant-type:token-exchange"


@pytest.fixture()
def fake() -> FakeId:
    return FakeId()


@pytest.fixture()
def client(fake: FakeId) -> Iterator[TestClient]:
    with web_client(fake) as c:
        yield c


@pytest.fixture()
def admin(client: TestClient, fake: FakeId) -> str:
    return login_as(client, fake, "admin", email="boss@example.test")


def _member(fake: FakeId, client: TestClient, role: str, email: str) -> str:
    sub = f"usr_{email.split('@')[0]}"
    fake.account(sub, email)
    wait_for_seq(client, fake.bind(sub, f"kb.{role}"))
    return sub


def test_non_admins_get_403(client: TestClient, fake: FakeId) -> None:
    login_as(client, fake, "editor")
    assert client.get("/kb/users").status_code == 403
    assert client.post("/kb/users/invite", data={"email": "a@b.test"}).status_code == 403


def test_members_are_listed_with_their_kb_role(
    client: TestClient, fake: FakeId, admin: str
) -> None:
    _member(fake, client, "editor", "ed@example.test")
    r = client.get("/kb/users")
    assert r.status_code == 200
    assert "ed@example.test" in r.text and "boss@example.test" in r.text
    assert "(you)" in r.text
    assert fake.calls[EXCHANGE] == 1  # one exchange for both admin calls (cached)


def test_invite_by_mail_and_by_link(client: TestClient, fake: FakeId, admin: str) -> None:
    r = client.post("/kb/users/invite", data={"email": "new@example.test", "role": "editor"})
    assert r.status_code == 200
    assert "Invited new@example.test as editor" in r.text
    (inv,) = fake.invitations
    assert (inv["role"], inv["scope"], inv["emailed"]) == ("kb.editor", "platform", True)

    r = client.post(
        "/kb/users/invite", data={"email": "link@example.test", "role": "viewer", "as_link": "1"}
    )
    assert "/invite?token=" in r.text  # shown once

    again = client.post("/kb/users/invite", data={"email": "new@example.test", "role": "viewer"})
    assert again.status_code == 409
    assert "already pending" in again.text


def test_invite_refuses_an_unknown_role(client: TestClient, admin: str) -> None:
    r = client.post("/kb/users/invite", data={"email": "x@example.test", "role": "superuser"})
    assert r.status_code == 422


def test_change_role_and_remove(client: TestClient, fake: FakeId, admin: str) -> None:
    sub = _member(fake, client, "editor", "ed@example.test")
    identity = client.app.state.identity  # type: ignore[attr-defined]

    r = client.post(f"/kb/users/{sub}/role", data={"role": "viewer"})
    assert r.status_code == 200 and "Role changed to viewer" in r.text
    assert identity.role_of(sub) == "viewer"  # the page waited for the feed

    r = client.post(f"/kb/users/{sub}/remove")
    assert r.status_code == 200 and "Removed" in r.text
    assert identity.role_of(sub) is None


def test_admins_cannot_change_or_remove_themselves(client: TestClient, admin: str) -> None:
    assert client.post(f"/kb/users/{admin}/role", data={"role": "viewer"}).status_code == 409
    assert client.post(f"/kb/users/{admin}/remove").status_code == 409


def test_pacod_id_refusals_are_explained(
    client: TestClient, fake: FakeId, admin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sub = _member(fake, client, "admin", "other@example.test")
    monkeypatch.setattr(fake, "_last_admin_would_go", lambda *_a: True)
    r = client.post(f"/kb/users/{sub}/role", data={"role": "editor"})
    assert r.status_code == 409
    assert "needs another admin" in r.text


def test_a_refused_admin_token_is_exchanged_again_once(
    client: TestClient, fake: FakeId, admin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = fake._admin
    refused = {"n": 0}

    def flaky(request: httpx.Request, path: str) -> httpx.Response:
        if refused["n"] == 0:
            refused["n"] += 1
            return httpx.Response(401, json={"error": {"details": {"code": "invalid_token"}}})
        return real(request, path)

    monkeypatch.setattr(fake, "_admin", flaky)
    assert client.get("/kb/users").status_code == 200
    assert fake.calls[EXCHANGE] == 2


def test_invitations_can_be_revoked(client: TestClient, fake: FakeId, admin: str) -> None:
    client.post("/kb/users/invite", data={"email": "gone@example.test", "role": "viewer"})
    (inv,) = fake.invitations
    r = client.post(f"/kb/users/invitations/{inv['id']}/revoke")
    assert r.status_code == 200 and "revoked" in r.text
    assert fake.invitations == []


def test_ids_are_validated(client: TestClient, admin: str) -> None:
    assert client.post("/kb/users/..%2Fadmin/role", data={"role": "viewer"}).status_code in (
        404,
        405,
    )
    assert client.post("/kb/users/invitations/x;y/revoke").status_code == 404


def test_former_local_accounts_are_offered_for_invitation(
    client: TestClient, fake: FakeId, admin: str
) -> None:
    path = os.environ["UI_DB_PATH"]
    con = sqlite3.connect(path)
    try:
        con.execute("DROP TABLE IF EXISTS ui_users")
        con.execute("CREATE TABLE ui_users (email TEXT, role TEXT, is_active BOOLEAN)")
        rows: list[tuple[Any, ...]] = [
            ("boss@example.test", "admin", 1),  # already a member
            ("Old@Example.test", "editor", 1),
            ("off@example.test", "viewer", 0),  # inactive: not offered
        ]
        con.executemany("INSERT INTO ui_users VALUES (?, ?, ?)", rows)
        con.commit()
        r = client.get("/kb/users")
    finally:
        con.execute("DROP TABLE ui_users")
        con.commit()
        con.close()
    assert "Former local accounts" in r.text
    assert "old@example.test" in r.text
    assert "off@example.test" not in r.text
    assert r.text.count("boss@example.test") == 2  # the members row and the sidebar only
