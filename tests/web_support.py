"""The UI against the fake pacod-id: a TestClient whose sign-in really runs (no backdoor)."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from urllib.parse import parse_qs, urlsplit

import httpx
from fastapi.testclient import TestClient

from tests.fake_id import ISSUER, FakeId


def wait_until(cond: Callable[[], bool], within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


@contextmanager
def web_client(
    fake: FakeId, pipeline: httpx.AsyncBaseTransport | None = None
) -> Iterator[TestClient]:
    """The UI app with *fake* as its pacod-id (and *pipeline*, if given, as the pipeline API),
    its feed followed before the first request."""
    from web.app import app

    app.state.id_transport = httpx.MockTransport(fake.handle)
    if pipeline is not None:
        app.state.pipeline_transport = pipeline
    try:
        with TestClient(app) as client:
            wait_until(lambda: app.state.identity is not None and app.state.identity.cache.ready)
            yield client
    finally:
        del app.state.id_transport
        if pipeline is not None:
            del app.state.pipeline_transport


def wait_for_seq(client: TestClient, seq: int) -> None:
    """Until the UI's copy of pacod-id's decisions has change *seq*."""
    identity = client.app.state.identity  # type: ignore[attr-defined]
    wait_until(lambda: identity.cache.seq >= seq)


def start_login(client: TestClient, next_path: str = "/kb") -> dict[str, str]:
    """GET /kb/login; returns the authorize URL's parameters."""
    r = client.get("/kb/login", params={"next": next_path}, follow_redirects=False)
    assert r.status_code == 303, r.text
    location = r.headers["location"]
    assert location.startswith(f"{ISSUER}/oidc/authorize?"), location
    return {k: v[0] for k, v in parse_qs(urlsplit(location).query).items()}


def login_as(
    client: TestClient,
    fake: FakeId,
    role: str | None = "admin",
    *,
    sub: str | None = None,
    email: str | None = None,
    next_path: str = "/kb",
) -> str:
    """Sign in through the real flow as a new account holding kb.<role> (None: no kb role)."""
    sub = sub or f"usr_{role or 'none'}{secrets.token_hex(3)}"
    fake.account(sub, email or f"{sub}@example.test")
    if role:
        wait_for_seq(client, fake.bind(sub, f"kb.{role}"))
    client.cookies.clear()
    q = start_login(client, next_path)
    code = fake.issue_code(sub, q["code_challenge"], q["redirect_uri"])
    r = client.get(
        "/kb/auth/callback",
        params={"code": code, "state": q["state"], "iss": ISSUER},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    assert r.headers["location"] == next_path
    return sub
