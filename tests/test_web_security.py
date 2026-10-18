"""Regression tests for P0 web-UI security fixes (redirects, XSS, admin-only routes)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.fake_id import FakeId
from tests.web_support import login_as, web_client
from web.routers.auth_routes import _safe_next


@pytest.fixture(scope="module")
def fake() -> FakeId:
    return FakeId()


@pytest.fixture(scope="module")
def client(fake: FakeId) -> Iterator[TestClient]:
    with web_client(fake) as c:
        yield c


# ---------------------------------------------------------------------------
# Open redirect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "/\\evil.com",
        "//evil.com",
        "https://evil.com",
        "/\tevil",
        "evil.com",
        "",
        None,
    ],
)
def test_safe_next_rejects_offsite(raw: str | None) -> None:
    assert _safe_next(raw) == "/kb"


@pytest.mark.parametrize("raw", ["/kb/pipeline", "/kb/pipeline/documents?status=active"])
def test_safe_next_keeps_local_paths(raw: str) -> None:
    assert _safe_next(raw) == raw


# ---------------------------------------------------------------------------
# Stored XSS via project name
# ---------------------------------------------------------------------------


def test_project_name_not_injected_into_js(client: TestClient, fake: FakeId) -> None:
    evil = "x');alert(document.cookie);//"
    data: dict[str, Any] = {
        "active": "evil",
        "projects": [{"slug": "evil", "name": evil, "collection_name": "c", "doc_count": 0}],
    }
    login_as(client, fake, "admin")
    with patch("web.routers.pipeline._fetch_projects_data", return_value=data):
        resp = client.get("/kb/pipeline/projects/evil/view")
    assert resp.status_code == 200
    assert "confirmDelete('" not in resp.text
    assert "confirmDelete(this.dataset.slug, this.dataset.name)" in resp.text
    assert 'data-name="x&#39;);alert(document.cookie);//"' in resp.text


def test_project_delete_hidden_for_editor(client: TestClient, fake: FakeId) -> None:
    data: dict[str, Any] = {
        "active": "p",
        "projects": [{"slug": "p", "name": "P", "collection_name": "c", "doc_count": 0}],
    }
    login_as(client, fake, "editor")
    with patch("web.routers.pipeline._fetch_projects_data", return_value=data):
        resp = client.get("/kb/pipeline/projects/p/view")
    assert resp.status_code == 200
    assert "Delete Project" not in resp.text


# ---------------------------------------------------------------------------
# Admin-only routes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,data",
    [
        ("/kb/pipeline/settings/llm-config", {}),
        ("/kb/pipeline/settings/llm-prompts/converter", {"value": "x"}),
        ("/kb/pipeline/settings/security", {"enabled": "false"}),
        ("/kb/pipeline/settings/converter", {"enabled": "true"}),
        ("/kb/pipeline/ai/ping", {"mode": "completion"}),
        ("/kb/pipeline/projects/default/delete", {}),
    ],
)
def test_editor_forbidden_on_admin_routes(
    client: TestClient, fake: FakeId, path: str, data: dict[str, str]
) -> None:
    login_as(client, fake, "editor")
    resp = client.post(path, data=data, follow_redirects=False)
    assert resp.status_code == 403, (path, resp.status_code)


@pytest.mark.parametrize(
    "path",
    [
        "/kb/pipeline/ai-management",
        "/kb/pipeline/usage/summary",
        "/kb/pipeline/ai-interactions",
        "/kb/pipeline/ai-interactions/detail/1",
    ],
)
def test_ai_data_is_admin_only_even_to_read(client: TestClient, fake: FakeId, path: str) -> None:
    # Interactions hold full prompts and document text.
    login_as(client, fake, "editor")
    assert client.get(path, follow_redirects=False).status_code == 403


def test_admin_renders_ai_management_with_the_pipeline_down(
    client: TestClient, fake: FakeId
) -> None:
    login_as(client, fake, "admin")
    resp = client.get("/kb/pipeline/ai-management")
    assert resp.status_code == 200
    assert "AI gateway" in resp.text
    assert "not set up" in resp.text
    assert "Provider API Keys" not in resp.text


def test_admin_passes_role_gate(client: TestClient, fake: FakeId) -> None:
    # The pipeline API is unreachable in tests, so anything but 403 proves the
    # role gate let the admin through.
    login_as(client, fake, "admin")
    resp = client.post("/kb/pipeline/settings/security", data={"enabled": "false"})
    assert resp.status_code != 403


def test_editor_keeps_write_access(client: TestClient, fake: FakeId) -> None:
    import httpx

    login_as(client, fake, "editor")
    try:
        resp = client.post("/kb/pipeline/rebuild")
    except httpx.ConnectError:
        return  # reached the handler (pipeline unreachable in tests) → gate passed
    assert resp.status_code != 403


# ---------------------------------------------------------------------------
# Viewers: read-only search allowed, mutations refused
# ---------------------------------------------------------------------------


def test_viewer_can_search_but_not_mutate(client: TestClient, fake: FakeId) -> None:
    import httpx

    login_as(client, fake, "viewer")
    try:
        resp = client.post("/kb/pipeline/query", data={"query": "x"})
        assert resp.status_code != 403
    except httpx.ConnectError:
        pass  # reached the handler; pipeline unreachable in tests
    assert client.post("/kb/pipeline/rebuild").status_code == 403


def test_dashboard_offers_ai_management_to_admins_only(client: TestClient, fake: FakeId) -> None:
    login_as(client, fake, "editor")
    assert "/kb/pipeline/ai-management" not in client.get("/kb/pipeline/").text
    login_as(client, fake, "admin")
    assert "/kb/pipeline/ai-management" in client.get("/kb/pipeline/").text
