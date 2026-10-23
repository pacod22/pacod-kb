"""The top bar's project picker: a switch reloads the page (HX-Refresh); a refused one does not,
and the bar keeps showing the project that is still active."""

from __future__ import annotations

import httpx
import pytest

from tests.fake_id import FakeId
from tests.web_support import login_as, web_client

_PROJECTS = [{"slug": "alpha", "name": "Alpha"}, {"slug": "beta", "name": "Beta"}]


def _pipeline(refuse: bool) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/settings" and request.method == "POST":
            if refuse:
                return httpx.Response(409, text="busy")
            return httpx.Response(200, json={"active_project": "beta"})
        if request.url.path == "/projects":
            active = "alpha" if refuse else "beta"
            return httpx.Response(200, json={"projects": _PROJECTS, "active": active})
        return httpx.Response(200, json={})

    return httpx.MockTransport(handle)


@pytest.fixture(scope="module")
def fake() -> FakeId:
    return FakeId()


def test_switch_reloads_the_page(fake: FakeId) -> None:
    with web_client(fake, pipeline=_pipeline(refuse=False)) as client:
        login_as(client, fake, "editor")
        r = client.post("/kb/pipeline/settings/project", data={"project": "beta", "sidebar": "1"})
        assert r.status_code == 200
        assert r.headers.get("HX-Refresh") == "true"
        assert 'id="project-bar"' in r.text
        assert '<option value="beta" selected>' in r.text


def test_refused_switch_keeps_the_active_project(fake: FakeId) -> None:
    with web_client(fake, pipeline=_pipeline(refuse=True)) as client:
        login_as(client, fake, "editor")
        r = client.post("/kb/pipeline/settings/project", data={"project": "beta", "sidebar": "1"})
        assert r.status_code == 200
        assert "HX-Refresh" not in r.headers
        assert "Cannot switch while a job is running." in r.text
        assert '<option value="alpha" selected>' in r.text
        assert '<option value="beta" selected>' not in r.text
