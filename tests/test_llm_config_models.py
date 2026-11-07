"""AI Management's model pickers: the gateway's models, then its templates (@name)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from pipeline import runtime_config
from pipeline.gateway import Gateway
from pipeline.ledger import ledger_session_factory

_LIST = {
    "object": "list",
    "data": [
        {"id": "openai/gpt-5", "gateway": {"kind": "model", "state": "active"}},
        {"id": "@careful", "gateway": {"kind": "template", "model": "anthropic/claude-sonnet-5-5"}},
        {"id": "anthropic/claude-sonnet-5-5", "gateway": {"kind": "model", "state": "active"}},
    ],
}


@pytest.fixture(autouse=True)
def _fresh_caches() -> Iterator[None]:
    runtime_config._CACHE.clear()
    yield
    runtime_config._CACHE.clear()


def _client(
    config: Any,
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response] | None,
) -> Iterator[TestClient]:
    from pipeline import api

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    previous_config = getattr(api.app.state, "config", None)
    api.app.state.config = config
    api.app.state.ledger_factory = ledger_session_factory(config)
    api.app.state.gateway = (
        Gateway("http://gw.test", "pgw_test", transport=httpx.MockTransport(handler))
        if handler is not None
        else Gateway("", "")
    )
    try:
        with TestClient(api.app, headers={"X-API-Key": "test-secret"}) as c:
            yield c
    finally:
        if previous_config is None:
            del api.app.state.config
        else:
            api.app.state.config = previous_config
        del api.app.state.ledger_factory
        del api.app.state.gateway


@pytest.fixture()
def api_client(config: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _client(config, monkeypatch, lambda req: httpx.Response(200, json=_LIST))


def test_the_picker_offers_the_models_then_the_templates(api_client: TestClient) -> None:
    state = api_client.get("/llm-config").json()
    gateway_part = state["available_models"][:3]
    assert gateway_part == ["anthropic/claude-sonnet-5-5", "openai/gpt-5", "@careful"]
    assert state["templates"] == {"@careful": "anthropic/claude-sonnet-5-5"}


def test_a_template_the_gateway_lists_can_be_saved(api_client: TestClient) -> None:
    resp = api_client.patch("/llm-config", json={"converter_model": "@careful"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["effective"]["converter_model"] == "@careful"
    assert resp.json()["templates"] == {"@careful": "anthropic/claude-sonnet-5-5"}

    status = api_client.get("/ai/gateway").json()
    assert status["converter_model"] == "@careful"
    assert status["converter_model_uses"] == "anthropic/claude-sonnet-5-5"


def test_a_name_the_gateway_does_not_list_is_refused_and_not_saved(
    api_client: TestClient,
) -> None:
    resp = api_client.patch("/llm-config", json={"converter_model": "@deleted"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "unknown_model"
    assert "@deleted" in resp.json()["detail"]["message"]
    runtime_config._CACHE.clear()
    assert api_client.get("/llm-config").json()["effective"]["converter_model"] != "@deleted"


def test_an_unchanged_name_is_not_checked(api_client: TestClient) -> None:
    # The form sends both names on every save; the env default security model is not one
    # the gateway lists here, yet changing only the token limit must succeed.
    current = api_client.get("/llm-config").json()["effective"]
    assert current["security_model"] not in {m["id"] for m in _LIST["data"]}
    resp = api_client.patch(
        "/llm-config",
        json={
            "converter_model": "openai/gpt-5",
            "security_model": current["security_model"],
            "converter_max_tokens": 1234,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["effective"]["converter_max_tokens"] == 1234


def test_without_a_list_nothing_is_checked(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    for handler in (down, None):  # unreachable, and not set up
        runtime_config._CACHE.clear()
        for client in _client(config, monkeypatch, handler):
            resp = client.patch("/llm-config", json={"converter_model": "@anything"})
            assert resp.status_code == 200, resp.text


def test_the_connection_test_lists_models_and_templates_apart(api_client: TestClient) -> None:
    ping = api_client.post("/ai/gateway/ping", json={"mode": "models"}).json()
    assert ping["ok"] is True
    assert ping["models"] == ["anthropic/claude-sonnet-5-5", "openai/gpt-5"]
    assert ping["templates"] == ["@careful"]


# ---- the UI's Models panel ---------------------------------------------------------------


def _ui_pipeline(refuse: bool) -> httpx.MockTransport:
    state = {
        "effective": {"converter_model": "anthropic/claude-sonnet-5-5"},
        "env_default": {},
        "available_models": ["anthropic/claude-sonnet-5-5", "openai/gpt-5", "@careful"],
        "templates": {"@careful": "anthropic/claude-sonnet-5-5"},
    }

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/llm-config" and request.method == "PATCH":
            if refuse:
                message = "The AI gateway does not offer @x <b>."
                detail = {"code": "unknown_model", "message": message}
                return httpx.Response(422, json={"detail": detail})
            return httpx.Response(200, json=state)
        if request.url.path == "/llm-config":
            return httpx.Response(200, json=state)
        return httpx.Response(200, json={})

    return httpx.MockTransport(handle)


@pytest.mark.parametrize("refuse", [False, True])
def test_the_models_panel_labels_templates_and_names_a_refused_model(refuse: bool) -> None:
    from tests.fake_id import FakeId
    from tests.web_support import login_as, web_client

    fake = FakeId()
    with web_client(fake, _ui_pipeline(refuse)) as client:
        login_as(client, fake, "admin")
        html = client.post(
            "/kb/pipeline/settings/llm-config",
            data={"converter_model": "@x", "security_model": "openai/gpt-5"},
        ).text
    assert "@careful — uses anthropic/claude-sonnet-5-5" in html
    assert "<select" in html
    if refuse:
        assert "The AI gateway does not offer @x &lt;b&gt;." in html
        # the value as typed stays selectable, marked as current
        assert '<option value="@x" selected>@x (current)</option>' in html
    else:
        assert "Saved." in html


def test_the_models_panel_lists_broker_models_the_gateway_lacks_as_disabled() -> None:
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("web/templates"), autoescape=True)
    env.globals["notice"] = lambda *a, **k: ""
    state = {
        "effective": {"converter_model": "openai/gpt-5"},
        "env_default": {},
        "available_models": ["openai/gpt-5", "@careful"],
        "templates": {"@careful": "openai/gpt-5"},
        "broker_models": [
            {"name": "acme/big", "price": "EUR input 1", "usable": False},
            {"name": "openai/gpt-5", "price": "", "usable": True},
        ],
        "broker_error": None,
    }
    html = env.get_template("pipeline/_llm_config_panel.html").render(llm_state=state)
    assert "<option disabled>acme/big (EUR input 1)</option>" in html
    assert "<option disabled>openai/gpt-5" not in html
    assert '<option value="openai/gpt-5" selected>' in html
