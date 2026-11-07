"""pacod-broker, read side: discovery, the catalogue, linked models' facts, the connection test."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from pipeline import broker, broker_config, runtime_config
from pipeline.broker_config import BrokerConnection
from pipeline.gateway import Gateway, GatewayModel
from pipeline.ledger import ledger_session_factory

URL = "http://broker.test"
DISCOVER = json.loads((Path(__file__).parent / "fixtures/broker/discover.json").read_text())


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    broker.reset_cache()
    broker_config.invalidate()
    runtime_config._CACHE.clear()
    yield
    broker.reset_cache()
    broker_config.invalidate()
    runtime_config._CACHE.clear()


def _conn(token: str = "pidt_kb_token_1234") -> BrokerConnection:
    return BrokerConnection(
        url=URL,
        token=token,
        timeout=5.0,
        sources={"url": "deployment", "timeout_seconds": "deployment", "token": "deployment"},
    )


def _hit(rid: str, name: str, *, kind: str = "model", **resource: Any) -> dict[str, Any]:
    return {
        "id": rid,
        "resource": {"id": rid, "kind": kind, "name": name, **resource},
        "provider": {"name": "acme"},
        "live": {"status": "active"},
    }


def _recording(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    return httpx.MockTransport(wrapped), seen


# ---- a hit ---------------------------------------------------------------------------


def test_a_recorded_hit_is_cut_to_what_the_page_shows() -> None:
    hit = broker.compact_hit(DISCOVER["hits"][0])
    assert set(hit) == {
        "id", "kind", "name", "summary", "provider", "price", "reputation", "status",
        "interfaces", "tags", "upstream_id", "gateway",
    }  # fmt: skip
    assert hit["provider"] == "Sofia GPU Cloud"
    assert hit["price"] == "EUR input 0.25, output 0.5"
    assert hit["status"] == "active"
    assert hit["reputation"] is not None and hit["reputation"]["of"] == "provider"
    assert hit["gateway"] is False


# ---- search --------------------------------------------------------------------------


def test_search_sends_the_words_a_known_kind_and_the_token() -> None:
    transport, seen = _recording(lambda req: httpx.Response(200, json=DISCOVER))
    hits = broker.search(_conn(), "  llama  ", "model", transport=transport)
    assert len(hits) == len(DISCOVER["hits"])
    assert seen[0].url.path == "/v1/discover"
    assert json.loads(seen[0].content) == {"q": "llama", "limit": 20, "kind": "model"}
    assert seen[0].headers["authorization"] == "Bearer pidt_kb_token_1234"

    broker.search(_conn(), "x", "spaceship", transport=transport)
    assert "kind" not in json.loads(seen[1].content)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (httpx.Response(401, json={"type": "unauthorized"}), "refused kb's credentials"),
        (httpx.Response(422, json={"type": "invalid"}), "answered 422, invalid"),
    ],
)
def test_a_refusal_is_reported_without_the_brokers_text(
    answer: httpx.Response, message: str
) -> None:
    transport = httpx.MockTransport(lambda req: answer)
    with pytest.raises(broker.BrokerUnavailableError, match=message):
        broker.search(_conn(), "x", transport=transport)


# ---- the catalogue -------------------------------------------------------------------


def test_the_catalogue_follows_the_cursor_and_stops_after_five_pages() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        page = int(body.get("cursor") or 0)
        if body["kind"] == "model":
            # An endless list: every page has a next one.
            return httpx.Response(
                200, json={"hits": [_hit(f"res_m{page}", f"m{page}")], "next_cursor": str(page + 1)}
            )
        return httpx.Response(200, json={"hits": [_hit("res_e", "e", kind="embedding")]})

    transport, seen = _recording(handler)
    out = broker.catalogue(_conn(), [], transport=transport)
    assert [(json.loads(r.content)["kind"], json.loads(r.content).get("cursor")) for r in seen] == [
        ("model", None), ("model", "1"), ("model", "2"), ("model", "3"), ("model", "4"),
        ("embedding", None),
    ]  # fmt: skip
    assert out["truncated"] is True
    ids = [i["id"] for i in out["items"]]
    assert ids == ["res_e", "res_m0", "res_m1", "res_m2", "res_m3", "res_m4"]

    broker.catalogue(_conn(), [], transport=transport)
    assert len(seen) == 6  # cached for a minute


def test_the_catalogue_marks_how_the_gateway_offers_each_entry() -> None:
    hits = [
        _hit("res_linked", "Sonnet"),
        _hit("res_tpl", "gw fast", tags=["gateway"], model={"upstream_id": "@sonnet-fast"}),
        _hit("res_tpl_unknown", "gw other", tags=["gateway"], model={"upstream_id": "@gone"}),
        # Not a gateway offering: an upstream id that looks like a template does not count.
        _hit("res_foreign", "foreign", model={"upstream_id": "@sonnet-fast"}),
    ]
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"hits": hits}))
    rows = [
        GatewayModel("anthropic/claude-sonnet-5-5", "model", broker_resource_id="res_linked"),
        GatewayModel("@sonnet-fast", "template", model="google/gemini-3-flash"),
    ]
    listed = broker.catalogue(_conn(), rows, transport=transport)["items"]
    items = {i["id"]: i["usage"] for i in listed}
    assert items["res_linked"] == {"state": "usable", "call": "anthropic/claude-sonnet-5-5"}
    assert items["res_tpl"] == {"state": "usable", "call": "@sonnet-fast"}
    assert items["res_tpl_unknown"] == {"state": "not_in_gateway", "call": None}
    assert items["res_foreign"] == {"state": "not_in_gateway", "call": None}


def test_a_dead_broker_ends_the_catalogue_after_one_call() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    transport, seen = _recording(handler)
    with pytest.raises(broker.BrokerUnavailableError, match="cannot be reached"):
        broker.catalogue(_conn(), [], transport=transport)
    assert len(seen) == 1


# ---- facts ---------------------------------------------------------------------------


def test_facts_are_read_once_a_minute_and_an_unknown_id_is_none() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        rid = req.url.path.rsplit("/", 1)[-1]
        if rid == "res_gone":
            return httpx.Response(404, json={"type": "not_found"})
        return httpx.Response(200, json={"id": rid, "kind": "model", "name": rid,
                                         "reputation": {"score": 0.8123, "label": "good"},
                                         "price_now": {"free": True}})  # fmt: skip

    transport, seen = _recording(handler)
    facts = broker.resource_facts(
        _conn(), ["res_a", "res_gone", "res_a", "not-a-resource"], transport=transport
    )
    assert facts["res_gone"] is None
    assert facts["res_a"] is not None
    assert facts["res_a"]["reputation"] == {"score": 0.812, "label": "good", "of": "resource"}
    assert facts["res_a"]["price"] == "free"
    assert "not-a-resource" not in facts
    assert len(seen) == 2

    broker.resource_facts(_conn(), ["res_a", "res_gone"], transport=transport)
    assert len(seen) == 2


def test_a_dead_broker_costs_one_call_however_many_models_are_linked() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    transport, seen = _recording(handler)
    rows = [
        GatewayModel(f"p/m{i}", "model", broker_resource_id=f"res_{i}") for i in range(20)
    ]
    view = broker.overview(_conn(), rows, {"converter": "p/m0"}, transport=transport)
    assert len(seen) == 1
    assert view["broker_error"] == "the broker cannot be reached (ConnectError)"
    assert all(entry["facts"] is None for entry in view["linked"])


def test_the_overview_follows_a_template_to_its_models_resource() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        rid = req.url.path.rsplit("/", 1)[-1]
        reputation = {"score": 0.9, "label": "excellent"}
        return httpx.Response(
            200, json={"id": rid, "kind": "model", "name": "Sonnet", "reputation": reputation}
        )

    rows = [
        GatewayModel("anthropic/claude-sonnet-5-5", "model", broker_resource_id="res_sonnet"),
        GatewayModel("openai/gpt-5", "model"),
        GatewayModel("@careful", "template", model="anthropic/claude-sonnet-5-5"),
    ]
    view = broker.overview(
        _conn(),
        rows,
        {"converter": "@careful", "security": "openai/gpt-5"},
        transport=httpx.MockTransport(handler),
    )
    converter, security = view["in_use"]
    assert (converter["uses"], converter["broker_resource_id"]) == (
        "anthropic/claude-sonnet-5-5",
        "res_sonnet",
    )
    assert converter["facts"]["reputation"]["label"] == "excellent"
    assert (security["uses"], security["broker_resource_id"], security["facts"]) == (
        None,
        None,
        None,
    )
    assert view["linked"] == [
        {
            "id": "anthropic/claude-sonnet-5-5",
            "broker_resource_id": "res_sonnet",
            "facts": converter["facts"],
            "used_as": ["converter"],
        }
    ]
    assert view["broker_error"] is None


# ---- the connection test --------------------------------------------------------------


def _broker(status: dict[str, int], me: httpx.Response | None = None) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path in status:
            return httpx.Response(status[req.url.path], text="ok")
        if req.url.path == "/v1/reputation/me":
            return me or httpx.Response(200, json={"client": "svc_kb", "provider_id": "prv_kb"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_the_probe_tells_unreachable_not_ready_refused_and_accepted_apart() -> None:
    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    out = broker.probe(_conn(), transport=httpx.MockTransport(down))
    assert (out["reachable"], out["ready"], out["accepted"]) == (False, False, None)

    out = broker.probe(_conn(), transport=_broker({"/healthz": 200, "/readyz": 503}))
    assert (out["reachable"], out["ready"], out["accepted"]) == (True, False, None)
    assert "not ready" in out["error"]

    refused = httpx.Response(401, json={"type": "unauthorized"})
    out = broker.probe(_conn(), transport=_broker({"/healthz": 200, "/readyz": 200}, refused))
    assert (out["ready"], out["accepted"]) == (True, False)
    assert "open mode" in out["hint"]

    out = broker.probe(_conn(), transport=_broker({"/healthz": 200, "/readyz": 200}))
    assert (out["accepted"], out["client"], out["provider_id"]) == (True, "svc_kb", "prv_kb")
    assert out["error"] is None


# ---- the routes ------------------------------------------------------------------------


@pytest.fixture()
def api_client(config: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from pipeline import api

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    state = api.app.state
    saved = {k: getattr(state, k, None) for k in ("config", "ledger_factory", "gateway")}
    state.config = replace(config, broker_url=URL, broker_token="pidt_kb_token_1234")
    state.ledger_factory = ledger_session_factory(config)
    state.gateway = Gateway("", "")  # not set up
    try:
        with TestClient(api.app, headers={"X-API-Key": "test-secret"}) as c:
            yield c
    finally:
        for key, value in saved.items():
            if value is None:
                if hasattr(state, key):
                    delattr(state, key)
            else:
                setattr(state, key, value)
        if hasattr(state, "broker_transport"):
            del state.broker_transport


def test_a_broker_that_is_down_costs_the_routes_their_facts_never_a_5xx(
    api_client: TestClient,
) -> None:
    from pipeline import api

    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    api.app.state.broker_transport = httpx.MockTransport(down)
    for path in ("/broker/catalogue", "/broker/search?q=llama"):
        resp = api_client.get(path)
        assert resp.status_code == 200, path
        assert resp.json()["broker_error"] == "the broker cannot be reached (ConnectError)", path
    # No gateway, so no linked models to look up: the overview asks the broker nothing.
    overview = api_client.get("/broker")
    assert overview.status_code == 200
    assert overview.json()["gateway_error"] == "the AI gateway is not set up"
    assert overview.json()["broker_error"] is None
    probe = api_client.post("/broker/config/test").json()
    assert probe["reachable"] is False


def test_the_config_routes_keep_omitted_fields_and_name_their_refusals(
    api_client: TestClient,
) -> None:
    described = api_client.get("/broker/config").json()
    assert described["url"] == {"value": URL, "source": "deployment", "default": URL}
    assert described["token"]["source"] == "deployment"

    resp = api_client.put("/broker/config", json={"url": "http://other.test"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "token_required"

    resp = api_client.put("/broker/config", json={"url": "http://other.test", "token": ""})
    assert resp.status_code == 200, resp.text
    assert resp.json()["url"]["source"] == "console"
    assert resp.json()["token"]["source"] == "none"
    assert resp.json()["timeout_seconds"]["source"] == "deployment"  # omitted: kept

    resp = api_client.put("/broker/config", json={"timeout_seconds": 9})
    assert resp.json()["url"]["value"] == "http://other.test"  # omitted again: kept
    assert resp.json()["timeout_seconds"] == {"value": 9.0, "source": "console", "default": 5.0}

    assert api_client.put("/broker/config", json={"surprise": 1}).status_code == 422

    resp = api_client.delete("/broker/config")
    assert resp.json()["url"]["source"] == "deployment"
    assert "pidt_kb_token_1234" not in resp.text


def test_an_address_that_is_not_the_brokers_api_costs_one_call_and_no_5xx(
    api_client: TestClient,
) -> None:
    # The broker's console (or a proxy's login page) answers 200 with HTML at every path.
    from pipeline import api

    seen: list[str] = []

    def html(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        return httpx.Response(200, text="<!doctype html><title>console</title>")

    listing = {"data": [{"id": f"p/m{i}", "gateway": {"kind": "model",
                                                     "broker_resource_id": f"res_{i}"}}
                        for i in range(6)]}  # fmt: skip
    api.app.state.gateway = Gateway(
        "http://gw.test",
        "pgw_test",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=listing)),
    )
    api.app.state.broker_transport = httpx.MockTransport(html)

    overview = api_client.get("/broker")
    assert overview.status_code == 200
    assert "not pacod-broker's API" in overview.json()["broker_error"]
    assert [p for p in seen if p.startswith("/v1/resources/")] == ["/v1/resources/res_0"]
    for path in ("/broker/catalogue", "/broker/search?q=llama"):
        resp = api_client.get(path)
        assert resp.status_code == 200, path
        assert "not pacod-broker's API" in resp.json()["broker_error"], path
    probe = api_client.post("/broker/config/test")
    assert probe.status_code == 200
    assert (probe.json()["reachable"], probe.json()["accepted"]) == (True, None)
    assert "not pacod-broker's API" in probe.json()["error"]


def test_a_role_that_is_off_is_marked_off() -> None:
    view = broker.overview(
        _conn(), [], {"converter": "a/b", "security": "c/d"}, enabled={"security": False}
    )
    assert [(u["role"], u["enabled"]) for u in view["in_use"]] == [
        ("converter", True),
        ("security", False),
    ]
