"""kb's connection to pacod-broker: the deployment's or an admin's; a token keeps to its address."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import httpx
import pytest
from sqlalchemy import select

from pipeline import broker, broker_config
from pipeline.broker_config import CONSOLE, DEPLOYMENT, KEYS, NONE, BrokerConfigError
from pipeline.db_models import AppSetting
from pipeline.document_store import _session_scope
from pipeline.ledger import ledger_session_factory

SECRET = "pipeline-api-secret-for-tests"
DEPLOYED = "http://broker-a.test"
TOKEN_A = "pidt_deployment_token_aaaa"


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    broker_config.invalidate()
    broker.reset_cache()
    yield
    broker_config.invalidate()
    broker.reset_cache()


@pytest.fixture()
def cfg(config: Any) -> Any:
    return replace(config, broker_url=DEPLOYED, broker_token=TOKEN_A, broker_timeout_secs=5.0)


@pytest.fixture()
def factory(config: Any) -> Any:
    return ledger_session_factory(config)


def _rows(factory: Any) -> dict[str, str]:
    with _session_scope(factory) as s:
        rows = s.execute(select(AppSetting).where(AppSetting.key.in_(KEYS))).scalars().all()
        return {r.key: r.value for r in rows}


def _conn(cfg: Any, factory: Any, secret: str = SECRET) -> broker_config.BrokerConnection:
    return broker_config.get_connection(cfg, factory, secret)


def test_the_deployment_is_in_force_until_an_admin_sets_something(cfg: Any, factory: Any) -> None:
    conn = _conn(cfg, factory)
    assert (conn.url, conn.token, conn.timeout) == (DEPLOYED, TOKEN_A, 5.0)
    assert conn.sources == {"url": DEPLOYMENT, "timeout_seconds": DEPLOYMENT, "token": DEPLOYMENT}
    assert TOKEN_A not in repr(conn)


def test_a_saved_token_belongs_to_its_address_and_is_encrypted(cfg: Any, factory: Any) -> None:
    other = "https://broker-b.test"
    conn = broker_config.save(
        cfg, factory, SECRET, {"url": other + "/"}, token="pidt_console_token_bbbb"
    )
    assert (conn.url, conn.token) == (other, "pidt_console_token_bbbb")
    assert conn.sources["url"] == CONSOLE and conn.sources["token"] == CONSOLE
    stored = _rows(factory)
    assert stored[broker_config.TOKEN_URL_KEY] == other
    assert "pidt_console_token_bbbb" not in stored[broker_config.TOKEN_KEY]

    # Back at the deployment's address, the deployment's token applies again; the console
    # token is sent nowhere but its own address.
    conn = broker_config.save(cfg, factory, SECRET, {"url": None})
    assert (conn.url, conn.token, conn.sources["token"]) == (DEPLOYED, TOKEN_A, DEPLOYMENT)


def test_another_address_without_its_token_is_refused_and_nothing_is_written(
    cfg: Any, factory: Any
) -> None:
    with pytest.raises(BrokerConfigError) as err:
        broker_config.save(cfg, factory, SECRET, {"url": "http://elsewhere.test"})
    assert err.value.code == "token_required"
    assert _rows(factory) == {}
    assert _conn(cfg, factory).url == DEPLOYED


def test_open_mode_for_another_address_sends_no_token_and_names_kb(
    cfg: Any, factory: Any
) -> None:
    conn = broker_config.save(cfg, factory, SECRET, {"url": "http://open.test"}, token="")
    assert (conn.token, conn.sources["token"]) == ("", NONE)

    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"hits": []})

    broker.search(conn, "llama", transport=httpx.MockTransport(handler))
    assert "authorization" not in seen[0].headers
    assert seen[0].headers["x-broker-client"] == "pacod-kb"


def test_without_a_token_in_use_a_new_address_needs_no_decision(cfg: Any, factory: Any) -> None:
    open_deployment = replace(cfg, broker_token="")
    conn = broker_config.save(open_deployment, factory, SECRET, {"url": "http://b.test"})
    assert (conn.url, conn.token) == ("http://b.test", "")


def test_the_deployments_token_never_follows_the_url_elsewhere(cfg: Any, factory: Any) -> None:
    # A stale token row for another address must not make the deployment's token travel.
    conn = broker_config.resolve(cfg, {broker_config.URL_KEY: "http://c.test"}, SECRET)
    assert (conn.token, conn.sources["token"]) == ("", NONE)


def test_a_changed_secret_makes_the_saved_token_unreadable(cfg: Any, factory: Any) -> None:
    broker_config.save(cfg, factory, SECRET, {"url": "http://b.test"}, token="pidt_bbbbbbbbbb")
    broker_config.invalidate()
    conn = _conn(cfg, factory, secret="another-secret")
    assert (conn.token, conn.token_unreadable, conn.sources["token"]) == ("", True, CONSOLE)
    assert broker_config.describe(cfg, conn, "another-secret")["token"]["unreadable"] is True


def test_no_secret_no_saved_token(cfg: Any, factory: Any) -> None:
    with pytest.raises(BrokerConfigError) as err:
        broker_config.save(cfg, factory, "", {}, token="pidt_bbbbbbbbbb")
    assert (err.value.code, err.value.status) == ("secrets_disabled", 503)
    assert _rows(factory) == {}


@pytest.mark.parametrize(
    "url",
    [
        "ftp://broker.test",
        "http://",
        "http://user:pw@broker.test",
        "http://broker.test/?q=1",
        "http://broker.test/#x",
        "http://bro ker.test",
        "http://broker.test:99999",
        "http://" + "a" * 600 + ".test",
    ],
)
def test_bad_addresses_are_refused(cfg: Any, factory: Any, url: str) -> None:
    with pytest.raises(BrokerConfigError) as err:
        broker_config.save(cfg, factory, SECRET, {"url": url}, token="")
    assert err.value.code == "invalid_url"


@pytest.mark.parametrize("value", [0.5, 31, True, "5"])
def test_timeouts_outside_the_bounds_are_refused(cfg: Any, factory: Any, value: Any) -> None:
    with pytest.raises(BrokerConfigError) as err:
        broker_config.save(cfg, factory, SECRET, {"timeout_seconds": value})
    assert err.value.code == "invalid_timeout"


def test_a_blank_url_turns_the_broker_off(cfg: Any, factory: Any) -> None:
    conn = broker_config.save(cfg, factory, SECRET, {"url": ""})
    assert not conn.configured
    with pytest.raises(broker.BrokerUnavailableError):
        broker.search(conn, "x")


def test_describe_never_contains_the_token(cfg: Any, factory: Any) -> None:
    broker_config.save(cfg, factory, SECRET, {"url": "http://b.test"}, token="pidt_secret_9876")
    text = repr(broker_config.describe(cfg, _conn(cfg, factory), SECRET))
    assert "pidt_secret" not in text
    described = broker_config.describe(cfg, _conn(cfg, factory), SECRET)
    assert described["token"]["last4"] == "9876"
    assert described["url"] == {"value": "http://b.test", "source": CONSOLE, "default": DEPLOYED}

    broker_config.save(cfg, factory, SECRET, {}, token="short")
    assert broker_config.describe(cfg, _conn(cfg, factory), SECRET)["token"]["last4"] is None


def test_reset_returns_to_the_deployment(cfg: Any, factory: Any) -> None:
    broker_config.save(
        cfg, factory, SECRET, {"url": "http://b.test", "timeout_seconds": 12}, token=""
    )
    assert _rows(factory)
    broker_config.reset(factory)
    assert _rows(factory) == {}
    conn = _conn(cfg, factory)
    assert (conn.url, conn.token, conn.timeout) == (DEPLOYED, TOKEN_A, 5.0)
