"""Runtime LLM settings: five fields, gateway model names, legacy keys honoured."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pipeline import runtime_config
from pipeline.runtime_config import (
    converter_settings,
    env_runtime_config,
    get_runtime_config,
    security_model,
    set_runtime_config,
    valid_model_name,
)


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    runtime_config._CACHE.clear()
    yield
    runtime_config._CACHE.clear()


@pytest.fixture()
def factory(tmp_path: Any) -> Any:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from pipeline.db_models import Base

    engine = create_engine(f"sqlite:///{tmp_path}/ledger.db")
    Base.metadata.create_all(engine)
    return sessionmaker(engine)


def test_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LLM_CONVERTER_MODEL", "LLM_CONVERTER_TEMPERATURE", "LLM_CONVERTER_MAX_TOKENS",
                "CONTENT_SECURITY_LLM_MODEL", "LLM_CONVERTER_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    assert env_runtime_config() == {
        "converter_model": "anthropic/claude-opus-5-5",
        "converter_max_tokens": 64000,
        "converter_temperature": 0.2,
        "converter_temperature_enabled": False,
        "security_model": "anthropic/claude-haiku-4-5",
    }


def test_temperature_is_sent_only_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CONVERTER_TEMPERATURE", "0.6")
    cfg = env_runtime_config()
    assert cfg["converter_temperature"] == 0.6
    assert cfg["converter_temperature_enabled"] is True


def test_overrides_persist_and_legacy_keys_are_read(factory: Any) -> None:
    from pipeline.db_models import AppSetting
    from pipeline.document_store import _session_scope

    with _session_scope(factory) as s:  # what an earlier release stored
        s.add(AppSetting(key="llm.primary_max_tokens", value="5000", updated_at="t"))
        s.add(AppSetting(key="llm.primary_model", value="claude-ignored", updated_at="t"))
    assert get_runtime_config(factory)["converter_max_tokens"] == 5000

    set_runtime_config(factory, {"converter_max_tokens": 7000, "security_model": "@cheap"})
    effective = get_runtime_config(factory)
    assert effective["converter_max_tokens"] == 7000  # the new key wins
    assert effective["security_model"] == "@cheap"
    assert effective["converter_model"] == "anthropic/claude-opus-5-5"  # primary_model ignored


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("anthropic/claude-opus-5-5", "anthropic/claude-opus-5-5"),
        ("@fast", "@fast"),
        ("claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"),
    ],
)
def test_valid_model_names(name: str, expected: str) -> None:
    assert valid_model_name(name) == expected


@pytest.mark.parametrize("name", ["", "has space/x", "a/b c", "@Bad Pool", "no-slash-unknown"])
def test_invalid_model_names(name: str) -> None:
    with pytest.raises(ValueError):
        valid_model_name(name)


def test_converter_and_security_settings_fall_back_to_config() -> None:
    class _Cfg:
        llm_converter_model = "anthropic/x"
        llm_converter_max_tokens = 100
        content_security_llm_model = "anthropic/y"

    cfg: Any = _Cfg()
    expected = runtime_config.ConverterSettings("anthropic/x", 100, None)
    assert converter_settings(None, cfg) == expected
    assert security_model(None, cfg) == "anthropic/y"
    runtime = {"converter_model": "@p", "converter_max_tokens": 5,
               "converter_temperature": 0.1, "converter_temperature_enabled": True,
               "security_model": "@s"}
    assert converter_settings(runtime, cfg) == runtime_config.ConverterSettings("@p", 5, 0.1)
    assert security_model(runtime, cfg) == "@s"
