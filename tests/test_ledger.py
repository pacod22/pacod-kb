"""The ledger keeps settings, prompts and usage in both storage backends."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pipeline import ledger, runtime_config
from pipeline.ledger import ledger_session_factory


@pytest.fixture(autouse=True)
def _fresh_caches() -> Iterator[None]:
    runtime_config._CACHE.clear()
    yield
    runtime_config._CACHE.clear()


@pytest.fixture()
def api_client(config: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from pipeline import api

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    from pipeline.gateway import Gateway

    api.app.state.ledger_factory = ledger_session_factory(config)
    api.app.state.gateway = Gateway("", "")  # not set up
    try:
        with TestClient(api.app, headers={"X-API-Key": "test-secret"}) as c:
            yield c
    finally:
        del api.app.state.ledger_factory
        del api.app.state.gateway


def test_filesystem_ledger_is_a_sqlite_file_in_the_index(config: Any) -> None:
    factory = ledger_session_factory(config)
    assert factory is ledger_session_factory(config)  # one per path
    assert (config.index_dir / "ledger.db").exists()


def test_runtime_settings_survive_a_restart_in_filesystem_mode(config: Any) -> None:
    runtime_config.set_runtime_config(
        ledger_session_factory(config), {"converter_max_tokens": 1234}, updated_by="t"
    )
    ledger._factories.clear()  # a new process opens the same file
    runtime_config._CACHE.clear()
    effective = runtime_config.get_runtime_config(ledger_session_factory(config))
    assert effective["converter_max_tokens"] == 1234


def test_saving_without_a_ledger_fails_loudly() -> None:
    from pipeline.prompt_config import set_prompt_text

    with pytest.raises(RuntimeError):
        runtime_config.set_runtime_config(None, {"converter_max_tokens": 1})
    with pytest.raises(RuntimeError):
        set_prompt_text(None, "converter_system", "x")


def test_prompt_and_config_edits_persist_through_the_api(api_client: TestClient) -> None:
    resp = api_client.patch("/llm-prompts/converter_system", json={"value": "Be terse."})
    assert resp.status_code == 200, resp.text
    resp = api_client.patch("/llm-config", json={"converter_max_tokens": 4321})
    assert resp.status_code == 200, resp.text

    prompts = {p["id"]: p for p in api_client.get("/llm-prompts").json()["prompts"]}
    assert prompts["converter_system"]["value"] == "Be terse."
    assert prompts["converter_system"]["overridden"] is True
    runtime_config._CACHE.clear()
    assert api_client.get("/llm-config").json()["effective"]["converter_max_tokens"] == 4321


def test_conversion_usage_is_recorded_in_filesystem_mode(
    config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy import select

    from pipeline.acquisition import ingest_document
    from pipeline.db_models import UsageEvent
    from pipeline.document_store import _session_scope
    from pipeline.parsers import llm

    def _fake_convert(result: Any, _config: Any, **_kw: Any) -> Any:
        return replace(
            result,
            extraction_method="vlm",
            converter="anthropic/claude-test",
            llm_usage={"input_tokens": 11, "output_tokens": 7, "cost": 0.02, "request_id": "r"},
        )

    class _Ready:
        def configured(self) -> bool:
            return True

    from pipeline import gateway

    monkeypatch.setattr(gateway, "get_gateway", lambda _config: _Ready())
    monkeypatch.setattr(llm, "convert_with_llm", _fake_convert)
    src = tmp_path / "note.md"
    src.write_text("# Note\n\nA short note about usage accounting.\n", encoding="utf-8")
    doc_id, folder = ingest_document(str(src), config, llm_convert=True)
    assert folder is not None

    with _session_scope(ledger_session_factory(config)) as s:
        rows = s.execute(select(UsageEvent)).scalars().all()
        recorded = [
            (r.turn_id, r.purpose, r.input_tokens, r.output_tokens, r.cost_usd, r.request_id)
            for r in rows
        ]
    assert recorded == [(doc_id, "document_conversion", 11, 7, 0.02, "r")]


def test_pipeline_api_fails_closed_without_a_secret(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline import api

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    api.app.state.config = replace(config)
    with TestClient(api.app) as c:
        assert c.get("/settings").status_code == 503
        assert c.get("/settings", headers={"X-API-Key": ""}).status_code == 503
        assert c.get("/health").status_code == 200
