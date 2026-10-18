"""Tests for pipeline/parsers/llm.py — document conversion through pacod-gw."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pipeline.gateway import (
    MAX_PDF_BYTES,
    Attribution,
    Completion,
    GatewayError,
    GatewayOutputError,
    Usage,
)
from pipeline.parsers import ParseResult
from pipeline.parsers.llm import _MAX_INPUT_CHARS, _truncate_body, convert_with_llm

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class FakeGateway:
    """Records calls; answers from a queue of Completions or exceptions."""

    def __init__(self, *answers: Completion | Exception, configured: bool = True) -> None:
        self.answers = list(answers) or [_completion()]
        self.calls: list[dict[str, Any]] = []
        self._configured = configured

    def configured(self) -> bool:
        return self._configured

    def complete(self, **kwargs: Any) -> Completion:
        self.calls.append(kwargs)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _completion(text: str = "# Clean\n\nMarkdown body.", **kw: Any) -> Completion:
    base: dict[str, Any] = {
        "text": text,
        "finish_reason": "stop",
        "model_served": "anthropic/claude-opus-5-5",
        "request_id": "req_1",
        "usage": Usage(10, 20, 3, cost=0.0123, currency="USD"),
        "dropped_params": (),
        "duration_ms": 42,
    }
    base.update(kw)
    return Completion(**base)


def _make_config(**overrides: Any) -> Any:
    """Return a Config-like namespace for tests (avoids needing KB_ROOT)."""
    defaults: dict[str, Any] = {
        "llm_converter_enabled": True,
        "llm_converter_model": "anthropic/claude-opus-5-5",
        "llm_converter_max_tokens": 4096,
        "llm_converter_timeout_secs": 1800.0,
    }
    defaults.update(overrides)

    class _Cfg:
        pass

    cfg = _Cfg()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_result(body: str = "Raw document text.", title: str = "Doc") -> ParseResult:
    return ParseResult(body=body, title=title, language="en")


def _convert(gw: FakeGateway, result: ParseResult | None = None, **kw: Any) -> ParseResult:
    return convert_with_llm(result or _make_result(), _make_config(), gateway=gw, **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _truncate_body
# ---------------------------------------------------------------------------


class TestTruncateBody:
    def test_short_body_unchanged(self) -> None:
        assert _truncate_body("hello") == "hello"

    def test_long_body_truncated(self) -> None:
        body = "x" * (_MAX_INPUT_CHARS + 10)
        assert len(_truncate_body(body)) == _MAX_INPUT_CHARS

    def test_exact_limit_unchanged(self) -> None:
        body = "x" * _MAX_INPUT_CHARS
        assert _truncate_body(body) == body


# ---------------------------------------------------------------------------
# convert_with_llm
# ---------------------------------------------------------------------------


class TestConvertWithLlm:
    def test_empty_body_returned_unchanged(self) -> None:
        gw = FakeGateway()
        result = _make_result(body="   ")
        assert _convert(gw, result) is result
        assert gw.calls == []

    def test_text_document_streams_through_the_gateway(self) -> None:
        gw = FakeGateway()
        out = _convert(gw, attribution=Attribution(task="document_conversion", trace="job_7"))
        call = gw.calls[0]
        assert call["model"] == "anthropic/claude-opus-5-5"
        assert call["stream"] is True
        assert call["max_tokens"] == 4096
        assert call["temperature"] is None  # the model's own sampling by default
        assert call["total_timeout"] == 1800.0
        assert call["attribution"].trace == "job_7"
        assert isinstance(call["user"], str) and "Raw document text." in call["user"]
        assert "Markdown" in call["system"]
        assert out.body == "# Clean\n\nMarkdown body."
        assert out.extraction_method == "vlm"
        assert out.converter == "anthropic/claude-opus-5-5"

    def test_usage_carries_gateway_cost_and_ids(self) -> None:
        out = _convert(FakeGateway())
        assert out.llm_usage == {
            "input_tokens": 10,
            "output_tokens": 20,
            "cached_tokens": 3,
            "cost": 0.0123,
            "currency": "USD",
            "request_id": "req_1",
            "model_served": "anthropic/claude-opus-5-5",
        }

    def test_converter_is_the_requested_model_when_none_served(self) -> None:
        gw = FakeGateway(_completion(model_served=None))
        assert _convert(gw).converter == "anthropic/claude-opus-5-5"

    def test_pdf_goes_as_a_file_part(self) -> None:
        gw = FakeGateway()
        _convert(gw, source_bytes=b"%PDF-1.7 data", source_ext=".pdf")
        parts = gw.calls[0]["user"]
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_data"].startswith("data:application/pdf;base64,")
        assert parts[1]["type"] == "text"

    def test_oversized_pdf_falls_back_to_text(self) -> None:
        gw = FakeGateway()
        _convert(gw, source_bytes=b"x" * (MAX_PDF_BYTES + 1), source_ext="pdf")
        assert isinstance(gw.calls[0]["user"], str)

    def test_non_pdf_bytes_are_not_sent(self) -> None:
        gw = FakeGateway()
        _convert(gw, source_bytes=b"<html>", source_ext=".html")
        assert isinstance(gw.calls[0]["user"], str)

    def test_model_without_documents_gets_the_text(self) -> None:
        gw = FakeGateway(
            GatewayError("no documents", status=400, code="unsupported_feature"), _completion()
        )
        out = _convert(gw, source_bytes=b"%PDF-1.7", source_ext=".pdf")
        assert len(gw.calls) == 2
        assert isinstance(gw.calls[0]["user"], list)
        assert isinstance(gw.calls[1]["user"], str)
        assert out.body.startswith("# Clean")

    def test_other_pdf_failures_are_not_retried_as_text(self) -> None:
        gw = FakeGateway(GatewayError("busy", status=503))
        with pytest.raises(GatewayError):
            _convert(gw, source_bytes=b"%PDF-1.7", source_ext=".pdf")
        assert len(gw.calls) == 1

    def test_unusable_answer_raises(self) -> None:
        gw = FakeGateway(GatewayOutputError("cut", code="length"))
        with pytest.raises(GatewayOutputError):
            _convert(gw)

    def test_runtime_config_sets_model_tokens_and_temperature(self) -> None:
        gw = FakeGateway()
        runtime = {
            "converter_model": "@fast",
            "converter_max_tokens": 9000,
            "converter_temperature": 0.4,
            "converter_temperature_enabled": True,
        }
        _convert(gw, runtime_config=runtime)
        call = gw.calls[0]
        assert (call["model"], call["max_tokens"], call["temperature"]) == ("@fast", 9000, 0.4)

    def test_temperature_off_in_runtime_config_sends_none(self) -> None:
        gw = FakeGateway()
        _convert(gw, runtime_config={"converter_temperature": 0.9})
        assert gw.calls[0]["temperature"] is None

    def test_uses_prompt_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pipeline import prompt_config

        prompts = {
            "converter_system": "SYSTEM OVERRIDE",
            "converter_user_text": "TEXT: {document_text}",
            "converter_user_pdf": "PDF OVERRIDE",
        }
        monkeypatch.setattr(prompt_config, "get_prompt_text", lambda _sf, pid: prompts[pid])
        gw = FakeGateway()
        _convert(gw)
        assert gw.calls[0]["system"] == "SYSTEM OVERRIDE"
        assert gw.calls[0]["user"] == "TEXT: Raw document text."
        _convert(gw, source_bytes=b"%PDF", source_ext=".pdf")
        assert gw.calls[1]["user"][1]["text"] == "PDF OVERRIDE"

    def test_original_metadata_preserved(self) -> None:
        out = _convert(FakeGateway(), ParseResult(body="Body", title="My Title", language="de"))
        assert (out.title, out.language) == ("My Title", "de")

    def test_implausibly_short_output_is_logged_not_rejected(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        gw = FakeGateway(_completion("tiny"))
        out = _convert(gw, _make_result(body="word " * 1000))
        assert out.body == "tiny"
        assert "check request req_1" in caplog.text


# ---------------------------------------------------------------------------
# Interaction log
# ---------------------------------------------------------------------------


def _ledger(tmp_path: Path) -> Any:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from pipeline.db_models import Base

    engine = create_engine(f"sqlite:///{tmp_path}/ledger.db")
    Base.metadata.create_all(engine)
    return sessionmaker(engine)


def _interactions(factory: Any) -> list[Any]:
    from sqlalchemy import select

    from pipeline.db_models import AiInteraction
    from pipeline.document_store import _session_scope

    with _session_scope(factory) as s:
        rows = s.execute(select(AiInteraction)).scalars().all()
        s.expunge_all()
        return list(rows)


def test_success_is_logged_with_request_id_and_cost(tmp_path: Path) -> None:
    factory = _ledger(tmp_path)
    gw = FakeGateway(_completion(dropped_params=("temperature",)))
    _convert(gw, session_factory=factory, attribution=Attribution("document_conversion", "usr_1"))
    (row,) = _interactions(factory)
    assert row.status == "ok"
    assert row.request_id == "req_1"
    assert row.model_served == "anthropic/claude-opus-5-5"
    assert (row.cost_usd, row.currency, row.cost_source) == (0.0123, "USD", "gateway")
    assert row.user_id == "usr_1"
    assert '"dropped_params": ["temperature"]' in (row.meta_json or "")


def test_failure_is_logged_with_the_gateway_request_id(tmp_path: Path) -> None:
    factory = _ledger(tmp_path)
    gw = FakeGateway(GatewayOutputError("cut off", code="length", request_id="req_x"))
    with pytest.raises(GatewayOutputError):
        _convert(gw, session_factory=factory)
    (row,) = _interactions(factory)
    assert (row.status, row.request_id, row.finish_reason) == ("error", "req_x", "length")
    assert row.cost_source == "unpriced"


# ---------------------------------------------------------------------------
# ingest_document keeps the deterministic parse
# ---------------------------------------------------------------------------


def _ingest_with(
    gw: FakeGateway, config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    from pipeline import gateway
    from pipeline.acquisition import ingest_document

    monkeypatch.setattr(gateway, "get_gateway", lambda _config: gw)
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nThe original deterministic body.\n", encoding="utf-8")
    _doc_id, folder = ingest_document(str(src), config, llm_convert=True)
    assert folder is not None
    return (folder / "document.md").read_text(encoding="utf-8")


def test_ingest_uses_the_conversion(config: Any, tmp_path: Path, monkeypatch: Any) -> None:
    gw = FakeGateway(_completion("# Title\n\nConverted body."))
    doc = _ingest_with(gw, replace(config, content_security_enabled=False), tmp_path, monkeypatch)
    assert "Converted body." in doc
    assert gw.calls[0]["attribution"].task == "document_conversion"


def test_ingest_keeps_the_parse_when_conversion_fails(
    config: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    gw = FakeGateway(GatewayOutputError("cut off", code="length"))
    doc = _ingest_with(gw, replace(config, content_security_enabled=False), tmp_path, monkeypatch)
    assert "The original deterministic body." in doc


def test_ingest_skips_conversion_without_a_gateway(
    config: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    gw = FakeGateway(configured=False)
    doc = _ingest_with(gw, replace(config, content_security_enabled=False), tmp_path, monkeypatch)
    assert "The original deterministic body." in doc
    assert gw.calls == []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfigFields:
    def _load(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, **env: str) -> Any:
        monkeypatch.setenv("KB_ROOT", str(tmp_path))
        monkeypatch.setenv("QDRANT_MODE", "embedded")
        for key in (
            "LLM_CONVERTER_ENABLED",
            "LLM_CONVERTER_PROVIDER",
            "LLM_CONVERTER_MODEL",
            "LLM_CONVERTER_MAX_TOKENS",
            "CONTENT_SECURITY_LLM_MODEL",
            "PACOD_GW_URL",
            "PACOD_GW_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from pipeline.config import load_config

        return load_config()

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        config = self._load(monkeypatch, tmp_path)
        assert config.llm_converter_enabled is True
        assert config.llm_converter_model == "anthropic/claude-opus-5-5"
        assert config.content_security_llm_model == "anthropic/claude-haiku-4-5"
        assert config.llm_converter_max_tokens == 64000
        assert (config.gateway_url, config.gateway_api_key) == ("", "")

    def test_gateway_and_model_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        config = self._load(
            monkeypatch,
            tmp_path,
            PACOD_GW_URL="http://gw:8080/",
            PACOD_GW_API_KEY="pgw_kb_x",
            LLM_CONVERTER_MODEL="@fast",
            LLM_CONVERTER_MAX_TOKENS="16384",
        )
        assert (config.gateway_url, config.gateway_api_key) == ("http://gw:8080", "pgw_kb_x")
        assert config.llm_converter_model == "@fast"
        assert config.llm_converter_max_tokens == 16384

    @pytest.mark.parametrize(
        ("model", "provider", "expected"),
        [
            ("claude-opus-5-5", "", "anthropic/claude-opus-5-5"),
            ("gemini-3.1-pro", "gemin", "gemini/gemini-3.1-pro"),
            ("gpt-5.5", "", "openai/gpt-5.5"),
            ("glm-5", "", "glm/glm-5"),
            ("my-model", "deepseek", "deepseek/my-model"),
            ("openai/gpt-5.5", "anthropic", "openai/gpt-5.5"),
        ],
    )
    def test_bare_legacy_model_names_are_qualified(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        model: str,
        provider: str,
        expected: str,
    ) -> None:
        env = {"LLM_CONVERTER_MODEL": model}
        if provider:
            env["LLM_CONVERTER_PROVIDER"] = provider
        assert self._load(monkeypatch, tmp_path, **env).llm_converter_model == expected

    def test_legacy_variables_are_warned_about(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from pipeline.config import warn_legacy_env

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-old")
        monkeypatch.delenv("SECRETS_KEY", raising=False)
        warn_legacy_env()
        assert "ANTHROPIC_API_KEY is set but ignored" in caplog.text
        assert "SECRETS_KEY" not in caplog.text
