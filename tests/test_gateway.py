"""The pacod-gw client: request shape, streaming, failures and retries."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from pipeline.gateway import (
    MAX_PDF_BYTES,
    Attribution,
    Gateway,
    GatewayError,
    GatewayNotConfiguredError,
    GatewayOutputError,
    pdf_part,
    text_part,
)

KEY = "pgw_kb_testsecret"
URL = "http://gw.test:8080"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.slept.append(secs)
        self.now += secs


def _gateway(
    handler: Callable[[httpx.Request], httpx.Response], clock: _Clock | None = None
) -> tuple[Gateway, _Clock]:
    clock = clock or _Clock()
    gw = Gateway(
        URL, KEY, transport=httpx.MockTransport(handler), sleep=clock.sleep, clock=clock
    )
    return gw, clock


def _ok(text: str = "# Doc", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "model": "anthropic/claude-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": 2},
                  "cost": {"amount": 0.0042, "currency": "USD"}},
    }
    body.update(extra)
    return body


def _call(gw: Gateway, **kw: Any) -> Any:
    args: dict[str, Any] = {
        "model": "anthropic/claude-test",
        "system": "sys",
        "user": "hello",
        "max_tokens": 100,
        "attribution": Attribution(task="document_conversion", trace="job_1"),
    }
    args.update(kw)
    return gw.complete(**args)


def _sse(*chunks: dict[str, Any] | str) -> bytes:
    lines = []
    for c in chunks:
        lines.append(f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n")
    return "".join(lines).encode()


def _delta(text: str | None = None, finish: str | None = None, **delta: Any) -> dict[str, Any]:
    d = dict(delta)
    if text is not None:
        d["content"] = text
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_plain_call_sends_openai_shape_with_key_and_attribution() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=_ok(), headers={"x-gw-request-id": "req_abc"})

    gw, _ = _gateway(handler)
    done = _call(
        gw,
        temperature=0.3,
        attribution=Attribution(
            task="document_conversion", trace="job_1", end_user="proj-a", user_id="usr_42"
        ),
    )
    req = seen[0]
    assert req.url == f"{URL}/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert req.headers["x-gw-task"] == "document_conversion"
    assert req.headers["x-gw-trace"] == "job_1"
    assert req.headers["x-gw-end-user"] == "proj-a"
    assert req.headers["x-gw-user-id"] == "usr_42"
    body = json.loads(req.content)
    assert body["model"] == "anthropic/claude-test"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.3
    assert "stream" not in body
    assert done.text == "# Doc"
    assert done.request_id == "req_abc"
    assert done.usage.input_tokens == 12 and done.usage.output_tokens == 5
    assert done.usage.cached_tokens == 2
    assert (done.usage.cost, done.usage.currency) == (0.0042, "USD")


def test_temperature_omitted_unless_given_and_bad_header_values_dropped() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=_ok())

    gw, _ = _gateway(handler)
    _call(gw, attribution=Attribution(task="x" * 65, trace="bad trace\n", user_id="usr_1"))
    body = json.loads(seen[0].content)
    assert "temperature" not in body
    assert "x-gw-task" not in seen[0].headers  # over 64 characters
    assert "x-gw-trace" not in seen[0].headers  # not an identifier
    assert seen[0].headers["x-gw-user-id"] == "usr_1"


def test_cost_falls_back_to_header_and_unpriced_is_none() -> None:
    def priced_by_header(_req: httpx.Request) -> httpx.Response:
        body = _ok()
        del body["usage"]["cost"]
        return httpx.Response(200, json=body, headers={"x-gw-cost": "0.5"})

    gw, _ = _gateway(priced_by_header)
    assert _call(gw).usage.cost == 0.5

    def unpriced(_req: httpx.Request) -> httpx.Response:
        body = _ok()
        del body["usage"]["cost"]
        return httpx.Response(200, json=body)

    gw, _ = _gateway(unpriced)
    usage = _call(gw).usage
    assert usage.cost is None and usage.currency is None


def test_pdf_part_is_an_openai_file_part_and_is_capped() -> None:
    part = pdf_part(b"%PDF-1.7 tiny", "doc.pdf")
    assert part["type"] == "file"
    assert part["file"]["filename"] == "doc.pdf"
    prefix, data = part["file"]["file_data"].split(",", 1)
    assert prefix == "data:application/pdf;base64"
    assert base64.b64decode(data) == b"%PDF-1.7 tiny"
    assert text_part("hi") == {"type": "text", "text": "hi"}
    with pytest.raises(ValueError):
        pdf_part(b"x" * (MAX_PDF_BYTES + 1))


def test_not_configured_raises_without_calling() -> None:
    gw = Gateway("", "", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert not gw.configured()
    with pytest.raises(GatewayNotConfiguredError):
        _call(gw)
    assert not Gateway(URL, "").configured()


def test_origin_hides_path_and_credentials() -> None:
    gw = Gateway("https://user:pw@gw.example.com:8443/base/", KEY)
    assert gw.origin == "https://gw.example.com:8443"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_assembles_text_and_reads_final_usage() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        body = _sse(
            _delta("# Ti"),
            _delta("tle\n\nbody"),
            _delta(finish="stop"),
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                                      "cost": {"amount": 0.01, "currency": "EUR"}}},
            "[DONE]",
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream", "x-gw-request-id": "req_s",
                     "x-gw-model-served": "anthropic/claude-opus-5-5",
                     "x-gw-dropped-params": "temperature"},
        )

    gw, _ = _gateway(handler)
    done = _call(gw, stream=True, temperature=0.5)
    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert done.text == "# Title\n\nbody"
    assert done.finish_reason == "stop"
    assert done.model_served == "anthropic/claude-opus-5-5"
    assert done.request_id == "req_s"
    assert done.dropped_params == ("temperature",)
    assert (done.usage.input_tokens, done.usage.cost, done.usage.currency) == (7, 0.01, "EUR")


def test_stream_error_chunk_raises_with_code() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        body = _sse(_delta("partial"), {"error": {"code": "upstream_error", "message": "x"}})
        return httpx.Response(200, content=body, headers={"x-gw-request-id": "req_e"})

    gw, _ = _gateway(handler)
    with pytest.raises(GatewayError) as err:
        _call(gw, stream=True)
    assert err.value.code == "upstream_error"
    assert err.value.request_id == "req_e"


def test_stream_cut_without_done_or_finish_is_an_output_error() -> None:
    gw, _ = _gateway(lambda _r: httpx.Response(200, content=_sse(_delta("half an ans"))))
    with pytest.raises(GatewayOutputError) as err:
        _call(gw, stream=True)
    assert err.value.code == "stream_cut"


def test_stream_total_deadline() -> None:
    clock = _Clock()

    class _SlowStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            for _ in range(5):
                clock.now += 10  # ten seconds between chunks
                yield _sse(_delta("x"))

    gw, _ = _gateway(lambda _r: httpx.Response(200, stream=_SlowStream()), clock)
    with pytest.raises(GatewayError) as err:
        _call(gw, stream=True, total_timeout=25)
    assert err.value.code == "total_timeout"


def test_idle_timeout_is_reported_and_not_retried() -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("silent", request=req)

    gw, _ = _gateway(handler)
    with pytest.raises(GatewayError) as err:
        _call(gw, stream=True)
    assert err.value.code == "idle_timeout"
    assert calls == 1


# ---------------------------------------------------------------------------
# Unusable answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("choice", "code"),
    [
        ({"message": {"content": "half"}, "finish_reason": "length"}, "length"),
        ({"message": {"content": ""}, "finish_reason": "content_filter"}, "refused"),
        ({"message": {"content": None, "refusal": "I can't"}, "finish_reason": "stop"}, "refused"),
        ({"message": {"content": "   "}, "finish_reason": "stop"}, "empty"),
    ],
)
def test_cut_declined_or_empty_answers_raise(choice: dict[str, Any], code: str) -> None:
    gw, _ = _gateway(lambda _r: httpx.Response(200, json=_ok(choices=[choice])))
    with pytest.raises(GatewayOutputError) as err:
        _call(gw)
    assert err.value.code == code


def test_stream_refusal_delta_is_declined() -> None:
    body = _sse(_delta(refusal="no"), _delta(finish="stop"), "[DONE]")
    gw, _ = _gateway(lambda _r: httpx.Response(200, content=body))
    with pytest.raises(GatewayOutputError) as err:
        _call(gw, stream=True)
    assert err.value.code == "refused"


# ---------------------------------------------------------------------------
# Failures and retries
# ---------------------------------------------------------------------------


def test_429_is_retried_once_after_retry_after() -> None:
    answers = [
        httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"code": "rate"}}),
        httpx.Response(200, json=_ok("second time")),
    ]
    gw, clock = _gateway(lambda _r: answers.pop(0))
    assert _call(gw).text == "second time"
    assert clock.slept == [3.0]


def test_429_twice_gives_up_and_retry_after_is_capped() -> None:
    gw, clock = _gateway(lambda _r: httpx.Response(429, headers={"retry-after": "120"}))
    with pytest.raises(GatewayError) as err:
        _call(gw)
    assert err.value.status == 429
    assert clock.slept == [30.0]


@pytest.mark.parametrize("status", [402, 504])
def test_budget_and_timeout_are_never_retried(status: int) -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    gw, clock = _gateway(handler)
    with pytest.raises(GatewayError):
        _call(gw)
    assert calls == 1 and clock.slept == []


def test_error_names_status_code_and_request_never_the_body() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"code": "unsupported_feature", "message": "SECRET PROMPT TEXT"}},
            headers={"x-gw-request-id": "req_9"},
        )

    gw, _ = _gateway(handler)
    with pytest.raises(GatewayError) as err:
        _call(gw)
    msg = str(err.value)
    assert err.value.code == "unsupported_feature"
    assert err.value.request_id == "req_9"
    assert "unsupported_feature" in msg and "req_9" in msg
    assert "SECRET" not in msg


def test_error_code_from_header_when_body_is_not_json() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"<html>", headers={"x-gw-error-type": "bad_key"})

    gw, _ = _gateway(handler)
    with pytest.raises(GatewayError) as err:
        _call(gw)
    assert "refused the key" in str(err.value)
    assert err.value.code == "bad_key"


def test_unreachable_gateway() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    gw, _ = _gateway(handler)
    with pytest.raises(GatewayError) as err:
        _call(gw)
    assert err.value.code == "unreachable"


# ---------------------------------------------------------------------------
# Models and ping
# ---------------------------------------------------------------------------


def test_models_are_listed_and_cached() -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert req.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "openai/gpt-x"}, {"id": "@fast"}]})

    gw, clock = _gateway(handler)
    assert gw.models() == ["@fast", "openai/gpt-x"]
    assert gw.models() == ["@fast", "openai/gpt-x"]
    assert calls == 1
    clock.now += 61
    gw.models()
    assert calls == 2


def test_ping_asks_for_a_json_answer() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_ok('{"reply": "pong"}'))

    gw, _ = _gateway(handler)
    assert gw.ping("anthropic/claude-test").text == '{"reply": "pong"}'
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["max_tokens"] == 64
