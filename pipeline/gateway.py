"""pacod-gw: the one way pacod-kb calls a language model.

Every LLM call — document conversion, the content-security second opinion, the AI
Management connection test — goes through :class:`Gateway`, which speaks the OpenAI
chat-completions shape to pacod-gw: ``POST {PACOD_GW_URL}/v1/chat/completions`` with a
``pgw_`` key. The gateway picks the provider from the model name
(``anthropic/claude-opus-5-5``, a pool ``@name``), translates a PDF file part for the
models that read documents, fails over, and prices each call; kb keeps the cost and the
gateway's request id with its usage rows.

Rules (shared with pacod-cms's AiGateway):

* attribution headers carry opaque ids only: ``x-gw-task`` (the feature),
  ``x-gw-trace`` (the job or document), ``x-gw-end-user`` (the project),
  ``x-gw-user-id`` (the person's ``usr_`` id);
* one retry, and only for a 429, after its ``Retry-After`` (at most 30 s) — never for
  a 402 (the budget is spent) or a 504 (the model may still be generating, and billing);
  the gateway already retries and fails over;
* an answer cut short (``length``), declined (``content_filter`` or a refusal) or empty
  is a failure: :class:`GatewayOutputError`;
* an error names the status, the gateway's error code and request id — never the body.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from pipeline.config import Config

logger = logging.getLogger(__name__)

# Raw PDF bytes sent inline; base64 makes 20 MiB about 26.7 MiB, under the gateway's
# default 32 MiB body limit (PGW_MAX_BODY_BYTES). Larger PDFs go as extracted text.
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_RETRY_AFTER_SECS = 30.0
_MODELS_TTL_SECS = 60.0

# What may be sent in an attribution header or quoted from a gateway answer.
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,200}")
_TASK_MAX = 64

_REASONS: dict[int, str] = {
    400: "The model cannot take this input",
    401: "The AI gateway refused the key",
    402: "The AI budget is used up",
    # The key is good, the call is not: a model the key may not use, or a gateway rule.
    403: "The AI gateway refused the call",
    404: "The AI gateway does not know this model",
    413: "The request is too large for the AI gateway",
    429: "The AI gateway is busy; try again later",
    503: "The AI gateway is unavailable; try again later",
    504: "The model did not answer in time",
}


def _token(value: str | None) -> str | None:
    if value and _TOKEN.fullmatch(value):
        return value
    return None


@dataclass(frozen=True)
class Attribution:
    """Who and what a call is for, as the gateway records it."""

    task: str
    user_id: str | None = None
    trace: str | None = None
    end_user: str | None = None

    def headers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if len(self.task) <= _TASK_MAX and _token(self.task):
            out["x-gw-task"] = self.task
        for name, value in (
            ("x-gw-trace", self.trace),
            ("x-gw-end-user", self.end_user),
            ("x-gw-user-id", self.user_id),
        ):
            if _token(value):
                out[name] = str(value)
        return out


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float | None = None  # None: the gateway did not price the call
    currency: str | None = None


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str
    model_served: str | None
    request_id: str | None
    usage: Usage
    dropped_params: tuple[str, ...] = ()
    duration_ms: int = 0


class GatewayError(Exception):
    """A call that produced nothing usable; the message is safe for logs and the UI."""

    def __init__(
        self, message: str, *, status: int = 0, code: str = "", request_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id


class GatewayNotConfiguredError(GatewayError):
    """No gateway address or key: AI features are off."""


class GatewayOutputError(GatewayError):
    """The model answered, but the answer is unusable: cut off, declined, empty, or cut."""


class _BusyError(Exception):
    def __init__(self, error: GatewayError, retry_after: float) -> None:
        self.error = error
        self.retry_after = retry_after


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def pdf_part(data: bytes, filename: str = "document.pdf") -> dict[str, Any]:
    """A PDF as an OpenAI ``file`` part; the gateway translates it per provider."""
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"PDF is {len(data)} bytes, over the {MAX_PDF_BYTES}-byte inline limit")
    encoded = base64.standard_b64encode(data).decode("ascii")
    return {
        "type": "file",
        "file": {"filename": filename, "file_data": f"data:application/pdf;base64,{encoded}"},
    }


def _usage_from(raw: Any, header_cost: str | None = None) -> Usage:
    if not isinstance(raw, dict):
        raw = {}
    details = raw.get("prompt_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    cost: float | None = None
    currency: str | None = None
    priced = raw.get("cost")
    amount = priced.get("amount") if isinstance(priced, dict) else priced
    if isinstance(amount, int | float) and not isinstance(amount, bool):
        cost = float(amount)
        cur = priced.get("currency") if isinstance(priced, dict) else None
        currency = str(cur) if cur else "USD"
    elif header_cost:
        try:
            cost, currency = float(header_cost.split()[0]), "USD"
        except (ValueError, IndexError):
            pass
    return Usage(
        input_tokens=int(raw.get("prompt_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cached_tokens=int(cached or 0),
        cost=cost,
        currency=currency,
    )


def _status_error(status: int, headers: httpx.Headers, body: bytes) -> GatewayError:
    """A refusal in words, with what helps find it in the gateway's log — never the body."""
    reason = _REASONS.get(status, f"The AI gateway answered {status}")
    code = ""
    try:
        err = json.loads(body).get("error")
        if isinstance(err, dict):
            code = str(err.get("code") or err.get("type") or "")
    except (ValueError, AttributeError):
        pass  # not JSON: the header below, or the status alone
    code = _token(code) or _token(headers.get("x-gw-error-type")) or ""
    request_id = _token(headers.get("x-gw-request-id"))
    details = [d for d in (code, f"request {request_id}" if request_id else "") if d]
    message = f"{reason} ({', '.join(details)})" if details else reason
    return GatewayError(message, status=status, code=code, request_id=request_id)


def _retry_after(headers: httpx.Headers) -> float:
    try:
        return min(float(headers.get("retry-after", "2").strip()), MAX_RETRY_AFTER_SECS)
    except ValueError:
        return 2.0


class Gateway:
    """pacod-gw's chat completions for kb (thread-safe; one per process)."""

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        connect_timeout: float = 10.0,
        idle_timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = (url or "").strip().rstrip("/")
        self._key = (api_key or "").strip()
        self._connect_timeout = connect_timeout
        self._idle_timeout = idle_timeout
        self._sleep = sleep
        self._clock = clock
        self._client = httpx.Client(transport=transport, follow_redirects=False)
        self._models: tuple[float, list[str]] | None = None

    @classmethod
    def from_config(cls, config: Config) -> Gateway:
        return cls(
            config.gateway_url,
            config.gateway_api_key,
            idle_timeout=config.gateway_idle_timeout_secs,
        )

    def configured(self) -> bool:
        """Whether there is a gateway to call: an address and a key."""
        return bool(self.url and self._key)

    @property
    def origin(self) -> str:
        """``scheme://host:port`` of the gateway, for display (no path, no credentials)."""
        if not self.url:
            return ""
        parts = urllib.parse.urlsplit(self.url)
        host = parts.hostname or ""
        return f"{parts.scheme}://{host}{f':{parts.port}' if parts.port else ''}"

    def close(self) -> None:
        self._client.close()

    # ---- calls ------------------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str | list[dict[str, Any]],
        max_tokens: int,
        attribution: Attribution,
        temperature: float | None = None,
        stream: bool = False,
        total_timeout: float = 600.0,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        """One completion; raises :class:`GatewayError` when nothing usable came back."""
        if not self.configured():
            raise GatewayNotConfiguredError(
                "AI is not set up: PACOD_GW_URL and PACOD_GW_API_KEY are blank",
                code="not_configured",
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if response_format is not None:
            body["response_format"] = response_format
        headers = {"authorization": f"Bearer {self._key}", **attribution.headers()}
        started = self._clock()
        deadline = started + total_timeout

        completion: Completion | None = None
        for attempt in (1, 2):
            try:
                if stream:
                    completion = self._stream(body, headers, deadline, total_timeout)
                else:
                    completion = self._plain(body, headers, deadline, total_timeout)
                break
            except _BusyError as busy:
                if attempt == 2 or self._clock() + busy.retry_after > deadline:
                    raise busy.error from None
                logger.info("AI gateway busy; retrying in %.0f s", busy.retry_after)
                self._sleep(busy.retry_after)
        assert completion is not None
        done = Completion(
            text=completion.text,
            finish_reason=completion.finish_reason,
            model_served=completion.model_served,
            request_id=completion.request_id,
            usage=completion.usage,
            dropped_params=completion.dropped_params,
            duration_ms=int((self._clock() - started) * 1000),
        )
        if done.dropped_params:
            logger.info(
                "AI gateway dropped parameters %s for %s", ",".join(done.dropped_params), model
            )
        _check_output(done)
        return done

    def models(self) -> list[str]:
        """The model names this key may use (``GET /v1/models``), cached for a minute."""
        now = self._clock()
        if self._models is not None and now - self._models[0] < _MODELS_TTL_SECS:
            return list(self._models[1])
        if not self.configured():
            raise GatewayNotConfiguredError(
                "AI is not set up: PACOD_GW_URL and PACOD_GW_API_KEY are blank",
                code="not_configured",
            )
        try:
            resp = self._client.get(
                f"{self.url}/v1/models",
                headers={"authorization": f"Bearer {self._key}"},
                timeout=self._timeout(15.0),
            )
        except httpx.TransportError as exc:
            raise GatewayError(
                f"The AI gateway cannot be reached ({type(exc).__name__})", code="unreachable"
            ) from None
        if resp.status_code != 200:
            raise _status_error(resp.status_code, resp.headers, resp.content)
        try:
            data = resp.json().get("data") or []
            ids = sorted({str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")})
        except (ValueError, AttributeError, KeyError):
            raise GatewayError("The AI gateway sent an unreadable model list") from None
        self._models = (now, ids)
        return list(ids)

    def ping(self, model: str) -> Completion:
        """The smallest real call there is: address, key and model, end to end."""
        return self.complete(
            model=model,
            system='Answer with a JSON object whose "reply" is the word pong.',
            user="ping",
            max_tokens=64,
            attribution=Attribution(task="ping"),
            total_timeout=30.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ping",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"reply": {"type": "string"}},
                        "required": ["reply"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    # ---- transport --------------------------------------------------------------

    def _timeout(self, total: float) -> httpx.Timeout:
        # The read timeout is the idle timeout: the longest silence between bytes.
        return httpx.Timeout(
            connect=self._connect_timeout,
            read=min(self._idle_timeout, total),
            write=60.0,
            pool=10.0,
        )

    def _plain(
        self, body: dict[str, Any], headers: dict[str, str], deadline: float, total: float
    ) -> Completion:
        try:
            resp = self._client.post(
                f"{self.url}/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=self._timeout(total),
            )
        except httpx.TimeoutException:
            raise GatewayError(
                f"The AI gateway sent nothing for {min(self._idle_timeout, total):.0f} s",
                code="timeout",
            ) from None
        except httpx.TransportError as exc:
            raise GatewayError(
                f"The AI gateway cannot be reached ({type(exc).__name__})", code="unreachable"
            ) from None
        if resp.status_code != 200:
            raise self._failure(resp)
        request_id = _token(resp.headers.get("x-gw-request-id"))
        try:
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
        except (ValueError, AttributeError, IndexError):
            raise GatewayOutputError(
                "The AI gateway sent an unreadable answer", code="unreadable", request_id=request_id
            ) from None
        content = message.get("content")
        if isinstance(content, list):  # content parts
            content = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        refusal = message.get("refusal")
        finish = str(choice.get("finish_reason") or "")
        if refusal:
            finish = "refusal"
        return Completion(
            text=str(content or ""),
            finish_reason=finish,
            model_served=resp.headers.get("x-gw-model-served") or data.get("model"),
            request_id=request_id,
            usage=_usage_from(data.get("usage"), resp.headers.get("x-gw-cost")),
            dropped_params=_dropped(resp.headers),
        )

    def _stream(
        self, body: dict[str, Any], headers: dict[str, str], deadline: float, total: float
    ) -> Completion:
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        parts: list[str] = []
        refused = False
        finish = ""
        usage_raw: Any = None
        ended = False
        request_id: str | None = None
        try:
            with self._client.stream(
                "POST",
                f"{self.url}/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=self._timeout(total),
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise self._failure(resp)
                request_id = _token(resp.headers.get("x-gw-request-id"))
                model_served = resp.headers.get("x-gw-model-served")
                dropped = _dropped(resp.headers)
                for line in resp.iter_lines():
                    if self._clock() > deadline:
                        raise GatewayError(
                            f"The model did not finish within {total:.0f} s",
                            code="total_timeout",
                            request_id=request_id,
                        )
                    if not line.startswith("data:"):
                        continue  # blank lines, comments, event names
                    data = line[5:].strip()
                    if data == "[DONE]":
                        ended = True
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error"):
                        err = chunk["error"] if isinstance(chunk["error"], dict) else {}
                        code = _token(str(err.get("code") or err.get("type") or "")) or ""
                        raise GatewayError(
                            "The AI gateway reported an error mid-answer"
                            + (f" ({code})" if code else ""),
                            code=code,
                            request_id=request_id,
                        )
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            parts.append(str(delta["content"]))
                        if delta.get("refusal"):
                            refused = True
                        if choice.get("finish_reason"):
                            finish = str(choice["finish_reason"])
                    if chunk.get("usage"):
                        usage_raw = chunk["usage"]
                    if not model_served and chunk.get("model"):
                        model_served = str(chunk["model"])
        except httpx.ReadTimeout:
            raise GatewayError(
                f"The AI gateway sent nothing for {min(self._idle_timeout, total):.0f} s",
                code="idle_timeout",
                request_id=request_id,
            ) from None
        except httpx.TimeoutException:
            raise GatewayError(
                "The AI gateway did not answer in time", code="timeout", request_id=request_id
            ) from None
        except httpx.TransportError as exc:
            raise GatewayError(
                f"The AI gateway cannot be reached ({type(exc).__name__})",
                code="unreachable",
                request_id=request_id,
            ) from None
        if not ended and not finish:
            raise GatewayOutputError(
                "The answer stream ended early", code="stream_cut", request_id=request_id
            )
        return Completion(
            text="".join(parts),
            finish_reason="refusal" if refused else finish,
            model_served=model_served,
            request_id=request_id,
            usage=_usage_from(usage_raw),
            dropped_params=dropped,
        )

    def _failure(self, resp: httpx.Response) -> Exception:
        error = _status_error(resp.status_code, resp.headers, resp.content)
        if resp.status_code == 429:
            return _BusyError(error, _retry_after(resp.headers))
        return error


def _dropped(headers: httpx.Headers) -> tuple[str, ...]:
    raw = headers.get("x-gw-dropped-params", "")
    return tuple(p.strip() for p in raw.split(",") if _token(p.strip()))


def _check_output(c: Completion) -> None:
    if c.finish_reason == "length":
        raise GatewayOutputError(
            "The answer was cut off (max_tokens reached)", code="length", request_id=c.request_id
        )
    if c.finish_reason in ("content_filter", "refusal"):
        raise GatewayOutputError("The model declined", code="refused", request_id=c.request_id)
    if not c.text.strip():
        raise GatewayOutputError(
            "The model returned no text", code="empty", request_id=c.request_id
        )


_lock = threading.Lock()
_gateways: dict[tuple[str, str, float], Gateway] = {}


def get_gateway(config: Config) -> Gateway:
    """The process-wide gateway client for this configuration."""
    key = (config.gateway_url, config.gateway_api_key, config.gateway_idle_timeout_secs)
    with _lock:
        gateway = _gateways.get(key)
        if gateway is None:
            gateway = Gateway.from_config(config)
            _gateways[key] = gateway
        return gateway
