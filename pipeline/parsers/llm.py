"""LLM-based document conversion, through pacod-gw.

``convert_with_llm`` sends the parsed document — or, for a PDF of at most
``gateway.MAX_PDF_BYTES``, the PDF itself as a file part — to the converter model
through the AI gateway (``pipeline/gateway.py``) and returns the Markdown it writes.
The gateway picks the provider from the model name (``anthropic/claude-opus-5-5``, a
pool ``@name``) and translates the PDF for the models that read documents; for one
that does not (``unsupported_feature``) the extracted text is sent instead.

A cut-off, refused or empty answer raises; ``ingest_document`` then keeps the
deterministic parse. Settings: LLM_CONVERTER_MODEL, LLM_CONVERTER_MAX_TOKENS,
LLM_CONVERTER_TEMPERATURE, LLM_CONVERTER_TIMEOUT_SECS, and their AI Management
overrides (``pipeline/runtime_config.py``).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.gateway import Attribution, Completion, Gateway, GatewayError
    from pipeline.parsers import ParseResult

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS: int = 200_000  # ~50 K tokens — truncate absurdly large text inputs

# A conversion much shorter than a long deterministic parse is logged (never rejected:
# conversion runs before the noise filter, and a real cleanup can shrink a page a lot).
_PLAUSIBLE_MIN_BODY = 2_000
_PLAUSIBLE_RATIO = 0.25


def convert_with_llm(
    result: ParseResult,
    config: Config,
    *,
    source_bytes: bytes | None = None,
    source_ext: str = "",
    runtime_config: dict[str, Any] | None = None,
    session_factory: Any | None = None,
    attribution: Attribution | None = None,
    gateway: Gateway | None = None,
) -> ParseResult:
    """Return a new ParseResult whose body was rewritten as Markdown by the converter model.

    Raises :class:`pipeline.gateway.GatewayError` when the gateway is not set up, the
    call fails, or the answer is cut off, refused or empty.
    """
    if not result.body.strip():
        return result

    from pipeline.gateway import (
        MAX_PDF_BYTES,
        Attribution,
        GatewayError,
        get_gateway,
        pdf_part,
        text_part,
    )
    from pipeline.prompt_config import get_prompt_text
    from pipeline.runtime_config import converter_settings

    gw = gateway or get_gateway(config)
    settings = converter_settings(runtime_config, config)
    attr = attribution or Attribution(task="document_conversion")
    system_prompt = get_prompt_text(session_factory, "converter_system")
    text_message = _render_user_prompt(
        get_prompt_text(session_factory, "converter_user_text"), _truncate_body(result.body)
    )
    pdf: bytes | None = None
    if source_ext.lstrip(".").lower() == "pdf" and source_bytes is not None:
        pdf = source_bytes
        if len(pdf) > MAX_PDF_BYTES:
            logger.warning(
                "PDF is %d bytes (> %d inline limit) — converting the extracted text instead",
                len(pdf),
                MAX_PDF_BYTES,
            )
            pdf = None

    def _call(user: str | list[dict[str, Any]]) -> Completion:
        return gw.complete(
            model=settings.model,
            system=system_prompt,
            user=user,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            attribution=attr,
            stream=True,
            total_timeout=config.llm_converter_timeout_secs,
        )

    meta: dict[str, Any] = {}
    request_log = text_message
    try:
        if pdf is not None:
            request_log = f"[PDF document: {len(pdf)} bytes]"
            try:
                user_pdf = get_prompt_text(session_factory, "converter_user_pdf")
                completion = _call([pdf_part(pdf), text_part(user_pdf)])
            except GatewayError as exc:
                if exc.code != "unsupported_feature":
                    raise
                logger.info(
                    "Model %s takes no documents — converting the extracted text", settings.model
                )
                meta["pdf_fallback"] = "unsupported_feature"
                request_log = text_message
                completion = _call(text_message)
        else:
            completion = _call(text_message)
    except Exception as exc:
        _log_interaction(
            session_factory,
            settings.model,
            attr,
            status="error",
            system_prompt=system_prompt,
            request_user=request_log,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            error=exc,
            meta=meta,
        )
        raise

    if completion.dropped_params:
        meta["dropped_params"] = list(completion.dropped_params)
    if (
        len(result.body) >= _PLAUSIBLE_MIN_BODY
        and len(completion.text) < _PLAUSIBLE_RATIO * len(result.body)
    ):
        logger.warning(
            "LLM conversion is %d chars for a %d-char document — check request %s",
            len(completion.text),
            len(result.body),
            completion.request_id or "?",
        )
        meta["implausibly_short"] = True
    _log_interaction(
        session_factory,
        settings.model,
        attr,
        status="ok",
        system_prompt=system_prompt,
        request_user=request_log,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        completion=completion,
        meta=meta,
    )

    usage = completion.usage
    llm_usage: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_tokens": usage.cached_tokens,
        "cost": usage.cost,
        "currency": usage.currency,
        "request_id": completion.request_id,
        "model_served": completion.model_served,
    }
    return dataclasses.replace(
        result,
        body=completion.text,
        extraction_method="vlm",
        converter=completion.model_served or settings.model,
        llm_usage=llm_usage,
    )


def _log_interaction(
    session_factory: Any | None,
    model: str,
    attribution: Attribution,
    *,
    status: str,
    system_prompt: str,
    request_user: str,
    temperature: float | None,
    max_tokens: int,
    completion: Completion | None = None,
    error: Exception | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Best-effort detailed interaction log for a document-conversion call."""
    if session_factory is None:
        return
    from pipeline.gateway import GatewayError
    from pipeline.llm_management import log_ai_interaction

    gw_error: GatewayError | None = error if isinstance(error, GatewayError) else None
    usage = completion.usage if completion is not None else None
    log_ai_interaction(
        session_factory=session_factory,
        provider=model,
        model=(completion.model_served if completion else None) or model,
        purpose="document_conversion",
        status=status,
        user_id=attribution.user_id,
        turn_id=attribution.trace,
        request_system=system_prompt,
        request_user=request_user,
        response_text=completion.text if completion else None,
        temperature=temperature,
        max_tokens=max_tokens,
        finish_reason=(
            completion.finish_reason if completion else (gw_error.code if gw_error else None)
        ),
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        cached_tokens=usage.cached_tokens if usage else 0,
        cost_usd=usage.cost if usage else None,
        currency=usage.currency if usage else None,
        request_id=(completion.request_id if completion else None)
        or (gw_error.request_id if gw_error else None),
        model_served=completion.model_served if completion else None,
        duration_ms=completion.duration_ms if completion else 0,
        error=str(error) if error is not None else None,
        meta=meta or None,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _truncate_body(body: str) -> str:
    if len(body) > _MAX_INPUT_CHARS:
        logger.warning(
            "Document body is %d chars, exceeds %d limit — truncating before LLM conversion.",
            len(body),
            _MAX_INPUT_CHARS,
        )
        return body[:_MAX_INPUT_CHARS]
    return body


def _render_user_prompt(template: str, document_text: str) -> str:
    try:
        rendered = template.format(document_text=document_text)
    except Exception:
        rendered = ""
    if rendered.strip():
        return rendered
    return f"Convert the following document to clean Markdown:\n\n{document_text}"
