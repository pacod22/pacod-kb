"""Regression tests for P0 security fixes (URL ingest, SSRF guard, fetch limits)."""

from __future__ import annotations

import socket
import ssl
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from pipeline.parsers import web

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> Any:
    def factory(headers: dict[str, str], verify: bool) -> httpx.Client:
        return httpx.Client(headers=headers, transport=httpx.MockTransport(handler))

    return factory


@pytest.fixture
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """Map hostnames to IPs without touching the network."""
    table: dict[str, str] = {}

    def getaddrinfo(host: str, *_: Any, **__: Any) -> list[Any]:
        if host not in table:
            raise socket.gaierror(f"unknown host {host}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], 0))]

    monkeypatch.setattr(web.socket, "getaddrinfo", getaddrinfo)
    yield table


# ---------------------------------------------------------------------------
# Scheme validation (local-file read via "URL" ingest)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["/etc/passwd", "file:///etc/passwd", "ftp://example.com/x", "/proc/self/environ", "https://"],
)
def test_require_http_url_rejects_non_http(bad: str) -> None:
    with pytest.raises(ValueError, match="Only http"):
        web.require_http_url(bad)


def test_require_http_url_accepts_http() -> None:
    assert web.require_http_url("  https://example.com/a?b=1 ") == "https://example.com/a?b=1"


def test_ingest_url_only_refuses_local_path(config: Any, tmp_path: Any) -> None:
    from pipeline.acquisition import ingest_document

    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    with pytest.raises(ValueError, match="Only http"):
        ingest_document(str(secret), config, url_only=True)


def test_ingest_url_request_model_rejects_path() -> None:
    from pydantic import ValidationError

    from pipeline.api import IngestUrlRequest

    with pytest.raises(ValidationError):
        IngestUrlRequest(url="/etc/passwd")
    assert IngestUrlRequest(url="https://example.com").url == "https://example.com"


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "192.168.1.1"])
def test_private_hosts_blocked_when_enabled(
    monkeypatch: pytest.MonkeyPatch, fake_dns: dict[str, str], ip: str
) -> None:
    monkeypatch.setenv("WEB_FETCH_ALLOW_PRIVATE", "false")
    fake_dns["internal.test"] = ip
    with pytest.raises(ValueError, match="non-public"):
        web._check_host("http://internal.test/x")


def test_public_host_allowed(monkeypatch: pytest.MonkeyPatch, fake_dns: dict[str, str]) -> None:
    monkeypatch.setenv("WEB_FETCH_ALLOW_PRIVATE", "false")
    fake_dns["public.test"] = "93.184.216.34"
    web._check_host("https://public.test/")


def test_private_allowed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEB_FETCH_ALLOW_PRIVATE", raising=False)
    web._check_host("http://127.0.0.1:6333/collections")  # no DNS, no raise


def test_redirect_to_private_host_blocked(
    monkeypatch: pytest.MonkeyPatch, fake_dns: dict[str, str]
) -> None:
    monkeypatch.setenv("WEB_FETCH_ALLOW_PRIVATE", "false")
    fake_dns["public.test"] = "93.184.216.34"
    fake_dns["internal.test"] = "10.1.2.3"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.test":
            return httpx.Response(302, headers={"Location": "http://internal.test/secret"})
        return httpx.Response(200, content=b"internal data")

    with patch.object(web, "_make_client", _mock_client(handler)):
        with pytest.raises(ValueError, match="non-public"):
            web._browser_fetch("https://public.test/start")


def test_redirect_to_file_scheme_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "file:///etc/passwd"})

    with patch.object(web, "_make_client", _mock_client(handler)):
        with pytest.raises(ValueError, match="Only http"):
            web._browser_fetch("https://example.com/")


def test_relative_redirect_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"Location": "/new"})
        return httpx.Response(200, content=b"moved here")

    with patch.object(web, "_make_client", _mock_client(handler)):
        assert web._browser_fetch("https://example.com/old") == b"moved here"


def test_too_many_redirects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "/loop"})

    with patch.object(web, "_make_client", _mock_client(handler)):
        with pytest.raises(ValueError, match="Too many redirects"):
            web._browser_fetch("https://example.com/loop")


def test_response_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_FETCH_MAX_BYTES", "10")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 100)

    with patch.object(web, "_make_client", _mock_client(handler)):
        with pytest.raises(ValueError, match="exceeds 10 bytes"):
            web._browser_fetch("https://example.com/big")


# ---------------------------------------------------------------------------
# Insecure-TLS fallback
# ---------------------------------------------------------------------------


def _ssl_failing_then_ok() -> tuple[Any, list[bool]]:
    calls: list[bool] = []

    def factory(headers: dict[str, str], verify: bool) -> httpx.Client:
        calls.append(verify)

        def handler(request: httpx.Request) -> httpx.Response:
            if verify:
                try:
                    raise ssl.SSLError("certificate verify failed")
                except ssl.SSLError as cause:
                    raise httpx.ConnectError("ssl failure", request=request) from cause
            return httpx.Response(200, content=b"ok")

        return httpx.Client(headers=headers, transport=httpx.MockTransport(handler))

    return factory, calls


def test_insecure_tls_retry_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOW_INSECURE_TLS", raising=False)
    factory, calls = _ssl_failing_then_ok()
    with patch.object(web, "_make_client", factory):
        assert web._browser_fetch("https://badcert.test/") == b"ok"
    assert calls == [True, False]


def test_insecure_tls_refused_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_INSECURE_TLS", "false")
    factory, calls = _ssl_failing_then_ok()
    with patch.object(web, "_make_client", factory):
        with pytest.raises(ValueError, match="Failed to fetch"):
            web._browser_fetch("https://badcert.test/")
    assert calls == [True]


# ---------------------------------------------------------------------------
# Auth hardening
# ---------------------------------------------------------------------------


def test_api_key_constant_time_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from pipeline import api

    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "s3cret")
    api.require_api_key("s3cret")
    for bad in (None, "", "s3cre", "s3cret!"):
        with pytest.raises(HTTPException):
            api.require_api_key(bad)


@pytest.mark.parametrize("value", ["", "change-me-session-secret"])
def test_session_secret_never_uses_placeholder(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    from web import app as web_app

    monkeypatch.setenv("UI_SESSION_SECRET", value)
    first, second = web_app._session_secret(), web_app._session_secret()
    assert first not in ("", "change-me-session-secret")
    assert first != second  # random per call


def test_session_secret_uses_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from web import app as web_app

    monkeypatch.setenv("UI_SESSION_SECRET", "a-real-secret-value")
    assert web_app._session_secret() == "a-real-secret-value"


# ---------------------------------------------------------------------------
# DELETE /documents/{prefix} — wildcard and ambiguity handling over HTTP
# ---------------------------------------------------------------------------


def test_delete_document_prefix_http_errors(config: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import asynccontextmanager

    from fastapi.testclient import TestClient

    from pipeline import api
    from pipeline.document_store import FilesystemDocumentStore

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    api.app.state.config = config
    api.app.state.document_store = FilesystemDocumentStore(config)
    api.app.state.vector_store = object()

    fs = FilesystemDocumentStore(config)
    base = {
        "title": "T",
        "source_file": "t.md",
        "source_uri": "t.md",
        "source_version": "",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "converted_at": "2026-01-01T00:00:00+00:00",
        "converter": "passthrough/1.0",
        "extraction_method": "deterministic",
        "language": "en",
        "doc_type": "generic",
        "supersedes": None,
        "valid_until": None,
    }
    for doc_id in ("12345678" + "a" * 56, "12345678" + "b" * 56):
        fm = {**base, "doc_id": doc_id, "source_sha256": doc_id, "title": doc_id[8:12]}
        fs.save(doc_id, fm, "# T\n\nbody\n", b"x", ".md")

    with TestClient(api.app, headers={"X-API-Key": "test-secret"}) as c:
        assert c.delete("/documents/%25").status_code == 400
        assert c.delete("/documents/12345678").status_code == 409
        assert c.delete("/documents/ffffffff").status_code == 404
