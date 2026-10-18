"""Web / HTML parser using trafilatura."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import httpx
import trafilatura

from pipeline.parsers import ParseResult
from pipeline.parsers.passthrough import _detect_language

logger = logging.getLogger(__name__)

# Headers that Chrome 131 sends on a cold navigation (no prior page).
# Together they defeat most simple UA / header-presence bot-detection checks.
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    # No 'br': httpx needs the optional 'brotli' package to decode it.
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

_TIMEOUT = httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=5.0)
_MAX_REDIRECTS = 5
_DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_DEADLINE_SECS = 180.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _max_bytes() -> int:
    raw = os.environ.get("WEB_FETCH_MAX_BYTES", "").strip()
    return int(raw) if raw.isdigit() else _DEFAULT_MAX_BYTES


def require_http_url(url: str) -> str:
    """Return *url* stripped, or raise ``ValueError`` unless it is an http(s) URL."""
    candidate = url.strip()
    parts = urllib.parse.urlsplit(candidate)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Only http(s) URLs are allowed, got: {url!r}")
    return candidate


def _check_host(url: str) -> None:
    """Reject hosts that resolve to non-public addresses (SSRF guard).

    Disabled by default for local deployments; set ``WEB_FETCH_ALLOW_PRIVATE=false``
    to block loopback, private, link-local and reserved targets.
    """
    if _env_bool("WEB_FETCH_ALLOW_PRIVATE", True):
        return
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve host {host!r}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError(f"Refusing to fetch {url}: {host} resolves to non-public address {ip}")


@dataclass(frozen=True)
class _Fetched:
    url: str
    status_code: int
    headers: httpx.Headers
    content: bytes


def _make_client(headers: dict[str, str], verify: bool) -> httpx.Client:
    return httpx.Client(follow_redirects=False, timeout=_TIMEOUT, headers=headers, verify=verify)


def _get_once(url: str, headers: dict[str, str], verify: bool, deadline: float) -> _Fetched:
    current = require_http_url(url)
    limit = _max_bytes()
    with _make_client(headers, verify) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            _check_host(current)
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location", "")
                    current = require_http_url(str(resp.url.join(location)))
                    continue
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > limit:
                        raise ValueError(f"Response from {current} exceeds {limit} bytes")
                    if time.monotonic() > deadline:
                        raise ValueError(f"Timed out fetching {current}")
                return _Fetched(current, resp.status_code, resp.headers, bytes(buf))
    raise ValueError(f"Too many redirects fetching {url}")


def _http_get(url: str, headers: dict[str, str] = _BROWSER_HEADERS) -> _Fetched:
    """GET *url* with manual redirect handling, size cap and overall deadline.

    Every hop is scheme-checked and (optionally) SSRF-checked.  On a TLS
    certificate error the request is retried without verification only when
    ``ALLOW_INSECURE_TLS`` is true (the default, for legacy sources).
    Raises ``ValueError`` on any failure.
    """
    deadline = time.monotonic() + _DEADLINE_SECS
    try:
        return _get_once(url, headers, True, deadline)
    except ValueError:
        raise
    except httpx.TransportError as exc:
        if not (_is_ssl_error(exc) and _env_bool("ALLOW_INSECURE_TLS", True)):
            raise ValueError(f"Failed to fetch {url}: {exc}") from exc
        logger.warning(
            "SSL error for %s — retrying WITHOUT certificate verification "
            "(set ALLOW_INSECURE_TLS=false to refuse)",
            url,
        )
        try:
            return _get_once(url, headers, False, deadline)
        except ValueError:
            raise
        except Exception as inner:
            raise ValueError(f"Failed to fetch {url}: {inner}") from inner
    except Exception as exc:
        raise ValueError(f"Failed to fetch {url}: {exc}") from exc


def _raise_for_status(resp: _Fetched, url: str) -> None:
    """Raise a ValueError with an actionable message for HTTP error responses.

    Cloudflare (and similar) managed JS challenges return 403 with a
    ``cf-mitigated: challenge`` header. These cannot be passed by any HTTP
    client (header/TLS spoofing does not help — the server requires a real
    browser to execute JavaScript), so direct a user to the manual dropzone
    workflow instead of surfacing a bare HTTP 403.
    """
    if resp.status_code < 400:
        return
    mitigated = resp.headers.get("cf-mitigated", "").lower()
    server = resp.headers.get("server", "").lower()
    if resp.status_code == 403 and ("challenge" in mitigated or "cloudflare" in server):
        raise ValueError(
            f"{url} is behind a Cloudflare challenge (HTTP 403) and cannot be "
            "fetched automatically. Open it in a browser, save the page as HTML "
            "or PDF, and drop the file into the dropzone to ingest it."
        )
    raise ValueError(f"HTTP {resp.status_code} fetching {url}")


def _is_ssl_error(exc: BaseException) -> bool:
    """Walk the exception chain looking for ssl.SSLError."""
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, ssl.SSLError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _browser_fetch(url: str) -> bytes:
    """Fetch *url* with browser-like headers and return the raw response bytes.

    Raises ``ValueError`` on non-http(s) URLs, HTTP errors or connection failures.
    """
    resp = _http_get(url)
    _raise_for_status(resp, url)
    return resp.content


def _apis_bg_html_to_markdown(paragraphs: list[dict[str, object]]) -> str:
    """Convert paragraph list from apis.bg DocTextJson into Markdown."""
    from bs4 import BeautifulSoup

    lines: list[str] = []
    for par in paragraphs:
        html = str(par.get("text", ""))
        if not html.strip():
            continue
        soup = BeautifulSoup(html, "html.parser")
        h3 = soup.find("h3")
        h4 = soup.find("h4")
        if h3:
            classes: list[object] = list(h3.get("class") or [])
            prefix = "# " if "doc-caption" in classes else "## "
            text = h3.get_text(" ", strip=True)
            if text:
                lines.append(f"{prefix}{text}")
        elif h4:
            text = h4.get_text(" ", strip=True)
            if text:
                lines.append(f"### {text}")
        else:
            text = soup.get_text(" ", strip=True)
            if text:
                lines.append(text)
    return "\n\n".join(lines)


# Matches: https://<subdomain>.apis.bg/doc/<id>[/<db>[/<topar>]]
_APIS_BG_DOC_RE = re.compile(
    r"^https://([a-zA-Z0-9_-]+)\.apis\.bg/doc/(\d+)(?:/(\d+))?", re.IGNORECASE
)


def _fetch_apis_bg(url: str) -> ParseResult | None:
    """Special-case handler for legislation.apis.bg and similar Angular SPAs at *.apis.bg.

    These sites render via Angular + a hidden REST backend (web-api.apis.bg).
    Trafilatura sees only the empty <app-root> shell; this fetches the real content
    from the backend API.

    Returns None if the URL doesn't match or any API call fails (caller falls
    back to regular fetch + trafilatura).
    """
    m = _APIS_BG_DOC_RE.match(url)
    if not m:
        return None
    subdomain, doc_id_str, db_index_str = m.group(1), m.group(2), m.group(3)
    if subdomain.lower() == "web-api":
        return None

    db_index = int(db_index_str) if db_index_str else 0
    api_base = f"https://web-api.apis.bg/api/obshtina-{subdomain}/"

    api_headers = {
        **_BROWSER_HEADERS,
        "Accept": "application/json",
        # Exclude 'br' (Brotli): httpx decompresses gzip/deflate natively but needs
        # the optional 'brotli' package for Brotli.  Forcing gzip avoids the dep.
        "Accept-Encoding": "gzip, deflate",
        "Origin": f"https://{subdomain}.apis.bg",
        "Referer": f"https://{subdomain}.apis.bg/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }

    def _api_get(path: str) -> object:
        resp = _http_get(api_base + path, api_headers)
        if resp.status_code >= 400:
            raise ValueError(f"HTTP {resp.status_code} from {api_base + path}")
        return json.loads(resp.content)

    try:
        info_data = _api_get(f"DocInfo?uniqueId={doc_id_str}&dbIndex={db_index}")
        assert isinstance(info_data, dict)
        doc_info = info_data.get("document", {})
        assert isinstance(doc_info, dict)
        title: str = str(doc_info.get("caption", url))

        text_data = _api_get(f"DocTextJson/?uniqueId={doc_id_str}&dbIndex={db_index}")
        assert isinstance(text_data, dict)
        paragraphs = text_data.get("paragraphs", [])
        assert isinstance(paragraphs, list)
    except Exception as exc:
        logger.warning("apis.bg API fetch failed for %s: %s — falling back to HTML", url, exc)
        return None

    body = _apis_bg_html_to_markdown(paragraphs)
    language = _detect_language(body)
    raw_snapshot = json.dumps(text_data, ensure_ascii=False).encode("utf-8")

    return ParseResult(
        body=body,
        title=title,
        language=language,
        extraction_method="deterministic",
        converter="apis.bg-api",
        raw_snapshot=raw_snapshot,
    )


class WebParser:
    def parse(self, source: Path | str) -> ParseResult:
        source_str = str(source)
        is_url = source_str.startswith(("http://", "https://"))

        if is_url:
            # Try SPA-specific handlers before the generic trafilatura path
            result = _fetch_apis_bg(source_str)
            if result is not None:
                return result
            snapshot = _browser_fetch(source_str)
        else:
            path = Path(source)
            snapshot = path.read_bytes()

        return self._parse_bytes(snapshot, source_str)

    def _parse_bytes(self, html_bytes: bytes, source_uri: str) -> ParseResult:
        """Parse pre-fetched HTML bytes (avoids double-fetching for URL sources).

        For known SPA URLs, the pre-fetched bytes may be compressed or empty
        (e.g. an Angular shell); in that case the SPA handler is tried first.
        """
        if source_uri.startswith(("http://", "https://")):
            result = _fetch_apis_bg(source_uri)
            if result is not None:
                return result

        raw_html = html_bytes.decode("utf-8", errors="replace")

        body = trafilatura.extract(
            raw_html,
            output_format="markdown",
            include_tables=True,
            include_comments=False,
            include_formatting=True,
            favor_recall=True,
        )
        if not body:
            logger.warning("trafilatura extracted no content from %s", source_uri)
            body = ""

        title = _extract_html_title(raw_html) or _first_heading(body) or source_uri
        language = _detect_language(body or raw_html)

        return ParseResult(
            body=body,
            title=title,
            language=language,
            extraction_method="deterministic",
            converter=f"trafilatura/{trafilatura.__version__}",
            raw_snapshot=html_bytes,
        )


def _extract_html_title(html: str) -> str:
    import html as html_lib
    import re

    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    return html_lib.unescape(m.group(1)).strip() if m else ""


def _first_heading(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""
