"""kb's connection to pacod-broker: the deployment's ``BROKER_*``, or an admin's override.

Three settings: the broker's address (``BROKER_URL``; blank = no broker), the timeout of
one call (``BROKER_TIMEOUT_SECONDS``, 1 to 30 s) and the token (``BROKER_TOKEN``: a pacod-id
service account's ``pidt_…`` with ``broker.consumer``; blank = open mode, the client then
names itself ``pacod-kb``). Each is the deployment's until an admin sets it on the Broker
page; the overrides are rows of the ledger's ``app_settings`` (both storage backends), and
Reset deletes them.

A token goes only to the address it was given for. The deployment's token belongs to
``BROKER_URL``; a token saved on the page is stored with the address it was saved for
(``broker.token_url``) and is sent nowhere else. Pointing kb at another broker without
giving that broker's token (or choosing open mode) is refused (``token_required``), so the
first broker's token never reaches the second.

A saved token is Fernet-encrypted under a key derived (HKDF-SHA256) from
``PIPELINE_API_SECRET``: a copy of the ledger alone yields no token. Changing the secret makes
a saved token unreadable; it is then sent nowhere and the page asks for it again. The token
is never in any answer: :func:`describe` reports its source and, for a long one, its last
four characters.
"""

from __future__ import annotations

import base64
import logging
import threading
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

if TYPE_CHECKING:
    from pipeline.config import Config

logger = logging.getLogger(__name__)

DEPLOYMENT: Final = "deployment"
CONSOLE: Final = "console"
NONE: Final = "none"

URL_KEY: Final = "broker.url"
TIMEOUT_KEY: Final = "broker.timeout_seconds"
TOKEN_KEY: Final = "broker.token"  # Fernet ciphertext; "" = open mode chosen on the page
TOKEN_URL_KEY: Final = "broker.token_url"  # the address the saved token belongs to
KEYS: Final = (URL_KEY, TIMEOUT_KEY, TOKEN_KEY, TOKEN_URL_KEY)

TIMEOUT_BOUNDS: Final = (1.0, 30.0)
_MAX_URL = 500
_MAX_TOKEN = 4096
_HKDF_INFO = b"pacod-kb broker token v1"


class _Keep:
    def __repr__(self) -> str:
        return "KEEP"


KEEP: Final = _Keep()  # save(token=KEEP): leave the token as it is


class BrokerConfigError(Exception):
    """A change refused before anything was written; ``code`` names the reason."""

    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class BrokerConnection:
    """The connection in force. ``sources`` maps url, timeout_seconds and token to
    ``deployment``, ``console`` or ``none`` (the token only)."""

    url: str
    token: str = field(repr=False)
    timeout: float
    sources: dict[str, str] = field(default_factory=dict, hash=False)
    token_unreadable: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.url)


# ---- encryption ---------------------------------------------------------------------


def _fernet(secret: str) -> Fernet:
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_HKDF_INFO).derive(
        secret.encode()
    )
    return Fernet(base64.urlsafe_b64encode(key))


def _decrypt(secret: str, ciphertext: str) -> str | None:
    if not secret:
        return None
    try:
        return _fernet(secret).decrypt(ciphertext.encode()).decode()
    except (InvalidToken, ValueError):
        return None


# ---- the ledger ---------------------------------------------------------------------


def _load_rows(session_factory: Any | None) -> dict[str, str]:
    if session_factory is None:
        return {}
    from sqlalchemy import select

    from pipeline.db_models import AppSetting
    from pipeline.document_store import _session_scope

    with _session_scope(session_factory) as s:
        rows = s.execute(select(AppSetting).where(AppSetting.key.in_(KEYS))).scalars().all()
        return {str(r.key): str(r.value) for r in rows}


def _write_rows(
    session_factory: Any,
    upserts: dict[str, str],
    deletes: set[str],
    updated_by: str,
) -> None:
    from sqlalchemy import delete, select

    from pipeline.db_models import AppSetting
    from pipeline.document_store import _session_scope

    now = datetime.now(UTC).isoformat()
    with _session_scope(session_factory) as s:
        if deletes:
            s.execute(delete(AppSetting).where(AppSetting.key.in_(sorted(deletes))))
        for key, value in upserts.items():
            row = s.execute(select(AppSetting).where(AppSetting.key == key)).scalar_one_or_none()
            if row is None:
                s.add(AppSetting(key=key, value=value, updated_by=updated_by, updated_at=now))
            else:
                row.value = value
                row.updated_by = updated_by
                row.updated_at = now


# ---- resolving ----------------------------------------------------------------------


def _timeout(raw: str | None) -> float | None:
    try:
        value = float(raw) if raw is not None else None
    except ValueError:
        return None
    if value is None or not TIMEOUT_BOUNDS[0] <= value <= TIMEOUT_BOUNDS[1]:
        return None
    return value


def resolve(config: Config, rows: dict[str, str], secret: str) -> BrokerConnection:
    """The connection the deployment and the stored overrides (*rows*) make together."""
    sources: dict[str, str] = {}
    if URL_KEY in rows:
        url, sources["url"] = rows[URL_KEY], CONSOLE
    else:
        url, sources["url"] = config.broker_url, DEPLOYMENT

    stored_timeout = _timeout(rows.get(TIMEOUT_KEY))
    if stored_timeout is not None:
        timeout, sources["timeout_seconds"] = stored_timeout, CONSOLE
    else:
        timeout, sources["timeout_seconds"] = config.broker_timeout_secs, DEPLOYMENT

    token, unreadable = "", False
    if url and rows.get(TOKEN_URL_KEY) == url:
        stored = rows.get(TOKEN_KEY, "")
        if stored:
            plain = _decrypt(secret, stored)
            token, unreadable = (plain, False) if plain is not None else ("", True)
            sources["token"] = CONSOLE
        else:
            sources["token"] = NONE  # open mode, chosen on the page
    elif url and url == config.broker_url:
        token = config.broker_token
        sources["token"] = DEPLOYMENT if token else NONE
    else:
        sources["token"] = NONE  # another address: the deployment's token stays home
    return BrokerConnection(
        url=url, token=token, timeout=timeout, sources=sources, token_unreadable=unreadable
    )


_lock = threading.Lock()
_cache: tuple[tuple[Any, ...], BrokerConnection] | None = None


def invalidate() -> None:
    global _cache
    with _lock:
        _cache = None


def get_connection(config: Config, session_factory: Any | None, secret: str) -> BrokerConnection:
    """The connection in force, cached until the next save or reset (one pipeline process).

    A ledger that cannot be read leaves the deployment's connection.
    """
    global _cache
    key = (
        id(session_factory),
        config.broker_url,
        config.broker_token,
        config.broker_timeout_secs,
        secret,
    )
    with _lock:
        if _cache is not None and _cache[0] == key:
            return _cache[1]
    try:
        rows = _load_rows(session_factory)
    except Exception as exc:
        logger.warning(
            "Broker settings unreadable; using the deployment's",
            extra={"error": type(exc).__name__},
        )
        return resolve(config, {}, secret)
    conn = resolve(config, rows, secret)
    with _lock:
        _cache = (key, conn)
    return conn


# ---- changing -----------------------------------------------------------------------


def check_url(value: str) -> str:
    """*value* as a broker address (``""`` = off); BrokerConfigError when it is not one."""
    url = value.strip().rstrip("/")
    if not url:
        return ""
    if len(url) > _MAX_URL:
        raise BrokerConfigError("invalid_url", f"The address is longer than {_MAX_URL} characters.")
    if any(c.isspace() or not c.isprintable() for c in url):
        raise BrokerConfigError("invalid_url", "The address may not contain spaces.")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise BrokerConfigError("invalid_url", "The address must be http(s)://host[:port][/path].")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BrokerConfigError(
            "invalid_url", "The address may not carry a user, a query or a fragment."
        )
    try:
        _ = parsed.port
    except ValueError:
        raise BrokerConfigError("invalid_url", "The address has an invalid port.") from None
    return url


def _check_token(value: str) -> str:
    token = value.strip()
    if len(token) > _MAX_TOKEN or any(not c.isprintable() or c.isspace() for c in token):
        raise BrokerConfigError("invalid_token", "The token must be one line without spaces.")
    return token


def save(
    config: Config,
    session_factory: Any | None,
    secret: str,
    changes: dict[str, Any],
    *,
    token: str | None | _Keep = KEEP,
    updated_by: str = "api",
) -> BrokerConnection:
    """Apply *changes* (``url``, ``timeout_seconds``; a value of None = the deployment's).

    *token*: KEEP leaves it; None drops the saved one (back to the deployment's pairing);
    ``""`` is open mode for the resulting address; any other text is the token for it.
    Everything is checked before anything is written.
    """
    if session_factory is None:
        raise RuntimeError("broker settings cannot be saved: no ledger session factory")
    rows = _load_rows(session_factory)
    before = resolve(config, rows, secret)
    after_rows = dict(rows)

    if "url" in changes:
        value = changes["url"]
        if value is None:
            after_rows.pop(URL_KEY, None)
        else:
            after_rows[URL_KEY] = check_url(str(value))
    if "timeout_seconds" in changes:
        value = changes["timeout_seconds"]
        if value is None:
            after_rows.pop(TIMEOUT_KEY, None)
        else:
            low, high = TIMEOUT_BOUNDS
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not low <= float(value) <= high
            ):
                raise BrokerConfigError(
                    "invalid_timeout", f"The timeout must be between {low:g} and {high:g} s."
                )
            after_rows[TIMEOUT_KEY] = repr(float(value))

    target_url = after_rows.get(URL_KEY, config.broker_url)
    if isinstance(token, _Keep):
        pass
    elif token is None:
        after_rows.pop(TOKEN_KEY, None)
        after_rows.pop(TOKEN_URL_KEY, None)
    else:
        if not target_url:
            raise BrokerConfigError("url_required", "Set the broker's address before its token.")
        plain = _check_token(token)
        if plain and not secret:
            raise BrokerConfigError(
                "secrets_disabled",
                "A token cannot be stored: the pipeline has no PIPELINE_API_SECRET.",
                status=503,
            )
        after_rows[TOKEN_KEY] = _fernet(secret).encrypt(plain.encode()).decode() if plain else ""
        after_rows[TOKEN_URL_KEY] = target_url

    after = resolve(config, after_rows, secret)
    if (
        isinstance(token, _Keep)
        and after.url
        and after.url != before.url
        and after.url != config.broker_url
        and before.token
        and not after.token
    ):
        raise BrokerConfigError(
            "token_required",
            "Another broker needs its own token (or open mode): kb's token goes only to the "
            "address it was given for.",
        )

    upserts = {k: v for k, v in after_rows.items() if rows.get(k) != v}
    deletes = set(rows) - set(after_rows)
    if upserts or deletes:
        _write_rows(session_factory, upserts, deletes, updated_by)
    invalidate()
    return after


def reset(session_factory: Any | None) -> None:
    """Back to the deployment's connection: every stored setting and token removed."""
    if session_factory is None:
        raise RuntimeError("broker settings cannot be reset: no ledger session factory")
    _write_rows(session_factory, {}, set(KEYS), "api")
    invalidate()


def describe(config: Config, conn: BrokerConnection, secret: str) -> dict[str, Any]:
    """The connection for the Broker page: values, their sources and defaults; no token."""
    token = conn.token
    return {
        "configured": conn.configured,
        "url": {"value": conn.url, "source": conn.sources["url"], "default": config.broker_url},
        "timeout_seconds": {
            "value": conn.timeout,
            "source": conn.sources["timeout_seconds"],
            "default": config.broker_timeout_secs,
        },
        "token": {
            "configured": bool(token),
            "source": conn.sources["token"],
            "last4": token[-4:] if len(token) >= 8 else None,
            "deployment_configured": bool(config.broker_token),
            "unreadable": conn.token_unreadable,
        },
        "editable": bool(secret),
        "bounds": {"timeout_seconds": {"min": TIMEOUT_BOUNDS[0], "max": TIMEOUT_BOUNDS[1]}},
    }
