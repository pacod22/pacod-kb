"""kb's log, after the pacod logging convention (pacod-demo/docs/LOGGING.md).

Both processes (kb-pipeline, kb-ui) and the `kb` CLI call :func:`setup` once at start. It puts
one handler on the root logger, takes over uvicorn's loggers (their lines become events like any
other) and turns uvicorn's access lines into request lines. Standard library only: the UI image
copies this package next to web/ and installs nothing for it.

Variables (the values are the convention's, the names kb's own):

- ``LOG_LEVEL``: debug, info, warn (or warning), error; any case; anything else is info.
- ``LOG_FORMAT``: json (one object per line, the default) or text; anything else is json.
- ``SERVICE_NAME``: the ``service`` of every event; each process passes its own default.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import UTC, datetime
from typing import Any, TextIO

__all__ = [
    "HEALTH_PATHS",
    "AccessFilter",
    "JsonFormatter",
    "TextFormatter",
    "event",
    "parse_format",
    "parse_level",
    "service_name",
    "setup",
]

LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
FORMATS = ("json", "text")

# Probes Docker (and pacod-caddy) run every few seconds: logged at debug.
HEALTH_PATHS = frozenset({"/health", "/healthz", "/kb/healthz"})

# The five keys every event starts with; a field of the same name gets a trailing underscore.
RESERVED = ("timestamp", "level", "service", "logger", "message")

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")
# The HTTP clients log every request at info, with the full URL (query included) and the feed's
# long polls among them: kept for debug.
_QUIET_BELOW_DEBUG = ("httpx", "httpcore")

# LogRecord's own attributes; everything else on a record came from extra= (or a filter).
# uvicorn's color_message is an ANSI copy of the message: never a field.
_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "color_message",
}

_ANSI = {"TRACE": "\x1b[2m", "DEBUG": "\x1b[34m", "INFO": "\x1b[32m", "WARN": "\x1b[33m"}
_ANSI_ERROR = "\x1b[31m"
_ANSI_RESET = "\x1b[0m"

_configured = False


def parse_level(value: str | None) -> int:
    """LOG_LEVEL's value as a logging level: unknown or unset is info."""
    return LEVELS.get((value or "").strip().lower(), logging.INFO)


def parse_format(value: str | None) -> str:
    """LOG_FORMAT's value: json or text; unknown or unset is json."""
    fmt = (value or "").strip().lower()
    return fmt if fmt in FORMATS else "json"


def service_name(default: str) -> str:
    return (os.environ.get("SERVICE_NAME") or "").strip() or default


def level_name(levelno: int) -> str:
    """The convention's level: Python's warning is warn, critical is error."""
    if levelno >= logging.ERROR:
        return "error"
    if levelno >= logging.WARNING:
        return "warn"
    if levelno >= logging.INFO:
        return "info"
    if levelno >= logging.DEBUG:
        return "debug"
    return "trace"


def timestamp(created: float) -> str:
    """RFC 3339, UTC, exactly three fraction digits and Z."""
    dt = datetime.fromtimestamp(created, tz=UTC)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{dt.microsecond // 1000:03d}Z"


def event(record: logging.LogRecord, service: str) -> dict[str, Any]:
    """The record as the convention's event: the five keys in order, then the fields (flat),
    then error and stack_trace when an exception is attached."""
    message = record.getMessage() or record.name
    out: dict[str, Any] = {
        "timestamp": timestamp(record.created),
        "level": level_name(record.levelno),
        "service": service,
        "logger": record.name,
        "message": message,
    }
    exc = record.exc_info
    has_error = bool(exc and exc[1] is not None) or bool(record.stack_info)
    for key, value in record.__dict__.items():
        if key in _RECORD_ATTRS or key.startswith("_"):
            continue
        while key in out or (has_error and key in ("error", "stack_trace")):
            key += "_"
        out[key] = value
    if exc and exc[1] is not None:
        err = exc[1]
        text = str(err)
        out["error"] = f"{type(err).__name__}: {text}" if text else type(err).__name__
        out["stack_trace"] = "".join(traceback.format_exception(*exc)).rstrip("\n")
    elif record.stack_info:
        out["stack_trace"] = record.stack_info
    return out


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        data = event(record, self.service)
        try:
            return json.dumps(
                data, ensure_ascii=False, separators=(",", ":"), default=str, allow_nan=False
            )
        except (TypeError, ValueError):
            # A field json cannot write (NaN, keys that are not strings): that field as a string,
            # the others as they were.
            return json.dumps(
                {k: _json_safe(v) for k, v in data.items()},
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _text_value(value: Any) -> str:
    """A field's value in a text line: quoted (JSON string syntax) when it has spaces or quotes
    or is empty; a list or mapping as compact JSON."""
    if isinstance(value, str):
        text = value
        quote = not text or any(c.isspace() or c == '"' for c in text)
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        quote = any(c.isspace() for c in text)
    return json.dumps(text, ensure_ascii=False) if quote else text


class TextFormatter(logging.Formatter):
    """`<timestamp> <LEVEL> <service> <logger>: <message> key=value ...`, then a stack trace on
    the next lines. Colour (the level only) when *color*: a terminal, never a container log."""

    def __init__(self, service: str, color: bool = False) -> None:
        super().__init__()
        self.service = service
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        data = event(record, self.service)
        level = data.pop("level").upper()
        padded = f"{level:<5}"
        if self.color:
            padded = f"{_ANSI.get(level, _ANSI_ERROR)}{padded}{_ANSI_RESET}"
        trace = data.pop("stack_trace", None)
        line = f"{data.pop('timestamp')} {padded} {data.pop('service')} {data.pop('logger')}: "
        line += str(data.pop("message"))
        for key, value in data.items():
            line += f" {key}={_text_value(value)}"
        if trace:
            line += "\n" + str(trace)
        return line


class AccessFilter(logging.Filter):
    """uvicorn's access line as the convention's request line: message `<METHOD> <path>
    <status>`, fields method, path (no query string: it can carry codes and tokens), status and
    client_ip; health checks at debug."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        # uvicorn: '%s - "%s %s HTTP/%s" %d' % (client, method, path?query, version, status)
        if not (isinstance(args, tuple) and len(args) == 5 and isinstance(args[4], int)):
            return True
        client, method, full_path, _version, status = args
        path = str(full_path).split("?", 1)[0]
        record.msg = f"{method} {path} {status}"
        record.args = ()
        record.method = str(method)
        record.path = path
        record.status = status
        if client:
            record.client_ip = str(client).rsplit(":", 1)[0]
        if path in HEALTH_PATHS:
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


class _StdHandler(logging.StreamHandler[TextIO]):
    """Writes to sys.stdout (or sys.stderr) as it is at each event, so a replaced stream (a
    test's capture) is followed."""

    def __init__(self, name: str) -> None:
        self._std = name
        super().__init__()

    @property
    def stream(self) -> TextIO:
        return sys.stderr if self._std == "stderr" else sys.stdout

    @stream.setter
    def stream(self, _value: TextIO) -> None:
        pass


def _is_ours(handler: logging.Handler) -> bool:
    return isinstance(handler, _StdHandler)


def setup(default_service: str, *, stream: str = "stdout", force: bool = False) -> None:
    """Configure the process's log once (later calls do nothing unless *force*).

    One handler on the root logger, at LOG_LEVEL, in LOG_FORMAT, on *stream* (stdout; the CLI
    uses stderr so its own output stays clean). uvicorn's loggers lose their handlers and
    propagate to it; its access lines become request lines. httpx and httpcore log only warnings
    unless LOG_LEVEL is debug. Handlers others put on the root logger (pytest's capture) stay.
    """
    global _configured
    if _configured and not force:
        return
    level = parse_level(os.environ.get("LOG_LEVEL"))
    fmt = parse_format(os.environ.get("LOG_FORMAT"))
    service = service_name(default_service)

    handler = _StdHandler(stream)
    # On the handler too: a health check's access line is turned into debug after the logger's
    # own level check, and propagated records skip the ancestors' levels.
    handler.setLevel(level)
    if fmt == "json":
        handler.setFormatter(JsonFormatter(service))
    else:
        isatty = getattr(handler.stream, "isatty", None)
        handler.setFormatter(TextFormatter(service, color=bool(isatty and isatty())))

    root = logging.getLogger()
    for old in [h for h in root.handlers if _is_ours(h)]:
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)

    for name in _UVICORN_LOGGERS:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.setLevel(logging.NOTSET)
        lg.propagate = True
    for name in _QUIET_BELOW_DEBUG:
        logging.getLogger(name).setLevel(
            logging.NOTSET if level <= logging.DEBUG else logging.WARNING
        )
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, AccessFilter) for f in access.filters):
        access.addFilter(AccessFilter())
    _configured = True
