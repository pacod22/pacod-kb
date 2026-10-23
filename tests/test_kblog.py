"""kblog: the pacod logging convention (pacod-demo/docs/LOGGING.md) for kb-pipeline and kb-ui."""

from __future__ import annotations

import io
import json
import logging
import re
import sys
from collections.abc import Iterator
from typing import Any

import pytest

import kblog

_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
_UVICORN = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")
_CLIENTS = ("httpx", "httpcore")


def _record(
    msg: str = "hello %s",
    args: tuple[Any, ...] = ("world",),
    level: int = logging.INFO,
    name: str = "pipeline.test",
    exc_info: Any = None,
    **extra: Any,
) -> logging.LogRecord:
    record = logging.LogRecord(name, level, __file__, 1, msg, args, exc_info)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def _json(record: logging.LogRecord, service: str = "kb-test") -> dict[str, Any]:
    line = kblog.JsonFormatter(service).format(record)
    assert "\n" not in line
    data: dict[str, Any] = json.loads(line)
    return data


@pytest.fixture()
def restore_logging() -> Iterator[None]:
    """setup() changes process-wide logging: put the root and uvicorn loggers back."""
    names = ["", *_UVICORN, *_CLIENTS]
    saved = {
        n: (
            list(logging.getLogger(n).handlers),
            logging.getLogger(n).level,
            logging.getLogger(n).propagate,
            list(logging.getLogger(n).filters),
        )
        for n in names
    }
    configured = kblog._configured
    yield
    for n, (handlers, level, propagate, filters) in saved.items():
        lg = logging.getLogger(n)
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate
        lg.filters[:] = filters
    kblog._configured = configured


# ---- the JSON event ------------------------------------------------------------------------------


def test_the_five_keys_come_first_in_order() -> None:
    data = _json(_record(count=3))
    assert list(data)[:5] == ["timestamp", "level", "service", "logger", "message"]
    assert data["service"] == "kb-test"
    assert data["logger"] == "pipeline.test"
    assert data["message"] == "hello world"
    assert data["level"] == "info"


def test_the_timestamp_is_utc_with_milliseconds() -> None:
    record = _record()
    record.created = 1792737667.6951  # 2026-10-23T06:41:07.695Z
    data = _json(record)
    assert _TS.match(data["timestamp"]), data["timestamp"]
    assert data["timestamp"] == "2026-10-23T06:41:07.695Z"
    record.created = 1792737667.0
    assert _json(record)["timestamp"].endswith(":07.000Z")


@pytest.mark.parametrize(
    ("levelno", "name"),
    [
        (5, "trace"),
        (logging.DEBUG, "debug"),
        (logging.INFO, "info"),
        (logging.WARNING, "warn"),
        (logging.ERROR, "error"),
        (logging.CRITICAL, "error"),
    ],
)
def test_levels_are_the_conventions(levelno: int, name: str) -> None:
    assert _json(_record(level=levelno))["level"] == name


def test_extra_fields_are_flat_with_json_types() -> None:
    logger = logging.getLogger("kblog.test.extra")
    records: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Keep()
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning(
            "done",
            extra={
                "count": 3,
                "ratio": 0.5,
                "ok": True,
                "none": None,
                "items": [1, "a"],
                "meta": {"k": 1},
            },
        )
    finally:
        logger.removeHandler(handler)
    data = _json(records[0])
    assert data["count"] == 3 and data["ratio"] == 0.5 and data["ok"] is True
    assert data["none"] is None and data["items"] == [1, "a"] and data["meta"] == {"k": 1}
    assert list(data)[5:] == ["count", "ratio", "ok", "none", "items", "meta"]
    # Nothing else: no process id, thread or host.
    assert set(data) == {
        "timestamp",
        "level",
        "service",
        "logger",
        "message",
        "count",
        "ratio",
        "ok",
        "none",
        "items",
        "meta",
    }


def test_unserialisable_values_become_strings() -> None:
    class Thing:
        def __str__(self) -> str:
            return "thing"

    data = _json(_record(obj=Thing(), nan=float("nan"), n=1.5, items=[1, 2], meta={"a": 1}))
    assert data["obj"] == "thing"
    assert data["nan"] == "nan"
    # The fields json can write stay as they were.
    assert data["n"] == 1.5 and data["items"] == [1, 2] and data["meta"] == {"a": 1}


def test_reserved_names_get_a_trailing_underscore() -> None:
    record = _record()
    # As extra={...} would set them (level is _record's own parameter).
    for key, value in {"timestamp": "t", "level": "l", "service": "s", "logger": "g"}.items():
        setattr(record, key, value)
    data = _json(record)
    assert data["timestamp_"] == "t" and data["level_"] == "l"
    assert data["service_"] == "s" and data["logger_"] == "g"
    assert data["level"] == "info" and data["service"] == "kb-test"


def test_uvicorns_colour_copy_is_never_a_field() -> None:
    data = _json(_record(color_message="\x1b[36mhello\x1b[0m"))
    assert "color_message" not in data
    assert "\x1b" not in json.dumps(data)


def test_an_empty_message_takes_the_loggers_name() -> None:
    assert _json(_record(msg="", args=()))["message"] == "pipeline.test"


def test_an_exception_adds_error_and_stack_trace() -> None:
    try:
        raise ValueError("bad input")
    except ValueError:
        exc = sys.exc_info()
    data = _json(_record(level=logging.ERROR, exc_info=exc, error="mine"))
    assert data["error"] == "ValueError: bad input"
    assert data["stack_trace"].startswith("Traceback (most recent call last):\n")
    assert data["stack_trace"].endswith("ValueError: bad input")
    assert data["error_"] == "mine"


def test_an_exception_without_text_is_its_type() -> None:
    try:
        raise KeyError()
    except KeyError:
        exc = sys.exc_info()
    assert _json(_record(exc_info=exc))["error"] == "KeyError"


# ---- the text line -------------------------------------------------------------------------------


def test_the_text_line_has_the_fixed_order() -> None:
    record = _record(level=logging.WARNING, count=3, note="two words", empty="")
    record.created = 1792737667.6951
    line = kblog.TextFormatter("kb-test").format(record)
    assert line == (
        "2026-10-23T06:41:07.695Z WARN  kb-test pipeline.test: hello world count=3 "
        'note="two words" empty=""'
    )
    assert "\x1b" not in line


def test_the_text_line_pads_levels_to_five() -> None:
    for levelno, label in (
        (logging.INFO, "INFO "),
        (logging.DEBUG, "DEBUG"),
        (logging.ERROR, "ERROR"),
    ):
        line = kblog.TextFormatter("s").format(_record(level=levelno))
        # 24 characters of timestamp, a space, the level in five, a space, the service.
        assert line[24:32] == f" {label} s"


def test_the_text_line_puts_the_stack_trace_on_the_next_lines() -> None:
    try:
        raise RuntimeError("down")
    except RuntimeError:
        exc = sys.exc_info()
    first, *rest = kblog.TextFormatter("s").format(_record(exc_info=exc)).split("\n")
    assert first.endswith('hello world error="RuntimeError: down"')
    assert rest[0] == "Traceback (most recent call last):"
    assert rest[-1] == "RuntimeError: down"


def test_colour_only_when_asked() -> None:
    plain = kblog.TextFormatter("s").format(_record())
    coloured = kblog.TextFormatter("s", color=True).format(_record())
    assert "\x1b[" not in plain
    assert "\x1b[" in coloured


# ---- uvicorn's access line -----------------------------------------------------------------------


def _access(path: str, status: int = 200, client: str = "172.18.0.9:51234") -> logging.LogRecord:
    record = _record(
        msg='%s - "%s %s HTTP/%s" %d',
        args=(client, "GET", path, "1.1", status),
        name="uvicorn.access",
    )
    assert kblog.AccessFilter().filter(record)
    return record


def test_the_access_line_becomes_a_request_line_without_the_query() -> None:
    data = _json(_access("/kb/auth/callback?code=secret&state=abc", 302))
    assert data["message"] == "GET /kb/auth/callback 302"
    assert data["method"] == "GET"
    assert data["path"] == "/kb/auth/callback"
    assert data["status"] == 302
    assert data["client_ip"] == "172.18.0.9"
    assert data["level"] == "info"
    assert "secret" not in json.dumps(data) and "HTTP/1.1" not in json.dumps(data)


@pytest.mark.parametrize("path", ["/health", "/healthz", "/kb/healthz", "/kb/healthz?x=1"])
def test_health_checks_are_debug(path: str) -> None:
    record = _access(path)
    assert record.levelno == logging.DEBUG and record.levelname == "DEBUG"
    assert _json(record)["level"] == "debug"


def test_an_access_line_without_a_client_has_no_client_ip() -> None:
    assert "client_ip" not in _json(_access("/kb", client=""))


def test_other_uvicorn_records_pass_untouched() -> None:
    record = _record(msg="Started server process [%d]", args=(42,), name="uvicorn.error")
    assert kblog.AccessFilter().filter(record)
    assert record.getMessage() == "Started server process [42]"


# ---- the variables -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "level"),
    [
        ("debug", logging.DEBUG),
        ("DEBUG", logging.DEBUG),
        (" Info ", logging.INFO),
        ("warn", logging.WARNING),
        ("WARNING", logging.WARNING),
        ("error", logging.ERROR),
        ("critical", logging.INFO),
        ("verbose", logging.INFO),
        ("", logging.INFO),
        (None, logging.INFO),
    ],
)
def test_log_level_values(value: str | None, level: int) -> None:
    assert kblog.parse_level(value) == level


@pytest.mark.parametrize(
    ("value", "fmt"),
    [
        ("json", "json"),
        ("JSON", "json"),
        ("text", "text"),
        (" Text", "text"),
        ("console", "json"),
        ("", "json"),
        (None, "json"),
    ],
)
def test_log_format_values(value: str | None, fmt: str) -> None:
    assert kblog.parse_format(value) == fmt


def test_service_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERVICE_NAME", raising=False)
    assert kblog.service_name("kb-ui") == "kb-ui"
    monkeypatch.setenv("SERVICE_NAME", " ")
    assert kblog.service_name("kb-ui") == "kb-ui"
    monkeypatch.setenv("SERVICE_NAME", "kb-ui-2")
    assert kblog.service_name("kb-ui") == "kb-ui-2"


# ---- setup ---------------------------------------------------------------------------------------


def _ours() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if kblog._is_ours(h)]


def test_setup_writes_json_to_stdout_and_takes_over_uvicorn(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "info")
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.setenv("SERVICE_NAME", "kb-ui")
    stray = logging.StreamHandler(io.StringIO())
    logging.getLogger("uvicorn.access").addHandler(stray)
    logging.getLogger("uvicorn.access").propagate = False
    kblog.setup("default", force=True)

    assert len(_ours()) == 1
    for name in _UVICORN:
        lg = logging.getLogger(name)
        assert lg.handlers == [] and lg.propagate and lg.level == logging.NOTSET

    logging.getLogger("uvicorn.error").info(
        "Started server process [%d]", 7, extra={"color_message": "\x1b[1m"}
    )
    access = logging.getLogger("uvicorn.access")
    access.info('%s - "%s %s HTTP/%s" %d', "10.0.0.1:1", "GET", "/kb/login?next=/x", "1.1", 302)
    access.info('%s - "%s %s HTTP/%s" %d', "10.0.0.1:1", "GET", "/kb/healthz", "1.1", 200)
    logging.getLogger("web.app").debug("hidden at info")

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["message"] for line in lines] == [
        "Started server process [7]",
        "GET /kb/login 302",
    ]
    assert all(line["service"] == "kb-ui" for line in lines)
    assert lines[1]["path"] == "/kb/login"


def test_setup_debug_shows_health_checks_and_text_has_no_ansi(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "TEXT")
    monkeypatch.delenv("SERVICE_NAME", raising=False)
    kblog.setup("kb-pipeline", force=True)
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5", "GET", "/health", "1.1", 200
    )
    out = capsys.readouterr().out
    assert re.match(
        r"^\S+Z DEBUG kb-pipeline uvicorn.access: GET /health 200 method=GET path=/health "
        r"status=200 client_ip=127.0.0.1\n$",
        out,
    ), out
    assert "\x1b" not in out


def test_setup_runs_once_and_keeps_foreign_handlers(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    foreign = logging.StreamHandler(io.StringIO())
    logging.getLogger().addHandler(foreign)
    monkeypatch.setenv("LOG_LEVEL", "info")
    kblog.setup("kb-pipeline", force=True)
    first = _ours()
    monkeypatch.setenv("LOG_LEVEL", "error")
    kblog.setup("kb-pipeline")  # already set up: nothing changes
    assert _ours() == first and first[0].level == logging.INFO
    kblog.setup("kb-pipeline", force=True)  # replaces its own handler only
    assert len(_ours()) == 1 and _ours()[0] is not first[0]
    assert _ours()[0].level == logging.ERROR
    assert foreign in logging.getLogger().handlers


def test_setup_on_stderr(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    kblog.setup("kb-pipeline", stream="stderr", force=True)
    logging.getLogger("pipeline.cli").warning("to stderr")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["message"] == "to stderr"


def test_load_config_does_not_reconfigure(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    kblog.setup("kb-pipeline", force=True)
    before = _ours()
    monkeypatch.setenv("KB_ROOT", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "warning")
    from pipeline.config import load_config

    config = load_config()
    assert _ours() == before
    assert config.log_level == "WARNING"


@pytest.mark.parametrize(("value", "level"), [("info", logging.WARNING), ("debug", logging.NOTSET)])
def test_http_clients_log_requests_only_at_debug(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch, value: str, level: int
) -> None:
    # httpx logs every request at info with its full URL (query included).
    monkeypatch.setenv("LOG_LEVEL", value)
    kblog.setup("kb-ui", force=True)
    for name in _CLIENTS:
        assert logging.getLogger(name).level == level
