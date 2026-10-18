"""The dashboard's switches apply to the dropzone and watch, not only to API ingests."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pipeline import api
from tests.test_phase3 import MockEmbedder


@pytest.fixture()
def runtime() -> Iterator[dict[str, Any]]:
    saved = dict(api._runtime)
    api._runtime.clear()
    try:
        yield api._runtime
    finally:
        api._runtime.clear()
        api._runtime.update(saved)


def _capture(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _fake_ingest(source: str, _config: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        raise ValueError("stop here")

    monkeypatch.setattr(module, "ingest_document", _fake_ingest)
    return calls


def test_options_follow_the_dashboard_switches(runtime: dict[str, Any]) -> None:
    runtime.update(llm_converter_enabled=False, content_security_enabled=False)
    off = api._ingest_options()
    assert (off.llm_convert, off.trusted) == (False, True)

    runtime.update(llm_converter_enabled=True, content_security_enabled=True)
    on = api._ingest_options()
    assert (on.llm_convert, on.trusted) == (True, False)
    # An explicit per-request choice still wins.
    assert api._ingest_options(llm_convert=False).llm_convert is False


def test_dropzone_honours_converter_off(
    config: Any, runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline import dropzone
    from pipeline.indexer import make_fts_db

    runtime.update(llm_converter_enabled=False, content_security_enabled=False)
    calls = _capture(monkeypatch, dropzone)
    config.kb_dropzone_dir.mkdir(parents=True, exist_ok=True)
    f = config.kb_dropzone_dir / "a.md"
    f.write_text("# A\n\nbody\n", encoding="utf-8")
    old = time.time() - 60
    os.utime(f, (old, old))

    fts = make_fts_db(config)
    try:
        dropzone.scan_dropzone(
            config, MockEmbedder(), object(), fts, options=api._ingest_options
        )  # type: ignore[arg-type]
    finally:
        fts.close()
    assert len(calls) == 1
    assert calls[0]["llm_convert"] is False
    assert calls[0]["trusted"] is True


def test_watch_honours_converter_off(
    config: Any, runtime: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline import watch
    from pipeline.indexer import make_fts_db

    runtime.update(llm_converter_enabled=False, content_security_enabled=True)
    calls = _capture(monkeypatch, watch)
    wf = config.kb_root / "watch.json"
    wf.write_text(json.dumps([{"url": "https://example.com/a"}]), encoding="utf-8")

    fts = make_fts_db(config)
    try:
        watch.watch_sources(
            config, wf, MockEmbedder(), object(), fts, options=api._ingest_options
        )  # type: ignore[arg-type]
    finally:
        fts.close()
    assert len(calls) == 1
    assert calls[0]["llm_convert"] is False
    assert calls[0]["trusted"] is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("auto", None), ("", None), ("true", True), ("yes", True), ("on", True), ("false", False),
     ("no", False)],
)
def test_upload_conversion_choice(raw: str, expected: bool | None) -> None:
    from pipeline.api_routes.ingest import _tri_state

    assert _tri_state(raw) is expected


def test_upload_conversion_choice_refuses_nonsense() -> None:
    from fastapi import HTTPException

    from pipeline.api_routes.ingest import _tri_state

    with pytest.raises(HTTPException):
        _tri_state("maybe")
