"""Who acts reaches the pipeline (X-KB-Actor), and deletions are erased and confirmed."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from pipeline import api
from pipeline.ledger import ledger_session_factory
from tests.fake_id import FakeId
from tests.web_support import login_as, wait_until, web_client

KEY = {"X-API-Key": "test-secret"}


@pytest.fixture()
def pipeline_client(config: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from pipeline.gateway import Gateway
    from pipeline.runtime_config import _CACHE

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    _CACHE.clear()
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    api.app.state.config = config
    api.app.state.ledger_factory = ledger_session_factory(config)
    api.app.state.gateway = Gateway("", "")
    saved_jobs = dict(api._jobs)
    try:
        with TestClient(api.app) as c:
            yield c
    finally:
        api._jobs.clear()
        api._jobs.update(saved_jobs)
        del api.app.state.ledger_factory
        del api.app.state.gateway
        _CACHE.clear()


# ---- the pipeline ----------------------------------------------------------------------------


def test_the_actor_is_captured_when_a_job_is_queued(
    pipeline_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline.api_routes import ingest

    monkeypatch.setattr(ingest, "_run_ingest_url", lambda *a, **k: None)
    api.app.state.embedder = api.app.state.vector_store = api.app.state.document_store = None
    r = pipeline_client.post(
        "/ingest/url",
        json={"url": "https://example.com/a"},
        headers={**KEY, "X-KB-Actor": "usr_alice1"},
    )
    assert r.status_code == 200, r.text
    assert api._jobs[r.json()["job_id"]]["params"]["actor"] == "usr_alice1"


def test_the_upload_route_keeps_the_actor_for_its_runner(
    pipeline_client: TestClient, config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline.api_routes import ingest

    seen: dict[str, Any] = {}
    monkeypatch.setattr(ingest, "_run_ingest_file", lambda *a, **k: seen.update(params=a[2]))
    api.app.state.embedder = api.app.state.vector_store = api.app.state.document_store = None
    r = pipeline_client.post(
        "/ingest/file",
        files={"file": ("n.md", b"# N\n\nbody\n", "text/markdown")},
        headers={**KEY, "X-KB-Actor": "usr_up1"},
    )
    assert r.status_code == 200, r.text
    # The route updates the job's params after creating it, and hands its dict to the runner.
    assert api._jobs[r.json()["job_id"]]["params"]["actor"] == "usr_up1"
    assert seen["params"]["actor"] == "usr_up1"


def test_a_malformed_actor_is_refused_and_needs_the_key(pipeline_client: TestClient) -> None:
    assert pipeline_client.get("/settings", headers={**KEY, "X-KB-Actor": "x;y"}).status_code == 400
    assert pipeline_client.get("/settings", headers={"X-KB-Actor": "usr_a"}).status_code == 403


def test_setting_changes_record_the_actor(pipeline_client: TestClient, config: Any) -> None:
    from sqlalchemy import select

    from pipeline.db_models import AppSetting
    from pipeline.document_store import _session_scope

    r = pipeline_client.patch(
        "/llm-config", json={"converter_max_tokens": 1000}, headers={**KEY, "X-KB-Actor": "usr_b"}
    )
    assert r.status_code == 200, r.text
    with _session_scope(ledger_session_factory(config)) as s:
        row = s.execute(
            select(AppSetting).where(AppSetting.key == "llm.converter_max_tokens")
        ).scalar_one()
        assert row.updated_by == "usr_b"


def test_ingest_runs_attribute_their_llm_calls(
    pipeline_client: TestClient, config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline import acquisition
    from pipeline.api_routes import ingest

    seen: dict[str, Any] = {}

    def fake_ingest(source: str, _config: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise ValueError("stop")

    monkeypatch.setattr(acquisition, "ingest_document", fake_ingest)
    params = {
        "url": "https://example.com/a",
        "doc_type": "generic",
        "source_version": "",
        "supersedes": None,
        "skip_index": True,
        "actor": "usr_c",
        "project": "proj-a",
    }
    job_id = api._create_job("ingest_url", params)
    # The runner records the failure on the job (it does not raise).
    ingest._run_ingest_url(job_id, params, config, None, None, None)  # type: ignore[arg-type]
    assert api._jobs[job_id]["status"] == "error"
    attr = seen["attribution"]
    assert (attr.user_id, attr.trace, attr.end_user) == ("usr_c", job_id, "proj-a")


def test_erase_anonymises_a_person(pipeline_client: TestClient, config: Any) -> None:
    from sqlalchemy import select

    from pipeline.db_models import UsageEvent
    from pipeline.document_store import _session_scope
    from pipeline.llm_management import record_usage_event

    record_usage_event(
        session_factory=ledger_session_factory(config),
        user_id="usr_gone",
        session_id=None,
        turn_id="doc",
        model="m/x",
        purpose="document_conversion",
        input_tokens=1,
        output_tokens=1,
        cached_tokens=0,
        cost_usd=None,
    )
    job_id = api._create_job("ingest_url", {"actor": "usr_gone"})
    r = pipeline_client.post("/people/usr_gone/erase", headers=KEY)
    assert r.status_code == 200, r.text
    assert r.json()["usage_events"] == 1
    with _session_scope(ledger_session_factory(config)) as s:
        assert [u.user_id for u in s.execute(select(UsageEvent)).scalars()] == ["deleted-user"]
    assert api._jobs[job_id]["params"]["actor"] == "deleted-user"
    assert pipeline_client.post("/people/bad id/erase", headers=KEY).status_code in (400, 404)


# ---- the UI ------------------------------------------------------------------------------------


class _Pipeline:
    """A stand-in pipeline API recording what the UI sends."""

    def __init__(self, erase_status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.erase_status = erase_status

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/erase"):
            return httpx.Response(self.erase_status, json={"usage_events": 0})
        return httpx.Response(200, json={})


def test_ui_calls_to_the_pipeline_say_who_acts() -> None:
    fake, pipe = FakeId(), _Pipeline()
    with web_client(fake, httpx.MockTransport(pipe.handle)) as client:
        sub = login_as(client, fake, "admin")
        pipe.requests.clear()
        assert client.get("/kb/pipeline/").status_code == 200
    # The dashboard's own calls (the sidebar middleware reads anonymously, before sign-in
    # is resolved, and only /settings and /projects).
    calls = [r for r in pipe.requests if r.url.path in ("/documents", "/collections")]
    assert calls
    assert {r.headers.get("x-kb-actor") for r in calls} == {sub}


def test_a_deleted_account_is_erased_then_confirmed() -> None:
    fake, pipe = FakeId(), _Pipeline()
    with web_client(fake, httpx.MockTransport(pipe.handle)) as client:
        sub = login_as(client, fake, "viewer")
        fake.delete(sub)
        wait_until(lambda: fake.deletions_done == [sub])
        assert client.get("/kb", follow_redirects=False).status_code == 303  # signed out
    erases = [r for r in pipe.requests if r.url.path == f"/people/{sub}/erase"]
    assert erases and erases[0].method == "POST"


def test_a_failed_erase_is_not_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    from web import app as web_app

    monkeypatch.setattr(web_app, "ERASE_ATTEMPTS", 2)
    monkeypatch.setattr(web_app, "ERASE_FIRST_WAIT_SECS", 0.01)
    fake, pipe = FakeId(), _Pipeline(erase_status=500)
    with web_client(fake, httpx.MockTransport(pipe.handle)) as client:
        sub = login_as(client, fake, "viewer")
        fake.delete(sub)
        wait_until(lambda: len([r for r in pipe.requests if r.url.path.endswith("/erase")]) >= 2)
    assert fake.deletions_done == []
