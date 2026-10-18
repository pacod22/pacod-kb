"""/query validation and scoping; collection switches keep vectors and keyword rows paired."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from pipeline import api
from pipeline.chunker import Chunk
from pipeline.indexer import make_fts_db
from pipeline.ledger import ledger_session_factory
from pipeline.projects import ProjectRegistry
from pipeline.vectorstore import QdrantVectorStore
from tests.test_phase3 import MockEmbedder

KEY = {"X-API-Key": "test-secret"}
_MISSING = object()


def _chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id, heading_path=["Doc"], heading_level=1,
        parent_chunk=None, sequence=0, byte_start=0, byte_end=len(text.encode()),
        embedded_text=text, valid_until=None, extraction_method="deterministic",
        confidence=1.0, text=text,
    )


@pytest.fixture()
def env(config: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Projects alpha (zebra doc) and beta (giraffe doc), plus an unowned collection."""
    from pipeline.gateway import Gateway
    from pipeline.runtime_config import _CACHE

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    _CACHE.clear()
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)

    registry = ProjectRegistry(config.kb_root)
    registry.create("alpha")
    registry.create("beta")
    qclient = QdrantClient(":memory:")
    stores = {name: QdrantVectorStore(qclient, name) for name in ("alpha", "beta", "loose")}
    embedder = MockEmbedder()
    fts = make_fts_db(config)
    for project, word in (("alpha", "zebra"), ("beta", "giraffe")):
        chunks = [_chunk(f"{project}#s#0", f"{project}-doc", f"{word} facts and figures")]
        stores[project].upsert(chunks, embedder.encode([c.embedded_text for c in chunks]))
        fts.upsert_batch(chunks, project)
    fts.close()

    state = {
        "config": config,
        "ledger_factory": ledger_session_factory(config),
        "gateway": Gateway("", ""),
        "embedder": embedder,
        "project_registry": registry,
        "vector_store": stores["alpha"],
        "document_store": None,
    }
    saved_state = {k: getattr(api.app.state, k, _MISSING) for k in state}
    saved_runtime = dict(api._runtime)
    saved_jobs = dict(api._jobs)
    for k, v in state.items():
        setattr(api.app.state, k, v)
    api._jobs.clear()
    api._runtime.update(active_project="alpha", active_collection="alpha")
    try:
        with TestClient(api.app) as client:
            yield {"client": client, "stores": stores}
    finally:
        api._runtime.clear()
        api._runtime.update(saved_runtime)
        api._jobs.clear()
        api._jobs.update(saved_jobs)
        for k, v in saved_state.items():
            if v is _MISSING:
                delattr(api.app.state, k)
            else:
                setattr(api.app.state, k, v)
        _CACHE.clear()


def _query(client: TestClient, **body: Any) -> Any:
    return client.post("/query", json={"wrap_untrusted": False, **body}, headers=KEY)


@pytest.mark.parametrize(
    "body",
    [
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 51},
        {"query": ""},
        {"query": "x", "as_of": "yesterday"},
        {"query": "x", "as_of": "2024-13-01"},
        {"query": "x", "as_of": "2024-02-30"},
    ],
)
def test_invalid_requests_are_422(env: dict[str, Any], body: dict[str, Any]) -> None:
    assert _query(env["client"], **body).status_code == 422


def test_valid_as_of_is_accepted(env: dict[str, Any]) -> None:
    r = _query(env["client"], query="zebra", as_of="2024-02-29")
    assert r.status_code == 200, r.text


def test_keyword_scope_follows_the_searched_collection(env: dict[str, Any]) -> None:
    # Searching beta must not fuse in alpha's keyword hit for "zebra".
    r = _query(env["client"], query="zebra", collection="beta")
    assert r.status_code == 200, r.text
    assert {hit["source_id"] for hit in r.json()["results"]} == {"beta-doc"}


def test_unowned_collection_is_vector_only(env: dict[str, Any]) -> None:
    r = _query(env["client"], query="zebra", collection="loose")
    assert r.status_code == 200, r.text
    assert r.json()["results"] == []


def test_active_unowned_collection_does_not_leak_keyword_rows(env: dict[str, Any]) -> None:
    client = env["client"]
    assert client.post("/settings", json={"active_collection": "loose"}, headers=KEY).is_success
    r = _query(client, query="zebra")
    assert r.json()["results"] == []


def test_selecting_a_project_collection_selects_the_project(env: dict[str, Any]) -> None:
    r = env["client"].post("/settings", json={"active_collection": "beta"}, headers=KEY)
    assert r.status_code == 200, r.text
    assert r.json()["active_project"] == "beta"
    assert api.app.state.vector_store.collection_name == "beta"


def test_writes_wait_for_a_project_while_an_unowned_collection_is_active(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline.api_routes import ingest

    monkeypatch.setattr(ingest, "_run_ingest_url", lambda *a, **k: None)
    client = env["client"]
    r = client.post("/settings", json={"active_collection": "loose"}, headers=KEY)
    assert r.status_code == 200, r.text
    assert r.json()["active_project"] == "alpha"  # selecting it did not pick a project

    for path, kwargs in (
        ("/ingest/url", {"json": {"url": "https://example.com/a"}}),
        ("/dropzone/scan", {}),
        ("/watch", {}),
        ("/reindex/abc", {}),
        ("/ingest/file", {"files": {"file": ("n.md", b"# N\n\nbody\n", "text/markdown")}}),
    ):
        r = client.post(path, headers=KEY, **kwargs)
        assert r.status_code == 409, (path, r.text)
        assert "belongs to no project" in r.json()["detail"]

    assert client.post("/settings", json={"active_project": "alpha"}, headers=KEY).is_success
    r = client.post("/ingest/url", json={"url": "https://example.com/a"}, headers=KEY)
    assert r.status_code == 200, r.text


def test_wrapping_loads_the_preamble_once_per_response(
    env: dict[str, Any], config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pipeline.prompt_config as prompt_config

    embedder = MockEmbedder()
    chunks = [_chunk(f"alpha#more#{i}", f"alpha-doc-{i}", f"zebra stripes {i}") for i in range(4)]
    env["stores"]["alpha"].upsert(chunks, embedder.encode([c.embedded_text for c in chunks]))
    fts = make_fts_db(config)
    fts.upsert_batch(chunks, "alpha")
    fts.close()

    calls: list[str] = []

    def fake_prompt(_factory: Any, key: str) -> str:
        calls.append(key)
        return "PREAMBLE"

    monkeypatch.setattr(prompt_config, "get_prompt_text", fake_prompt)
    r = env["client"].post("/query", json={"query": "zebra", "top_k": 5}, headers=KEY)
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) >= 3
    assert all(hit["text"].startswith("PREAMBLE\n") for hit in results)
    assert calls == ["wrap_untrusted_preamble"]
