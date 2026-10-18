from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi.testclient import TestClient

from pipeline import api
from pipeline.projects import ProjectRegistry


class _DummyStore:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._docs)


class _DummyVectorStore:
    def __init__(self) -> None:
        self.client = object()


class _DummyQdrantVectorStore:
    def __init__(self, client: Any, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name


@asynccontextmanager
async def _noop_lifespan(_: Any):
    yield


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test-secret"}


def _seed_project_state(*, kb_root, docs: list[dict[str, Any]]) -> str:
    registry = ProjectRegistry(kb_root)
    project = registry.create(name="Project Alpha", collection_name="alpha-col")
    api.app.state.project_registry = registry
    api.app.state.document_store = _DummyStore(docs)
    api.app.state.vector_store = _DummyVectorStore()
    return project.slug


def test_patch_project_returns_409_when_job_is_running(kb_root, monkeypatch) -> None:
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api, "QdrantVectorStore", _DummyQdrantVectorStore)
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop_lifespan)

    old_jobs = dict(api._jobs)
    old_runtime = dict(api._runtime)
    api._jobs.clear()
    api._runtime.clear()

    try:
        slug = _seed_project_state(kb_root=kb_root, docs=[])
        api._runtime["active_project"] = "other-project"
        api._jobs["j-running"] = {"status": "running"}

        client = TestClient(api.app)
        response = client.patch(
            f"/projects/{slug}",
            json={"collection_name": "alpha-new"},
            headers=_auth_headers(),
        )
        client.close()

        assert response.status_code == 409
        assert "running" in response.text
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)
        api._runtime.clear()
        api._runtime.update(old_runtime)


def test_patch_project_returns_409_for_active_project(kb_root, monkeypatch) -> None:
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api, "QdrantVectorStore", _DummyQdrantVectorStore)
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop_lifespan)

    old_jobs = dict(api._jobs)
    old_runtime = dict(api._runtime)
    api._jobs.clear()
    api._runtime.clear()

    try:
        slug = _seed_project_state(kb_root=kb_root, docs=[])
        api._runtime["active_project"] = slug

        client = TestClient(api.app)
        response = client.patch(
            f"/projects/{slug}",
            json={"collection_name": "alpha-new"},
            headers=_auth_headers(),
        )
        client.close()

        assert response.status_code == 409
        assert "active project" in response.text
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)
        api._runtime.clear()
        api._runtime.update(old_runtime)


def test_patch_project_returns_409_when_project_has_documents(kb_root, monkeypatch) -> None:
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api, "QdrantVectorStore", _DummyQdrantVectorStore)
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop_lifespan)

    old_jobs = dict(api._jobs)
    old_runtime = dict(api._runtime)
    api._jobs.clear()
    api._runtime.clear()

    try:
        slug = _seed_project_state(
            kb_root=kb_root,
            docs=[{"doc_id": "doc-1", "project": "project-alpha"}],
        )
        api._runtime["active_project"] = "other-project"

        client = TestClient(api.app)
        response = client.patch(
            f"/projects/{slug}",
            json={"collection_name": "alpha-new"},
            headers=_auth_headers(),
        )
        client.close()

        assert response.status_code == 409
        assert "already has documents" in response.text
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)
        api._runtime.clear()
        api._runtime.update(old_runtime)


def test_list_projects_counts_legacy_docs_for_default_project(kb_root, monkeypatch) -> None:
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api, "QdrantVectorStore", _DummyQdrantVectorStore)
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop_lifespan)

    old_jobs = dict(api._jobs)
    old_runtime = dict(api._runtime)
    api._jobs.clear()
    api._runtime.clear()

    try:
        slug = _seed_project_state(
            kb_root=kb_root,
            docs=[{"doc_id": "legacy-1"}],
        )

        client = TestClient(api.app)
        response = client.get("/projects", headers=_auth_headers())
        client.close()

        assert response.status_code == 200
        payload = response.json()
        projects = payload.get("projects", [])
        target = next((p for p in projects if p.get("slug") == slug), None)
        assert target is not None
        assert target.get("doc_count") == 1
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)
        api._runtime.clear()
        api._runtime.update(old_runtime)


