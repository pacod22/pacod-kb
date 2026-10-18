"""Tests for the regenerate-from-vector-db and convert-without-index operations."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

from sqlalchemy import create_engine

from pipeline import api

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _noop_lifespan(_: Any):
    yield


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test-secret"}


class _DummyStore:
    def __init__(self) -> None:
        self.storage_backend = "filesystem"

    def list_all(self) -> list[dict[str, Any]]:
        return []


class _FakeQdrantClient:
    """Minimal Qdrant client stub for tests."""

    def __init__(self, points: list[dict[str, Any]]) -> None:
        self._points = points  # list of payload dicts

    def scroll(
        self,
        collection_name: str,
        limit: int = 1000,
        offset: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[Any], None]:
        # Return all points in a single page regardless of limit
        records = []
        for p in self._points:
            rec = MagicMock()
            rec.payload = p
            records.append(rec)
        return records, None


class _FakeVectorStore:
    def __init__(self, client: Any, collection_name: str = "test-col") -> None:
        self.client = client
        self.collection_name = collection_name


# ─────────────────────────────────────────────────────────────────────────────
# Regenerate-from-vector-db tests
# ─────────────────────────────────────────────────────────────────────────────


def _run_regen_sync(
    job_id: str,
    project_slug: str | None,
    config: Any,
    vector_store: Any,
    store: Any,
) -> None:
    """Run regenerate task synchronously (bypasses BackgroundTasks)."""
    api._run_regenerate_from_vector_db(job_id, project_slug, config, vector_store, store)


def test_regenerate_collects_unique_docs(kb_root, monkeypatch, config):
    """_run_regenerate_from_vector_db aggregates chunk counts per doc_id."""
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    old_jobs = dict(api._jobs)
    api._jobs.clear()
    try:
        # Two chunks for doc1, one for doc2
        qdrant_points = [
            {"source_id": "doc1", "title": "Doc One"},
            {"source_id": "doc1", "title": "Doc One"},
            {"source_id": "doc2", "title": "Doc Two"},
        ]
        client = _FakeQdrantClient(qdrant_points)
        vs = _FakeVectorStore(client)
        store = _DummyStore()

        job_id = api._create_job("regenerate_from_vector_db", {})
        _run_regen_sync(job_id, None, config, vs, store)

        job = api._jobs[job_id]
        assert job["status"] == "done", job["error"]
        result = job["result"]
        assert result is not None
        assert result["doc_count"] == 2
        docs_by_id = {d["doc_id"]: d for d in result["documents"]}
        assert docs_by_id["doc1"]["chunk_count"] == 2
        assert docs_by_id["doc2"]["chunk_count"] == 1
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)


def test_regenerate_handles_empty_collection(kb_root, monkeypatch, config):
    """_run_regenerate_from_vector_db returns doc_count=0 for an empty collection."""
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    old_jobs = dict(api._jobs)
    api._jobs.clear()
    try:
        client = _FakeQdrantClient([])
        vs = _FakeVectorStore(client)
        store = _DummyStore()

        job_id = api._create_job("regenerate_from_vector_db", {})
        _run_regen_sync(job_id, None, config, vs, store)

        job = api._jobs[job_id]
        assert job["status"] == "done", job["error"]
        assert job["result"]["doc_count"] == 0
        assert job["result"]["documents"] == []
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)


def test_regenerate_skips_points_without_source_id(kb_root, monkeypatch, config):
    """Points lacking source_id in payload are silently ignored."""
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    old_jobs = dict(api._jobs)
    api._jobs.clear()
    try:
        qdrant_points = [
            {"title": "No source_id here"},          # missing source_id
            {"source_id": "", "title": "Empty ID"},   # empty source_id
            {"source_id": "valid-doc", "title": "OK"},
        ]
        client = _FakeQdrantClient(qdrant_points)
        vs = _FakeVectorStore(client)
        store = _DummyStore()

        job_id = api._create_job("regenerate_from_vector_db", {})
        _run_regen_sync(job_id, None, config, vs, store)

        job = api._jobs[job_id]
        assert job["status"] == "done", job["error"]
        assert job["result"]["doc_count"] == 1
        assert job["result"]["documents"][0]["doc_id"] == "valid-doc"
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)


def test_regenerate_persists_docs_to_database_project(kb_root, monkeypatch, config):
    """Database-mode regeneration persists recovered docs and assigns the target project."""
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    old_jobs = dict(api._jobs)
    api._jobs.clear()
    try:
        from pipeline.document_store import DatabaseDocumentStore
        from pipeline.projects import DatabaseProjectRegistry

        db_config = replace(config, storage_backend="database")
        engine = create_engine(f"sqlite:///{kb_root}/regen.db")
        store = DatabaseDocumentStore(engine)
        registry = DatabaseProjectRegistry(store._factory)
        project = registry.create(name="Recovered Docs", collection_name="recovered-col")

        qdrant_points = [
            {
                "source_id": "doc1",
                "title": "Recovered Doc",
                "source_type": "report",
                "language": "en",
                "url": "https://example.com/recovered",
                "extraction_method": "deterministic",
                "sequence": 0,
                "byte_start": 0,
                "raw_text": "First paragraph.",
            },
            {
                "source_id": "doc1",
                "title": "Recovered Doc",
                "source_type": "report",
                "language": "en",
                "url": "https://example.com/recovered",
                "extraction_method": "deterministic",
                "sequence": 1,
                "byte_start": 16,
                "raw_text": "Second paragraph.",
            },
        ]
        client = _FakeQdrantClient(qdrant_points)
        vs = _FakeVectorStore(client, collection_name=project.collection_name)

        job_id = api._create_job("regenerate_from_vector_db", {})
        _run_regen_sync(job_id, project.slug, db_config, vs, store)

        job = api._jobs[job_id]
        assert job["status"] == "done", job["error"]
        result = job["result"]
        assert result is not None
        assert result["doc_count"] == 1
        assert result["persisted"] == 1
        assert result["updated"] == 0

        manifest = store.load_manifest("doc1")
        assert manifest["project"] == project.slug
        assert manifest["chunk_count"] == 2
        assert manifest["title"] == "Recovered Doc"
        assert manifest["source_uri"] == "https://example.com/recovered"
        assert any(e["event"] == "regenerated_from_vector_db" for e in manifest["events"])

        body = store.load_body("doc1")
        assert "First paragraph." in body
        assert "Second paragraph." in body
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)


# ─────────────────────────────────────────────────────────────────────────────
# Convert-without-index tests
# ─────────────────────────────────────────────────────────────────────────────


def _run_convert_sync(
    job_id: str,
    project_slug: str | None,
    config: Any,
    embedder: Any,
    store: Any,
) -> None:
    api._run_convert_without_index(job_id, project_slug, config, embedder, store)


def test_convert_without_index_empty_dropzone(kb_root, monkeypatch, config):
    """_run_convert_without_index returns zeroes when dropzone is empty."""
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    old_jobs = dict(api._jobs)
    api._jobs.clear()
    try:
        # Ensure dropzone directory exists but is empty
        config.kb_dropzone_dir.mkdir(parents=True, exist_ok=True)
        store = _DummyStore()

        job_id = api._create_job("convert_without_index", {})
        _run_convert_sync(job_id, None, config, None, store)

        job = api._jobs[job_id]
        assert job["status"] == "done", job["error"]
        result = job["result"]
        assert result is not None
        assert result["converted"] == 0
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)


def test_convert_without_index_converts_text_file(kb_root, monkeypatch, config):
    """_run_convert_without_index parses a .md file and saves JSON to converted_staging/."""
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    old_jobs = dict(api._jobs)
    api._jobs.clear()
    try:
        # Put a markdown file in the dropzone
        dropzone = config.kb_dropzone_dir
        dropzone.mkdir(parents=True, exist_ok=True)
        (dropzone / "test_doc.md").write_text(
            "---\ntitle: Test Doc\n---\n\nHello world.", encoding="utf-8"
        )

        store = _DummyStore()
        job_id = api._create_job("convert_without_index", {})
        _run_convert_sync(job_id, None, config, None, store)

        job = api._jobs[job_id]
        assert job["status"] == "done", job["error"]
        result = job["result"]
        assert result is not None
        assert result["converted"] == 1
        assert result["failed"] == 0
        assert result["total"] == 1

        # Verify JSON output was written
        staging = config.kb_root / "converted_staging"
        assert staging.exists()
        files = list(staging.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["source_file"] == "test_doc.md"
        assert "body" in data
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)


def test_convert_without_index_no_dropzone_dir(kb_root, monkeypatch, config):
    """_run_convert_without_index fails gracefully when dropzone directory is missing."""
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    old_jobs = dict(api._jobs)
    api._jobs.clear()
    try:
        # Make sure dropzone does NOT exist
        import shutil
        if config.kb_dropzone_dir.exists():
            shutil.rmtree(config.kb_dropzone_dir)

        store = _DummyStore()
        job_id = api._create_job("convert_without_index", {})
        _run_convert_sync(job_id, None, config, None, store)

        job = api._jobs[job_id]
        # Should fail with a clear error message
        assert job["status"] == "error"
        assert "Dropzone" in job["error"] or "dropzone" in job["error"].lower()
    finally:
        api._jobs.clear()
        api._jobs.update(old_jobs)


