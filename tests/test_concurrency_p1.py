"""Regression tests for P1 concurrency/correctness fixes."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from pipeline import api


@pytest.fixture()
def clean_jobs() -> Iterator[None]:
    saved_jobs, saved_runtime = dict(api._jobs), dict(api._runtime)
    api._jobs.clear()
    yield
    api._jobs.clear()
    api._jobs.update(saved_jobs)
    api._runtime.clear()
    api._runtime.update(saved_runtime)


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------


def test_concurrent_job_mutation_is_safe(clean_jobs: None) -> None:
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            while not stop.is_set():
                jid = api._create_job("t", {})
                api._update_job_progress(jid, current=1, total=2, message="x")
                api._finish_job(jid, result={})
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        api._job_busy()
        sorted(api._job_snapshot(), key=lambda j: str(j["created_at"]))
    stop.set()
    for t in threads:
        t.join()
    assert not errors
    assert len(api._jobs) <= api._MAX_JOBS + 4


def test_eviction_never_drops_live_jobs(clean_jobs: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_MAX_JOBS", 3)
    live = [api._create_job("t", {}) for _ in range(3)]
    api._set_running(live[0])
    for _ in range(5):
        api._finish_job(api._create_job("t", {}), result={})
    for jid in live:
        assert jid in api._jobs
    assert api._get_job(live[0])["status"] == "running"  # type: ignore[index]


def test_pending_job_blocks_switch(clean_jobs: None) -> None:
    api._create_job("ingest_url", {})
    assert api._job_busy()
    with pytest.raises(HTTPException) as exc:
        with api._exclusive():
            pass
    assert exc.value.status_code == 409


def test_restart_unknown_type_creates_no_orphan(clean_jobs: None) -> None:
    with pytest.raises(HTTPException):
        api._restart_job({"type": "mystery", "params": {}}, background_tasks=None, request=None)  # type: ignore[arg-type]
    with pytest.raises(HTTPException):
        api._restart_job({"type": "reindex", "params": {}}, background_tasks=None, request=None)  # type: ignore[arg-type]
    assert api._jobs == {}


def test_stale_jobs_marked_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from pipeline.db_models import Base, KbJob

    engine = create_engine(f"sqlite:///{tmp_path}/jobs.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    with factory.begin() as s:
        for jid, status in (("a", "running"), ("b", "pending"), ("c", "done")):
            s.add(KbJob(job_id=jid, type="t", status=status, created_at="2026", params_json="{}"))
    monkeypatch.setattr(api, "_job_session_factory", factory)
    assert api._fail_stale_jobs() == 2
    with factory() as s:
        statuses = dict(s.execute(select(KbJob.job_id, KbJob.status)).all())
    assert statuses == {"a": "error", "b": "error", "c": "done"}


# ---------------------------------------------------------------------------
# Settings: switching collection
# ---------------------------------------------------------------------------


def test_switch_to_missing_collection_is_refused(
    clean_jobs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qdrant_client import QdrantClient

    from pipeline.vectorstore import QdrantVectorStore

    @asynccontextmanager
    async def _noop(_: Any) -> Any:
        yield

    client = QdrantClient(":memory:")
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api.app.router, "lifespan_context", _noop)
    api.app.state.vector_store = QdrantVectorStore(client, "existing")
    with TestClient(api.app, headers={"X-API-Key": "test-secret"}) as c:
        resp = c.post("/settings", json={"active_collection": "typo"})
        assert resp.status_code == 404
        assert not client.collection_exists("typo")
        api._create_job("ingest_url", {})
        resp = c.post("/settings", json={"active_collection": "existing"})
        assert resp.status_code == 409
    client.close()


# ---------------------------------------------------------------------------
# Indexing atomicity
# ---------------------------------------------------------------------------


class _MockEmbedder:
    VECTOR_DIM = 1024

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def encode(
        self, texts: list[str], *, progress_cb: Callable[[int, int], None] | None = None
    ) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding backend down")
        return [[0.01 * (i + 1)] * self.VECTOR_DIM for i in range(len(texts))]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


_EN = (
    "\n\nThis paragraph is written in plain English so that language detection "
    "is reliable for the knowledge base regression tests.\n"
)


def test_failed_reindex_keeps_previous_index(config: Any, tmp_path: Path) -> None:
    from qdrant_client import QdrantClient

    from pipeline.acquisition import ingest_document
    from pipeline.indexer import index_document, make_fts_db
    from pipeline.vectorstore import QdrantVectorStore

    src = tmp_path / "d.md"
    src.write_text("# Doc\n\nbody" + _EN, encoding="utf-8")
    _, folder = ingest_document(str(src), config)
    client = QdrantClient(":memory:")
    vs = QdrantVectorStore(client, "kb")
    fts = make_fts_db(config)
    try:
        index_document(folder, config, _MockEmbedder(), vs, fts)
        before = client.count("kb").count
        assert before > 0
        with pytest.raises(RuntimeError):
            index_document(folder, config, _MockEmbedder(fail=True), vs, fts)
        assert client.count("kb").count == before
    finally:
        fts.close()
        client.close()


def test_quarantined_doc_is_purged_from_index(config: Any, tmp_path: Path) -> None:
    from qdrant_client import QdrantClient

    from pipeline.acquisition import ingest_document
    from pipeline.indexer import index_document, make_fts_db
    from pipeline.vectorstore import QdrantVectorStore

    src = tmp_path / "q.md"
    src.write_text("# Q\n\nbody" + _EN, encoding="utf-8")
    _, folder = ingest_document(str(src), config)
    assert folder is not None
    client = QdrantClient(":memory:")
    vs = QdrantVectorStore(client, "kb")
    fts = make_fts_db(config)
    try:
        index_document(folder, config, _MockEmbedder(), vs, fts)
        mp = folder / "manifest.json"
        m = json.loads(mp.read_text())
        m["status"] = "review"
        mp.write_text(json.dumps(m))
        assert index_document(folder, config, _MockEmbedder(), vs, fts) == []
        assert client.count("kb").count == 0
    finally:
        fts.close()
        client.close()


# ---------------------------------------------------------------------------
# Dropzone
# ---------------------------------------------------------------------------


def _age(path: Path) -> None:
    old = time.time() - 60
    os.utime(path, (old, old))


def test_dropzone_ignores_dotfiles_and_reindexes_unindexed(config: Any) -> None:
    from qdrant_client import QdrantClient

    from pipeline.acquisition import ingest_document
    from pipeline.dropzone import scan_dropzone
    from pipeline.indexer import make_fts_db
    from pipeline.vectorstore import QdrantVectorStore

    dz = config.kb_dropzone_dir
    dz.mkdir(parents=True, exist_ok=True)
    (dz / ".DS_Store").write_bytes(b"\x00\x01")
    _age(dz / ".DS_Store")

    # Acquired earlier but never indexed (chunk_count 0)
    doc = dz / "late.md"
    doc.write_text("# Late\n\nindex me" + _EN, encoding="utf-8")
    ingest_document(str(doc), config)
    _age(doc)

    client = QdrantClient(":memory:")
    fts = make_fts_db(config)
    try:
        results = scan_dropzone(config, _MockEmbedder(), QdrantVectorStore(client, "kb"), fts)
    finally:
        fts.close()
    assert [r["file"] for r in results] == ["late.md"]
    assert results[0]["status"] == "ingested" and results[0]["chunks"] > 0
    assert (dz / ".DS_Store").exists()
    assert client.count("kb").count > 0
    client.close()


# ---------------------------------------------------------------------------
# Database backend parity
# ---------------------------------------------------------------------------


def test_db_store_round_trips_meta_fields_and_truncates(tmp_path: Path) -> None:
    from sqlalchemy import create_engine

    from pipeline.document_store import DatabaseDocumentStore

    store = DatabaseDocumentStore(create_engine(f"sqlite:///{tmp_path}/kb.db"))
    doc_id = "ab" * 32
    fm = {
        "doc_id": doc_id,
        "title": "T" * 900,
        "source_file": "a.html",
        "source_uri": "https://example.com/a",
        "source_sha256": doc_id,
        "language": "en",
        "meta_author": "Ada",
        "meta_published_date": "2026-01-02",
        "meta_publication_venue": "Example Times",
    }
    store.save(doc_id, fm, "# T\n\nbody\n", b"x", ".html")
    m = store.load_manifest(doc_id)
    assert m["meta_author"] == "Ada"
    assert m["meta_publication_venue"] == "Example Times"
    assert len(m["title"]) == 512
    store.update_manifest(doc_id, meta_author="Grace", chunk_count=3)
    m = store.load_manifest(doc_id)
    assert (m["meta_author"], m["meta_published_date"], m["chunk_count"]) == (
        "Grace",
        "2026-01-02",
        3,
    )
    with pytest.raises(KeyError):
        store.load_manifest("cd" * 32)
    with pytest.raises(KeyError):
        store.load_body("cd" * 32)


# ---------------------------------------------------------------------------
# _job_slot (shared background-job wrapper)
# ---------------------------------------------------------------------------


def test_job_slot_success_runs_and_releases(clean_jobs: None) -> None:
    jid = api._create_job("t", {})
    with api._job_slot(jid) as acquired:
        assert acquired and api._get_job(jid)["status"] == "running"  # type: ignore[index]
        api._finish_job(jid, result={"ok": True})
    assert not api._mutation_lock.locked()
    assert api._get_job(jid)["status"] == "done"  # type: ignore[index]


def test_job_slot_records_error_and_releases(clean_jobs: None) -> None:
    jid = api._create_job("t", {})
    with api._job_slot(jid) as acquired:
        assert acquired
        raise RuntimeError("boom")
    assert not api._mutation_lock.locked()
    job = api._get_job(jid)
    assert job is not None and job["status"] == "error" and job["error"] == "boom"


def test_job_slot_busy_skips_body(clean_jobs: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the lock timeout the job fails (it waits that long first)."""
    monkeypatch.setattr(api, "_JOB_LOCK_TIMEOUT_SECS", 0.01)
    jid = api._create_job("t", {})
    api._mutation_lock.acquire()
    try:
        ran = False
        with api._job_slot(jid) as acquired:
            if acquired:
                ran = True
        assert not ran
        assert api._get_job(jid)["error"] == "another job is already running"  # type: ignore[index]
    finally:
        api._mutation_lock.release()


def test_jobs_queue_in_order_instead_of_failing(clean_jobs: None) -> None:
    """A job submitted while another runs waits its turn (it used to fail at once)."""
    import threading

    api._shutdown_jobs()
    release = threading.Event()
    order: list[str] = []

    def runner(jid: str, name: str, wait: bool) -> None:
        with api._job_slot(jid) as acquired:
            if not acquired:
                return
            order.append(name)
            if wait:
                release.wait(5)
            api._finish_job(jid, result={})

    first, second = api._create_job("t", {}), api._create_job("t", {})
    pool = api.ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(runner, first, "first", True)
        pool.submit(runner, second, "second", False)
        for _ in range(100):
            if api._get_job(first)["status"] == "running":  # type: ignore[index]
                break
            threading.Event().wait(0.01)
        assert api._get_job(second)["status"] == "pending"  # type: ignore[index]
        release.set()
        pool.shutdown(wait=True)
    finally:
        release.set()
    assert order == ["first", "second"]
    assert api._get_job(second)["status"] == "done"  # type: ignore[index]
