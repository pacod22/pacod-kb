from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import BackgroundTasks
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_jobs_persist_to_database(tmp_path: Path) -> None:
    from pipeline import api
    from pipeline.db_models import Base

    engine = create_engine(f"sqlite:///{tmp_path}/jobs.db")
    Base.metadata.create_all(engine)

    old_factory = api._job_session_factory
    old_jobs = dict(api._jobs)
    try:
        api._jobs.clear()
        api._job_session_factory = sessionmaker(engine)

        job_id = api._create_job("ingest_url", {"url": "https://example.com"})
        api._set_running(job_id)
        api._update_job_progress(job_id, current=1, total=2, message="halfway")
        api._finish_job(job_id, result={"doc_id": "abc"})

        loaded = api._load_job_from_db(job_id)
        assert loaded is not None
        assert loaded["status"] == "done"
        assert loaded["type"] == "ingest_url"
        assert loaded["params"]["url"] == "https://example.com"
        assert loaded["result"]["doc_id"] == "abc"
        assert loaded["progress"]["percent"] == 50

        listed = api._list_jobs_from_db(limit=10)
        assert any(j["job_id"] == job_id for j in listed)
    finally:
        api._job_session_factory = old_factory
        api._jobs.clear()
        api._jobs.update(old_jobs)


def test_delete_job_from_db_removes_row(tmp_path: Path) -> None:
    from pipeline import api
    from pipeline.db_models import Base

    engine = create_engine(f"sqlite:///{tmp_path}/jobs_delete.db")
    Base.metadata.create_all(engine)

    old_factory = api._job_session_factory
    old_jobs = dict(api._jobs)
    try:
        api._jobs.clear()
        api._job_session_factory = sessionmaker(engine)

        job_id = api._create_job("watch", {"watch_file": None})
        assert api._load_job_from_db(job_id) is not None
        assert api._delete_job_from_db(job_id) is True
        assert api._load_job_from_db(job_id) is None
    finally:
        api._job_session_factory = old_factory
        api._jobs.clear()
        api._jobs.update(old_jobs)


def test_restart_job_ingest_file_uses_replay_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline import api

    class _Capture:
        """The job queue, recording what is submitted instead of running it."""

        def __init__(self) -> None:
            self.tasks: list[Any] = []

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> None:
            self.tasks.append(SimpleNamespace(func=fn, args=args, kwargs=kwargs))

    tasks = _Capture()
    monkeypatch.setattr(api, "_job_executor", lambda: tasks)

    old_jobs = dict(api._jobs)
    replay_path = tmp_path / "job_replays" / "old-job" / "doc.pdf"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_bytes(b"replay-bytes")

    params = {
        "filename": "doc.pdf",
        "doc_type": "generic",
        "source_version": "",
        "supersedes": None,
        "skip_index": False,
        "trusted": False,
        "json_convert": True,
        "llm_convert": None,
        "validate_language": None,
        "replay_relpath": "job_replays/old-job/doc.pdf",
    }
    job = {"type": "ingest_file", "params": params}
    request: Any = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(kb_root=tmp_path),
                embedder=object(),
                vector_store=object(),
                document_store=object(),
            )
        )
    )

    try:
        api._jobs.clear()
        new_job_id = api._restart_job(job, background_tasks=BackgroundTasks(), request=request)
        assert new_job_id in api._jobs
        assert len(tasks.tasks) == 1
        tmp_path_str = str(tasks.tasks[0].args[1])
        assert Path(tmp_path_str).name == "doc.pdf"
        assert Path(tmp_path_str).read_bytes() == b"replay-bytes"
    finally:
        for t in getattr(tasks, "tasks", []):
            if len(t.args) > 1:
                shutil.rmtree(Path(str(t.args[1])).parent, ignore_errors=True)
        api._jobs.clear()
        api._jobs.update(old_jobs)


def test_delete_job_replay_artifact_removes_parent_dir(tmp_path: Path) -> None:
    from pipeline import api

    artifact = tmp_path / "job_replays" / "dead-job" / "source.pdf"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"bytes")
    config: Any = SimpleNamespace(kb_root=tmp_path)

    api._delete_job_replay_artifact(
        config,
        {"replay_relpath": "job_replays/dead-job/source.pdf"},
    )
    assert not artifact.exists()
    assert not artifact.parent.exists()


