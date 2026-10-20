"""Regression tests for P0 project-isolation and supersession fixes.

Covers: per-project rebuild, default-project tagging for dropzone/CLI-style
ingests, the legacy ``"default"`` re-tag migration, and watch supersession
safety (quarantine + content revert).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

URL = "https://example.com/watched"


class MockEmbedder:
    VECTOR_DIM = 1024

    def encode(
        self, texts: list[str], *, progress_cb: Callable[[int, int], None] | None = None
    ) -> list[list[float]]:
        import random

        return [
            [random.Random(hash(t) ^ i).gauss(0, 1) for i in range(self.VECTOR_DIM)] for t in texts
        ]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


@pytest.fixture()
def qdrant() -> Iterator[Any]:
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture()
def fts(config: Any) -> Iterator[Any]:
    from pipeline.indexer import make_fts_db

    db = make_fts_db(config)
    yield db
    db.close()


def _vs(client: Any, name: str) -> Any:
    from pipeline.vectorstore import QdrantVectorStore

    return QdrantVectorStore(client, name)


_FILLER = (
    "\n\nThis paragraph is written in plain English so that language detection "
    "is reliable, and it describes the document for the knowledge base tests.\n"
)


def _src(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text + _FILLER, encoding="utf-8")
    return p


def _fts_rows(fts: Any, project: str) -> int:
    row = fts._con.execute("SELECT count(*) FROM chunks_meta WHERE project = ?", (project,))
    return int(row.fetchone()[0])


def _ingest_index(
    config: Any, path: Path, vs: Any, fts: Any, project: str | None
) -> tuple[str, Path]:
    from pipeline.acquisition import ingest_document
    from pipeline.indexer import index_document

    doc_id, folder = ingest_document(str(path), config, project=project)
    assert folder is not None
    index_document(folder, config, MockEmbedder(), vs, fts)
    return doc_id, folder


# ---------------------------------------------------------------------------
# Default-project tagging
# ---------------------------------------------------------------------------


def test_ingest_without_project_uses_registry_default(config: Any, tmp_path: Path) -> None:
    from pipeline.acquisition import ingest_document
    from pipeline.projects import ProjectRegistry

    _, folder = ingest_document(str(_src(tmp_path, "a.md", "# A\n\nalpha text\n")), config)
    assert folder is not None
    manifest = json.loads((folder / "manifest.json").read_text())
    default_slug = ProjectRegistry(config.kb_root).list()[0].slug
    assert manifest["project"] == default_slug
    assert manifest["project"] != "default"


def test_dropzone_tags_requested_project(config: Any, qdrant: Any, fts: Any) -> None:
    import os
    import time

    from pipeline.dropzone import scan_dropzone

    dz = config.kb_dropzone_dir
    dz.mkdir(parents=True, exist_ok=True)
    f = dz / "note.md"
    f.write_text("# Note\n\nbeta project content\n" + _FILLER, encoding="utf-8")
    old = time.time() - 60
    os.utime(f, (old, old))

    results = scan_dropzone(config, MockEmbedder(), _vs(qdrant, "beta"), fts, project="beta")
    assert results[0]["status"] == "ingested"
    manifests = [json.loads(p.read_text()) for p in config.documents_dir.glob("*/manifest.json")]
    assert [m["project"] for m in manifests] == ["beta"]
    assert _fts_rows(fts, "beta") > 0
    assert qdrant.count("beta").count > 0


# ---------------------------------------------------------------------------
# Per-project rebuild
# ---------------------------------------------------------------------------


def test_rebuild_only_touches_target_project(
    config: Any, qdrant: Any, fts: Any, tmp_path: Path
) -> None:
    from pipeline.versioning import rebuild_index

    alpha, beta = _vs(qdrant, "alpha"), _vs(qdrant, "beta")
    _ingest_index(config, _src(tmp_path, "a.md", "# A\n\nalpha words\n"), alpha, fts, "alpha")
    _ingest_index(config, _src(tmp_path, "b.md", "# B\n\nbeta words\n"), beta, fts, "beta")
    beta_points = qdrant.count("beta").count
    beta_rows = _fts_rows(fts, "beta")

    ok, err = rebuild_index(
        config, MockEmbedder(), alpha, fts, project="alpha", default_project="alpha"
    )
    assert (ok, err) == (1, 0)
    assert qdrant.count("alpha").count > 0
    assert qdrant.count("beta").count == beta_points  # untouched
    assert _fts_rows(fts, "beta") == beta_rows  # untouched
    # alpha's collection holds only alpha's document
    points, _ = qdrant.scroll("alpha", limit=100, with_payload=True)
    assert {p.payload["project"] for p in points} == {"alpha"}


def test_rebuild_all_projects_keeps_collections_separate(
    config: Any, qdrant: Any, fts: Any, tmp_path: Path
) -> None:
    from pipeline.projects import ProjectRegistry
    from pipeline.versioning import rebuild_all_projects

    registry = ProjectRegistry(config.kb_root)
    a = registry.create(name="Alpha", collection_name="alpha")
    b = registry.create(name="Beta", collection_name="beta")
    _ingest_index(
        config, _src(tmp_path, "a.md", "# A\n\nalpha\n"), _vs(qdrant, "alpha"), fts, a.slug
    )
    _ingest_index(config, _src(tmp_path, "b.md", "# B\n\nbeta\n"), _vs(qdrant, "beta"), fts, b.slug)

    ok, err = rebuild_all_projects(config, MockEmbedder(), qdrant, fts, registry)
    assert (ok, err) == (2, 0)
    for coll, slug in (("alpha", a.slug), ("beta", b.slug)):
        points, _ = qdrant.scroll(coll, limit=100, with_payload=True)
        assert points and {p.payload["project"] for p in points} == {slug}


# ---------------------------------------------------------------------------
# Legacy "default" re-tag migration
# ---------------------------------------------------------------------------


def test_retag_legacy_default_is_idempotent(
    config: Any, qdrant: Any, fts: Any, tmp_path: Path
) -> None:
    from pipeline.document_store import FilesystemDocumentStore
    from pipeline.normalize import strip_front_matter
    from pipeline.projects import ProjectRegistry, retag_legacy_default_project

    registry = ProjectRegistry(config.kb_root)
    default = registry.ensure_default(config.qdrant_collection)
    vs = _vs(qdrant, default.collection_name)
    _, folder = _ingest_index(
        config, _src(tmp_path, "l.md", "# Legacy\n\nold words\n"), vs, fts, "default"
    )
    assert _fts_rows(fts, "default") > 0
    store = FilesystemDocumentStore(config)

    first = retag_legacy_default_project(registry, default.slug, store, fts, vs)
    assert first["documents"] == 1 and first["fts_rows"] > 0
    assert json.loads((folder / "manifest.json").read_text())["project"] == default.slug
    front, _ = strip_front_matter((folder / "document.md").read_text())
    assert front["project"] == default.slug
    assert _fts_rows(fts, "default") == 0
    points, _ = qdrant.scroll(default.collection_name, limit=100, with_payload=True)
    assert {p.payload["project"] for p in points} == {default.slug}

    second = retag_legacy_default_project(registry, default.slug, store, fts, vs)
    assert second == {"documents": 0, "fts_rows": 0}


def test_retag_skipped_when_real_default_project_exists(config: Any, fts: Any) -> None:
    from unittest.mock import MagicMock

    from pipeline.projects import ProjectRegistry, retag_legacy_default_project

    registry = ProjectRegistry(config.kb_root)
    registry.ensure_default(config.qdrant_collection)
    registry.create(name="default", collection_name="real-default")
    store = MagicMock()
    counts = retag_legacy_default_project(registry, "kb_chunks", store, fts)
    assert counts == {"documents": 0, "fts_rows": 0}
    store.list_all.assert_not_called()


# ---------------------------------------------------------------------------
# Watch supersession safety
# ---------------------------------------------------------------------------


def _watch_file(config: Any) -> Path:
    wf = config.kb_root / "watch.json"
    wf.write_text(json.dumps([{"url": URL, "doc_type": "web_page"}]))
    return wf


def _set_source_uri(folder: Path, url: str) -> None:
    mp = folder / "manifest.json"
    m = json.loads(mp.read_text())
    m["source_uri"] = url
    mp.write_text(json.dumps(m))


def test_quarantined_new_version_does_not_supersede(
    config: Any, qdrant: Any, fts: Any, tmp_path: Path
) -> None:
    from pipeline.watch import watch_sources

    vs = _vs(qdrant, "kb")
    prior_id, prior = _ingest_index(config, _src(tmp_path, "v1.md", "# V1\n\nv1\n"), vs, fts, None)
    _set_source_uri(prior, URL)
    new_id, new = _ingest_index(config, _src(tmp_path, "v2.md", "# V2\n\nv2\n"), vs, fts, None)
    m = json.loads((new / "manifest.json").read_text())
    m["status"] = "review"
    (new / "manifest.json").write_text(json.dumps(m))

    with patch("pipeline.watch.ingest_document", return_value=(new_id, new)):
        with patch("pipeline.watch.index_document", return_value=[]):
            results = watch_sources(config, _watch_file(config), MockEmbedder(), vs, fts)

    assert results[0]["status"] == "review"
    assert json.loads((prior / "manifest.json").read_text())["valid_until"] is None


def test_reverted_content_reinstates_old_version(
    config: Any, qdrant: Any, fts: Any, tmp_path: Path
) -> None:
    from pipeline.versioning import supersede_document
    from pipeline.watch import watch_sources

    vs = _vs(qdrant, "kb")
    a_id, a = _ingest_index(config, _src(tmp_path, "a.md", "# A\n\nversion a\n"), vs, fts, None)
    b_id, b = _ingest_index(config, _src(tmp_path, "b.md", "# B\n\nversion b\n"), vs, fts, None)
    _set_source_uri(a, URL)
    _set_source_uri(b, URL)
    supersede_document(a_id, "2026-01-01", config, vs, fts)  # history: A → B

    # Source reverts to A's bytes: sha256 dedup hands back the old document.
    with patch("pipeline.watch.ingest_document", return_value=(a_id, a)):
        results = watch_sources(config, _watch_file(config), MockEmbedder(), vs, fts)

    assert results[0]["status"] == "updated"
    assert json.loads((a / "manifest.json").read_text())["valid_until"] is None
    assert json.loads((b / "manifest.json").read_text())["valid_until"] is not None
    current = fts._con.execute(
        "SELECT DISTINCT doc_id FROM chunks_meta WHERE valid_until IS NULL"
    ).fetchall()
    assert [r[0] for r in current] == [a_id]


def test_corrupt_projects_json_is_backed_up_not_overwritten(config: Any) -> None:
    from pipeline.projects import ProjectRegistry

    path = config.kb_root / "projects.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        ProjectRegistry(config.kb_root)
    assert path.read_text() == "{not json"
    assert list(config.kb_root.glob("projects.json.corrupt-*"))


# ---------------------------------------------------------------------------
# Real API lifespan: legacy re-tag migration runs at startup
# ---------------------------------------------------------------------------


def test_lifespan_retags_legacy_default_documents(
    config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from pipeline import api
    from pipeline.acquisition import ingest_document

    _, folder = ingest_document(
        str(_src(tmp_path, "old.md", "# Old\n\nlegacy doc\n")), config, project="default"
    )
    assert folder is not None
    assert not (config.kb_root / "projects.json").exists()  # startup creates the default

    class _Embedder(MockEmbedder):
        def __init__(self, _model: str) -> None:
            pass

    monkeypatch.setattr(api, "make_embedder", lambda _config: _Embedder("mock"))
    monkeypatch.setattr(api, "_PIPELINE_API_SECRET", "test-secret")
    monkeypatch.setattr(api, "_DROPZONE_INTERVAL", 0)
    saved_runtime, saved_jobs = dict(api._runtime), dict(api._jobs)
    try:
        with TestClient(api.app, headers={"X-API-Key": "test-secret"}) as c:
            docs = c.get("/documents").json()["documents"]
            active = api._runtime["active_project"]
            rebuild_resp = c.post("/rebuild")
    finally:
        api._runtime.clear()
        api._runtime.update(saved_runtime)
        api._jobs.clear()
        api._jobs.update(saved_jobs)

    assert active == "kb_chunks"
    assert [d["project"] for d in docs] == ["kb_chunks"]
    assert json.loads((folder / "manifest.json").read_text())["project"] == "kb_chunks"
    assert rebuild_resp.status_code == 200


def test_restart_of_legacy_rebuild_job_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from pipeline import api

    monkeypatch.setitem(api._runtime, "active_project", "alpha")
    assert api._rebuild_params({}) == {"project": "alpha", "all_projects": False}
    monkeypatch.setitem(api._runtime, "active_project", "")
    with pytest.raises(HTTPException) as exc:
        api._rebuild_params({})
    assert exc.value.status_code == 409
    assert api._rebuild_params({"all_projects": True})["all_projects"] is True
