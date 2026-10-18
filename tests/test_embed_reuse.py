"""Unchanged chunks keep their vectors on supersession and rebuild (EMBED_REUSE)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from pipeline.indexer import FtsDb, index_document
from pipeline.vectorstore import QdrantVectorStore
from tests.test_phase3 import MockEmbedder


class _CountingEmbedder(MockEmbedder):
    def __init__(self, model_name: str | None = "test-model") -> None:
        if model_name is not None:
            self.model_name = model_name
        self.embedded: list[str] = []

    def encode(self, texts: list[str], *, progress_cb: Any = None) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().encode(texts, progress_cb=progress_cb)


_SECTIONS = [
    f"## Section {i}\n\nParagraph {i} about topic {i} with enough words." for i in range(6)
]


def _body(changed: int | None = None) -> str:
    parts = list(_SECTIONS)
    if changed is not None:
        parts[changed] = f"## Section {changed}\n\nRewritten paragraph {changed}, new wording."
    return "# Act\n\n" + "\n\n".join(parts) + "\n"


def _index(config: Any, emb: Any, vstore: Any, fts: Any, doc_id: str, body: str) -> list[Any]:
    manifest = {"doc_id": doc_id, "title": "Act", "status": "ok", "project": "p"}
    return index_document(None, config, emb, vstore, fts, doc_data=(manifest, body))


@pytest.fixture()
def stores(tmp_path: Path) -> Any:
    vstore = QdrantVectorStore(QdrantClient(":memory:"), "c")
    fts = FtsDb(tmp_path / "fts5.db")
    yield vstore, fts
    fts.close()


def test_a_new_version_embeds_only_changed_chunks(config: Any, stores: Any) -> None:
    vstore, fts = stores
    emb = _CountingEmbedder()
    first = _index(config, emb, vstore, fts, "a" * 64, _body())
    assert len(emb.embedded) == len(first)

    emb.embedded.clear()
    second = _index(config, emb, vstore, fts, "b" * 64, _body(changed=2))
    assert len(second) == len(first)
    assert len(emb.embedded) == 1
    assert "Rewritten paragraph 2" in emb.embedded[0]

    # Reused vectors are the stored ones: the unchanged chunk ranks the same either way.
    old = {c.text: c.chunk_id for c in first}
    new = {c.text: c.chunk_id for c in second}
    text = next(t for t in old if "Paragraph 4" in t)
    v_old = vstore.client.retrieve("c", [_uuid(old[text])], with_vectors=True)[0].vector
    v_new = vstore.client.retrieve("c", [_uuid(new[text])], with_vectors=True)[0].vector
    assert v_old == pytest.approx(v_new)


def _uuid(chunk_id: str) -> str:
    from pipeline.vectorstore import _chunk_id_to_uuid

    return _chunk_id_to_uuid(chunk_id)


def test_vectors_of_another_model_are_not_reused(config: Any, stores: Any) -> None:
    vstore, fts = stores
    _index(config, _CountingEmbedder("model-a"), vstore, fts, "a" * 64, _body())
    emb = _CountingEmbedder("model-b")
    chunks = _index(config, emb, vstore, fts, "b" * 64, _body(changed=2))
    assert len(emb.embedded) == len(chunks)


def test_untagged_vectors_need_the_opt_in(config: Any, stores: Any) -> None:
    vstore, fts = stores
    _index(config, _CountingEmbedder(None), vstore, fts, "a" * 64, _body())  # no model tag

    emb = _CountingEmbedder()
    chunks = _index(config, emb, vstore, fts, "b" * 64, _body(changed=2))
    assert len(emb.embedded) == len(chunks)

    emb.embedded.clear()
    opted_in = dataclasses.replace(config, embed_reuse_untagged=True)
    _index(opted_in, emb, vstore, fts, "c" * 64, _body(changed=3))
    assert len(emb.embedded) == 1


def test_reuse_can_be_switched_off(config: Any, stores: Any) -> None:
    vstore, fts = stores
    emb = _CountingEmbedder()
    _index(config, emb, vstore, fts, "a" * 64, _body())
    emb.embedded.clear()
    off = dataclasses.replace(config, embed_reuse=False)
    chunks = _index(off, emb, vstore, fts, "b" * 64, _body(changed=2))
    assert len(emb.embedded) == len(chunks)


def test_rebuild_snapshot_feeds_reindexing(config: Any, stores: Any) -> None:
    vstore, fts = stores
    emb = _CountingEmbedder()
    first = _index(config, emb, vstore, fts, "a" * 64, _body())
    snapshot = vstore.snapshot_vectors(model="test-model")
    assert len(snapshot) == len(first)

    vstore.drop_all()
    emb.embedded.clear()
    manifest = {"doc_id": "a" * 64, "title": "Act", "status": "ok", "project": "p"}
    again = index_document(
        None, config, emb, vstore, fts, doc_data=(manifest, _body()), vector_cache=snapshot
    )
    assert len(again) == len(first)
    assert emb.embedded == []


def test_rebuilding_twice_embeds_nothing_the_second_time(config: Any, stores: Any) -> None:
    """End to end through rebuild_index (filesystem archive): unchanged chunks keep vectors."""
    from pipeline.acquisition import ingest_document
    from pipeline.versioning import rebuild_index

    vstore, fts = stores
    src = config.kb_root / "act.md"
    src.write_text(_body(), encoding="utf-8")
    emb = _CountingEmbedder()
    _, folder = ingest_document(str(src), config, llm_convert=False)
    first = index_document(folder, config, emb, vstore, fts)
    assert len(emb.embedded) == len(first)

    emb.embedded.clear()
    assert rebuild_index(config, emb, vstore, fts) == (1, 0)
    assert emb.embedded == []
    assert vstore.client.count("c").count == len(first)

    emb.embedded.clear()
    assert rebuild_index(config, emb, vstore, fts, reuse_vectors=False) == (1, 0)
    assert len(emb.embedded) == len(first)
