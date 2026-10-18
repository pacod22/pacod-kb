"""PostgreSQL keyword store: analysed (OR) search, ordering, as_of. Needs KB_TEST_PG_URL.

    KB_TEST_PG_URL=postgresql+psycopg2://kb:kb@localhost:5432/kb uv run pytest tests/test_pg_fts.py
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from pipeline.chunker import Chunk
from pipeline.indexer import PostgresFtsDb

pytestmark = pytest.mark.skipif(
    not os.environ.get("KB_TEST_PG_URL"), reason="needs KB_TEST_PG_URL (a PostgreSQL database)"
)


def _chunk(cid: str, text: str, valid_until: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=cid.split("#")[0], heading_path=["Doc"], heading_level=1,
        parent_chunk=None, sequence=0, byte_start=0, byte_end=len(text.encode()),
        embedded_text=text, valid_until=valid_until, extraction_method="deterministic",
        confidence=1.0, text=text,
    )


@pytest.fixture()
def pg() -> Iterator[PostgresFtsDb]:
    from sqlalchemy import create_engine, text

    engine = create_engine(os.environ["KB_TEST_PG_URL"])
    with engine.begin() as con:
        con.execute(text("DROP TABLE IF EXISTS kb_chunks_fts"))
    db = PostgresFtsDb(engine)
    db.upsert_batch(
        [
            _chunk("a#s#0", "Законите за защита на личните данни се прилагат"),
            _chunk("b#s#0", "Закон за данъците данъците данъците данъците"),
            _chunk("c#s#0", "The data subject may obtain erasure of personal data"),
            _chunk("d#s#0", "Чл. 32 Личният живот е неприкосновен"),
            _chunk("e#s#0", "Old erasure rules for personal data", valid_until="2020-01-01"),
        ],
        "p",
    )
    yield db
    with engine.begin() as con:
        con.execute(text("DROP TABLE IF EXISTS kb_chunks_fts"))
    engine.dispose()


def _ids(db: PostgresFtsDb, query: str, **kw: Any) -> list[str]:
    return [cid for cid, _ in db.search(query, 10, **kw)]


def test_legacy_still_requires_every_word(pg: PostgresFtsDb) -> None:
    assert _ids(pg, "What is erasure of personal data") == []


def test_or_finds_partial_matches(pg: PostgresFtsDb) -> None:
    assert _ids(pg, "What is erasure of personal data", mode="or")[0] == "c#s#0"


def test_prefix_matches_inflected_forms(pg: PostgresFtsDb) -> None:
    assert {"a#s#0", "b#s#0", "d#s#0"} <= set(_ids(pg, "закона за личния живот", mode="or"))


def test_more_matched_terms_rank_first_despite_repetition(pg: PostgresFtsDb) -> None:
    # b repeats one term; a matches every term once: no IDF, so count terms first.
    assert _ids(pg, "законите личните данни", mode="or")[0] == "a#s#0"


def test_article_phrase(pg: PostgresFtsDb) -> None:
    assert _ids(pg, "чл. 32", mode="or")[0] == "d#s#0"


def test_as_of_and_current_filters(pg: PostgresFtsDb) -> None:
    assert "e#s#0" not in _ids(pg, "erasure personal data", mode="or")
    assert "e#s#0" in _ids(pg, "erasure personal data", mode="or", as_of="2019-06-01")
    assert "e#s#0" not in _ids(pg, "erasure personal data", mode="or", as_of="2020-01-01")
    assert pg.valid_until_values_after("2019-12-31") == ["2020-01-01"]
