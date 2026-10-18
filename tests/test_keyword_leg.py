"""The analysed keyword leg: query analysis and the OR / AND-then-OR searches."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.chunker import Chunk
from pipeline.indexer import FtsDb, build_fts5_match, build_pg_tsquery
from pipeline.text_lexicon import QueryTerm, analyze_query, light_stem


def _terms(query: str) -> set[tuple[str, bool]]:
    return {(t.text, t.prefix) for t in analyze_query(query).terms}


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("законите", ("закон", True)),
        ("личните", ("личн", True)),
        ("персональных", ("персональн", True)),
        ("personenbezogener", ("personenbezogen", True)),
        ("obradom", ("obrad", True)),
        ("processing", ("process", True)),
        ("data", ("data", False)),
        ("leak", ("leak", False)),
    ],
)
def test_light_stem(word: str, expected: tuple[str, bool]) -> None:
    assert light_stem(word) == expected


def test_question_words_are_dropped_and_numbers_kept() -> None:
    kq = analyze_query("What does Article 17 say?")
    assert ("17", False) in {(t.text, t.prefix) for t in kq.terms}
    assert not {"what", "does"} & {t.text for t in kq.terms}
    assert kq.phrases == (("article", "17"),)


def test_article_abbreviations_become_phrases() -> None:
    assert analyze_query("чл. 32 от Конституцията").phrases == (("чл", "32"),)
    assert ("чл", False) in _terms("чл. 32")


def test_acronyms_survive_even_when_they_spell_a_stopword() -> None:
    assert {("dsgvo", False), ("it", False)} <= _terms("DSGVO und IT")
    assert ("und", False) not in _terms("DSGVO und IT")


def test_only_stopwords_leaves_nothing() -> None:
    assert analyze_query("what is the").empty
    assert build_fts5_match("what is the", "or") is None
    assert build_pg_tsquery("what is the") is None


def test_yo_adds_the_ye_spelling() -> None:
    texts = {t.text for t in analyze_query("актёров").terms}
    assert {"актёр", "актер"} <= texts


def test_lowercase_not_casefold() -> None:
    # The index lower-cases; casefold would turn ß into ss, which it never contains.
    assert QueryTerm("straß", True) in analyze_query("Straße").terms


def test_builders() -> None:
    assert build_fts5_match('a "quoted" (query)', "legacy") == '"a" "quoted" "query"'
    assert build_fts5_match("законите за данни", "or") == '"закон"* OR "данн"*'
    assert build_fts5_match("законите за данни", "and") == '"закон"* AND "данн"*'
    assert build_fts5_match("данни", "and") is None  # one term: the OR search covers it
    any_q, parts = build_pg_tsquery("чл. 32 законите") or ("", [])
    assert parts == ["'чл'", "'32'", "'закон':*", "'чл' <-> '32'"]
    assert any_q == "('чл') | ('32') | ('закон':*) | ('чл' <-> '32')"


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=cid.split("#")[0], heading_path=["Doc"], heading_level=1,
        parent_chunk=None, sequence=0, byte_start=0, byte_end=len(text.encode()),
        embedded_text=text, valid_until=None, extraction_method="deterministic",
        confidence=1.0, text=text,
    )


@pytest.fixture()
def fts(tmp_path: Path) -> FtsDb:
    db = FtsDb(tmp_path / "fts5.db")
    db.upsert_batch(
        [
            _chunk("a#s#0", "Законите за защита на личните данни се прилагат"),
            _chunk("b#s#0", "Закон за данъците"),
            _chunk("c#s#0", "The data subject may obtain erasure of personal data"),
            _chunk("d#s#0", "Чл. 32 Личният живот е неприкосновен"),
        ],
        "p",
    )
    return db


def test_legacy_requires_every_word(fts: FtsDb) -> None:
    assert fts.search("What is erasure of personal data", 10) == []


def test_or_mode_finds_partial_matches(fts: FtsDb) -> None:
    ids = [c for c, _ in fts.search("What is erasure of personal data", 10, mode="or")]
    assert ids[0] == "c#s#0"


def test_or_mode_matches_inflected_forms_by_prefix(fts: FtsDb) -> None:
    ids = {c for c, _ in fts.search("закона за личния живот", 10, mode="or")}
    assert {"a#s#0", "b#s#0", "d#s#0"} <= ids


def test_article_phrase_finds_the_article(fts: FtsDb) -> None:
    ids = [c for c, _ in fts.search("чл. 32", 10, mode="or")]
    assert ids[0] == "d#s#0"


def test_and_or_ranks_all_term_matches_first(fts: FtsDb) -> None:
    ids = [c for c, _ in fts.search("законите личните данни", 10, mode="and_or")]
    assert ids[0] == "a#s#0"
    assert "b#s#0" in ids  # OR matches follow


def test_nothing_searchable_returns_empty(fts: FtsDb) -> None:
    assert fts.search("what is the", 10, mode="or") == []
    assert fts.search("what is the", 10, mode="and_or") == []


def test_postgres_analysed_search_fails_open_on_timeout() -> None:
    """A statement timeout (OperationalError) leaves the query vector-only, not failed."""
    from sqlalchemy.exc import OperationalError

    from pipeline.indexer import PostgresFtsDb

    class _Engine:
        def begin(self) -> None:
            raise OperationalError("SELECT", {}, Exception("canceling statement due to timeout"))

    db = PostgresFtsDb.__new__(PostgresFtsDb)
    db._engine = _Engine()  # type: ignore[assignment]
    db._timeout_ms = 1
    assert db.search("erasure of personal data", 10, mode="or") == []


def test_unknown_keyword_mode_is_refused_at_startup(
    monkeypatch: pytest.MonkeyPatch, kb_root: Path
) -> None:
    from pipeline.config import load_config

    monkeypatch.setenv("KB_ROOT", str(kb_root))
    monkeypatch.setenv("KEYWORD_QUERY_MODE", "fuzzy")
    with pytest.raises(SystemExit, match="KEYWORD_QUERY_MODE"):
        load_config()
    monkeypatch.setenv("KEYWORD_QUERY_MODE", "and_or")
    assert load_config().keyword_query_mode == "and_or"
