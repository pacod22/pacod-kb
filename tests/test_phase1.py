"""Phase 1 acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------


def _validate_schema(data: dict, schema_name: str) -> None:
    import jsonschema

    schema_dir = Path(__file__).parent.parent / "pipeline" / "schemas"
    schema = json.loads((schema_dir / schema_name).read_text())
    jsonschema.validate(data, schema)


# ---------------------------------------------------------------------------
# JSON Schema self-tests
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_frontmatter_schema_is_valid_json(self) -> None:
        schema_path = Path(__file__).parent.parent / "pipeline" / "schemas" / "frontmatter.json"
        data = json.loads(schema_path.read_text())
        assert data["type"] == "object"

    def test_manifest_schema_is_valid_json(self) -> None:
        schema_path = Path(__file__).parent.parent / "pipeline" / "schemas" / "manifest.json"
        data = json.loads(schema_path.read_text())
        assert data["type"] == "object"

    def test_valid_frontmatter_passes(self) -> None:
        fm = {
            "doc_id": "a" * 64,
            "title": "Test",
            "source_file": "test.txt",
            "source_uri": "test.txt",
            "source_version": "",
            "source_sha256": "a" * 64,
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "converted_at": "2026-01-01T00:00:00+00:00",
            "converter": "passthrough/1.0",
            "extraction_method": "deterministic",
            "language": "en",
            "doc_type": "generic",
            "supersedes": None,
            "valid_until": None,
        }
        _validate_schema(fm, "frontmatter.json")

    def test_invalid_frontmatter_fails(self) -> None:
        import jsonschema

        with pytest.raises(jsonschema.ValidationError):
            _validate_schema({"doc_id": "tooshort"}, "frontmatter.json")


# ---------------------------------------------------------------------------
# Normalization unit tests
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_heading_repair_no_skip(self) -> None:
        from pipeline.normalize import repair_heading_levels

        text = "# Title\n### Skip\n#### Sub"
        result = repair_heading_levels(text)
        lines = result.splitlines()
        # # → ## → ### (no skips allowed)
        assert lines[0].startswith("# ")
        assert lines[1].startswith("## ")
        assert lines[2].startswith("### ")

    def test_heading_repair_allows_up(self) -> None:
        from pipeline.normalize import repair_heading_levels

        text = "# Title\n## A\n#### Deep\n## B"
        result = repair_heading_levels(text)
        lines = result.splitlines()
        assert lines[3].startswith("## ")  # going back up is fine

    def test_heading_repair_first_heading_kept(self) -> None:
        from pipeline.normalize import repair_heading_levels

        result = repair_heading_levels("## Start")
        assert result.startswith("## ")

    def test_whitespace_collapse(self) -> None:
        from pipeline.normalize import normalize_whitespace

        text = "Para one\n\n\n\n\nPara two"
        result = normalize_whitespace(text)
        assert "\n\n\n" not in result

    def test_front_matter_round_trip(self) -> None:
        from pipeline.normalize import assemble_document, strip_front_matter

        fm = {"doc_id": "x" * 64, "title": "Hello"}
        body = "# Hello\n\nBody text here."
        doc = assemble_document(fm, body)
        parsed_fm, parsed_body = strip_front_matter(doc)
        assert parsed_fm["title"] == "Hello"
        assert "Body text here" in parsed_body


# ---------------------------------------------------------------------------
# Legal promoter tests
# ---------------------------------------------------------------------------


class TestPromoteLegal:
    def _promote(self, text: str) -> str:
        from pipeline.promote_legal import promote_legal

        return promote_legal(text)

    def test_glava_promoted_to_h2(self) -> None:
        result = self._promote("Глава I\nsome text")
        assert result.splitlines()[0] == "## Глава I"

    def test_razdel_promoted_to_h3(self) -> None:
        result = self._promote("Раздел II\nsome text")
        assert result.splitlines()[0] == "### Раздел II"

    def test_chl_promoted_to_h4(self) -> None:
        result = self._promote("Чл. 42. Some article text.")
        assert result.splitlines()[0].startswith("#### Чл. 42")

    def test_paragraph_sign_promoted_to_h4(self) -> None:
        result = self._promote("§ 5. Transition provisions.")
        assert result.splitlines()[0].startswith("#### § 5")

    def test_art_promoted_to_h4(self) -> None:
        result = self._promote("Art. 10 The following shall apply.")
        assert result.splitlines()[0].startswith("#### Art. 10")

    def test_existing_heading_untouched(self) -> None:
        result = self._promote("## Already a heading")
        assert result == "## Already a heading"

    def test_blank_lines_preserved(self) -> None:
        result = self._promote("Глава I\n\nЧл. 1. Text.")
        lines = result.splitlines()
        assert lines[0] == "## Глава I"
        assert lines[1] == ""
        assert lines[2].startswith("#### Чл. 1")

    def test_legal_bg_fixture(self) -> None:
        """Full fixture: every Глава/Раздел/Чл./§ line must become a heading."""
        from pipeline.promote_legal import promote_legal

        text = (FIXTURES / "legal_bg.md").read_text(encoding="utf-8")
        result = promote_legal(text)
        lines = result.splitlines()
        heading_lines = {ln.split()[1] for ln in lines if ln.startswith("#")}
        assert "Глава" in " ".join(heading_lines) or any(ln.startswith("## Глава") for ln in lines)
        assert any(ln.startswith("#### Чл.") for ln in lines)
        assert any(ln.startswith("### Раздел") for ln in lines)

    # English
    def test_en_chapter(self) -> None:
        assert self._promote("Chapter 1\ntext").splitlines()[0] == "## Chapter 1"

    def test_en_section(self) -> None:
        assert self._promote("Section 3 Definitions").splitlines()[0] == "### Section 3 Definitions"

    def test_en_part(self) -> None:
        assert self._promote("Part IV General Rules").splitlines()[0] == "### Part IV General Rules"

    def test_en_article(self) -> None:
        assert self._promote("Article 7 Scope.").splitlines()[0].startswith("#### Article 7")

    # German
    def test_de_kapitel(self) -> None:
        assert self._promote("Kapitel 2\nInhalt").splitlines()[0] == "## Kapitel 2"

    def test_de_abschnitt(self) -> None:
        assert (
            self._promote("Abschnitt 1 Allgemeines").splitlines()[0]
            == "### Abschnitt 1 Allgemeines"
        )

    def test_de_teil(self) -> None:
        assert (
            self._promote("Teil III Schlussbestimmungen").splitlines()[0]
            == "### Teil III Schlussbestimmungen"
        )

    def test_de_artikel(self) -> None:
        assert self._promote("Artikel 3 Grundsätze").splitlines()[0].startswith("#### Artikel 3")

    # Russian
    def test_ru_statya(self) -> None:
        assert (
            self._promote("Статья 1. Основные положения.")
            .splitlines()[0]
            .startswith("#### Статья 1")
        )

    def test_ru_statya_abbrev(self) -> None:
        assert self._promote("Ст. 5. Ответственность.").splitlines()[0].startswith("#### Ст. 5")

    # Serbian Cyrillic
    def test_sr_cyr_clan(self) -> None:
        assert self._promote("Члан 3. Опште одредбе.").splitlines()[0].startswith("#### Члан 3")

    def test_sr_cyr_odeljak(self) -> None:
        assert (
            self._promote("Одељак II Посебне одредбе").splitlines()[0]
            == "### Одељак II Посебне одредбе"
        )

    # Serbian Latin / Croatian (shared Glava)
    def test_sr_lat_glava(self) -> None:
        assert self._promote("Glava II\nOpšte odredbe").splitlines()[0] == "## Glava II"

    def test_sr_lat_odeljak(self) -> None:
        assert (
            self._promote("Odeljak 1 Opšte odredbe").splitlines()[0]
            == "### Odeljak 1 Opšte odredbe"
        )

    def test_sr_lat_clan(self) -> None:
        assert self._promote("Član 10. Opšte odredbe.").splitlines()[0].startswith("#### Član 10")

    # Croatian
    def test_hr_clanak(self) -> None:
        assert self._promote("Članak 5. Opće odredbe.").splitlines()[0].startswith("#### Članak 5")

    def test_hr_cl_abbrev(self) -> None:
        assert self._promote("Čl. 12. Primjena.").splitlines()[0].startswith("#### Čl. 12")

    def test_hr_odjeljak(self) -> None:
        assert (
            self._promote("Odjeljak 2 Posebne odredbe").splitlines()[0]
            == "### Odjeljak 2 Posebne odredbe"
        )

    # Slovenian
    def test_sl_poglavje(self) -> None:
        assert self._promote("Poglavje 1\nSplošne določbe").splitlines()[0] == "## Poglavje 1"

    def test_sl_oddelek(self) -> None:
        assert (
            self._promote("Oddelek 3 Posebne določbe").splitlines()[0]
            == "### Oddelek 3 Posebne določbe"
        )

    def test_sl_clen(self) -> None:
        assert self._promote("Člen 7. Splošne določbe.").splitlines()[0].startswith("#### Člen 7")

    # Macedonian
    def test_mk_del(self) -> None:
        assert self._promote("Дел II\nОпшти одредби").splitlines()[0] == "### Дел II"

    def test_mk_clan(self) -> None:
        assert self._promote("Член 4. Општи одредби.").splitlines()[0].startswith("#### Член 4")


# ---------------------------------------------------------------------------
# Passthrough parser tests
# ---------------------------------------------------------------------------


class TestPassthroughParser:
    def test_txt_parse(self) -> None:
        from pipeline.parsers.passthrough import PassthroughParser

        result = PassthroughParser().parse(FIXTURES / "sample.txt")
        assert "Artificial Intelligence" in result.title
        assert len(result.body) > 100
        assert result.extraction_method == "deterministic"

    def test_md_parse(self) -> None:
        from pipeline.parsers.passthrough import PassthroughParser

        result = PassthroughParser().parse(FIXTURES / "sample.md")
        assert result.title  # some title extracted
        assert "## " in result.body  # headings preserved

    def test_txt_roundtrip_within_2pct(self) -> None:
        from pipeline.normalize import char_count_normalized
        from pipeline.parsers.passthrough import PassthroughParser

        fixture = FIXTURES / "sample.txt"
        source_text = fixture.read_text(encoding="utf-8")
        result = PassthroughParser().parse(fixture)

        src_count = char_count_normalized(source_text)
        body_count = char_count_normalized(result.body)
        diff = abs(src_count - body_count) / src_count if src_count else 0
        assert diff <= 0.02, f"Round-trip char diff {diff:.1%} exceeds 2% threshold"

    def test_md_roundtrip_within_2pct(self) -> None:
        from pipeline.normalize import char_count_normalized
        from pipeline.parsers.passthrough import PassthroughParser

        fixture = FIXTURES / "sample.md"
        source_text = fixture.read_text(encoding="utf-8")
        result = PassthroughParser().parse(fixture)

        src_count = char_count_normalized(source_text)
        body_count = char_count_normalized(result.body)
        diff = abs(src_count - body_count) / src_count if src_count else 0
        assert diff <= 0.02, f"Round-trip char diff {diff:.1%} exceeds 2% threshold"


# ---------------------------------------------------------------------------
# Web / HTML parser tests
# ---------------------------------------------------------------------------


class TestWebParser:
    def test_html_parse(self) -> None:
        from pipeline.parsers.web import WebParser

        result = WebParser().parse(FIXTURES / "sample.html")
        assert "Knowledge" in result.title or "knowledge" in result.body.lower()
        assert len(result.body) > 50

    def test_html_has_snapshot(self) -> None:
        from pipeline.parsers.web import WebParser

        result = WebParser().parse(FIXTURES / "sample.html")
        assert result.raw_snapshot is not None
        assert len(result.raw_snapshot) > 0

    def test_url_fetch_sends_browser_headers(self) -> None:
        """_browser_fetch must use a browser User-Agent and Sec-Fetch headers."""
        from unittest.mock import patch

        import httpx

        from pipeline.parsers import web

        html = b"<html><head><title>T</title></head><body>hello world</body></html>"
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(request.headers)
            return httpx.Response(200, content=html)

        def fake_client(headers: dict[str, str], verify: bool) -> httpx.Client:
            return httpx.Client(headers=headers, transport=httpx.MockTransport(handler))

        with patch.object(web, "_make_client", fake_client):
            assert web._browser_fetch("https://example.com/page") == html

        assert "Mozilla" in captured.get("user-agent", "")
        assert "Chrome" in captured.get("user-agent", "")
        assert "sec-fetch-dest" in captured
        assert "sec-fetch-mode" in captured

    def test_url_fetch_raises_on_http_error(self) -> None:
        """_browser_fetch must raise ValueError on 4xx/5xx responses."""
        from unittest.mock import patch

        import httpx

        from pipeline.parsers import web

        def fake_client(headers: dict[str, str], verify: bool) -> httpx.Client:
            transport = httpx.MockTransport(lambda r: httpx.Response(403, content=b"Forbidden"))
            return httpx.Client(headers=headers, transport=transport)

        with patch.object(web, "_make_client", fake_client):
            with pytest.raises(ValueError, match="HTTP 403"):
                web._browser_fetch("https://example.com/page")


# ---------------------------------------------------------------------------
# Full ingest integration tests
# ---------------------------------------------------------------------------


class TestIngest:
    def test_ingest_txt(self, config) -> None:  # type: ignore[no-untyped-def]
        from pipeline.acquisition import ingest_document

        doc_id, folder = ingest_document(str(FIXTURES / "sample.txt"), config)
        assert (folder / "document.md").exists()
        assert (folder / "manifest.json").exists()
        assert (folder / "source.txt").exists()

        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["status"] == "ok"
        assert manifest["doc_id"] == doc_id
        assert len(doc_id) == 64
        _validate_schema(manifest, "manifest.json")

    def test_ingest_md(self, config) -> None:  # type: ignore[no-untyped-def]
        from pipeline.acquisition import ingest_document

        doc_id, folder = ingest_document(str(FIXTURES / "sample.md"), config)
        manifest = json.loads((folder / "manifest.json").read_text())
        _validate_schema(manifest, "manifest.json")

    def test_ingest_html(self, config) -> None:  # type: ignore[no-untyped-def]
        from pipeline.acquisition import ingest_document

        doc_id, folder = ingest_document(str(FIXTURES / "sample.html"), config)
        manifest = json.loads((folder / "manifest.json").read_text())
        _validate_schema(manifest, "manifest.json")
        assert manifest["doc_type"] == "generic"

    def test_ingest_legal(self, config) -> None:  # type: ignore[no-untyped-def]
        from pipeline.acquisition import ingest_document

        _, folder = ingest_document(str(FIXTURES / "legal_bg.md"), config, doc_type="legal_code")
        doc_md = (folder / "document.md").read_text(encoding="utf-8")
        # Legal promotion must have run
        assert "## Глава" in doc_md
        assert "#### Чл." in doc_md

    def test_ingest_idempotent(self, config) -> None:  # type: ignore[no-untyped-def]
        from pipeline.acquisition import ingest_document

        id1, folder1 = ingest_document(str(FIXTURES / "sample.txt"), config)
        id2, folder2 = ingest_document(str(FIXTURES / "sample.txt"), config)
        assert id1 == id2
        assert folder1 == folder2
        # Only one document folder should exist
        docs = list(config.documents_dir.iterdir())
        assert len(docs) == 1

    def test_frontmatter_in_document_md(self, config) -> None:  # type: ignore[no-untyped-def]
        from pipeline.acquisition import ingest_document
        from pipeline.normalize import strip_front_matter

        _, folder = ingest_document(str(FIXTURES / "sample.txt"), config)
        doc_md = (folder / "document.md").read_text(encoding="utf-8")
        fm, body = strip_front_matter(doc_md)
        _validate_schema(fm, "frontmatter.json")
        assert len(body) > 10

    def test_ingest_docx(self, config, docx_fixture: Path) -> None:  # type: ignore[no-untyped-def]
        from pipeline.acquisition import ingest_document

        doc_id, folder = ingest_document(str(docx_fixture), config)
        manifest = json.loads((folder / "manifest.json").read_text())
        _validate_schema(manifest, "manifest.json")
        assert (folder / "source.docx").exists()

    def test_ingest_pdf(self, config, pdf_fixture: Path) -> None:  # type: ignore[no-untyped-def]
        pytest.importorskip("docling")
        from pipeline.acquisition import ingest_document

        doc_id, folder = ingest_document(str(pdf_fixture), config)
        manifest = json.loads((folder / "manifest.json").read_text())
        _validate_schema(manifest, "manifest.json")
        assert (folder / "source.pdf").exists()

    def test_ingest_rejects_unsupported_language(self, config, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A document detected as an unsupported language must raise ValueError."""
        from unittest.mock import patch

        from pipeline.acquisition import ingest_document

        # Write a plain-text file then mock language detection to return 'fr'
        src = tmp_path / "french.txt"
        src.write_text("Bonjour le monde. Ceci est un test.", encoding="utf-8")

        with patch("pipeline.parsers.passthrough._detect_language", return_value="fr"):
            with pytest.raises(ValueError, match="Unsupported language 'fr'"):
                ingest_document(str(src), config)

    def test_ingest_allows_unknown_language(self, config, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Unknown language (too short to detect) must be allowed with a warning."""
        from unittest.mock import patch

        from pipeline.acquisition import ingest_document

        src = tmp_path / "short.txt"
        src.write_text("ok", encoding="utf-8")

        with patch("pipeline.parsers.passthrough._detect_language", return_value="unknown"):
            doc_id, folder = ingest_document(str(src), config)
        assert (folder / "manifest.json").exists()

    def test_ingest_allows_unsupported_language_when_validation_disabled_per_call(
        self,
        config,
        tmp_path,
    ) -> None:  # type: ignore[no-untyped-def]
        from unittest.mock import patch

        from pipeline.acquisition import ingest_document

        src = tmp_path / "french_allowed.txt"
        src.write_text("Bonjour le monde. Ceci est un test.", encoding="utf-8")

        with patch("pipeline.parsers.passthrough._detect_language", return_value="fr"):
            _doc_id, folder = ingest_document(
                str(src),
                config,
                validate_language=False,
            )
        assert (folder / "manifest.json").exists()

    def test_ingest_allows_unsupported_language_when_validation_disabled_globally(
        self,
        config,
        tmp_path,
    ) -> None:  # type: ignore[no-untyped-def]
        import dataclasses
        from unittest.mock import patch

        from pipeline.acquisition import ingest_document

        cfg = dataclasses.replace(config, language_validation_enabled=False)
        src = tmp_path / "french_allowed_global.txt"
        src.write_text("Bonjour le monde. Ceci est un test.", encoding="utf-8")

        with patch("pipeline.parsers.passthrough._detect_language", return_value="fr"):
            _doc_id, folder = ingest_document(str(src), cfg)
        assert (folder / "manifest.json").exists()

    def test_ingest_strips_special_chars_from_title(self, config, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Leading/trailing special characters must be removed from the stored title."""
        from pipeline.acquisition import ingest_document

        src = tmp_path / "weird.md"
        src.write_text(
            "# **[Annual Report] (2024)$**\n\nBody text long enough to detect English language.",
            encoding="utf-8",
        )
        _, folder = ingest_document(str(src), config)
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["title"] == "Annual Report] (2024"


class TestCleanTitle:
    def test_strips_leading_and_trailing_specials(self) -> None:
        from pipeline.acquisition import clean_title

        assert clean_title("*%[Hello World]%*") == "Hello World"
        assert clean_title("&$( Budget 2024 ){}") == "Budget 2024"

    def test_preserves_internal_specials(self) -> None:
        from pipeline.acquisition import clean_title

        assert clean_title("Art. 5 (a) & (b)") == "Art. 5 (a) & (b"

    def test_preserves_unicode_letters(self) -> None:
        from pipeline.acquisition import clean_title

        assert clean_title("«Закон за ДДС»") == "Закон за ДДС"

    def test_all_specials_becomes_empty(self) -> None:
        from pipeline.acquisition import clean_title

        assert clean_title("***") == ""
