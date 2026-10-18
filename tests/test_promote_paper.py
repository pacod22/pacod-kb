"""Unit tests for pipeline/promote_paper.py."""

from __future__ import annotations

import pytest

from pipeline.promote_paper import promote_paper


class TestPromotePaper:
    def test_abstract_promoted(self) -> None:
        body = "Abstract\n\nThis paper studies X."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Abstract"

    def test_abstract_with_colon(self) -> None:
        body = "Abstract:\n\nThis paper studies X."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Abstract"

    def test_introduction_promoted(self) -> None:
        body = "Introduction\n\nWe present a new approach."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Introduction"

    def test_methods_promoted(self) -> None:
        body = "Methods\n\nWe used dataset A."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Methods"

    def test_materials_and_methods_promoted(self) -> None:
        body = "Materials and Methods\n\nSamples were collected."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Materials and Methods"

    def test_results_promoted(self) -> None:
        body = "Results\n\nFigure 1 shows the accuracy."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Results"

    def test_discussion_promoted(self) -> None:
        body = "Discussion\n\nOur findings suggest."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Discussion"

    def test_conclusion_promoted(self) -> None:
        body = "Conclusion\n\nWe demonstrated X."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Conclusion"

    def test_references_promoted(self) -> None:
        body = "References\n\n[1] Doe et al."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## References"

    def test_bibliography_promoted(self) -> None:
        body = "Bibliography\n\n[1] Doe et al."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Bibliography"

    def test_acknowledgments_promoted(self) -> None:
        body = "Acknowledgments\n\nWe thank our reviewers."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Acknowledgments"

    def test_appendix_promoted(self) -> None:
        body = "Appendix\n\nAdditional figures."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Appendix"

    def test_case_insensitive_upper(self) -> None:
        body = "ABSTRACT\n\nText."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## ABSTRACT"

    def test_case_insensitive_lower(self) -> None:
        body = "abstract\n\nText."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## abstract"

    def test_existing_heading_unchanged(self) -> None:
        body = "## Already a Heading\n\nContent."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Already a Heading"

    def test_existing_h1_unchanged(self) -> None:
        body = "# Title of Paper\n\nContent."
        result = promote_paper(body)
        assert result.splitlines()[0] == "# Title of Paper"

    def test_blank_line_preserved(self) -> None:
        body = "Abstract\n\nSome text.\n\nIntroduction\n\nMore text."
        result = promote_paper(body)
        lines = result.splitlines()
        assert lines[0] == "## Abstract"
        assert lines[1] == ""
        assert lines[4] == "## Introduction"

    def test_inline_text_not_promoted(self) -> None:
        body = "The abstract of this paper describes methods."
        result = promote_paper(body)
        assert result == body

    def test_partial_match_not_promoted(self) -> None:
        """'Abstract:' with text on same line is NOT promoted (it's not a standalone header)."""
        body = "Abstract: This paper describes X."
        result = promote_paper(body)
        # Not in frozenset since it has text after the colon
        assert "##" not in result

    def test_german_zusammenfassung(self) -> None:
        body = "Zusammenfassung\n\nDiese Arbeit."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Zusammenfassung"

    def test_german_einleitung(self) -> None:
        body = "Einleitung\n\nWir präsentieren."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Einleitung"

    def test_german_referenzen(self) -> None:
        body = "Referenzen\n\n[1] Mueller et al."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Referenzen"

    def test_bulgarian_abstract(self) -> None:
        body = "Абстракт\n\nТова изследване."
        result = promote_paper(body)
        assert result.splitlines()[0] == "## Абстракт"

    def test_full_paper_structure(self) -> None:
        body = (
            "# A Study of Things\n\n"
            "Abstract\n\n"
            "We study things.\n\n"
            "Keywords: things, stuff\n\n"
            "Introduction\n\n"
            "Background information.\n\n"
            "Methods\n\n"
            "We used method A.\n\n"
            "Results\n\n"
            "Figure 1.\n\n"
            "Discussion\n\n"
            "Findings suggest.\n\n"
            "Conclusion\n\n"
            "We conclude.\n\n"
            "References\n\n"
            "[1] Doe 2020."
        )
        result = promote_paper(body)
        lines = result.splitlines()
        assert lines[0] == "# A Study of Things"
        promoted = [ln for ln in lines if ln.startswith("## ")]
        assert "## Abstract" in promoted
        assert "## Introduction" in promoted
        assert "## Methods" in promoted
        assert "## Results" in promoted
        assert "## Discussion" in promoted
        assert "## Conclusion" in promoted
        assert "## References" in promoted

    def test_empty_body(self) -> None:
        assert promote_paper("") == ""

    def test_body_unchanged_no_headers(self) -> None:
        body = "This is plain text without any section headers."
        assert promote_paper(body) == body


@pytest.mark.parametrize(
    "header",
    [
        "abstract",
        "keywords",
        "key words",
        "introduction",
        "background",
        "methods",
        "methodology",
        "materials and methods",
        "results",
        "findings",
        "discussion",
        "conclusion",
        "conclusions",
        "summary",
        "acknowledgments",
        "acknowledgements",
        "references",
        "bibliography",
        "appendix",
        "appendices",
    ],
)
def test_all_english_headers_promoted(header: str) -> None:
    body = f"{header.title()}\n\nContent."
    result = promote_paper(body)
    assert result.startswith("## "), f"Header {header!r} was not promoted"
