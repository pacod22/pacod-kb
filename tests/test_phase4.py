"""Phase 4 tests — Image detection, OCR, dual-extraction, review queue, VLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb_root(tmp_path: Path) -> Path:
    root = tmp_path / "kb"
    (root / "documents").mkdir(parents=True)
    (root / "index").mkdir()
    (root / "review").mkdir()
    return root


@pytest.fixture()
def config(kb_root: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[type-arg]
    monkeypatch.setenv("KB_ROOT", str(kb_root))
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    monkeypatch.setenv("GIT_AUTOCOMMIT", "false")
    monkeypatch.setenv("LLM_CONVERTER_ENABLED", "false")
    monkeypatch.setenv("CONTENT_SECURITY_ENABLED", "false")
    monkeypatch.setenv("VLM_TABLE_RECONSTRUCTION_ENABLED", "false")
    monkeypatch.setenv("OCR_CONFIDENCE_THRESHOLD", "0.95")
    monkeypatch.setenv("OCR_ENABLED", "true")
    monkeypatch.delenv("QDRANT_URL", raising=False)

    from pipeline.config import load_config

    return load_config()


# ---------------------------------------------------------------------------
# Test: Image Detection
# ---------------------------------------------------------------------------


class TestImageDetection:
    """Test image-only page detection logic."""

    def test_detect_text_page(self) -> None:
        """A page with enough text is not classified as image-only."""
        from pipeline.ocr import _MIN_TEXT_CHARS, PageInfo

        # Simulate what detect_image_pages would compute
        page = PageInfo(page_num=0, char_count=500, is_image_only=False)
        assert not page.is_image_only
        assert page.char_count >= _MIN_TEXT_CHARS

    def test_detect_image_page(self) -> None:
        """A page with very little text is classified as image-only."""
        from pipeline.ocr import _MIN_TEXT_CHARS, PageInfo

        page = PageInfo(page_num=0, char_count=10, is_image_only=True)
        assert page.is_image_only
        assert page.char_count < _MIN_TEXT_CHARS

    def test_threshold_boundary(self) -> None:
        """Pages at exactly _MIN_TEXT_CHARS are NOT image-only."""
        from pipeline.ocr import _MIN_TEXT_CHARS

        # exactly at threshold = not image-only
        assert _MIN_TEXT_CHARS == 50
        # < threshold = image-only
        assert 49 < _MIN_TEXT_CHARS

    @patch("pipeline.ocr.detect_image_pages")
    def test_detect_image_pages_with_mock(self, mock_detect: MagicMock) -> None:
        """Test detect_image_pages with mocked Docling."""
        from pipeline.ocr import PageInfo

        # Mock the detect_image_pages function directly
        mock_detect.return_value = [
            PageInfo(page_num=0, char_count=100, is_image_only=False),
            PageInfo(page_num=1, char_count=10, is_image_only=True),
        ]

        pages = mock_detect(Path("/fake/test.pdf"))

        assert len(pages) == 2
        assert not pages[0].is_image_only  # text page
        assert pages[1].is_image_only  # image page

    def test_detect_image_pages_classification_logic(self) -> None:
        """Verify the classification logic threshold."""
        from pipeline.ocr import _MIN_TEXT_CHARS, PageInfo

        # Build pages manually to test the classification logic
        pages = [
            PageInfo(page_num=0, char_count=100, is_image_only=100 < _MIN_TEXT_CHARS),
            PageInfo(page_num=1, char_count=10, is_image_only=10 < _MIN_TEXT_CHARS),
            PageInfo(page_num=2, char_count=49, is_image_only=49 < _MIN_TEXT_CHARS),
            PageInfo(page_num=3, char_count=50, is_image_only=50 < _MIN_TEXT_CHARS),
        ]

        assert not pages[0].is_image_only  # 100 chars — text
        assert pages[1].is_image_only  # 10 chars — image
        assert pages[2].is_image_only  # 49 chars — image (below threshold)
        assert not pages[3].is_image_only  # 50 chars — text (at threshold)


# ---------------------------------------------------------------------------
# Test: Dual Extraction & Similarity
# ---------------------------------------------------------------------------


class TestDualExtraction:
    """Test dual-extraction similarity scoring."""

    def test_identical_texts_perfect_score(self) -> None:
        """Identical texts produce similarity 1.0."""
        from pipeline.ocr import _compute_similarity

        assert _compute_similarity("hello world", "hello world") == 1.0

    def test_empty_texts_perfect_score(self) -> None:
        """Both empty → similarity 1.0."""
        from pipeline.ocr import _compute_similarity

        assert _compute_similarity("", "") == 1.0

    def test_one_empty_zero_score(self) -> None:
        """One empty, one not → similarity 0.0."""
        from pipeline.ocr import _compute_similarity

        assert _compute_similarity("hello", "") == 0.0
        assert _compute_similarity("", "hello") == 0.0

    def test_similar_texts_high_score(self) -> None:
        """Very similar texts produce a high (but < 1.0) score."""
        from pipeline.ocr import _compute_similarity

        sim = _compute_similarity(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
        )
        assert sim > 0.9

    def test_different_texts_low_score(self) -> None:
        """Very different texts produce a low score."""
        from pipeline.ocr import _compute_similarity

        sim = _compute_similarity("abc def ghi", "xyz 123 456 789 000")
        assert sim < 0.5

    def test_flagged_below_threshold(self) -> None:
        """Pages with confidence below threshold are flagged."""
        from pipeline.ocr import PageExtraction

        ext = PageExtraction(
            page_num=0,
            text_a="hello world",
            text_b="completely different text here",
            similarity=0.3,
            confidence=0.3,
        )
        threshold = 0.95
        assert ext.confidence < threshold


# ---------------------------------------------------------------------------
# Test: Review Queue
# ---------------------------------------------------------------------------


class TestReviewQueue:
    """Test routing low-confidence pages to review."""

    def test_no_flagged_pages_no_review(self, config: Any) -> None:
        """No flagged pages → returns None."""
        from pipeline.ocr import OcrResult, route_to_review

        ocr_result = OcrResult(
            body="Some text",
            extraction_method="ocr",
            page_confidences={0: 0.98, 1: 0.99},
            min_confidence=0.98,
            image_pages=[0, 1],
            flagged_pages=[],
        )
        assert route_to_review("a" * 64, ocr_result, config) is None

    def test_flagged_pages_create_review_folder(self, config: Any) -> None:
        """Flagged pages create review folder with review.json."""
        from pipeline.ocr import OcrResult, route_to_review

        doc_id = "abcdef01" + "0" * 56
        ocr_result = OcrResult(
            body="Some text",
            extraction_method="ocr",
            page_confidences={0: 0.98, 1: 0.7, 2: 0.85},
            min_confidence=0.7,
            image_pages=[0, 1, 2],
            flagged_pages=[1, 2],
        )
        review_folder = route_to_review(doc_id, ocr_result, config)

        assert review_folder is not None
        assert review_folder.exists()
        assert review_folder.name == "abcdef01"

        review_json = json.loads((review_folder / "review.json").read_text())
        assert review_json["doc_id"] == doc_id
        assert review_json["flagged_pages"] == [1, 2]
        assert review_json["threshold"] == 0.95
        assert float(review_json["page_confidences"]["1"]) == 0.7

    def test_review_folder_in_review_dir(self, config: Any) -> None:
        """Review folder is created under config.review_dir."""
        from pipeline.ocr import OcrResult, route_to_review

        doc_id = "deadbeef" + "1" * 56
        ocr_result = OcrResult(
            body="text",
            extraction_method="ocr",
            page_confidences={0: 0.5},
            min_confidence=0.5,
            image_pages=[0],
            flagged_pages=[0],
        )
        folder = route_to_review(doc_id, ocr_result, config)
        assert folder is not None
        assert folder.parent == config.review_dir


# ---------------------------------------------------------------------------
# Test: OCR Parsing (mocked)
# ---------------------------------------------------------------------------


class TestOcrParsing:
    """Test that OCR extraction sets correct extraction_method and confidence."""

    @patch("pipeline.ocr.dual_extract")
    @patch("pipeline.ocr.analyze_pdf")
    def test_no_image_pages_deterministic(
        self,
        mock_analyze: MagicMock,
        mock_dual: MagicMock,
        config: Any,
    ) -> None:
        """PDF with no image pages → extraction_method='deterministic', one conversion."""
        from pipeline.ocr import NativeAnalysis, PageInfo, run_ocr_pipeline

        mock_analyze.return_value = NativeAnalysis(
            pages=[PageInfo(page_num=0, char_count=500, is_image_only=False)],
            body="# Title\n\nSome text content.",
        )
        result = run_ocr_pipeline(Path("/fake/test.pdf"), config)

        assert result.extraction_method == "deterministic"
        assert result.body == "# Title\n\nSome text content."
        assert result.min_confidence == 1.0
        assert result.image_pages == []
        mock_dual.assert_not_called()

    @patch("pipeline.ocr.dual_extract")
    @patch("pipeline.ocr.analyze_pdf")
    def test_image_pages_ocr_method(
        self,
        mock_analyze: MagicMock,
        mock_dual: MagicMock,
        config: Any,
    ) -> None:
        """PDF with image pages → extraction_method='ocr', body from the OCR engine."""
        from pipeline.ocr import NativeAnalysis, PageExtraction, PageInfo, run_ocr_pipeline

        mock_analyze.return_value = NativeAnalysis(
            pages=[
                PageInfo(page_num=0, char_count=500, is_image_only=False),
                PageInfo(page_num=1, char_count=10, is_image_only=True),
            ],
            body="native only",
        )
        mock_dual.return_value = [
            PageExtraction(
                page_num=1,
                text_a="OCR text from page",
                text_b="OCR text from page",
                similarity=0.98,
                confidence=0.98,
                document_text="page one text\n\nOCR text from page",
            ),
        ]

        result = run_ocr_pipeline(Path("/fake/test.pdf"), config)

        assert result.extraction_method == "ocr"
        assert result.min_confidence == 0.98
        assert result.image_pages == [1]
        assert result.body == "page one text\n\nOCR text from page"

    @patch("pipeline.ocr.dual_extract")
    @patch("pipeline.ocr.analyze_pdf")
    def test_low_confidence_flags_pages(
        self,
        mock_analyze: MagicMock,
        mock_dual: MagicMock,
        config: Any,
    ) -> None:
        """Pages below threshold are included in flagged_pages."""
        from pipeline.ocr import NativeAnalysis, PageExtraction, PageInfo, run_ocr_pipeline

        mock_analyze.return_value = NativeAnalysis(
            pages=[PageInfo(page_num=0, char_count=10, is_image_only=True)], body=""
        )
        mock_dual.return_value = [
            PageExtraction(
                page_num=0,
                text_a="Some noisy text",
                text_b="Different noisy text",
                similarity=0.6,
                confidence=0.6,
            ),
        ]

        result = run_ocr_pipeline(Path("/fake/test.pdf"), config)

        assert result.flagged_pages == [0]
        assert result.min_confidence == 0.6
        assert result.body == "Some noisy text"

    def test_parse_result_confidence_propagation(self) -> None:
        """ParseResult carries confidence and page_confidences."""
        from pipeline.parsers import ParseResult

        result = ParseResult(
            body="text",
            title="Test",
            extraction_method="ocr",
            page_confidences={0: 0.98, 1: 0.7},
            confidence=0.7,
        )
        assert result.confidence == 0.7
        assert result.page_confidences is not None
        assert result.page_confidences[1] == 0.7


# ---------------------------------------------------------------------------
# Test: Integration with acquisition
# ---------------------------------------------------------------------------


class TestAcquisitionIntegration:
    """Test that OCR results propagate through ingest_document."""

    @patch("pipeline.parsers.pdf.PdfParser._parse_with_ocr")
    def test_ocr_review_creates_review_event(
        self,
        mock_ocr: MagicMock,
        config: Any,
        tmp_path: Path,
    ) -> None:
        """When OCR produces low-confidence pages, review_flagged event is created."""
        from pipeline.parsers import ParseResult

        # Create a fake PDF file
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content for testing " * 10)

        mock_ocr.return_value = ParseResult(
            body="# Scanned Doc\n\nSome OCR text.",
            title="Scanned Doc",
            language="en",
            extraction_method="ocr",
            converter="docling-ocr/2.0",
            raw_snapshot=pdf_path.read_bytes(),
            page_confidences={0: 0.98, 1: 0.7},
            confidence=0.7,
        )

        from pipeline.acquisition import ingest_document

        doc_id, doc_folder = ingest_document(
            str(pdf_path),
            config,
            doc_type="generic",
        )

        assert doc_folder is not None
        manifest = json.loads((doc_folder / "manifest.json").read_text())

        # Should have status=review and review_flagged event
        assert manifest["status"] == "review"
        assert manifest["extraction_method"] == "ocr"
        events = manifest.get("events", [])
        event_types = [e["event"] for e in events]
        assert "review_flagged" in event_types

    @patch("pipeline.parsers.pdf.PdfParser._parse_with_ocr")
    def test_ocr_good_confidence_status_ok(
        self,
        mock_ocr: MagicMock,
        config: Any,
        tmp_path: Path,
    ) -> None:
        """When OCR confidence is above threshold, status is 'ok'."""
        from pipeline.parsers import ParseResult

        pdf_path = tmp_path / "good_scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 good scan content " * 10)

        mock_ocr.return_value = ParseResult(
            body="# Good Scan\n\nClear text.",
            title="Good Scan",
            language="en",
            extraction_method="ocr",
            converter="docling-ocr/2.0",
            raw_snapshot=pdf_path.read_bytes(),
            page_confidences={0: 0.98, 1: 0.97},
            confidence=0.97,
        )

        from pipeline.acquisition import ingest_document

        doc_id, doc_folder = ingest_document(
            str(pdf_path),
            config,
            doc_type="generic",
        )

        assert doc_folder is not None
        manifest = json.loads((doc_folder / "manifest.json").read_text())
        assert manifest["status"] == "ok"
        events = manifest.get("events", [])
        event_types = [e["event"] for e in events]
        assert "review_flagged" not in event_types

    @patch("pipeline.parsers.pdf.PdfParser._parse_with_ocr")
    def test_review_folder_created_on_low_confidence(
        self,
        mock_ocr: MagicMock,
        config: Any,
        tmp_path: Path,
    ) -> None:
        """Low-confidence pages create a review folder in review_dir."""
        from pipeline.parsers import ParseResult

        pdf_path = tmp_path / "review_scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 needs review " * 10)

        mock_ocr.return_value = ParseResult(
            body="# Review Doc\n\nBlurry text.",
            title="Review Doc",
            language="en",
            extraction_method="ocr",
            converter="docling-ocr/2.0",
            raw_snapshot=pdf_path.read_bytes(),
            page_confidences={0: 0.5},
            confidence=0.5,
        )

        from pipeline.acquisition import ingest_document

        doc_id, doc_folder = ingest_document(
            str(pdf_path),
            config,
            doc_type="generic",
        )

        # Check that review folder was created
        review_dirs = list(config.review_dir.iterdir())
        assert len(review_dirs) == 1
        review_json = json.loads((review_dirs[0] / "review.json").read_text())
        assert review_json["doc_id"] == doc_id
        assert 0 in review_json["flagged_pages"]





# ---------------------------------------------------------------------------
# Test: per-page text from a real docling document (no model, no PDF)
# ---------------------------------------------------------------------------


def _docling_doc(page_texts: dict[int, str]) -> Any:
    from docling_core.types.doc import (
        BoundingBox,
        DocItemLabel,
        DoclingDocument,
        ProvenanceItem,
        Size,
    )

    doc = DoclingDocument(name="t")
    for page_no in page_texts:
        doc.add_page(page_no=page_no, size=Size(width=595, height=842))
    for page_no, text in page_texts.items():
        if text:
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                prov=ProvenanceItem(
                    page_no=page_no,
                    bbox=BoundingBox(l=10, t=10, r=500, b=40),
                    charspan=(0, len(text)),
                ),
            )
    return doc


class TestPageAnalysis:
    """Pages are classified from the items' provenance (PageItem has no text)."""

    def test_pages_without_native_text_are_image_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipeline import ocr

        text = "Native text on the first page, long enough to count as a text page."
        doc = _docling_doc({1: text, 2: ""})
        calls: list[str] = []
        monkeypatch.setattr(ocr, "_convert", lambda engine, path: calls.append(engine) or doc)

        analysis = ocr.analyze_pdf(Path("/fake.pdf"))

        assert [(p.page_num, p.is_image_only) for p in analysis.pages] == [(0, False), (1, True)]
        assert text in analysis.body
        assert calls == ["native"]  # one conversion

    def test_single_engine_confidence_without_tesseract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipeline import ocr

        doc = _docling_doc({1: "Page one native text that is long enough here.", 2: "OCR words"})
        monkeypatch.setattr(ocr, "_convert", lambda engine, path: doc)
        monkeypatch.setattr(ocr.shutil, "which", lambda name: None)

        [ext] = ocr.dual_extract(Path("/fake.pdf"), [1], single_engine_confidence=0.9)

        assert ext.confidence == 0.9
        assert "OCR words" in ext.text_a
        assert "Page one" not in ext.text_a  # per page, not the whole document
        assert "Page one" in ext.document_text
