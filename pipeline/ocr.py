"""Phase 4: Image-only page detection, OCR, and dual-extraction confidence scoring.

Detects image-only pages in PDFs, runs OCR via Docling's OCR backend,
performs dual-extraction similarity checks, and routes low-confidence
pages to a review queue.
"""

from __future__ import annotations

import difflib
import functools
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.config import Config

logger = logging.getLogger(__name__)

# Minimum characters on a page for it to be considered "text" (not image-only)
_MIN_TEXT_CHARS = 50


@dataclass
class PageInfo:
    """Metadata about a single PDF page."""

    page_num: int  # 0-based
    char_count: int
    is_image_only: bool


@dataclass
class PageExtraction:
    """Result of dual-extraction comparison for a single page."""

    page_num: int
    text_a: str
    text_b: str
    similarity: float
    confidence: float  # = similarity (used for downstream propagation)
    document_text: str = ""  # engine A's reading of the whole document


@dataclass
class OcrResult:
    """Aggregated OCR result for an entire PDF document."""

    body: str
    extraction_method: str  # "deterministic" | "ocr"
    page_confidences: dict[int, float] = field(default_factory=dict)
    min_confidence: float = 1.0
    image_pages: list[int] = field(default_factory=list)
    flagged_pages: list[int] = field(default_factory=list)  # pages below threshold


@dataclass
class NativeAnalysis:
    """One conversion without OCR: per-page classification plus the document body."""

    pages: list[PageInfo]
    body: str


@functools.cache
def _converter(engine: str) -> Any:
    """Docling converters of the OCR path, built once (construction loads models).

    "native": no OCR. "auto": docling's default OCR engine (rapidocr here). "tesseract":
    the tesseract CLI, the independent second opinion for dual extraction.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "docling is required for PDF parsing. Install it with: uv add docling"
        ) from exc

    if engine == "native":
        options = PdfPipelineOptions(do_ocr=False)
    elif engine == "tesseract":
        options = PdfPipelineOptions(do_ocr=True, ocr_options=TesseractCliOcrOptions())
    else:
        options = PdfPipelineOptions(do_ocr=True)
    from pipeline.parsers.pdf import apply_table_mode

    apply_table_mode(options)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _convert(engine: str, pdf_path: Path) -> Any:
    """Convert with the page and size limits of the standard PDF path."""
    from pipeline.parsers.pdf import _limit

    return _converter(engine).convert(
        str(pdf_path),
        max_num_pages=_limit("PDF_MAX_PAGES", 2000),
        max_file_size=_limit("PDF_MAX_BYTES", 200 * 1024 * 1024),
    ).document


def _page_text(doc: Any, page_num: int) -> str:
    """Markdown of one page (0-based) of a converted document."""
    return str(doc.export_to_markdown(page_no=page_num + 1))


def analyze_pdf(pdf_path: Path) -> NativeAnalysis:
    """Classify pages as text or image-only from one conversion without OCR.

    A page with fewer than _MIN_TEXT_CHARS characters of native text is image-only.
    The body is returned too, so a PDF without image-only pages is converted once.
    """
    doc = _convert("native", pdf_path)
    page_nums = sorted(int(n) - 1 for n in (getattr(doc, "pages", None) or {})) or [0]
    pages: list[PageInfo] = []
    for page_num in page_nums:
        count = len(_page_text(doc, page_num).strip())
        pages.append(PageInfo(page_num, count, is_image_only=count < _MIN_TEXT_CHARS))
    return NativeAnalysis(pages=pages, body=str(doc.export_to_markdown()))


def detect_image_pages(pdf_path: Path) -> list[PageInfo]:
    """Which pages are image-only (scanned); see analyze_pdf."""
    return analyze_pdf(pdf_path).pages


def _compute_similarity(text_a: str, text_b: str) -> float:
    """Compute character-level similarity between two text extractions.

    Returns a float in [0.0, 1.0] where 1.0 = identical.
    """
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    return difflib.SequenceMatcher(None, text_a, text_b).ratio()


def dual_extract(
    pdf_path: Path, image_page_nums: list[int], *, single_engine_confidence: float = 0.8
) -> list[PageExtraction]:
    """OCR the image-only pages with two independent engines and compare them per page.

    Engine A is docling's default OCR; engine B is the tesseract CLI when installed.
    Each page's confidence is the similarity of the two readings of that page. With
    one engine, every page gets *single_engine_confidence*.
    """
    if not image_page_nums:
        return []
    doc_a = _convert("auto", pdf_path)
    body_a = str(doc_a.export_to_markdown())
    doc_b: Any = None
    if shutil.which("tesseract"):
        try:
            doc_b = _convert("tesseract", pdf_path)
        except Exception as exc:  # noqa: BLE001 — fall back to one engine
            logger.warning("Second OCR engine failed (%s); single extraction", exc)
    else:
        logger.info("tesseract not installed: OCR confidence from a single engine")

    extractions: list[PageExtraction] = []
    for page_num in image_page_nums:
        text_a = _page_text(doc_a, page_num)
        text_b = _page_text(doc_b, page_num) if doc_b is not None else ""
        sim = _compute_similarity(text_a, text_b) if doc_b is not None else single_engine_confidence
        extractions.append(
            PageExtraction(
                page_num=page_num, text_a=text_a, text_b=text_b, similarity=sim,
                confidence=sim, document_text=body_a,
            )
        )
    return extractions


def run_ocr_pipeline(pdf_path: Path, config: Config) -> OcrResult:
    """Full OCR pipeline: detect image pages, extract, score confidence.

    Returns an OcrResult with the extracted body and per-page confidence scores.
    """
    threshold = config.ocr_confidence_threshold
    analysis = analyze_pdf(pdf_path)
    pages = analysis.pages
    image_pages = [p.page_num for p in pages if p.is_image_only]

    if not image_pages:
        return OcrResult(
            body=analysis.body,
            extraction_method="deterministic",
            page_confidences={p.page_num: 1.0 for p in pages},
            min_confidence=1.0,
            image_pages=[],
            flagged_pages=[],
        )

    logger.info(
        "Detected %d image-only page(s) in %s — running OCR",
        len(image_pages),
        pdf_path.name,
    )
    extractions = dual_extract(
        pdf_path,
        image_pages,
        single_engine_confidence=config.ocr_single_engine_confidence,
    )

    page_confidences: dict[int, float] = {
        p.page_num: 1.0 for p in pages if not p.is_image_only
    }
    flagged_pages: list[int] = []
    for ext in extractions:
        page_confidences[ext.page_num] = ext.confidence
        if ext.confidence < threshold:
            flagged_pages.append(ext.page_num)
    # The OCR engine's whole-document reading (native text pages included).
    body = next(
        (e.document_text for e in extractions if e.document_text),
        "\n\n".join(e.text_a for e in extractions),
    )
    min_confidence = min(page_confidences.values()) if page_confidences else 0.0
    return OcrResult(
        body=body,
        extraction_method="ocr",
        page_confidences=page_confidences,
        min_confidence=min_confidence,
        image_pages=image_pages,
        flagged_pages=flagged_pages,
    )


def route_to_review(
    doc_id: str,
    ocr_result: OcrResult,
    config: Config,
) -> Path | None:
    """Write flagged pages to the review queue.

    Creates ``{review_dir}/{doc_id_short}/`` with a ``review.json`` manifest
    listing flagged pages and their confidence scores.

    Returns the review folder path if anything was flagged, else None.
    """
    if not ocr_result.flagged_pages:
        return None

    doc_id_short = doc_id[:8]
    review_folder = config.review_dir / doc_id_short
    review_folder.mkdir(parents=True, exist_ok=True)

    review_data: dict[str, Any] = {
        "doc_id": doc_id,
        "flagged_pages": ocr_result.flagged_pages,
        "page_confidences": {
            str(k): v for k, v in ocr_result.page_confidences.items()
        },
        "threshold": config.ocr_confidence_threshold,
        "image_pages": ocr_result.image_pages,
    }

    (review_folder / "review.json").write_text(
        json.dumps(review_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.warning(
        "Document %s has %d page(s) below confidence threshold (%.2f) — routed to review",
        doc_id_short,
        len(ocr_result.flagged_pages),
        config.ocr_confidence_threshold,
    )

    return review_folder

