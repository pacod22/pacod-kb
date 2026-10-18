"""PDF parser using Docling, with Phase 4 image detection and OCR support."""

from __future__ import annotations

import functools
import logging
import os
import threading
from pathlib import Path
from typing import Any

from pipeline.parsers import ParseResult
from pipeline.parsers.passthrough import _detect_language

logger = logging.getLogger(__name__)

_CONVERT_LOCK = threading.Lock()


def _limit(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() else default


def apply_table_mode(options: Any) -> None:
    """DOCLING_TABLE_MODE=fast trades table-structure accuracy for speed (default accurate)."""
    if os.environ.get("DOCLING_TABLE_MODE", "accurate").strip().lower() == "fast":
        from docling.datamodel.pipeline_options import TableFormerMode

        options.table_structure_options.mode = TableFormerMode.FAST


@functools.cache
def _default_converter() -> Any:
    """Build docling's converter once — construction loads its layout models."""
    from docling.document_converter import DocumentConverter

    if os.environ.get("DOCLING_TABLE_MODE", "accurate").strip().lower() != "fast":
        return DocumentConverter()  # docling's own defaults (TableFormer accurate)
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption

    options = PdfPipelineOptions()
    apply_table_mode(options)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


class PdfParser:
    def parse(self, source: Path | str, *, config: object | None = None) -> ParseResult:
        path = Path(source)
        try:
            import docling.document_converter  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "docling is required for PDF parsing. Install it with: uv add docling"
            ) from exc

        # Phase 4: If config is available, use the OCR pipeline for image detection
        if config is not None and _has_ocr_config(config):
            return self._parse_with_ocr(path, config)

        # Standard deterministic extraction (Phase 1 behavior)
        with _CONVERT_LOCK:
            result = _default_converter().convert(
                str(path),
                max_num_pages=_limit("PDF_MAX_PAGES", 2000),
                max_file_size=_limit("PDF_MAX_BYTES", 200 * 1024 * 1024),
            )
        body = result.document.export_to_markdown()

        title = _extract_title_from_md(body) or path.stem
        language = _detect_language(body[:2000])

        try:
            import importlib.metadata

            version = importlib.metadata.version("docling")
        except Exception:
            version = "unknown"

        return ParseResult(
            body=body,
            title=title,
            language=language,
            extraction_method="deterministic",
            converter=f"docling/{version}",
            raw_snapshot=path.read_bytes(),
        )

    def _parse_with_ocr(self, path: Path, config: object) -> ParseResult:
        """Parse PDF with image detection and OCR pipeline (Phase 4)."""
        # Type narrowing: config must be a Config instance with the right attrs
        from pipeline.config import Config
        from pipeline.ocr import run_ocr_pipeline

        assert isinstance(config, Config)

        # One conversion at a time, as on the standard path (docling is not thread-safe).
        with _CONVERT_LOCK:
            ocr_result = run_ocr_pipeline(path, config)

        title = _extract_title_from_md(ocr_result.body) or path.stem
        language = _detect_language(ocr_result.body[:2000])

        try:
            import importlib.metadata

            version = importlib.metadata.version("docling")
        except Exception:
            version = "unknown"

        converter_name = f"docling/{version}"
        if ocr_result.extraction_method == "ocr":
            converter_name = f"docling-ocr/{version}"

        return ParseResult(
            body=ocr_result.body,
            title=title,
            language=language,
            extraction_method=ocr_result.extraction_method,  # type: ignore[arg-type]
            converter=converter_name,
            raw_snapshot=path.read_bytes(),
            page_confidences=ocr_result.page_confidences,
            confidence=ocr_result.min_confidence,
        )


def _has_ocr_config(config: object) -> bool:
    """Check if config has OCR enabled."""
    return hasattr(config, "ocr_enabled") and getattr(config, "ocr_enabled", False) is True


def _extract_title_from_md(md: str) -> str:
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return ""
