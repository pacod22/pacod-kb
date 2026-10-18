"""Parser protocol and format dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass
class ParseResult:
    body: str
    title: str
    language: str = "unknown"
    extraction_method: Literal["deterministic", "ocr", "vlm"] = "deterministic"
    converter: str = ""
    llm_usage: dict[str, Any] | None = None  # tokens, cost, currency, request_id, model_served
    raw_snapshot: bytes | None = None
    page_confidences: dict[int, float] | None = None
    confidence: float = 1.0


class Parser(Protocol):
    def parse(self, source: Path | str) -> ParseResult:
        """Parse *source* (path or URL) and return a ParseResult."""
        ...


def get_parser(ext: str, source_uri: str = "", json_convert: bool = True) -> Parser:
    """Return the appropriate parser for a file extension or URL.

    *json_convert* controls how ``.json`` files are handled:
    - ``True`` (default): convert to heading-aware Markdown via ``JsonParser``
    - ``False``: store verbatim as raw text via ``RawJsonParser``
    """
    from pipeline.parsers.docx import DocxParser
    from pipeline.parsers.json_parser import JsonParser, RawJsonParser
    from pipeline.parsers.passthrough import PassthroughParser
    from pipeline.parsers.pdf import PdfParser
    from pipeline.parsers.web import WebParser

    ext = ext.lower().lstrip(".")

    if source_uri.startswith(("http://", "https://")):
        return WebParser()
    if ext == "pdf":
        return PdfParser()
    if ext == "doc":
        # mammoth/pandoc read only the OOXML (.docx) format.
        raise ValueError("Legacy .doc files are not supported; save the file as .docx first")
    if ext == "docx":
        return DocxParser()
    if ext == "json":
        return JsonParser() if json_convert else RawJsonParser()
    if ext in ("md", "txt"):
        return PassthroughParser()
    if ext in ("html", "htm"):
        return WebParser()
    # Default: try passthrough
    return PassthroughParser()


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {"pdf", "docx", "json", "md", "txt", "html", "htm"}
)
