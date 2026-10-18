"""DOCX parser: Pandoc primary, mammoth fallback."""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.parsers import ParseResult
from pipeline.parsers.passthrough import _detect_language

logger = logging.getLogger(__name__)


class DocxParser:
    def parse(self, source: Path | str) -> ParseResult:
        path = Path(source)
        snapshot = path.read_bytes()

        body, converter = _try_pandoc(path)
        if body is None:
            logger.info("Pandoc unavailable or failed, falling back to mammoth for %s", path.name)
            body, converter = _try_mammoth(path)

        title = _first_heading_or_stem(body, path.stem)
        language = _detect_language(body[:2000])

        return ParseResult(
            body=body,
            title=title,
            language=language,
            extraction_method="deterministic",
            converter=converter,
            raw_snapshot=snapshot,
        )


def _try_pandoc(path: Path) -> tuple[str | None, str]:
    try:
        import pypandoc  # type: ignore[import-untyped]

        body: str = pypandoc.convert_file(
            str(path),
            "markdown",
            format="docx",
            extra_args=["--wrap=none", "--markdown-headings=atx"],
        )
        version: str = str(pypandoc.get_pandoc_version())
        return body, f"pandoc/{version}"
    except Exception as exc:
        logger.debug("pandoc failed: %s", exc)
        return None, ""


def _try_mammoth(path: Path) -> tuple[str, str]:
    import mammoth  # type: ignore[import-untyped]
    import trafilatura

    with open(path, "rb") as fh:
        result = mammoth.convert_to_html(fh)

    html: str = result.value
    body: str | None = trafilatura.extract(
        html,
        output_format="markdown",
        include_tables=True,
        include_formatting=True,
        favor_recall=True,
    )
    if not body:
        with open(path, "rb") as fh2:
            body = str(mammoth.extract_raw_text(fh2).value)

    version: str = str(getattr(mammoth, "__version__", "unknown"))
    return body or "", f"mammoth/{version}"


def _first_heading_or_stem(md: str, fallback: str) -> str:
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return fallback
