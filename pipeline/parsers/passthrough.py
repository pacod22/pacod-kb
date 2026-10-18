"""MD / TXT passthrough parser — preserves content verbatim."""

from __future__ import annotations

from pathlib import Path

from pipeline.parsers import ParseResult


class PassthroughParser:
    def parse(self, source: Path | str) -> ParseResult:
        path = Path(source)
        snapshot = path.read_bytes()
        raw = snapshot.decode("utf-8", errors="replace")
        title = _extract_title(raw, path.stem)
        language = _detect_language(raw)
        return ParseResult(
            body=raw,
            title=title,
            language=language,
            extraction_method="deterministic",
            converter="passthrough/1.0",
            raw_snapshot=snapshot,
        )


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return fallback


def _detect_language(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect  # type: ignore[import-untyped]

        DetectorFactory.seed = 0
        sample = text[:2000]
        result = detect(sample)
        return str(result)
    except Exception:
        return "unknown"

