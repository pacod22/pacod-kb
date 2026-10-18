"""JSON parser: render structured JSON into heading-aware Markdown.

Plain JSON dumped verbatim chunks and embeds poorly (the heading-aware chunker
sees no headings, language detection returns "unknown", and the title is the
first brace).  This parser walks the parsed structure and emits Markdown:
mapping keys become headings (nested mappings/sequences) or ``**key:** value``
lines (scalars), and sequences of objects get a heading derived from each
item's ``name`` / ``title`` / ``id`` so the search breadcrumb stays meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.parsers import ParseResult
from pipeline.parsers.passthrough import _detect_language

_MAX_HEADING_DEPTH = 6
_TITLE_KEYS = ("title", "name", "heading", "id")


class RawJsonParser:
    """Store JSON verbatim — no conversion to Markdown.

    The raw JSON text becomes the document body; the title is taken from the
    filename stem so it is always meaningful (unlike the first brace/bracket
    that a generic passthrough would return).
    """

    def parse(self, source: Path | str) -> ParseResult:
        from pipeline.parsers.passthrough import _detect_language as _dl

        path = Path(source)
        snapshot = path.read_bytes()
        try:
            text = snapshot.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Could not decode JSON file as UTF-8: {exc}") from exc

        return ParseResult(
            body=text,
            title=path.stem,
            language=_dl(text[:2000]),
            extraction_method="deterministic",
            converter="json-raw/1.0",
            raw_snapshot=snapshot,
        )


class JsonParser:
    def parse(self, source: Path | str) -> ParseResult:
        path = Path(source)
        snapshot = path.read_bytes()
        try:
            data: Any = json.loads(snapshot.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        title = _extract_title(data, path.stem)
        body = "\n".join(_render(data, depth=1)).strip() + "\n"
        language = _detect_language(body[:2000])

        return ParseResult(
            body=body,
            title=title,
            language=language,
            extraction_method="deterministic",
            converter="json/1.0",
            raw_snapshot=snapshot,
        )


def _extract_title(data: Any, fallback: str) -> str:
    if isinstance(data, dict):
        for key in _TITLE_KEYS:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return fallback


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _heading_for_item(item: Any, index: int) -> str:
    if isinstance(item, dict):
        for key in _TITLE_KEYS:
            val = item.get(key)
            if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                text = str(val).strip()
                if text:
                    return text
    return f"Item {index}"


def _render(value: Any, depth: int) -> list[str]:
    """Render parsed JSON *value* to a list of Markdown lines."""
    level = min(depth + 1, _MAX_HEADING_DEPTH)
    prefix = "#" * level
    lines: list[str] = []

    if isinstance(value, dict):
        if not value:
            lines.append("_(empty object)_")
            lines.append("")
            return lines
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{prefix} {key}")
                lines.append("")
                lines.extend(_render(val, depth + 1))
            else:
                lines.append(f"**{key}:** {_scalar(val)}")
                lines.append("")
        return lines

    if isinstance(value, list):
        if not value:
            lines.append("_(empty list)_")
            lines.append("")
            return lines
        if all(not isinstance(item, (dict, list)) for item in value):
            for item in value:
                lines.append(f"- {_scalar(item)}")
            lines.append("")
            return lines
        for index, item in enumerate(value, start=1):
            lines.append(f"{prefix} {_heading_for_item(item, index)}")
            lines.append("")
            lines.extend(_render(item, depth + 1))
        return lines

    lines.append(_scalar(value))
    lines.append("")
    return lines
