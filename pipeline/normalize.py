"""Canonical Markdown assembly: heading-level repair, whitespace normalization,
front matter injection."""

from __future__ import annotations

import re
from typing import Any

import yaml


def repair_heading_levels(text: str) -> str:
    """Ensure no heading skips a level (# → ### becomes # → ##).

    Rule: a heading may only go one level deeper than the previous heading.
    Going back up (## after ####) is always allowed.
    """
    lines = text.splitlines()
    out: list[str] = []
    prev_level = 0

    for line in lines:
        m = re.match(r"^(#{1,6})(\s+.*|$)", line)
        if m:
            level = len(m.group(1))
            # Clamp: never jump more than one level deeper
            if prev_level == 0:
                allowed = level  # first heading keeps its level
            else:
                allowed = min(level, prev_level + 1)
            prev_level = allowed
            rest = line[level:]  # everything after the hashes
            out.append("#" * allowed + rest)
        else:
            out.append(line)

    return "\n".join(out)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines to at most two, strip trailing spaces."""
    # Strip trailing whitespace from each line
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Collapse 2+ consecutive blank lines → 1 blank line
    result: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)
    return "\n".join(result).strip()


def build_front_matter(data: dict[str, Any]) -> str:
    """Serialize *data* as a YAML front-matter block."""
    return "---\n" + yaml.dump(data, allow_unicode=True, sort_keys=False) + "---\n"


def assemble_document(front_matter: dict[str, Any], body: str) -> str:
    """Return the canonical document.md content."""
    normalized = normalize_whitespace(repair_heading_levels(body))
    return build_front_matter(front_matter) + "\n" + normalized + "\n"


def strip_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split a document.md into (front_matter_dict, body_text)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    yaml_block = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    data: dict[str, Any] = yaml.safe_load(yaml_block) or {}
    return data, body


def char_count_normalized(text: str) -> int:
    """Character count after stripping all whitespace — used for round-trip check."""
    return len(re.sub(r"\s+", "", text))
