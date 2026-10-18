"""Rule-based heading promotion for legal codes.

Activated only when doc_type == 'legal_code'.  Supported languages:
English, German, Bulgarian, Russian, Serbian (Cyrillic and Latin),
Macedonian, Croatian, and Slovenian.

Sub-article markers (ал. / Abs. / numbered paragraphs) stay as inline text.
"""

from __future__ import annotations

import re
from typing import NamedTuple

_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")


class _Rule(NamedTuple):
    pattern: re.Pattern[str]
    level: int


# Latin "Section/Part"-style keywords also start ordinary sentences ("Part of
# the reason…", "Teil der Regelung…"), so they must be followed by a number,
# roman numeral or single letter. Cyrillic keywords keep word ordinals
# ("Глава първа") and need no numeral.
_NUM = r"\s+(?:\d+[a-z]?|[IVXLCDM]+|[A-Z])\b\.?"

# Headings are short; a long line starting with a keyword is prose.
_MAX_HEADING_LEN = 120

# Ordered H2 → H3 → H4; first match wins.
_RULES: list[_Rule] = [
    # ── H2: Chapter ──────────────────────────────────────────────────────────
    _Rule(re.compile(r"^(Глава|ГЛАВА)\b", re.UNICODE), 2),           # BG / RU / MK / SR-Cyr
    _Rule(re.compile(r"^Chapter" + _NUM, re.IGNORECASE), 2),          # EN
    _Rule(re.compile(r"^Kapitel" + _NUM, re.IGNORECASE), 2),          # DE
    _Rule(re.compile(r"^Glava\b", re.IGNORECASE), 2),                 # SR-Lat / HR
    _Rule(re.compile(r"^Poglavje\b", re.IGNORECASE), 2),              # SL

    # ── H3: Section / Part ───────────────────────────────────────────────────
    _Rule(re.compile(r"^(Раздел|РАЗДЕЛ)\b", re.UNICODE), 3),          # BG / RU
    _Rule(re.compile(r"^(Дел|ДЕЛ)\b", re.UNICODE), 3),                # MK
    _Rule(re.compile(r"^(Одељак|ОДЕЉАК|Одсек|ОДСЕК)\b", re.UNICODE), 3),  # SR-Cyr
    _Rule(re.compile(r"^(Section|Part)" + _NUM, re.IGNORECASE), 3),   # EN
    _Rule(re.compile(r"^(Abschnitt|Teil)" + _NUM, re.IGNORECASE), 3), # DE
    _Rule(re.compile(r"^(Odeljak|Odsek)" + _NUM, re.IGNORECASE), 3),  # SR-Lat
    _Rule(re.compile(r"^(Odjeljak|Dio)" + _NUM, re.IGNORECASE), 3),   # HR
    _Rule(re.compile(r"^(Oddelek|Del)" + _NUM, re.IGNORECASE), 3),    # SL

    # ── H4: Article ──────────────────────────────────────────────────────────
    _Rule(re.compile(r"^(Чл\.|Член)\s*\d+", re.UNICODE), 4),          # BG / MK
    _Rule(re.compile(r"^(Статья|СТАТЬЯ|Ст\.)\s*\d+", re.UNICODE), 4), # RU
    _Rule(re.compile(r"^(Члан|ЧЛАН)\s*\d+", re.UNICODE), 4),          # SR-Cyr
    _Rule(re.compile(r"^Article\s+\d+", re.IGNORECASE), 4),           # EN
    _Rule(re.compile(r"^Artikel\s+\d+", re.IGNORECASE), 4),           # DE
    _Rule(re.compile(r"^Član\s*\d+", re.IGNORECASE), 4),              # SR-Lat
    _Rule(re.compile(r"^Članak\s*\d+", re.IGNORECASE), 4),            # HR
    _Rule(re.compile(r"^Čl\.\s*\d+", re.IGNORECASE), 4),              # HR (abbrev)
    _Rule(re.compile(r"^Člen\s+\d+", re.IGNORECASE), 4),              # SL
    _Rule(re.compile(r"^§\s*\d+"), 4),                                # Generic (DE / BG / …)
    _Rule(re.compile(r"^Art\.\s*\d+"), 4),                            # Generic
]


def promote_legal(text: str) -> str:
    """Return *text* with legal-structure lines promoted to Markdown headings."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.rstrip()
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
        # Leave code, existing headings, blank and long (prose) lines untouched
        if (
            in_fence
            or stripped.startswith("#")
            or not stripped
            or len(stripped) > _MAX_HEADING_LEN
        ):
            out.append(line)
            continue
        for rule in _RULES:
            if rule.pattern.match(stripped):
                out.append("#" * rule.level + " " + stripped)
                break
        else:
            out.append(line)
    result = "\n".join(out)
    return result + "\n" if text.endswith("\n") else result
