"""Rule-based heading promotion for academic papers.

Activated when doc_type == 'paper'.  Promotes bare section-header lines
(Abstract, Introduction, Methods, Results, Discussion, Conclusion, References, etc.)
to Markdown H2 headings.  Supports English, German, and Bulgarian/Russian.

Lines that already start with '#' or are blank are left untouched.
Matching is case-insensitive; an optional trailing colon is ignored.
Multi-word headers (e.g. "Materials and Methods") are matched in full.
"""

from __future__ import annotations

# Lowercase canonical header names.  The full (stripped, lowercased) line must
# equal one of these entries (after removing a trailing colon, if present).
_PAPER_HEADERS: frozenset[str] = frozenset({
    # ── Abstract ────────────────────────────────────────────────────────────────
    "abstract",
    "zusammenfassung",          # DE
    "абстракт",                 # BG / RU
    "резюме",                   # BG (alternate)
    # ── Keywords ────────────────────────────────────────────────────────────────
    "keywords", "key words",
    "schlüsselwörter",          # DE
    "ключови думи",             # BG
    "ключевые слова",           # RU
    # ── Introduction / Background ───────────────────────────────────────────────
    "introduction",
    "background",
    "overview",
    "literature review",
    "related work", "related works",
    "einleitung",               # DE
    "hintergrund",              # DE
    "въведение",                # BG
    "введение",                 # RU
    # ── Methods ─────────────────────────────────────────────────────────────────
    "methods", "method",
    "methodology",
    "materials and methods",
    "experimental setup",
    "experimental design",
    "experimental",
    "experiments",
    "methoden", "methodik",     # DE
    "materialien und methoden", # DE
    "методи",                   # BG
    "методология",              # BG / RU
    # ── Results ─────────────────────────────────────────────────────────────────
    "results",
    "findings",
    "experimental results",
    "ergebnisse",               # DE
    "резултати",                # BG
    "результаты",               # RU
    # ── Discussion ──────────────────────────────────────────────────────────────
    "discussion",
    "diskussion",               # DE
    "дискусия",                 # BG
    "обсуждение",               # RU
    # ── Conclusion ──────────────────────────────────────────────────────────────
    "conclusion", "conclusions",
    "summary",
    "final remarks",
    "fazit",                    # DE
    "заключение",               # BG / RU
    "извод",                    # BG (alternate)
    # ── Acknowledgments ─────────────────────────────────────────────────────────
    "acknowledgments", "acknowledgements",
    "acknowledgment", "acknowledgement",
    "danksagung",               # DE
    "благодарности",            # BG
    # ── References ──────────────────────────────────────────────────────────────
    "references", "bibliography", "works cited",
    "literaturverzeichnis", "referenzen",  # DE
    "литература",               # BG / RU
    "библиография",             # BG / RU
    # ── Appendix ────────────────────────────────────────────────────────────────
    "appendix", "appendices",
    "supplementary material", "supplementary materials",
    "supplemental material",
    "anhang",                   # DE
    "приложение",               # BG / RU
})


def promote_paper(body: str) -> str:
    """Return *body* with bare section-header lines promoted to Markdown H2."""
    out: list[str] = []
    in_fence = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        # Leave code, existing headings and blank lines untouched
        if in_fence or stripped.startswith("#") or not stripped:
            out.append(line)
            continue
        # Strip optional trailing colon for matching
        candidate = stripped.rstrip(":")
        if candidate.lower() in _PAPER_HEADERS:
            out.append(f"## {candidate}")
        else:
            out.append(line)
    result = "\n".join(out)
    return result + "\n" if body.endswith("\n") else result
