"""Structural noise removal for extracted document bodies.

Strips known-boilerplate paragraphs from Markdown body text after parsing and
before chunking.  Conservative by design — only removes paragraphs that are
very clearly not document content:

  • Isolated page numbers (single digits on their own line)
  • Running headers / footers (identical short text appearing 3+ times)
  • Pure link-navigation blocks (≥ 3 markdown links, ≥ 80% link density)
  • Known short boilerplate phrases (cookie notices, JS warnings, social share,
    newsletter signups, "continue reading", "powered by", etc.)

Headings (#…) and table blocks are always kept.  Paragraphs longer than 200
characters are only checked for link-nav density; they are never matched
against phrase patterns (to avoid false positives in legitimate content).

Activated in acquisition.py for all doc_types when
``config.content_noise_filter_enabled`` is True (default).
"""

from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Paragraph consisting solely of a page number (1-4 digits) and optional
# decorative characters (dashes, bullets, whitespace).
_PAGE_NUM_RE = re.compile(r'^[\-–—·•|\s]*\d{1,4}[\-–—·•|\s]*$')

# Full-paragraph boilerplate: the *entire* stripped paragraph must match.
# Only applied to short paragraphs (≤ _MAX_BOILERPLATE_LEN characters).
_BOILERPLATE_RE = re.compile(
    r'^(?:'
    # JavaScript / browser warnings
    r'(?:please\s+)?enable\s+javascript(?:\s+to[\w\s]+)?'
    r'|javascript\s+is\s+(?:required|disabled|not\s+enabled)'
    r'|javascript\s+required'
    r'|your\s+browser\s+does\s+not\s+support\s+\w+'
    r'|browser\s+not\s+supported'
    # Cookie / GDPR banners
    r'|(?:this\s+)?(?:site|website)\s+uses?\s+cookies?(?:\s+[\w,\s]+)?'
    r'|we\s+use\s+cookies?(?:\s+to\s+[\w\s]+)?'
    r'|accept\s+(?:all\s+)?cookies?'
    r'|cookie\s+(?:settings?|policy|preferences?|notice|banner)'
    r'|privacy\s+and\s+cookie\s+(?:settings?|policy)'
    r'|manage\s+cookie\s+(?:settings?|preferences?)'
    # Social sharing (only when the entire paragraph is about sharing)
    r'|share\s+(?:this\s+)?(?:article|post|page)?(?:\s+on\s+[\w\s]+)?'
    r'|tweet\s+this'
    r'|copy\s+link'
    # Newsletter / mailing list subscription
    r'|subscribe\s+to\s+(?:our\s+)?(?:newsletter|updates?|mailing\s+list)'
    r'|sign\s+up\s+for\s+(?:our\s+)?(?:newsletter|updates?|e?mails?)'
    r'|join\s+our\s+(?:newsletter|mailing\s+list)'
    # Paywall / gated content fragments
    r'|continue\s+reading\.{0,3}'
    r'|read\s+(?:the\s+)?(?:full\s+)?(?:article|story)\.{0,3}'
    r'|(?:sign|log)\s+(?:up|in)\s+to\s+(?:read|continue|access)'
    r'|subscribe\s+to\s+(?:read|access|view)\s+(?:this|the\s+full)'
    r'|to\s+(?:read|view)\s+(?:the\s+)?(?:full\s+)?(?:article|story|content)'
    # "Powered by" / generator footers
    r'|powered\s+by\s+\w+(?:\s+\w+)?'
    r'|built\s+with\s+\w+(?:\s+\w+)?'
    r'|created\s+with\s+\w+(?:\s+\w+)?'
    r')$',
    re.IGNORECASE | re.UNICODE,
)

# Markdown links, excluding images ("![alt](src)" is content, not navigation).
_MD_LINK_RE = re.compile(r'(?<!!)\[([^\]]*)\]\([^)]*\)')

# Fenced code blocks are copied verbatim: short repeated lines such as "}" or
# "return x" would otherwise look like running headers or page numbers.
_FENCED_BLOCK_RE = re.compile(
    r'^[ ]{0,3}(`{3,}|~{3,})[^\n]*\n.*?(?:^[ ]{0,3}\1[`~]*[ \t]*$|\Z)',
    re.MULTILINE | re.DOTALL,
)

# Paragraphs longer than this are exempt from boilerplate-phrase matching.
_MAX_BOILERPLATE_LEN: int = 200

# Identical short paragraphs appearing this many times are running headers.
_RUNNING_MIN_COUNT: int = 3
_RUNNING_MAX_LEN: int = 120


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strip_noise(body: str) -> str:
    """Return *body* with structural noise paragraphs removed.

    Splits on blank lines, evaluates each paragraph, reassembles the kept
    paragraphs separated by double newlines.  A trailing newline is preserved
    if the input ends with one.
    """
    if not body.strip():
        return body

    # Alternate prose / fenced-code segments; only prose is filtered.
    segments: list[tuple[bool, str]] = []
    pos = 0
    for m in _FENCED_BLOCK_RE.finditer(body):
        segments.append((False, body[pos : m.start()]))
        segments.append((True, m.group().strip('\n')))
        pos = m.end()
    segments.append((False, body[pos:]))

    prose = [
        p for is_code, seg in segments if not is_code for p in re.split(r'\n\n+', seg)
    ]

    # Identify running headers: short paragraphs that appear 3+ times.
    freq: Counter[str] = Counter(
        p.strip().lower()
        for p in prose
        if 0 < len(p.strip()) <= _RUNNING_MAX_LEN
    )
    running: set[str] = {
        text for text, count in freq.items() if count >= _RUNNING_MIN_COUNT
    }

    parts: list[str] = []
    for is_code, seg in segments:
        if is_code:
            parts.append(seg)
        else:
            parts.extend(p for p in re.split(r'\n\n+', seg) if _should_keep(p, running))

    result = '\n\n'.join(p.strip('\n') for p in parts)
    if body.endswith('\n'):
        result += '\n'
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _should_keep(para: str, running: set[str]) -> bool:
    """Return True if *para* should be retained."""
    stripped = para.strip()

    if not stripped:
        return False

    # Headings are always kept.
    if stripped.startswith('#'):
        return True

    # Table blocks (3+ pipe characters) are always kept.
    if stripped.count('|') >= 3:
        return True

    # Long paragraphs: only check for pure link-nav density.
    if len(stripped) > _MAX_BOILERPLATE_LEN:
        return not _is_link_nav(stripped)

    # Isolated page numbers.
    if _PAGE_NUM_RE.fullmatch(stripped):
        return False

    # Running headers / footers.
    if stripped.lower() in running:
        return False

    # Known short boilerplate phrases.
    if _BOILERPLATE_RE.fullmatch(stripped):
        return False

    # Pure link-navigation blocks.
    if _is_link_nav(stripped):
        return False

    return True


def _is_link_nav(text: str) -> bool:
    """True if *text* is a pure link-navigation block.

    Requires at least 3 markdown links AND link-content density ≥ 80%
    (i.e., at most 20% of significant characters are outside link markup).
    """
    links = _MD_LINK_RE.findall(text)
    if len(links) < 3:
        return False
    residue = _MD_LINK_RE.sub('', text)
    sig_total = len(re.sub(r'[\s\W]', '', text))
    sig_residue = len(re.sub(r'[\s\W]', '', residue))
    if sig_total == 0:
        return False
    return (1.0 - sig_residue / sig_total) >= 0.80
