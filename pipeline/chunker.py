"""Structure-aware document chunker.

Splits canonical Markdown body text into Chunk records.  Hard constraints:
  (2) Heading line always accompanies its first paragraph; table rows are never
      split from their header or each other; ал./чл. boundary respected by
      keeping legal article bodies ≤ max_tokens (handled at ingest time via
      promote_legal — chunker just sees H4 sections).
  (3) Tables are always single chunks; embedded_text prepends nearest preceding
      paragraph as caption context.
  (4) Breadcrumb format: "[DocTitle > H1 > H2]\n" prepended to embedded_text.

byte_start / byte_end are UTF-8 byte offsets into the stripped document body
(result of strip_front_matter()).  This matches the PLAN spec; char offsets
would diverge for Cyrillic content.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_BLANK_SEP_RE = re.compile(r"[ \t]*\n(?:[ \t]*\n)+")
_FENCE_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})", re.MULTILINE)
# Sentence / line boundaries used to split a single over-long paragraph.
_UNIT_SPLIT_RE = re.compile(r"(?<=[.!?…;:])\s+|\n")
# Characters of the doc_id used in chunk ids. 8 hex chars collide at ~1% for
# 10k documents (overwriting another document's FTS row and Qdrant point).
_CHUNK_ID_PREFIX = 16

_DEFAULT_MAX_TOKENS = 500


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    heading_path: list[str]
    heading_level: int
    parent_chunk: str | None
    sequence: int
    byte_start: int
    byte_end: int
    embedded_text: str
    valid_until: str | None
    extraction_method: str
    confidence: float
    text: str = ""          # raw chunk text without breadcrumb
    content_hash: str = ""  # SHA1[:16] of text for change detection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    heading_path: list[str]
    heading_level: int  # 0 = preamble (no heading)
    char_start: int
    char_end: int


@dataclass
class _Piece:
    char_start: int
    char_end: int
    caption_span: tuple[int, int] | None = field(default=None)  # for tables


class _ByteOffsets:
    """Char → UTF-8 byte offset conversion in O(stride) per lookup.

    Re-encoding ``body[:pos]`` on every call made chunking quadratic in body
    size; checkpoints every *stride* chars keep each lookup cheap.
    """

    def __init__(self, body: str, stride: int = 4096) -> None:
        self._body = body
        self._stride = stride
        self._marks = [0]
        for start in range(0, len(body), stride):
            self._marks.append(self._marks[-1] + len(body[start : start + stride].encode("utf-8")))

    def __call__(self, char_pos: int) -> int:
        k = char_pos // self._stride
        base = k * self._stride
        return self._marks[k] + len(self._body[base:char_pos].encode("utf-8"))


def _fenced_ranges(body: str) -> list[tuple[int, int]]:
    """Char ranges covered by fenced code blocks (``` or ~~~)."""
    ranges: list[tuple[int, int]] = []
    open_at: int | None = None
    fence = ""
    for m in _FENCE_RE.finditer(body):
        marker = m.group(1)
        if open_at is None:
            open_at, fence = m.start(), marker
        elif marker[0] == fence[0] and len(marker) >= len(fence):
            ranges.append((open_at, m.end()))
            open_at = None
    if open_at is not None:
        ranges.append((open_at, len(body)))
    return ranges


def _split_long_span(
    body: str, start: int, end: int, max_tokens: int, token_counter: Callable[[str], int]
) -> list[tuple[int, int]]:
    """Split one paragraph that exceeds *max_tokens* at sentence/line boundaries.

    Without this a single huge paragraph became one chunk far past the
    embedder's window and was silently truncated at embedding time.
    """
    text = body[start:end]
    # A token covers at least one character, so short text cannot be too long.
    if len(text) <= max_tokens or token_counter(text) <= max_tokens:
        return [(start, end)]

    units: list[tuple[int, int]] = []
    prev = 0
    for m in _UNIT_SPLIT_RE.finditer(text):
        if m.end() > prev:
            units.append((prev, m.end()))
            prev = m.end()
    if prev < len(text):
        units.append((prev, len(text)))

    out: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    cur_tokens = 0
    for us, ue in units:
        n = token_counter(text[us:ue])
        if n > max_tokens:
            # A single unbroken run (no sentence/line breaks): hard-split by length.
            if cur_start is not None:
                out.append((cur_start, cur_end))
                cur_start, cur_tokens = None, 0
            step = max(1, (ue - us) * max_tokens // n)
            out.extend((k, min(k + step, ue)) for k in range(us, ue, step))
            continue
        if cur_start is not None and cur_tokens + n > max_tokens:
            out.append((cur_start, cur_end))
            cur_start, cur_tokens = None, 0
        if cur_start is None:
            cur_start = us
        cur_end = ue
        cur_tokens += n
    if cur_start is not None:
        out.append((cur_start, cur_end))
    return [(start + s, start + e) for s, e in out if text[s:e].strip()]


def _slugify_path(path: list[str]) -> str:
    if not path:
        return "preamble"
    full = " > ".join(path)
    path_hash = hashlib.md5(full.encode()).hexdigest()[:8]
    leaf = re.sub(r"[^\w\s-]", "", path[-1].lower(), flags=re.UNICODE)
    leaf = re.sub(r"[\s_]+", "-", leaf, flags=re.UNICODE).strip("-")[:30]
    return f"{leaf}-{path_hash}" if leaf else path_hash


def _build_breadcrumb(doc_title: str, heading_path: list[str]) -> str:
    parts = [doc_title, *heading_path]
    return "[" + " > ".join(parts) + "]\n"


def _is_table_block(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    table_lines = sum(1 for ln in lines if ln.strip().startswith("|"))
    return table_lines > len(lines) // 2


def _split_sections(body: str) -> list[_Section]:
    fenced = _fenced_ranges(body)
    # "# comment" lines inside code blocks are not headings.
    matches = [
        m for m in _HEADING_RE.finditer(body) if not any(s < m.start() < e for s, e in fenced)
    ]
    sections: list[_Section] = []

    if not matches:
        if body.strip():
            sections.append(_Section([], 0, 0, len(body)))
        return sections

    # Preamble: content before first heading
    if matches[0].start() > 0 and body[: matches[0].start()].strip():
        sections.append(_Section([], 0, 0, matches[0].start()))

    current_path: list[str] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading_text = m.group(2).strip()
        current_path = current_path[: level - 1] + [heading_text]

        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append(
            _Section(
                heading_path=list(current_path),
                heading_level=level,
                char_start=m.start(),
                char_end=section_end,
            )
        )

    return sections


def _find_para_spans(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Return (start, end) body-relative char spans of non-empty paragraphs."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BLANK_SEP_RE.finditer(text):
        seg = text[pos : m.start()]
        if seg.strip():
            spans.append((offset + pos, offset + m.start()))
        pos = m.end()
    if pos < len(text) and text[pos:].strip():
        spans.append((offset + pos, offset + len(text)))
    return spans


def _make_pieces(
    body: str,
    section: _Section,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[_Piece]:
    """Split one section into _Pieces, honouring heading+para and table rules."""
    sec_text = body[section.char_start : section.char_end]
    spans = [
        sub
        for ps, pe in _find_para_spans(sec_text, offset=section.char_start)
        for sub in (
            [(ps, pe)]
            if _is_table_block(body[ps:pe])  # tables are never split
            else _split_long_span(body, ps, pe, max_tokens, token_counter)
        )
    ]

    if not spans:
        return [_Piece(section.char_start, section.char_end)]

    total_tokens = token_counter(sec_text)
    if total_tokens <= max_tokens:
        return [_Piece(section.char_start, section.char_end)]

    # Need to split — process paragraph by paragraph
    pieces: list[_Piece] = []
    must_bind_next = True  # heading para always bound to first content para
    last_non_table: tuple[int, int] | None = None

    current_start = spans[0][0]
    current_end = spans[0][1]
    current_tokens = token_counter(body[current_start:current_end])

    for ps, pe in spans[1:]:
        is_table = _is_table_block(body[ps:pe])

        if is_table:
            # Flush current non-table accumulator
            if body[current_start:current_end].strip():
                pieces.append(_Piece(current_start, current_end))
            # Table gets its own piece with nearest preceding para as caption
            pieces.append(_Piece(ps, pe, caption_span=last_non_table))
            current_start = pe
            current_end = pe
            current_tokens = 0
            must_bind_next = False
            last_non_table = None
            continue

        p_tokens = token_counter(body[ps:pe])

        if must_bind_next:
            # Heading + first content are inseparable
            current_end = pe
            current_tokens += p_tokens
            must_bind_next = False
            last_non_table = (ps, pe)
            continue

        if current_tokens + p_tokens > max_tokens and current_end > current_start:
            if body[current_start:current_end].strip():
                pieces.append(_Piece(current_start, current_end))
            current_start = ps
            current_end = pe
            current_tokens = p_tokens
        else:
            current_end = pe
            current_tokens += p_tokens

        last_non_table = (ps, pe)

    if current_end > current_start and body[current_start:current_end].strip():
        pieces.append(_Piece(current_start, current_end))

    return pieces if pieces else [_Piece(section.char_start, section.char_end)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    body: str,
    doc_id: str,
    doc_title: str,
    token_counter: Callable[[str], int],
    *,
    valid_until: str | None = None,
    extraction_method: str = "deterministic",
    confidence: float = 1.0,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    """Chunk *body* into a list of Chunk records.

    body must be the stripped body text (output of strip_front_matter).
    byte_start / byte_end are UTF-8 byte offsets into body.encode('utf-8').
    """
    body_bytes = body.encode("utf-8")
    to_byte = _ByteOffsets(body)
    sections = _split_sections(body)

    chunks: list[Chunk] = []
    seq = 0
    section_seq: dict[str, int] = {}  # path_slug → seq within that section
    parent_by_level: dict[int, str] = {}  # heading_level → chunk_id of latest section

    for section in sections:
        pieces = _make_pieces(body, section, max_tokens, token_counter)
        path_slug = _slugify_path(section.heading_path)

        # Compute parent from ancestor levels before updating parent_by_level
        parent_chunk: str | None = None
        if section.heading_level > 0:
            for lvl in range(section.heading_level - 1, 0, -1):
                if lvl in parent_by_level:
                    parent_chunk = parent_by_level[lvl]
                    break

        first_chunk_id: str | None = None
        for piece in pieces:
            seq_in_section = section_seq.get(path_slug, -1) + 1
            section_seq[path_slug] = seq_in_section
            chunk_id = f"{doc_id[:_CHUNK_ID_PREFIX]}#{path_slug}#{seq_in_section}"

            byte_start = to_byte(piece.char_start)
            byte_end = to_byte(piece.char_end)
            chunk_text = body_bytes[byte_start:byte_end].decode("utf-8")

            breadcrumb = _build_breadcrumb(doc_title, section.heading_path)
            if piece.caption_span is not None:
                cs, ce = piece.caption_span
                cap_bs = to_byte(cs)
                cap_be = to_byte(ce)
                caption = body_bytes[cap_bs:cap_be].decode("utf-8")
                embedded_text = breadcrumb + caption + "\n\n" + chunk_text
            else:
                embedded_text = breadcrumb + chunk_text

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                heading_path=list(section.heading_path),
                heading_level=section.heading_level,
                parent_chunk=parent_chunk,
                sequence=seq,
                byte_start=byte_start,
                byte_end=byte_end,
                embedded_text=embedded_text,
                valid_until=valid_until,
                extraction_method=extraction_method,
                confidence=confidence,
                text=chunk_text,
                content_hash=hashlib.sha1(chunk_text.encode()).hexdigest()[:16],
            )
            chunks.append(chunk)

            if first_chunk_id is None:
                first_chunk_id = chunk_id
            seq += 1

        # Update parent tracker with the first chunk of this section
        if section.heading_level > 0 and first_chunk_id is not None:
            parent_by_level[section.heading_level] = first_chunk_id
            # Clear deeper levels (they are no longer valid ancestors)
            for lvl in [lv for lv in parent_by_level if lv > section.heading_level]:
                del parent_by_level[lvl]

    return chunks
