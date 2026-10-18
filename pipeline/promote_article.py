"""Metadata extraction for news articles ingested from URLs.

Uses trafilatura.bare_extraction() on the raw HTML bytes already fetched
during acquisition — no extra network call.  Returns a dict of optional
metadata fields; values are None when trafilatura cannot detect them.

Only called by acquisition.py when doc_type == 'article' and the source
is a URL (is_url == True).
"""

from __future__ import annotations

from typing import Any


def extract_article_meta(source_bytes: bytes) -> dict[str, str | None]:
    """Extract author, publication date, and outlet name from raw HTML bytes.

    Returns keys: meta_author, meta_published_date, meta_publication_venue.
    All values are strings or None.
    """
    try:
        import trafilatura

        doc = trafilatura.bare_extraction(source_bytes, include_links=False)
        if doc is None:
            return _empty()
        return {
            "meta_author": _field(doc, "author"),
            "meta_published_date": _field(doc, "date"),
            "meta_publication_venue": _field(doc, "sitename"),
        }
    except Exception:
        return _empty()


def _field(doc: Any, key: str) -> str | None:
    """Extract a string field from a trafilatura Document or dict result."""
    val = doc.get(key) if isinstance(doc, dict) else getattr(doc, key, None)
    return str(val) if val else None


def _empty() -> dict[str, str | None]:
    return {
        "meta_author": None,
        "meta_published_date": None,
        "meta_publication_venue": None,
    }
