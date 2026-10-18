"""Document acquisition: hashing, snapshotting, dedup, manifest writing."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import subprocess
import unicodedata
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.config import Config
from pipeline.fs_layout import find_folder_by_doc_id, new_doc_folder, write_json_atomic
from pipeline.normalize import assemble_document, char_count_normalized
from pipeline.parsers import ParseResult, get_parser
from pipeline.projects import default_project_slug
from pipeline.schema_validation import validate as validate_schema

if TYPE_CHECKING:
    from pipeline.document_store import DocumentStore
    from pipeline.gateway import Attribution

logger = logging.getLogger(__name__)


# ISO 639-1 codes for the languages with full pipeline support.
# langdetect returns these codes; "unknown" is always allowed (short texts).
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {"en", "de", "bg", "ru", "sr", "mk", "hr", "sl", "unknown"}
)


@dataclass(frozen=True)
class IngestOptions:
    """The switches an ingest follows: the dashboard's, with a request's overrides.

    Every API-side ingest path (single URL, URL batch, upload, dropzone, watch)
    builds one through ``pipeline.api._ingest_options`` so the runtime switches
    apply everywhere; the CLI follows the env config instead.
    """

    llm_convert: bool | None = None
    trusted: bool = False
    validate_language: bool | None = None
    llm_runtime_config: dict[str, Any] | None = None

    def kwargs(self) -> dict[str, Any]:
        return {
            "llm_convert": self.llm_convert,
            "trusted": self.trusted,
            "validate_language": self.validate_language,
            "llm_runtime_config": self.llm_runtime_config,
        }


def _validate(data: dict[str, Any], schema_name: str) -> None:
    validate_schema(data, schema_name)


_gateway_warned = False


def _warn_gateway_unconfigured() -> None:
    """Conversion is on but no gateway is set up: keep the deterministic parse, say so once."""
    global _gateway_warned
    if not _gateway_warned:
        _gateway_warned = True
        logger.warning(
            "LLM conversion skipped: PACOD_GW_URL / PACOD_GW_API_KEY are not set "
            "(documents keep their deterministic parse)"
        )


def clean_title(text: str) -> str:
    """Strip leading/trailing special characters from a document title.

    Removes any run of non-alphanumeric characters (e.g. ``* % [ ] & $ ( ) { }``,
    quotes, punctuation, whitespace) from the start and end of *text*, while
    leaving characters inside the title untouched. Unicode-aware, so letters in
    Cyrillic and other scripts are preserved.
    """
    return re.sub(r"^[^\w]+|[^\w]+$", "", text, flags=re.UNICODE)


def slugify(text: str, max_len: int = 50) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "-", text)
    text = text.strip("-")
    return (text or "untitled")[:max_len]


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _find_existing(doc_id: str, config: Config) -> Path | None:
    """Return the existing doc folder if doc_id is already ingested."""
    return find_folder_by_doc_id(config.documents_dir, doc_id)


def _git_commit(config: Config, doc_folder: Path, title: str, doc_id_short: str) -> None:
    if not config.git_autocommit:
        return
    try:
        repo_root = config.kb_root
        rel = doc_folder.relative_to(repo_root)
        subprocess.run(["git", "-C", str(repo_root), "add", str(rel)], check=True)
        msg = f"ingest: add '{title}' [{doc_id_short}]"
        subprocess.run(["git", "-C", str(repo_root), "commit", "-m", msg], check=True)
        logger.info("git commit: %s", msg)
    except Exception as exc:
        logger.warning("git autocommit failed (non-fatal): %s", exc)


def ingest_document(
    source: str,
    config: Config,
    *,
    doc_type: str = "generic",
    source_version: str = "",
    supersedes: str | None = None,
    trusted: bool = False,
    store: DocumentStore | None = None,
    json_convert: bool = True,
    llm_convert: bool | None = None,
    llm_runtime_config: dict[str, Any] | None = None,
    validate_language: bool | None = None,
    project: str | None = None,
    url_only: bool = False,
    attribution: Attribution | None = None,
) -> tuple[str, Path | None]:
    """Ingest a document and return (doc_id, doc_folder).

    When *store* is provided (database mode) the folder is always None.
    Raises ValueError on parse failure.  Returns immediately (no-op) if the
    same sha256 has already been ingested.

    *json_convert* — when ``False`` and the source is a ``.json`` file, the
    raw JSON text is stored verbatim instead of being converted to Markdown.
    *project* — project slug this document belongs to; ``None`` means the
    default project (resolved through the project registry).
    *url_only* — when ``True`` the source must be an http(s) URL; local paths
    are rejected.  Set by every caller whose source string comes from a user
    as a "URL" so it can never be used to read files from the server.
    """
    source_str = str(source)
    if url_only:
        from pipeline.parsers.web import require_http_url

        source_str = require_http_url(source_str)
    is_url = source_str.startswith(("http://", "https://"))

    # --- fetch / read source bytes ---
    if is_url:
        from pipeline.parsers.web import _browser_fetch

        source_bytes = _browser_fetch(source_str)
        source_ext = ".html"
        raw_name = source_str.split("/")[-1].split("?")[0] or "page.html"
        source_filename = unicodedata.normalize(
            "NFC", urllib.parse.unquote(raw_name, encoding="utf-8")
        )
    else:
        source_path = Path(source_str)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        source_bytes = source_path.read_bytes()
        source_ext = source_path.suffix
        source_filename = unicodedata.normalize("NFC", source_path.name)

    doc_id = sha256_of_bytes(source_bytes)
    doc_id_short = doc_id[:8]

    # --- idempotency check ---
    if store is not None:
        if store.sha256_exists(doc_id):
            logger.info("Document %s already ingested (sha256 match) — skipping", doc_id_short)
            return doc_id, None
    else:
        existing = _find_existing(doc_id, config)
        if existing is not None:
            logger.info("Document %s already ingested (sha256 match) — skipping", doc_id_short)
            return doc_id, existing

    # --- parse ---
    now_utc = datetime.now(UTC).isoformat()
    parser = get_parser(source_ext.lstrip("."), source_uri=source_str, json_convert=json_convert)

    try:
        if is_url:
            # WebParser already fetched — re-use snapshot bytes
            from pipeline.parsers.web import WebParser

            wp = WebParser()
            result: ParseResult = wp._parse_bytes(source_bytes, source_str)
        else:
            # For PDFs, pass config to enable Phase 4 OCR pipeline
            if source_ext.lstrip(".").lower() == "pdf":
                from pipeline.parsers.pdf import PdfParser

                pdf_parser = PdfParser()
                result = pdf_parser.parse(Path(source_str), config=config)
            else:
                result = parser.parse(Path(source_str))
    except Exception as exc:
        logger.error("Parse failed for %s: %s", source_str, exc)
        raise ValueError(f"Parse failed: {exc}") from exc

    # --- language check ---
    effective_language_validation = (
        validate_language if validate_language is not None else config.language_validation_enabled
    )
    lang = result.language
    if effective_language_validation and lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{lang}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES - {'unknown'}))}."
        )
    if effective_language_validation and lang == "unknown":
        logger.warning("Could not detect language for %s — proceeding anyway", source_filename)
    if not effective_language_validation and lang not in SUPPORTED_LANGUAGES:
        logger.warning(
            "Language validation disabled; allowing unsupported language '%s' for %s",
            lang,
            source_filename,
        )

    from pipeline.ledger import ledger_session_factory

    session_factory = ledger_session_factory(config, store)
    security_on = config.content_security_enabled and not trusted
    quarantined = False
    security_summary = ""

    # What the gateway records for this document's LLM calls: the job (or, without
    # one, the document) as the trace, the project as the end user.
    from pipeline.gateway import Attribution as _Attribution
    from pipeline.gateway import get_gateway

    base_attr = attribution or _Attribution(task="document_conversion")
    base_attr = dataclasses.replace(
        base_attr,
        trace=base_attr.trace or doc_id[:16],
        end_user=base_attr.end_user or project,
    )

    def _security_scan(text: str, scan_title: str) -> None:
        nonlocal quarantined, security_summary
        from pipeline.content_security import scan_content

        scan = scan_content(
            text,
            scan_title,
            config,
            session_factory=session_factory,
            runtime_config=llm_runtime_config,
            attribution=dataclasses.replace(base_attr, task="content_security"),
        )
        security_summary = scan.summary()
        if scan.verdict == "block":
            raise ValueError(f"Content security check failed: {security_summary}")
        quarantined = quarantined or scan.verdict == "quarantine"

    # --- content security scan on the deterministic parse ---
    # Runs *before* any LLM sees the text: the converter could act on injected
    # instructions, and conversion strips the very signals the scanner looks for
    # (zero-width/bidi characters, hidden markup, script blocks).
    if security_on:
        _security_scan(result.body, clean_title(result.title or source_filename) or "untitled")

    # --- LLM conversion (optional; replaces body with LLM-formatted Markdown) ---
    # per-call llm_convert overrides the global config flag when explicitly set
    effective_llm = llm_convert if llm_convert is not None else config.llm_converter_enabled
    converted = False
    if effective_llm and quarantined:
        logger.warning("Skipping LLM conversion for %s: content flagged for review", source_str)
    elif effective_llm and not get_gateway(config).configured():
        _warn_gateway_unconfigured()
    elif effective_llm:
        from pipeline.parsers.llm import convert_with_llm

        try:
            result = convert_with_llm(
                result,
                config,
                source_bytes=source_bytes,
                source_ext=source_ext,
                runtime_config=llm_runtime_config,
                session_factory=session_factory,
                attribution=dataclasses.replace(base_attr, task="document_conversion"),
            )
            converted = True
        except Exception as exc:
            # Conversion is an optional polish step: a truncated/refused/failed
            # call must not lose the document.
            logger.warning(
                "LLM conversion failed for %s (%s) — keeping deterministic parse",
                source_str,
                exc,
            )

    # --- content noise filter (strip boilerplate / running headers / page numbers) ---
    body = result.body
    if config.content_noise_filter_enabled:
        from pipeline.content_filter import strip_noise

        body = strip_noise(body)

    # --- legal / paper heading promotion ---
    if doc_type in {"legal_code", "regulation", "directive"}:
        from pipeline.promote_legal import promote_legal

        body = promote_legal(body)
    elif doc_type == "paper":
        from pipeline.promote_paper import promote_paper

        body = promote_paper(body)

    # --- finalize display title (strip leading/trailing special characters) ---
    title = clean_title(result.title or source_filename) or "untitled"

    # --- re-scan the LLM output (the model may have introduced or surfaced content) ---
    if security_on and converted:
        _security_scan(body, title)

    # --- Phase 4: OCR review queue for low-confidence pages ---
    ocr_review = False
    if result.page_confidences is not None:
        flagged_pages = [
            p
            for p, conf in result.page_confidences.items()
            if conf < config.ocr_confidence_threshold
        ]
        if flagged_pages:
            ocr_review = True
            from pipeline.ocr import OcrResult, route_to_review

            ocr_result = OcrResult(
                body=body,
                extraction_method=result.extraction_method,
                page_confidences=result.page_confidences,
                min_confidence=result.confidence,
                image_pages=[p for p, conf in result.page_confidences.items() if conf < 1.0],
                flagged_pages=flagged_pages,
            )
            route_to_review(doc_id, ocr_result, config)

    # --- type-specific metadata (article from URL: extract HTML meta tags) ---
    type_meta: dict[str, str | None] = {}
    if doc_type == "article" and is_url:
        from pipeline.promote_article import extract_article_meta

        type_meta = extract_article_meta(source_bytes)

    # --- front matter (common to both storage modes) ---
    front_matter: dict[str, Any] = {
        "doc_id": doc_id,
        "title": title,
        "source_file": source_filename,
        "source_uri": source_str,
        "source_version": source_version,
        "source_sha256": doc_id,
        "retrieved_at": now_utc,
        "converted_at": now_utc,
        "converter": result.converter,
        "extraction_method": result.extraction_method,
        "language": result.language,
        "doc_type": doc_type,
        "supersedes": supersedes,
        "valid_until": None,
        "project": project or default_project_slug(config, getattr(store, "_factory", None)),
        **type_meta,
    }
    _validate(front_matter, "frontmatter.json")

    doc_status = "review" if (quarantined or ocr_review) else "ok"
    snapshot_bytes = result.raw_snapshot if result.raw_snapshot is not None else source_bytes

    # --- round-trip character-count check (warn only) ---
    _check_roundtrip(source_bytes, source_ext, body)

    manifest: dict[str, Any] = {
        **front_matter,
        "status": doc_status,
        "error": None,
        "chunk_count": 0,
    }
    _validate(manifest, "manifest.json")

    usage_payload: dict[str, Any] | None = None
    if result.llm_usage and result.converter:
        # The gateway's figures: tokens, its price (cost/currency) and request id.
        from pipeline.llm_management import record_usage_event, to_usage_payload

        usage_payload = to_usage_payload(result.llm_usage)
        usage_payload["model"] = result.converter
        # The ledger exists in both backends (pipeline/ledger.py).
        record_usage_event(
            session_factory=session_factory,
            user_id=base_attr.user_id,
            session_id=None,
            turn_id=doc_id,
            model=str(usage_payload["model"]),
            purpose="document_conversion",
            input_tokens=int(usage_payload.get("input_tokens", 0)),
            output_tokens=int(usage_payload.get("output_tokens", 0)),
            cached_tokens=int(usage_payload.get("cached_tokens", 0)),
            cost_usd=usage_payload.get("cost"),
            currency=usage_payload.get("currency"),
            request_id=usage_payload.get("request_id"),
        )

    if store is not None:
        # --- database (or custom store) mode ---
        store.save(doc_id, front_matter, body, snapshot_bytes, source_ext, status=doc_status)
        store.append_event(doc_id, "ingested", source_str)
        if usage_payload is not None:
            store.append_event(doc_id, "llm_usage", json.dumps(usage_payload, ensure_ascii=False))
        if quarantined:
            store.append_event(
                doc_id,
                "security_flagged",
                f"quarantined for review: {security_summary}",
            )
            logger.warning(
                "Content security: %s quarantined (status=review, not indexed): %s",
                source_filename,
                security_summary,
            )
        if ocr_review:
            store.append_event(
                doc_id,
                "review_flagged",
                f"OCR confidence below threshold on pages: {flagged_pages}",
            )
        logger.info("Ingested %s → store", source_filename)
        return doc_id, None

    # --- filesystem mode (existing behavior) ---
    slug = slugify(title)
    config.documents_dir.mkdir(parents=True, exist_ok=True)
    doc_folder = new_doc_folder(config.documents_dir, doc_id, slug)
    doc_folder.mkdir(parents=True, exist_ok=True)

    doc_md = assemble_document(front_matter, body)
    (doc_folder / "document.md").write_text(doc_md, encoding="utf-8")
    (doc_folder / f"source{source_ext}").write_bytes(snapshot_bytes)
    write_json_atomic(doc_folder / "manifest.json", manifest)

    from pipeline.events import append_event

    append_event(doc_folder / "manifest.json", "ingested", source_str)
    if usage_payload is not None:
        append_event(
            doc_folder / "manifest.json",
            "llm_usage",
            json.dumps(usage_payload, ensure_ascii=False),
        )
    if quarantined:
        append_event(
            doc_folder / "manifest.json",
            "security_flagged",
            f"quarantined for review: {security_summary}",
        )
        logger.warning(
            "Content security: %s quarantined (status=review, not indexed): %s",
            source_filename,
            security_summary,
        )
    if ocr_review:
        append_event(
            doc_folder / "manifest.json",
            "review_flagged",
            f"OCR confidence below threshold on pages: {flagged_pages}",
        )

    logger.info("Ingested %s → %s", source_filename, doc_folder.name)
    _git_commit(config, doc_folder, front_matter["title"], doc_id_short)
    return doc_id, doc_folder


def _check_roundtrip(source_bytes: bytes, source_ext: str, extracted_body: str) -> None:
    """Log a warning if char count diverges by more than 2% (text formats only)."""
    ext = source_ext.lstrip(".").lower()
    if ext not in ("txt", "md"):
        return
    try:
        source_text = source_bytes.decode("utf-8", errors="replace")
        src_count = char_count_normalized(source_text)
        body_count = char_count_normalized(extracted_body)
        if src_count == 0:
            return
        diff = abs(src_count - body_count) / src_count
        if diff > 0.02:
            logger.warning(
                "Round-trip char count divergence %.1f%% (source=%d, body=%d)",
                diff * 100,
                src_count,
                body_count,
            )
    except Exception:
        pass
