"""Load and validate configuration from environment / .env file."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    kb_root: Path
    documents_dir: Path
    kb_dropzone_dir: Path
    kb_done_dir: Path
    kb_failed_dir: Path
    index_dir: Path
    review_dir: Path
    qdrant_mode: str
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection: str
    embedding_model: str
    git_autocommit: bool
    log_level: str
    content_security_enabled: bool
    content_security_wrap_output: bool
    content_security_llm_enabled: bool
    content_security_llm_model: str
    content_security_block_threshold: float
    content_security_quarantine_threshold: float
    content_noise_filter_enabled: bool
    llm_converter_enabled: bool
    llm_converter_model: str
    llm_converter_max_tokens: int
    llm_converter_timeout_secs: float
    content_security_llm_timeout_secs: float
    # pacod-gw (pipeline/gateway.py): every LLM call goes through it.
    gateway_url: str
    gateway_api_key: str
    gateway_idle_timeout_secs: float
    language_validation_enabled: bool
    storage_backend: str
    database_host: str | None
    database_port: int
    database_name: str
    database_user: str | None
    database_password: str | None
    ocr_confidence_threshold: float
    ocr_enabled: bool
    # Inference (docs/retrieval-eval.md). 0 threads = derive from the CPU quota.
    torch_num_threads: int = 0
    embed_batch_tokens: int = 1024
    query_embed_cache_size: int = 256
    # Keyword leg (pipeline/text_lexicon.py): legacy | or | and_or
    keyword_query_mode: str = "legacy"
    rrf_keyword_weight: float = 1.0
    keyword_pg_timeout_ms: int = 1500
    # Second-stage reranker (pipeline/reranker.py)
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = ""
    reranker_trust_remote_code: bool = False
    reranker_code_revision: str = ""
    rerank_top_n: int = 10
    rerank_max_length: int = 384
    rerank_batch_size: int = 16
    rerank_gate_timeout_ms: int = 1000
    retrieval_candidate_k: int = 50
    inference_device: str = "auto"
    # Embedding reuse: unchanged chunks keep their vectors on supersession and rebuild
    embed_reuse: bool = True
    embed_reuse_untagged: bool = False
    rebuild_reuse_max_points: int = 100_000
    ocr_single_engine_confidence: float = 0.8
    # The shared embeddings service (pacod-embeddings): blank = the model in this process
    embeddings_url: str = ""
    embeddings_timeout_secs: float = 120.0
    # Its /rerank instead of a cross-encoder in this process (blank = local model)
    reranker_url: str = ""
    reranker_timeout_ms: int = 5000


# Shared components on pacod-net (pacod-demo/docs/NETWORK.md).
DEFAULT_QDRANT_URL = "http://pacod-qdrant:6333"
DEFAULT_DATABASE_HOST = "pacod-postgres"


# Model names are the gateway's: ``provider/model`` or a pool ``@name``.
DEFAULT_CONVERTER_MODEL = "anthropic/claude-opus-5-5"
DEFAULT_SECURITY_MODEL = "anthropic/claude-haiku-4-5"
# The converter streams through the gateway, so a long document can use a large cap.
DEFAULT_CONVERTER_MAX_TOKENS = 64000

_BARE_MODEL_PROVIDERS = (
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini-", "gemini"),
    ("deepseek-", "deepseek"),
    ("glm-", "glm"),
)

# Variables of the direct-provider era: read by nothing now (one warning at start).
LEGACY_ENV: dict[str, str] = {
    "ANTHROPIC_API_KEY": "provider keys belong to the gateway (PGW_PROVIDER_*_KEY)",
    "LLM_CONVERTER_API_KEY": "provider keys belong to the gateway (PGW_PROVIDER_*_KEY)",
    "LLM_CONVERTER_BASE_URL": "set PACOD_GW_URL instead",
    "LLM_CONVERTER_PROVIDER": "name the model as provider/model (only qualifies a bare name)",
    "SECRETS_KEY": "kb no longer stores provider keys",
    "TABLE_RECONSTRUCTION_MODEL": "table reconstruction was removed",
    "VLM_TABLE_RECONSTRUCTION_ENABLED": "table reconstruction was removed",
}


def qualify_model(name: str, provider: str = "") -> str:
    """A gateway model name for *name*: ``provider/model`` or ``@pool`` as given.

    A bare legacy name (``claude-opus-5-5``) gets the provider from *provider* (the old
    LLM_CONVERTER_PROVIDER) or from its prefix; an unknown bare name is left alone and
    the gateway resolves it when exactly one of its models has that id.
    """
    raw = name.strip()
    if not raw or "/" in raw or raw.startswith("@"):
        return raw
    hint = provider.strip().lower()
    hint = {"gemin": "gemini"}.get(hint, hint)
    if hint:
        return f"{hint}/{raw}"
    lowered = raw.lower()
    for prefix, owner in _BARE_MODEL_PROVIDERS:
        if lowered.startswith(prefix):
            return f"{owner}/{raw}"
    return raw


def warn_legacy_env() -> None:
    """Log once per legacy variable that is still set (it changes nothing now)."""
    for var, hint in LEGACY_ENV.items():
        if os.environ.get(var, "").strip():
            logging.getLogger(__name__).warning("%s is set but ignored: %s", var, hint)


def load_config() -> Config:
    kb_root_str = os.environ.get("KB_ROOT", "").strip()
    if not kb_root_str:
        raise SystemExit("KB_ROOT is required but not set. Set it in .env or the environment.")

    kb_root = Path(kb_root_str)

    dropzone_env = os.environ.get("KB_DROPZONE", "").strip()
    kb_dropzone_dir = Path(dropzone_env) if dropzone_env else kb_root / "dropzone"

    done_env = os.environ.get("KB_DONE", "").strip()
    kb_done_dir = Path(done_env) if done_env else kb_root / "done"

    failed_env = os.environ.get("KB_FAILED", "").strip()
    kb_failed_dir = Path(failed_env) if failed_env else kb_root / "failed"

    qdrant_mode = os.environ.get("QDRANT_MODE", "embedded").strip()
    if qdrant_mode not in ("embedded", "server"):
        raise SystemExit(f"QDRANT_MODE must be 'embedded' or 'server', got: {qdrant_mode!r}")

    qdrant_url: str | None = os.environ.get("QDRANT_URL", "").strip() or None
    if qdrant_mode == "server" and not qdrant_url:
        qdrant_url = DEFAULT_QDRANT_URL

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    def _validate_storage_backend(value: str) -> str:
        if value not in ("filesystem", "database"):
            raise SystemExit(
                f"KB_STORAGE_BACKEND must be 'filesystem' or 'database', got: {value!r}"
            )
        return value

    def _int_env(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            raise SystemExit(f"{name} must be an integer, got: {raw!r}") from None

    def _float_env(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            raise SystemExit(f"{name} must be a number, got: {raw!r}") from None

    storage_backend = _validate_storage_backend(
        os.environ.get("KB_STORAGE_BACKEND", "filesystem").strip()
    )
    database_host = os.environ.get("KB_DATABASE_HOST", "").strip() or DEFAULT_DATABASE_HOST
    database_port = _int_env("KB_DATABASE_PORT", 5432)
    keyword_query_mode = os.environ.get("KEYWORD_QUERY_MODE", "legacy").strip().lower() or "legacy"
    if keyword_query_mode not in ("legacy", "or", "and_or"):
        raise SystemExit(
            f"KEYWORD_QUERY_MODE must be legacy, or or and_or, got: {keyword_query_mode!r}"
        )
    database_name = os.environ.get("KB_DATABASE_NAME", "kb").strip() or "kb"
    database_user = os.environ.get("KB_DATABASE_USER", "").strip() or None
    database_password = os.environ.get("KB_DATABASE_PASSWORD", "").strip() or None

    if storage_backend == "database":
        missing: list[str] = []
        if not database_host:
            missing.append("KB_DATABASE_HOST")
        if not database_user:
            missing.append("KB_DATABASE_USER")
        if not database_password:
            missing.append("KB_DATABASE_PASSWORD")
        if missing:
            joined = ", ".join(missing)
            raise SystemExit(f"KB_STORAGE_BACKEND=database requires PostgreSQL settings: {joined}.")

    return Config(
        kb_root=kb_root,
        kb_dropzone_dir=kb_dropzone_dir,
        kb_done_dir=kb_done_dir,
        kb_failed_dir=kb_failed_dir,
        documents_dir=kb_root / "documents",
        index_dir=kb_root / "index",
        review_dir=kb_root / "review",
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
        qdrant_collection=os.environ.get("QDRANT_COLLECTION", "kb_chunks"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        git_autocommit=os.environ.get("GIT_AUTOCOMMIT", "true").lower() == "true",
        log_level=log_level,
        content_security_enabled=os.environ.get("CONTENT_SECURITY_ENABLED", "true").lower()
        == "true",
        content_security_wrap_output=os.environ.get("CONTENT_SECURITY_WRAP_OUTPUT", "true").lower()
        == "true",
        content_security_llm_enabled=os.environ.get("CONTENT_SECURITY_LLM_ENABLED", "false").lower()
        == "true",
        content_security_llm_model=qualify_model(
            os.environ.get("CONTENT_SECURITY_LLM_MODEL", "") or DEFAULT_SECURITY_MODEL
        ),
        content_security_block_threshold=_float_env("CONTENT_SECURITY_BLOCK_THRESHOLD", 1.0),
        content_security_quarantine_threshold=_float_env(
            "CONTENT_SECURITY_QUARANTINE_THRESHOLD", 0.5
        ),
        content_noise_filter_enabled=os.environ.get("CONTENT_NOISE_FILTER_ENABLED", "true").lower()
        == "true",
        llm_converter_enabled=os.environ.get("LLM_CONVERTER_ENABLED", "true").lower() == "true",
        llm_converter_model=qualify_model(
            os.environ.get("LLM_CONVERTER_MODEL", "") or DEFAULT_CONVERTER_MODEL,
            os.environ.get("LLM_CONVERTER_PROVIDER", ""),
        ),
        llm_converter_max_tokens=_int_env("LLM_CONVERTER_MAX_TOKENS", DEFAULT_CONVERTER_MAX_TOKENS),
        llm_converter_timeout_secs=_float_env("LLM_CONVERTER_TIMEOUT_SECS", 1800.0),
        content_security_llm_timeout_secs=_float_env("CONTENT_SECURITY_LLM_TIMEOUT_SECS", 15.0),
        gateway_url=os.environ.get("PACOD_GW_URL", "").strip().rstrip("/"),
        gateway_api_key=os.environ.get("PACOD_GW_API_KEY", "").strip(),
        gateway_idle_timeout_secs=_float_env("PACOD_GW_IDLE_TIMEOUT_SECS", 120.0),
        language_validation_enabled=os.environ.get("LANGUAGE_VALIDATION_ENABLED", "true").lower()
        == "true",
        storage_backend=storage_backend,
        database_host=database_host,
        database_port=database_port,
        database_name=database_name,
        database_user=database_user,
        database_password=database_password,
        ocr_confidence_threshold=_float_env("OCR_CONFIDENCE_THRESHOLD", 0.95),
        ocr_enabled=os.environ.get("OCR_ENABLED", "false").lower() == "true",
        torch_num_threads=_int_env("TORCH_NUM_THREADS", 0),
        embed_batch_tokens=_int_env("EMBED_BATCH_TOKENS", 1024),
        query_embed_cache_size=_int_env("QUERY_EMBED_CACHE_SIZE", 256),
        keyword_query_mode=keyword_query_mode,
        rrf_keyword_weight=_float_env("RRF_KEYWORD_WEIGHT", 1.0),
        keyword_pg_timeout_ms=_int_env("KEYWORD_PG_TIMEOUT_MS", 1500),
        reranker_enabled=os.environ.get("RERANKER_ENABLED", "false").lower() == "true",
        reranker_model=os.environ.get("RERANKER_MODEL", "").strip() or "BAAI/bge-reranker-v2-m3",
        reranker_revision=os.environ.get("RERANKER_REVISION", "").strip(),
        reranker_trust_remote_code=os.environ.get("RERANKER_TRUST_REMOTE_CODE", "false").lower()
        == "true",
        reranker_code_revision=os.environ.get("RERANKER_CODE_REVISION", "").strip(),
        rerank_top_n=max(1, _int_env("RERANK_TOP_N", 10)),
        rerank_max_length=max(16, _int_env("RERANK_MAX_LENGTH", 384)),
        rerank_batch_size=max(1, _int_env("RERANK_BATCH_SIZE", 16)),
        rerank_gate_timeout_ms=max(1, _int_env("RERANK_GATE_TIMEOUT_MS", 1000)),
        retrieval_candidate_k=max(1, _int_env("RETRIEVAL_CANDIDATE_K", 50)),
        inference_device=os.environ.get("INFERENCE_DEVICE", "auto").strip().lower() or "auto",
        embed_reuse=os.environ.get("EMBED_REUSE", "true").lower() == "true",
        embed_reuse_untagged=os.environ.get("EMBED_REUSE_UNTAGGED", "false").lower() == "true",
        rebuild_reuse_max_points=max(0, _int_env("REBUILD_REUSE_MAX_POINTS", 100_000)),
        ocr_single_engine_confidence=_float_env("OCR_SINGLE_ENGINE_CONFIDENCE", 0.8),
        embeddings_url=os.environ.get("EMBEDDINGS_URL", "").strip().rstrip("/"),
        embeddings_timeout_secs=max(1.0, _float_env("EMBEDDINGS_TIMEOUT_SECS", 120.0)),
        reranker_url=os.environ.get("RERANKER_URL", "").strip().rstrip("/"),
        reranker_timeout_ms=max(1, _int_env("RERANKER_TIMEOUT_MS", 5000)),
    )
