"""Runtime-overridable LLM settings, stored in the ledger's ``app_settings`` table.

Five settings, each with an env default and an AI Management override that applies
on the next call without a restart:

* ``converter_model`` / ``converter_max_tokens`` / ``converter_temperature`` /
  ``converter_temperature_enabled`` — document conversion;
* ``security_model`` — the content-security second opinion.

Model names are the gateway's (``provider/model`` or ``@pool``). The keys earlier
releases actually read (``llm.primary_max_tokens``, ``llm.primary_temperature``,
``llm.primary_temperature_enabled``) are honoured when the new key is absent.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipeline.config import (
    DEFAULT_CONVERTER_MAX_TOKENS,
    DEFAULT_CONVERTER_MODEL,
    DEFAULT_SECURITY_MODEL,
    qualify_model,
)

if TYPE_CHECKING:
    from pipeline.config import Config


@dataclass(frozen=True)
class _FieldSpec:
    key: str
    kind: str
    legacy_key: str | None = None


_FIELDS: dict[str, _FieldSpec] = {
    "converter_model": _FieldSpec("llm.converter_model", "str"),
    "converter_max_tokens": _FieldSpec("llm.converter_max_tokens", "int", "llm.primary_max_tokens"),
    "converter_temperature": _FieldSpec(
        "llm.converter_temperature", "float", "llm.primary_temperature"
    ),
    "converter_temperature_enabled": _FieldSpec(
        "llm.converter_temperature_enabled", "bool", "llm.primary_temperature_enabled"
    ),
    "security_model": _FieldSpec("llm.security_model", "str"),
}

# A gateway model name: a pool, or provider/model.
MODEL_NAME_RE = re.compile(r"^(@[a-z0-9_-]{1,64}|[a-z0-9-]{1,64}/[A-Za-z0-9._:-]{1,190})$")

_CACHE: dict[str, Any] = {"expires_at": 0.0, "effective": None, "env_default": None}
_CACHE_TTL_SECS = 60.0


def valid_model_name(name: str) -> str:
    """*name* as a gateway model name (bare legacy names qualified); ValueError if not one."""
    qualified = qualify_model(name)
    if not MODEL_NAME_RE.fullmatch(qualified):
        raise ValueError(f"not a gateway model name (provider/model or @pool): {name!r}")
    return qualified


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_value(kind: str, raw: str) -> Any:
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        return _parse_bool(raw)
    return raw


def _serialize_value(kind: str, value: Any) -> str:
    if kind == "bool":
        return "true" if bool(value) else "false"
    return str(value)


def env_runtime_config() -> dict[str, Any]:
    temperature = os.environ.get("LLM_CONVERTER_TEMPERATURE", "").strip()
    max_tokens = os.environ.get("LLM_CONVERTER_MAX_TOKENS", "").strip()
    return {
        "converter_model": qualify_model(
            os.environ.get("LLM_CONVERTER_MODEL", "") or DEFAULT_CONVERTER_MODEL,
            os.environ.get("LLM_CONVERTER_PROVIDER", ""),
        ),
        "converter_max_tokens": int(max_tokens) if max_tokens else DEFAULT_CONVERTER_MAX_TOKENS,
        # Faithful conversion wants the model's default sampling: no temperature unless
        # one is configured (the gateway drops it for models that reject it anyway).
        "converter_temperature": float(temperature) if temperature else 0.2,
        "converter_temperature_enabled": bool(temperature),
        "security_model": qualify_model(
            os.environ.get("CONTENT_SECURITY_LLM_MODEL", "") or DEFAULT_SECURITY_MODEL
        ),
    }


def _load_overrides(session_factory: Any | None) -> dict[str, str]:
    if session_factory is None:
        return {}
    try:
        from sqlalchemy import select

        from pipeline.db_models import AppSetting
        from pipeline.document_store import _session_scope

        keys = [spec.key for spec in _FIELDS.values()]
        keys += [spec.legacy_key for spec in _FIELDS.values() if spec.legacy_key]
        with _session_scope(session_factory) as s:
            rows = s.execute(select(AppSetting).where(AppSetting.key.in_(keys))).scalars().all()
            return {str(r.key): str(r.value) for r in rows}
    except Exception:
        return {}


def get_runtime_config(session_factory: Any | None) -> dict[str, Any]:
    now = time.monotonic()
    cached = _CACHE.get("effective")
    if cached is not None and now < float(_CACHE.get("expires_at", 0.0)):
        return dict(cached)

    env_default = env_runtime_config()
    effective = dict(env_default)
    overrides = _load_overrides(session_factory)
    for field_name, spec in _FIELDS.items():
        raw = overrides.get(spec.key)
        if raw is None and spec.legacy_key:
            raw = overrides.get(spec.legacy_key)
        if raw is None:
            continue
        try:
            effective[field_name] = _parse_value(spec.kind, raw)
        except Exception:
            # A malformed stored value is ignored; the default for that field stays.
            continue

    _CACHE["effective"] = dict(effective)
    _CACHE["env_default"] = dict(env_default)
    _CACHE["expires_at"] = now + _CACHE_TTL_SECS
    return effective


def get_runtime_config_state(
    session_factory: Any | None, available: list[str] | None = None
) -> dict[str, Any]:
    """Effective values, env defaults, and the model names to offer (the gateway's)."""
    effective = get_runtime_config(session_factory)
    env_default = _CACHE.get("env_default")
    if not isinstance(env_default, dict):
        env_default = env_runtime_config()
    models = set(available or [])
    models.update(str(effective.get(k) or "") for k in ("converter_model", "security_model"))
    models.discard("")
    return {
        "effective": effective,
        "env_default": dict(env_default),
        "available_models": sorted(models),
    }


def set_runtime_config(
    session_factory: Any | None,
    updates: dict[str, Any],
    *,
    updated_by: str = "api",
) -> dict[str, Any]:
    if session_factory is None:
        # The ledger exists in both backends; reaching this is a wiring bug, and
        # answering "saved" would silently drop the change.
        raise RuntimeError("runtime settings cannot be saved: no ledger session factory")

    from datetime import UTC, datetime

    from sqlalchemy import select

    from pipeline.db_models import AppSetting
    from pipeline.document_store import _session_scope

    now = datetime.now(UTC).isoformat()
    with _session_scope(session_factory) as s:
        for field_name, value in updates.items():
            spec = _FIELDS.get(field_name)
            if spec is None or value is None:
                continue
            row = s.execute(
                select(AppSetting).where(AppSetting.key == spec.key)
            ).scalar_one_or_none()
            serialized = _serialize_value(spec.kind, value)
            if row is None:
                s.add(
                    AppSetting(
                        key=spec.key,
                        value=serialized,
                        updated_by=updated_by,
                        updated_at=now,
                    )
                )
            else:
                row.value = serialized
                row.updated_by = updated_by
                row.updated_at = now

    _CACHE["expires_at"] = 0.0
    return get_runtime_config_state(session_factory)


@dataclass(frozen=True)
class ConverterSettings:
    model: str
    max_tokens: int
    temperature: float | None  # None: send no temperature


def converter_settings(runtime: dict[str, Any] | None, config: Config) -> ConverterSettings:
    """What document conversion sends: the runtime overrides, else the static config."""
    if runtime is None:
        return ConverterSettings(config.llm_converter_model, config.llm_converter_max_tokens, None)
    temperature = (
        float(runtime.get("converter_temperature", 0.2))
        if runtime.get("converter_temperature_enabled")
        else None
    )
    return ConverterSettings(
        model=str(runtime.get("converter_model") or config.llm_converter_model),
        max_tokens=int(runtime.get("converter_max_tokens") or config.llm_converter_max_tokens),
        temperature=temperature,
    )


def security_model(runtime: dict[str, Any] | None, config: Config) -> str:
    """The content-security screen's model: the runtime override, else the static config."""
    if runtime is not None and runtime.get("security_model"):
        return str(runtime["security_model"])
    return config.content_security_llm_model
