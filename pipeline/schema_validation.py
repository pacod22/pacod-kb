"""Cached JSON-schema validation for pipeline/schemas/*.json.

``jsonschema.validate`` re-reads nothing but re-checks the schema and builds a
new validator on every call; with one call per chunk that dominated indexing
of large documents. Validators are built once per schema file.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema.exceptions import best_match

_SCHEMA_DIR = Path(__file__).parent / "schemas"


@functools.cache
def _validator(name: str) -> Any:
    schema: Any = json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))
    cls: Any = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def validate(data: dict[str, Any], schema_name: str) -> None:
    """Raise ``jsonschema.ValidationError`` (best match) if *data* is invalid."""
    error = best_match(_validator(schema_name).iter_errors(data))
    if error is not None:
        raise error
