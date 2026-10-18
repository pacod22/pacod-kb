"""Route contract: the API surface must not change during refactors.

Snapshots hold the *ordered* route table (Starlette matches the first route
that fits, so moving routes between routers can let a parameterised path
shadow a literal one) and the OpenAPI schema. Regenerate deliberately with
``KB_UPDATE_ROUTE_SNAPSHOTS=1 uv run pytest tests/test_route_contract.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import iter_route_contexts

_SNAP_DIR = Path(__file__).parent / "snapshots"


def _routes(app: Any) -> list[list[Any]]:
    # FastAPI >= 0.143 keeps included routers as nodes in app.routes; walk the
    # effective routes, in match order, through its public iterator.
    out: list[list[Any]] = []
    for r in iter_route_contexts(app.routes):
        methods = sorted(r.methods or [])
        endpoint = r.endpoint
        name = getattr(endpoint, "__name__", type(r.original_route).__name__)
        out.append([r.path, methods, name])
    return out


def _apps() -> dict[str, Any]:
    from pipeline.api import app as pipeline_app
    from web.app import app as web_app

    return {"pipeline": pipeline_app, "web": web_app}


@pytest.mark.parametrize("name", ["pipeline", "web"])
def test_route_table_and_openapi_unchanged(name: str) -> None:
    app = _apps()[name]
    current = {"routes": _routes(app), "openapi": app.openapi()}
    snap = _SNAP_DIR / f"{name}_routes.json"
    if os.environ.get("KB_UPDATE_ROUTE_SNAPSHOTS") == "1" or not snap.exists():
        snap.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    expected = json.loads(snap.read_text(encoding="utf-8"))
    assert current["routes"] == expected["routes"]
    assert json.loads(json.dumps(current["openapi"], sort_keys=True)) == expected["openapi"]
