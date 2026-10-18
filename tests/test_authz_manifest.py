"""kb's manifest (web/authz/kb.yaml) through pacod-id's rules engine."""

from __future__ import annotations

from pathlib import Path

import pytest
from pacod_id.authz import Manifest, ManifestError
from pacod_id.vectors import run

_ROOT = Path(__file__).parent.parent
_MANIFEST = _ROOT / "web" / "authz" / "kb.yaml"


def _load(name: str) -> Manifest:
    if name != "kb":
        raise ManifestError("unknown", name)
    return Manifest.from_yaml(_MANIFEST.read_text(encoding="utf-8"))


def test_manifest_is_valid_and_has_no_implicit_role() -> None:
    manifest = _load("kb")
    assert manifest.product == "kb"
    assert {r.name for r in manifest.roles()} == {"kb.viewer", "kb.editor", "kb.admin"}
    assert not any(r.implicit for r in manifest.roles())


def test_vectors() -> None:
    checks, failures = run(
        (Path(__file__).parent / "authz" / "kb_vectors.yaml").read_text(encoding="utf-8"), _load
    )
    assert checks >= 18
    assert failures == []


@pytest.mark.parametrize("role", ["viewer", "editor", "admin"])
def test_every_role_is_held_at_the_platform(role: str) -> None:
    held = _load("kb").role(f"kb.{role}")
    assert held is not None
    assert held.at == ("platform",)
