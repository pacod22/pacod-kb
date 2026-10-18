"""docker/pacod-id/pacod-id.toml: the kb realm and the UI's two clients, as pacod-id reads them."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

_TOML = Path(__file__).parent.parent / "docker" / "pacod-id" / "pacod-id.toml"
_ENV = {"KB_ID_ISSUER": "http://127.0.0.1:18070", "KB_UI_ORIGIN": "http://localhost:8080"}


def _load(env: dict[str, str] = _ENV) -> dict[str, Any]:
    """Expand ${VAR} the way pacod-id does (an unset or empty one is fatal), then parse."""

    def expand(m: re.Match[str]) -> str:
        value = env.get(m.group(1), "")
        if not value:
            raise KeyError(m.group(1))
        return value

    return tomllib.loads(re.sub(r"\$\{([A-Z0-9_]+)\}", expand, _TOML.read_text(encoding="utf-8")))


def _clients(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in cfg["client"]}


def test_one_kb_realm_on_the_kb_brand_closed_to_sign_up() -> None:
    cfg = _load()
    (realm,) = cfg["realm"]
    assert realm["name"] == "kb"
    assert realm["brand"] == "kb"
    assert realm["issuer"] == _ENV["KB_ID_ISSUER"]
    assert realm["self_signup"] is False


def test_ui_client_redirects_are_exact_and_built_from_the_origin() -> None:
    ui = _clients(_load())["pacod-kb-ui"]
    origin = _ENV["KB_UI_ORIGIN"]
    assert ui["kind"] == "confidential"
    assert ui["realm"] == "kb"
    assert ui["redirect_uris"] == [f"{origin}/kb/auth/callback"]
    assert ui["post_logout_redirect_uris"] == [f"{origin}/kb/login?signed_out=1"]
    assert ui["initiate_login_uri"] == f"{origin}/kb/login"
    assert ui["secret_env"] == "KB_ID_UI_SECRET"


def test_service_client_is_the_kb_product_with_the_same_audience() -> None:
    clients = _clients(_load())
    service = clients["pacod-kb-service"]
    assert service["kind"] == "service"
    assert service["product"] == "kb"
    assert service["secret_env"] == "KB_ID_SERVICE_SECRET"
    # Token exchange takes only user tokens issued for the service client's own audience.
    assert service["audience"] == clients["pacod-kb-ui"]["audience"] == "pacod-kb"


def test_every_variable_must_be_set() -> None:
    with pytest.raises(KeyError):
        _load({"KB_ID_ISSUER": "http://127.0.0.1:18070"})
