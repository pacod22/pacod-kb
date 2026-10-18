"""docker-compose.yml gives each service its own environment, and nothing else."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent
_COMPOSE = yaml.safe_load((_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
_SERVICES: dict[str, Any] = _COMPOSE["services"]

# Read by the pipeline only to warn that they are ignored (config.LEGACY_ENV); never passed.
_LEGACY = {
    "ANTHROPIC_API_KEY",
    "LLM_CONVERTER_API_KEY",
    "LLM_CONVERTER_BASE_URL",
    "SECRETS_KEY",
    "TABLE_RECONSTRUCTION_MODEL",
    "VLM_TABLE_RECONSTRUCTION_ENABLED",
}


def _env_names(service: str) -> set[str]:
    env = _SERVICES[service].get("environment") or {}
    if isinstance(env, dict):
        return set(env)
    return {str(item).split("=", 1)[0] for item in env}


def _variables_read(package: str) -> set[str]:
    """Every environment variable *package* reads by a literal name."""
    readers = {"get", "getenv", "_int_env", "_float_env", "_env_bool", "_limit"}
    names: set[str] = set()
    for path in (_ROOT / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and node.args):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in readers:
                continue
            if name == "get" and not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
            ):
                continue  # dict.get, not os.environ.get
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.isupper():
                    names.add(arg.value)
    return names


def test_no_service_receives_the_whole_env_file() -> None:
    for name, service in _SERVICES.items():
        assert "env_file" not in service, name


def test_the_pipeline_receives_every_variable_it_reads() -> None:
    missing = _variables_read("pipeline") - _env_names("pipeline") - _LEGACY
    assert not missing, sorted(missing)


def test_the_pipeline_never_sees_identity_secrets() -> None:
    names = _env_names("pipeline")
    assert not {n for n in names if n.startswith(("PID_", "KB_ID_", "ID_ADMIN"))}
    assert not names & _LEGACY


def test_the_ui_never_sees_the_gateway_or_identity_server_secrets() -> None:
    names = _env_names("ui")
    assert "PACOD_GW_API_KEY" not in names
    assert not {n for n in names if n.startswith(("PGW_", "PID_"))}


def test_profile_gated_services_never_demand_variables() -> None:
    # ${VAR:?} is interpolated even with the service's profile off; `make check-env` checks.
    text = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert not re.search(r"\$\{[A-Z0-9_]+:?\?", text)


def test_pacod_id_reaches_no_pipeline_network() -> None:
    nets = {name: set(_SERVICES[name].get("networks", [])) for name in _SERVICES}
    assert not nets["pipeline"] & nets["id"]
    assert not nets["pipeline"] & nets["id-postgres"]
    assert nets["ui"] & nets["id"]
    assert nets["pipeline"] & nets["gw"]


def test_env_example_has_no_comment_compose_would_read_as_a_value() -> None:
    # docker compose takes `KEY=   # comment` (a blank value, then a comment) as the value
    # "# comment"; such comments belong on the line above.
    lines = (_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    bad = [line for line in lines if re.match(r"^[A-Z0-9_]+=\s+#", line)]
    assert not bad, bad
