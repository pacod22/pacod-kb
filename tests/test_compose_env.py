"""The compose files: each service its own environment and nothing else, on pacod-net alone.

docker-compose.yml is the app file (kb-pipeline, kb-ui), docker-compose.standalone.yml kb's own
copies of the shared components (pacod-demo/docs/NETWORK.md).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent
_FILES = ("docker-compose.yml", "docker-compose.standalone.yml")
_COMPOSE = {name: yaml.safe_load((_ROOT / name).read_text(encoding="utf-8")) for name in _FILES}
_APP: dict[str, Any] = _COMPOSE["docker-compose.yml"]["services"]
_SERVICES: dict[str, Any] = {
    name: service for doc in _COMPOSE.values() for name, service in doc["services"].items()
}

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
    missing = _variables_read("pipeline") - _env_names("kb-pipeline") - _LEGACY
    assert not missing, sorted(missing)


def test_the_pipeline_never_sees_identity_secrets() -> None:
    names = _env_names("kb-pipeline")
    assert not {n for n in names if n.startswith(("PID_", "KB_ID_", "ID_ADMIN"))}
    assert not names & _LEGACY


def test_the_ui_never_sees_the_gateway_or_identity_server_secrets() -> None:
    names = _env_names("kb-ui")
    assert "PACOD_GW_API_KEY" not in names
    assert not {n for n in names if n.startswith(("PGW_", "PID_"))}


def test_profile_gated_services_never_demand_variables() -> None:
    # ${VAR:?} is interpolated even with the service's profile off; `make check-env` checks.
    for name in _FILES:
        text = (_ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"\$\{[A-Z0-9_]+:?\?", text), name


def test_every_service_joins_pacod_net_and_only_id_traffic_joins_pacod_admin_net() -> None:
    # pacod-admin-net carries pacod-id's internal listener alone: pacod-id there under the alias
    # it binds the listener to, and the UI, its one caller here.
    external = {"external": True}
    for doc in _COMPOSE.values():
        assert doc["networks"] == {"pacod-net": external, "pacod-admin-net": external}
    for name, service in _SERVICES.items():
        networks = service["networks"]
        if name == "pacod-id":
            alias = {"aliases": ["pacod-id-internal"]}
            assert networks == {"pacod-net": None, "pacod-admin-net": alias}
            assert service["environment"]["PID_INTERNAL_LISTEN"] == "pacod-id-internal:8071"
        elif name == "kb-ui":
            assert networks == ["pacod-net", "pacod-admin-net"], name
        else:
            assert networks == ["pacod-net"], name
        assert service["container_name"] == name


def test_the_app_file_holds_kb_alone_and_runs_without_a_profile() -> None:
    assert set(_APP) == {"kb-pipeline", "kb-ui"}
    for name, service in _APP.items():
        assert "profiles" not in service, name
        # Shared components come from either stack: never a hard dependency.
        for dep, spec in (service.get("depends_on") or {}).items():
            if dep not in _APP:
                assert spec.get("required") is False, (name, dep)


def test_shared_components_carry_their_shared_names() -> None:
    standalone = set(_COMPOSE["docker-compose.standalone.yml"]["services"])
    assert standalone == {
        "pacod-qdrant",
        "pacod-postgres",
        "pacod-db-init",
        "pacod-id",
        "pacod-mailpit",
        "pacod-gw",
        "gw-mock",
    }


def test_env_example_has_no_comment_compose_would_read_as_a_value() -> None:
    # docker compose takes `KEY=   # comment` (a blank value, then a comment) as the value
    # "# comment"; such comments belong on the line above.
    lines = (_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    bad = [line for line in lines if re.match(r"^[A-Z0-9_]+=\s+#", line)]
    assert not bad, bad


def _env(service: str) -> dict[str, str]:
    env = _SERVICES[service].get("environment") or {}
    if isinstance(env, dict):
        return {k: str(v) for k, v in env.items()}
    return dict(str(item).split("=", 1) for item in env)


def test_every_service_rotates_its_log() -> None:
    # pacod-demo/docs/LOGGING.md, section 2: each file its own x-logging anchor.
    block = {"driver": "json-file", "options": {"max-size": "10m", "max-file": "5"}}
    for name, doc in _COMPOSE.items():
        assert doc["x-logging"] == block, name
    for name, service in _SERVICES.items():
        assert service.get("logging") == block, name


def test_kb_services_pass_the_log_switches() -> None:
    # kblog reads these (and only these) in both processes.
    assert _variables_read("kblog") == {"LOG_LEVEL", "LOG_FORMAT", "SERVICE_NAME"}
    for name in ("kb-pipeline", "kb-ui"):
        env = _env(name)
        assert env["SERVICE_NAME"] == name
        assert env["LOG_LEVEL"] == "${LOG_LEVEL:-info}"
        assert env["LOG_FORMAT"] == "${LOG_FORMAT:-json}"


def test_bundled_components_pass_their_log_switches_through() -> None:
    assert _env("pacod-id")["PID_LOG_FORMAT"] == "${PID_LOG_FORMAT:-text}"
    assert _env("pacod-id")["PID_LOG"] == "${PID_LOG:-info,sqlx=warn}"
    assert _env("pacod-gw")["PGW_LOG_FORMAT"] == "${PGW_LOG_FORMAT:-text}"
    assert _env("pacod-gw")["RUST_LOG"] == "${RUST_LOG:-info}"
