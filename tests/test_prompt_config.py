from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.prompt_config import get_prompt_text, list_prompt_state, set_prompt_text


@pytest.fixture()
def session_factory(tmp_path: Path) -> Any:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from pipeline.db_models import Base

    engine = create_engine(f"sqlite:///{tmp_path}/prompt_config.db")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_prompt_defaults_without_db() -> None:
    state = list_prompt_state(None)
    assert state["prompts"]
    ids = {p["id"] for p in state["prompts"]}
    assert "converter_system" in ids
    assert "price_extraction_user" in ids


def test_set_and_get_prompt_override(session_factory: Any) -> None:
    updated = set_prompt_text(
        session_factory,
        "converter_user_text",
        "Custom conversion template:\n{document_text}",
        updated_by="test",
    )
    row = next(p for p in updated["prompts"] if p["id"] == "converter_user_text")
    assert row["overridden"] is True
    assert "Custom conversion template" in row["value"]

    text = get_prompt_text(session_factory, "converter_user_text")
    assert text.startswith("Custom conversion")


def test_reset_prompt_to_default_removes_override(session_factory: Any) -> None:
    defaults = list_prompt_state(session_factory)
    default_value = next(p for p in defaults["prompts"] if p["id"] == "wrap_untrusted_preamble")[
        "default"
    ]

    set_prompt_text(
        session_factory,
        "wrap_untrusted_preamble",
        "override",
        updated_by="test",
    )
    state = set_prompt_text(
        session_factory,
        "wrap_untrusted_preamble",
        default_value,
        updated_by="test",
    )

    row = next(p for p in state["prompts"] if p["id"] == "wrap_untrusted_preamble")
    assert row["overridden"] is False
    assert row["value"] == default_value


def test_unknown_prompt_raises_key_error(session_factory: Any) -> None:
    with pytest.raises(KeyError):
        set_prompt_text(session_factory, "nope", "x", updated_by="test")

