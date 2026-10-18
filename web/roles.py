"""kb's roles, least to most privileged: pacod-id's kb.viewer, kb.editor, kb.admin."""

from __future__ import annotations

ROLES: tuple[str, ...] = ("viewer", "editor", "admin")
_RANK = {role: i for i, role in enumerate(ROLES)}


def role_satisfies(user_role: str | None, required: str) -> bool:
    """True when ``user_role`` is at least as privileged as ``required``."""
    return _RANK.get(user_role or "", -1) >= _RANK.get(required, len(ROLES))
