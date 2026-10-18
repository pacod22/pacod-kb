"""The local accounts of earlier releases, read once more so admins can invite them.

Before sign-in moved to pacod-id, the UI kept its own accounts in ``ui_users`` (same SQLite
file as the sessions). Nothing writes that table any more; the members page lists its active
rows that are not yet kb members or invited, with an Invite button pre-filled with the old
role. Read-only, and an install that never had the table simply has no former accounts.
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, inspect, text

from web.roles import ROLES

logger = logging.getLogger(__name__)


def former_accounts(url: str) -> list[tuple[str, str]]:
    """``(email, role)`` of the active ``ui_users`` rows, lowercased email; [] without it."""
    engine = create_engine(url)
    try:
        if not inspect(engine).has_table("ui_users"):
            return []
        with engine.connect() as con:
            rows = con.execute(text("SELECT email, role FROM ui_users WHERE is_active")).all()
    except Exception as exc:  # a damaged or foreign table: show none
        logger.warning("Former local accounts unreadable: %s", exc)
        return []
    finally:
        engine.dispose()
    out = []
    for email, role in rows:
        role = str(role or "viewer").lower()
        out.append((str(email).strip().lower(), role if role in ROLES else "viewer"))
    return sorted(set(out))
