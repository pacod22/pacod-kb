"""The signed-in person, for the pipeline: every UI call to it says who acts (X-KB-Actor).

``current_principal`` sets the context variable; the shared pipeline client's request hook
adds the header, so no call site changes. The pipeline trusts it only alongside the API key
and records it on jobs, settings changes and usage (and the gateway's ``x-gw-user-id``).
"""

from __future__ import annotations

from contextvars import ContextVar

import httpx

current_actor: ContextVar[str | None] = ContextVar("kb_ui_actor", default=None)


async def add_actor_header(request: httpx.Request) -> None:
    actor = current_actor.get()
    if actor:
        request.headers["X-KB-Actor"] = actor
