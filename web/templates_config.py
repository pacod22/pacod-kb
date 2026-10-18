"""Shared Jinja2 templates instance — imported by app and routers."""

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates

from web.roles import role_satisfies


def _current_user_processor(request: Request) -> dict[str, Any]:
    """Expose the signed-in person (and a role helper) to every template.

    Only reads what the auth dependencies resolved for this request
    (``request.state.principal``); it never resolves the person itself.
    """
    user = getattr(request.state, "principal", None)
    identity = getattr(request.app.state, "identity", None)
    return {
        "current_user": user,
        "user_can": (lambda required: user is not None and role_satisfies(user.role, required)),
        "id_account_url": identity.settings.account_url if identity is not None else "",
        "active_project": getattr(request.state, "active_project", ""),
        "sidebar_projects": getattr(request.state, "sidebar_projects", []),
    }


templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates"),
    context_processors=[_current_user_processor],
)
