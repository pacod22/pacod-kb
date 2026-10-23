"""The UI uses the pacod design (pacod-id's @pacod/ui tokens, plain CSS), not Bootstrap.

Every template and every piece of HTML a router builds may use only the class vocabulary of
web/static/style.css: these tests refuse Bootstrap's classes and attributes, and check that both
page frames load the vendored tokens and fonts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
TEMPLATES = sorted((WEB / "templates").rglob("*.html"))
ROUTERS = sorted((WEB / "routers").glob("*.py"))

# Classes that exist only in Bootstrap (style.css defines none of them).
_BOOTSTRAP_CLASS = re.compile(
    r"^("
    r"btn-outline-[a-z]+|btn-(secondary|success|info|warning|light|dark|group(-sm)?|close(-white)?)"
    r"|form-(control|select|check|label|text|switch|range)(-[a-z]+)?"
    r"|col(-[a-z0-9]+)*|g-[0-5]|gx-[0-5]|gy-[0-5]|vr"
    r"|d-(flex|none|block|inline|inline-block|inline-flex|grid|table-cell)"
    r"|d-(sm|md|lg|xl)-[a-z-]+"
    r"|align-(items|self|content|baseline|middle)(-[a-z]+)?|justify-content-[a-z]+"
    r"|flex-(wrap|grow|shrink|column|row|fill|nowrap)(-[0-9])?"
    r"|gap-[0-5]|[mp][tbsexy]?-(auto|[0-5])|[mp][tbsexy]?-(sm|md|lg)-[0-5a-z]+"
    r"|text-(muted|end|start|nowrap|break|truncate|secondary|white|dark|light|primary|info"
    r"|decoration-none|uppercase|lowercase|bg-[a-z]+|body|black)"
    r"|fw-[a-z]+|fs-[1-6]|lh-[a-z0-9]+|bg-[a-z]+|h-100|w-100|w-50|mw-100|min-vh-100|vh-100"
    r"|alert(-[a-z]+)?|table-(sm|hover|light|dark|responsive|striped|bordered|borderless)"
    r"|card-(body|header|footer|title|text)|spinner-(border|grow)(-sm)?|progress(-bar)?"
    r"|nav-(pills|tabs|link|item)|navbar(-[a-z]+)?|list-group(-[a-z]+)?"
    r"|modal(-[a-z]+)?|offcanvas(-[a-z]+)?|fade|show|collapse|dropdown(-[a-z]+)?"
    r"|badge-(xs|xxs|micro)|rounded(-[a-z0-9]+)?|shadow(-[a-z]+)?|border(-[a-z0-9]+)?"
    r"|font-monospace|sticky-top|fixed-top|position-[a-z]+|overflow-[a-z]+|float-[a-z]+"
    r"|h[1-6]|display-[1-6]|lead|visually-hidden-focusable|text-xxs|text-xs|text-s|btn-sm-block"
    r")$"
)
_CLASS_ATTR = re.compile(r"""class\s*=\s*(["'])(.*?)\1""", re.S)
_JINJA = re.compile(r"{%.*?%}|{{.*?}}|{#.*?#}", re.S)


def _bootstrap_classes(text: str) -> list[str]:
    found: list[str] = []
    for match in _CLASS_ATTR.finditer(text):
        value = _JINJA.sub(" ", match.group(2))
        for token in value.split():
            # Quoted alternatives inside a Jinja expression ('btn-primary' if x else ...).
            token = token.strip("'\"")
            if _BOOTSTRAP_CLASS.match(token):
                found.append(token)
    # Class names chosen by an expression: {{ 'a' if x else 'b' }} inside class="...".
    for expr in re.findall(r"class\s*=\s*[\"'][^\"']*?{{(.*?)}}", text, re.S):
        for token in re.findall(r"'([a-z0-9-]+)'", expr):
            if _BOOTSTRAP_CLASS.match(token):
                found.append(token)
    return found


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: str(p.relative_to(WEB / "templates")))
def test_template_uses_no_bootstrap(path: Path) -> None:
    text = path.read_text()
    assert "bootstrap" not in text.lower(), "Bootstrap is not loaded any more"
    assert "data-bs-" not in text
    assert _bootstrap_classes(text) == []


@pytest.mark.parametrize("path", ROUTERS, ids=lambda p: p.name)
def test_router_html_uses_no_bootstrap(path: Path) -> None:
    assert _bootstrap_classes(path.read_text()) == []


def test_frames_load_the_pacod_design() -> None:
    head = (WEB / "templates" / "_head_assets.html").read_text()
    links = re.findall(r'href="/kb/static/([^"]+\.css)"', head)
    assert links == ["pacod-ui/tokens.css", "pacod-ui/console.css", "fonts/fonts.css", "style.css"]
    for link in links:
        assert (WEB / "static" / link).is_file(), link
    for frame in ("base.html", "auth_base.html"):
        assert '{% include "_head_assets.html" %}' in (WEB / "templates" / frame).read_text()


def test_fonts_css_points_at_vendored_files() -> None:
    css = (WEB / "static" / "fonts" / "fonts.css").read_text()
    files = re.findall(r"url\(([^)]+\.woff2)\)", css)
    assert files, "no font faces"
    for f in files:
        assert (WEB / "static" / "fonts" / f).is_file(), f
    for family in ("Inter Variable", "Manrope Variable", "JetBrains Mono Variable"):
        assert family in css


def test_tokens_are_pacod_ui() -> None:
    tokens = (WEB / "static" / "pacod-ui" / "tokens.css").read_text()
    console = (WEB / "static" / "pacod-ui" / "console.css").read_text()
    assert "--pid-accent:" in tokens and "light-dark(" in tokens
    assert "--accent: var(--pid-accent);" in console
