from pathlib import Path

from fastapi.testclient import TestClient

from web.app import app


def test_favicon_redirects_to_static_asset() -> None:
    with TestClient(app) as client:
        response = client.get("/favicon.ico", follow_redirects=False)

    assert response.status_code in {302, 307}
    assert response.headers.get("location") == "/kb/static/favicon.ico"


def test_favicon_asset_bundle_exists() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    expected = {
        "favicon.ico",
        "favicon.svg",
        "favicon-16x16.png",
        "favicon-32x32.png",
        "apple-touch-icon.png",
        "site.webmanifest",
    }

    missing = [name for name in expected if not (static_dir / name).is_file()]
    assert not missing, f"Missing favicon assets: {missing}"

