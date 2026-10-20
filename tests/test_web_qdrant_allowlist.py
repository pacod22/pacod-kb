"""The Qdrant pages show and touch only kb's collections (Qdrant may be shared on pacod-net).

kb's are the names with KB_QDRANT_COLLECTION_PREFIX ("kb_") and the collections kb's projects own
(GET /projects on the pipeline); pacod-ap's pacod_* collections, or anything else, are a 404.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tests.fake_id import FakeId
from tests.web_support import login_as, web_client

_FOREIGN = "pacod_starter_kb"
_PROJECT_COLLECTION = "legal"  # a project's collection from before the prefix


def _pipeline(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/projects":
        projects = [
            {"slug": "kb_chunks", "collection_name": "kb_chunks"},
            {"slug": "legal", "collection_name": _PROJECT_COLLECTION},
        ]
        return httpx.Response(200, json={"projects": projects, "active": "kb_chunks"})
    return httpx.Response(200, json={})


@pytest.fixture(scope="module")
def fake() -> FakeId:
    return FakeId()


@pytest.fixture()
def qdrant() -> MagicMock:
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name=n) for n in ("kb_chunks", _PROJECT_COLLECTION, _FOREIGN)]
    )
    client.get_collection.return_value = SimpleNamespace(points_count=3, indexed_vectors_count=3)
    client.retrieve.return_value = []
    return client


@pytest.fixture()
def client(fake: FakeId, qdrant: MagicMock) -> Iterator[TestClient]:
    from web.app import app
    from web.dependencies import get_qdrant_client

    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    try:
        with web_client(fake, pipeline=httpx.MockTransport(_pipeline)) as c:
            login_as(c, fake, "admin")
            yield c
    finally:
        app.dependency_overrides.pop(get_qdrant_client, None)


def _collection_routes() -> list[tuple[str, str]]:
    from web.routers.qdrant import router

    out: list[tuple[str, str]] = []
    for route in router.routes:
        assert isinstance(route, APIRoute)
        if "{collection}" not in route.path:
            continue
        path = route.path.replace("{collection}", _FOREIGN)
        path = re.sub(r"\{[a-z_]+\}", "x", path)
        out.extend((method, path) for method in sorted(route.methods))
    return out


def test_every_collection_route_is_covered() -> None:
    # Browse, search, export, points, delete, rename, snapshots: the walk below sees them all.
    assert len(_collection_routes()) >= 13


@pytest.mark.parametrize(("method", "path"), _collection_routes())
def test_a_foreign_collection_is_not_found(
    client: TestClient, qdrant: MagicMock, method: str, path: str
) -> None:
    r = client.request(method, path, follow_redirects=False)
    assert r.status_code == 404, (method, path, r.status_code)
    # Qdrant never heard of the foreign collection.
    touched = {str(a) for c in qdrant.mock_calls for a in (*c.args, *c.kwargs.values())}
    assert _FOREIGN not in touched


def test_the_list_shows_only_kb_collections(client: TestClient) -> None:
    r = client.get("/kb/qdrant/")
    assert r.status_code == 200
    assert "kb_chunks" in r.text
    assert _PROJECT_COLLECTION in r.text
    assert _FOREIGN not in r.text


@pytest.mark.parametrize("name", ["kb_chunks", _PROJECT_COLLECTION])
def test_kb_collections_are_reachable(client: TestClient, name: str) -> None:
    r = client.get(f"/kb/qdrant/{name}/export", params={"point_ids": ["p1"]})
    assert r.status_code == 200
    assert r.json()["collection"] == name


def test_new_collections_need_the_prefix(client: TestClient, qdrant: MagicMock) -> None:
    r = client.post("/kb/qdrant/", data={"name": "pacod_other_kb"})
    assert r.status_code == 400
    assert "kb_" in r.text
    qdrant.create_collection.assert_not_called()

    r = client.post("/kb/qdrant/", data={"name": "kb_new"})
    assert r.status_code == 200
    assert qdrant.create_collection.call_args.kwargs["collection_name"] == "kb_new"


def test_a_rename_keeps_the_prefix(client: TestClient, qdrant: MagicMock) -> None:
    r = client.post("/kb/qdrant/kb_chunks/rename", data={"new_name": "pacod_x"})
    assert r.status_code == 400
    qdrant.create_collection.assert_not_called()
    qdrant.delete_collection.assert_not_called()


def test_an_empty_prefix_allows_every_collection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_QDRANT_COLLECTION_PREFIX", "")
    assert _FOREIGN in client.get("/kb/qdrant/").text
    r = client.get(f"/kb/qdrant/{_FOREIGN}/export", params={"point_ids": ["p1"]})
    assert r.status_code == 200


def test_without_the_pipeline_only_the_prefix_counts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unreachable(_client: Any) -> set[str]:
        return set()

    monkeypatch.setattr("web.routers.qdrant._project_collections", unreachable)
    assert client.get("/kb/qdrant/kb_chunks/export", params={"point_ids": ["p"]}).status_code == 200
    r = client.get(f"/kb/qdrant/{_PROJECT_COLLECTION}/export", params={"point_ids": ["p"]})
    assert r.status_code == 404
