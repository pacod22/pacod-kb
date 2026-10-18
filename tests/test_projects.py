from __future__ import annotations

import json

from pipeline.projects import ProjectRegistry


def test_create_project_uses_custom_collection_name(kb_root) -> None:
    registry = ProjectRegistry(kb_root)

    project = registry.create(
        name="Legal Research",
        description="EU legal corpus",
        collection_name="eu-legal-vectors",
    )

    assert project.slug == "legal-research"
    assert project.collection_name == "eu-legal-vectors"


def test_create_project_defaults_collection_name_to_slug(kb_root) -> None:
    registry = ProjectRegistry(kb_root)

    project = registry.create(name="Policy Drafts")

    assert project.slug == "policy-drafts"
    assert project.collection_name == "policy-drafts"


def test_create_project_rejects_duplicate_collection_name(kb_root) -> None:
    registry = ProjectRegistry(kb_root)
    registry.create(name="Team One", collection_name="shared-collection")

    try:
        registry.create(name="Team Two", collection_name="shared-collection")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "already assigned" in str(exc)


def test_load_legacy_projects_json_backfills_collection_name(kb_root) -> None:
    projects_file = kb_root / "projects.json"
    projects_file.write_text(
        json.dumps(
            {
                "legacy": {
                    "slug": "legacy",
                    "name": "Legacy",
                    "created_at": "2026-07-05T00:00:00+00:00",
                    "description": "old format",
                }
            }
        ),
        encoding="utf-8",
    )

    registry = ProjectRegistry(kb_root)
    project = registry.get("legacy")

    assert project is not None
    assert project.collection_name == "legacy"


def test_update_collection_name_updates_and_persists(kb_root) -> None:
    registry = ProjectRegistry(kb_root)
    created = registry.create(name="Team One")

    updated = registry.update_collection_name(created.slug, "team-one-v2")
    reloaded = ProjectRegistry(kb_root).get(created.slug)

    assert updated.collection_name == "team-one-v2"
    assert reloaded is not None
    assert reloaded.collection_name == "team-one-v2"


def test_update_collection_name_validates_uniqueness_and_empty(kb_root) -> None:
    registry = ProjectRegistry(kb_root)
    left = registry.create(name="Alpha", collection_name="alpha-col")
    registry.create(name="Beta", collection_name="beta-col")

    try:
        registry.update_collection_name(left.slug, "beta-col")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "already assigned" in str(exc)

    try:
        registry.update_collection_name(left.slug, "")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "cannot be empty" in str(exc)


