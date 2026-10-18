"""Structured lifecycle event logging into manifest JSON files."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pipeline.fs_layout import write_json_atomic


def append_event(manifest_path: Path, event: str, detail: str = "") -> None:
    """Append a structured lifecycle event to *manifest_path*'s ``events`` list.

    Reads the manifest, appends ``{"event": ..., "at": <ISO-UTC>, "detail": ...}``,
    and writes it back.  Safe to call on manifests that pre-date the events field.
    """
    data: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = data.get("events")
    events: list[dict[str, str]] = raw if isinstance(raw, list) else []
    events.append(
        {
            "event": event,
            "at": datetime.now(UTC).isoformat(),
            "detail": detail,
        }
    )
    data["events"] = events
    write_json_atomic(manifest_path, data)
