from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .models import PickPlaceScenario


_TUPLE3_FIELDS = {
    "cube_start",
    "place_target",
    "table_center",
    "table_size",
    "grasp_tcp_offset",
    "obstacle_center",
    "obstacle_size",
}


def _tuple_or_none(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(item) for item in value)


def _support_blocks(value: Any) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    blocks = []
    for item in value or ():
        if isinstance(item, dict):
            center = item["center"]
            size = item["size"]
        else:
            center, size = item
        blocks.append(
            (
                tuple(float(component) for component in center),
                tuple(float(component) for component in size),
            )
        )
    return tuple(blocks)


def scenario_from_json_dict(payload: dict[str, Any]) -> PickPlaceScenario:
    allowed = {field.name for field in fields(PickPlaceScenario)}
    data: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        if key == "support_blocks":
            data[key] = _support_blocks(value)
        elif key in _TUPLE3_FIELDS:
            data[key] = _tuple_or_none(value)
        elif key == "object_size":
            data[key] = tuple(float(item) for item in value)
        else:
            data[key] = value
    return PickPlaceScenario(**data)


def load_scenario_json(path: str | Path) -> PickPlaceScenario:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "projected_scenario" in payload:
        payload = payload["projected_scenario"]
    return scenario_from_json_dict(payload)
