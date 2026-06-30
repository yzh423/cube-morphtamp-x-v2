from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .models import PickPlaceScenario, ReplayFrame
from .objects import object_half_height, object_spec
from .scenario_io import scenario_from_json_dict


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _tuple3(value: Any) -> tuple[float, float, float]:
    if value is None or len(value) != 3:
        raise ValueError(f"expected a 3-vector, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _tuple4(value: Any) -> tuple[float, float, float, float]:
    if value is None or len(value) != 4:
        raise ValueError(f"expected a quaternion 4-vector, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def frame_from_json_dict(payload: dict[str, Any]) -> ReplayFrame:
    return ReplayFrame(
        phase_name=str(payload["phase_name"]),
        tcp_position=_tuple3(payload["tcp_position"]),
        tcp_quat=_tuple4(payload.get("tcp_quat", (1.0, 0.0, 0.0, 0.0))),
        object_position=_tuple3(payload["object_position"]),
        object_quat=_tuple4(payload.get("object_quat", (1.0, 0.0, 0.0, 0.0))),
        gripper_width=float(payload["gripper_width"]),
        object_attached=bool(payload["object_attached"]),
        weld_active=bool(payload.get("weld_active", payload["object_attached"])),
        progress=float(payload.get("progress", 0.0)),
    )


def replay_payload_from_json(path: str | Path) -> tuple[PickPlaceScenario, tuple[ReplayFrame, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "scenario" not in payload or "frames" not in payload:
        raise ValueError("replay JSON must contain 'scenario' and 'frames'")
    scenario = scenario_from_json_dict(dict(payload["scenario"]))
    frames = tuple(frame_from_json_dict(dict(frame)) for frame in payload["frames"])
    return scenario, frames


def _object_half_extents(scenario: PickPlaceScenario) -> tuple[float, float, float]:
    size = tuple(float(value) for value in scenario.object_size)
    geom = scenario.object_geom_type
    if geom == "box":
        return (size[0], size[1], size[2])
    if geom == "sphere":
        return (size[0], size[0], size[0])
    if geom in {"cylinder", "capsule"}:
        return (size[0], size[0], size[1])
    return (float(scenario.cube_size) * 0.5,) * 3


def _support_surfaces(scenario: PickPlaceScenario) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    surfaces = [("table", scenario.table_center, scenario.table_size)]
    surfaces.extend(
        (f"support_{index}", center, size)
        for index, (center, size) in enumerate(scenario.support_blocks)
    )
    return surfaces


def _best_support_gap(
    scenario: PickPlaceScenario,
    object_position: tuple[float, float, float],
) -> tuple[float, str | None]:
    half_height = object_half_height(object_spec(scenario.object_type))
    bottom_z = float(object_position[2] - half_height)
    best_gap = -math.inf
    best_name = None
    x, y, _ = object_position
    for name, center, half in _support_surfaces(scenario):
        if abs(float(x) - float(center[0])) <= float(half[0]) and abs(float(y) - float(center[1])) <= float(half[1]):
            support_top = float(center[2] + half[2])
            gap = bottom_z - support_top
            if abs(gap) < abs(best_gap) or best_name is None:
                best_gap = gap
                best_name = name
    return (float(best_gap), best_name)


def _aabb_clearance(
    center_a: tuple[float, float, float],
    half_a: tuple[float, float, float],
    center_b: tuple[float, float, float],
    half_b: tuple[float, float, float],
) -> float:
    delta = np.abs(np.asarray(center_a, dtype=float) - np.asarray(center_b, dtype=float))
    overlap_distance = delta - (np.asarray(half_a, dtype=float) + np.asarray(half_b, dtype=float))
    outside = np.maximum(overlap_distance, 0.0)
    if np.any(overlap_distance > 0.0):
        return float(np.linalg.norm(outside))
    return float(np.max(overlap_distance))


def _json_number(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def evaluate_physical_evidence(
    scenario: PickPlaceScenario,
    frames: tuple[ReplayFrame, ...],
    *,
    position_tolerance: float = 0.035,
    retention_tolerance: float = 0.015,
    support_tolerance: float = 0.006,
    clearance_tolerance: float = 0.0,
) -> dict[str, Any]:
    failure_reasons: list[str] = []
    if not frames:
        return {
            "schema_version": 1,
            "evidence_scope": "geometric replay consistency; not force-controlled contact dynamics",
            "success": False,
            "failure_reasons": ["empty_replay"],
            "checks": {},
        }

    attached = tuple(frame for frame in frames if frame.object_attached)
    pre_attach = tuple(frame for frame in frames if frame.phase_name in {"home", "pregrasp", "grasp", "close_gripper"})
    max_pre_attach_motion = max(
        (_distance(frame.object_position, scenario.cube_start) for frame in pre_attach),
        default=0.0,
    )
    static_passed = max_pre_attach_motion <= 1e-6
    if not static_passed:
        failure_reasons.append("object_static_before_attach")

    max_retention_error = 0.0
    retention_passed = bool(attached)
    if attached:
        first = attached[0]
        attachment_offset = np.asarray(first.object_position, dtype=float) - np.asarray(first.tcp_position, dtype=float)
        for frame in attached:
            current_offset = np.asarray(frame.object_position, dtype=float) - np.asarray(frame.tcp_position, dtype=float)
            max_retention_error = max(max_retention_error, float(np.linalg.norm(current_offset - attachment_offset)))
        retention_passed = max_retention_error <= retention_tolerance
    if not retention_passed:
        failure_reasons.append("grasp_retention")

    start_gap, start_support = _best_support_gap(scenario, scenario.cube_start)
    target_gap, target_support = _best_support_gap(scenario, scenario.place_target)
    endpoints_supported = (
        start_support is not None
        and target_support is not None
    )
    if not endpoints_supported:
        failure_reasons.append("endpoint_support")

    release_error = _distance(frames[-1].object_position, scenario.place_target)
    support_gap, support_name = _best_support_gap(scenario, frames[-1].object_position)
    support_gap_error = (
        math.inf
        if support_name is None or target_support is None
        else float(support_gap - target_gap)
    )
    release_supported = (
        support_name is not None
        and target_support is not None
        and abs(support_gap_error) <= support_tolerance
        and release_error <= position_tolerance
    )
    if not release_supported:
        failure_reasons.append("release_support")

    min_obstacle_clearance = None
    obstacle_passed = True
    if scenario.obstacle_center is not None and scenario.obstacle_size is not None:
        half = _object_half_extents(scenario)
        min_obstacle_clearance = min(
            _aabb_clearance(frame.object_position, half, scenario.obstacle_center, scenario.obstacle_size)
            for frame in frames
        )
        obstacle_passed = float(min_obstacle_clearance) >= -abs(clearance_tolerance)
        if not obstacle_passed:
            failure_reasons.append("obstacle_clearance")

    checks = {
        "object_static_before_attach": {
            "passed": static_passed,
            "max_pre_attach_motion": _json_number(max_pre_attach_motion),
        },
        "grasp_retention": {
            "passed": retention_passed,
            "attached_frame_count": len(attached),
            "max_retention_error": _json_number(max_retention_error),
            "retention_tolerance": retention_tolerance,
        },
        "endpoint_support": {
            "passed": endpoints_supported,
            "start_support": start_support,
            "target_support": target_support,
            "start_support_gap": _json_number(start_gap),
            "target_support_gap": _json_number(target_gap),
            "note": (
                "Gaps are reported against MuJoCo support geometry. Pass/fail uses "
                "the planned scenario start/target support references so legacy "
                "full-size table-height scenarios remain auditable."
            ),
            "support_tolerance": support_tolerance,
        },
        "release_support": {
            "passed": release_supported,
            "release_error": _json_number(release_error),
            "support": support_name,
            "support_gap": _json_number(support_gap),
            "planned_target_gap": _json_number(target_gap),
            "support_gap_error": _json_number(support_gap_error),
            "position_tolerance": position_tolerance,
            "support_tolerance": support_tolerance,
        },
        "obstacle_clearance": {
            "passed": obstacle_passed,
            "min_object_obstacle_clearance": _json_number(min_obstacle_clearance),
            "clearance_tolerance": clearance_tolerance,
        },
    }
    return {
        "schema_version": 1,
        "evidence_scope": "geometric replay consistency; not force-controlled contact dynamics",
        "success": not failure_reasons,
        "failure_reasons": failure_reasons,
        "checks": checks,
    }


def evaluate_replay_json(
    path: str | Path,
    *,
    position_tolerance: float = 0.035,
    retention_tolerance: float = 0.015,
    support_tolerance: float = 0.006,
    clearance_tolerance: float = 0.0,
) -> dict[str, Any]:
    scenario, frames = replay_payload_from_json(path)
    return evaluate_physical_evidence(
        scenario,
        frames,
        position_tolerance=position_tolerance,
        retention_tolerance=retention_tolerance,
        support_tolerance=support_tolerance,
        clearance_tolerance=clearance_tolerance,
    )
