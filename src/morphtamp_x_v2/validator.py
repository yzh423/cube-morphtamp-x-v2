from __future__ import annotations

from typing import Any
import math

import numpy as np

from .models import PickPlaceScenario, ReplayFrame


def _distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def evaluate_replay(
    scenario: PickPlaceScenario,
    frames: tuple[ReplayFrame, ...],
    *,
    position_tolerance: float = 0.025,
) -> dict[str, Any]:
    if not frames:
        return {
            "success": False,
            "failure_reasons": ["empty_replay"],
            "release_error": None,
            "max_lift": None,
            "attached_frame_count": 0,
        }
    final_object = frames[-1].object_position
    release_error = _distance(final_object, scenario.place_target)
    max_lift = max(float(frame.object_position[2]) for frame in frames) - float(scenario.cube_start[2])
    attached_frames = [frame for frame in frames if frame.object_attached]
    early_motion = [
        frame
        for frame in frames
        if frame.phase_name in {"home", "pregrasp", "grasp", "close_gripper"}
        and _distance(frame.object_position, scenario.cube_start) > 1e-9
    ]
    failure_reasons: list[str] = []
    if release_error > position_tolerance:
        failure_reasons.append(f"release_error:{release_error:.6g}")
    if max_lift < scenario.lift_height * 0.75:
        failure_reasons.append(f"insufficient_lift:{max_lift:.6g}")
    if not attached_frames:
        failure_reasons.append("never_attached")
    if early_motion:
        failure_reasons.append("object_moved_before_attach")
    return {
        "success": not failure_reasons,
        "failure_reasons": failure_reasons,
        "release_error": None if not math.isfinite(release_error) else release_error,
        "max_lift": None if not math.isfinite(max_lift) else max_lift,
        "attached_frame_count": len(attached_frames),
        "total_frames": len(frames),
    }
