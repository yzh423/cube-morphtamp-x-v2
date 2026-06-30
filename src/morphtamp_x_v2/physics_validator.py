from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from .viewer import (
    _cube_address,
    _cube_qvel_address,
    _gripper_addresses,
    _joint_addresses,
    _mocap_id,
    _set_cube,
    _set_mocap_pose,
    _set_weld,
    _update_weld_relative_pose,
    _weld_id,
)


def _geom_body_name(mujoco: Any, model: Any, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")


def _geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")


def _contact_is_obstacle_violation(
    *,
    geom1_name: str,
    body1_name: str,
    geom2_name: str,
    body2_name: str,
) -> bool:
    names = {geom1_name, body1_name, geom2_name, body2_name}
    is_object = "v2_cube" in {body1_name, body2_name} or "v2_object" in {geom1_name, geom2_name}
    support_names = {name for name in names if name == "v2_table" or name.startswith("v2_support_")}
    if "v2_obstacle" in names:
        allowed = {"", "world", "v2_table", "v2_place_target", "v2_obstacle"} | support_names
        other_names = names - {"v2_obstacle"}
        return any(name not in allowed for name in other_names)
    if support_names:
        if is_object:
            return False
        allowed = {"", "world", "v2_table", "v2_place_target"} | support_names
        other_names = names - support_names
        return any(name not in allowed for name in other_names)
    return False


def _contact_is_gripper_object(
    *,
    geom1_name: str,
    body1_name: str,
    geom2_name: str,
    body2_name: str,
) -> bool:
    names = " ".join((geom1_name, body1_name, geom2_name, body2_name)).lower()
    object_side = "v2_cube" in {body1_name, body2_name} or "v2_object" in {geom1_name, geom2_name}
    gripper_side = any(token in names for token in ("finger", "gripper", "hand"))
    support_side = any(
        name == "v2_table" or name.startswith("v2_support_")
        for name in {geom1_name, body1_name, geom2_name, body2_name}
    )
    return object_side and gripper_side and not support_side


def _gripper_contact_side(
    *,
    geom1_name: str,
    body1_name: str,
    geom2_name: str,
    body2_name: str,
) -> str:
    names = " ".join((geom1_name, body1_name, geom2_name, body2_name)).lower()
    if "left" in names and "finger" in names:
        return "left_finger"
    if "right" in names and "finger" in names:
        return "right_finger"
    if "finger" in names:
        return "finger_unknown_side"
    if "hand" in names or "gripper" in names:
        return "palm_or_hand"
    return "unknown"


def _scan_obstacle_violations(mujoco: Any, model: Any, data: Any, *, phase_name: str) -> list[str]:
    violations: list[str] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        if float(contact.dist) > 0.001:
            continue
        geom1 = _geom_name(mujoco, model, int(contact.geom1))
        geom2 = _geom_name(mujoco, model, int(contact.geom2))
        body1 = _geom_body_name(mujoco, model, int(contact.geom1))
        body2 = _geom_body_name(mujoco, model, int(contact.geom2))
        if _contact_is_obstacle_violation(
            geom1_name=geom1,
            body1_name=body1,
            geom2_name=geom2,
            body2_name=body2,
        ):
            violations.append(f"{phase_name}:{geom1 or body1}/{geom2 or body2}:{float(contact.dist):.6g}")
    return violations


def _scan_gripper_object_contacts(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    phase_name: str,
) -> tuple[int, float, str | None, list[str], set[str], dict[str, float]]:
    count = 0
    max_penetration = 0.0
    max_example: str | None = None
    examples: list[str] = []
    sides: set[str] = set()
    max_penetration_by_side: dict[str, float] = {}
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = _geom_name(mujoco, model, int(contact.geom1))
        geom2 = _geom_name(mujoco, model, int(contact.geom2))
        body1 = _geom_body_name(mujoco, model, int(contact.geom1))
        body2 = _geom_body_name(mujoco, model, int(contact.geom2))
        if not _contact_is_gripper_object(
            geom1_name=geom1,
            body1_name=body1,
            geom2_name=geom2,
            body2_name=body2,
        ):
            continue
        count += 1
        side = _gripper_contact_side(
            geom1_name=geom1,
            body1_name=body1,
            geom2_name=geom2,
            body2_name=body2,
        )
        sides.add(side)
        penetration = max(0.0, -float(contact.dist))
        max_penetration_by_side[side] = max(
            penetration,
            max_penetration_by_side.get(side, 0.0),
        )
        example = f"{phase_name}:{geom1 or body1}/{geom2 or body2}:{float(contact.dist):.6g}"
        if penetration > max_penetration:
            max_penetration = penetration
            max_example = example
        if len(examples) < 8:
            examples.append(example)
    return count, max_penetration, max_example, examples, sides, max_penetration_by_side


def _body_position(mujoco: Any, model: Any, data: Any, body_name: str) -> np.ndarray:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise ValueError(f"body {body_name!r} not found")
    return np.asarray(data.xpos[body_id], dtype=float).copy()


def _json_number(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _build_dynamics_evidence(
    *,
    final_error: float,
    position_tolerance: float,
    weld_frames: int,
    max_weld_tracking_error: float,
    weld_tracking_tolerance: float,
    obstacle_collisions: int,
    gripper_object_contacts: int,
    gripper_object_max_penetration: float,
    gripper_object_contact_sides: tuple[str, ...] | set[str] | None = None,
    gripper_object_max_penetration_by_side: dict[str, float] | None = None,
    max_gripper_penetration: float | None = None,
    require_two_sided_grasp: bool = False,
    table_contacts: int,
    settle_steps: int,
) -> dict[str, Any]:
    weld_retention_passed = int(weld_frames) > 0
    reference_tracking_passed = (
        float(max_weld_tracking_error) <= float(weld_tracking_tolerance)
    )
    grasp_contact_passed = int(gripper_object_contacts) > 0
    contact_sides = set(gripper_object_contact_sides or ())
    two_sided_contact_passed = (
        not contact_sides
        or {"left_finger", "right_finger"}.issubset(contact_sides)
    )
    gripper_penetration_passed = (
        max_gripper_penetration is None
        or float(gripper_object_max_penetration) <= float(max_gripper_penetration)
    )
    penetration_by_side = {
        str(side): _json_number(value)
        for side, value in sorted((gripper_object_max_penetration_by_side or {}).items())
    }
    dominant_side = None
    if penetration_by_side:
        dominant_side = max(
            penetration_by_side,
            key=lambda side: float(penetration_by_side[side] or 0.0),
        )
    release_settle_passed = (
        float(final_error) <= float(position_tolerance)
        and int(table_contacts) > 0
    )
    obstacle_clearance_passed = int(obstacle_collisions) == 0
    checks = {
        "weld_retention": {
            "passed": weld_retention_passed,
            "weld_frames": int(weld_frames),
            "max_weld_tracking_error": _json_number(max_weld_tracking_error),
            "weld_tracking_tolerance": float(weld_tracking_tolerance),
        },
        "reference_tracking": {
            "passed": reference_tracking_passed,
            "diagnostic_only": True,
            "max_weld_tracking_error": _json_number(max_weld_tracking_error),
            "weld_tracking_tolerance": float(weld_tracking_tolerance),
            "note": (
                "Compares MuJoCo weld-carried object pose with scripted replay "
                "reference. It is diagnostic because Panda hand-body and TCP "
                "frames can differ while the object remains welded, released, "
                "and supported successfully."
            ),
        },
        "grasp_contact": {
            "passed": grasp_contact_passed,
            "gripper_object_contacts": int(gripper_object_contacts),
            "gripper_object_contact_sides": sorted(contact_sides),
            "gripper_object_max_penetration": _json_number(gripper_object_max_penetration),
        },
        "two_sided_grasp_contact": {
            "passed": two_sided_contact_passed,
            "required_sides": ["left_finger", "right_finger"],
            "observed_sides": sorted(contact_sides),
            "required_for_success": bool(require_two_sided_grasp),
            "diagnostic_only": not bool(require_two_sided_grasp),
        },
        "gripper_penetration": {
            "passed": gripper_penetration_passed,
            "gripper_object_max_penetration": _json_number(gripper_object_max_penetration),
            "gripper_object_max_penetration_by_side": penetration_by_side,
            "dominant_penetration_side": dominant_side,
            "max_allowed_penetration": _json_number(max_gripper_penetration),
        },
        "release_settle": {
            "passed": release_settle_passed,
            "final_error": _json_number(final_error),
            "position_tolerance": float(position_tolerance),
            "table_contacts": int(table_contacts),
            "settle_steps": int(settle_steps),
        },
        "obstacle_clearance": {
            "passed": obstacle_clearance_passed,
            "obstacle_collisions": int(obstacle_collisions),
        },
    }
    failure_reasons = [
        name for name, check in checks.items()
        if not bool(check["passed"]) and not bool(check.get("diagnostic_only", False))
    ]
    return {
        "schema_version": 1,
        "evidence_scope": (
            "MuJoCo equality-weld replay dynamics; conservative simulation evidence, "
            "not force-controlled real grasp validation"
        ),
        "success": not failure_reasons,
        "failure_reasons": failure_reasons,
        "checks": checks,
    }


def validate_physics_replay(
    xml_path: str | Path,
    replay_path: str | Path,
    *,
    settle_steps: int = 80,
    position_tolerance: float = 0.035,
    weld_tracking_tolerance: float = 0.03,
    frame_substeps: int = 4,
    max_gripper_penetration: float | None = None,
    require_two_sided_grasp: bool = False,
) -> dict[str, Any]:
    try:
        import mujoco
    except ModuleNotFoundError as error:
        raise RuntimeError("physics validation requires mujoco") from error

    payload = json.loads(Path(replay_path).read_text(encoding="utf-8"))
    joint_replay = payload.get("joint_replay")
    if not joint_replay:
        raise ValueError("replay JSON does not contain joint_replay")
    scenario = payload["scenario"]
    target = np.asarray(scenario["place_target"], dtype=float)
    frames = tuple(joint_replay["frames"])

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_addr = _joint_addresses(mujoco, model, list(joint_replay["joint_names"]))
    gripper_addr = _gripper_addresses(mujoco, model)
    cube_addr = _cube_address(mujoco, model)
    cube_qvel = _cube_qvel_address(mujoco, model)
    weld = _weld_id(mujoco, model)
    mocap = _mocap_id(mujoco, model)

    weld_was_active = False
    weld_frames = 0
    obstacle_violations: list[str] = []
    gripper_object_contacts = 0
    gripper_object_max_penetration = 0.0
    gripper_object_max_penetration_example: str | None = None
    gripper_object_examples: list[str] = []
    gripper_object_contact_sides: set[str] = set()
    gripper_object_max_penetration_by_side: dict[str, float] = {}
    max_weld_tracking_error = 0.0
    substeps = max(int(frame_substeps), 1)
    for frame in frames:
        data.qpos[joint_addr] = np.asarray(frame["q"], dtype=float)
        if len(gripper_addr):
            data.qpos[gripper_addr] = float(frame["gripper_width"]) / float(len(gripper_addr))
        weld_active = bool(frame["weld_active"])
        tcp_position = list(frame.get("tcp_position", frame["object_position"]))
        tcp_quat = list(frame.get("tcp_quat", frame.get("object_quat", [1.0, 0.0, 0.0, 0.0])))
        if weld_active:
            _set_mocap_pose(data, mocap, position=tcp_position, quat=tcp_quat)
        if weld_active and not weld_was_active:
            _set_cube(
                data,
                cube_addr,
                list(frame["object_position"]),
                cube_qvel,
                quat=list(frame.get("object_quat", [1.0, 0.0, 0.0, 0.0])),
            )
            mujoco.mj_forward(model, data)
            _update_weld_relative_pose(mujoco, model, data, weld)
        if not weld_active:
            _set_cube(
                data,
                cube_addr,
                list(frame["object_position"]),
                cube_qvel,
                quat=list(frame.get("object_quat", [1.0, 0.0, 0.0, 0.0])),
            )
        _set_weld(data, weld, weld_active)
        for _ in range(substeps):
            if weld_active:
                _set_mocap_pose(data, mocap, position=tcp_position, quat=tcp_quat)
            mujoco.mj_step(model, data)
            obstacle_violations.extend(_scan_obstacle_violations(mujoco, model, data, phase_name=str(frame["phase_name"])))
            if weld_active or str(frame["phase_name"]) in {"close_gripper", "attach"}:
                count, penetration, max_example, examples, sides, side_penetrations = _scan_gripper_object_contacts(
                    mujoco,
                    model,
                    data,
                    phase_name=str(frame["phase_name"]),
                )
                gripper_object_contacts += count
                gripper_object_contact_sides.update(sides)
                for side, side_penetration in side_penetrations.items():
                    gripper_object_max_penetration_by_side[side] = max(
                        gripper_object_max_penetration_by_side.get(side, 0.0),
                        side_penetration,
                    )
                if penetration > gripper_object_max_penetration:
                    gripper_object_max_penetration = penetration
                    gripper_object_max_penetration_example = max_example
                gripper_object_examples.extend(examples)
        if weld_active:
            cube_position = _body_position(mujoco, model, data, "v2_cube")
            reference_position = np.asarray(frame["object_position"], dtype=float)
            max_weld_tracking_error = max(
                max_weld_tracking_error,
                float(np.linalg.norm(cube_position - reference_position)),
            )
        weld_was_active = weld_active
        weld_frames += int(weld_active)

    _set_weld(data, weld, False)
    for _ in range(max(int(settle_steps), 0)):
        mujoco.mj_step(model, data)
        obstacle_violations.extend(_scan_obstacle_violations(mujoco, model, data, phase_name="settle"))

    cube_position = _body_position(mujoco, model, data, "v2_cube")
    final_error = float(np.linalg.norm(cube_position - target))
    table_contacts = 0
    cube_contacts = 0
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        if float(contact.dist) > 0.001:
            continue
        body1 = _geom_body_name(mujoco, model, int(contact.geom1))
        body2 = _geom_body_name(mujoco, model, int(contact.geom2))
        pair = {body1, body2}
        if "v2_cube" in pair:
            cube_contacts += 1
            if "world" in pair or "v2_table" in pair or any("table" in item for item in pair):
                table_contacts += 1
    unique_obstacle_violations = tuple(dict.fromkeys(obstacle_violations))
    unique_gripper_examples = tuple(dict.fromkeys(gripper_object_examples))
    evidence = _build_dynamics_evidence(
        final_error=final_error,
        position_tolerance=position_tolerance,
        weld_frames=weld_frames,
        max_weld_tracking_error=max_weld_tracking_error,
        weld_tracking_tolerance=weld_tracking_tolerance,
        obstacle_collisions=len(unique_obstacle_violations),
        gripper_object_contacts=gripper_object_contacts,
        gripper_object_contact_sides=tuple(sorted(gripper_object_contact_sides)),
        gripper_object_max_penetration=gripper_object_max_penetration,
        gripper_object_max_penetration_by_side=gripper_object_max_penetration_by_side,
        max_gripper_penetration=max_gripper_penetration,
        require_two_sided_grasp=require_two_sided_grasp,
        table_contacts=table_contacts,
        settle_steps=int(settle_steps),
    )
    evidence.update({
        "final_object_position": [float(item) for item in cube_position],
        "target_position": [float(item) for item in target],
        "final_error": final_error,
        "max_weld_tracking_error": max_weld_tracking_error,
        "weld_frames": weld_frames,
        "cube_contacts": cube_contacts,
        "table_contacts": table_contacts,
        "obstacle_collisions": len(unique_obstacle_violations),
        "obstacle_collision_examples": list(unique_obstacle_violations[:8]),
        "grasp_contact_present": gripper_object_contacts > 0,
        "gripper_object_contacts": gripper_object_contacts,
        "gripper_object_contact_sides": sorted(gripper_object_contact_sides),
        "gripper_object_max_penetration": gripper_object_max_penetration,
        "gripper_object_max_penetration_by_side": {
            side: float(value)
            for side, value in sorted(gripper_object_max_penetration_by_side.items())
        },
        "gripper_object_max_penetration_example": gripper_object_max_penetration_example,
        "gripper_object_contact_examples": list(unique_gripper_examples[:8]),
        "settle_steps": int(settle_steps),
        "frame_substeps": substeps,
    })
    return evidence
