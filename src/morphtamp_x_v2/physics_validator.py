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
    _set_cube,
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
) -> tuple[int, float, str | None, list[str]]:
    count = 0
    max_penetration = 0.0
    max_example: str | None = None
    examples: list[str] = []
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
        penetration = max(0.0, -float(contact.dist))
        example = f"{phase_name}:{geom1 or body1}/{geom2 or body2}:{float(contact.dist):.6g}"
        if penetration > max_penetration:
            max_penetration = penetration
            max_example = example
        if len(examples) < 8:
            examples.append(example)
    return count, max_penetration, max_example, examples


def _body_position(mujoco: Any, model: Any, data: Any, body_name: str) -> np.ndarray:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise ValueError(f"body {body_name!r} not found")
    return np.asarray(data.xpos[body_id], dtype=float).copy()


def validate_physics_replay(
    xml_path: str | Path,
    replay_path: str | Path,
    *,
    settle_steps: int = 80,
    position_tolerance: float = 0.035,
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

    weld_was_active = False
    weld_frames = 0
    obstacle_violations: list[str] = []
    gripper_object_contacts = 0
    gripper_object_max_penetration = 0.0
    gripper_object_max_penetration_example: str | None = None
    gripper_object_examples: list[str] = []
    for frame in frames:
        data.qpos[joint_addr] = np.asarray(frame["q"], dtype=float)
        if len(gripper_addr):
            data.qpos[gripper_addr] = float(frame["gripper_width"]) / float(len(gripper_addr))
        weld_active = bool(frame["weld_active"])
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
        mujoco.mj_step(model, data)
        obstacle_violations.extend(_scan_obstacle_violations(mujoco, model, data, phase_name=str(frame["phase_name"])))
        if weld_active or str(frame["phase_name"]) in {"close_gripper", "attach"}:
            count, penetration, max_example, examples = _scan_gripper_object_contacts(
                mujoco,
                model,
                data,
                phase_name=str(frame["phase_name"]),
            )
            gripper_object_contacts += count
            if penetration > gripper_object_max_penetration:
                gripper_object_max_penetration = penetration
                gripper_object_max_penetration_example = max_example
            gripper_object_examples.extend(examples)
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
    failure_reasons: list[str] = []
    if final_error > position_tolerance:
        failure_reasons.append(f"final_error:{final_error:.6g}")
    if weld_frames <= 0:
        failure_reasons.append("weld_never_active")
    unique_obstacle_violations = tuple(dict.fromkeys(obstacle_violations))
    if unique_obstacle_violations:
        failure_reasons.append(f"obstacle_collision:{len(unique_obstacle_violations)}")
    unique_gripper_examples = tuple(dict.fromkeys(gripper_object_examples))
    return {
        "schema_version": 1,
        "success": not failure_reasons,
        "failure_reasons": failure_reasons,
        "final_object_position": [float(item) for item in cube_position],
        "target_position": [float(item) for item in target],
        "final_error": final_error,
        "weld_frames": weld_frames,
        "cube_contacts": cube_contacts,
        "table_contacts": table_contacts,
        "obstacle_collisions": len(unique_obstacle_violations),
        "obstacle_collision_examples": list(unique_obstacle_violations[:8]),
        "grasp_contact_present": gripper_object_contacts > 0,
        "gripper_object_contacts": gripper_object_contacts,
        "gripper_object_max_penetration": gripper_object_max_penetration,
        "gripper_object_max_penetration_example": gripper_object_max_penetration_example,
        "gripper_object_contact_examples": list(unique_gripper_examples[:8]),
        "settle_steps": int(settle_steps),
    }
