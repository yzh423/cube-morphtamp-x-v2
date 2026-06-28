from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from morphtamp_x_v2.viewer import (
    _apply_replay_frame,
    _cube_address,
    _cube_qvel_address,
    _gripper_addresses,
    _joint_addresses,
    _weld_id,
)


def _name(mujoco, model, obj, index: int) -> str:
    return str(mujoco.mj_id2name(model, obj, int(index)) or "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--phase", default="attach")
    args = parser.parse_args()

    import mujoco

    payload = json.loads(args.replay.read_text(encoding="utf-8"))
    replay = payload["joint_replay"]
    frames = [frame for frame in replay["frames"] if frame["phase_name"] == args.phase]
    if not frames:
        raise SystemExit(f"phase {args.phase!r} not found")
    frame = frames[len(frames) // 2]

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    joint_addr = _joint_addresses(mujoco, model, list(replay["joint_names"]))
    gripper_addr = _gripper_addresses(mujoco, model)
    cube_addr = _cube_address(mujoco, model)
    cube_qvel = _cube_qvel_address(mujoco, model)
    weld = _weld_id(mujoco, model)
    _apply_replay_frame(data, frame, joint_addr, gripper_addr, cube_addr, cube_qvel, weld)
    mujoco.mj_forward(model, data)

    cube_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "v2_cube"))
    cube_pos = np.asarray(data.xpos[cube_body], dtype=float)
    print("phase", args.phase)
    print("cube", cube_pos.tolist())
    print("target tcp", frame["target_position"])
    print("solved tcp", frame["tcp_position"])
    print("gripper_width", frame["gripper_width"])

    rows = []
    for geom_id in range(int(model.ngeom)):
        geom_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        body_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        lowered = f"{geom_name} {body_name}".lower()
        if any(token in lowered for token in ("finger", "hand", "gripper")):
            pos = np.asarray(data.geom_xpos[geom_id], dtype=float)
            rows.append((float(np.linalg.norm(pos - cube_pos)), geom_name, body_name, pos.tolist()))
    for distance, geom_name, body_name, pos in sorted(rows)[:12]:
        print(f"{distance:.6f}", geom_name or "<unnamed>", body_name, pos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
