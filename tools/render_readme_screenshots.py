from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from morphtamp_x_v2.viewer import (  # noqa: E402
    _apply_replay_frame,
    _cube_address,
    _cube_qvel_address,
    _gripper_addresses,
    _joint_addresses,
    _weld_id,
)


SHOWCASES = (
    {
        "object": "cube",
        "task": "over_barrier",
        "name": "cube_over_barrier",
        "phase": "transport",
        "caption": "Cube transfer over a solid obstacle with Panda joint replay.",
    },
    {
        "object": "sphere",
        "task": "folded_transfer",
        "name": "sphere_folded_transfer",
        "phase": "transport",
        "caption": "Sphere task requiring a folded arm posture and carried-object replay.",
    },
    {
        "object": "plate",
        "task": "shelf_place",
        "name": "plate_shelf_place",
        "phase": "place",
        "caption": "Plate-style object placement with object-specific grasp selection.",
    },
)


def _run_case(case: dict[str, str], panda_xml: Path, work_dir: Path) -> tuple[Path, Path]:
    output_dir = work_dir / case["name"]
    cmd = [
        sys.executable,
        "-m",
        "morphtamp_x_v2.cli",
        "run",
        "--object",
        case["object"],
        "--task",
        case["task"],
        "--panda-xml",
        str(panda_xml),
        "--auto-fit-panda",
        "--position-tolerance",
        "0.05",
        "--full-candidate-limit",
        "2",
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{ROOT / 'tools'}"
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return output_dir / "scene.xml", output_dir / "replay.json"


def _select_frame(frames: list[dict[str, Any]], phase_name: str) -> dict[str, Any]:
    matches = [frame for frame in frames if frame.get("phase") == phase_name]
    if not matches:
        matches = frames
    return matches[len(matches) // 2]


def _render_replay_frame(
    xml_path: Path,
    replay_path: Path,
    output_path: Path,
    *,
    phase_name: str,
    width: int,
    height: int,
) -> None:
    import mujoco
    import numpy as np

    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    joint_replay = payload["joint_replay"]
    frames = list(joint_replay["frames"])
    frame = _select_frame(frames, phase_name)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_addr = _joint_addresses(mujoco, model, list(joint_replay["joint_names"]))
    gripper_addr = _gripper_addresses(mujoco, model)
    cube_addr = _cube_address(mujoco, model)
    cube_qvel = _cube_qvel_address(mujoco, model)
    weld = _weld_id(mujoco, model)
    _apply_replay_frame(
        data,
        frame,
        joint_addr,
        gripper_addr,
        cube_addr,
        cube_qvel,
        weld,
        enable_visual_weld=False,
    )
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    # A stable three-quarter view that keeps the table, obstacle/object, and
    # Panda arm visible across the README showcase tasks.
    object_position = np.asarray(frame["object_position"], dtype=float)
    camera.lookat[:] = np.asarray([0.11, 0.00, max(0.43, object_position[2])], dtype=float)
    camera.distance = 1.48
    camera.azimuth = 150.0
    camera.elevation = -14.0

    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as error:
        raise RuntimeError("render_readme_screenshots requires imageio") from error
    imageio.imwrite(output_path, image)
    renderer.close()


def build_readme_screenshots(
    *,
    panda_xml: Path,
    output_dir: Path,
    work_dir: Path,
    width: int = 640,
    height: int = 480,
) -> list[dict[str, str]]:
    generated: list[dict[str, str]] = []
    for case in SHOWCASES:
        xml_path, replay_path = _run_case(case, panda_xml, work_dir)
        output_path = output_dir / f"{case['name']}.png"
        _render_replay_frame(
            xml_path,
            replay_path,
            output_path,
            phase_name=case["phase"],
            width=width,
            height=height,
        )
        generated.append(
            {
                "object": case["object"],
                "task": case["task"],
                "path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
                "caption": case["caption"],
            }
        )
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render README simulation screenshots")
    parser.add_argument("--panda-xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets" / "readme")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "results" / "readme_screenshot_runs")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(argv)
    generated = build_readme_screenshots(
        panda_xml=args.panda_xml.expanduser(),
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        width=args.width,
        height=args.height,
    )
    for item in generated:
        print(f"{item['object']}:{item['task']} -> {item['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
