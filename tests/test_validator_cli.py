from __future__ import annotations

import json

from morphtamp_x_v2.cli import main
from morphtamp_x_v2.cli import parser
from morphtamp_x_v2.cli import run_execution_success
from morphtamp_x_v2.models import PickPlaceScenario
from morphtamp_x_v2.planner import plan_pick_place
from morphtamp_x_v2.replay import build_replay_frames
from morphtamp_x_v2.validator import evaluate_replay


PANDA_LIKE_XML = """
<mujoco model="panda_like">
  <worldbody>
    <body name="panda_hand"/>
  </worldbody>
</mujoco>
"""


def test_evaluate_replay_accepts_complete_pick_place_sequence():
    scenario = PickPlaceScenario()
    frames = build_replay_frames(scenario, plan_pick_place(scenario), transition_frames=5, hold_frames=1)

    metrics = evaluate_replay(scenario, frames)

    assert metrics["success"] is True
    assert metrics["release_error"] == 0.0
    assert metrics["attached_frame_count"] > 0


def test_run_execution_success_requires_joint_replay_success_when_present():
    metrics = {"success": True}

    assert run_execution_success(metrics, None) is True
    assert run_execution_success(metrics, {"success": True}) is True
    assert run_execution_success(metrics, {"success": False}) is False
    assert run_execution_success({"success": False}, {"success": True}) is False


def test_cli_plan_writes_json(tmp_path):
    output = tmp_path / "plan.json"

    assert main(["plan", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["phases"][0]["name"] == "home"


def test_cli_build_scene_writes_scene_and_json(tmp_path):
    panda = tmp_path / "panda.xml"
    panda.write_text(PANDA_LIKE_XML, encoding="utf-8")
    output_xml = tmp_path / "scene.xml"
    output_json = tmp_path / "scene.json"

    assert main(["build-scene", "--panda-xml", str(panda), "--output-xml", str(output_xml), "--output-json", str(output_json)]) == 0

    assert output_xml.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["scene"]["object_body"] == "v2_cube"


def test_cli_run_writes_replay_metrics(tmp_path):
    output_dir = tmp_path / "run"

    assert main(["run", "--object", "plate", "--output-dir", str(output_dir)]) == 0

    payload = json.loads((output_dir / "replay.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["success"] is True
    assert payload["grasp_strategy"]["name"] == "plate_center_stabilized_pinch"
    assert payload["grasp_strategy"]["approach_axis"] == "top"
    assert payload["grasp_selection"]["selected_strategy"] == payload["grasp_strategy"]["name"]
    assert len(payload["grasp_selection"]["candidate_evaluations"]) >= 2


def test_cli_accepts_panda_ik_and_view_options():
    run = parser().parse_args(
        [
            "run",
            "--panda-xml",
            "panda.xml",
            "--auto-fit-panda",
            "--position-tolerance",
            "0.04",
            "--output-dir",
            "out",
        ]
    )
    view = parser().parse_args(
        [
            "view",
            "--xml",
            "scene.xml",
            "--replay",
            "replay.json",
            "--interactive",
            "--fps",
            "30",
        ]
    )

    assert run.command == "run"
    assert run.auto_fit_panda is True
    assert run.position_tolerance == 0.04
    assert view.command == "view"
    assert view.interactive is True
    assert view.fps == 30


def test_cli_accepts_physics_validation_options():
    parsed = parser().parse_args(
        [
            "validate-physics",
            "--xml",
            "scene.xml",
            "--replay",
            "replay.json",
            "--settle-steps",
            "25",
            "--position-tolerance",
            "0.05",
            "--output",
            "physics.json",
        ]
    )

    assert parsed.command == "validate-physics"
    assert str(parsed.xml) == "scene.xml"
    assert str(parsed.replay) == "replay.json"
    assert parsed.settle_steps == 25
    assert parsed.position_tolerance == 0.05
    assert str(parsed.output) == "physics.json"


def test_cli_view_reports_joint_replay_metadata(tmp_path, capsys):
    scene = tmp_path / "scene.xml"
    replay = tmp_path / "replay.json"
    scene.write_text("<mujoco/>", encoding="utf-8")
    replay.write_text(
        json.dumps(
            {
                "joint_replay": {
                    "success": True,
                    "joint_names": [f"joint{i}" for i in range(1, 8)],
                    "frames": [{"q": [0.0] * 7}],
                }
            }
        ),
        encoding="utf-8",
    )

    assert main(["view", "--xml", str(scene), "--replay", str(replay)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["has_joint_replay"] is True
    assert payload["frame_count"] == 1
    assert payload["success"] is True
