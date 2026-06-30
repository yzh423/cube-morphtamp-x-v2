from __future__ import annotations

import json
from dataclasses import replace

import pytest

from morphtamp_x_v2.cli import main, parser
from morphtamp_x_v2.physics_evidence import evaluate_physical_evidence
from morphtamp_x_v2.planner import plan_pick_place
from morphtamp_x_v2.replay import build_replay_frames
from morphtamp_x_v2.tasks import make_scenario


def test_physical_evidence_accepts_nominal_attached_transfer():
    scenario = make_scenario(object_type="cube", task_name="over_barrier")
    frames = build_replay_frames(scenario, plan_pick_place(scenario))

    evidence = evaluate_physical_evidence(scenario, frames)

    assert evidence["success"] is True
    assert evidence["checks"]["grasp_retention"]["passed"] is True
    assert evidence["checks"]["release_support"]["passed"] is True
    assert evidence["checks"]["obstacle_clearance"]["passed"] is True
    assert evidence["checks"]["grasp_retention"]["attached_frame_count"] > 0


def test_physical_evidence_rejects_object_that_does_not_follow_gripper():
    scenario = make_scenario(object_type="cube", task_name="over_barrier")
    frames = build_replay_frames(scenario, plan_pick_place(scenario))
    corrupted = tuple(
        replace(frame, object_position=scenario.cube_start)
        if frame.object_attached
        else frame
        for frame in frames
    )

    evidence = evaluate_physical_evidence(
        scenario,
        corrupted,
        retention_tolerance=0.01,
    )

    assert evidence["success"] is False
    assert "grasp_retention" in evidence["failure_reasons"]
    assert evidence["checks"]["grasp_retention"]["max_retention_error"] > 0.01


def test_physical_evidence_rejects_final_object_below_support_surface():
    scenario = make_scenario(object_type="sphere", task_name="tabletop_easy")
    frames = build_replay_frames(scenario, plan_pick_place(scenario))
    bad_final = replace(
        frames[-1],
        object_position=(
            scenario.place_target[0],
            scenario.place_target[1],
            scenario.table_center[2],
        ),
    )
    corrupted = frames[:-1] + (bad_final,)

    evidence = evaluate_physical_evidence(scenario, corrupted)

    assert evidence["success"] is False
    assert "release_support" in evidence["failure_reasons"]
    assert evidence["checks"]["release_support"]["support_gap"] < -0.005


def test_cli_validate_task_physics_writes_report(tmp_path):
    scenario = make_scenario(object_type="cube", task_name="over_barrier")
    frames = build_replay_frames(scenario, plan_pick_place(scenario))
    replay = tmp_path / "replay.json"
    output = tmp_path / "physics.json"
    replay.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenario": scenario.to_json_dict(),
                "frames": [frame.to_json_dict() for frame in frames],
            }
        ),
        encoding="utf-8",
    )

    args = parser().parse_args(
        [
            "validate-task-physics",
            "--replay",
            str(replay),
            "--output",
            str(output),
        ]
    )

    assert args.command == "validate-task-physics"
    assert main(["validate-task-physics", "--replay", str(replay), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert payload["evidence_scope"].startswith("geometric replay")


def test_benchmark_rows_include_physical_evidence(tmp_path):
    output = tmp_path / "benchmark.json"

    assert main(
        [
            "benchmark",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--output",
            str(output),
        ]
    ) == 0

    row = json.loads(output.read_text(encoding="utf-8"))["results"][0]
    assert row["physical_evidence"]["success"] is True
    assert row["physical_evidence"]["checks"]["release_support"]["passed"] is True
