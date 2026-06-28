from __future__ import annotations

import json

from morphtamp_x_v2.cli import main, parser
from morphtamp_x_v2.candidate_report import build_candidate_report, compact_failure_reasons, write_candidate_report
from morphtamp_x_v2.grasp_planner import CandidateEvaluation
from morphtamp_x_v2.tasks import make_scenario


def test_build_candidate_report_contains_metrics_paths_and_selected_flag():
    scenario = make_scenario(object_type="plate", task_name="shelf_place")
    report = build_candidate_report(
        scenario,
        selected_strategy="plate_clearance_top_pinch",
        evaluations=(
            CandidateEvaluation(
                strategy_name="plate_clearance_top_pinch",
                success=True,
                max_position_error=0.002,
                max_orientation_error=0.0,
                min_joint_margin=0.8,
                collision_count=0,
                path_length=0.4,
                joint_path_length=1.2,
                max_joint_step=0.2,
                max_condition_number=3.0,
                failure_reasons=(),
            ),
            CandidateEvaluation(
                strategy_name="plate_edge_pinch_6d",
                success=False,
                max_position_error=0.09,
                max_orientation_error=1.1,
                min_joint_margin=0.2,
                collision_count=0,
                path_length=0.5,
                joint_path_length=5.0,
                max_joint_step=1.6,
                max_condition_number=5.0,
                failure_reasons=("grasp:orientation_error:1.1",),
            ),
        ),
    )

    assert report["schema_version"] == 1
    assert report["selected_strategy"] == "plate_clearance_top_pinch"
    assert len(report["candidates"]) == 2
    selected = report["candidates"][0]
    failed = report["candidates"][1]
    assert selected["selected"] is True
    assert selected["metrics"]["joint_path_length"] == 1.2
    assert len(selected["path_points"]) >= 6
    assert all(len(point) == 3 for point in selected["path_points"])
    assert failed["selected"] is False
    assert failed["failure_reasons"] == ["grasp:orientation_error:1.1"]


def test_write_candidate_report_outputs_nonempty_html_and_json(tmp_path):
    scenario = make_scenario(object_type="cube", task_name="over_barrier")
    report = build_candidate_report(
        scenario,
        selected_strategy="box_side_pinch",
        evaluations=(
            CandidateEvaluation(
                strategy_name="box_side_pinch",
                success=True,
                max_position_error=0.001,
                min_joint_margin=0.6,
                collision_count=0,
                path_length=0.5,
                failure_reasons=(),
                joint_path_length=1.4,
                max_joint_step=0.3,
                max_condition_number=3.2,
            ),
        ),
    )
    report["candidate_replays"] = {
        "candidates": [
            {
                "strategy_name": "box_side_pinch",
                "replay_json": "candidate_replays/01_box_side_pinch/replay.json",
                "scene_xml": "candidate_replays/01_box_side_pinch/scene.xml",
            }
        ]
    }

    outputs = write_candidate_report(report, tmp_path)

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    html = outputs["html"].read_text(encoding="utf-8")
    assert payload["selected_strategy"] == "box_side_pinch"
    assert "box_side_pinch" in html
    assert "Candidate trajectory comparison" in html
    assert "<polyline" in html
    assert "joint path" in html
    assert "candidate_replays/01_box_side_pinch/replay.json" in html


def test_compact_failure_reasons_groups_repeated_phase_codes():
    summary = compact_failure_reasons(
        (
            "grasp:ik_error:0.09",
            "grasp:ik_error:0.08",
            "place:orientation_error:0.9",
            "place:orientation_error:0.8",
            "place:collision:1",
        )
    )

    assert summary == (
        "grasp:ik_error x2",
        "place:orientation_error x2",
        "place:collision x1",
    )


def test_cli_compare_grasps_writes_candidate_report(tmp_path):
    parsed = parser().parse_args(
        [
            "compare-grasps",
            "--object",
            "cube",
            "--task",
            "over_barrier",
            "--output-dir",
            str(tmp_path / "comparison"),
        ]
    )
    assert parsed.command == "compare-grasps"

    assert main(
        [
            "compare-grasps",
            "--object",
            "cube",
            "--task",
            "over_barrier",
            "--output-dir",
            str(tmp_path / "comparison"),
        ]
    ) == 0

    assert (tmp_path / "comparison" / "candidate_report.json").exists()
    assert (tmp_path / "comparison" / "candidate_report.html").exists()
    assert (tmp_path / "comparison" / "candidate_replays" / "manifest.json").exists()
    report = json.loads((tmp_path / "comparison" / "candidate_report.json").read_text(encoding="utf-8"))
    assert report["selected_strategy"]
    assert len(report["candidates"]) >= 2
    manifest = json.loads((tmp_path / "comparison" / "candidate_replays" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert len(manifest["candidates"]) == len(report["candidates"])
    for row in manifest["candidates"]:
        assert row["strategy_name"]
        assert (tmp_path / "comparison" / row["replay_json"]).exists()
