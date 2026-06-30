from __future__ import annotations

import json

import morphtamp_x_v2.cli as cli
from morphtamp_x_v2.dynamics_analysis import build_dynamics_benchmark_analysis


def sample_dynamics_payload() -> dict:
    return {
        "schema_version": 1,
        "total_runs": 3,
        "successful_runs": 1,
        "results": [
            {
                "object_type": "cube",
                "task_name": "tabletop_easy",
                "success": True,
                "failure_reasons": [],
                "grasp_strategy": {"name": "box_side_pinch"},
                "dynamics_evidence": {
                    "checks": {
                        "gripper_penetration": {
                            "gripper_object_max_penetration": 0.003,
                            "dominant_penetration_side": "right_finger",
                        },
                        "two_sided_grasp_contact": {"passed": True},
                        "grasp_contact": {
                            "gripper_object_contact_sides": [
                                "left_finger",
                                "right_finger",
                            ],
                        },
                    },
                    "final_error": 0.001,
                },
            },
            {
                "object_type": "cube",
                "task_name": "over_barrier",
                "success": False,
                "failure_reasons": ["dynamics:gripper_penetration"],
                "grasp_strategy": {"name": "box_top_center_pinch"},
                "dynamics_evidence": {
                    "checks": {
                        "gripper_penetration": {
                            "gripper_object_max_penetration": 0.010,
                            "dominant_penetration_side": "palm_or_hand",
                        },
                        "two_sided_grasp_contact": {"passed": True},
                        "grasp_contact": {
                            "gripper_object_contact_sides": [
                                "left_finger",
                                "palm_or_hand",
                                "right_finger",
                            ],
                        },
                    },
                    "final_error": 0.002,
                },
            },
            {
                "object_type": "sphere",
                "task_name": "over_barrier",
                "success": False,
                "failure_reasons": ["dynamics:two_sided_grasp_contact"],
                "grasp_strategy": {"name": "spherical_high_pinch"},
                "dynamics_evidence": {
                    "checks": {
                        "gripper_penetration": {
                            "gripper_object_max_penetration": 0.002,
                            "dominant_penetration_side": "left_finger",
                        },
                        "two_sided_grasp_contact": {"passed": False},
                        "grasp_contact": {
                            "gripper_object_contact_sides": ["left_finger"],
                        },
                    },
                    "final_error": 0.003,
                },
            },
        ],
    }


def test_build_dynamics_benchmark_analysis_summarizes_failures_and_contacts():
    analysis = build_dynamics_benchmark_analysis(sample_dynamics_payload())

    assert analysis["overall"]["success_rate"] == 1 / 3
    assert analysis["by_object"]["cube"]["successes"] == 1
    assert analysis["by_object"]["cube"]["runs"] == 2
    assert analysis["by_task"]["over_barrier"]["success_rate"] == 0.0
    assert analysis["failure_counts"] == {
        "dynamics:gripper_penetration": 1,
        "dynamics:two_sided_grasp_contact": 1,
    }
    assert analysis["max_gripper_penetration"]["value"] == 0.010
    assert analysis["max_gripper_penetration"]["object_type"] == "cube"
    assert analysis["dominant_penetration_side_counts"] == {
        "left_finger": 1,
        "palm_or_hand": 1,
        "right_finger": 1,
    }
    assert analysis["selected_strategy_counts"]["box_side_pinch"] == 1
    assert "## Dynamics benchmark analysis" in analysis["markdown"]
    assert "dynamics:gripper_penetration" in analysis["markdown"]


def test_analyze_dynamics_benchmark_cli_writes_json_and_markdown(tmp_path):
    input_path = tmp_path / "dynamics.json"
    output_json = tmp_path / "analysis.json"
    output_md = tmp_path / "analysis.md"
    input_path.write_text(json.dumps(sample_dynamics_payload()), encoding="utf-8")

    assert cli.main(
        [
            "analyze-dynamics-benchmark",
            "--input",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    ) == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overall"]["runs"] == 3
    assert output_md.read_text(encoding="utf-8").startswith(
        "## Dynamics benchmark analysis"
    )
