from __future__ import annotations

import csv
import json

from morphtamp_x_v2.benchmark_analysis import summarize_benchmark, summarize_grasp_selection
from morphtamp_x_v2.cli import main, parser


def sample_benchmark_payload() -> dict:
    return {
        "schema_version": 1,
        "objects": ["cube", "sphere"],
        "tasks": ["tabletop_easy", "over_barrier"],
        "total_runs": 4,
        "successful_runs": 3,
        "results": [
            {
                "object_type": "cube",
                "task_name": "tabletop_easy",
                "success": True,
                "path_length": 0.4,
                "joint_metrics": {"success": True, "max_position_error": 0.01, "failure_reasons": [], "joint_count": 7},
                "grasp_selection_summary": {
                    "selected_strategy": "cube_top",
                    "coarse_candidates": 3,
                    "full_candidates": 1,
                    "coarse_pruned_candidates": 2,
                    "coarse_successes": 2,
                    "full_successes": 1,
                    "rejection_code_counts": {"orientation_error": 1},
                },
            },
            {
                "object_type": "cube",
                "task_name": "over_barrier",
                "success": True,
                "path_length": 0.9,
                "joint_metrics": {"success": False, "max_position_error": 0.08, "failure_reasons": ["ik_residual"], "joint_count": 7},
                "grasp_selection_summary": {
                    "selected_strategy": "cube_side",
                    "coarse_candidates": 3,
                    "full_candidates": 2,
                    "coarse_pruned_candidates": 1,
                    "coarse_successes": 1,
                    "full_successes": 0,
                    "rejection_code_counts": {"ik_error": 2},
                },
            },
            {
                "object_type": "sphere",
                "task_name": "tabletop_easy",
                "success": True,
                "path_length": 0.35,
                "joint_metrics": {"success": True, "max_position_error": 0.015, "failure_reasons": [], "joint_count": 7},
            },
            {
                "object_type": "sphere",
                "task_name": "over_barrier",
                "success": True,
                "path_length": 0.7,
                "joint_metrics": {"success": True, "max_position_error": 0.02, "failure_reasons": [], "joint_count": 7},
            },
        ],
    }


def test_summarize_grasp_selection_reports_coarse_to_fine_evidence():
    selection = {
        "selected_strategy": "top_pinch",
        "coarse_candidate_evaluations": [
            {"strategy_name": "top_pinch", "success": True, "failure_reasons": []},
            {
                "strategy_name": "edge_6d",
                "success": False,
                "failure_reasons": ["grasp:orientation_error:3.14", "grasp:ik_error:0.12"],
            },
            {"strategy_name": "side_pinch", "success": False, "failure_reasons": ["lift:collision:1"]},
        ],
        "candidate_evaluations": [
            {"strategy_name": "top_pinch", "success": True, "failure_reasons": []},
        ],
    }

    summary = summarize_grasp_selection(selection)

    assert summary["selected_strategy"] == "top_pinch"
    assert summary["coarse_candidates"] == 3
    assert summary["full_candidates"] == 1
    assert summary["coarse_pruned_candidates"] == 2
    assert summary["coarse_successes"] == 1
    assert summary["full_successes"] == 1
    assert summary["rejection_code_counts"] == {
        "collision": 1,
        "ik_error": 1,
        "orientation_error": 1,
    }


def test_cli_accepts_analysis_and_morphology_benchmark_options():
    analysis = parser().parse_args(
        [
            "analyze-benchmark",
            "--input",
            "benchmark.json",
            "--output-json",
            "summary.json",
            "--output-csv",
            "summary.csv",
            "--output-md",
            "summary.md",
        ]
    )
    morph = parser().parse_args(
        [
            "morphology-benchmark",
            "--arm-designs",
            "short_arm",
            "nominal_panda",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--minimum-reach-margin",
            "0.03",
            "--output",
            "morph.json",
        ]
    )

    assert analysis.command == "analyze-benchmark"
    assert morph.command == "morphology-benchmark"
    assert morph.arm_designs == ["short_arm", "nominal_panda"]
    assert morph.minimum_reach_margin == 0.03


def test_analyze_benchmark_writes_json_csv_and_markdown(tmp_path):
    input_path = tmp_path / "benchmark.json"
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"
    output_md = tmp_path / "summary.md"
    input_path.write_text(json.dumps(sample_benchmark_payload()), encoding="utf-8")

    assert (
        main(
            [
                "analyze-benchmark",
                "--input",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["overall"]["success_rate"] == 0.75
    assert summary["by_object"]["cube"]["success_rate"] == 0.5
    assert summary["by_task"]["over_barrier"]["success_rate"] == 0.5
    assert summary["best_successful"]["object_type"] == "sphere"
    assert summary["grasp_planning"]["mean_coarse_candidates"] == 3.0
    assert summary["grasp_planning"]["mean_full_candidates"] == 1.5
    assert summary["grasp_planning"]["rows_with_grasp_data"] == 2
    assert summary["grasp_planning"]["rows_without_grasp_data"] == 2
    assert summary["grasp_planning"]["coarse_pruned_candidates"] == 3
    assert summary["grasp_planning"]["selected_strategy_counts"]["cube_top"] == 1
    assert summary["grasp_planning"]["rejection_code_counts"] == {
        "ik_error": 2,
        "orientation_error": 1,
    }
    assert summary["failure_taxonomy"]["overall_code_counts"]["joint_ik_residual"] == 1
    assert summary["failure_taxonomy"]["overall_code_counts"]["grasp_ik_error"] == 2
    assert summary["failure_taxonomy"]["by_task"]["over_barrier"]["joint_ik_residual"] == 1
    assert summary["failed_cases"] == [
        {
            "object_type": "cube",
            "task_name": "over_barrier",
            "selected_strategy": "cube_side",
            "path_length": 0.9,
            "max_position_error": 0.08,
            "failure_reasons": ["joint:ik_residual"],
            "failure_reason_counts": {"joint:ik_residual": 1},
        }
    ]
    assert summary["task_splits"]["development"]
    assert "over_barrier" in summary["task_splits"]["main"]

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert {row["group"] for row in rows} >= {"overall", "object", "task"}
    markdown = output_md.read_text(encoding="utf-8")
    assert "# MorphTAMP-X Benchmark Summary" in markdown
    assert "## Grasp planning efficiency" in markdown
    assert "Mean coarse candidates" in markdown
    assert "## Failure taxonomy" in markdown
    assert "## Failed cases" in markdown
    assert "`cube`" in markdown
    assert "`joint_ik_residual`" in markdown


def test_analyze_benchmark_reports_missing_grasp_selection_data():
    payload = sample_benchmark_payload()
    for row in payload["results"]:
        row.pop("grasp_selection_summary", None)
        row.pop("grasp_selection", None)

    summary = summarize_benchmark(payload)

    assert summary["grasp_planning"]["rows_with_grasp_data"] == 0
    assert summary["grasp_planning"]["rows_without_grasp_data"] == 4
    assert summary["grasp_planning"]["mean_coarse_candidates"] == 0.0


def test_task_splits_classify_all_library_benchmark_tasks():
    payload = {
        "schema_version": 1,
        "results": [
            {
                "object_type": "cube",
                "task_name": task_name,
                "success": True,
                "path_length": 1.0,
                "joint_metrics": {"success": True, "max_position_error": 0.0, "failure_reasons": []},
            }
            for task_name in [
                "tabletop_easy",
                "over_barrier",
                "narrow_slot",
                "shelf_pick",
                "shelf_place",
                "folded_transfer",
                "far_corner",
                "around_wall",
                "compound_shelf_barrier",
                "diagonal_reach_around",
                "under_bridge",
            ]
        ],
    }

    summary = summarize_benchmark(payload)

    assert summary["task_splits"]["unassigned"] == []
    assert "shelf_pick" in summary["task_splits"]["main"]


def test_benchmark_rows_include_grasp_selection_summary(tmp_path):
    output = tmp_path / "benchmark.json"

    assert (
        main(
            [
                "benchmark",
                "--objects",
                "plate",
                "--tasks",
                "tabletop_easy",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    row = payload["results"][0]
    assert row["grasp_selection_summary"]["selected_strategy"] == row["grasp_selection"]["selected_strategy"]
    assert row["grasp_selection_summary"]["full_candidates"] == len(
        row["grasp_selection"]["candidate_evaluations"]
    )


def test_morphology_benchmark_scores_designs_and_selects_pareto(tmp_path):
    output = tmp_path / "morphology.json"

    assert (
        main(
            [
                "morphology-benchmark",
                "--arm-designs",
                "short_arm",
                "nominal_panda",
                "long_forearm",
                "--objects",
                "cube",
                "--tasks",
                "tabletop_easy",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["total_runs"] == 3
    assert payload["best_design"] in {"short_arm", "nominal_panda", "long_forearm"}
    assert payload["pareto_designs"]
    assert all("morphology_cost" in row for row in payload["results"])


def test_morphology_benchmark_can_require_positive_reach_margin(tmp_path):
    output = tmp_path / "morphology_margin.json"

    assert (
        main(
            [
                "morphology-benchmark",
                "--arm-designs",
                "short_arm",
                "nominal_panda",
                "--objects",
                "tall_box",
                "--tasks",
                "compound_shelf_barrier",
                "--minimum-reach-margin",
                "0.03",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = {row["arm_design"]: row for row in payload["results"]}
    assert rows["short_arm"]["success"] is False
    assert rows["nominal_panda"]["success"] is True
    assert any(reason.startswith("reach_margin:") for reason in rows["short_arm"]["failure_reasons"])
    assert payload["best_design"] == "nominal_panda"
