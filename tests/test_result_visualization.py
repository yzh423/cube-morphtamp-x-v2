from __future__ import annotations

import json

from morphtamp_x_v2.cli import main, parser


def benchmark_payload() -> dict:
    return {
        "schema_version": 1,
        "results": [
            {
                "object_type": "cube",
                "task_name": "easy",
                "success": True,
                "path_length": 0.3,
                "joint_metrics": {"success": True, "max_position_error": 0.01},
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
                "task_name": "hard",
                "success": True,
                "path_length": 0.8,
                "joint_metrics": {"success": False, "max_position_error": 0.07},
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
                "task_name": "easy",
                "success": True,
                "path_length": 0.25,
                "joint_metrics": {"success": True, "max_position_error": 0.02},
                "grasp_selection_summary": {
                    "selected_strategy": "sphere_center",
                    "coarse_candidates": 2,
                    "full_candidates": 1,
                    "coarse_pruned_candidates": 1,
                    "coarse_successes": 2,
                    "full_successes": 1,
                    "rejection_code_counts": {},
                },
            },
            {
                "object_type": "sphere",
                "task_name": "hard",
                "success": True,
                "path_length": 0.6,
                "joint_metrics": {"success": True, "max_position_error": 0.03},
                "grasp_selection_summary": {
                    "selected_strategy": "sphere_center",
                    "coarse_candidates": 2,
                    "full_candidates": 1,
                    "coarse_pruned_candidates": 1,
                    "coarse_successes": 2,
                    "full_successes": 1,
                    "rejection_code_counts": {"collision": 1},
                },
            },
        ],
    }


def morphology_payload() -> dict:
    return {
        "schema_version": 1,
        "best_design": "short_arm",
        "pareto_designs": ["short_arm", "nominal_panda"],
        "design_summaries": [
            {"arm_design": "short_arm", "success_rate": 0.75, "morphology_cost": 0.72, "mean_path_length": 0.5, "minimum_reach_margin": 0.1},
            {"arm_design": "nominal_panda", "success_rate": 1.0, "morphology_cost": 0.86, "mean_path_length": 0.55, "minimum_reach_margin": 0.2},
            {"arm_design": "long_forearm", "success_rate": 1.0, "morphology_cost": 0.92, "mean_path_length": 0.58, "minimum_reach_margin": 0.3},
        ],
    }


def test_cli_accepts_visualize_results_options():
    parsed = parser().parse_args(
        [
            "visualize-results",
            "--benchmark",
            "benchmark.json",
            "--morphology",
            "morphology.json",
            "--output-dir",
            "figures",
        ]
    )

    assert parsed.command == "visualize-results"
    assert parsed.output_dir.name == "figures"


def test_visualize_results_writes_svg_dashboard_and_manifest(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    morphology = tmp_path / "morphology.json"
    output_dir = tmp_path / "figures"
    benchmark.write_text(json.dumps(benchmark_payload()), encoding="utf-8")
    morphology.write_text(json.dumps(morphology_payload()), encoding="utf-8")

    assert (
        main(
            [
                "visualize-results",
                "--benchmark",
                str(benchmark),
                "--morphology",
                str(morphology),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    manifest = json.loads((output_dir / "visualization_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    expected = {
        "fig1_system_pipeline": output_dir / "main" / "fig1_system_pipeline.svg",
        "fig2_object_task_success_heatmap": output_dir / "main" / "fig2_object_task_success_heatmap.svg",
        "fig3_grasp_candidate_funnel": output_dir / "main" / "fig3_grasp_candidate_funnel.svg",
        "fig4_rejection_reason_distribution": output_dir / "main" / "fig4_rejection_reason_distribution.svg",
        "fig5_morphology_pareto": output_dir / "main" / "fig5_morphology_pareto.svg",
        "fig6_kinematic_quality": output_dir / "main" / "fig6_kinematic_quality.svg",
        "fig7_selected_strategy_distribution": output_dir / "diagnostics" / "fig7_selected_strategy_distribution.svg",
        "fig8_task_difficulty_ranking": output_dir / "diagnostics" / "fig8_task_difficulty_ranking.svg",
        "fig9_object_difficulty_ranking": output_dir / "diagnostics" / "fig9_object_difficulty_ranking.svg",
        "fig10_coarse_vs_full_cost": output_dir / "diagnostics" / "fig10_coarse_vs_full_cost.svg",
        "fig11_collision_failure_map": output_dir / "diagnostics" / "fig11_collision_failure_map.svg",
    }
    for figure_id, path in expected.items():
        assert path.exists(), figure_id
        assert manifest["figures"][figure_id] == str(path)
    assert (output_dir / "dashboard.html").exists()
    dashboard = (output_dir / "dashboard.html").read_text(encoding="utf-8")
    assert "Main Paper Figures" in dashboard
    assert "Diagnostic Figures" in dashboard
    assert "fig2_object_task_success_heatmap.svg" in dashboard
    assert "fig10_coarse_vs_full_cost.svg" in dashboard


def test_visualize_results_warns_when_grasp_candidate_data_is_missing(tmp_path):
    payload = benchmark_payload()
    for row in payload["results"]:
        row.pop("grasp_selection_summary", None)
        row.pop("grasp_selection", None)
    benchmark = tmp_path / "benchmark_without_grasp.json"
    morphology = tmp_path / "morphology.json"
    output_dir = tmp_path / "figures"
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    morphology.write_text(json.dumps(morphology_payload()), encoding="utf-8")

    assert (
        main(
            [
                "visualize-results",
                "--benchmark",
                str(benchmark),
                "--morphology",
                str(morphology),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    manifest = json.loads((output_dir / "visualization_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_quality"]["grasp_rows_with_data"] == 0
    assert manifest["data_quality"]["grasp_rows_without_data"] == 4
    assert any("grasp candidate" in warning for warning in manifest["data_quality"]["warnings"])
    dashboard = (output_dir / "dashboard.html").read_text(encoding="utf-8")
    assert "Data Quality Warnings" in dashboard
    assert "grasp candidate" in dashboard
