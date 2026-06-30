from __future__ import annotations

import json

from morphtamp_x_v2.ablation import build_ablation_report, write_ablation_markdown
from morphtamp_x_v2.cli import main, parser


def test_ablation_report_compares_optimized_design_to_baselines():
    final_summary = {
        "optimization": {
            "best_design": "opt_demo",
            "best_morphology_cost": 0.75,
            "best_success_rate": 1.0,
            "best_minimum_reach_margin": 0.05,
            "minimum_reach_margin_constraint": 0.03,
            "best_minimum_sigma": 0.2,
            "best_maximum_condition_number": 5.0,
            "constraints": {"minimum_sigma": 0.08, "maximum_condition_number": 30.0},
        }
    }
    morphology = {
        "design_summaries": [
            {"arm_design": "short_arm", "success_rate": 0.8, "morphology_cost": 0.72, "minimum_reach_margin": -0.01},
            {"arm_design": "nominal_panda", "success_rate": 1.0, "morphology_cost": 0.86, "minimum_reach_margin": 0.12},
        ],
    }
    benchmark_summary = {
        "grasp_planning": {
            "rows_with_grasp_data": 10,
            "mean_coarse_candidates": 4.0,
            "mean_full_candidates": 2.0,
        }
    }

    report = build_ablation_report(
        final_summary=final_summary,
        morphology_benchmark=morphology,
        benchmark_summary=benchmark_summary,
    )

    assert report["schema_version"] == 1
    assert report["optimized_design"]["name"] == "opt_demo"
    assert report["baselines"]["short_arm"]["success_rate"] == 0.8
    assert report["components"]["reach_margin_constraint"]["active"] is True
    assert report["components"]["singularity_constraint"]["active"] is True
    assert report["components"]["multi_grasp_selection"]["active"] is True
    assert report["recommendation"]["paper_ready"] is True


def test_ablation_report_markdown_mentions_failure_boundary():
    report = build_ablation_report(
        final_summary={"optimization": {"best_design": "opt", "best_success_rate": 1.0}},
        morphology_benchmark={"design_summaries": []},
        benchmark_summary={},
    )
    markdown = write_ablation_markdown(report)

    assert "Ablation" in markdown
    assert "failure boundary" in markdown.lower()


def test_cli_ablation_report_writes_outputs(tmp_path):
    final_summary = tmp_path / "final.json"
    morphology = tmp_path / "morphology.json"
    output_json = tmp_path / "ablation.json"
    output_md = tmp_path / "ablation.md"
    final_summary.write_text(json.dumps({"optimization": {"best_design": "opt", "best_success_rate": 1.0}}), encoding="utf-8")
    morphology.write_text(json.dumps({"design_summaries": []}), encoding="utf-8")

    args = parser().parse_args(
        [
            "ablation-report",
            "--final-summary",
            str(final_summary),
            "--morphology",
            str(morphology),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert args.command == "ablation-report"
    assert main(
        [
            "ablation-report",
            "--final-summary",
            str(final_summary),
            "--morphology",
            str(morphology),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    ) == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["schema_version"] == 1
    assert "Ablation" in output_md.read_text(encoding="utf-8")
