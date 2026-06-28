from __future__ import annotations

import json

from morphtamp_x_v2.cli import main, parser
from morphtamp_x_v2.design_analysis import (
    build_design_recommendation,
    build_pareto_report,
    segment_importance,
)


def sample_optimizer_payload() -> dict:
    return {
        "schema_version": 1,
        "optimization_type": "continuous_grid_search",
        "minimum_reach_margin": 0.05,
        "design_summaries": [
            {
                "arm_design": "short",
                "success_rate": 0.5,
                "morphology_cost": 0.70,
                "objective": 500.7,
                "mean_path_length": 2.0,
                "minimum_reach_margin": 0.01,
                "design": {"upper_scale": 0.7, "forearm_scale": 0.7, "wrist_scale": 0.7, "base_shift": [0, 0, 0]},
            },
            {
                "arm_design": "balanced",
                "success_rate": 1.0,
                "morphology_cost": 0.82,
                "objective": 0.86,
                "mean_path_length": 2.0,
                "minimum_reach_margin": 0.06,
                "design": {"upper_scale": 0.9, "forearm_scale": 0.9, "wrist_scale": 0.82, "base_shift": [0, 0, 0]},
            },
            {
                "arm_design": "long",
                "success_rate": 1.0,
                "morphology_cost": 0.95,
                "objective": 0.98,
                "mean_path_length": 1.5,
                "minimum_reach_margin": 0.18,
                "design": {"upper_scale": 1.1, "forearm_scale": 1.0, "wrist_scale": 1.0, "base_shift": [0, 0, 0]},
            },
        ],
    }


def test_pareto_report_identifies_tradeoff_designs():
    report = build_pareto_report(sample_optimizer_payload())

    assert report["best_objective"]["arm_design"] == "balanced"
    assert report["lowest_cost_feasible"]["arm_design"] == "balanced"
    assert report["highest_margin_feasible"]["arm_design"] == "long"
    assert "balanced" in report["markdown"]
    assert "Pareto" in report["markdown"]


def test_segment_importance_summarizes_success_by_scale():
    report = segment_importance(sample_optimizer_payload())

    assert report["segments"]["upper_scale"]["0.900"]["success_rate"] == 1.0
    assert report["segments"]["wrist_scale"]["0.700"]["success_rate"] == 0.5
    assert "upper_scale" in report["markdown"]
    assert "wrist_scale" in report["markdown"]


def test_design_recommendation_card_explains_choice_and_risks():
    card = build_design_recommendation(sample_optimizer_payload())

    assert card["recommended_design"]["arm_design"] == "balanced"
    assert "Why this design" in card["markdown"]
    assert "Risks" in card["markdown"]
    assert "0.9" in card["markdown"]


def test_design_analysis_cli_writes_json_and_markdown(tmp_path):
    input_path = tmp_path / "optimizer.json"
    output_json = tmp_path / "analysis.json"
    output_md = tmp_path / "analysis.md"
    input_path.write_text(json.dumps(sample_optimizer_payload()), encoding="utf-8")

    parsed = parser().parse_args(
        [
            "design-analysis",
            "--input",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )
    assert parsed.command == "design-analysis"

    assert main(
        [
            "design-analysis",
            "--input",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    ) == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["recommendation"]["recommended_design"]["arm_design"] == "balanced"
    assert "Segment importance" in output_md.read_text(encoding="utf-8")
