from __future__ import annotations

import json

from morphtamp_x_v2.cli import main, parser
from morphtamp_x_v2.failure_analysis import build_failure_analysis
from morphtamp_x_v2.robustness import perturb_scenario, run_robustness_benchmark
from morphtamp_x_v2.tasks import make_scenario


def test_perturb_scenario_is_seed_reproducible_and_preserves_object_height():
    scenario = make_scenario(object_type="cube", task_name="tabletop_easy")

    first = perturb_scenario(scenario, seed=7, position_noise=0.01, obstacle_noise=0.02)
    second = perturb_scenario(scenario, seed=7, position_noise=0.01, obstacle_noise=0.02)

    assert first == second
    assert first.cube_start[2] == scenario.cube_start[2]
    assert first.place_target[2] == scenario.place_target[2]
    assert first.cube_start[:2] != scenario.cube_start[:2]


def test_run_robustness_benchmark_reports_trials_and_summary():
    payload = run_robustness_benchmark(
        objects=("cube",),
        tasks=("tabletop_easy",),
        trials=3,
        seed=11,
        position_noise=0.005,
        obstacle_noise=0.0,
    )

    assert payload["schema_version"] == 1
    assert payload["total_runs"] == 3
    assert payload["summary"]["success_rate"] == 1.0
    assert {row["trial"] for row in payload["results"]} == {0, 1, 2}
    assert all("perturbation" in row for row in payload["results"])


def test_robustness_cli_writes_json(tmp_path):
    output = tmp_path / "robustness.json"
    parsed = parser().parse_args(
        [
            "robustness-benchmark",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--trials",
            "2",
            "--seed",
            "5",
            "--output",
            str(output),
        ]
    )
    assert parsed.command == "robustness-benchmark"

    assert main(
        [
            "robustness-benchmark",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--trials",
            "2",
            "--seed",
            "5",
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["success_rate"] == 1.0


def test_failure_analysis_markdown_groups_by_design_task_object_and_reason(tmp_path):
    payload = {
        "schema_version": 1,
        "results": [
            {
                "success": False,
                "arm_design": "short_arm",
                "object_type": "cube",
                "task_name": "diagonal_reach_around",
                "failure_reasons": ["reach_margin:0.02", "collision"],
            },
            {
                "success": False,
                "arm_design": "short_arm",
                "object_type": "sphere",
                "task_name": "diagonal_reach_around",
                "failure_reasons": ["reach_margin:0.01"],
            },
            {
                "success": True,
                "arm_design": "nominal_panda",
                "object_type": "cube",
                "task_name": "tabletop_easy",
                "failure_reasons": [],
            },
        ],
    }

    report = build_failure_analysis(payload)

    assert report["failed_runs"] == 2
    assert report["by_design"]["short_arm"] == 2
    assert report["by_task"]["diagonal_reach_around"] == 2
    assert report["reason_codes"]["reach_margin"] == 2
    assert "short_arm" in report["markdown"]
    assert "reach_margin" in report["markdown"]


def test_failure_analysis_cli_writes_markdown_and_json(tmp_path):
    input_path = tmp_path / "input.json"
    output_json = tmp_path / "failure.json"
    output_md = tmp_path / "failure.md"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "results": [
                    {
                        "success": False,
                        "object_type": "cube",
                        "task_name": "over_barrier",
                        "failure_reasons": ["joint:ik_residual"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "failure-analysis",
            "--input",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    ) == 0

    assert json.loads(output_json.read_text(encoding="utf-8"))["failed_runs"] == 1
    assert "joint" in output_md.read_text(encoding="utf-8")
