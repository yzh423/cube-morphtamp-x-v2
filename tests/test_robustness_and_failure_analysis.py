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
    assert payload["summary"]["by_object"]["cube"]["successful_runs"] == 3
    assert payload["summary"]["by_task"]["tabletop_easy"]["success_rate"] == 1.0
    assert payload["summary"]["mean_path_length"] is not None
    assert {row["trial"] for row in payload["results"]} == {0, 1, 2}
    assert all("perturbation" in row for row in payload["results"])


def test_panda_robustness_evaluates_perturbed_scenario(monkeypatch):
    seen = []

    def fake_run_static_case(object_type, task_name, **kwargs):
        scenario = kwargs["scenario"]
        seen.append(scenario)
        return {
            "object_type": object_type,
            "task_name": task_name,
            "success": True,
            "failure_reasons": [],
            "path_length": 0.1,
            "metrics": {"success": True, "max_position_error": 0.0},
            "joint_metrics": {"success": True, "max_position_error": 0.0},
            "scenario": scenario.to_json_dict(),
        }

    monkeypatch.setattr(
        "morphtamp_x_v2.robustness._run_static_case",
        fake_run_static_case,
    )

    nominal = make_scenario(object_type="cube", task_name="tabletop_easy")
    payload = run_robustness_benchmark(
        objects=("cube",),
        tasks=("tabletop_easy",),
        trials=1,
        seed=11,
        position_noise=0.02,
        obstacle_noise=0.0,
        panda_xml="fake_panda.xml",
    )

    assert len(seen) == 1
    assert seen[0].cube_start != nominal.cube_start
    assert payload["results"][0]["scenario"]["cube_start"] == list(seen[0].cube_start)


def test_robustness_benchmark_reuses_incremental_result_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "robustness_cache.json"
    calls = []

    def fake_run_static_case(object_type, task_name, **kwargs):
        calls.append((object_type, task_name, kwargs["scenario"].cube_start))
        scenario = kwargs["scenario"]
        return {
            "object_type": object_type,
            "task_name": task_name,
            "success": True,
            "failure_reasons": [],
            "path_length": 0.1,
            "metrics": {"success": True, "max_position_error": 0.0},
            "joint_metrics": {"success": True, "max_position_error": 0.0},
            "scenario": scenario.to_json_dict(),
        }

    monkeypatch.setattr(
        "morphtamp_x_v2.robustness._run_static_case",
        fake_run_static_case,
    )

    first = run_robustness_benchmark(
        objects=("cube",),
        tasks=("tabletop_easy",),
        trials=2,
        seed=11,
        panda_xml="fake_panda.xml",
        results_cache=cache_path,
    )

    assert first["result_cache"]["hits"] == 0
    assert first["result_cache"]["misses"] == 2
    assert len(calls) == 2
    assert cache_path.exists()

    calls.clear()
    second = run_robustness_benchmark(
        objects=("cube",),
        tasks=("tabletop_easy",),
        trials=2,
        seed=11,
        panda_xml="fake_panda.xml",
        results_cache=cache_path,
    )

    assert second["result_cache"]["hits"] == 2
    assert second["result_cache"]["misses"] == 0
    assert calls == []
    assert second["total_runs"] == 2


def test_robustness_benchmark_reuses_cache_when_trial_count_increases(tmp_path, monkeypatch):
    cache_path = tmp_path / "robustness_cache.json"
    calls = []

    def fake_run_static_case(object_type, task_name, **kwargs):
        calls.append((object_type, task_name, kwargs["scenario"].cube_start))
        scenario = kwargs["scenario"]
        return {
            "object_type": object_type,
            "task_name": task_name,
            "success": True,
            "failure_reasons": [],
            "path_length": 0.1,
            "metrics": {"success": True, "release_error": 0.0},
            "joint_metrics": {
                "success": True,
                "max_position_error": 0.01,
                "min_sigma": 0.25,
                "max_condition_number": 4.0,
                "joint_path_length": 1.5,
            },
            "scenario": scenario.to_json_dict(),
        }

    monkeypatch.setattr(
        "morphtamp_x_v2.robustness._run_static_case",
        fake_run_static_case,
    )

    first = run_robustness_benchmark(
        objects=("cube",),
        tasks=("tabletop_easy",),
        trials=2,
        seed=11,
        panda_xml="fake_panda.xml",
        results_cache=cache_path,
    )
    assert first["result_cache"]["misses"] == 2

    calls.clear()
    second = run_robustness_benchmark(
        objects=("cube",),
        tasks=("tabletop_easy",),
        trials=3,
        seed=11,
        panda_xml="fake_panda.xml",
        results_cache=cache_path,
    )

    assert second["result_cache"]["hits"] == 2
    assert second["result_cache"]["misses"] == 1
    assert len(calls) == 1
    assert second["total_runs"] == 3


def test_robustness_summary_reports_panda_quality_metrics(monkeypatch):
    def fake_run_static_case(object_type, task_name, **kwargs):
        scenario = kwargs["scenario"]
        return {
            "object_type": object_type,
            "task_name": task_name,
            "success": True,
            "failure_reasons": [],
            "path_length": 0.5 if object_type == "cube" else 0.7,
            "metrics": {"success": True, "release_error": 0.0},
            "joint_metrics": {
                "success": True,
                "max_position_error": 0.01 if object_type == "cube" else 0.02,
                "min_sigma": 0.30 if object_type == "cube" else 0.20,
                "max_condition_number": 3.0 if object_type == "cube" else 5.0,
                "joint_path_length": 1.5,
                "failure_reasons": [],
            },
            "scenario": scenario.to_json_dict(),
        }

    monkeypatch.setattr(
        "morphtamp_x_v2.robustness._run_static_case",
        fake_run_static_case,
    )

    payload = run_robustness_benchmark(
        objects=("cube", "sphere"),
        tasks=("tabletop_easy",),
        trials=1,
        seed=11,
        panda_xml="fake_panda.xml",
    )

    assert payload["summary"]["max_position_error"] == 0.02
    assert payload["summary"]["mean_position_error"] == 0.015
    assert payload["summary"]["min_sigma"] == 0.20
    assert payload["summary"]["max_condition_number"] == 5.0
    assert payload["summary"]["mean_path_length"] == 0.6
    assert payload["summary"]["by_object"]["cube"]["success_rate"] == 1.0
    assert payload["summary"]["failed_runs"] == []


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
    assert report["reason_codes"]["task"] == 3
    assert "short_arm" in report["markdown"]
    assert "reach_margin" in report["markdown"]


def test_failure_analysis_counts_failed_panda_joint_metrics():
    payload = {
        "schema_version": 1,
        "results": [
            {
                "success": True,
                "object_type": "cube",
                "task_name": "over_barrier",
                "failure_reasons": [],
                "joint_metrics": {
                    "success": False,
                    "failure_reasons": ["ik_residual:0.08", "collision:1"],
                },
            },
            {
                "success": True,
                "object_type": "sphere",
                "task_name": "tabletop_easy",
                "failure_reasons": [],
                "joint_metrics": {"success": True, "failure_reasons": []},
            },
        ],
    }

    report = build_failure_analysis(payload)

    assert report["failed_runs"] == 1
    assert report["reason_codes"]["joint"] == 2
    assert report["by_task"]["over_barrier"] == 1
    assert report["representative_failures"][0]["failure_reasons"] == [
        "joint:ik_residual:0.08",
        "joint:collision:1",
    ]


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
