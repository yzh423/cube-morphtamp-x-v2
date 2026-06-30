from __future__ import annotations

import json

from morphtamp_x_v2.cli import main, parser
from morphtamp_x_v2.protocols import benchmark_protocol


def test_protocol_catalog_separates_auto_fit_development_from_fixed_evaluation():
    development = benchmark_protocol("development_auto_fit")
    fixed = benchmark_protocol("fixed")
    heldout = benchmark_protocol("heldout_fixed")
    stress = benchmark_protocol("stress_fixed")

    assert development.auto_fit_panda is True
    assert fixed.auto_fit_panda is False
    assert heldout.auto_fit_panda is False
    assert stress.auto_fit_panda is False
    assert set(heldout.tasks).isdisjoint({"tabletop_easy", "precision_drop"})
    assert {"compound_shelf_barrier", "diagonal_reach_around", "under_bridge"} <= set(heldout.tasks)
    assert {"reach_boundary_far", "tight_under_bridge", "high_shelf_barrier"} <= set(stress.tasks)


def test_benchmark_cli_accepts_protocol_and_records_it(tmp_path):
    output = tmp_path / "heldout.json"
    parsed = parser().parse_args(
        [
            "benchmark",
            "--protocol",
            "heldout_fixed",
            "--objects",
            "cube",
            "--output",
            str(output),
        ]
    )
    assert parsed.protocol == "heldout_fixed"

    assert main(
        [
            "benchmark",
            "--protocol",
            "heldout_fixed",
            "--objects",
            "cube",
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol"]["name"] == "heldout_fixed"
    assert payload["auto_fit_panda"] is False
    assert payload["objects"] == ["cube"]
    assert set(payload["tasks"]) == set(benchmark_protocol("heldout_fixed").tasks)


def test_stress_benchmark_cli_uses_stress_protocol(tmp_path):
    output = tmp_path / "stress.json"
    parsed = parser().parse_args(
        [
            "stress-benchmark",
            "--objects",
            "cube",
            "--output",
            str(output),
        ]
    )

    assert parsed.command == "stress-benchmark"
    assert main(["stress-benchmark", "--objects", "cube", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol"]["name"] == "stress_fixed"
    assert "reach_boundary_far" in payload["tasks"]
    assert "tight_under_bridge" in payload["tasks"]


def test_benchmark_cli_reuses_incremental_result_cache(tmp_path, monkeypatch):
    import morphtamp_x_v2.cli as cli

    calls = {"count": 0}

    def fake_case(object_type, task_name, **kwargs):
        calls["count"] += 1
        return {
            "object_type": object_type,
            "task_name": task_name,
            "success": True,
            "failure_reasons": [],
            "path_length": 1.0,
            "lift_height": 0.1,
            "metrics": {"success": True},
            "joint_metrics": None,
            "physical_evidence": {"success": True},
            "scenario": {"cube_start": [0.4, 0.0, 0.06], "place_target": [0.5, 0.1, 0.06]},
        }

    monkeypatch.setattr(cli, "_run_static_case", fake_case)
    cache = tmp_path / "benchmark_cache.json"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    assert main(
        [
            "benchmark",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--results-cache",
            str(cache),
            "--output",
            str(first_output),
        ]
    ) == 0
    assert calls["count"] == 1

    assert main(
        [
            "benchmark",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--results-cache",
            str(cache),
            "--output",
            str(second_output),
        ]
    ) == 0
    assert calls["count"] == 1
    assert json.loads(second_output.read_text(encoding="utf-8"))["cache"]["reused_rows"] == 1
