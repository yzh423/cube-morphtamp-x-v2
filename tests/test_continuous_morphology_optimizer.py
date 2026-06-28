from __future__ import annotations

import json

from morphtamp_x_v2.cli import main, parser
from morphtamp_x_v2.morphology_optimizer import generate_scale_grid, optimize_morphology


def test_generate_scale_grid_includes_base_shift_and_segment_scales():
    designs = list(
        generate_scale_grid(
            upper_scales=(0.9, 1.0),
            forearm_scales=(0.9,),
            wrist_scales=(0.95,),
            base_x_values=(0.0, 0.05),
        )
    )

    assert [design.name for design in designs] == [
        "opt_u0.900_f0.900_w0.950_bx0.000",
        "opt_u0.900_f0.900_w0.950_bx0.050",
        "opt_u1.000_f0.900_w0.950_bx0.000",
        "opt_u1.000_f0.900_w0.950_bx0.050",
    ]
    assert designs[1].base_shift == (0.05, 0.0, 0.0)


def test_optimize_morphology_prefers_lowest_cost_design_satisfying_margin():
    result = optimize_morphology(
        objects=("cube",),
        tasks=("tabletop_easy",),
        upper_scales=(0.75, 1.0),
        forearm_scales=(0.75, 1.0),
        wrist_scales=(0.8,),
        base_x_values=(0.0,),
        minimum_reach_margin=0.03,
    )

    assert result["total_candidates"] == 4
    assert result["best_design"] is not None
    best = result["best_design"]
    assert best["success_rate"] == 1.0
    assert best["minimum_reach_margin"] >= 0.03
    assert result["pareto_designs"]


def test_optimize_morphology_cli_writes_json(tmp_path):
    output = tmp_path / "continuous.json"
    parsed = parser().parse_args(
        [
            "optimize-morphology",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--scale-values",
            "0.85",
            "1.0",
            "--base-x-values",
            "0.0",
            "--minimum-reach-margin",
            "0.03",
            "--output",
            str(output),
        ]
    )
    assert parsed.command == "optimize-morphology"

    assert main(
        [
            "optimize-morphology",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--scale-values",
            "0.85",
            "1.0",
            "--base-x-values",
            "0.0",
            "--minimum-reach-margin",
            "0.03",
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["optimization_type"] == "continuous_grid_search"
    assert payload["best_design"]["success_rate"] == 1.0


def test_optimize_morphology_can_apply_singularity_constraints():
    result = optimize_morphology(
        objects=("cube",),
        tasks=("tabletop_easy",),
        upper_scales=(0.85, 1.0),
        forearm_scales=(0.85,),
        wrist_scales=(0.85,),
        base_x_values=(0.0,),
        minimum_reach_margin=0.03,
        minimum_sigma=0.2,
        maximum_condition_number=10.0,
    )

    assert result["singularity_constraints"] == {
        "minimum_sigma": 0.2,
        "maximum_condition_number": 10.0,
    }
    assert all("singularity_passed" in row for row in result["design_summaries"])
