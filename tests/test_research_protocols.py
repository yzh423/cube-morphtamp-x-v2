from __future__ import annotations

import json

from morphtamp_x_v2.cli import main, parser
from morphtamp_x_v2.protocols import benchmark_protocol


def test_protocol_catalog_separates_auto_fit_development_from_fixed_evaluation():
    development = benchmark_protocol("development_auto_fit")
    fixed = benchmark_protocol("fixed")
    heldout = benchmark_protocol("heldout_fixed")

    assert development.auto_fit_panda is True
    assert fixed.auto_fit_panda is False
    assert heldout.auto_fit_panda is False
    assert set(heldout.tasks).isdisjoint({"tabletop_easy", "precision_drop"})
    assert {"compound_shelf_barrier", "diagonal_reach_around", "under_bridge"} <= set(heldout.tasks)


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
