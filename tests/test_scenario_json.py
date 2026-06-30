from __future__ import annotations

import json

from morphtamp_x_v2.cli import main, parser
from morphtamp_x_v2.scenario_io import load_scenario_json, scenario_from_json_dict
from morphtamp_x_v2.tasks import make_scenario


def test_scenario_json_round_trips_projected_robocasa_case(tmp_path):
    scenario = make_scenario(object_type="mug_proxy", task_name="shelf_pick")
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario.to_json_dict()), encoding="utf-8")

    loaded = load_scenario_json(path)

    assert loaded.object_type == "mug_proxy"
    assert loaded.task_name == "shelf_pick"
    assert loaded.cube_start == scenario.cube_start
    assert loaded.support_blocks == scenario.support_blocks


def test_scenario_from_json_accepts_support_block_dicts():
    payload = make_scenario(object_type="cube", task_name="shelf_place").to_json_dict()

    scenario = scenario_from_json_dict(payload)

    assert scenario.support_blocks
    assert isinstance(scenario.support_blocks[0][0], tuple)
    assert isinstance(scenario.support_blocks[0][1], tuple)


def test_cli_run_accepts_scenario_json(tmp_path):
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(make_scenario(object_type="mug_proxy", task_name="shelf_pick").to_json_dict()),
        encoding="utf-8",
    )
    output = tmp_path / "run"

    args = parser().parse_args(
        [
            "run",
            "--scenario-json",
            str(scenario_path),
            "--output-dir",
            str(output),
        ]
    )

    assert args.scenario_json == scenario_path
    assert main(["run", "--scenario-json", str(scenario_path), "--output-dir", str(output)]) == 0
    payload = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    assert payload["scenario"]["object_type"] == "mug_proxy"
    assert payload["scenario"]["task_name"] == "shelf_pick"
