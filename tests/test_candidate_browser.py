from __future__ import annotations

import json

from morphtamp_x_v2.candidate_browser import (
    CandidateSwitchState,
    load_candidate_replays,
)
from morphtamp_x_v2.cli import parser


def write_manifest(tmp_path):
    root = tmp_path / "comparison"
    replay_root = root / "candidate_replays"
    replay_root.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "selected_strategy": "strategy_b",
        "candidates": [
            {
                "index": 1,
                "strategy_name": "strategy_a",
                "selected": False,
                "success": True,
                "scene_xml": "candidate_replays/01_strategy_a/scene.xml",
                "replay_json": "candidate_replays/01_strategy_a/replay.json",
            },
            {
                "index": 2,
                "strategy_name": "strategy_b",
                "selected": True,
                "success": True,
                "scene_xml": "candidate_replays/02_strategy_b/scene.xml",
                "replay_json": "candidate_replays/02_strategy_b/replay.json",
            },
            {
                "index": 3,
                "strategy_name": "strategy_c",
                "selected": False,
                "success": False,
                "scene_xml": "candidate_replays/03_strategy_c/scene.xml",
                "replay_json": "candidate_replays/03_strategy_c/replay.json",
            },
        ],
    }
    path = replay_root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_candidate_replays_resolves_paths_and_selected_index(tmp_path):
    manifest = write_manifest(tmp_path)

    candidates, selected_index = load_candidate_replays(manifest)

    assert selected_index == 1
    assert tuple(candidate.strategy_name for candidate in candidates) == (
        "strategy_a",
        "strategy_b",
        "strategy_c",
    )
    assert candidates[0].xml_path == tmp_path / "comparison" / "candidate_replays" / "01_strategy_a" / "scene.xml"
    assert candidates[1].replay_path == tmp_path / "comparison" / "candidate_replays" / "02_strategy_b" / "replay.json"


def test_candidate_switch_state_supports_next_previous_and_digits(tmp_path):
    candidates, selected_index = load_candidate_replays(write_manifest(tmp_path))
    state = CandidateSwitchState(candidates=candidates, candidate_index=selected_index)

    assert state.current.strategy_name == "strategy_b"
    assert state.apply_key(ord("n")) == "reload"
    assert state.current.strategy_name == "strategy_c"
    assert state.apply_key(ord("p")) == "reload"
    assert state.current.strategy_name == "strategy_b"
    assert state.apply_key(ord("1")) == "reload"
    assert state.current.strategy_name == "strategy_a"
    assert state.apply_key(ord("3")) == "reload"
    assert state.current.strategy_name == "strategy_c"
    assert state.apply_key(ord("h")) == "help"
    assert state.apply_key(ord("q")) == "quit"


def test_cli_accepts_browse_candidates_options():
    parsed = parser().parse_args(
        [
            "browse-candidates",
            "--manifest",
            "results/comparison/candidate_replays/manifest.json",
            "--fps",
            "45",
            "--playback-speed",
            "0.4",
        ]
    )

    assert parsed.command == "browse-candidates"
    assert parsed.manifest.name == "manifest.json"
    assert parsed.fps == 45
    assert parsed.playback_speed == 0.4
