from __future__ import annotations

import json

import morphtamp_x_v2.cli as cli


def test_cli_accepts_dynamics_benchmark_options():
    parsed = cli.parser().parse_args(
        [
            "dynamics-benchmark",
            "--objects",
            "cube",
            "sphere",
            "--tasks",
            "tabletop_easy",
            "over_barrier",
            "--panda-xml",
            "panda.xml",
            "--auto-fit-panda",
            "--position-tolerance",
            "0.05",
            "--full-candidate-limit",
            "1",
            "--settle-steps",
            "40",
            "--frame-substeps",
            "8",
            "--dynamics-gate",
            "ultra_strict_4mm",
            "--max-gripper-penetration",
            "0.004",
            "--require-two-sided-grasp",
            "--output-dir",
            "results/dynamics",
            "--output",
            "results/dynamics/summary.json",
        ]
    )

    assert parsed.command == "dynamics-benchmark"
    assert parsed.objects == ["cube", "sphere"]
    assert parsed.tasks == ["tabletop_easy", "over_barrier"]
    assert parsed.auto_fit_panda is True
    assert parsed.full_candidate_limit == 1
    assert parsed.dynamics_gate == "ultra_strict_4mm"
    assert parsed.max_gripper_penetration == 0.004
    assert parsed.require_two_sided_grasp is True


def test_dynamics_benchmark_practical_gate_sets_penetration_and_two_sided_defaults(
    tmp_path, monkeypatch
):
    calls = []

    def fake_write_replay_case(**kwargs):
        case_dir = kwargs["output_dir"]
        case_dir.mkdir(parents=True, exist_ok=True)
        scene = case_dir / "scene.xml"
        replay = case_dir / "replay.json"
        scene.write_text("<mujoco/>", encoding="utf-8")
        replay.write_text("{}", encoding="utf-8")
        return {
            "scene_xml": scene,
            "replay_path": replay,
            "execution_success": True,
            "scenario": {
                "object_type": kwargs["object_type"],
                "task_name": kwargs["task_name"],
            },
        }

    def fake_validate_physics_replay(xml_path, replay_path, **kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "failure_reasons": [],
            "checks": {
                "gripper_penetration": {
                    "passed": True,
                    "max_allowed_penetration": kwargs["max_gripper_penetration"],
                },
                "two_sided_grasp_contact": {
                    "passed": True,
                    "required_for_success": kwargs["require_two_sided_grasp"],
                },
            },
        }

    monkeypatch.setattr(cli, "_write_replay_case", fake_write_replay_case)
    monkeypatch.setattr(cli, "validate_physics_replay", fake_validate_physics_replay)
    output = tmp_path / "summary.json"

    assert cli.main(
        [
            "dynamics-benchmark",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--panda-xml",
            "panda.xml",
            "--dynamics-gate",
            "panda_practical_8mm",
            "--output-dir",
            str(tmp_path / "cases"),
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert calls[0]["max_gripper_penetration"] == 0.008
    assert calls[0]["require_two_sided_grasp"] is True
    assert payload["settings"]["dynamics_gate"]["name"] == "panda_practical_8mm"
    assert payload["settings"]["dynamics_gate"]["max_gripper_penetration"] == 0.008
    assert payload["settings"]["dynamics_gate"]["require_two_sided_grasp"] is True


def test_dynamics_benchmark_results_cache_resumes_completed_rows(tmp_path, monkeypatch):
    calls = []

    def fake_write_replay_case(**kwargs):
        calls.append(kwargs)
        case_dir = kwargs["output_dir"]
        case_dir.mkdir(parents=True, exist_ok=True)
        scene = case_dir / "scene.xml"
        replay = case_dir / "replay.json"
        scene.write_text("<mujoco/>", encoding="utf-8")
        replay.write_text("{}", encoding="utf-8")
        return {
            "scene_xml": scene,
            "replay_path": replay,
            "execution_success": True,
            "scenario": {
                "object_type": kwargs["object_type"],
                "task_name": kwargs["task_name"],
            },
        }

    def fake_validate_physics_replay(xml_path, replay_path, **kwargs):
        return {
            "success": True,
            "failure_reasons": [],
            "checks": {
                "gripper_penetration": {
                    "passed": True,
                    "max_allowed_penetration": kwargs["max_gripper_penetration"],
                }
            },
        }

    monkeypatch.setattr(cli, "_write_replay_case", fake_write_replay_case)
    monkeypatch.setattr(cli, "validate_physics_replay", fake_validate_physics_replay)
    cache = tmp_path / "dynamics_cache.json"
    output_1 = tmp_path / "summary_1.json"
    output_2 = tmp_path / "summary_2.json"
    common_args = [
        "dynamics-benchmark",
        "--objects",
        "cube",
        "--tasks",
        "tabletop_easy",
        "--panda-xml",
        "panda.xml",
        "--results-cache",
        str(cache),
        "--output-dir",
        str(tmp_path / "cases"),
    ]

    assert cli.main([*common_args, "--output", str(output_1)]) == 0
    assert len(calls) == 1

    assert cli.main([*common_args, "--output", str(output_2)]) == 0
    payload = json.loads(output_2.read_text(encoding="utf-8"))
    cache_payload = json.loads(cache.read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert payload["cache"]["reused_rows"] == 1
    assert payload["cache"]["computed_rows"] == 0
    assert cache_payload["cache_type"] == "dynamics_benchmark_results"
    assert cache_payload["results"][0]["object_type"] == "cube"


def test_dynamics_benchmark_writes_case_rows_and_summary(tmp_path, monkeypatch):
    calls = []

    def fake_write_replay_case(**kwargs):
        case_dir = kwargs["output_dir"]
        case_dir.mkdir(parents=True, exist_ok=True)
        scene = case_dir / "scene.xml"
        replay = case_dir / "replay.json"
        scene.write_text("<mujoco/>", encoding="utf-8")
        replay.write_text("{}", encoding="utf-8")
        return {
            "scene_xml": scene,
            "replay_path": replay,
            "execution_success": True,
            "scenario": {
                "object_type": kwargs["object_type"],
                "task_name": kwargs["task_name"],
            },
        }

    def fake_validate_physics_replay(xml_path, replay_path, **kwargs):
        calls.append((xml_path, replay_path, kwargs))
        return {
            "success": "cube" in str(replay_path),
            "failure_reasons": [] if "cube" in str(replay_path) else ["gripper_penetration"],
            "checks": {
                "gripper_penetration": {
                    "passed": "cube" in str(replay_path),
                    "max_allowed_penetration": kwargs["max_gripper_penetration"],
                }
            },
        }

    monkeypatch.setattr(cli, "_write_replay_case", fake_write_replay_case)
    monkeypatch.setattr(cli, "validate_physics_replay", fake_validate_physics_replay)
    output = tmp_path / "summary.json"

    assert cli.main(
        [
            "dynamics-benchmark",
            "--objects",
            "cube",
            "sphere",
            "--tasks",
            "tabletop_easy",
            "--panda-xml",
            "panda.xml",
            "--max-gripper-penetration",
            "0.004",
            "--require-two-sided-grasp",
            "--output-dir",
            str(tmp_path / "cases"),
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["total_runs"] == 2
    assert payload["successful_runs"] == 1
    assert payload["summary"]["success_rate"] == 0.5
    assert payload["results"][0]["object_type"] == "cube"
    assert payload["results"][0]["success"] is True
    assert payload["results"][1]["failure_reasons"] == ["dynamics:gripper_penetration"]
    assert calls[0][2]["max_gripper_penetration"] == 0.004
    assert calls[0][2]["require_two_sided_grasp"] is True


def test_dynamics_benchmark_retries_grasp_strategies_for_strict_two_sided_mode(
    tmp_path, monkeypatch
):
    attempted_strategies = []

    def fake_write_replay_case(**kwargs):
        strategy_name = kwargs["grasp_strategy_name"]
        attempted_strategies.append(strategy_name)
        case_dir = kwargs["output_dir"]
        case_dir.mkdir(parents=True, exist_ok=True)
        scene = case_dir / "scene.xml"
        replay = case_dir / "replay.json"
        scene.write_text("<mujoco/>", encoding="utf-8")
        replay.write_text("{}", encoding="utf-8")
        return {
            "scene_xml": scene,
            "replay_path": replay,
            "execution_success": True,
            "scenario": {
                "object_type": kwargs["object_type"],
                "task_name": kwargs["task_name"],
            },
            "payload": {
                "grasp_strategy": {
                    "name": strategy_name,
                }
            },
        }

    def fake_validate_physics_replay(xml_path, replay_path, **kwargs):
        success = "box_top_center_pinch" in str(replay_path)
        return {
            "success": success,
            "failure_reasons": [] if success else ["two_sided_grasp_contact"],
            "checks": {
                "two_sided_grasp_contact": {
                    "passed": success,
                    "required_for_success": kwargs["require_two_sided_grasp"],
                }
            },
        }

    monkeypatch.setattr(cli, "_write_replay_case", fake_write_replay_case)
    monkeypatch.setattr(cli, "validate_physics_replay", fake_validate_physics_replay)
    output = tmp_path / "summary.json"

    assert cli.main(
        [
            "dynamics-benchmark",
            "--objects",
            "cube",
            "--tasks",
            "tabletop_easy",
            "--panda-xml",
            "panda.xml",
            "--full-candidate-limit",
            "3",
            "--require-two-sided-grasp",
            "--output-dir",
            str(tmp_path / "cases"),
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    row = payload["results"][0]
    assert payload["successful_runs"] == 1
    assert row["success"] is True
    assert row["grasp_strategy"]["name"] == "box_top_center_pinch"
    assert row["candidate_dynamics_attempts"][0]["success"] is False
    assert row["candidate_dynamics_attempts"][1]["success"] is True
    assert attempted_strategies[:2] == ["box_side_pinch", "box_top_center_pinch"]


def test_dynamics_benchmark_selects_lowest_penetration_failed_candidate(
    tmp_path, monkeypatch
):
    attempted_strategies = []
    penetration_by_strategy = {
        "box_side_pinch": 0.012,
        "box_top_center_pinch": 0.006,
        "box_x_side_pinch": 0.009,
    }

    def fake_write_replay_case(**kwargs):
        strategy_name = kwargs["grasp_strategy_name"]
        attempted_strategies.append(strategy_name)
        case_dir = kwargs["output_dir"]
        case_dir.mkdir(parents=True, exist_ok=True)
        scene = case_dir / "scene.xml"
        replay = case_dir / "replay.json"
        scene.write_text("<mujoco/>", encoding="utf-8")
        replay.write_text("{}", encoding="utf-8")
        return {
            "scene_xml": scene,
            "replay_path": replay,
            "execution_success": True,
            "scenario": {
                "object_type": kwargs["object_type"],
                "task_name": kwargs["task_name"],
            },
            "payload": {
                "grasp_strategy": {
                    "name": strategy_name,
                }
            },
        }

    def fake_validate_physics_replay(xml_path, replay_path, **kwargs):
        replay_text = str(replay_path)
        strategy_name = next(
            name for name in penetration_by_strategy if name in replay_text
        )
        penetration = penetration_by_strategy[strategy_name]
        return {
            "success": False,
            "failure_reasons": ["gripper_penetration"],
            "final_error": 0.01,
            "checks": {
                "two_sided_grasp_contact": {
                    "passed": True,
                    "required_for_success": kwargs["require_two_sided_grasp"],
                },
                "gripper_penetration": {
                    "passed": False,
                    "gripper_object_max_penetration": penetration,
                    "dominant_penetration_side": "left_finger",
                },
            },
        }

    monkeypatch.setattr(cli, "_write_replay_case", fake_write_replay_case)
    monkeypatch.setattr(cli, "validate_physics_replay", fake_validate_physics_replay)
    output = tmp_path / "summary.json"

    assert cli.main(
        [
            "dynamics-benchmark",
            "--objects",
            "cube",
            "--tasks",
            "over_barrier",
            "--panda-xml",
            "panda.xml",
            "--full-candidate-limit",
            "3",
            "--require-two-sided-grasp",
            "--max-gripper-penetration",
            "0.004",
            "--output-dir",
            str(tmp_path / "cases"),
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    row = payload["results"][0]
    assert payload["successful_runs"] == 0
    assert attempted_strategies == [
        "box_side_pinch",
        "box_top_center_pinch",
        "box_x_side_pinch",
    ]
    assert row["grasp_strategy"]["name"] == "box_top_center_pinch"
    assert row["dynamics_evidence"]["checks"]["gripper_penetration"][
        "gripper_object_max_penetration"
    ] == 0.006
    assert len(row["candidate_dynamics_attempts"]) == 3
