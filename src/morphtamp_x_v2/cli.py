from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from .benchmark_analysis import (
    load_benchmark,
    summarize_benchmark,
    summarize_grasp_selection,
    write_summary_csv,
    write_summary_markdown,
)
from .candidate_report import build_candidate_report, write_candidate_report
from .candidate_browser import launch_candidate_viewer
from .grasp_planner import path_length_for_phases, select_grasp_plan
from .grasp_strategies import GraspStrategy, strategies_for_object, strategy_for_object
from .interactive_browse import launch_browse_viewer
from .models import PickPlaceScenario
from .morphology import ARM_DESIGN_TYPES, arm_design, pareto_designs
from .panda_ik import auto_fit_panda_scenario, solve_joint_replay
from .physics_evidence import evaluate_physical_evidence, evaluate_replay_json
from .physics_validator import validate_physics_replay
from .planner import plan_pick_place
from .replay import build_replay_frames
from .result_visualization import visualize_results
from .scene_builder import build_pick_place_scene
from .scenario_io import load_scenario_json
from .objects import OBJECT_TYPES
from .objects import object_spec
from .protocols import PROTOCOL_TYPES, resolve_protocol_inputs
from .task_health import check_all_tasks
from .tasks import TASK_TYPES, make_scenario, task_spec
from .validator import evaluate_replay
from .viewer import launch_viewer, replay_report


DYNAMICS_GATE_PRESETS: dict[str, dict[str, object]] = {
    "custom": {
        "name": "custom",
        "description": "Caller-provided dynamics thresholds.",
        "max_gripper_penetration": None,
        "require_two_sided_grasp": False,
    },
    "ultra_strict_4mm": {
        "name": "ultra_strict_4mm",
        "description": (
            "Ultra-strict diagnostic gate for exposing mesh/contact artifacts; "
            "requires two-sided finger contact and <=4 mm penetration."
        ),
        "max_gripper_penetration": 0.004,
        "require_two_sided_grasp": True,
    },
    "panda_practical_8mm": {
        "name": "panda_practical_8mm",
        "description": (
            "Practical Menagerie Panda simulation gate; requires two-sided "
            "finger contact and <=8 mm gripper-object penetration."
        ),
        "max_gripper_penetration": 0.008,
        "require_two_sided_grasp": True,
    },
}


def _dynamics_gate_payload(args: argparse.Namespace) -> dict[str, object]:
    name = str(getattr(args, "dynamics_gate", "custom"))
    preset = DYNAMICS_GATE_PRESETS[name]
    max_penetration = getattr(args, "max_gripper_penetration", None)
    if max_penetration is None:
        max_penetration = preset["max_gripper_penetration"]
    require_two_sided = bool(
        getattr(args, "require_two_sided_grasp", False)
        or preset["require_two_sided_grasp"]
    )
    return {
        "name": name,
        "description": preset["description"],
        "max_gripper_penetration": max_penetration,
        "require_two_sided_grasp": require_two_sided,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="morphtamp-x-v2")
    sub = root.add_subparsers(dest="command", required=True)

    sub.add_parser("list-objects")
    sub.add_parser("list-tasks")
    sub.add_parser("check-tasks")

    plan = sub.add_parser("plan")
    plan.add_argument("--object", dest="object_type", choices=OBJECT_TYPES, default="cube")
    plan.add_argument("--task", choices=TASK_TYPES, default="tabletop_easy")
    plan.add_argument("--scenario-json", type=Path)
    plan.add_argument("--grasp-strategy")
    plan.add_argument("--output", type=Path)

    build = sub.add_parser("build-scene")
    build.add_argument("--panda-xml", type=Path, required=True)
    build.add_argument("--output-xml", type=Path, required=True)
    build.add_argument("--output-json", type=Path)
    build.add_argument("--auto-fit-panda", action="store_true")
    build.add_argument("--object", dest="object_type", choices=OBJECT_TYPES, default="cube")
    build.add_argument("--task", choices=TASK_TYPES, default="tabletop_easy")
    build.add_argument("--scenario-json", type=Path)

    run = sub.add_parser("run")
    run.add_argument("--panda-xml", type=Path)
    run.add_argument("--auto-fit-panda", action="store_true")
    run.add_argument("--object", dest="object_type", choices=OBJECT_TYPES, default="cube")
    run.add_argument("--task", choices=TASK_TYPES, default="tabletop_easy")
    run.add_argument("--scenario-json", type=Path)
    run.add_argument("--grasp-strategy")
    run.add_argument("--full-candidate-limit", type=int, default=2)
    run.add_argument("--position-tolerance", type=float, default=0.035)
    run.add_argument("--output-dir", type=Path, default=Path("results/v2_run"))
    run.add_argument("--json", action="store_true")

    view = sub.add_parser("view")
    view.add_argument("--xml", type=Path, required=True)
    view.add_argument("--replay", type=Path, required=True)
    view.add_argument("--interactive", action="store_true")
    view.add_argument("--fps", type=float, default=60.0)
    view.add_argument("--playback-speed", type=float, default=0.55)
    view.add_argument("--output", type=Path)

    browse = sub.add_parser("browse")
    browse.add_argument("--panda-xml", type=Path, required=True)
    browse.add_argument("--auto-fit-panda", action="store_true")
    browse.add_argument("--objects", nargs="+", choices=OBJECT_TYPES, default=list(OBJECT_TYPES))
    browse.add_argument("--tasks", nargs="+", choices=TASK_TYPES, default=list(TASK_TYPES))
    browse.add_argument("--position-tolerance", type=float, default=0.035)
    browse.add_argument("--output-dir", type=Path, default=Path("results/browser"))
    browse.add_argument("--fps", type=float, default=60.0)
    browse.add_argument("--playback-speed", type=float, default=0.55)

    for name in ("validate-physics", "validate-dynamics"):
        validate = sub.add_parser(name)
        validate.add_argument("--xml", type=Path, required=True)
        validate.add_argument("--replay", type=Path, required=True)
        validate.add_argument("--settle-steps", type=int, default=80)
        validate.add_argument("--position-tolerance", type=float, default=0.035)
        validate.add_argument("--weld-tracking-tolerance", type=float, default=0.03)
        validate.add_argument("--frame-substeps", type=int, default=4)
        validate.add_argument("--dynamics-gate", choices=tuple(DYNAMICS_GATE_PRESETS), default="custom")
        validate.add_argument("--max-gripper-penetration", type=float)
        validate.add_argument("--require-two-sided-grasp", action="store_true")
        validate.add_argument("--output", type=Path)

    validate_task = sub.add_parser("validate-task-physics")
    validate_task.add_argument("--replay", type=Path, required=True)
    validate_task.add_argument("--position-tolerance", type=float, default=0.035)
    validate_task.add_argument("--retention-tolerance", type=float, default=0.015)
    validate_task.add_argument("--support-tolerance", type=float, default=0.006)
    validate_task.add_argument("--clearance-tolerance", type=float, default=0.0)
    validate_task.add_argument("--output", type=Path)

    dynamics = sub.add_parser("dynamics-benchmark")
    dynamics.add_argument("--objects", nargs="+", choices=OBJECT_TYPES, default=["cube", "sphere", "plate", "mug_proxy"])
    dynamics.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_TYPES,
        default=["tabletop_easy", "over_barrier", "shelf_place", "narrow_slot"],
    )
    dynamics.add_argument("--panda-xml", type=Path, required=True)
    dynamics.add_argument("--auto-fit-panda", action="store_true")
    dynamics.add_argument("--position-tolerance", type=float, default=0.05)
    dynamics.add_argument("--full-candidate-limit", type=int, default=1)
    dynamics.add_argument("--settle-steps", type=int, default=80)
    dynamics.add_argument("--weld-tracking-tolerance", type=float, default=0.03)
    dynamics.add_argument("--frame-substeps", type=int, default=8)
    dynamics.add_argument("--dynamics-gate", choices=tuple(DYNAMICS_GATE_PRESETS), default="custom")
    dynamics.add_argument("--max-gripper-penetration", type=float)
    dynamics.add_argument("--require-two-sided-grasp", action="store_true")
    dynamics.add_argument(
        "--results-cache",
        type=Path,
        help="Optional incremental cache for expensive dynamics benchmark rows; enables resume after timeout.",
    )
    dynamics.add_argument(
        "--no-reuse-results-cache",
        dest="reuse_results_cache",
        action="store_false",
        default=True,
        help="Ignore existing dynamics benchmark cache rows and recompute all cases.",
    )
    dynamics.add_argument("--output-dir", type=Path, default=Path("results/dynamics_benchmark"))
    dynamics.add_argument("--output", type=Path)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--protocol", choices=PROTOCOL_TYPES)
    benchmark.add_argument("--objects", nargs="+", choices=OBJECT_TYPES)
    benchmark.add_argument("--tasks", nargs="+", choices=TASK_TYPES)
    benchmark.add_argument("--panda-xml", type=Path)
    benchmark.add_argument("--auto-fit-panda", action="store_true")
    benchmark.add_argument("--full-candidate-limit", type=int, default=2)
    benchmark.add_argument("--position-tolerance", type=float, default=0.05)
    benchmark.add_argument(
        "--results-cache",
        type=Path,
        help="Optional incremental cache for expensive benchmark rows; enables resume after timeout.",
    )
    benchmark.add_argument(
        "--no-reuse-results-cache",
        dest="reuse_results_cache",
        action="store_false",
        default=True,
        help="Ignore existing benchmark cache rows and recompute all cases.",
    )
    benchmark.add_argument("--output", type=Path, required=True)

    stress = sub.add_parser("stress-benchmark")
    stress.add_argument("--objects", nargs="+", choices=OBJECT_TYPES)
    stress.add_argument("--tasks", nargs="+", choices=TASK_TYPES)
    stress.add_argument("--panda-xml", type=Path)
    stress.add_argument("--auto-fit-panda", action="store_true")
    stress.add_argument("--full-candidate-limit", type=int, default=2)
    stress.add_argument("--position-tolerance", type=float, default=0.05)
    stress.add_argument(
        "--results-cache",
        type=Path,
        help="Optional incremental cache for expensive stress benchmark rows; enables resume after timeout.",
    )
    stress.add_argument(
        "--no-reuse-results-cache",
        dest="reuse_results_cache",
        action="store_false",
        default=True,
        help="Ignore existing stress benchmark cache rows and recompute all cases.",
    )
    stress.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser("compare-grasps")
    compare.add_argument("--panda-xml", type=Path)
    compare.add_argument("--auto-fit-panda", action="store_true")
    compare.add_argument("--object", dest="object_type", choices=OBJECT_TYPES, default="cube")
    compare.add_argument("--task", choices=TASK_TYPES, default="tabletop_easy")
    compare.add_argument("--scenario-json", type=Path)
    compare.add_argument("--full-candidate-limit", type=int, default=3)
    compare.add_argument("--position-tolerance", type=float, default=0.05)
    compare.add_argument("--output-dir", type=Path, required=True)

    browse_candidates = sub.add_parser("browse-candidates")
    browse_candidates.add_argument("--manifest", type=Path, required=True)
    browse_candidates.add_argument("--fps", type=float, default=60.0)
    browse_candidates.add_argument("--playback-speed", type=float, default=0.55)

    analysis = sub.add_parser("analyze-benchmark")
    analysis.add_argument("--input", type=Path, required=True)
    analysis.add_argument("--output-json", type=Path, required=True)
    analysis.add_argument("--output-csv", type=Path)
    analysis.add_argument("--output-md", type=Path)

    dynamics_analysis = sub.add_parser("analyze-dynamics-benchmark")
    dynamics_analysis.add_argument("--input", type=Path, required=True)
    dynamics_analysis.add_argument("--output-json", type=Path, required=True)
    dynamics_analysis.add_argument("--output-md", type=Path)

    morph = sub.add_parser("morphology-benchmark")
    morph.add_argument("--protocol", choices=PROTOCOL_TYPES)
    morph.add_argument("--arm-designs", nargs="+", choices=ARM_DESIGN_TYPES, default=list(ARM_DESIGN_TYPES))
    morph.add_argument("--objects", nargs="+", choices=OBJECT_TYPES)
    morph.add_argument("--tasks", nargs="+", choices=TASK_TYPES)
    morph.add_argument("--panda-xml", type=Path)
    morph.add_argument("--auto-fit-panda", action="store_true")
    morph.add_argument("--full-candidate-limit", type=int, default=2)
    morph.add_argument("--position-tolerance", type=float, default=0.05)
    morph.add_argument(
        "--minimum-reach-margin",
        type=float,
        default=-0.02,
        help=(
            "Required design reach margin in meters. The default preserves the "
            "legacy permissive proxy; use a positive value such as 0.03 for "
            "robust reachability."
        ),
    )
    morph.add_argument("--output", type=Path, required=True)

    optimize = sub.add_parser("optimize-morphology")
    optimize.add_argument("--protocol", choices=PROTOCOL_TYPES)
    optimize.add_argument("--objects", nargs="+", choices=OBJECT_TYPES)
    optimize.add_argument("--tasks", nargs="+", choices=TASK_TYPES)
    optimize.add_argument("--panda-xml", type=Path)
    optimize.add_argument("--auto-fit-panda", action="store_true")
    optimize.add_argument("--full-candidate-limit", type=int, default=2)
    optimize.add_argument("--position-tolerance", type=float, default=0.05)
    optimize.add_argument("--scale-values", nargs="+", type=float, default=[0.82, 0.9, 1.0, 1.1])
    optimize.add_argument("--upper-scales", nargs="+", type=float)
    optimize.add_argument("--forearm-scales", nargs="+", type=float)
    optimize.add_argument("--wrist-scales", nargs="+", type=float)
    optimize.add_argument("--base-x-values", nargs="+", type=float, default=[0.0, 0.03, 0.05])
    optimize.add_argument("--base-y-values", nargs="+", type=float, default=[0.0])
    optimize.add_argument("--minimum-reach-margin", type=float, default=0.03)
    optimize.add_argument("--path-cost-weight", type=float, default=0.02)
    optimize.add_argument("--minimum-sigma", type=float)
    optimize.add_argument("--maximum-condition-number", type=float)
    optimize.add_argument(
        "--base-results-cache",
        type=Path,
        help=(
            "Optional incremental cache for expensive object-task base evaluations. "
            "When present, completed base cases are reused and missing cases are appended."
        ),
    )
    optimize.add_argument(
        "--no-reuse-base-results",
        dest="reuse_base_results",
        action="store_false",
        default=True,
        help="Ignore existing base-results cache entries and recompute base cases.",
    )
    optimize.add_argument("--output", type=Path, required=True)

    robust = sub.add_parser("robustness-benchmark")
    robust.add_argument("--protocol", choices=PROTOCOL_TYPES)
    robust.add_argument("--objects", nargs="+", choices=OBJECT_TYPES)
    robust.add_argument("--tasks", nargs="+", choices=TASK_TYPES)
    robust.add_argument("--panda-xml", type=Path)
    robust.add_argument("--auto-fit-panda", action="store_true")
    robust.add_argument("--full-candidate-limit", type=int, default=2)
    robust.add_argument("--position-tolerance", type=float, default=0.05)
    robust.add_argument("--trials", type=int, default=10)
    robust.add_argument("--seed", type=int, default=42)
    robust.add_argument("--position-noise", type=float, default=0.01)
    robust.add_argument("--obstacle-noise", type=float, default=0.01)
    robust.add_argument(
        "--results-cache",
        type=Path,
        help="Optional incremental cache for expensive robustness runs; enables resume after interruption.",
    )
    robust.add_argument(
        "--no-reuse-results-cache",
        dest="reuse_results_cache",
        action="store_false",
        default=True,
        help="Ignore existing robustness cache rows and recompute all trials.",
    )
    robust.add_argument("--output", type=Path, required=True)

    failure = sub.add_parser("failure-analysis")
    failure.add_argument("--input", type=Path, required=True)
    failure.add_argument("--output-json", type=Path, required=True)
    failure.add_argument("--output-md", type=Path)

    design_analysis = sub.add_parser("design-analysis")
    design_analysis.add_argument("--input", type=Path, required=True)
    design_analysis.add_argument("--output-json", type=Path, required=True)
    design_analysis.add_argument("--output-md", type=Path)

    claim_audit = sub.add_parser("claim-audit")
    claim_audit.add_argument("--input", type=Path, required=True)
    claim_audit.add_argument("--output-json", type=Path, required=True)
    claim_audit.add_argument("--output-md", type=Path)

    ablation = sub.add_parser("ablation-report")
    ablation.add_argument("--final-summary", type=Path, required=True)
    ablation.add_argument("--morphology", type=Path)
    ablation.add_argument("--benchmark-summary", type=Path)
    ablation.add_argument("--output-json", type=Path, required=True)
    ablation.add_argument("--output-md", type=Path)

    robocasa_snapshot = sub.add_parser("robocasa-snapshot")
    robocasa_snapshot.add_argument("--env-name", default="PickPlaceCoffee")
    robocasa_snapshot.add_argument("--robot", default="PandaOmron")
    robocasa_snapshot.add_argument("--behavior", default="machine_to_counter")
    robocasa_snapshot.add_argument("--seed", type=int, default=42)
    robocasa_snapshot.add_argument("--from-json", type=Path)
    robocasa_snapshot.add_argument("--output-json", type=Path, required=True)
    robocasa_snapshot.add_argument("--output-scenario", type=Path)

    visualize = sub.add_parser("visualize-results")
    visualize.add_argument("--benchmark", type=Path, required=True)
    visualize.add_argument("--morphology", type=Path, required=True)
    visualize.add_argument("--output-dir", type=Path, required=True)
    return root


def _write_json(payload: dict, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_execution_success(metrics: dict, joint_replay: dict | None) -> bool:
    if not bool(metrics.get("success", False)):
        return False
    if joint_replay is None:
        return True
    return bool(joint_replay.get("success", False))


def _scenario_from_args(args: argparse.Namespace) -> PickPlaceScenario:
    scenario_json = getattr(args, "scenario_json", None)
    if scenario_json is not None:
        return load_scenario_json(scenario_json)
    object_type = getattr(args, "object_type", "cube")
    task_name = getattr(args, "task", "tabletop_easy")
    if getattr(args, "auto_fit_panda", False):
        return auto_fit_panda_scenario(
            args.panda_xml,
            object_type=object_type,
            task_name=task_name,
        )
    return make_scenario(object_type=object_type, task_name=task_name)


def _grasp_strategy_payload(scenario: PickPlaceScenario, strategy: GraspStrategy | None = None) -> dict:
    strategy = strategy or strategy_for_object(scenario.object_type)
    return {
        "name": strategy.name,
        "approach_axis": strategy.approach_axis,
        "tcp_offset": list(strategy.tcp_offset),
        "tcp_quat": list(strategy.tcp_quat),
        "requires_orientation": strategy.requires_orientation,
        "orientation_tolerance": strategy.orientation_tolerance,
        "orientation_weight": strategy.orientation_weight,
        "hold_orientation": strategy.hold_orientation,
        "description": strategy.description,
    }


def _strategy_named_for_object(scenario: PickPlaceScenario, name: str | None) -> GraspStrategy | None:
    if name is None:
        return None
    for strategy in strategies_for_object(scenario.object_type):
        if strategy.name == name:
            return strategy
    available = ", ".join(strategy.name for strategy in strategies_for_object(scenario.object_type))
    raise ValueError(
        f"unknown grasp strategy {name!r} for object {scenario.object_type!r}; "
        f"available strategies: {available}"
    )


def _explicit_grasp_selection_payload(strategy: GraspStrategy) -> dict:
    return {
        "selection_mode": "explicit",
        "selected_strategy": strategy.name,
        "selected_evaluation": None,
        "candidate_evaluations": [],
    }


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def _write_replay_case(
    *,
    object_type: str,
    task_name: str,
    panda_xml: Path,
    auto_fit_panda: bool,
    position_tolerance: float,
    full_candidate_limit: int,
    output_dir: Path,
    grasp_strategy_name: str | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = (
        auto_fit_panda_scenario(
            panda_xml,
            object_type=object_type,
            task_name=task_name,
        )
        if auto_fit_panda
        else make_scenario(object_type=object_type, task_name=task_name)
    )
    if grasp_strategy_name is None:
        selection = select_grasp_plan(
            scenario,
            panda_xml=panda_xml,
            output_dir=output_dir / "candidate_scenes",
            position_tolerance=position_tolerance,
            full_candidate_limit=full_candidate_limit,
        )
        selected_strategy = selection.selected_strategy
        phases = selection.phases
        grasp_selection = selection.to_json_dict()
    else:
        selected_strategy = _strategy_named_for_object(scenario, grasp_strategy_name)
        assert selected_strategy is not None
        phases = plan_pick_place(scenario, grasp_strategy=selected_strategy)
        grasp_selection = _explicit_grasp_selection_payload(selected_strategy)

    frames = build_replay_frames(scenario, phases)
    metrics = evaluate_replay(scenario, frames)
    physical_evidence = evaluate_physical_evidence(
        scenario,
        frames,
        position_tolerance=position_tolerance,
    )
    scene = build_pick_place_scene(
        panda_xml=panda_xml,
        scenario=scenario,
        output_xml=output_dir / "scene.xml",
    )
    joint_replay = solve_joint_replay(
        scene.output_xml,
        frames,
        tolerance=position_tolerance,
        orientation_tolerance=float(selected_strategy.orientation_tolerance),
    ).to_json_dict()
    payload = {
        "schema_version": 1,
        "scenario": scenario.to_json_dict(),
        "grasp_strategy": _grasp_strategy_payload(scenario, selected_strategy),
        "grasp_selection": grasp_selection,
        "phases": [phase.to_json_dict() for phase in phases],
        "frames": [frame.to_json_dict() for frame in frames],
        "metrics": metrics,
        "physical_evidence": physical_evidence,
        "scene": scene.to_json_dict(),
        "joint_replay": joint_replay,
    }
    payload["execution_success"] = bool(
        run_execution_success(metrics, joint_replay)
        and physical_evidence["success"]
    )
    replay_path = output_dir / "replay.json"
    replay_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "scene_xml": scene.output_xml,
        "replay_path": replay_path,
        "execution_success": payload["execution_success"],
        "scenario": scenario.to_json_dict(),
        "payload": payload,
    }


def _dynamics_grasp_strategy_names(
    object_type: str,
    *,
    require_two_sided_grasp: bool,
    full_candidate_limit: int,
) -> list[str | None]:
    if not require_two_sided_grasp:
        return [None]
    limit = max(1, int(full_candidate_limit))
    return [strategy.name for strategy in strategies_for_object(object_type)[:limit]]


def _dynamics_failure_reasons(replay_case: dict, dynamics: dict) -> list[str]:
    failure_reasons = []
    if not bool(replay_case["execution_success"]):
        failure_reasons.append("execution")
    failure_reasons.extend(f"dynamics:{reason}" for reason in dynamics["failure_reasons"])
    return failure_reasons


def _replay_case_grasp_strategy_payload(
    replay_case: dict,
    fallback_strategy_name: str | None,
) -> dict:
    payload = replay_case.get("payload") or {}
    strategy = payload.get("grasp_strategy")
    if isinstance(strategy, dict) and strategy.get("name"):
        return strategy
    return {"name": fallback_strategy_name or "auto_selected"}


def _finite_metric(value: object, default: float = 1.0e9) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or abs(number) == float("inf"):
        return default
    return number


def _dynamics_attempt_quality(attempt: dict) -> tuple:
    """Lower is better; designed for selecting the least-bad physics candidate."""

    evidence = attempt.get("dynamics_evidence") or {}
    checks = evidence.get("checks") or {}
    two_sided = checks.get("two_sided_grasp_contact") or {}
    penetration = checks.get("gripper_penetration") or {}
    dominant_side = penetration.get("dominant_penetration_side")
    max_penetration = penetration.get("gripper_object_max_penetration")
    if max_penetration is None:
        by_side = penetration.get("gripper_object_max_penetration_by_side") or {}
        max_penetration = max(
            (_finite_metric(value, default=0.0) for value in by_side.values()),
            default=1.0e9,
        )
    return (
        0 if bool(attempt.get("success")) else 1,
        0 if bool(two_sided.get("passed", True)) else 1,
        len(attempt.get("failure_reasons") or []),
        1 if dominant_side == "palm_or_hand" else 0,
        _finite_metric(max_penetration),
        _finite_metric(evidence.get("final_error")),
        str(attempt.get("strategy_name") or ""),
    )


def _run_dynamics_benchmark(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate = _dynamics_gate_payload(args)
    cache_path = getattr(args, "results_cache", None)
    cache_config = _dynamics_benchmark_cache_config(args, gate=gate)
    rows_by_key = _load_dynamics_benchmark_cache(
        cache_path,
        expected_config=cache_config,
        reuse=getattr(args, "reuse_results_cache", True),
    )
    reused_rows = 0
    computed_rows = 0
    results = []
    for object_type in args.objects:
        for task_name in args.tasks:
            key = _benchmark_cache_key(object_type, task_name)
            if key in rows_by_key:
                results.append(rows_by_key[key])
                reused_rows += 1
                continue
            case_dir = args.output_dir / f"{_safe_name(object_type)}__{_safe_name(task_name)}"
            row = {
                "object_type": object_type,
                "task_name": task_name,
                "case_dir": str(case_dir),
            }
            try:
                attempts = []
                scored_attempts = []
                strategy_names = _dynamics_grasp_strategy_names(
                    object_type,
                    require_two_sided_grasp=bool(gate["require_two_sided_grasp"]),
                    full_candidate_limit=args.full_candidate_limit,
                )
                for index, strategy_name in enumerate(strategy_names, start=1):
                    attempt_dir = (
                        case_dir
                        if strategy_name is None
                        else case_dir / "attempts" / f"{index:02d}_{_safe_name(strategy_name)}"
                    )
                    replay_case = _write_replay_case(
                        object_type=object_type,
                        task_name=task_name,
                        panda_xml=args.panda_xml,
                        auto_fit_panda=args.auto_fit_panda,
                        position_tolerance=args.position_tolerance,
                        full_candidate_limit=args.full_candidate_limit,
                        output_dir=attempt_dir,
                        grasp_strategy_name=strategy_name,
                    )
                    dynamics = validate_physics_replay(
                        replay_case["scene_xml"],
                        replay_case["replay_path"],
                        settle_steps=args.settle_steps,
                        position_tolerance=args.position_tolerance,
                        weld_tracking_tolerance=args.weld_tracking_tolerance,
                        frame_substeps=args.frame_substeps,
                        max_gripper_penetration=gate["max_gripper_penetration"],
                        require_two_sided_grasp=bool(gate["require_two_sided_grasp"]),
                    )
                    success = bool(replay_case["execution_success"] and dynamics["success"])
                    failure_reasons = _dynamics_failure_reasons(replay_case, dynamics)
                    grasp_strategy = _replay_case_grasp_strategy_payload(
                        replay_case,
                        strategy_name,
                    )
                    attempt = {
                        "strategy_name": grasp_strategy["name"],
                        "success": success,
                        "failure_reasons": failure_reasons,
                        "scene_xml": str(replay_case["scene_xml"]),
                        "replay_json": str(replay_case["replay_path"]),
                        "execution_success": bool(replay_case["execution_success"]),
                        "dynamics_evidence": dynamics,
                    }
                    attempts.append(attempt)
                    scored_attempts.append(
                        {
                            "attempt": attempt,
                            "replay_case": replay_case,
                            "dynamics": dynamics,
                            "success": success,
                            "failure_reasons": failure_reasons,
                        }
                    )

                selected_attempt = min(
                    scored_attempts,
                    key=lambda item: _dynamics_attempt_quality(item["attempt"]),
                )
                assert selected_attempt is not None
                replay_case = selected_attempt["replay_case"]
                dynamics = selected_attempt["dynamics"]
                success = bool(selected_attempt["success"])
                failure_reasons = list(selected_attempt["failure_reasons"])
                row.update(
                    {
                        "success": success,
                        "failure_reasons": failure_reasons,
                        "scene_xml": str(replay_case["scene_xml"]),
                        "replay_json": str(replay_case["replay_path"]),
                        "scenario": replay_case["scenario"],
                        "grasp_strategy": _replay_case_grasp_strategy_payload(
                            replay_case,
                            None,
                        ),
                        "candidate_selection": {
                            "method": "dynamics_quality_score",
                            "selected_strategy": selected_attempt["attempt"][
                                "strategy_name"
                            ],
                            "selected_quality": list(
                                _dynamics_attempt_quality(selected_attempt["attempt"])
                            ),
                        },
                        "candidate_dynamics_attempts": attempts,
                        "execution_success": bool(replay_case["execution_success"]),
                        "dynamics_evidence": dynamics,
                    }
                )
            except Exception as error:  # pragma: no cover - exercised by integration failures
                row.update(
                    {
                        "success": False,
                        "failure_reasons": [f"error:{type(error).__name__}:{error}"],
                        "dynamics_evidence": None,
                    }
                )
            results.append(row)
            rows_by_key[key] = row
            computed_rows += 1
            _write_dynamics_benchmark_cache(
                cache_path,
                config=cache_config,
                rows_by_key=rows_by_key,
            )
    successful = [row for row in results if row["success"]]
    payload = {
        "schema_version": 1,
        "evidence_scope": "batch MuJoCo equality-weld dynamics validation",
        "objects": list(args.objects),
        "tasks": list(args.tasks),
        "total_runs": len(results),
        "successful_runs": len(successful),
        "summary": {
            "success_rate": 0.0 if not results else len(successful) / len(results),
            "failed_runs": len(results) - len(successful),
        },
        "cache": {
            "path": None if cache_path is None else str(cache_path),
            "reused_rows": reused_rows,
            "computed_rows": computed_rows,
            "total_cached_rows": len(rows_by_key),
        },
        "settings": {
            "panda_xml": str(args.panda_xml),
            "auto_fit_panda": bool(args.auto_fit_panda),
            "position_tolerance": float(args.position_tolerance),
            "full_candidate_limit": int(args.full_candidate_limit),
            "settle_steps": int(args.settle_steps),
            "frame_substeps": int(args.frame_substeps),
            "weld_tracking_tolerance": float(args.weld_tracking_tolerance),
            "dynamics_gate": gate,
            "max_gripper_penetration": gate["max_gripper_penetration"],
            "require_two_sided_grasp": bool(gate["require_two_sided_grasp"]),
        },
        "results": results,
    }
    return payload


def _dynamics_benchmark_cache_config(args: argparse.Namespace, *, gate: dict[str, object]) -> dict:
    return {
        "panda_xml": str(Path(args.panda_xml).expanduser()),
        "auto_fit_panda": bool(args.auto_fit_panda),
        "position_tolerance": float(args.position_tolerance),
        "full_candidate_limit": int(args.full_candidate_limit),
        "settle_steps": int(args.settle_steps),
        "weld_tracking_tolerance": float(args.weld_tracking_tolerance),
        "frame_substeps": int(args.frame_substeps),
        "dynamics_gate": gate,
    }


def _load_dynamics_benchmark_cache(
    path: Path | None,
    *,
    expected_config: dict,
    reuse: bool,
) -> dict[str, dict]:
    if path is None or not reuse or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _benchmark_cache_compatible(payload.get("config"), expected_config):
        return {}
    rows = {}
    for row in payload.get("results", ()):
        key = _benchmark_cache_key(str(row["object_type"]), str(row["task_name"]))
        rows[key] = dict(row)
    return rows


def _write_dynamics_benchmark_cache(
    path: Path | None,
    *,
    config: dict,
    rows_by_key: dict[str, dict],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "cache_type": "dynamics_benchmark_results",
        "config": config,
        "results": [rows_by_key[key] for key in sorted(rows_by_key)],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_candidate_replays(
    scenario: PickPlaceScenario,
    evaluations: list | tuple,
    output_dir: Path,
    *,
    selected_strategy: str,
    panda_xml: Path | None = None,
    position_tolerance: float = 0.05,
) -> dict:
    replay_root = output_dir / "candidate_replays"
    replay_root.mkdir(parents=True, exist_ok=True)
    output_base = output_dir.resolve()
    strategy_lookup = {strategy.name: strategy for strategy in strategies_for_object(scenario.object_type)}
    rows = []
    for index, evaluation in enumerate(evaluations, start=1):
        strategy = strategy_lookup.get(evaluation.strategy_name)
        if strategy is None:
            continue
        candidate_dir = replay_root / f"{index:02d}_{_safe_name(strategy.name)}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        phases = plan_pick_place(scenario, grasp_strategy=strategy)
        frames = build_replay_frames(scenario, phases)
        metrics = evaluate_replay(scenario, frames)
        payload = {
            "schema_version": 1,
            "candidate_index": index,
            "scenario": scenario.to_json_dict(),
            "grasp_strategy": _grasp_strategy_payload(scenario, strategy),
            "selected": strategy.name == selected_strategy,
            "candidate_evaluation": evaluation.to_json_dict(),
            "phases": [phase.to_json_dict() for phase in phases],
            "frames": [frame.to_json_dict() for frame in frames],
            "metrics": metrics,
        }
        scene_path = None
        if panda_xml is not None:
            scene = build_pick_place_scene(
                panda_xml=panda_xml,
                scenario=scenario,
                output_xml=candidate_dir / "scene.xml",
            )
            joint_replay = solve_joint_replay(
                scene.output_xml,
                frames,
                tolerance=position_tolerance,
                orientation_tolerance=float(strategy.orientation_tolerance),
            ).to_json_dict()
            payload["scene"] = scene.to_json_dict()
            payload["joint_replay"] = joint_replay
            payload["execution_success"] = run_execution_success(metrics, joint_replay)
            scene_path = scene.output_xml
        else:
            payload["execution_success"] = bool(metrics["success"])
        replay_path = candidate_dir / "replay.json"
        replay_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "index": index,
                "strategy_name": strategy.name,
                "selected": strategy.name == selected_strategy,
                "success": bool(payload["execution_success"]),
                "replay_json": str(replay_path.resolve().relative_to(output_base)),
                "scene_xml": None if scene_path is None else str(scene_path.resolve().relative_to(output_base)),
            }
        )
    manifest = {
        "schema_version": 1,
        "selected_strategy": selected_strategy,
        "candidates": rows,
    }
    manifest_path = replay_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest, "manifest_path": manifest_path}


def _run_static_case(
    object_type: str,
    task_name: str,
    *,
    scenario: PickPlaceScenario | None = None,
    panda_xml: Path | None = None,
    auto_fit_panda: bool = False,
    position_tolerance: float = 0.05,
    full_candidate_limit: int = 2,
) -> dict:
    if scenario is None and panda_xml is not None and auto_fit_panda:
        scenario = auto_fit_panda_scenario(
            panda_xml,
            object_type=object_type,
            task_name=task_name,
        )
    elif scenario is None:
        scenario = make_scenario(object_type=object_type, task_name=task_name)
    if panda_xml is None:
        selection = select_grasp_plan(
            scenario,
            position_tolerance=position_tolerance,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="morphtamp_x_v2_grasp_select_") as selection_tmp:
            selection = select_grasp_plan(
                scenario,
                panda_xml=panda_xml,
                output_dir=Path(selection_tmp),
                position_tolerance=position_tolerance,
                full_candidate_limit=full_candidate_limit,
            )
    phases = selection.phases
    frames = build_replay_frames(scenario, phases)
    metrics = evaluate_replay(scenario, frames)
    physical_evidence = evaluate_physical_evidence(
        scenario,
        frames,
        position_tolerance=position_tolerance,
    )
    joint_metrics = None
    if panda_xml is not None:
        with tempfile.TemporaryDirectory(prefix="morphtamp_x_v2_") as temporary:
            scene = build_pick_place_scene(
                panda_xml=panda_xml,
                scenario=scenario,
                output_xml=Path(temporary) / "scene.xml",
            )
            replay = solve_joint_replay(
                scene.output_xml,
                frames,
                tolerance=position_tolerance,
                orientation_tolerance=float(selection.selected_strategy.orientation_tolerance),
            )
        joint_metrics = {
            "success": selection.selected_evaluation.success,
            "max_position_error": selection.selected_evaluation.max_position_error,
            "max_orientation_error": selection.selected_evaluation.max_orientation_error,
            "min_joint_margin": selection.selected_evaluation.min_joint_margin,
            "joint_path_length": selection.selected_evaluation.joint_path_length,
            "max_joint_step": selection.selected_evaluation.max_joint_step,
            "energy_proxy": selection.selected_evaluation.energy_proxy,
            "smoothness_proxy": selection.selected_evaluation.smoothness_proxy,
            "min_sigma": selection.selected_evaluation.min_sigma,
            "max_condition_number": selection.selected_evaluation.max_condition_number,
            "failure_reasons": list(selection.selected_evaluation.failure_reasons),
            "joint_count": len(replay.joint_names),
            "candidate_count": len(selection.evaluations),
        }
    path_length = path_length_for_phases(phases)
    grasp_selection = selection.to_json_dict()
    return {
        "object_type": object_type,
        "task_name": task_name,
        "success": bool(metrics["success"] and physical_evidence["success"]),
        "failure_reasons": list(metrics["failure_reasons"]) + [
            f"physical:{reason}" for reason in physical_evidence["failure_reasons"]
        ],
        "path_length": path_length,
        "lift_height": scenario.lift_height,
        "has_obstacle": scenario.obstacle_center is not None,
        "grasp_strategy": _grasp_strategy_payload(scenario, selection.selected_strategy),
        "grasp_selection": grasp_selection,
        "grasp_selection_summary": summarize_grasp_selection(grasp_selection),
        "metrics": metrics,
        "physical_evidence": physical_evidence,
        "joint_metrics": joint_metrics,
        "scenario": scenario.to_json_dict(),
    }


def _workspace_demand(row: dict) -> float:
    scenario = row["scenario"]
    points = [
        scenario["cube_start"],
        scenario["place_target"],
    ]
    if scenario.get("obstacle_center") is not None:
        points.append(scenario["obstacle_center"])
    import numpy as np

    return max(float(np.linalg.norm(np.asarray(point, dtype=float))) for point in points)


def _run_morphology_case(
    design_name: str,
    object_type: str,
    task_name: str,
    *,
    panda_xml: Path | None = None,
    auto_fit_panda: bool = False,
    position_tolerance: float = 0.05,
    full_candidate_limit: int = 2,
    minimum_reach_margin: float = -0.02,
) -> dict:
    design = arm_design(design_name)
    row = _run_static_case(
        object_type,
        task_name,
        panda_xml=panda_xml,
        auto_fit_panda=auto_fit_panda,
        position_tolerance=position_tolerance,
        full_candidate_limit=full_candidate_limit,
    )
    demand = _workspace_demand(row)
    reach_margin = design.reach_proxy - demand
    base_success = row["success"] and (row["joint_metrics"] is None or row["joint_metrics"]["success"])
    morphology_success = base_success and reach_margin >= minimum_reach_margin
    failure_reasons = list(row["failure_reasons"])
    if not morphology_success and reach_margin < minimum_reach_margin:
        failure_reasons.append(f"reach_margin:{reach_margin:.6f}")
    return {
        **row,
        "arm_design": design.name,
        "design": design.to_json_dict(),
        "morphology_cost": design.morphology_cost,
        "reach_proxy": design.reach_proxy,
        "workspace_demand": demand,
        "reach_margin": reach_margin,
        "minimum_reach_margin": minimum_reach_margin,
        "success": morphology_success,
        "failure_reasons": failure_reasons,
    }


def _summarize_morphology(results: list[dict]) -> list[dict]:
    summaries = []
    for design_name in sorted({row["arm_design"] for row in results}):
        rows = [row for row in results if row["arm_design"] == design_name]
        successes = [row for row in rows if row["success"]]
        summaries.append(
            {
                "arm_design": design_name,
                "runs": len(rows),
                "successes": len(successes),
                "success_rate": 0.0 if not rows else len(successes) / len(rows),
                "morphology_cost": arm_design(design_name).morphology_cost,
                "mean_path_length": 0.0
                if not successes
                else sum(float(row["path_length"]) for row in successes) / len(successes),
                "minimum_reach_margin": min(float(row["reach_margin"]) for row in rows),
            }
        )
    return summaries


def _run_benchmark_payload(
    args: argparse.Namespace,
    *,
    protocol_name: str | None = None,
) -> tuple[dict, list[dict], dict | None]:
    objects, tasks, auto_fit_panda, protocol_payload = resolve_protocol_inputs(
        protocol_name=protocol_name,
        objects=args.objects,
        tasks=args.tasks,
        auto_fit_panda=args.auto_fit_panda,
    )
    cache_path = getattr(args, "results_cache", None)
    cache_config = _benchmark_cache_config(
        args,
        protocol_name=protocol_name,
        auto_fit_panda=auto_fit_panda,
    )
    rows_by_key = _load_benchmark_cache(
        cache_path,
        expected_config=cache_config,
        reuse=getattr(args, "reuse_results_cache", True),
    )
    results = []
    reused_rows = 0
    computed_rows = 0
    for object_type in objects:
        for task_name in tasks:
            key = _benchmark_cache_key(object_type, task_name)
            if key in rows_by_key:
                row = rows_by_key[key]
                reused_rows += 1
            else:
                row = _run_static_case(
                    object_type,
                    task_name,
                    panda_xml=args.panda_xml,
                    auto_fit_panda=auto_fit_panda,
                    position_tolerance=args.position_tolerance,
                    full_candidate_limit=args.full_candidate_limit,
                )
                rows_by_key[key] = row
                computed_rows += 1
                _write_benchmark_cache(
                    cache_path,
                    config=cache_config,
                    rows_by_key=rows_by_key,
                )
            results.append(row)
    feasible = [
        row
        for row in results
        if row["success"] and (row["joint_metrics"] is None or row["joint_metrics"]["success"])
    ]
    best = min(feasible, key=lambda row: (row["path_length"], row["lift_height"])) if feasible else None
    payload = {
        "schema_version": 1,
        "protocol": protocol_payload,
        "objects": list(objects),
        "tasks": list(tasks),
        "total_runs": len(results),
        "successful_runs": len(feasible),
        "panda_xml": None if args.panda_xml is None else str(args.panda_xml),
        "auto_fit_panda": bool(auto_fit_panda),
        "cache": {
            "path": None if cache_path is None else str(cache_path),
            "reused_rows": reused_rows,
            "computed_rows": computed_rows,
            "total_cached_rows": len(rows_by_key),
        },
        "best": None if best is None else {
            "object_type": best["object_type"],
            "task_name": best["task_name"],
            "path_length": best["path_length"],
        },
        "results": results,
    }
    return payload, feasible, best


def _benchmark_cache_key(object_type: str, task_name: str) -> str:
    return f"{object_type}::{task_name}"


def _benchmark_cache_config(
    args: argparse.Namespace,
    *,
    protocol_name: str | None,
    auto_fit_panda: bool,
) -> dict:
    return {
        "protocol_name": protocol_name,
        "panda_xml": None if args.panda_xml is None else str(Path(args.panda_xml).expanduser()),
        "auto_fit_panda": bool(auto_fit_panda),
        "position_tolerance": float(args.position_tolerance),
        "full_candidate_limit": int(args.full_candidate_limit),
    }


def _benchmark_cache_compatible(cached_config: object, expected_config: dict) -> bool:
    return isinstance(cached_config, dict) and cached_config == expected_config


def _load_benchmark_cache(
    path: Path | None,
    *,
    expected_config: dict,
    reuse: bool,
) -> dict[str, dict]:
    if path is None or not reuse or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _benchmark_cache_compatible(payload.get("config"), expected_config):
        return {}
    rows = {}
    for row in payload.get("results", ()):
        key = _benchmark_cache_key(str(row["object_type"]), str(row["task_name"]))
        rows[key] = dict(row)
    return rows


def _write_benchmark_cache(
    path: Path | None,
    *,
    config: dict,
    rows_by_key: dict[str, dict],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "cache_type": "object_task_benchmark_results",
        "config": config,
        "results": [rows_by_key[key] for key in sorted(rows_by_key)],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "list-objects":
        payload = {
            "schema_version": 1,
            "objects": list(OBJECT_TYPES),
            "details": {name: object_spec(name).__dict__ for name in OBJECT_TYPES},
        }
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "list-tasks":
        payload = {
            "schema_version": 1,
            "tasks": list(TASK_TYPES),
            "details": {name: task_spec(name).__dict__ for name in TASK_TYPES},
        }
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "check-tasks":
        payload = check_all_tasks(objects=OBJECT_TYPES, tasks=TASK_TYPES)
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0 if payload["passed"] else 1
    if args.command in {"benchmark", "stress-benchmark"}:
        protocol_name = "stress_fixed" if args.command == "stress-benchmark" else args.protocol
        payload, feasible, best = _run_benchmark_payload(args, protocol_name=protocol_name)
        _write_json(payload, args.output)
        label = "stress-benchmark" if args.command == "stress-benchmark" else "benchmark"
        print(
            f"{label}: success={len(feasible)}/{payload['total_runs']} "
            f"best={None if best is None else best['object_type'] + ':' + best['task_name']} "
            f"output={args.output}"
        )
        return 0
    if args.command == "compare-grasps":
        scenario = _scenario_from_args(args)
        if args.panda_xml is None:
            selection = select_grasp_plan(
                scenario,
                position_tolerance=args.position_tolerance,
            )
        else:
            selection = select_grasp_plan(
                scenario,
                panda_xml=args.panda_xml,
                output_dir=args.output_dir / "candidate_scenes",
                position_tolerance=args.position_tolerance,
                full_candidate_limit=args.full_candidate_limit,
            )
        report = build_candidate_report(
            scenario,
            selected_strategy=selection.selected_strategy.name,
            evaluations=selection.evaluations,
        )
        report["coarse_candidate_evaluations"] = [
            row.to_json_dict() for row in selection.coarse_evaluations
        ]
        replay_manifest = _write_candidate_replays(
            scenario,
            selection.evaluations,
            args.output_dir,
            selected_strategy=selection.selected_strategy.name,
            panda_xml=args.panda_xml,
            position_tolerance=args.position_tolerance,
        )
        report["candidate_replays"] = replay_manifest["manifest"]
        outputs = write_candidate_report(report, args.output_dir)
        print(
            f"compare-grasps: candidates={len(selection.evaluations)} "
            f"selected={selection.selected_strategy.name} html={outputs['html']} "
            f"replays={replay_manifest['manifest_path']}"
        )
        return 0
    if args.command == "browse-candidates":
        launch_candidate_viewer(
            args.manifest,
            fps=args.fps,
            playback_speed=args.playback_speed,
        )
        return 0
    if args.command == "analyze-benchmark":
        summary = summarize_benchmark(load_benchmark(args.input))
        _write_json(summary, args.output_json)
        if args.output_csv is not None:
            write_summary_csv(summary, args.output_csv)
        if args.output_md is not None:
            write_summary_markdown(summary, args.output_md)
        print(
            f"analyze-benchmark: success_rate={summary['overall']['success_rate']:.3f} "
            f"output={args.output_json}"
        )
        return 0
    if args.command == "morphology-benchmark":
        objects, tasks, auto_fit_panda, protocol_payload = resolve_protocol_inputs(
            protocol_name=args.protocol,
            objects=args.objects,
            tasks=args.tasks,
            auto_fit_panda=args.auto_fit_panda,
        )
        results = [
            _run_morphology_case(
                design_name,
                object_type,
                task_name,
                panda_xml=args.panda_xml,
                auto_fit_panda=auto_fit_panda,
                position_tolerance=args.position_tolerance,
                full_candidate_limit=args.full_candidate_limit,
                minimum_reach_margin=args.minimum_reach_margin,
            )
            for design_name in args.arm_designs
            for object_type in objects
            for task_name in tasks
        ]
        summaries = _summarize_morphology(results)
        feasible_summaries = [row for row in summaries if row["success_rate"] > 0.0]
        best = min(
            feasible_summaries,
            key=lambda row: (-row["success_rate"], row["morphology_cost"], row["mean_path_length"]),
            default=None,
        )
        payload = {
            "schema_version": 1,
            "evidence_scope": "equivalent arm morphology proxy plus optional Panda IK verification",
            "protocol": protocol_payload,
            "arm_designs": list(args.arm_designs),
            "objects": list(objects),
            "tasks": list(tasks),
            "total_runs": len(results),
            "successful_runs": sum(1 for row in results if row["success"]),
            "best_design": None if best is None else best["arm_design"],
            "pareto_designs": pareto_designs(summaries),
            "design_summaries": summaries,
            "results": results,
        }
        _write_json(payload, args.output)
        print(
            f"morphology-benchmark: success={payload['successful_runs']}/{payload['total_runs']} "
            f"best={payload['best_design']} output={args.output}"
        )
        return 0
    if args.command == "optimize-morphology":
        from .morphology_optimizer import optimize_morphology

        objects, tasks, auto_fit_panda, protocol_payload = resolve_protocol_inputs(
            protocol_name=args.protocol,
            objects=args.objects,
            tasks=args.tasks,
            auto_fit_panda=args.auto_fit_panda,
        )
        scale_values = tuple(args.scale_values)
        payload = optimize_morphology(
            objects=objects,
            tasks=tasks,
            upper_scales=tuple(args.upper_scales or scale_values),
            forearm_scales=tuple(args.forearm_scales or scale_values),
            wrist_scales=tuple(args.wrist_scales or scale_values),
            base_x_values=tuple(args.base_x_values),
            base_y_values=tuple(args.base_y_values),
            panda_xml=args.panda_xml,
            auto_fit_panda=auto_fit_panda,
            position_tolerance=args.position_tolerance,
            full_candidate_limit=args.full_candidate_limit,
            minimum_reach_margin=args.minimum_reach_margin,
            path_cost_weight=args.path_cost_weight,
            minimum_sigma=args.minimum_sigma,
            maximum_condition_number=args.maximum_condition_number,
            base_results_cache=args.base_results_cache,
            reuse_base_results=args.reuse_base_results,
        )
        payload["protocol"] = protocol_payload
        _write_json(payload, args.output)
        print(
            f"optimize-morphology: feasible={payload['feasible_candidates']}/{payload['total_candidates']} "
            f"best={None if payload['best_design'] is None else payload['best_design']['arm_design']} "
            f"output={args.output}"
        )
        return 0
    if args.command == "robustness-benchmark":
        from .robustness import run_robustness_benchmark

        objects, tasks, auto_fit_panda, protocol_payload = resolve_protocol_inputs(
            protocol_name=args.protocol,
            objects=args.objects,
            tasks=args.tasks,
            auto_fit_panda=args.auto_fit_panda,
        )
        payload = run_robustness_benchmark(
            objects=objects,
            tasks=tasks,
            trials=args.trials,
            seed=args.seed,
            position_noise=args.position_noise,
            obstacle_noise=args.obstacle_noise,
            panda_xml=args.panda_xml,
            auto_fit_panda=auto_fit_panda,
            position_tolerance=args.position_tolerance,
            full_candidate_limit=args.full_candidate_limit,
            results_cache=args.results_cache,
            reuse_results_cache=args.reuse_results_cache,
        )
        payload["protocol"] = protocol_payload
        _write_json(payload, args.output)
        print(
            f"robustness-benchmark: success={payload['successful_runs']}/{payload['total_runs']} "
            f"output={args.output}"
        )
        return 0
    if args.command == "failure-analysis":
        from .failure_analysis import build_failure_analysis

        payload = build_failure_analysis(json.loads(args.input.read_text(encoding="utf-8")))
        _write_json(payload, args.output_json)
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(payload["markdown"], encoding="utf-8")
        print(f"failure-analysis: failed={payload['failed_runs']} output={args.output_json}")
        return 0
    if args.command == "design-analysis":
        from .design_analysis import build_design_analysis

        payload = build_design_analysis(json.loads(args.input.read_text(encoding="utf-8")))
        _write_json(payload, args.output_json)
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(payload["markdown"], encoding="utf-8")
        recommendation = payload["recommendation"]["recommended_design"]
        print(
            "design-analysis: "
            f"recommended={None if recommendation is None else recommendation['arm_design']} "
            f"output={args.output_json}"
        )
        return 0
    if args.command == "claim-audit":
        from .claim_audit import load_summary, write_claim_audit_files

        audit = write_claim_audit_files(
            load_summary(args.input),
            output_json=args.output_json,
            output_md=args.output_md,
        )
        print(
            f"claim-audit: supported={audit['overall']['supported_claims']}/"
            f"{audit['overall']['claims']} output={args.output_json}"
        )
        return 0
    if args.command == "ablation-report":
        from .ablation import load_json, write_ablation_files

        report = write_ablation_files(
            final_summary=load_json(args.final_summary) or {},
            morphology_benchmark=load_json(args.morphology),
            benchmark_summary=load_json(args.benchmark_summary),
            output_json=args.output_json,
            output_md=args.output_md,
        )
        print(
            "ablation-report: "
            f"paper_ready={report['recommendation']['paper_ready']} "
            f"output={args.output_json}"
        )
        return 0
    if args.command == "robocasa-snapshot":
        from .robocasa_bridge import (
            create_live_robocasa_snapshot,
            extract_snapshot_from_mapping,
            write_robocasa_snapshot_files,
        )

        if args.from_json is None:
            snapshot = create_live_robocasa_snapshot(
                env_name=args.env_name,
                robot=args.robot,
                behavior=args.behavior,
                seed=args.seed,
            )
        else:
            snapshot = extract_snapshot_from_mapping(
                json.loads(args.from_json.read_text(encoding="utf-8"))
            )
        write_robocasa_snapshot_files(
            snapshot,
            output_json=args.output_json,
            output_scenario=args.output_scenario,
        )
        print(
            f"robocasa-snapshot: env={snapshot.env_name} behavior={snapshot.behavior} "
            f"object={snapshot.object_type_hint} task={snapshot.task_name_hint} "
            f"output={args.output_json}"
        )
        return 0
    if args.command == "visualize-results":
        manifest = visualize_results(args.benchmark, args.morphology, args.output_dir)
        print(f"visualize-results: dashboard={manifest['outputs']['dashboard']}")
        return 0
    if args.command == "browse":
        launch_browse_viewer(
            panda_xml=args.panda_xml,
            output_dir=args.output_dir,
            objects=tuple(args.objects),
            tasks=tuple(args.tasks),
            auto_fit_panda=args.auto_fit_panda,
            position_tolerance=args.position_tolerance,
            fps=args.fps,
            playback_speed=args.playback_speed,
        )
        return 0
    if args.command == "validate-task-physics":
        payload = evaluate_replay_json(
            args.replay,
            position_tolerance=args.position_tolerance,
            retention_tolerance=args.retention_tolerance,
            support_tolerance=args.support_tolerance,
            clearance_tolerance=args.clearance_tolerance,
        )
        _write_json(payload, args.output)
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "dynamics-benchmark":
        payload = _run_dynamics_benchmark(args)
        output = args.output or (args.output_dir / "dynamics_benchmark.json")
        _write_json(payload, output)
        print(
            f"dynamics-benchmark: success={payload['successful_runs']}/{payload['total_runs']} "
            f"output={output}"
        )
        return 0
    if args.command == "analyze-dynamics-benchmark":
        from .dynamics_analysis import build_dynamics_benchmark_analysis

        payload = build_dynamics_benchmark_analysis(
            json.loads(args.input.read_text(encoding="utf-8"))
        )
        _write_json(payload, args.output_json)
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(payload["markdown"], encoding="utf-8")
        print(
            "analyze-dynamics-benchmark: "
            f"success_rate={payload['overall']['success_rate']:.3f} "
            f"output={args.output_json}"
        )
        return 0
    scenario = _scenario_from_args(args)
    if args.command == "build-scene":
        result = build_pick_place_scene(
            panda_xml=args.panda_xml,
            scenario=scenario,
            output_xml=args.output_xml,
        )
        payload = {"schema_version": 1, "scene": result.to_json_dict(), "scenario": scenario.to_json_dict()}
        _write_json(payload, args.output_json)
        print(f"build-scene: output={result.output_xml}")
        return 0

    explicit_strategy = _strategy_named_for_object(scenario, getattr(args, "grasp_strategy", None))
    if explicit_strategy is None:
        selection = select_grasp_plan(scenario)
        selected_strategy = selection.selected_strategy
        grasp_selection_payload = selection.to_json_dict()
        phases = selection.phases
    else:
        selection = None
        selected_strategy = explicit_strategy
        grasp_selection_payload = _explicit_grasp_selection_payload(explicit_strategy)
        phases = plan_pick_place(scenario, grasp_strategy=explicit_strategy)
    if args.command == "plan":
        payload = {
            "schema_version": 1,
            "scenario": scenario.to_json_dict(),
            "grasp_strategy": _grasp_strategy_payload(scenario, selected_strategy),
            "grasp_selection": grasp_selection_payload,
            "phases": [phase.to_json_dict() for phase in phases],
        }
        _write_json(payload, args.output)
        if args.output is None:
            print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "run":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.panda_xml is not None and explicit_strategy is None:
            selection = select_grasp_plan(
                scenario,
                panda_xml=args.panda_xml,
                output_dir=args.output_dir / "candidate_scenes",
                position_tolerance=args.position_tolerance,
                full_candidate_limit=args.full_candidate_limit,
            )
            selected_strategy = selection.selected_strategy
            grasp_selection_payload = selection.to_json_dict()
            phases = selection.phases
        frames = build_replay_frames(scenario, phases)
        metrics = evaluate_replay(scenario, frames)
        physical_evidence = evaluate_physical_evidence(
            scenario,
            frames,
            position_tolerance=args.position_tolerance,
        )
        payload = {
            "schema_version": 1,
            "scenario": scenario.to_json_dict(),
            "grasp_strategy": _grasp_strategy_payload(scenario, selected_strategy),
            "grasp_selection": grasp_selection_payload,
            "phases": [phase.to_json_dict() for phase in phases],
            "frames": [frame.to_json_dict() for frame in frames],
            "metrics": metrics,
            "physical_evidence": physical_evidence,
        }
        if args.panda_xml is not None:
            scene = build_pick_place_scene(
                panda_xml=args.panda_xml,
                scenario=scenario,
                output_xml=args.output_dir / "scene.xml",
            )
            joint_replay = solve_joint_replay(
                scene.output_xml,
                frames,
                tolerance=args.position_tolerance,
                orientation_tolerance=float(selected_strategy.orientation_tolerance),
            ).to_json_dict()
            payload["scene"] = scene.to_json_dict()
            payload["joint_replay"] = joint_replay
        else:
            joint_replay = None
        payload["execution_success"] = bool(
            run_execution_success(metrics, joint_replay)
            and physical_evidence["success"]
        )
        replay_path = args.output_dir / "replay.json"
        replay_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(payload, sort_keys=True, allow_nan=False))
        else:
            joint_summary = "" if joint_replay is None else f" joint_success={joint_replay['success']}"
            print(
                f"run: success={payload['execution_success']} "
                f"static_success={metrics['success']}{joint_summary} output={replay_path}"
            )
        return 0
    if args.command == "view":
        payload = replay_report(args.xml, args.replay)
        _write_json(payload, args.output)
        if args.interactive:
            launch_viewer(args.xml, args.replay, fps=args.fps, playback_speed=args.playback_speed)
        else:
            print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    if args.command in {"validate-physics", "validate-dynamics"}:
        gate = _dynamics_gate_payload(args)
        payload = validate_physics_replay(
            args.xml,
            args.replay,
            settle_steps=args.settle_steps,
            position_tolerance=args.position_tolerance,
            weld_tracking_tolerance=args.weld_tracking_tolerance,
            frame_substeps=args.frame_substeps,
            max_gripper_penetration=gate["max_gripper_penetration"],
            require_two_sided_grasp=bool(gate["require_two_sided_grasp"]),
        )
        payload["settings"] = {
            "dynamics_gate": gate,
            "max_gripper_penetration": gate["max_gripper_penetration"],
            "require_two_sided_grasp": bool(gate["require_two_sided_grasp"]),
        }
        _write_json(payload, args.output)
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
