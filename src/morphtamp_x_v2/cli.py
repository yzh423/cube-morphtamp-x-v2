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
from .physics_validator import validate_physics_replay
from .planner import plan_pick_place
from .replay import build_replay_frames
from .result_visualization import visualize_results
from .scene_builder import build_pick_place_scene
from .objects import OBJECT_TYPES
from .objects import object_spec
from .protocols import PROTOCOL_TYPES, resolve_protocol_inputs
from .task_health import check_all_tasks
from .tasks import TASK_TYPES, make_scenario, task_spec
from .validator import evaluate_replay
from .viewer import launch_viewer, replay_report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="morphtamp-x-v2")
    sub = root.add_subparsers(dest="command", required=True)

    sub.add_parser("list-objects")
    sub.add_parser("list-tasks")
    sub.add_parser("check-tasks")

    plan = sub.add_parser("plan")
    plan.add_argument("--object", dest="object_type", choices=OBJECT_TYPES, default="cube")
    plan.add_argument("--task", choices=TASK_TYPES, default="tabletop_easy")
    plan.add_argument("--grasp-strategy")
    plan.add_argument("--output", type=Path)

    build = sub.add_parser("build-scene")
    build.add_argument("--panda-xml", type=Path, required=True)
    build.add_argument("--output-xml", type=Path, required=True)
    build.add_argument("--output-json", type=Path)
    build.add_argument("--auto-fit-panda", action="store_true")
    build.add_argument("--object", dest="object_type", choices=OBJECT_TYPES, default="cube")
    build.add_argument("--task", choices=TASK_TYPES, default="tabletop_easy")

    run = sub.add_parser("run")
    run.add_argument("--panda-xml", type=Path)
    run.add_argument("--auto-fit-panda", action="store_true")
    run.add_argument("--object", dest="object_type", choices=OBJECT_TYPES, default="cube")
    run.add_argument("--task", choices=TASK_TYPES, default="tabletop_easy")
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

    validate = sub.add_parser("validate-physics")
    validate.add_argument("--xml", type=Path, required=True)
    validate.add_argument("--replay", type=Path, required=True)
    validate.add_argument("--settle-steps", type=int, default=80)
    validate.add_argument("--position-tolerance", type=float, default=0.035)
    validate.add_argument("--output", type=Path)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--protocol", choices=PROTOCOL_TYPES)
    benchmark.add_argument("--objects", nargs="+", choices=OBJECT_TYPES)
    benchmark.add_argument("--tasks", nargs="+", choices=TASK_TYPES)
    benchmark.add_argument("--panda-xml", type=Path)
    benchmark.add_argument("--auto-fit-panda", action="store_true")
    benchmark.add_argument("--full-candidate-limit", type=int, default=2)
    benchmark.add_argument("--position-tolerance", type=float, default=0.05)
    benchmark.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser("compare-grasps")
    compare.add_argument("--panda-xml", type=Path)
    compare.add_argument("--auto-fit-panda", action="store_true")
    compare.add_argument("--object", dest="object_type", choices=OBJECT_TYPES, default="cube")
    compare.add_argument("--task", choices=TASK_TYPES, default="tabletop_easy")
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
    robust.add_argument("--output", type=Path, required=True)

    failure = sub.add_parser("failure-analysis")
    failure.add_argument("--input", type=Path, required=True)
    failure.add_argument("--output-json", type=Path, required=True)
    failure.add_argument("--output-md", type=Path)

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
    panda_xml: Path | None = None,
    auto_fit_panda: bool = False,
    position_tolerance: float = 0.05,
    full_candidate_limit: int = 2,
) -> dict:
    if panda_xml is not None and auto_fit_panda:
        scenario = auto_fit_panda_scenario(
            panda_xml,
            object_type=object_type,
            task_name=task_name,
        )
    else:
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
        "success": bool(metrics["success"]),
        "failure_reasons": list(metrics["failure_reasons"]),
        "path_length": path_length,
        "lift_height": scenario.lift_height,
        "has_obstacle": scenario.obstacle_center is not None,
        "grasp_strategy": _grasp_strategy_payload(scenario, selection.selected_strategy),
        "grasp_selection": grasp_selection,
        "grasp_selection_summary": summarize_grasp_selection(grasp_selection),
        "metrics": metrics,
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
    if args.command == "benchmark":
        objects, tasks, auto_fit_panda, protocol_payload = resolve_protocol_inputs(
            protocol_name=args.protocol,
            objects=args.objects,
            tasks=args.tasks,
            auto_fit_panda=args.auto_fit_panda,
        )
        results = [
            _run_static_case(
                object_type,
                task_name,
                panda_xml=args.panda_xml,
                auto_fit_panda=auto_fit_panda,
                position_tolerance=args.position_tolerance,
                full_candidate_limit=args.full_candidate_limit,
            )
            for object_type in objects
            for task_name in tasks
        ]
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
            "best": None if best is None else {
                "object_type": best["object_type"],
                "task_name": best["task_name"],
                "path_length": best["path_length"],
            },
            "results": results,
        }
        _write_json(payload, args.output)
        print(
            f"benchmark: success={len(feasible)}/{len(results)} "
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
        payload = {
            "schema_version": 1,
            "scenario": scenario.to_json_dict(),
            "grasp_strategy": _grasp_strategy_payload(scenario, selected_strategy),
            "grasp_selection": grasp_selection_payload,
            "phases": [phase.to_json_dict() for phase in phases],
            "frames": [frame.to_json_dict() for frame in frames],
            "metrics": metrics,
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
        payload["execution_success"] = run_execution_success(metrics, joint_replay)
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
    if args.command == "validate-physics":
        payload = validate_physics_replay(
            args.xml,
            args.replay,
            settle_steps=args.settle_steps,
            position_tolerance=args.position_tolerance,
        )
        _write_json(payload, args.output)
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
