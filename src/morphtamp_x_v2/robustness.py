from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any

import numpy as np

from .cli import _run_static_case
from .models import PickPlaceScenario
from .tasks import make_scenario


def _noise2(rng: np.random.Generator, scale: float) -> tuple[float, float]:
    if scale <= 0.0:
        return (0.0, 0.0)
    return tuple(float(value) for value in rng.uniform(-scale, scale, size=2))


def perturb_scenario(
    scenario: PickPlaceScenario,
    *,
    seed: int,
    position_noise: float = 0.01,
    obstacle_noise: float = 0.01,
) -> PickPlaceScenario:
    rng = np.random.default_rng(int(seed))
    start_dx, start_dy = _noise2(rng, position_noise)
    target_dx, target_dy = _noise2(rng, position_noise)
    obstacle_center = scenario.obstacle_center
    if obstacle_center is not None:
        obstacle_dx, obstacle_dy = _noise2(rng, obstacle_noise)
        obstacle_center = (
            float(obstacle_center[0] + obstacle_dx),
            float(obstacle_center[1] + obstacle_dy),
            float(obstacle_center[2]),
        )
    return replace(
        scenario,
        task_name=f"{scenario.task_name}_perturbed",
        cube_start=(
            float(scenario.cube_start[0] + start_dx),
            float(scenario.cube_start[1] + start_dy),
            float(scenario.cube_start[2]),
        ),
        place_target=(
            float(scenario.place_target[0] + target_dx),
            float(scenario.place_target[1] + target_dy),
            float(scenario.place_target[2]),
        ),
        obstacle_center=obstacle_center,
    )


def _evaluate_scenario_direct(
    scenario: PickPlaceScenario,
    *,
    object_type: str,
    task_name: str,
) -> dict[str, Any]:
    from .grasp_planner import select_grasp_plan, path_length_for_phases
    from .replay import build_replay_frames
    from .validator import evaluate_replay

    selection = select_grasp_plan(scenario)
    phases = selection.phases
    frames = build_replay_frames(scenario, phases)
    metrics = evaluate_replay(scenario, frames)
    return {
        "object_type": object_type,
        "task_name": task_name,
        "success": bool(metrics["success"]),
        "failure_reasons": list(metrics["failure_reasons"]),
        "path_length": path_length_for_phases(phases),
        "metrics": metrics,
        "scenario": scenario.to_json_dict(),
        "grasp_selection": selection.to_json_dict(),
    }


def run_robustness_benchmark(
    *,
    objects: tuple[str, ...],
    tasks: tuple[str, ...],
    trials: int,
    seed: int,
    position_noise: float = 0.01,
    obstacle_noise: float = 0.01,
    panda_xml: Any | None = None,
    auto_fit_panda: bool = False,
    position_tolerance: float = 0.05,
    full_candidate_limit: int = 2,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for object_type in objects:
        for task_name in tasks:
            nominal = make_scenario(object_type=object_type, task_name=task_name)
            stable_offset = int(
                hashlib.sha256(f"{object_type}:{task_name}".encode("utf-8")).hexdigest()[:8],
                16,
            )
            for trial in range(int(trials)):
                trial_seed = int(seed) + 1009 * trial + stable_offset
                perturbed = perturb_scenario(
                    nominal,
                    seed=trial_seed,
                    position_noise=position_noise,
                    obstacle_noise=obstacle_noise,
                )
                if panda_xml is None:
                    row = _evaluate_scenario_direct(
                        perturbed,
                        object_type=object_type,
                        task_name=task_name,
                    )
                else:
                    # Preserve the existing full Panda verification path for
                    # commanded cases; the nominal task name is retained so
                    # auto-fit and task-library semantics stay explicit.
                    row = _run_static_case(
                        object_type,
                        task_name,
                        panda_xml=panda_xml,
                        auto_fit_panda=auto_fit_panda,
                        position_tolerance=position_tolerance,
                        full_candidate_limit=full_candidate_limit,
                    )
                    row["scenario"] = perturbed.to_json_dict()
                row["trial"] = trial
                row["trial_seed"] = trial_seed
                row["perturbation"] = {
                    "position_noise": position_noise,
                    "obstacle_noise": obstacle_noise,
                }
                results.append(row)
    successes = [row for row in results if row["success"] and (row.get("joint_metrics") is None or row["joint_metrics"]["success"])]
    max_error = max(
        (
            float(row.get("joint_metrics", {}).get("max_position_error", row.get("metrics", {}).get("max_position_error", 0.0)))
            for row in results
        ),
        default=0.0,
    )
    return {
        "schema_version": 1,
        "benchmark_type": "robustness",
        "objects": list(objects),
        "tasks": list(tasks),
        "trials": int(trials),
        "seed": int(seed),
        "total_runs": len(results),
        "successful_runs": len(successes),
        "summary": {
            "success_rate": 0.0 if not results else len(successes) / len(results),
            "max_position_error": max_error,
        },
        "results": results,
    }
