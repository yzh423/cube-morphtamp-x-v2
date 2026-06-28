from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
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


def _cache_row_key(object_type: str, task_name: str, trial: int) -> str:
    return f"{object_type}\u241f{task_name}\u241f{int(trial)}"


def _cache_config(
    *,
    objects: tuple[str, ...],
    tasks: tuple[str, ...],
    trials: int,
    seed: int,
    position_noise: float,
    obstacle_noise: float,
    panda_xml: Any | None,
    auto_fit_panda: bool,
    position_tolerance: float,
    full_candidate_limit: int,
) -> dict[str, Any]:
    return {
        "objects": list(objects),
        "tasks": list(tasks),
        "trials": int(trials),
        "seed": int(seed),
        "position_noise": float(position_noise),
        "obstacle_noise": float(obstacle_noise),
        "panda_xml": None if panda_xml is None else str(Path(panda_xml).expanduser()),
        "auto_fit_panda": bool(auto_fit_panda),
        "position_tolerance": float(position_tolerance),
        "full_candidate_limit": int(full_candidate_limit),
    }


def _load_result_cache(
    path: Path | None,
    *,
    expected_config: dict[str, Any],
    reuse: bool,
) -> dict[str, dict[str, Any]]:
    if path is None or not reuse or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _cache_config_is_compatible(payload.get("config"), expected_config):
        return {}
    rows = {}
    for row in payload.get("results", ()):
        rows[_cache_row_key(str(row["object_type"]), str(row["task_name"]), int(row["trial"]))] = row
    return rows


def _cache_config_is_compatible(
    cached_config: object,
    expected_config: dict[str, Any],
) -> bool:
    """Return whether cached robustness rows can be reused.

    The number of requested trials is intentionally excluded from the strict
    comparison. A cache from 5 trials should remain useful when extending the
    same experiment to 10 trials, and a 10-trial cache should also support a
    smaller diagnostic rerun. Individual row keys still include the trial index,
    so incompatible or missing trial rows are never fabricated.
    """
    if not isinstance(cached_config, dict):
        return False
    ignored = {"trials"}
    return {
        key: value for key, value in cached_config.items() if key not in ignored
    } == {
        key: value for key, value in expected_config.items() if key not in ignored
    }


def _write_result_cache(
    path: Path | None,
    *,
    config: dict[str, Any],
    rows_by_key: dict[str, dict[str, Any]],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "cache_type": "robustness_benchmark_results",
        "config": config,
        "results": [rows_by_key[key] for key in sorted(rows_by_key)],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _row_success(row: dict[str, Any]) -> bool:
    joint_metrics = row.get("joint_metrics")
    return bool(row.get("success", False)) and (
        joint_metrics is None or bool(joint_metrics.get("success", False))
    )


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _position_error(row: dict[str, Any]) -> float | None:
    joint_metrics = row.get("joint_metrics") or {}
    metrics = row.get("metrics") or {}
    for key, container in (
        ("max_position_error", joint_metrics),
        ("max_position_error", metrics),
        ("release_error", metrics),
    ):
        value = _finite_float(container.get(key))
        if value is not None:
            return value
    return None


def _joint_metric(row: dict[str, Any], key: str) -> float | None:
    return _finite_float((row.get("joint_metrics") or {}).get(key))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {
        name: {
            "runs": len(group_rows),
            "successful_runs": sum(1 for row in group_rows if _row_success(row)),
            "success_rate": 0.0
            if not group_rows
            else sum(1 for row in group_rows if _row_success(row)) / len(group_rows),
        }
        for name, group_rows in sorted(grouped.items())
    }


def summarize_robustness_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in results if _row_success(row)]
    position_errors = [
        value for row in results if (value := _position_error(row)) is not None
    ]
    sigma_values = [
        value for row in results if (value := _joint_metric(row, "min_sigma")) is not None
    ]
    condition_values = [
        value
        for row in results
        if (value := _joint_metric(row, "max_condition_number")) is not None
    ]
    path_lengths = [
        value
        for row in results
        if (value := _finite_float(row.get("path_length"))) is not None
    ]
    failed_runs = [
        {
            "object_type": row.get("object_type"),
            "task_name": row.get("task_name"),
            "trial": row.get("trial"),
            "failure_reasons": list(row.get("failure_reasons", ())),
            "joint_failure_reasons": list(
                (row.get("joint_metrics") or {}).get("failure_reasons", ())
            ),
        }
        for row in results
        if not _row_success(row)
    ]
    return {
        "success_rate": 0.0 if not results else len(successes) / len(results),
        "max_position_error": None if not position_errors else max(position_errors),
        "mean_position_error": _mean(position_errors),
        "min_sigma": None if not sigma_values else min(sigma_values),
        "mean_sigma": _mean(sigma_values),
        "max_condition_number": None if not condition_values else max(condition_values),
        "mean_condition_number": _mean(condition_values),
        "mean_path_length": _mean(path_lengths),
        "by_object": _group_summary(results, "object_type"),
        "by_task": _group_summary(results, "task_name"),
        "failed_runs": failed_runs,
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
    results_cache: str | Path | None = None,
    reuse_results_cache: bool = True,
) -> dict[str, Any]:
    objects = tuple(objects)
    tasks = tuple(tasks)
    cache_path = None if results_cache is None else Path(results_cache)
    config = _cache_config(
        objects=objects,
        tasks=tasks,
        trials=trials,
        seed=seed,
        position_noise=position_noise,
        obstacle_noise=obstacle_noise,
        panda_xml=panda_xml,
        auto_fit_panda=auto_fit_panda,
        position_tolerance=position_tolerance,
        full_candidate_limit=full_candidate_limit,
    )
    cached_rows = _load_result_cache(
        cache_path,
        expected_config=config,
        reuse=reuse_results_cache,
    )
    hits = 0
    misses = 0
    results: list[dict[str, Any]] = []
    for object_type in objects:
        for task_name in tasks:
            nominal = make_scenario(object_type=object_type, task_name=task_name)
            stable_offset = int(
                hashlib.sha256(f"{object_type}:{task_name}".encode("utf-8")).hexdigest()[:8],
                16,
            )
            for trial in range(int(trials)):
                cache_key = _cache_row_key(object_type, task_name, trial)
                if cache_key in cached_rows:
                    hits += 1
                    results.append(cached_rows[cache_key])
                    continue
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
                        scenario=perturbed,
                        panda_xml=panda_xml,
                        auto_fit_panda=auto_fit_panda,
                        position_tolerance=position_tolerance,
                        full_candidate_limit=full_candidate_limit,
                    )
                row["trial"] = trial
                row["trial_seed"] = trial_seed
                row["perturbation"] = {
                    "position_noise": position_noise,
                    "obstacle_noise": obstacle_noise,
                }
                cached_rows[cache_key] = row
                results.append(row)
                misses += 1
                _write_result_cache(cache_path, config=config, rows_by_key=cached_rows)
    summary = summarize_robustness_results(results)
    return {
        "schema_version": 1,
        "benchmark_type": "robustness",
        "objects": list(objects),
        "tasks": list(tasks),
        "trials": int(trials),
        "seed": int(seed),
        "total_runs": len(results),
        "successful_runs": sum(1 for row in results if _row_success(row)),
        "summary": summary,
        "result_cache": {
            "enabled": cache_path is not None,
            "path": None if cache_path is None else str(cache_path),
            "reuse": bool(reuse_results_cache),
            "hits": hits,
            "misses": misses,
        },
        "results": results,
    }
