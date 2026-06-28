from __future__ import annotations

from typing import Any, Iterable

from .cli import _run_static_case, _workspace_demand
from .morphology import ArmDesign, pareto_designs


def generate_scale_grid(
    *,
    upper_scales: Iterable[float],
    forearm_scales: Iterable[float],
    wrist_scales: Iterable[float],
    base_x_values: Iterable[float],
    base_y_values: Iterable[float] = (0.0,),
) -> Iterable[ArmDesign]:
    for upper in upper_scales:
        for forearm in forearm_scales:
            for wrist in wrist_scales:
                for base_x in base_x_values:
                    for base_y in base_y_values:
                        yield ArmDesign(
                            name=(
                                f"opt_u{upper:.3f}_f{forearm:.3f}_w{wrist:.3f}"
                                f"_bx{base_x:.3f}"
                                + ("" if abs(float(base_y)) < 1e-12 else f"_by{base_y:.3f}")
                            ),
                            upper_scale=float(upper),
                            forearm_scale=float(forearm),
                            wrist_scale=float(wrist),
                            base_shift=(float(base_x), float(base_y), 0.0),
                            description="continuous grid-search candidate",
                        )


def _evaluate_design(
    design: ArmDesign,
    base_rows: list[dict[str, Any]],
    *,
    minimum_reach_margin: float,
    path_cost_weight: float,
    minimum_sigma: float | None,
    maximum_condition_number: float | None,
) -> dict[str, Any]:
    rows = []
    sigma_values: list[float] = []
    condition_values: list[float] = []
    for row in base_rows:
        demand = _workspace_demand(row)
        reach_margin = design.reach_proxy - demand
        joint_metrics = row.get("joint_metrics") or {}
        sigma = joint_metrics.get("min_sigma")
        condition = joint_metrics.get("max_condition_number")
        if sigma is not None:
            sigma_values.append(float(sigma))
        if condition is not None:
            condition_values.append(float(condition))
        sigma_passed = minimum_sigma is None or sigma is None or float(sigma) >= minimum_sigma
        condition_passed = (
            maximum_condition_number is None
            or condition is None
            or float(condition) <= maximum_condition_number
        )
        singularity_passed = sigma_passed and condition_passed
        base_success = bool(row["success"]) and (
            row.get("joint_metrics") is None or bool(row["joint_metrics"]["success"])
        )
        success = base_success and reach_margin >= minimum_reach_margin and singularity_passed
        failure_reasons = list(row.get("failure_reasons", ()))
        if not success and reach_margin < minimum_reach_margin:
            failure_reasons.append(f"reach_margin:{reach_margin:.6f}")
        if not singularity_passed:
            if minimum_sigma is not None and sigma is not None and float(sigma) < minimum_sigma:
                failure_reasons.append(f"sigma_min:{float(sigma):.6f}")
            if (
                maximum_condition_number is not None
                and condition is not None
                and float(condition) > maximum_condition_number
            ):
                failure_reasons.append(f"condition_number:{float(condition):.6f}")
        rows.append(
            {
                "object_type": row["object_type"],
                "task_name": row["task_name"],
                "success": success,
                "path_length": row["path_length"],
                "workspace_demand": demand,
                "reach_margin": reach_margin,
                "min_sigma": sigma,
                "max_condition_number": condition,
                "singularity_passed": singularity_passed,
                "failure_reasons": failure_reasons,
            }
        )
    successes = [row for row in rows if row["success"]]
    success_rate = 0.0 if not rows else len(successes) / len(rows)
    mean_path = 0.0 if not successes else sum(float(row["path_length"]) for row in successes) / len(successes)
    objective = design.morphology_cost + path_cost_weight * mean_path
    if success_rate < 1.0:
        objective += 1000.0 * (1.0 - success_rate)
    return {
        "arm_design": design.name,
        "design": design.to_json_dict(),
        "runs": len(rows),
        "successes": len(successes),
        "success_rate": success_rate,
        "morphology_cost": design.morphology_cost,
        "objective": objective,
        "mean_path_length": mean_path,
        "minimum_reach_margin": min(float(row["reach_margin"]) for row in rows) if rows else 0.0,
        "minimum_sigma": None if not sigma_values else min(sigma_values),
        "maximum_condition_number": None if not condition_values else max(condition_values),
        "singularity_passed": all(bool(row["singularity_passed"]) for row in rows),
        "results": rows,
    }


def optimize_morphology(
    *,
    objects: tuple[str, ...],
    tasks: tuple[str, ...],
    upper_scales: Iterable[float],
    forearm_scales: Iterable[float],
    wrist_scales: Iterable[float],
    base_x_values: Iterable[float],
    base_y_values: Iterable[float] = (0.0,),
    panda_xml: Any | None = None,
    auto_fit_panda: bool = False,
    position_tolerance: float = 0.05,
    full_candidate_limit: int = 2,
    minimum_reach_margin: float = 0.03,
    path_cost_weight: float = 0.02,
    minimum_sigma: float | None = None,
    maximum_condition_number: float | None = None,
) -> dict[str, Any]:
    base_rows = [
        _run_static_case(
            object_type,
            task_name,
            panda_xml=panda_xml,
            auto_fit_panda=auto_fit_panda,
            position_tolerance=position_tolerance,
            full_candidate_limit=full_candidate_limit,
        )
        for object_type in objects
        for task_name in tasks
    ]
    candidates = list(
        generate_scale_grid(
            upper_scales=upper_scales,
            forearm_scales=forearm_scales,
            wrist_scales=wrist_scales,
            base_x_values=base_x_values,
            base_y_values=base_y_values,
        )
    )
    summaries = [
        _evaluate_design(
            design,
            base_rows,
            minimum_reach_margin=minimum_reach_margin,
            path_cost_weight=path_cost_weight,
            minimum_sigma=minimum_sigma,
            maximum_condition_number=maximum_condition_number,
        )
        for design in candidates
    ]
    feasible = [row for row in summaries if row["success_rate"] >= 1.0]
    best = min(feasible, key=lambda row: (row["objective"], row["morphology_cost"])) if feasible else None
    return {
        "schema_version": 1,
        "optimization_type": "continuous_grid_search",
        "objects": list(objects),
        "tasks": list(tasks),
        "total_candidates": len(summaries),
        "feasible_candidates": len(feasible),
        "minimum_reach_margin": minimum_reach_margin,
        "singularity_constraints": {
            "minimum_sigma": minimum_sigma,
            "maximum_condition_number": maximum_condition_number,
        },
        "path_cost_weight": path_cost_weight,
        "best_design": best,
        "pareto_designs": pareto_designs(summaries),
        "design_summaries": summaries,
        "base_case_results": base_rows,
    }
