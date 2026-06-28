from __future__ import annotations

from collections import defaultdict
from typing import Any


def _designs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("design_summaries", ()))


def _feasible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if float(row.get("success_rate", 0.0)) >= 1.0]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = _feasible(rows)
    front: list[dict[str, Any]] = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            better_or_equal = (
                float(other["morphology_cost"]) <= float(row["morphology_cost"])
                and float(other["mean_path_length"]) <= float(row["mean_path_length"])
                and float(other["minimum_reach_margin"]) >= float(row["minimum_reach_margin"])
            )
            strictly_better = (
                float(other["morphology_cost"]) < float(row["morphology_cost"])
                or float(other["mean_path_length"]) < float(row["mean_path_length"])
                or float(other["minimum_reach_margin"]) > float(row["minimum_reach_margin"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda row: (float(row["morphology_cost"]), float(row["mean_path_length"])))


def build_pareto_report(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _designs(payload)
    feasible = _feasible(rows)
    best_objective = min(feasible, key=lambda row: float(row.get("objective", float("inf"))), default=None)
    lowest_cost = min(feasible, key=lambda row: float(row["morphology_cost"]), default=None)
    lowest_path = min(feasible, key=lambda row: float(row["mean_path_length"]), default=None)
    highest_margin = max(feasible, key=lambda row: float(row["minimum_reach_margin"]), default=None)
    front = _pareto_front(rows)

    lines = [
        "# Pareto Morphology Report",
        "",
        f"Total designs: {len(rows)}",
        f"Feasible designs: {len(feasible)}",
        "",
        "## Key designs",
        "",
        "| Role | Design | Cost | Mean path | Min reach margin | Objective |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, row in (
        ("best objective", best_objective),
        ("lowest feasible cost", lowest_cost),
        ("lowest path", lowest_path),
        ("highest reach margin", highest_margin),
    ):
        if row is None:
            lines.append(f"| {label} | none | - | - | - | - |")
        else:
            lines.append(
                f"| {label} | `{row['arm_design']}` | {_fmt(row['morphology_cost'])} | "
                f"{_fmt(row['mean_path_length'])} | {_fmt(row['minimum_reach_margin'])} | "
                f"{_fmt(row.get('objective', 0.0))} |"
            )
    lines += [
        "",
        "## Pareto front",
        "",
        "| Design | Cost | Mean path | Min reach margin |",
        "|---|---:|---:|---:|",
    ]
    if not front:
        lines.append("| none | - | - | - |")
    else:
        for row in front:
            lines.append(
                f"| `{row['arm_design']}` | {_fmt(row['morphology_cost'])} | "
                f"{_fmt(row['mean_path_length'])} | {_fmt(row['minimum_reach_margin'])} |"
            )
    lines.append("")
    return {
        "schema_version": 1,
        "best_objective": best_objective,
        "lowest_cost_feasible": lowest_cost,
        "lowest_path_feasible": lowest_path,
        "highest_margin_feasible": highest_margin,
        "pareto_front": front,
        "markdown": "\n".join(lines),
    }


def segment_importance(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _designs(payload)
    segment_stats: dict[str, dict[str, dict[str, float]]] = {}
    for segment in ("upper_scale", "forearm_scale", "wrist_scale"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get("design", {}).get(segment)
            groups[f"{float(value):.3f}"].append(row)
        segment_stats[segment] = {}
        for value, grouped in sorted(groups.items()):
            successes = sum(1 for row in grouped if float(row.get("success_rate", 0.0)) >= 1.0)
            mean_success_rate = sum(float(row.get("success_rate", 0.0)) for row in grouped) / len(grouped)
            segment_stats[segment][value] = {
                "runs": len(grouped),
                "fully_feasible_designs": successes,
                "full_feasibility_rate": 0.0 if not grouped else successes / len(grouped),
                "success_rate": mean_success_rate,
                "mean_success_rate": mean_success_rate,
                "mean_cost": sum(float(row["morphology_cost"]) for row in grouped) / len(grouped),
                "mean_reach_margin": sum(float(row["minimum_reach_margin"]) for row in grouped) / len(grouped),
            }

    lines = ["# Segment importance", ""]
    for segment, stats in segment_stats.items():
        lines += [
            f"## `{segment}`",
            "",
            "| Value | Success rate | Mean cost | Mean reach margin |",
            "|---:|---:|---:|---:|",
        ]
        for value, row in stats.items():
            lines.append(
                f"| {value} | {_fmt(row['success_rate'])} | "
                f"{_fmt(row['mean_cost'])} | {_fmt(row['mean_reach_margin'])} |"
            )
        lines.append("")
    return {"schema_version": 1, "segments": segment_stats, "markdown": "\n".join(lines)}


def build_design_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
    pareto = build_pareto_report(payload)
    recommended = pareto["best_objective"] or pareto["lowest_cost_feasible"]
    lines = ["# Design Recommendation", ""]
    if recommended is None:
        lines += [
            "No feasible design was found under the current constraints.",
            "",
            "## Risks",
            "",
            "- Relaxing constraints may produce a candidate, but the resulting claim should be labeled infeasible under the requested protocol.",
        ]
    else:
        design = recommended["design"]
        lines += [
            f"Recommended design: `{recommended['arm_design']}`",
            "",
            "| Parameter | Value |",
            "|---|---:|",
            f"| upper_scale | {_fmt(design['upper_scale'])} |",
            f"| forearm_scale | {_fmt(design['forearm_scale'])} |",
            f"| wrist_scale | {_fmt(design['wrist_scale'])} |",
            f"| base_shift | `{design.get('base_shift')}` |",
            f"| morphology_cost | {_fmt(recommended['morphology_cost'])} |",
            f"| minimum_reach_margin | {_fmt(recommended['minimum_reach_margin'])} |",
            f"| mean_path_length | {_fmt(recommended['mean_path_length'])} |",
            "",
            "## Why this design",
            "",
            "- It is feasible under the selected optimization constraints.",
            "- It has the best objective among feasible candidates.",
            "- It provides a bounded cost/reachability trade-off rather than minimizing length alone.",
            "",
            "## Risks",
            "",
            "- The recommendation is tied to the current object/task distribution.",
            "- The morphology cost is an equivalent proxy, not a manufactured Panda redesign.",
            "- The evidence is simulation-stage and does not prove force-controlled real-world grasping.",
        ]
    markdown = "\n".join(lines) + "\n"
    return {
        "schema_version": 1,
        "recommended_design": recommended,
        "markdown": markdown,
    }


def build_design_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    pareto = build_pareto_report(payload)
    importance = segment_importance(payload)
    recommendation = build_design_recommendation(payload)
    markdown = "\n\n".join(
        [
            pareto["markdown"],
            importance["markdown"],
            recommendation["markdown"],
        ]
    )
    return {
        "schema_version": 1,
        "pareto": pareto,
        "segment_importance": importance,
        "recommendation": recommendation,
        "markdown": markdown,
    }
