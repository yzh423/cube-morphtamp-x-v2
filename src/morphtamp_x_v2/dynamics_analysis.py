from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _rate(successes: int, runs: int) -> float:
    return 0.0 if runs == 0 else successes / runs


def _row_penetration(row: dict[str, Any]) -> float | None:
    try:
        return float(
            row["dynamics_evidence"]["checks"]["gripper_penetration"][
                "gripper_object_max_penetration"
            ]
        )
    except (KeyError, TypeError, ValueError):
        return None


def _row_contact_sides(row: dict[str, Any]) -> tuple[str, ...]:
    try:
        sides = row["dynamics_evidence"]["checks"]["grasp_contact"][
            "gripper_object_contact_sides"
        ]
    except (KeyError, TypeError):
        return ()
    return tuple(str(side) for side in sides)


def _table(rows: list[tuple[str, int, int]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "successes": successes,
            "runs": runs,
            "success_rate": _rate(successes, runs),
        }
        for name, successes, runs in rows
    ]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "## Dynamics benchmark analysis",
        "",
        f"- Runs: {payload['overall']['runs']}",
        f"- Successes: {payload['overall']['successes']}",
        f"- Success rate: {payload['overall']['success_rate']:.3f}",
        "",
        "### By object",
        "",
        "| Object | Successes | Runs | Success rate |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["by_object_rows"]:
        lines.append(
            f"| `{row['name']}` | {row['successes']} | {row['runs']} | "
            f"{row['success_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### By task",
            "",
            "| Task | Successes | Runs | Success rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["by_task_rows"]:
        lines.append(
            f"| `{row['name']}` | {row['successes']} | {row['runs']} | "
            f"{row['success_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Failure counts",
            "",
            "| Failure reason | Count |",
            "|---|---:|",
        ]
    )
    if payload["failure_counts"]:
        for reason, count in payload["failure_counts"].items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("| none | 0 |")
    max_pen = payload["max_gripper_penetration"]
    lines.extend(
        [
            "",
            "### Contact quality",
            "",
            (
                "- Max gripper penetration: "
                f"{max_pen['value']:.6g} m "
                f"({max_pen['object_type']} / {max_pen['task_name']})"
                if max_pen["value"] is not None
                else "- Max gripper penetration: unavailable"
            ),
            f"- Two-sided contact successes: {payload['contact_quality']['two_sided_passes']}",
            "",
            "Dominant penetration side counts:",
            "",
            "| Side | Count |",
            "|---|---:|",
        ]
    )
    if payload["dominant_penetration_side_counts"]:
        for side, count in payload["dominant_penetration_side_counts"].items():
            lines.append(f"| `{side}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "### Selected strategy counts",
            "",
            "| Strategy | Count |",
            "|---|---:|",
        ]
    )
    if payload["selected_strategy_counts"]:
        for strategy, count in payload["selected_strategy_counts"].items():
            lines.append(f"| `{strategy}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.append("")
    return "\n".join(lines)


def build_dynamics_benchmark_analysis(benchmark: dict[str, Any]) -> dict[str, Any]:
    results = list(benchmark.get("results") or [])
    successes = sum(1 for row in results if bool(row.get("success")))
    by_object: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    failure_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    contact_sides: Counter[str] = Counter()
    dominant_penetration_sides: Counter[str] = Counter()
    two_sided_passes = 0
    max_penetration: dict[str, Any] = {
        "value": None,
        "object_type": None,
        "task_name": None,
    }

    for row in results:
        object_type = str(row.get("object_type", "unknown"))
        task_name = str(row.get("task_name", "unknown"))
        success = bool(row.get("success"))
        by_object[object_type][1] += 1
        by_task[task_name][1] += 1
        if success:
            by_object[object_type][0] += 1
            by_task[task_name][0] += 1
        for reason in row.get("failure_reasons") or ():
            failure_counts[str(reason)] += 1
        strategy_name = (row.get("grasp_strategy") or {}).get("name")
        if strategy_name:
            strategy_counts[str(strategy_name)] += 1
        for side in _row_contact_sides(row):
            contact_sides[side] += 1
        dominant_side = (
            row.get("dynamics_evidence", {})
            .get("checks", {})
            .get("gripper_penetration", {})
            .get("dominant_penetration_side")
        )
        if dominant_side:
            dominant_penetration_sides[str(dominant_side)] += 1
        two_sided = (
            row.get("dynamics_evidence", {})
            .get("checks", {})
            .get("two_sided_grasp_contact", {})
            .get("passed")
        )
        if two_sided:
            two_sided_passes += 1
        penetration = _row_penetration(row)
        if penetration is not None and (
            max_penetration["value"] is None
            or penetration > float(max_penetration["value"])
        ):
            max_penetration = {
                "value": penetration,
                "object_type": object_type,
                "task_name": task_name,
            }

    by_object_rows = _table(
        (name, values[0], values[1])
        for name, values in sorted(by_object.items())
    )
    by_task_rows = _table(
        (name, values[0], values[1])
        for name, values in sorted(by_task.items())
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_schema_version": benchmark.get("schema_version"),
        "overall": {
            "runs": len(results),
            "successes": successes,
            "failures": len(results) - successes,
            "success_rate": _rate(successes, len(results)),
        },
        "by_object": {row["name"]: row for row in by_object_rows},
        "by_object_rows": by_object_rows,
        "by_task": {row["name"]: row for row in by_task_rows},
        "by_task_rows": by_task_rows,
        "failure_counts": dict(sorted(failure_counts.items())),
        "selected_strategy_counts": dict(sorted(strategy_counts.items())),
        "contact_side_counts": dict(sorted(contact_sides.items())),
        "dominant_penetration_side_counts": dict(
            sorted(dominant_penetration_sides.items())
        ),
        "contact_quality": {
            "two_sided_passes": two_sided_passes,
            "two_sided_rate": _rate(two_sided_passes, len(results)),
        },
        "max_gripper_penetration": max_penetration,
    }
    payload["markdown"] = _markdown(payload)
    return payload
