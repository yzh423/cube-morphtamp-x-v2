from __future__ import annotations

from collections import Counter
from typing import Any


def _reason_code(reason: str) -> str:
    text = str(reason)
    if ":" in text:
        return text.split(":", 1)[0]
    return text


def _row_success(row: dict[str, Any]) -> bool:
    joint_metrics = row.get("joint_metrics")
    return bool(row.get("success", False)) and (
        joint_metrics is None or bool(joint_metrics.get("success", False))
    )


def _failure_reasons(row: dict[str, Any]) -> list[str]:
    reasons = [f"task:{reason}" for reason in row.get("failure_reasons", ())]
    joint_metrics = row.get("joint_metrics")
    if isinstance(joint_metrics, dict) and not bool(joint_metrics.get("success", True)):
        joint_reasons = list(joint_metrics.get("failure_reasons", ()))
        if joint_reasons:
            reasons.extend(f"joint:{reason}" for reason in joint_reasons)
        else:
            reasons.append("joint:unspecified")
    return reasons


def _counter_table(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"## {title}", "", "| Item | Count |", "|---|---:|"]
    if not counter:
        lines.append("| none | 0 |")
    else:
        for key, count in counter.most_common():
            lines.append(f"| `{key}` | {count} |")
    lines.append("")
    return lines


def build_failure_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("results", ()))
    failures = [row for row in rows if not _row_success(row)]
    by_design: Counter[str] = Counter()
    by_task: Counter[str] = Counter()
    by_object: Counter[str] = Counter()
    reason_codes: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for row in failures:
        by_design[str(row.get("arm_design", "unspecified"))] += 1
        by_task[str(row.get("task_name", "unknown_task"))] += 1
        by_object[str(row.get("object_type", "unknown_object"))] += 1
        reasons = _failure_reasons(row)
        if not reasons:
            reason_codes["unspecified"] += 1
        for reason in reasons:
            reason_codes[_reason_code(str(reason))] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "arm_design": row.get("arm_design"),
                    "object_type": row.get("object_type"),
                    "task_name": row.get("task_name"),
                    "failure_reasons": reasons,
                }
            )

    lines = [
        "# Failure Analysis",
        "",
        f"Evaluated runs: {len(rows)}",
        f"Failed runs: {len(failures)}",
        "",
    ]
    lines += _counter_table("Failures by design", by_design)
    lines += _counter_table("Failures by task", by_task)
    lines += _counter_table("Failures by object", by_object)
    lines += _counter_table("Failure reason codes", reason_codes)
    lines += [
        "## Representative failures",
        "",
        "| Design | Object | Task | Reasons |",
        "|---|---|---|---|",
    ]
    if not examples:
        lines.append("| none | none | none | none |")
    else:
        for row in examples:
            reasons = ", ".join(f"`{reason}`" for reason in row["failure_reasons"]) or "`unspecified`"
            lines.append(
                f"| `{row.get('arm_design') or 'unspecified'}` | "
                f"`{row.get('object_type')}` | `{row.get('task_name')}` | {reasons} |"
            )
    lines.append("")

    return {
        "schema_version": 1,
        "evaluated_runs": len(rows),
        "failed_runs": len(failures),
        "by_design": dict(by_design),
        "by_task": dict(by_task),
        "by_object": dict(by_object),
        "reason_codes": dict(reason_codes),
        "representative_failures": examples,
        "markdown": "\n".join(lines),
    }
