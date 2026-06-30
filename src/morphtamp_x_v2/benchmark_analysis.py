from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TASK_SPLITS: dict[str, tuple[str, ...]] = {
    "development": (
        "tabletop_easy",
        "over_barrier",
    ),
    "main": (
        "shelf_place",
        "shelf_pick",
        "narrow_slot",
        "folded_transfer",
        "far_corner",
        "around_wall",
        "over_barrier",
    ),
    "heldout": (
        "compound_shelf_barrier",
        "diagonal_reach_around",
        "under_bridge",
    ),
}


@dataclass(frozen=True)
class GroupStats:
    runs: int
    successes: int
    success_rate: float
    mean_path_length: float | None
    max_position_error: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "mean_path_length": self.mean_path_length,
            "max_position_error": self.max_position_error,
        }


def _row_success(row: dict[str, Any]) -> bool:
    joint_metrics = row.get("joint_metrics")
    return bool(row.get("success")) and (joint_metrics is None or bool(joint_metrics.get("success")))


def _position_error(row: dict[str, Any]) -> float | None:
    joint_metrics = row.get("joint_metrics")
    if not isinstance(joint_metrics, dict):
        return None
    value = joint_metrics.get("max_position_error")
    return None if value is None else float(value)


def _failure_code(reason: Any) -> str:
    parts = str(reason).split(":")
    if len(parts) >= 2:
        return parts[1]
    return str(reason)


def _normalize_failure_code(reason: Any) -> str:
    text = str(reason)
    parts = text.split(":")
    if len(parts) >= 2:
        return parts[1]
    return text.replace(" ", "_")


def summarize_grasp_selection(selection: dict[str, Any] | None) -> dict[str, Any]:
    selection = selection if isinstance(selection, dict) else {}
    coarse = list(selection.get("coarse_candidate_evaluations") or ())
    full = list(selection.get("candidate_evaluations") or ())
    full_names = {str(row.get("strategy_name")) for row in full if isinstance(row, dict)}
    code_counts: dict[str, int] = {}
    for row in coarse:
        if not isinstance(row, dict):
            continue
        for reason in row.get("failure_reasons") or ():
            code = _failure_code(reason)
            code_counts[code] = code_counts.get(code, 0) + 1
    return {
        "selected_strategy": selection.get("selected_strategy"),
        "coarse_candidates": len(coarse),
        "full_candidates": len(full),
        "coarse_pruned_candidates": max(0, len(coarse) - len(full_names)),
        "coarse_successes": sum(1 for row in coarse if isinstance(row, dict) and bool(row.get("success"))),
        "full_successes": sum(1 for row in full if isinstance(row, dict) and bool(row.get("success"))),
        "rejection_code_counts": dict(sorted(code_counts.items())),
    }


def _stats(rows: Iterable[dict[str, Any]]) -> GroupStats:
    items = list(rows)
    successes = [row for row in items if _row_success(row)]
    path_lengths = [float(row["path_length"]) for row in successes if row.get("path_length") is not None]
    errors = [error for row in items if (error := _position_error(row)) is not None]
    runs = len(items)
    return GroupStats(
        runs=runs,
        successes=len(successes),
        success_rate=0.0 if runs == 0 else len(successes) / runs,
        mean_path_length=None if not path_lengths else sum(path_lengths) / len(path_lengths),
        max_position_error=None if not errors else max(errors),
    )


def _aggregate_grasp_planning(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runs = len(rows)
    rows_with_grasp_data = 0
    selected_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    coarse_total = 0
    full_total = 0
    pruned_total = 0
    coarse_successes = 0
    full_successes = 0
    for row in rows:
        has_grasp_data = isinstance(row.get("grasp_selection_summary"), dict) or isinstance(
            row.get("grasp_selection"), dict
        )
        if has_grasp_data:
            rows_with_grasp_data += 1
        summary = row.get("grasp_selection_summary")
        if not isinstance(summary, dict):
            summary = summarize_grasp_selection(row.get("grasp_selection"))
        selected = summary.get("selected_strategy")
        if selected is not None:
            selected_counts[str(selected)] = selected_counts.get(str(selected), 0) + 1
        coarse_total += int(summary.get("coarse_candidates") or 0)
        full_total += int(summary.get("full_candidates") or 0)
        pruned_total += int(summary.get("coarse_pruned_candidates") or 0)
        coarse_successes += int(summary.get("coarse_successes") or 0)
        full_successes += int(summary.get("full_successes") or 0)
        for code, count in dict(summary.get("rejection_code_counts") or {}).items():
            rejection_counts[str(code)] = rejection_counts.get(str(code), 0) + int(count)
    grasp_runs = rows_with_grasp_data
    return {
        "runs": runs,
        "rows_with_grasp_data": rows_with_grasp_data,
        "rows_without_grasp_data": runs - rows_with_grasp_data,
        "mean_coarse_candidates": 0.0 if grasp_runs == 0 else coarse_total / grasp_runs,
        "mean_full_candidates": 0.0 if grasp_runs == 0 else full_total / grasp_runs,
        "coarse_pruned_candidates": pruned_total,
        "coarse_successes": coarse_successes,
        "full_successes": full_successes,
        "selected_strategy_counts": dict(sorted(selected_counts.items())),
        "rejection_code_counts": dict(sorted(rejection_counts.items())),
    }


def _increment(counter: dict[str, int], code: str, amount: int = 1) -> None:
    counter[code] = counter.get(code, 0) + int(amount)


def summarize_physical_evidence(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    rows_with_evidence = 0
    physical_successes = 0
    failure_reason_counts: dict[str, int] = {}
    failed_check_counts: dict[str, int] = {}
    for row in items:
        evidence = row.get("physical_evidence")
        if not isinstance(evidence, dict):
            continue
        rows_with_evidence += 1
        if bool(evidence.get("success")):
            physical_successes += 1
        for reason in evidence.get("failure_reasons") or ():
            _increment(failure_reason_counts, _normalize_failure_code(reason))
        checks = evidence.get("checks")
        if isinstance(checks, dict):
            for name, check in checks.items():
                if isinstance(check, dict) and check.get("passed") is False:
                    _increment(failed_check_counts, str(name))
    physical_failures = rows_with_evidence - physical_successes
    return {
        "rows": len(items),
        "rows_with_physical_evidence": rows_with_evidence,
        "rows_without_physical_evidence": len(items) - rows_with_evidence,
        "physical_successes": physical_successes,
        "physical_failures": physical_failures,
        "physical_success_rate": 0.0
        if rows_with_evidence == 0
        else physical_successes / rows_with_evidence,
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "failed_check_counts": dict(sorted(failed_check_counts.items())),
    }


def _row_failure_codes(row: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in row.get("failure_reasons") or ():
        _increment(counts, f"task_{_normalize_failure_code(reason)}")
    joint = row.get("joint_metrics")
    if isinstance(joint, dict):
        for reason in joint.get("failure_reasons") or ():
            _increment(counts, f"joint_{_normalize_failure_code(reason)}")
    summary = row.get("grasp_selection_summary")
    if not isinstance(summary, dict):
        summary = summarize_grasp_selection(row.get("grasp_selection"))
    for code, count in dict(summary.get("rejection_code_counts") or {}).items():
        _increment(counts, f"grasp_{code}", int(count))
    evidence = row.get("physical_evidence")
    if isinstance(evidence, dict):
        for reason in evidence.get("failure_reasons") or ():
            _increment(counts, f"physical_{_normalize_failure_code(reason)}")
    return dict(sorted(counts.items()))


def _aggregate_failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall: dict[str, int] = {}
    by_object: dict[str, dict[str, int]] = {}
    by_task: dict[str, dict[str, int]] = {}
    failed_rows = 0
    for row in rows:
        counts = _row_failure_codes(row)
        if counts:
            failed_rows += 1
        object_name = str(row["object_type"])
        task_name = str(row["task_name"])
        object_counts = by_object.setdefault(object_name, {})
        task_counts = by_task.setdefault(task_name, {})
        for code, count in counts.items():
            _increment(overall, code, count)
            _increment(object_counts, code, count)
            _increment(task_counts, code, count)
    return {
        "rows": len(rows),
        "rows_with_failures": failed_rows,
        "overall_code_counts": dict(sorted(overall.items())),
        "by_object": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_object.items())
        },
        "by_task": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_task.items())
        },
    }


def _selected_strategy(row: dict[str, Any]) -> str | None:
    summary = row.get("grasp_selection_summary")
    if isinstance(summary, dict) and summary.get("selected_strategy") is not None:
        return str(summary["selected_strategy"])
    selection = row.get("grasp_selection")
    if isinstance(selection, dict) and selection.get("selected_strategy") is not None:
        return str(selection["selected_strategy"])
    strategy = row.get("grasp_strategy")
    if isinstance(strategy, dict) and strategy.get("name") is not None:
        return str(strategy["name"])
    return None


def _failed_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        if _row_success(row):
            continue
        reasons = [f"task:{reason}" for reason in row.get("failure_reasons") or ()]
        joint = row.get("joint_metrics")
        if isinstance(joint, dict):
            reasons.extend(f"joint:{reason}" for reason in joint.get("failure_reasons") or ())
        reason_counts: dict[str, int] = {}
        for reason in reasons:
            _increment(reason_counts, reason)
        cases.append(
            {
                "object_type": row["object_type"],
                "task_name": row["task_name"],
                "selected_strategy": _selected_strategy(row),
                "path_length": row.get("path_length"),
                "max_position_error": _position_error(row),
                "failure_reasons": sorted(reason_counts),
                "failure_reason_counts": dict(sorted(reason_counts.items())),
            }
        )
    return cases


def _task_splits_for_results(tasks: list[str]) -> dict[str, list[str]]:
    task_set = set(tasks)
    assigned: set[str] = set()
    splits: dict[str, list[str]] = {}
    for split_name, split_tasks in TASK_SPLITS.items():
        present = [task for task in split_tasks if task in task_set]
        splits[split_name] = present
        assigned.update(present)
    splits["unassigned"] = sorted(task_set - assigned)
    return splits


def summarize_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    results = list(payload.get("results", ()))
    objects = sorted({str(row["object_type"]) for row in results})
    tasks = sorted({str(row["task_name"]) for row in results})
    successful = [row for row in results if _row_success(row)]
    best = min(
        successful,
        key=lambda row: (
            float(row.get("path_length", float("inf"))),
            _position_error(row) if _position_error(row) is not None else float("inf"),
        ),
        default=None,
    )
    return {
        "schema_version": 1,
        "source_schema_version": payload.get("schema_version"),
        "overall": _stats(results).to_json_dict(),
        "by_object": {
            name: _stats(row for row in results if row["object_type"] == name).to_json_dict()
            for name in objects
        },
        "by_task": {
            name: _stats(row for row in results if row["task_name"] == name).to_json_dict()
            for name in tasks
        },
        "hardest_tasks": sorted(
            tasks,
            key=lambda name: (
                _stats(row for row in results if row["task_name"] == name).success_rate,
                _stats(row for row in results if row["task_name"] == name).mean_path_length or 0.0,
            ),
        ),
        "best_successful": None
        if best is None
        else {
            "object_type": best["object_type"],
            "task_name": best["task_name"],
            "path_length": best.get("path_length"),
            "max_position_error": _position_error(best),
        },
        "grasp_planning": _aggregate_grasp_planning(results),
        "physical_evidence": summarize_physical_evidence(results),
        "failure_taxonomy": _aggregate_failure_taxonomy(results),
        "failed_cases": _failed_cases(results),
        "task_splits": _task_splits_for_results(tasks),
    }


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "name",
                "runs",
                "successes",
                "success_rate",
                "mean_path_length",
                "max_position_error",
            ],
        )
        writer.writeheader()

        def emit(group: str, name: str, stats: dict[str, Any]) -> None:
            writer.writerow({"group": group, "name": name, **stats})

        emit("overall", "all", summary["overall"])
        for name, stats in summary["by_object"].items():
            emit("object", name, stats)
        for name, stats in summary["by_task"].items():
            emit("task", name, stats)


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = summary["overall"]
    lines = [
        "# MorphTAMP-X Benchmark Summary",
        "",
        f"- Runs: {overall['runs']}",
        f"- Successes: {overall['successes']}",
        f"- Success rate: {overall['success_rate']:.3f}",
        "",
    ]
    best = summary.get("best_successful")
    if best is not None:
        lines += [
            "## Best successful case",
            "",
            f"- Object: `{best['object_type']}`",
            f"- Task: `{best['task_name']}`",
            f"- Path length: {best['path_length']:.6f}",
            "",
        ]
    grasp = summary.get("grasp_planning", {})
    if grasp:
        lines += [
            "## Grasp planning efficiency",
            "",
            f"- Mean coarse candidates: {float(grasp['mean_coarse_candidates']):.3f}",
            f"- Mean full candidates: {float(grasp['mean_full_candidates']):.3f}",
            f"- Coarse-pruned candidates: {int(grasp['coarse_pruned_candidates'])}",
            f"- Coarse successes: {int(grasp['coarse_successes'])}",
            f"- Full successes: {int(grasp['full_successes'])}",
            "",
        ]
        rejection_counts = dict(grasp.get("rejection_code_counts") or {})
        if rejection_counts:
            lines += ["### Coarse rejection codes", "", "| Code | Count |", "|---|---:|"]
            for code, count in rejection_counts.items():
                lines.append(f"| `{code}` | {count} |")
            lines.append("")
    physical = summary.get("physical_evidence", {})
    if physical:
        lines += [
            "## Physical evidence",
            "",
            f"- Rows with physical evidence: {int(physical['rows_with_physical_evidence'])}",
            f"- Physical successes: {int(physical['physical_successes'])}",
            f"- Physical success rate: {float(physical['physical_success_rate']):.3f}",
            "",
        ]
        failed_checks = dict(physical.get("failed_check_counts") or {})
        if failed_checks:
            lines += ["### Failed physical checks", "", "| Check | Count |", "|---|---:|"]
            for check_name, count in failed_checks.items():
                lines.append(f"| `{check_name}` | {count} |")
            lines.append("")
    failure = summary.get("failure_taxonomy", {})
    failure_counts = dict(failure.get("overall_code_counts") or {})
    if failure_counts:
        lines += ["## Failure taxonomy", "", "| Code | Count |", "|---|---:|"]
        for code, count in failure_counts.items():
            lines.append(f"| `{code}` | {count} |")
        lines.append("")
    failed_cases = list(summary.get("failed_cases") or ())
    if failed_cases:
        lines += [
            "## Failed cases",
            "",
            "| Object | Task | Strategy | Max error | Reasons |",
            "|---|---|---|---:|---|",
        ]
        for row in failed_cases:
            reason_counts = dict(row.get("failure_reason_counts") or {})
            reasons = ", ".join(
                f"`{reason}`×{count}" for reason, count in reason_counts.items()
            )
            error = row.get("max_position_error")
            error_text = "" if error is None else f"{float(error):.6f}"
            lines.append(
                f"| `{row['object_type']}` | `{row['task_name']}` | "
                f"`{row.get('selected_strategy')}` | {error_text} | {reasons} |"
            )
        lines.append("")
    splits = dict(summary.get("task_splits") or {})
    if splits:
        lines += ["## Task splits", ""]
        for split_name in ("development", "main", "heldout", "unassigned"):
            values = splits.get(split_name, [])
            if values:
                joined = ", ".join(f"`{value}`" for value in values)
                lines.append(f"- {split_name}: {joined}")
        lines.append("")
    lines += ["## By object", "", "| Object | Success rate | Runs |", "|---|---:|---:|"]
    for name, stats in summary["by_object"].items():
        lines.append(f"| `{name}` | {stats['success_rate']:.3f} | {stats['runs']} |")
    lines += ["", "## By task", "", "| Task | Success rate | Runs |", "|---|---:|---:|"]
    for name, stats in summary["by_task"].items():
        lines.append(f"| `{name}` | {stats['success_rate']:.3f} | {stats['runs']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_benchmark(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
