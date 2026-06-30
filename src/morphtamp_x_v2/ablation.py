from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _design_summaries(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows = [] if payload is None else payload.get("design_summaries", [])
    return {str(row.get("arm_design")): dict(row) for row in rows if row.get("arm_design")}


def _grasp_rows(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return summary.get("grasp_planning", {}) if isinstance(summary.get("grasp_planning"), dict) else {}


def build_ablation_report(
    *,
    final_summary: dict[str, Any],
    morphology_benchmark: dict[str, Any] | None = None,
    benchmark_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    optimization = _section(final_summary, "optimization")
    constraints = _section(optimization, "constraints")
    baselines = _design_summaries(morphology_benchmark)
    grasp = _grasp_rows(benchmark_summary)
    optimized = {
        "name": optimization.get("best_design"),
        "success_rate": optimization.get("best_success_rate"),
        "morphology_cost": optimization.get("best_morphology_cost"),
        "minimum_reach_margin": optimization.get("best_minimum_reach_margin"),
        "minimum_sigma": optimization.get("best_minimum_sigma"),
        "maximum_condition_number": optimization.get("best_maximum_condition_number"),
    }
    reach_active = float(optimization.get("minimum_reach_margin_constraint") or 0.0) > 0.0
    singularity_active = (
        constraints.get("minimum_sigma") is not None
        or constraints.get("maximum_condition_number") is not None
    )
    multi_grasp_active = int(grasp.get("rows_with_grasp_data") or 0) > 0
    short_arm = baselines.get("short_arm")
    nominal = baselines.get("nominal_panda")
    failure_boundary = {
        "short_arm_exposes_boundary": bool(
            short_arm is not None and float(short_arm.get("success_rate", 0.0)) < 1.0
        ),
        "short_arm_success_rate": None if short_arm is None else short_arm.get("success_rate"),
        "nominal_success_rate": None if nominal is None else nominal.get("success_rate"),
    }
    components = {
        "reach_margin_constraint": {
            "active": reach_active,
            "evidence": optimization.get("best_minimum_reach_margin"),
            "threshold": optimization.get("minimum_reach_margin_constraint"),
        },
        "singularity_constraint": {
            "active": singularity_active,
            "min_sigma": optimization.get("best_minimum_sigma"),
            "max_condition_number": optimization.get("best_maximum_condition_number"),
            "thresholds": constraints,
        },
        "multi_grasp_selection": {
            "active": multi_grasp_active,
            "rows_with_grasp_data": grasp.get("rows_with_grasp_data", 0),
            "mean_coarse_candidates": grasp.get("mean_coarse_candidates", 0.0),
            "mean_full_candidates": grasp.get("mean_full_candidates", 0.0),
        },
    }
    paper_ready = (
        optimized["name"] is not None
        and float(optimization.get("best_success_rate") or 0.0) >= 1.0
        and reach_active
        and singularity_active
    )
    return {
        "schema_version": 1,
        "optimized_design": optimized,
        "baselines": baselines,
        "components": components,
        "failure_boundary": failure_boundary,
        "recommendation": {
            "paper_ready": paper_ready,
            "next_missing_evidence": [
                name
                for name, active in {
                    "multi_grasp_selection": multi_grasp_active,
                    "short_arm_failure_boundary": failure_boundary["short_arm_exposes_boundary"],
                }.items()
                if not active
            ],
        },
    }


def write_ablation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Ablation and Failure-Boundary Report",
        "",
        "This report summarizes which method components are supported by the current evidence.",
        "",
        "## Optimized design",
        "",
        f"- Design: `{report['optimized_design'].get('name')}`",
        f"- Success rate: {report['optimized_design'].get('success_rate')}",
        f"- Morphology cost: {report['optimized_design'].get('morphology_cost')}",
        "",
        "## Component evidence",
        "",
        "| Component | Active | Key evidence |",
        "|---|---:|---|",
    ]
    for name, row in report["components"].items():
        lines.append(f"| `{name}` | {bool(row.get('active'))} | `{json.dumps(row, sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "## Baseline and failure boundary",
            "",
            "| Design | Success rate | Morphology cost | Min reach margin |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in sorted(report["baselines"].items()):
        lines.append(
            f"| `{name}` | {row.get('success_rate')} | "
            f"{row.get('morphology_cost')} | {row.get('minimum_reach_margin')} |"
        )
    lines.extend(
        [
            "",
            "A useful morphology benchmark should expose a failure boundary: a too-short "
            "design should become cheaper but fail reach or stress constraints. If no "
            "baseline fails, the benchmark may be too easy for a strong optimization claim.",
            "",
        ]
    )
    return "\n".join(lines)


def load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_ablation_files(
    *,
    final_summary: dict[str, Any],
    morphology_benchmark: dict[str, Any] | None,
    benchmark_summary: dict[str, Any] | None,
    output_json: str | Path,
    output_md: str | Path | None = None,
) -> dict[str, Any]:
    report = build_ablation_report(
        final_summary=final_summary,
        morphology_benchmark=morphology_benchmark,
        benchmark_summary=benchmark_summary,
    )
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if output_md is not None:
        output_md = Path(output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(write_ablation_markdown(report), encoding="utf-8")
    return report
