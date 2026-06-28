from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .benchmark_analysis import summarize_benchmark


def _successful(row: dict[str, Any]) -> bool:
    joint = row.get("joint_metrics")
    return bool(row.get("success")) and (joint is None or bool(joint.get("success")))


def _joint_error(row: dict[str, Any]) -> float | None:
    joint = row.get("joint_metrics")
    if not isinstance(joint, dict):
        return None
    value = joint.get("max_position_error")
    return None if value is None else float(value)


def _svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#1f2937">{html.escape(text)}</text>'
    )


def _heat_color(value: float, vmax: float) -> str:
    if vmax <= 0.0:
        return "#dcfce7"
    t = max(0.0, min(1.0, value / vmax))
    if t < 0.5:
        local = t / 0.5
        r = int(34 + (250 - 34) * local)
        g = int(197 + (204 - 197) * local)
        b = int(94 + (21 - 94) * local)
    else:
        local = (t - 0.5) / 0.5
        r = int(250 + (239 - 250) * local)
        g = int(204 + (68 - 204) * local)
        b = int(21 + (68 - 21) * local)
    return f"#{r:02x}{g:02x}{b:02x}"


def _object_task_grid(benchmark: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows = list(benchmark.get("results", ()))
    objects = sorted({str(row["object_type"]) for row in rows})
    tasks = sorted({str(row["task_name"]) for row in rows})
    return rows, objects, tasks


def write_success_matrix_svg(benchmark: dict[str, Any], path: Path) -> None:
    rows, objects, tasks = _object_task_grid(benchmark)
    cell = 58
    left = 130
    top = 70
    width = left + cell * len(tasks) + 30
    height = top + cell * len(objects) + 50
    lookup = {(row["object_type"], row["task_name"]): row for row in rows}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 28, "Object-Task Success Matrix", size=18),
    ]
    for j, task in enumerate(tasks):
        parts.append(_svg_text(left + j * cell + cell / 2, 56, task, size=10))
    for i, obj in enumerate(objects):
        y = top + i * cell
        parts.append(_svg_text(left - 10, y + cell / 2 + 4, obj, size=11, anchor="end"))
        for j, task in enumerate(tasks):
            x = left + j * cell
            row = lookup.get((obj, task))
            success = row is not None and _successful(row)
            color = "#22c55e" if success else "#ef4444"
            label = "OK" if success else "FAIL"
            error = None if row is None else _joint_error(row)
            sub = "" if error is None else f"{error:.2f}m"
            parts += [
                f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" rx="8" fill="{color}" opacity="0.86"/>',
                _svg_text(x + cell / 2 - 2, y + 25, label, size=11),
                _svg_text(x + cell / 2 - 2, y + 43, sub, size=9),
            ]
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_error_heatmap_svg(benchmark: dict[str, Any], path: Path) -> None:
    rows, objects, tasks = _object_task_grid(benchmark)
    errors = [error for row in rows if (error := _joint_error(row)) is not None]
    vmax = max(errors) if errors else 1.0
    cell = 64
    left = 130
    top = 76
    width = left + cell * len(tasks) + 120
    height = top + cell * len(objects) + 78
    lookup = {(row["object_type"], row["task_name"]): row for row in rows}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 28, "Object-Task TCP Error Heatmap", size=18),
        _svg_text(width / 2, 48, "Darker red means larger max TCP / IK error", size=11),
    ]
    for j, task in enumerate(tasks):
        parts.append(_svg_text(left + j * cell + cell / 2, 66, task, size=10))
    for i, obj in enumerate(objects):
        y = top + i * cell
        parts.append(_svg_text(left - 10, y + cell / 2 + 4, obj, size=11, anchor="end"))
        for j, task in enumerate(tasks):
            x = left + j * cell
            row = lookup.get((obj, task))
            error = None if row is None else _joint_error(row)
            color = "#e5e7eb" if error is None else _heat_color(error, vmax)
            label = "n/a" if error is None else f"{error:.3f}"
            parts += [
                f'<rect x="{x}" y="{y}" width="{cell - 5}" height="{cell - 5}" rx="8" fill="{color}" opacity="0.9"/>',
                _svg_text(x + cell / 2 - 2, y + cell / 2 + 4, label, size=10),
            ]
    parts.append(_svg_text(width - 62, height - 32, f"max={vmax:.3f} m", size=10))
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_morphology_pareto_svg(morphology: dict[str, Any], path: Path) -> None:
    rows = list(morphology.get("design_summaries", ()))
    width, height = 720, 420
    left, right, top, bottom = 80, 30, 45, 65
    plot_w = width - left - right
    plot_h = height - top - bottom
    costs = [float(row["morphology_cost"]) for row in rows] or [0.0, 1.0]
    rates = [float(row["success_rate"]) for row in rows] or [0.0, 1.0]
    cmin, cmax = min(costs), max(costs)
    rmin, rmax = min(0.0, min(rates)), max(1.0, max(rates))
    if cmax == cmin:
        cmax = cmin + 1.0
    pareto = set(morphology.get("pareto_designs", ()))

    def sx(cost: float) -> float:
        return left + (cost - cmin) / (cmax - cmin) * plot_w

    def sy(rate: float) -> float:
        return top + (rmax - rate) / (rmax - rmin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 26, "Morphology Cost vs Success Rate", size=18),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#374151"/>',
        _svg_text(width / 2, height - 18, "Morphology cost / equivalent reach length", size=12),
        f'<text x="18" y="{height/2}" font-family="Arial, sans-serif" font-size="12" fill="#1f2937" transform="rotate(-90 18 {height/2})">Success rate</text>',
    ]
    for row in rows:
        name = str(row["arm_design"])
        x = sx(float(row["morphology_cost"]))
        y = sy(float(row["success_rate"]))
        color = "#2563eb" if name in pareto else "#94a3b8"
        radius = 8 if name in pareto else 6
        parts += [
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="0.92"/>',
            _svg_text(x + 7, y - 8, name, size=10, anchor="start"),
        ]
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_reach_margin_svg(morphology: dict[str, Any], path: Path) -> None:
    rows = sorted(
        list(morphology.get("design_summaries", ())),
        key=lambda row: float(row.get("minimum_reach_margin", 0.0)),
    )
    width, height = 760, 420
    left, right, top, bottom = 160, 40, 50, 55
    plot_w = width - left - right
    bar_h = 28
    gap = 14
    margins = [float(row.get("minimum_reach_margin", 0.0)) for row in rows] or [0.0]
    mmin, mmax = min(margins + [0.0]), max(margins + [0.0])
    if mmax == mmin:
        mmax = mmin + 1.0

    def sx(value: float) -> float:
        return left + (value - mmin) / (mmax - mmin) * plot_w

    zero_x = sx(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 28, "Minimum Reach Margin by Morphology", size=18),
        f'<line x1="{zero_x:.1f}" y1="{top - 8}" x2="{zero_x:.1f}" y2="{height - bottom + 14}" stroke="#111827" stroke-dasharray="4 4"/>',
        _svg_text(zero_x, height - 18, "zero margin", size=10),
    ]
    for index, row in enumerate(rows):
        name = str(row["arm_design"])
        margin = float(row.get("minimum_reach_margin", 0.0))
        y = top + index * (bar_h + gap)
        x0 = min(zero_x, sx(margin))
        x1 = max(zero_x, sx(margin))
        color = "#2563eb" if margin >= 0.0 else "#ef4444"
        parts += [
            _svg_text(left - 12, y + 19, name, size=11, anchor="end"),
            f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(2.0, x1 - x0):.1f}" height="{bar_h}" rx="6" fill="{color}" opacity="0.85"/>',
            _svg_text(x1 + 8 if margin >= 0.0 else x0 - 8, y + 19, f"{margin:.3f} m", size=10, anchor="start" if margin >= 0.0 else "end"),
        ]
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_bar_chart_svg(
    values: dict[str, float],
    path: Path,
    *,
    title: str,
    x_label: str = "Count",
    color: str = "#2563eb",
) -> None:
    items = [(str(name), float(value)) for name, value in values.items()]
    width = 760
    left, right, top, bottom = 210, 42, 56, 58
    bar_h = 28
    gap = 14
    height = max(260, top + len(items) * (bar_h + gap) + bottom)
    plot_w = width - left - right
    vmax = max([value for _, value in items] + [1.0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 30, title, size=18),
        f'<line x1="{left}" y1="{height - bottom + 8}" x2="{left + plot_w}" y2="{height - bottom + 8}" stroke="#374151"/>',
        _svg_text(width / 2, height - 18, x_label, size=12),
    ]
    if not items:
        parts.append(_svg_text(width / 2, height / 2, "No data", size=13))
    for index, (name, value) in enumerate(items):
        y = top + index * (bar_h + gap)
        bar_w = 0.0 if vmax <= 0 else value / vmax * plot_w
        parts += [
            _svg_text(left - 12, y + 19, name, size=11, anchor="end"),
            f'<rect x="{left:.1f}" y="{y:.1f}" width="{max(2.0, bar_w):.1f}" height="{bar_h}" rx="6" fill="{color}" opacity="0.86"/>',
            _svg_text(left + bar_w + 10, y + 19, f"{value:.3g}", size=10, anchor="start"),
        ]
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_grasp_efficiency_svg(summary: dict[str, Any], path: Path) -> None:
    grasp = dict(summary.get("grasp_planning") or {})
    if int(grasp.get("rows_with_grasp_data") or 0) == 0:
        values = {}
    else:
        values = {
            "Mean coarse candidates": float(grasp.get("mean_coarse_candidates") or 0.0),
            "Mean full candidates": float(grasp.get("mean_full_candidates") or 0.0),
            "Total coarse-pruned": float(grasp.get("coarse_pruned_candidates") or 0.0),
            "Coarse successes": float(grasp.get("coarse_successes") or 0.0),
            "Full successes": float(grasp.get("full_successes") or 0.0),
        }
    _write_bar_chart_svg(
        values,
        path,
        title="Coarse-to-Fine Grasp Planning Efficiency",
        x_label="Candidates / successes",
        color="#0072B2",
    )


def write_selected_strategy_svg(summary: dict[str, Any], path: Path) -> None:
    grasp = dict(summary.get("grasp_planning") or {})
    values = {
        name: float(count)
        for name, count in sorted(
            dict(grasp.get("selected_strategy_counts") or {}).items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
    }
    _write_bar_chart_svg(
        values,
        path,
        title="Selected Grasp Strategy Distribution",
        x_label="Selections",
        color="#009E73",
    )


def write_rejection_code_svg(summary: dict[str, Any], path: Path) -> None:
    grasp = dict(summary.get("grasp_planning") or {})
    values = {
        name: float(count)
        for name, count in sorted(
            dict(grasp.get("rejection_code_counts") or {}).items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
    }
    _write_bar_chart_svg(
        values,
        path,
        title="Coarse Grasp Rejection Codes",
        x_label="Rejections",
        color="#D55E00",
    )


def write_system_pipeline_svg(path: Path) -> None:
    width, height = 980, 260
    boxes = [
        ("Object/task library", ("Objects", "Scenes", "Constraints")),
        ("Grasp candidates", ("Top", "Side", "Edge / 6D")),
        ("Coarse IK filter", ("Key-frame", "Panda check", "Fast prune")),
        ("Full replay", ("7D IK", "Collision", "Joint margin")),
        ("Morphology score", ("Reach", "Cost", "Energy proxy")),
        ("Reports", ("Pareto", "Figures", "Evidence")),
    ]
    box_w, box_h = 135, 82
    start_x, y = 32, 94
    gap = 25
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#374151"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 30, "MorphTAMP-X Pipeline", size=20),
        _svg_text(width / 2, 54, "Task-driven arm morphology evaluation with coarse-to-fine grasp planning", size=12),
    ]
    for index, (title, lines) in enumerate(boxes):
        x = start_x + index * (box_w + gap)
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="12" '
            'fill="#e0f2fe" stroke="#0369a1" stroke-width="1.4"/>'
        )
        parts.append(_svg_text(x + box_w / 2, y + 23, title, size=12))
        for line_index, line in enumerate(lines):
            parts.append(_svg_text(x + box_w / 2, y + 43 + 14 * line_index, line, size=10))
        if index < len(boxes) - 1:
            x1 = x + box_w + 5
            x2 = x + box_w + gap - 5
            yy = y + box_h / 2
            parts.append(
                f'<line x1="{x1}" y1="{yy}" x2="{x2}" y2="{yy}" '
                'stroke="#374151" stroke-width="1.6" marker-end="url(#arrow)"/>'
            )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_kinematic_quality_svg(benchmark: dict[str, Any], path: Path) -> None:
    values: dict[str, float] = {}
    for row in benchmark.get("results", ()):
        joint = row.get("joint_metrics")
        if not isinstance(joint, dict) or joint.get("max_position_error") is None:
            continue
        values[f"{row['object_type']} / {row['task_name']}"] = float(joint["max_position_error"])
    _write_bar_chart_svg(
        dict(sorted(values.items(), key=lambda item: item[1], reverse=True)),
        path,
        title="Kinematic Quality: Max TCP / IK Error",
        x_label="Max position error (m)",
        color="#CC79A7",
    )


def _difficulty_values(summary: dict[str, Any], group: str) -> dict[str, float]:
    rows = dict(summary.get(group) or {})
    values = {
        name: 1.0 - float(stats.get("success_rate", 0.0))
        for name, stats in rows.items()
    }
    return dict(sorted(values.items(), key=lambda item: (-item[1], item[0])))


def write_task_difficulty_svg(summary: dict[str, Any], path: Path) -> None:
    _write_bar_chart_svg(
        _difficulty_values(summary, "by_task"),
        path,
        title="Task Difficulty Ranking",
        x_label="Failure rate",
        color="#E69F00",
    )


def write_object_difficulty_svg(summary: dict[str, Any], path: Path) -> None:
    _write_bar_chart_svg(
        _difficulty_values(summary, "by_object"),
        path,
        title="Object Difficulty Ranking",
        x_label="Failure rate",
        color="#56B4E9",
    )


def write_collision_failure_map_svg(benchmark: dict[str, Any], path: Path) -> None:
    values: dict[str, float] = {}
    for row in benchmark.get("results", ()):
        summary = row.get("grasp_selection_summary")
        if not isinstance(summary, dict):
            continue
        count = float(dict(summary.get("rejection_code_counts") or {}).get("collision", 0))
        if count > 0.0:
            values[f"{row['object_type']} / {row['task_name']}"] = count
    _write_bar_chart_svg(
        dict(sorted(values.items(), key=lambda item: (-item[1], item[0]))),
        path,
        title="Collision-Related Coarse Rejections",
        x_label="Collision rejection count",
        color="#ef4444",
    )


def _figure_number(figure_id: str) -> int:
    head = figure_id.split("_", 1)[0]
    if not head.startswith("fig"):
        return 999
    try:
        return int(head[3:])
    except ValueError:
        return 999


def _figure_title(figure_id: str) -> str:
    return figure_id.replace("_", " ").title()


def data_quality_report(summary: dict[str, Any]) -> dict[str, Any]:
    grasp = dict(summary.get("grasp_planning") or {})
    runs = int(grasp.get("runs") or summary.get("overall", {}).get("runs") or 0)
    with_data = int(grasp.get("rows_with_grasp_data") or 0)
    without_data = int(grasp.get("rows_without_grasp_data") or max(0, runs - with_data))
    warnings: list[str] = []
    if runs > 0 and with_data == 0:
        warnings.append(
            "Benchmark rows contain no grasp candidate data; grasp funnel, strategy, "
            "and collision-rejection figures are diagnostic placeholders. Regenerate "
            "the benchmark with the current planner so rows include grasp_selection_summary."
        )
    elif without_data > 0:
        warnings.append(
            f"{without_data} / {runs} benchmark rows are missing grasp candidate data; "
            "grasp-planning averages are computed only from rows with grasp evidence."
        )
    return {
        "grasp_rows_total": runs,
        "grasp_rows_with_data": with_data,
        "grasp_rows_without_data": without_data,
        "warnings": warnings,
    }


def write_dashboard(
    output_dir: Path,
    *,
    figures: dict[str, str],
    summary: dict[str, Any],
    data_quality: dict[str, Any],
) -> Path:
    best = summary.get("best_successful")
    main_figures = {
        key: value
        for key, value in figures.items()
        if _figure_number(key) <= 6
    }
    diagnostic_figures = {
        key: value
        for key, value in figures.items()
        if _figure_number(key) >= 7
    }
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>MorphTAMP-X Visualization</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;color:#111827}"
        "section{margin-top:28px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}"
        "figure{margin:0;padding:14px;border:1px solid #e5e7eb;border-radius:14px;background:#fff}"
        "img{max-width:100%;display:block} figcaption{font-weight:700;margin-bottom:8px}"
        "code{background:#f3f4f6;padding:2px 4px;border-radius:4px}</style>",
        "</head><body>",
        "<h1>MorphTAMP-X Result Dashboard</h1>",
        f"<p>Overall success rate: <strong>{summary['overall']['success_rate']:.3f}</strong></p>",
    ]
    if best is not None:
        body.append(
            f"<p>Best successful object/task: <code>{html.escape(best['object_type'])}</code> / "
            f"<code>{html.escape(best['task_name'])}</code></p>"
        )
    warnings = list(data_quality.get("warnings") or ())
    if warnings:
        body += [
            "<section><h2>Data Quality Warnings</h2>",
            "<ul>",
        ]
        for warning in warnings:
            body.append(f"<li>{html.escape(str(warning))}</li>")
        body += ["</ul></section>"]
    for section_title, group in (
        ("Main Paper Figures", main_figures),
        ("Diagnostic Figures", diagnostic_figures),
    ):
        body += [f"<section><h2>{section_title}</h2>", "<div class='grid'>"]
        for figure_id, figure_path in sorted(group.items(), key=lambda item: _figure_number(item[0])):
            body += [
                "<figure>",
                f"<figcaption>{html.escape(_figure_title(figure_id))}</figcaption>",
                f"<img src='{html.escape(figure_path)}' alt='{html.escape(figure_id)}'>",
                "</figure>",
            ]
        body += ["</div></section>"]
    body.append("</body></html>")
    path = output_dir / "dashboard.html"
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def visualize_results(benchmark_path: Path, morphology_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_dir = output_dir / "main"
    diagnostics_dir = output_dir / "diagnostics"
    evidence_dir = output_dir / "evidence"
    for directory in (main_dir, diagnostics_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    morphology = json.loads(morphology_path.read_text(encoding="utf-8"))
    summary = summarize_benchmark(benchmark)
    data_quality = data_quality_report(summary)
    figure_paths = {
        "fig1_system_pipeline": main_dir / "fig1_system_pipeline.svg",
        "fig2_object_task_success_heatmap": main_dir / "fig2_object_task_success_heatmap.svg",
        "fig3_grasp_candidate_funnel": main_dir / "fig3_grasp_candidate_funnel.svg",
        "fig4_rejection_reason_distribution": main_dir / "fig4_rejection_reason_distribution.svg",
        "fig5_morphology_pareto": main_dir / "fig5_morphology_pareto.svg",
        "fig6_kinematic_quality": main_dir / "fig6_kinematic_quality.svg",
        "fig7_selected_strategy_distribution": diagnostics_dir / "fig7_selected_strategy_distribution.svg",
        "fig8_task_difficulty_ranking": diagnostics_dir / "fig8_task_difficulty_ranking.svg",
        "fig9_object_difficulty_ranking": diagnostics_dir / "fig9_object_difficulty_ranking.svg",
        "fig10_coarse_vs_full_cost": diagnostics_dir / "fig10_coarse_vs_full_cost.svg",
        "fig11_collision_failure_map": diagnostics_dir / "fig11_collision_failure_map.svg",
    }
    write_system_pipeline_svg(figure_paths["fig1_system_pipeline"])
    write_success_matrix_svg(benchmark, figure_paths["fig2_object_task_success_heatmap"])
    write_grasp_efficiency_svg(summary, figure_paths["fig3_grasp_candidate_funnel"])
    write_rejection_code_svg(summary, figure_paths["fig4_rejection_reason_distribution"])
    write_morphology_pareto_svg(morphology, figure_paths["fig5_morphology_pareto"])
    write_kinematic_quality_svg(benchmark, figure_paths["fig6_kinematic_quality"])
    write_selected_strategy_svg(summary, figure_paths["fig7_selected_strategy_distribution"])
    write_task_difficulty_svg(summary, figure_paths["fig8_task_difficulty_ranking"])
    write_object_difficulty_svg(summary, figure_paths["fig9_object_difficulty_ranking"])
    write_grasp_efficiency_svg(summary, figure_paths["fig10_coarse_vs_full_cost"])
    write_collision_failure_map_svg(benchmark, figure_paths["fig11_collision_failure_map"])
    relative_figures = {
        figure_id: str(path.relative_to(output_dir))
        for figure_id, path in figure_paths.items()
    }
    dashboard = write_dashboard(
        output_dir,
        figures=relative_figures,
        summary=summary,
        data_quality=data_quality,
    )
    figure_groups = {
        "main": [key for key in figure_paths if _figure_number(key) <= 6],
        "diagnostics": [key for key in figure_paths if _figure_number(key) >= 7],
        "evidence": [],
    }
    manifest = {
        "schema_version": 1,
        "benchmark": str(benchmark_path),
        "morphology": str(morphology_path),
        "figure_groups": figure_groups,
        "figures": {figure_id: str(path) for figure_id, path in figure_paths.items()},
        "data_quality": data_quality,
        "outputs": {"dashboard": str(dashboard)},
        "overall_success_rate": summary["overall"]["success_rate"],
        "best_successful": summary["best_successful"],
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
