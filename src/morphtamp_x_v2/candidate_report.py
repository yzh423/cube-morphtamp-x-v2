from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .grasp_planner import CandidateEvaluation
from .grasp_strategies import strategies_for_object
from .models import PickPlaceScenario
from .planner import plan_pick_place


def _metric_payload(evaluation: CandidateEvaluation) -> dict[str, Any]:
    def finite(value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    return {
        "success": evaluation.success,
        "max_position_error": finite(evaluation.max_position_error),
        "max_orientation_error": finite(evaluation.max_orientation_error),
        "min_joint_margin": finite(evaluation.min_joint_margin),
        "collision_count": evaluation.collision_count,
        "path_length": finite(evaluation.path_length),
        "joint_path_length": finite(evaluation.joint_path_length),
        "max_joint_step": finite(evaluation.max_joint_step),
        "max_condition_number": finite(evaluation.max_condition_number),
    }


def _path_for_strategy(
    scenario: PickPlaceScenario,
    strategy_name: str,
) -> tuple[tuple[float, float, float], ...]:
    strategies = {strategy.name: strategy for strategy in strategies_for_object(scenario.object_type)}
    strategy = strategies.get(strategy_name)
    if strategy is None:
        return ()
    phases = plan_pick_place(scenario, grasp_strategy=strategy)
    return tuple(tuple(float(value) for value in phase.tcp_position) for phase in phases)


def compact_failure_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    counts: dict[tuple[str, str], int] = {}
    order: list[tuple[str, str]] = []
    for reason in reasons:
        parts = str(reason).split(":")
        key = (
            parts[0] if len(parts) > 0 and parts[0] else "unknown_phase",
            parts[1] if len(parts) > 1 and parts[1] else "unknown_failure",
        )
        if key not in counts:
            order.append(key)
            counts[key] = 0
        counts[key] += 1
    return tuple(f"{phase}:{code} x{counts[(phase, code)]}" for phase, code in order)


def build_candidate_report(
    scenario: PickPlaceScenario,
    *,
    selected_strategy: str,
    evaluations: Iterable[CandidateEvaluation],
) -> dict[str, Any]:
    candidates = []
    for evaluation in evaluations:
        path_points = _path_for_strategy(scenario, evaluation.strategy_name)
        candidates.append(
            {
                "strategy_name": evaluation.strategy_name,
                "selected": evaluation.strategy_name == selected_strategy,
                "metrics": _metric_payload(evaluation),
                "failure_reasons": list(evaluation.failure_reasons),
                "failure_summary": list(compact_failure_reasons(evaluation.failure_reasons)),
                "path_points": [list(point) for point in path_points],
            }
        )
    return {
        "schema_version": 1,
        "title": "Candidate trajectory comparison",
        "scenario": scenario.to_json_dict(),
        "selected_strategy": selected_strategy,
        "candidates": candidates,
    }


def _bounds(candidates: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    points = [
        point
        for candidate in candidates
        for point in candidate.get("path_points", ())
    ]
    if not points:
        return 0.0, 1.0, 0.0, 1.0
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 0.06
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _polyline(points: list[list[float]], *, bounds: tuple[float, float, float, float], color: str) -> str:
    if len(points) < 2:
        return ""
    xmin, xmax, ymin, ymax = bounds
    width, height = 520.0, 320.0
    left, top = 34.0, 26.0

    def project(point: list[float]) -> tuple[float, float]:
        x = left + (float(point[0]) - xmin) / (xmax - xmin) * width
        y = top + (ymax - float(point[1])) / (ymax - ymin) * height
        return x, y

    projected = [project(point) for point in points]
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in projected)
    start = projected[0]
    end = projected[-1]
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        'stroke-width="3.0" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{start[0]:.1f}" cy="{start[1]:.1f}" r="4" fill="{color}" opacity="0.65"/>'
        f'<circle cx="{end[0]:.1f}" cy="{end[1]:.1f}" r="6" fill="{color}"/>'
    )


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _write_html(report: dict[str, Any], path: Path) -> None:
    candidates = list(report.get("candidates", ()))
    bounds = _bounds(candidates)
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#f59e0b", "#0891b2"]
    svg_parts = [
        '<svg viewBox="0 0 590 380" width="100%" role="img" '
        'aria-label="candidate top-down TCP trajectories">',
        '<rect width="590" height="380" fill="#f8fafc" rx="12"/>',
        '<text x="295" y="20" text-anchor="middle" font-family="Arial" font-size="15" fill="#111827">Top-down TCP path comparison</text>',
    ]
    for index, candidate in enumerate(candidates):
        color = "#16a34a" if candidate.get("selected") else colors[index % len(colors)]
        svg_parts.append(_polyline(candidate.get("path_points", ()), bounds=bounds, color=color))
    svg_parts.append("</svg>")

    cards = []
    replay_lookup = {
        str(row.get("strategy_name")): row
        for row in dict(report.get("candidate_replays") or {}).get("candidates", ())
    }
    for index, candidate in enumerate(candidates):
        metrics = dict(candidate.get("metrics") or {})
        failures = list(candidate.get("failure_summary") or candidate.get("failure_reasons") or ())
        selected = bool(candidate.get("selected"))
        badge = "SELECTED" if selected else ("OK" if metrics.get("success") else "REJECTED")
        card_class = "selected" if selected else ""
        replay = replay_lookup.get(str(candidate["strategy_name"]), {})
        replay_json = replay.get("replay_json")
        scene_xml = replay.get("scene_xml")
        replay_links = ""
        if replay_json:
            replay_links += (
                f"<p class='links'>Replay JSON: "
                f"<code>{html.escape(str(replay_json))}</code></p>"
            )
        if scene_xml:
            replay_links += (
                f"<p class='links'>Scene XML: "
                f"<code>{html.escape(str(scene_xml))}</code></p>"
            )
        failure_html = (
            "<li>none</li>"
            if not failures
            else "".join(f"<li>{html.escape(str(reason))}</li>" for reason in failures)
        )
        metric_rows = "".join(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(_format_metric(metrics.get(key)))}</td>"
            "</tr>"
            for label, key in (
                ("success", "success"),
                ("max IK error", "max_position_error"),
                ("orientation error", "max_orientation_error"),
                ("joint margin", "min_joint_margin"),
                ("collisions", "collision_count"),
                ("TCP path", "path_length"),
                ("joint path", "joint_path_length"),
                ("max joint step", "max_joint_step"),
                ("max cond.", "max_condition_number"),
            )
        )
        cards.append(
            f"<article class='card {card_class}'>"
            f"<h3>{index + 1}. {html.escape(str(candidate['strategy_name']))} <span>{badge}</span></h3>"
            f"<table>{metric_rows}</table>"
            f"{replay_links}"
            "<h4>Failure reasons</h4>"
            f"<ul>{failure_html}</ul>"
            "</article>"
        )

    document = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Candidate trajectory comparison</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:28px;color:#111827;background:#f1f5f9}",
        "main{max-width:1180px;margin:auto}",
        ".panel{background:white;border:1px solid #e5e7eb;border-radius:16px;padding:18px;margin-bottom:18px}",
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:14px}",
        ".card{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:14px}",
        ".card.selected{border-color:#16a34a;box-shadow:0 0 0 2px rgba(22,163,74,.15)}",
        "h1{margin:0 0 8px} h3{margin:0 0 10px;font-size:16px} h3 span{float:right;font-size:11px;background:#e5e7eb;border-radius:999px;padding:3px 8px}",
        ".selected h3 span{background:#dcfce7;color:#166534}",
        "table{width:100%;border-collapse:collapse;font-size:13px}td{border-bottom:1px solid #e5e7eb;padding:5px 2px}td:last-child{text-align:right;font-variant-numeric:tabular-nums}",
        "ul{font-size:12px;margin-top:4px;padding-left:18px}",
        ".links{font-size:12px;margin:8px 0 0;color:#475569}",
        "code{background:#e5e7eb;border-radius:4px;padding:2px 5px}",
        "</style></head><body><main>",
        "<section class='panel'>",
        "<h1>Candidate trajectory comparison</h1>",
        f"<p>Object/task: <code>{html.escape(str(report['scenario']['object_type']))}</code> / "
        f"<code>{html.escape(str(report['scenario']['task_name']))}</code>. "
        f"Selected strategy: <code>{html.escape(str(report['selected_strategy']))}</code>.</p>",
        "\n".join(svg_parts),
        "</section>",
        "<section class='cards'>",
        "\n".join(cards),
        "</section>",
        "</main></body></html>",
    ]
    path.write_text("\n".join(document), encoding="utf-8")


def write_candidate_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "candidate_report.json"
    html_path = output_dir / "candidate_report.html"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_html(report, html_path)
    return {"json": json_path, "html": html_path}
