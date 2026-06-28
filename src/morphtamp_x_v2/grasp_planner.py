from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .grasp_strategies import GraspStrategy, strategies_for_object
from .models import Phase, PickPlaceScenario, ReplayFrame
from .planner import plan_pick_place
from .replay import build_replay_frames
from .scene_builder import build_pick_place_scene
from .panda_ik import solve_joint_replay
from .validator import evaluate_replay


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class CandidateEvaluation:
    strategy_name: str
    success: bool
    max_position_error: float
    min_joint_margin: float
    collision_count: int
    path_length: float
    failure_reasons: tuple[str, ...]
    max_orientation_error: float = 0.0
    joint_path_length: float = 0.0
    max_joint_step: float = 0.0
    max_condition_number: float | None = None
    min_sigma: float | None = None

    def score_key(self) -> tuple[int, int, float, float, int, float, float, float, float, float, float]:
        residual = float(self.max_position_error)
        residual_for_ranking = 0.0 if residual <= 1e-4 else residual
        orientation_error = float(self.max_orientation_error)
        orientation_for_ranking = 0.0 if orientation_error <= 1e-3 else orientation_error
        condition = (
            float(self.max_condition_number)
            if self.max_condition_number is not None and math.isfinite(float(self.max_condition_number))
            else 0.0
        )
        sigma_penalty = (
            -float(self.min_sigma)
            if self.min_sigma is not None and math.isfinite(float(self.min_sigma))
            else 0.0
        )
        return (
            0 if self.success else 1,
            int(self.collision_count),
            residual_for_ranking,
            orientation_for_ranking,
            0 if self.min_joint_margin >= 0.02 else 1,
            -float(self.min_joint_margin),
            sigma_penalty,
            condition,
            float(self.joint_path_length),
            float(self.max_joint_step),
            float(self.path_length),
        )

    def to_json_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "success": self.success,
            "max_position_error": _json_number(self.max_position_error),
            "max_orientation_error": _json_number(self.max_orientation_error),
            "min_joint_margin": _json_number(self.min_joint_margin),
            "collision_count": self.collision_count,
            "path_length": self.path_length,
            "joint_path_length": self.joint_path_length,
            "max_joint_step": self.max_joint_step,
            "max_condition_number": None
            if self.max_condition_number is None
            else _json_number(float(self.max_condition_number)),
            "min_sigma": None if self.min_sigma is None else _json_number(float(self.min_sigma)),
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class GraspPlanSelection:
    selected_strategy: GraspStrategy
    selected_evaluation: CandidateEvaluation
    evaluations: tuple[CandidateEvaluation, ...]
    phases: tuple[Phase, ...]
    coarse_evaluations: tuple[CandidateEvaluation, ...] = ()

    def to_json_dict(self) -> dict:
        return {
            "selected_strategy": self.selected_strategy.name,
            "selected_evaluation": self.selected_evaluation.to_json_dict(),
            "candidate_evaluations": [row.to_json_dict() for row in self.evaluations],
            "coarse_candidate_evaluations": [row.to_json_dict() for row in self.coarse_evaluations],
        }


def path_length_for_phases(phases: tuple[Phase, ...]) -> float:
    total = 0.0
    previous = None
    for phase in phases:
        if previous is not None:
            total += float(np.linalg.norm(np.asarray(phase.tcp_position) - np.asarray(previous)))
        previous = phase.tcp_position
    return total


def joint_motion_metrics(q_sequence: tuple[tuple[float, ...], ...]) -> tuple[float, float]:
    """Return cumulative 7D joint distance and largest adjacent joint step."""

    if len(q_sequence) < 2:
        return 0.0, 0.0
    total = 0.0
    max_step = 0.0
    previous = np.asarray(q_sequence[0], dtype=float)
    for q in q_sequence[1:]:
        current = np.asarray(q, dtype=float)
        step = float(np.linalg.norm(current - previous))
        total += step
        max_step = max(max_step, step)
        previous = current
    return float(total), float(max_step)


def replay_joint_motion_metrics(replay) -> tuple[float, float, float | None, float | None]:
    joint_path, max_step = joint_motion_metrics(tuple(frame.q for frame in replay.frames))
    finite_conditions = [
        float(frame.condition_number)
        for frame in replay.frames
        if frame.condition_number is not None and math.isfinite(float(frame.condition_number))
    ]
    max_condition = max(finite_conditions) if finite_conditions else None
    finite_sigmas = [
        float(frame.min_singular_value)
        for frame in replay.frames
        if getattr(frame, "min_singular_value", None) is not None
        and math.isfinite(float(frame.min_singular_value))
    ]
    min_sigma = min(finite_sigmas) if finite_sigmas else None
    return joint_path, max_step, max_condition, min_sigma


def build_coarse_replay_frames(
    scenario: PickPlaceScenario,
    phases: tuple[Phase, ...],
    *,
    key_phase_names: tuple[str, ...] = ("home", "grasp", "lift", "transport", "place", "release"),
) -> tuple[ReplayFrame, ...]:
    full_frames = build_replay_frames(scenario, phases, transition_frames=2, hold_frames=1)
    selected: list[ReplayFrame] = []
    for phase_name in key_phase_names:
        candidates = [frame for frame in full_frames if frame.phase_name == phase_name]
        if candidates:
            selected.append(candidates[-1])
    return tuple(selected)


def select_best_candidate(evaluations: tuple[CandidateEvaluation, ...]) -> CandidateEvaluation:
    if not evaluations:
        raise ValueError("no grasp candidates to select from")
    return min(evaluations, key=lambda row: row.score_key())


def _evaluate_static_candidate(
    scenario: PickPlaceScenario,
    strategy: GraspStrategy,
) -> tuple[CandidateEvaluation, tuple[Phase, ...]]:
    phases = plan_pick_place(scenario, grasp_strategy=strategy)
    frames = build_replay_frames(scenario, phases)
    metrics = evaluate_replay(scenario, frames)
    success = bool(metrics["success"])
    return (
        CandidateEvaluation(
            strategy_name=strategy.name,
            success=success,
            max_position_error=0.0 if success else float("inf"),
            min_joint_margin=float("inf"),
            collision_count=0,
            path_length=path_length_for_phases(phases),
            failure_reasons=tuple(str(item) for item in metrics["failure_reasons"]),
            max_orientation_error=0.0,
        ),
        phases,
    )


def _evaluate_panda_candidate(
    scenario: PickPlaceScenario,
    strategy: GraspStrategy,
    *,
    panda_xml: Path,
    output_xml: Path,
    position_tolerance: float,
) -> tuple[CandidateEvaluation, tuple[Phase, ...]]:
    phases = plan_pick_place(scenario, grasp_strategy=strategy)
    frames = build_replay_frames(scenario, phases)
    scene = build_pick_place_scene(
        panda_xml=panda_xml,
        scenario=scenario,
        output_xml=output_xml,
    )
    replay = solve_joint_replay(
        scene.output_xml,
        frames,
        tolerance=position_tolerance,
        orientation_tolerance=float(strategy.orientation_tolerance),
    )
    collision_count = sum(int(frame.collision_count) for frame in replay.frames)
    margins = [float(frame.joint_margin) for frame in replay.frames if math.isfinite(float(frame.joint_margin))]
    min_margin = min(margins) if margins else 0.0
    joint_path, max_step, max_condition, min_sigma = replay_joint_motion_metrics(replay)
    success = bool(replay.success and collision_count == 0)
    return (
        CandidateEvaluation(
            strategy_name=strategy.name,
            success=success,
            max_position_error=float(replay.max_position_error),
            min_joint_margin=float(min_margin),
            collision_count=int(collision_count),
            path_length=path_length_for_phases(phases),
            failure_reasons=tuple(str(item) for item in replay.failure_reasons),
            max_orientation_error=float(replay.max_orientation_error),
            joint_path_length=joint_path,
            max_joint_step=max_step,
            max_condition_number=max_condition,
            min_sigma=min_sigma,
        ),
        phases,
    )


def _evaluate_coarse_panda_candidate(
    scenario: PickPlaceScenario,
    strategy: GraspStrategy,
    *,
    panda_xml: Path,
    output_xml: Path,
    position_tolerance: float,
) -> tuple[CandidateEvaluation, tuple[Phase, ...]]:
    phases = plan_pick_place(scenario, grasp_strategy=strategy)
    frames = build_coarse_replay_frames(scenario, phases)
    scene = build_pick_place_scene(
        panda_xml=panda_xml,
        scenario=scenario,
        output_xml=output_xml,
    )
    replay = solve_joint_replay(
        scene.output_xml,
        frames,
        tolerance=position_tolerance,
        orientation_tolerance=float(strategy.orientation_tolerance),
    )
    collision_count = sum(int(frame.collision_count) for frame in replay.frames)
    margins = [float(frame.joint_margin) for frame in replay.frames if math.isfinite(float(frame.joint_margin))]
    min_margin = min(margins) if margins else 0.0
    joint_path, max_step, max_condition, min_sigma = replay_joint_motion_metrics(replay)
    success = bool(replay.success and collision_count == 0)
    return (
        CandidateEvaluation(
            strategy_name=strategy.name,
            success=success,
            max_position_error=float(replay.max_position_error),
            min_joint_margin=float(min_margin),
            collision_count=int(collision_count),
            path_length=path_length_for_phases(phases),
            failure_reasons=tuple(str(item) for item in replay.failure_reasons),
            max_orientation_error=float(replay.max_orientation_error),
            joint_path_length=joint_path,
            max_joint_step=max_step,
            max_condition_number=max_condition,
            min_sigma=min_sigma,
        ),
        phases,
    )


def select_grasp_plan(
    scenario: PickPlaceScenario,
    *,
    panda_xml: Path | None = None,
    output_dir: Path | None = None,
    position_tolerance: float = 0.035,
    full_candidate_limit: int = 2,
) -> GraspPlanSelection:
    strategies = strategies_for_object(scenario.object_type)
    evaluated: list[tuple[CandidateEvaluation, tuple[Phase, ...], GraspStrategy]] = []
    if panda_xml is None:
        for strategy in strategies:
            evaluation, phases = _evaluate_static_candidate(scenario, strategy)
            evaluated.append((evaluation, phases, strategy))
        selected_evaluation = select_best_candidate(tuple(item[0] for item in evaluated))
        for evaluation, phases, strategy in evaluated:
            if evaluation.strategy_name == selected_evaluation.strategy_name:
                return GraspPlanSelection(
                    selected_strategy=strategy,
                    selected_evaluation=evaluation,
                    evaluations=tuple(item[0] for item in evaluated),
                    phases=phases,
                )
        raise RuntimeError("selected grasp candidate disappeared")

    if output_dir is None:
        raise ValueError("output_dir is required when panda_xml is provided")

    coarse_evaluated: list[tuple[CandidateEvaluation, tuple[Phase, ...], GraspStrategy]] = []
    for strategy in strategies:
        candidate_xml = output_dir / f"candidate_coarse_{strategy.name}.xml"
        evaluation, phases = _evaluate_coarse_panda_candidate(
            scenario,
            strategy,
            panda_xml=panda_xml,
            output_xml=candidate_xml,
            position_tolerance=position_tolerance,
        )
        coarse_evaluated.append((evaluation, phases, strategy))

    coarse_sorted = sorted(coarse_evaluated, key=lambda item: item[0].score_key())
    limit = max(1, min(int(full_candidate_limit), len(coarse_sorted)))
    for _, _, strategy in coarse_sorted[:limit]:
        candidate_xml = output_dir / f"candidate_full_{strategy.name}.xml"
        evaluation, phases = _evaluate_panda_candidate(
            scenario,
            strategy,
            panda_xml=panda_xml,
            output_xml=candidate_xml,
            position_tolerance=position_tolerance,
        )
        evaluated.append((evaluation, phases, strategy))

    selected_evaluation = select_best_candidate(tuple(item[0] for item in evaluated))
    for evaluation, phases, strategy in evaluated:
        if evaluation.strategy_name == selected_evaluation.strategy_name:
            return GraspPlanSelection(
                selected_strategy=strategy,
                selected_evaluation=evaluation,
                evaluations=tuple(item[0] for item in evaluated),
                phases=phases,
                coarse_evaluations=tuple(item[0] for item in coarse_evaluated),
            )
    raise RuntimeError("selected grasp candidate disappeared")
