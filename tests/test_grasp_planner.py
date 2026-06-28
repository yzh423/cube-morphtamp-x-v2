from __future__ import annotations

import pytest

import morphtamp_x_v2.grasp_planner as grasp_planner
from morphtamp_x_v2.grasp_planner import (
    CandidateEvaluation,
    build_coarse_replay_frames,
    path_length_for_phases,
    select_best_candidate,
    select_grasp_plan,
)
from morphtamp_x_v2.grasp_strategies import strategies_for_object
from morphtamp_x_v2.planner import plan_pick_place
from morphtamp_x_v2.tasks import make_scenario


def test_path_length_for_phases_is_positive_for_different_candidates():
    scenario = make_scenario(object_type="cube", task_name="tabletop_easy")
    first, second = strategies_for_object("cube")[:2]

    first_length = path_length_for_phases(plan_pick_place(scenario, grasp_strategy=first))
    second_length = path_length_for_phases(plan_pick_place(scenario, grasp_strategy=second))

    assert first_length > 0.0
    assert second_length > 0.0
    first_grasp = {phase.name: phase for phase in plan_pick_place(scenario, grasp_strategy=first)}["grasp"]
    second_grasp = {phase.name: phase for phase in plan_pick_place(scenario, grasp_strategy=second)}["grasp"]
    assert second_grasp.tcp_position != pytest.approx(first_grasp.tcp_position)


def test_build_coarse_replay_frames_keeps_key_phase_endpoints_only():
    scenario = make_scenario(object_type="cube", task_name="tabletop_easy")
    phases = plan_pick_place(scenario)

    frames = build_coarse_replay_frames(
        scenario,
        phases,
        key_phase_names=("home", "grasp", "lift", "place"),
    )

    assert tuple(frame.phase_name for frame in frames) == ("home", "grasp", "lift", "place")
    assert all(frame.progress == pytest.approx(1.0) for frame in frames)
    phase_by_name = {phase.name: phase for phase in phases}
    assert frames[1].tcp_position == pytest.approx(phase_by_name["grasp"].tcp_position)
    assert frames[-1].object_position == pytest.approx(scenario.place_target)


def test_select_best_candidate_prefers_success_then_low_error_then_margin_and_path():
    worse_success = CandidateEvaluation(
        strategy_name="bad",
        success=False,
        max_position_error=0.001,
        min_joint_margin=2.0,
        collision_count=0,
        path_length=0.1,
        failure_reasons=("failed",),
    )
    high_error = CandidateEvaluation(
        strategy_name="high_error",
        success=True,
        max_position_error=0.03,
        min_joint_margin=1.0,
        collision_count=0,
        path_length=0.1,
        failure_reasons=(),
    )
    best = CandidateEvaluation(
        strategy_name="best",
        success=True,
        max_position_error=0.01,
        min_joint_margin=0.5,
        collision_count=0,
        path_length=0.2,
        failure_reasons=(),
    )

    selected = select_best_candidate((worse_success, high_error, best))

    assert selected.strategy_name == "best"


def test_select_best_candidate_treats_tiny_ik_error_differences_as_equivalent():
    lower_margin = CandidateEvaluation(
        strategy_name="lower_margin",
        success=True,
        max_position_error=2.0e-10,
        min_joint_margin=0.4,
        collision_count=0,
        path_length=0.1,
        failure_reasons=(),
    )
    higher_margin = CandidateEvaluation(
        strategy_name="higher_margin",
        success=True,
        max_position_error=3.0e-10,
        min_joint_margin=0.8,
        collision_count=0,
        path_length=0.1,
        failure_reasons=(),
    )

    selected = select_best_candidate((lower_margin, higher_margin))

    assert selected.strategy_name == "higher_margin"


def test_select_best_candidate_prefers_lower_orientation_error_after_position_error():
    high_orientation_error = CandidateEvaluation(
        strategy_name="high_orientation_error",
        success=True,
        max_position_error=1.0e-6,
        max_orientation_error=0.40,
        min_joint_margin=0.8,
        collision_count=0,
        path_length=0.1,
        failure_reasons=(),
    )
    low_orientation_error = CandidateEvaluation(
        strategy_name="low_orientation_error",
        success=True,
        max_position_error=1.0e-6,
        max_orientation_error=0.05,
        min_joint_margin=0.4,
        collision_count=0,
        path_length=0.3,
        failure_reasons=(),
    )

    selected = select_best_candidate((high_orientation_error, low_orientation_error))

    assert selected.strategy_name == "low_orientation_error"


def test_select_best_candidate_prefers_smoother_joint_motion_when_feasible():
    jerky = CandidateEvaluation(
        strategy_name="jerky",
        success=True,
        max_position_error=1.0e-6,
        max_orientation_error=1.0e-6,
        min_joint_margin=0.8,
        collision_count=0,
        path_length=0.1,
        joint_path_length=3.0,
        max_joint_step=1.2,
        failure_reasons=(),
    )
    smooth = CandidateEvaluation(
        strategy_name="smooth",
        success=True,
        max_position_error=1.0e-6,
        max_orientation_error=1.0e-6,
        min_joint_margin=0.8,
        collision_count=0,
        path_length=0.2,
        joint_path_length=1.0,
        max_joint_step=0.3,
        failure_reasons=(),
    )

    selected = select_best_candidate((jerky, smooth))

    assert selected.strategy_name == "smooth"


def test_select_grasp_plan_runs_full_panda_only_for_coarse_top_k(tmp_path, monkeypatch):
    scenario = make_scenario(object_type="cube", task_name="tabletop_easy")
    strategies = strategies_for_object("cube")
    preferred = strategies[-1].name
    coarse_calls = []
    full_calls = []

    def fake_coarse_candidate(scenario, strategy, *, panda_xml, output_xml, position_tolerance):
        coarse_calls.append(strategy.name)
        phases = plan_pick_place(scenario, grasp_strategy=strategy)
        return (
            CandidateEvaluation(
                strategy_name=strategy.name,
                success=True,
                max_position_error=0.001 if strategy.name == preferred else 0.1,
                min_joint_margin=0.5,
                collision_count=0,
                path_length=path_length_for_phases(phases),
                failure_reasons=(),
            ),
            phases,
        )

    def fake_full_candidate(scenario, strategy, *, panda_xml, output_xml, position_tolerance):
        full_calls.append(strategy.name)
        phases = plan_pick_place(scenario, grasp_strategy=strategy)
        return (
            CandidateEvaluation(
                strategy_name=strategy.name,
                success=True,
                max_position_error=0.0,
                min_joint_margin=1.0,
                collision_count=0,
                path_length=path_length_for_phases(phases),
                failure_reasons=(),
            ),
            phases,
        )

    monkeypatch.setattr(grasp_planner, "_evaluate_coarse_panda_candidate", fake_coarse_candidate)
    monkeypatch.setattr(grasp_planner, "_evaluate_panda_candidate", fake_full_candidate)

    selection = select_grasp_plan(
        scenario,
        panda_xml=tmp_path / "panda.xml",
        output_dir=tmp_path,
        full_candidate_limit=1,
    )

    assert tuple(coarse_calls) == tuple(strategy.name for strategy in strategies)
    assert tuple(full_calls) == (preferred,)
    assert selection.selected_strategy.name == preferred
    assert len(selection.coarse_evaluations) == len(strategies)
    assert len(selection.evaluations) == 1
    assert selection.to_json_dict()["coarse_candidate_evaluations"]


def test_candidate_evaluation_serializes_without_nan_or_infinity():
    row = CandidateEvaluation(
        strategy_name="candidate",
        success=False,
        max_position_error=float("inf"),
        max_orientation_error=float("inf"),
        min_joint_margin=float("inf"),
        collision_count=4,
        path_length=1.0,
        failure_reasons=("ik",),
    ).to_json_dict()

    assert row["max_position_error"] is None
    assert row["max_orientation_error"] is None
    assert row["min_joint_margin"] is None
    assert row["joint_path_length"] == 0.0
    assert row["max_joint_step"] == 0.0
    assert row["min_sigma"] is None
    assert row["collision_count"] == 4
