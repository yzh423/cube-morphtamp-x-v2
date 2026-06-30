from __future__ import annotations

import pytest

from morphtamp_x_v2.grasp_strategies import strategies_for_object, strategy_for_object
from morphtamp_x_v2.objects import OBJECT_TYPES
from morphtamp_x_v2.planner import plan_pick_place
from morphtamp_x_v2.tasks import make_scenario


def test_every_object_type_has_a_named_grasp_strategy():
    for object_type in OBJECT_TYPES:
        strategy = strategy_for_object(object_type)

        assert strategy.name
        assert strategy.approach_axis in {"top", "side_y", "side_x", "edge"}
        assert len(strategy.tcp_offset) == 3
        assert len(strategy.tcp_quat) == 4
        assert strategy.hold_orientation in {"upright", "free", "edge_stable"}


@pytest.mark.parametrize("object_type", ["cube", "sphere", "mug_proxy", "plate", "capsule"])
def test_objects_expose_multiple_candidate_grasp_strategies(object_type):
    strategies = strategies_for_object(object_type)

    assert len(strategies) >= 2
    assert len({strategy.name for strategy in strategies}) == len(strategies)
    assert strategies[0] == strategy_for_object(object_type)


def test_planner_can_generate_different_paths_for_explicit_grasp_candidates():
    scenario = make_scenario(object_type="cube", task_name="tabletop_easy")
    strategies = strategies_for_object("cube")
    center = next(strategy for strategy in strategies if strategy.name == "box_side_pinch")
    top = next(strategy for strategy in strategies if strategy.name == "box_top_center_pinch")

    center_phases = {phase.name: phase for phase in plan_pick_place(scenario, grasp_strategy=center)}
    top_phases = {phase.name: phase for phase in plan_pick_place(scenario, grasp_strategy=top)}

    assert center_phases["grasp"].tcp_position == pytest.approx(scenario.cube_start)
    assert top_phases["grasp"].tcp_position != pytest.approx(center_phases["grasp"].tcp_position)
    assert top_phases["grasp"].grasp_strategy == top.name


def test_cube_exposes_lateral_calibration_grasp_candidates():
    strategies = {strategy.name: strategy for strategy in strategies_for_object("cube")}

    assert strategies["box_side_pinch_y_neg"].tcp_offset == pytest.approx((0.0, -0.004, 0.0))
    assert strategies["box_side_pinch_y_pos"].tcp_offset == pytest.approx((0.0, 0.004, 0.0))


def test_orientation_required_strategy_is_written_to_grasp_phases():
    scenario = make_scenario(object_type="plate", task_name="tabletop_easy")
    edge = next(strategy for strategy in strategies_for_object("plate") if strategy.requires_orientation)

    phases = {phase.name: phase for phase in plan_pick_place(scenario, grasp_strategy=edge)}

    assert phases["grasp"].tcp_quat == pytest.approx(edge.tcp_quat)
    assert phases["grasp"].orientation_required is True
    assert phases["grasp"].orientation_tolerance == pytest.approx(edge.orientation_tolerance)


@pytest.mark.parametrize(
    ("object_type", "expected_strategy"),
    [
        ("cube", "box_side_pinch"),
        ("sphere", "spherical_center_pinch"),
        ("plate", "plate_center_stabilized_pinch"),
        ("mug_proxy", "cup_body_side_pinch"),
        ("capsule", "long_axis_side_pinch"),
    ],
)
def test_planner_records_object_specific_grasp_strategy(object_type, expected_strategy):
    scenario = make_scenario(object_type=object_type, task_name="tabletop_easy")

    phases = {phase.name: phase for phase in plan_pick_place(scenario)}

    assert phases["grasp"].grasp_strategy == expected_strategy
    assert phases["close_gripper"].grasp_strategy == expected_strategy
    assert phases["attach"].grasp_strategy == expected_strategy


def test_plate_uses_edge_offset_instead_of_cube_center_grasp():
    cube = make_scenario(object_type="cube", task_name="tabletop_easy")
    plate = make_scenario(object_type="plate", task_name="tabletop_easy")

    cube_grasp = {phase.name: phase for phase in plan_pick_place(cube)}["grasp"]
    plate_grasp = {phase.name: phase for phase in plan_pick_place(plate)}["grasp"]

    assert cube_grasp.tcp_position == pytest.approx(cube.cube_start)
    assert plate_grasp.tcp_position == pytest.approx(plate.cube_start)
    assert plate_grasp.grasp_strategy == "plate_center_stabilized_pinch"


def test_plate_shelf_place_keeps_object_center_on_shelf_target_until_orientation_ik_exists():
    scenario = make_scenario(object_type="plate", task_name="shelf_place")

    phases = {phase.name: phase for phase in plan_pick_place(scenario)}

    assert phases["grasp"].tcp_position == pytest.approx(scenario.cube_start)
    assert phases["place"].tcp_position == pytest.approx(scenario.place_target)
    assert phases["place"].tcp_position[2] == pytest.approx(0.22 + scenario.object_size[1])
