from __future__ import annotations

import pytest

from morphtamp_x_v2.models import PickPlaceScenario
from morphtamp_x_v2.planner import plan_pick_place
from morphtamp_x_v2.grasp_strategies import strategies_for_object
from morphtamp_x_v2.tasks import make_scenario


def test_plan_pick_place_has_realistic_grasp_order():
    scenario = PickPlaceScenario()

    phases = plan_pick_place(scenario)
    names = [phase.name for phase in phases]

    assert names == [
        "home",
        "pregrasp",
        "grasp",
        "close_gripper",
        "attach",
        "lift",
        "transport",
        "descend",
        "place",
        "open_gripper",
        "release",
        "return_home",
    ]
    assert phases[names.index("grasp")].object_attached is False
    assert phases[names.index("close_gripper")].object_attached is False
    assert phases[names.index("attach")].object_attached is True
    assert phases[names.index("release")].object_attached is False


def test_plan_pick_place_places_grasp_and_place_at_cube_centers():
    scenario = PickPlaceScenario(
        cube_start=(0.40, -0.10, 0.06),
        place_target=(0.52, 0.14, 0.06),
    )

    phases = {phase.name: phase for phase in plan_pick_place(scenario)}

    assert phases["grasp"].tcp_position == pytest.approx(scenario.cube_start)
    assert phases["place"].tcp_position == pytest.approx(scenario.place_target)
    assert phases["pregrasp"].tcp_position[2] > phases["grasp"].tcp_position[2]
    assert phases["transport"].tcp_position[2] >= phases["lift"].tcp_position[2]


def test_plan_pick_place_lifts_above_obstacle_with_wrist_clearance():
    scenario = PickPlaceScenario(
        cube_start=(0.40, -0.14, 0.20),
        place_target=(0.52, 0.14, 0.20),
        lift_height=0.10,
        obstacle_center=(0.46, 0.0, 0.30),
        obstacle_size=(0.02, 0.18, 0.10),
    )

    phases = {phase.name: phase for phase in plan_pick_place(scenario)}
    obstacle_top = scenario.obstacle_center[2] + scenario.obstacle_size[2]

    assert phases["lift"].tcp_position[2] >= obstacle_top + 0.26 - 1e-12
    assert phases["transport"].tcp_position[2] >= obstacle_top + 0.26 - 1e-12


def test_plan_pick_place_adds_side_bypass_for_wall_like_obstacle():
    scenario = make_scenario(object_type="cube", task_name="around_wall")

    phases = plan_pick_place(scenario)
    names = [phase.name for phase in phases]

    assert "bypass_obstacle" in names
    bypass = phases[names.index("bypass_obstacle")]
    assert scenario.obstacle_center is not None
    assert scenario.obstacle_size is not None
    assert abs(bypass.tcp_position[0] - scenario.obstacle_center[0]) > scenario.obstacle_size[0] + 0.08


def test_library_scenario_aligns_side_pinch_tcp_with_object_center():
    scenario = make_scenario(object_type="cube", task_name="tabletop_easy")
    phases = {phase.name: phase for phase in plan_pick_place(scenario)}

    assert phases["grasp"].tcp_position == pytest.approx(scenario.cube_start)
    assert phases["place"].tcp_position == pytest.approx(scenario.place_target)


def test_directional_tasks_insert_visible_motion_style_waypoints():
    extend = [phase.name for phase in plan_pick_place(make_scenario(object_type="cube", task_name="near_to_far_reach"))]
    retract = [phase.name for phase in plan_pick_place(make_scenario(object_type="cube", task_name="far_to_near_retract"))]
    folded = [phase.name for phase in plan_pick_place(make_scenario(object_type="cube", task_name="folded_transfer"))]
    compound = [phase.name for phase in plan_pick_place(make_scenario(object_type="cube", task_name="compound_shelf_barrier"))]

    assert "extend_far" in extend
    assert "retract_near" in retract
    assert folded.index("fold_near_body") < folded.index("extend_far") < folded.index("transport")
    assert "high_clearance" in compound
    assert "extend_far" in compound


def test_attached_phases_keep_selected_grasp_orientation():
    scenario = make_scenario(object_type="plate", task_name="tabletop_easy")
    strategy = next(
        item for item in strategies_for_object("plate")
        if item.name == "plate_edge_pinch_6d"
    )

    phases = {phase.name: phase for phase in plan_pick_place(scenario, grasp_strategy=strategy)}

    for name in [
        "attach",
        "lift",
        "transport",
        "descend",
        "place",
        "open_gripper",
        "release",
    ]:
        assert phases[name].grasp_strategy == strategy.name
        assert phases[name].tcp_quat == strategy.tcp_quat
        assert phases[name].orientation_required is True

    assert phases["return_home"].grasp_strategy is None
