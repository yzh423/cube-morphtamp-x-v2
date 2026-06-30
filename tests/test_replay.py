from __future__ import annotations

import pytest

from morphtamp_x_v2.models import Phase, PickPlaceScenario
from morphtamp_x_v2.pose_math import quat_angle_error
from morphtamp_x_v2.planner import plan_pick_place
from morphtamp_x_v2.replay import _smooth, build_replay_frames
from morphtamp_x_v2.tasks import make_scenario


def test_replay_keeps_object_static_before_attach():
    scenario = PickPlaceScenario()
    frames = build_replay_frames(scenario, plan_pick_place(scenario), transition_frames=5, hold_frames=1)

    before_attach = [
        frame
        for frame in frames
        if frame.phase_name in {"home", "pregrasp", "grasp", "close_gripper"}
    ]

    assert before_attach
    assert all(frame.object_position == pytest.approx(scenario.cube_start) for frame in before_attach)
    assert all(not frame.weld_active for frame in before_attach)


def test_replay_carries_object_rigidly_between_attach_and_open_gripper():
    scenario = PickPlaceScenario(cube_start=(0.42, -0.12, 0.06), place_target=(0.50, 0.18, 0.06))
    frames = build_replay_frames(scenario, plan_pick_place(scenario), transition_frames=5, hold_frames=1)

    carried = [
        frame
        for frame in frames
        if frame.phase_name in {"attach", "lift", "transport", "descend", "place", "open_gripper"}
    ]

    assert carried
    assert all(frame.object_attached for frame in carried)
    assert all(frame.weld_active for frame in carried)
    assert carried[-1].object_position == pytest.approx(scenario.place_target)


def test_replay_releases_only_after_gripper_opens():
    scenario = PickPlaceScenario(open_width=0.08, closed_width=0.03)
    frames = build_replay_frames(scenario, plan_pick_place(scenario), transition_frames=7, hold_frames=1)

    open_frames = [frame for frame in frames if frame.phase_name == "open_gripper"]
    release_frames = [frame for frame in frames if frame.phase_name == "release"]

    assert open_frames[0].weld_active is True
    assert open_frames[-1].weld_active is True
    assert open_frames[-1].gripper_width == pytest.approx(scenario.open_width)
    assert release_frames[0].weld_active is False
    assert release_frames[-1].object_position == pytest.approx(scenario.place_target)


def test_replay_keeps_object_orientation_continuous_through_attach_and_release():
    scenario = PickPlaceScenario()
    frames = build_replay_frames(scenario, plan_pick_place(scenario), transition_frames=5, hold_frames=1)

    quats = [frame.object_quat for frame in frames]

    assert all(quat == pytest.approx((1.0, 0.0, 0.0, 0.0)) for quat in quats)
    attach_quat = next(frame.object_quat for frame in frames if frame.phase_name == "attach")
    release_quat = next(frame.object_quat for frame in frames if frame.phase_name == "release")
    assert attach_quat == pytest.approx(release_quat)


def test_default_replay_is_slow_enough_for_visual_inspection():
    scenario = PickPlaceScenario()
    frames = build_replay_frames(scenario, plan_pick_place(scenario))

    assert len(frames) >= 250


def test_replay_smoother_has_zero_slope_like_endpoints():
    epsilon = 1e-3

    assert _smooth(0.0) == pytest.approx(0.0)
    assert _smooth(1.0) == pytest.approx(1.0)
    assert _smooth(epsilon) < 4.0 * epsilon
    assert 1.0 - _smooth(1.0 - epsilon) < 4.0 * epsilon


def test_replay_keeps_object_attached_during_motion_style_waypoints():
    scenario = make_scenario(object_type="sphere", task_name="folded_transfer")
    frames = build_replay_frames(scenario, plan_pick_place(scenario), transition_frames=5, hold_frames=1)

    waypoint_frames = [
        frame
        for frame in frames
        if frame.phase_name in {"fold_near_body", "extend_far", "bypass_obstacle"}
    ]

    assert waypoint_frames
    assert all(frame.object_attached for frame in waypoint_frames)
    assert all(frame.weld_active for frame in waypoint_frames)
    assert any(frame.object_position[2] > scenario.cube_start[2] + 0.15 for frame in waypoint_frames)


def test_replay_interpolates_tcp_orientation_from_phase_quaternions():
    scenario = make_scenario(object_type="plate", task_name="tabletop_easy")
    phases = plan_pick_place(scenario)
    frames = build_replay_frames(scenario, phases, transition_frames=5, hold_frames=1)

    assert all(len(frame.tcp_quat) == 4 for frame in frames)
    grasp_phase = next(phase for phase in phases if phase.name == "grasp")
    grasp_frame = next(frame for frame in frames if frame.phase_name == "grasp" and frame.progress == pytest.approx(1.0))

    assert quat_angle_error(grasp_frame.tcp_quat, grasp_phase.tcp_quat) < 1e-9


def test_replay_carries_object_with_tcp_relative_rigid_transform():
    ninety_z = (0.7071067811865476, 0.0, 0.0, 0.7071067811865475)
    minus_ninety_z = (0.7071067811865476, 0.0, 0.0, -0.7071067811865475)
    identity = (1.0, 0.0, 0.0, 0.0)
    scenario = PickPlaceScenario(
        cube_start=(1.0, 0.0, 0.0),
        place_target=(0.0, 1.0, 0.0),
    )
    phases = (
        Phase("home", (0.0, 0.0, 0.0), 0.08, False, "home", tcp_quat=identity),
        Phase("grasp", (0.0, 0.0, 0.0), 0.08, False, "grasp", tcp_quat=ninety_z),
        Phase("attach", (0.0, 0.0, 0.0), 0.03, True, "attach", tcp_quat=ninety_z),
        Phase("transport", (0.0, 0.0, 0.0), 0.03, True, "transport", tcp_quat=identity),
        Phase("open_gripper", (0.0, 0.0, 0.0), 0.08, True, "open", tcp_quat=identity),
        Phase("release", (0.0, 0.0, 0.0), 0.08, False, "release", tcp_quat=identity),
    )

    frames = build_replay_frames(scenario, phases, transition_frames=2, hold_frames=1)
    attach_frame = next(frame for frame in frames if frame.phase_name == "attach" and frame.progress == pytest.approx(1.0))
    transport_frame = next(frame for frame in frames if frame.phase_name == "transport" and frame.progress == pytest.approx(1.0))
    release_frame = next(frame for frame in frames if frame.phase_name == "release" and frame.progress == pytest.approx(1.0))

    assert attach_frame.object_position == pytest.approx(scenario.cube_start)
    assert attach_frame.object_quat == pytest.approx(identity)
    assert transport_frame.object_position == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)
    assert quat_angle_error(transport_frame.object_quat, minus_ninety_z) < 1e-9
    assert release_frame.object_position == pytest.approx(transport_frame.object_position)
    assert quat_angle_error(release_frame.object_quat, transport_frame.object_quat) < 1e-9


def test_replay_frames_preserve_phase_orientation_requirements():
    scenario = PickPlaceScenario()
    frames = build_replay_frames(
        scenario,
        (
            Phase("home", scenario.cube_start, 0.08, False, "home"),
            Phase(
                "attach",
                scenario.cube_start,
                0.04,
                True,
                "attach",
                orientation_mode="hold_attach",
                orientation_weight=0.08,
            ),
            Phase(
                "transport",
                scenario.place_target,
                0.04,
                True,
                "transport",
                orientation_mode="hold_attach",
                orientation_weight=0.08,
            ),
        ),
        transition_frames=3,
        hold_frames=1,
    )

    transport = next(
        frame
        for frame in frames
        if frame.phase_name == "transport" and frame.progress == pytest.approx(1.0)
    )

    assert transport.orientation_mode == "hold_attach"
    assert transport.orientation_weight > 0.0
    assert transport.orientation_required is False
