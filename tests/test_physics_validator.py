from __future__ import annotations

from morphtamp_x_v2.physics_validator import _contact_is_gripper_object, _contact_is_obstacle_violation


def test_obstacle_contact_between_robot_and_obstacle_is_hard_violation():
    assert (
        _contact_is_obstacle_violation(
            geom1_name="v2_obstacle",
            body1_name="world",
            geom2_name="finger_geom",
            body2_name="right_finger",
        )
        is True
    )


def test_obstacle_contact_with_table_is_not_robot_trajectory_violation():
    assert (
        _contact_is_obstacle_violation(
            geom1_name="v2_obstacle",
            body1_name="world",
            geom2_name="v2_table",
            body2_name="world",
        )
        is False
    )


def test_support_contact_between_robot_and_support_is_hard_violation():
    assert (
        _contact_is_obstacle_violation(
            geom1_name="v2_support_0",
            body1_name="world",
            geom2_name="finger_geom",
            body2_name="right_finger",
        )
        is True
    )


def test_support_contact_with_object_is_allowed_physical_support():
    assert (
        _contact_is_obstacle_violation(
            geom1_name="v2_support_0",
            body1_name="world",
            geom2_name="v2_object_geom",
            body2_name="v2_cube",
        )
        is False
    )


def test_gripper_object_contact_detects_finger_to_manipulated_object():
    assert (
        _contact_is_gripper_object(
            geom1_name="v2_object_geom",
            body1_name="v2_cube",
            geom2_name="finger_pad_collision",
            body2_name="right_finger",
        )
        is True
    )


def test_gripper_object_contact_ignores_table_contact():
    assert (
        _contact_is_gripper_object(
            geom1_name="v2_object_geom",
            body1_name="v2_cube",
            geom2_name="v2_table",
            body2_name="world",
        )
        is False
    )


def test_gripper_object_contact_ignores_support_contact():
    assert (
        _contact_is_gripper_object(
            geom1_name="v2_object_geom",
            body1_name="v2_cube",
            geom2_name="v2_support_0",
            body2_name="world",
        )
        is False
    )
