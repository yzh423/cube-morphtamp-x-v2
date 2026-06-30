from __future__ import annotations

from morphtamp_x_v2.physics_validator import (
    _build_dynamics_evidence,
    _contact_is_gripper_object,
    _contact_is_obstacle_violation,
    _gripper_contact_side,
)


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


def test_gripper_contact_side_distinguishes_left_right_and_palm():
    assert _gripper_contact_side(
        geom1_name="v2_object_geom",
        body1_name="v2_cube",
        geom2_name="left_finger",
        body2_name="left_finger",
    ) == "left_finger"
    assert _gripper_contact_side(
        geom1_name="right_finger",
        body1_name="right_finger",
        geom2_name="v2_object_geom",
        body2_name="v2_cube",
    ) == "right_finger"
    assert _gripper_contact_side(
        geom1_name="hand_collision",
        body1_name="hand",
        geom2_name="v2_object_geom",
        body2_name="v2_cube",
    ) == "palm_or_hand"


def test_build_dynamics_evidence_accepts_weld_tracking_release_and_contact():
    evidence = _build_dynamics_evidence(
        final_error=0.01,
        position_tolerance=0.03,
        weld_frames=12,
        max_weld_tracking_error=0.004,
        weld_tracking_tolerance=0.02,
        obstacle_collisions=0,
        gripper_object_contacts=5,
        gripper_object_contact_sides=("left_finger", "right_finger"),
        gripper_object_max_penetration=0.001,
        table_contacts=2,
        settle_steps=80,
    )

    assert evidence["success"] is True
    assert evidence["evidence_scope"].startswith("MuJoCo equality-weld")
    assert evidence["failure_reasons"] == []
    assert evidence["checks"]["weld_retention"]["passed"] is True
    assert evidence["checks"]["grasp_contact"]["passed"] is True
    assert evidence["checks"]["two_sided_grasp_contact"]["passed"] is True
    assert evidence["checks"]["release_settle"]["passed"] is True
    assert evidence["checks"]["obstacle_clearance"]["passed"] is True


def test_build_dynamics_evidence_rejects_palm_only_grasp_contact():
    evidence = _build_dynamics_evidence(
        final_error=0.002,
        position_tolerance=0.03,
        weld_frames=10,
        max_weld_tracking_error=0.01,
        weld_tracking_tolerance=0.02,
        obstacle_collisions=0,
        gripper_object_contacts=4,
        gripper_object_contact_sides=("palm_or_hand",),
        gripper_object_max_penetration=0.001,
        max_gripper_penetration=0.004,
        require_two_sided_grasp=True,
        table_contacts=2,
        settle_steps=80,
    )

    assert evidence["success"] is False
    assert "two_sided_grasp_contact" in evidence["failure_reasons"]
    assert evidence["checks"]["grasp_contact"]["passed"] is True
    assert evidence["checks"]["two_sided_grasp_contact"]["passed"] is False
    assert evidence["checks"]["two_sided_grasp_contact"]["required_for_success"] is True


def test_build_dynamics_evidence_rejects_failed_weld_release_and_contact():
    evidence = _build_dynamics_evidence(
        final_error=0.08,
        position_tolerance=0.03,
        weld_frames=0,
        max_weld_tracking_error=0.05,
        weld_tracking_tolerance=0.02,
        obstacle_collisions=2,
        gripper_object_contacts=0,
        gripper_object_max_penetration=0.0,
        max_gripper_penetration=0.004,
        table_contacts=0,
        settle_steps=80,
    )

    assert evidence["success"] is False
    assert evidence["failure_reasons"] == [
        "weld_retention",
        "grasp_contact",
        "release_settle",
        "obstacle_clearance",
    ]
    assert evidence["checks"]["weld_retention"]["max_weld_tracking_error"] == 0.05
    assert evidence["checks"]["release_settle"]["final_error"] == 0.08


def test_build_dynamics_evidence_rejects_excessive_gripper_penetration():
    evidence = _build_dynamics_evidence(
        final_error=0.002,
        position_tolerance=0.03,
        weld_frames=10,
        max_weld_tracking_error=0.01,
        weld_tracking_tolerance=0.02,
        obstacle_collisions=0,
        gripper_object_contacts=3,
        gripper_object_max_penetration=0.009,
        max_gripper_penetration=0.004,
        table_contacts=2,
        settle_steps=80,
    )

    assert evidence["success"] is False
    assert "gripper_penetration" in evidence["failure_reasons"]
    assert evidence["checks"]["gripper_penetration"]["passed"] is False
    assert evidence["checks"]["gripper_penetration"]["max_allowed_penetration"] == 0.004


def test_build_dynamics_evidence_reports_gripper_penetration_by_contact_side():
    evidence = _build_dynamics_evidence(
        final_error=0.001,
        position_tolerance=0.01,
        weld_frames=5,
        max_weld_tracking_error=0.001,
        weld_tracking_tolerance=0.01,
        obstacle_collisions=0,
        gripper_object_contacts=4,
        gripper_object_contact_sides=("left_finger", "palm_or_hand", "right_finger"),
        gripper_object_max_penetration=0.006,
        gripper_object_max_penetration_by_side={
            "left_finger": 0.002,
            "right_finger": 0.003,
            "palm_or_hand": 0.006,
        },
        max_gripper_penetration=0.004,
        table_contacts=2,
        settle_steps=20,
    )

    check = evidence["checks"]["gripper_penetration"]
    assert check["gripper_object_max_penetration_by_side"] == {
        "left_finger": 0.002,
        "palm_or_hand": 0.006,
        "right_finger": 0.003,
    }
    assert check["dominant_penetration_side"] == "palm_or_hand"


def test_build_dynamics_evidence_treats_reference_tracking_as_diagnostic():
    evidence = _build_dynamics_evidence(
        final_error=0.002,
        position_tolerance=0.03,
        weld_frames=10,
        max_weld_tracking_error=0.08,
        weld_tracking_tolerance=0.02,
        obstacle_collisions=0,
        gripper_object_contacts=3,
        gripper_object_max_penetration=0.001,
        max_gripper_penetration=0.004,
        table_contacts=2,
        settle_steps=80,
    )

    assert evidence["success"] is True
    assert evidence["checks"]["weld_retention"]["passed"] is True
    assert evidence["checks"]["reference_tracking"]["passed"] is False
    assert evidence["checks"]["reference_tracking"]["diagnostic_only"] is True
