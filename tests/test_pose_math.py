from __future__ import annotations

import math

import pytest

from morphtamp_x_v2.pose_math import (
    IDENTITY_QUAT,
    quat_angle_error,
    quat_normalize,
    quat_rotate_vector,
    quat_slerp,
)


def test_quat_angle_error_is_zero_for_identical_orientation():
    assert quat_angle_error(IDENTITY_QUAT, IDENTITY_QUAT) == pytest.approx(0.0)


def test_quat_angle_error_detects_half_turn():
    half_turn_z = (0.0, 0.0, 0.0, 1.0)

    assert quat_angle_error(IDENTITY_QUAT, half_turn_z) == pytest.approx(math.pi)


def test_quat_slerp_keeps_unit_length_and_endpoints():
    target = quat_normalize((0.0, 1.0, 0.0, 0.0))

    assert quat_slerp(IDENTITY_QUAT, target, 0.0) == pytest.approx(IDENTITY_QUAT)
    assert quat_slerp(IDENTITY_QUAT, target, 1.0) == pytest.approx(target)
    midpoint = quat_slerp(IDENTITY_QUAT, target, 0.5)

    assert sum(value * value for value in midpoint) == pytest.approx(1.0)
    assert quat_angle_error(IDENTITY_QUAT, midpoint) == pytest.approx(math.pi / 2.0)


def test_quat_rotate_vector_applies_right_handed_rotation_without_rescaling():
    ninety_z = quat_normalize((math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)))

    rotated = quat_rotate_vector(ninety_z, (2.0, 0.0, 0.0))

    assert rotated == pytest.approx((0.0, 2.0, 0.0), abs=1e-9)
