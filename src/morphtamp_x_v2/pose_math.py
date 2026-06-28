from __future__ import annotations

import math

import numpy as np


IDENTITY_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def quat_normalize(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    values = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        return IDENTITY_QUAT
    values = values / norm
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def quat_dot(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    return float(np.dot(np.asarray(first, dtype=float), np.asarray(second, dtype=float)))


def quat_conjugate(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = quat_normalize(quat)
    return (w, -x, -y, -z)


def quat_multiply(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = quat_normalize(first)
    w2, x2, y2, z2 = quat_normalize(second)
    return quat_normalize(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def quat_angle_error(
    current: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    dot = abs(quat_dot(quat_normalize(current), quat_normalize(target)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * math.acos(dot))


def quat_error_vector(
    current: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> np.ndarray:
    delta = quat_multiply(target, quat_conjugate(current))
    if delta[0] < 0.0:
        delta = tuple(-value for value in delta)
    return 2.0 * np.asarray(delta[1:4], dtype=float)


def quat_rotate_vector(
    quat: tuple[float, float, float, float],
    vector: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    q = np.asarray(quat_normalize(quat), dtype=float)
    v = np.asarray(vector, dtype=float)
    scalar = float(q[0])
    axis = q[1:4]
    return (
        2.0 * float(np.dot(axis, v)) * axis
        + (scalar * scalar - float(np.dot(axis, axis))) * v
        + 2.0 * scalar * np.cross(axis, v)
    )


def quat_slerp(
    start: tuple[float, float, float, float],
    end: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    q0 = np.asarray(quat_normalize(start), dtype=float)
    q1 = np.asarray(quat_normalize(end), dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(tuple(q0 + alpha * (q1 - q0)))
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta0 = math.sin(theta0)
    theta = theta0 * alpha
    scale0 = math.sin(theta0 - theta) / sin_theta0
    scale1 = math.sin(theta) / sin_theta0
    return quat_normalize(tuple(scale0 * q0 + scale1 * q1))
