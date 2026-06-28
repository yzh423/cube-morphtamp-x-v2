from __future__ import annotations

import numpy as np

from .models import Phase, PickPlaceScenario, ReplayFrame
from .pose_math import quat_conjugate, quat_multiply, quat_rotate_vector, quat_slerp


def _tuple3(vector: np.ndarray) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _smooth(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * alpha * (10.0 - 15.0 * alpha + 6.0 * alpha * alpha)


def _lerp(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    return _tuple3(np.asarray(start, dtype=float) * (1.0 - alpha) + np.asarray(end, dtype=float) * alpha)


def build_replay_frames(
    scenario: PickPlaceScenario,
    phases: tuple[Phase, ...],
    *,
    transition_frames: int = 24,
    hold_frames: int = 6,
) -> tuple[ReplayFrame, ...]:
    if not phases:
        return ()
    frames: list[ReplayFrame] = []
    object_position = scenario.cube_start
    object_quat = (1.0, 0.0, 0.0, 0.0)
    attach_phase = next(phase for phase in phases if phase.name == "attach")
    attach_tcp = np.asarray(attach_phase.tcp_position, dtype=float)
    attach_quat = attach_phase.tcp_quat
    object_quat_at_attach = object_quat
    carry_offset_local = quat_rotate_vector(
        quat_conjugate(attach_quat),
        np.asarray(scenario.cube_start, dtype=float) - attach_tcp,
    )
    carry_quat_local = quat_multiply(quat_conjugate(attach_quat), object_quat_at_attach)

    previous_tcp = phases[0].tcp_position
    previous_quat = phases[0].tcp_quat
    previous_width = phases[0].gripper_width
    for phase_index, phase in enumerate(phases):
        count = hold_frames if phase_index == 0 else transition_frames
        for index in range(max(count, 1)):
            raw = 1.0 if count <= 1 else index / float(count - 1)
            alpha = _smooth(raw)
            tcp = _lerp(previous_tcp, phase.tcp_position, alpha)
            tcp_quat = quat_slerp(previous_quat, phase.tcp_quat, alpha)
            width = float(previous_width * (1.0 - alpha) + phase.gripper_width * alpha)

            if phase.object_attached:
                object_quat = quat_multiply(tcp_quat, carry_quat_local)
                object_position = _tuple3(
                    np.asarray(tcp, dtype=float)
                    + quat_rotate_vector(tcp_quat, carry_offset_local)
                )
                attached = True
                weld = True
            elif phase.name in {"home", "pregrasp", "grasp", "close_gripper"}:
                object_position = scenario.cube_start
                object_quat = object_quat_at_attach
                attached = False
                weld = False
            else:
                attached = False
                weld = False

            frames.append(
                ReplayFrame(
                    phase_name=phase.name,
                    tcp_position=tcp,
                    tcp_quat=tcp_quat,
                    object_position=object_position,
                    object_quat=object_quat,
                    gripper_width=width,
                    object_attached=attached,
                    weld_active=weld,
                    progress=float(raw),
                )
            )
        previous_tcp = phase.tcp_position
        previous_quat = phase.tcp_quat
        previous_width = phase.gripper_width
    return tuple(frames)
