from __future__ import annotations

import numpy as np
import pytest

from morphtamp_x_v2.viewer import _apply_replay_frame
from morphtamp_x_v2.viewer import _advance_frame_cursor
from morphtamp_x_v2.viewer import _frame_delay
from morphtamp_x_v2.viewer import _set_mocap_pose


class FakeData:
    def __init__(self) -> None:
        self.qpos = np.zeros(16, dtype=float)
        self.qvel = np.ones(16, dtype=float)
        self.eq_active = np.zeros(1, dtype=int)
        self.mocap_pos = np.zeros((1, 3), dtype=float)
        self.mocap_quat = np.zeros((1, 4), dtype=float)


def test_apply_replay_frame_updates_cube_pose_while_weld_is_active():
    data = FakeData()
    joint_addr = np.asarray([0, 1, 2], dtype=int)
    gripper_addr = np.asarray([3, 4], dtype=int)
    cube_addr = 5
    cube_qvel = 7

    first = {
        "q": [0.1, 0.2, 0.3],
        "gripper_width": 0.02,
        "object_position": [0.4, -0.1, 0.2],
        "weld_active": True,
    }
    second = {
        "q": [0.2, 0.3, 0.4],
        "gripper_width": 0.02,
        "object_position": [0.5, 0.1, 0.3],
        "object_quat": [0.70710678, 0.0, 0.70710678, 0.0],
        "weld_active": True,
    }

    _apply_replay_frame(data, first, joint_addr, gripper_addr, cube_addr, cube_qvel, weld_id=0)
    assert data.qpos[cube_addr : cube_addr + 3] == pytest.approx(first["object_position"])

    _apply_replay_frame(data, second, joint_addr, gripper_addr, cube_addr, cube_qvel, weld_id=0)

    assert data.qpos[joint_addr] == pytest.approx(second["q"])
    assert data.qpos[gripper_addr] == pytest.approx([0.01, 0.01])
    assert data.qpos[cube_addr : cube_addr + 3] == pytest.approx(second["object_position"])
    assert data.qpos[cube_addr + 3 : cube_addr + 7] == pytest.approx(second["object_quat"])
    assert data.eq_active[0] == 1


def test_apply_replay_frame_can_keep_visual_weld_disabled_during_kinematic_playback():
    data = FakeData()
    frame = {
        "q": [0.1, 0.2, 0.3],
        "gripper_width": 0.02,
        "object_position": [0.44, -0.12, 0.30],
        "object_quat": [1.0, 0.0, 0.0, 0.0],
        "weld_active": True,
    }

    semantic_weld = _apply_replay_frame(
        data,
        frame,
        np.asarray([0, 1, 2], dtype=int),
        np.asarray([3, 4], dtype=int),
        cube_addr=5,
        cube_qvel=7,
        weld_id=0,
        enable_visual_weld=False,
    )

    assert semantic_weld is True
    assert data.qpos[5:8] == pytest.approx(frame["object_position"])
    assert data.eq_active[0] == 0


def test_set_mocap_pose_updates_invisible_grasp_anchor():
    data = FakeData()

    _set_mocap_pose(
        data,
        0,
        position=[0.4, -0.2, 0.5],
        quat=[0.5, 0.5, 0.5, 0.5],
    )

    assert data.mocap_pos[0] == pytest.approx([0.4, -0.2, 0.5])
    assert data.mocap_quat[0] == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_frame_delay_supports_uncapped_playback():
    assert _frame_delay(75.0) == pytest.approx(1.0 / 75.0)
    assert _frame_delay(0.0) == pytest.approx(0.0)
    assert _frame_delay(-1.0) == pytest.approx(0.0)


def test_frame_cursor_decouples_render_fps_from_playback_speed():
    assert _advance_frame_cursor(10.0, playback_speed=0.5, frame_count=100) == pytest.approx(10.5)
    assert _advance_frame_cursor(99.8, playback_speed=0.5, frame_count=100) == pytest.approx(0.0)
    assert _advance_frame_cursor(10.0, playback_speed=0.0, frame_count=100) == pytest.approx(10.0)
