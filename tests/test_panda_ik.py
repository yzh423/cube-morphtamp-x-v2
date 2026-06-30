from __future__ import annotations

import pytest

import morphtamp_x_v2.panda_ik as panda_ik_module
from morphtamp_x_v2.models import Phase, PickPlaceScenario
from morphtamp_x_v2.objects import object_half_height, object_spec
from morphtamp_x_v2.panda_ik import (
    DEFAULT_FINGERTIP_EXTENSION,
    MuJoCoPandaIK,
    auto_fit_panda_scenario,
    solve_joint_replay,
)
from morphtamp_x_v2.planner import plan_pick_place
from morphtamp_x_v2.replay import build_replay_frames


PANDA_CHAIN_XML = """
<mujoco model="panda_chain">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base">
      <body name="link1">
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
        <body name="link2" pos="0 0 0.18">
          <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
          <body name="link3" pos="0.18 0 0">
            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            <joint name="joint3" type="hinge" axis="0 1 0" range="-2.9 2.9"/>
            <body name="link4" pos="0.18 0 0">
              <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
              <joint name="joint4" type="hinge" axis="0 1 0" range="-3.0 0.0"/>
              <body name="link5" pos="0.15 0 0">
                <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
                <joint name="joint5" type="hinge" axis="0 1 0" range="-2.9 2.9"/>
                <body name="link6" pos="0.12 0 0">
                  <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
                  <joint name="joint6" type="hinge" axis="0 1 0" range="-0.1 3.7"/>
                  <body name="link7" pos="0.10 0 0">
                    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
                    <joint name="joint7" type="hinge" axis="0 1 0" range="-2.9 2.9"/>
                    <body name="hand" pos="0.08 0 0">
                      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


PANDA_CHAIN_WITH_FINGERS_XML = PANDA_CHAIN_XML.replace(
    '<body name="hand" pos="0.08 0 0">\n                      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>\n                    </body>',
    '<body name="hand" pos="0.08 0 0">\n'
    '                      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>\n'
    '                      <body name="left_finger" pos="0.05 0.03 0">\n'
    '                        <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>\n'
    '                        <geom name="left_finger_geom" type="box" size="0.01 0.005 0.03"/>\n'
    '                      </body>\n'
    '                      <body name="right_finger" pos="0.05 -0.03 0">\n'
    '                        <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>\n'
    '                        <geom name="right_finger_geom" type="box" size="0.01 0.005 0.03"/>\n'
    '                      </body>\n'
    '                    </body>',
)


def test_mujoco_panda_ik_detects_seven_arm_joints(tmp_path):
    pytest.importorskip("mujoco")
    path = tmp_path / "panda.xml"
    path.write_text(PANDA_CHAIN_XML, encoding="utf-8")

    ik = MuJoCoPandaIK(path)

    assert ik.joint_names == tuple(f"joint{i}" for i in range(1, 8))
    assert ik.eef_body == "hand"


def test_auto_fit_panda_scenario_places_task_near_current_tcp(tmp_path):
    pytest.importorskip("mujoco")
    path = tmp_path / "panda.xml"
    path.write_text(PANDA_CHAIN_XML, encoding="utf-8")

    scenario = auto_fit_panda_scenario(path, lateral_offset=0.05, vertical_drop=0.08)

    assert scenario.cube_start[1] < scenario.place_target[1]
    assert scenario.cube_start[0] > 0.20
    assert scenario.cube_start[2] > scenario.table_center[2]
    assert scenario.table_center[2] > 0.0
    assert scenario.table_size[0] < 0.40


def test_auto_fit_panda_scenario_shifts_support_blocks_with_target(tmp_path):
    pytest.importorskip("mujoco")
    path = tmp_path / "panda.xml"
    path.write_text(PANDA_CHAIN_XML, encoding="utf-8")

    scenario = auto_fit_panda_scenario(
        path,
        object_type="sphere",
        task_name="shelf_place",
        lateral_offset=0.05,
        vertical_drop=0.08,
    )

    assert scenario.support_blocks
    target_support = min(
        scenario.support_blocks,
        key=lambda block: abs(block[0][0] - scenario.place_target[0])
        + abs(block[0][1] - scenario.place_target[1]),
    )
    center, half_size = target_support
    support_top = center[2] + half_size[2]
    object_radius = object_half_height(object_spec(scenario.object_type))

    assert scenario.place_target[2] == pytest.approx(support_top + object_radius)


def test_panda_ik_uses_gripper_center_when_finger_bodies_exist(tmp_path):
    pytest.importorskip("mujoco")
    path = tmp_path / "panda.xml"
    path.write_text(PANDA_CHAIN_WITH_FINGERS_XML, encoding="utf-8")

    ik = MuJoCoPandaIK(path)
    ik.set_q(ik.neutral_q)

    task = ik.task_position()
    hand = ik.eef_position()

    assert len(ik.gripper_body_ids) == 2
    assert task[0] > hand[0]


def test_panda_ik_task_point_is_distal_to_finger_geom_center(tmp_path):
    pytest.importorskip("mujoco")
    path = tmp_path / "panda.xml"
    path.write_text(PANDA_CHAIN_WITH_FINGERS_XML, encoding="utf-8")

    ik = MuJoCoPandaIK(path)
    ik.set_q(ik.neutral_q)

    finger_center = ik.data.geom_xpos[ik.gripper_geom_ids].mean(axis=0)
    hand = ik.eef_position()
    task = ik.task_position()

    assert task[0] > finger_center[0]
    assert (task[0] - finger_center[0]) == pytest.approx(ik.fingertip_extension)
    assert ik.fingertip_extension > 0.0


def test_default_fingertip_extension_matches_calibrated_grasp_clearance():
    assert DEFAULT_FINGERTIP_EXTENSION == pytest.approx(0.024)


def test_panda_ik_selects_geoms_from_distinct_finger_bodies(tmp_path):
    pytest.importorskip("mujoco")
    path = tmp_path / "panda.xml"
    path.write_text(
        PANDA_CHAIN_WITH_FINGERS_XML.replace(
            '<geom name="left_finger_geom" type="box" size="0.01 0.005 0.03"/>',
            '<geom name="left_finger_geom_a" type="box" size="0.01 0.005 0.03"/>'
            '<geom name="left_finger_geom_b" type="box" size="0.01 0.005 0.03"/>',
        ),
        encoding="utf-8",
    )

    ik = MuJoCoPandaIK(path)
    body_ids = {int(ik.model.geom_bodyid[int(geom_id)]) for geom_id in ik.gripper_geom_ids}

    assert len(ik.gripper_geom_ids) == 2
    assert len(body_ids) == 2


def test_solve_joint_replay_returns_q_for_each_frame(tmp_path):
    pytest.importorskip("mujoco")
    path = tmp_path / "panda.xml"
    path.write_text(PANDA_CHAIN_XML, encoding="utf-8")
    scenario = auto_fit_panda_scenario(path, lateral_offset=0.03, vertical_drop=0.06)
    phases = plan_pick_place(scenario)
    frames = build_replay_frames(scenario, phases, transition_frames=2, hold_frames=1)

    replay = solve_joint_replay(path, frames, tolerance=0.08)

    assert len(replay.frames) == len(frames)
    assert all(len(frame.q) == 7 for frame in replay.frames)
    assert replay.max_position_error < 0.20
    assert replay.joint_path_length >= 0.0
    assert replay.energy_proxy >= 0.0
    assert replay.smoothness_proxy >= 0.0


def test_joint_replay_frame_schema_contains_7d_quality_metrics(tmp_path):
    pytest.importorskip("mujoco")
    path = tmp_path / "panda.xml"
    path.write_text(PANDA_CHAIN_XML, encoding="utf-8")
    scenario = auto_fit_panda_scenario(path, lateral_offset=0.03, vertical_drop=0.06)
    frames = build_replay_frames(scenario, plan_pick_place(scenario), transition_frames=1, hold_frames=1)

    replay = solve_joint_replay(path, frames, tolerance=0.20)
    row = replay.frames[0].to_json_dict()

    assert "collision_count" in row
    assert "joint_margin" in row
    assert "condition_number" in row
    assert "min_singular_value" in row
    assert "target_quat" in row
    assert "tcp_quat" in row
    assert "orientation_error" in row
    assert "object_quat" in row
    assert len(row["target_quat"]) == 4
    assert len(row["tcp_quat"]) == 4
    assert len(row["object_quat"]) == 4
    assert row["collision_count"] == 0
    assert row["joint_margin"] >= 0.0
    assert row["min_singular_value"] is None or row["min_singular_value"] >= 0.0


def test_solve_joint_replay_reports_orientation_failures_for_6d_targets(tmp_path):
    pytest.importorskip("mujoco")
    path = tmp_path / "panda.xml"
    path.write_text(PANDA_CHAIN_XML, encoding="utf-8")
    scenario = auto_fit_panda_scenario(path, object_type="plate", lateral_offset=0.03, vertical_drop=0.06)
    edge = next(strategy for strategy in __import__(
        "morphtamp_x_v2.grasp_strategies",
        fromlist=["strategies_for_object"],
    ).strategies_for_object("plate") if strategy.requires_orientation)
    frames = build_replay_frames(
        scenario,
        plan_pick_place(scenario, grasp_strategy=edge),
        transition_frames=1,
        hold_frames=1,
    )

    replay = solve_joint_replay(path, frames, tolerance=0.20, orientation_tolerance=0.20)
    row = replay.frames[0].to_json_dict()

    assert "max_orientation_error" in replay.to_json_dict()
    assert "energy_proxy" in replay.to_json_dict()
    assert "smoothness_proxy" in replay.to_json_dict()
    assert "orientation_error" in row
    assert isinstance(row["orientation_error"], float)


def test_solve_joint_replay_uses_frame_level_orientation_requirement(monkeypatch):
    import numpy as np

    calls = []
    learned_quat = (0.5, 0.5, 0.5, 0.5)

    class FakeIK:
        joint_names = tuple(f"joint{i}" for i in range(1, 8))
        eef_body = "hand"
        eef_site = None
        ready_q = np.zeros(7)

        def __init__(self, panda_xml):
            self.q = np.zeros(7)

        def solve(self, target, **kwargs):
            calls.append(
                {
                    "orientation_weight": kwargs["orientation_weight"],
                    "target_quat": kwargs["target_quat"],
                }
            )
            return np.zeros(7), 0.0, 0.0

        def set_q(self, q):
            self.q = np.asarray(q, dtype=float)

        def set_gripper_width(self, width):
            return None

        def set_object_position(self, position, quat=None):
            return None

        def task_position(self):
            return np.zeros(3)

        def task_quat(self):
            return learned_quat

        def collision_violation_count(self):
            return 0

        def joint_margin(self, q):
            return 1.0

        def condition_number(self):
            return 1.0

        def min_singular_value(self):
            return 1.0

    monkeypatch.setattr(panda_ik_module, "MuJoCoPandaIK", FakeIK)
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
                tcp_quat=(1.0, 0.0, 0.0, 0.0),
                orientation_required=False,
                orientation_weight=0.08,
                orientation_mode="hold_attach",
            ),
            Phase(
                "transport",
                scenario.place_target,
                0.04,
                True,
                "transport",
                tcp_quat=(1.0, 0.0, 0.0, 0.0),
                orientation_required=False,
                orientation_weight=0.08,
                orientation_mode="hold_attach",
            ),
        ),
        transition_frames=1,
        hold_frames=1,
    )

    solve_joint_replay("fake.xml", frames, orientation_tolerance=3.141592653589793)

    assert calls
    assert calls[1]["orientation_weight"] == 0.0
    assert calls[1]["target_quat"] is None
    assert calls[2]["orientation_weight"] > 0.0
    assert calls[2]["target_quat"] == learned_quat
