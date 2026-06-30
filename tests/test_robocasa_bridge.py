from __future__ import annotations

from morphtamp_x_v2.models import PickPlaceScenario
from morphtamp_x_v2.robocasa_bridge import (
    RoboCasaSnapshot,
    extract_snapshot_from_environment,
    extract_snapshot_from_mapping,
    scenario_from_robocasa_snapshot,
    task_hint_for_robocasa,
)


def test_pick_place_coffee_behavior_maps_to_object_and_task_hint():
    hint = task_hint_for_robocasa("PickPlaceCoffee", "machine_to_counter")

    assert hint["object_type"] == "mug_proxy"
    assert hint["task_name"] == "shelf_pick"
    assert hint["supported"] is True

    reverse = task_hint_for_robocasa("PickPlaceCoffee", "counter_to_machine")
    assert reverse["object_type"] == "mug_proxy"
    assert reverse["task_name"] == "shelf_place"


def test_snapshot_from_mapping_preserves_robocasa_metadata_and_world_points():
    snapshot = extract_snapshot_from_mapping(
        {
            "env_name": "PickPlaceCoffee",
            "robot": "PandaOmron",
            "behavior": "machine_to_counter",
            "seed": 42,
            "object_position": [0.45, -3.42, 1.0],
            "target_position": [0.55, -2.95, 0.96],
            "fixture_positions": {"coffee_machine": [0.27, -3.42, 1.09]},
        }
    )

    assert snapshot.env_name == "PickPlaceCoffee"
    assert snapshot.robot == "PandaOmron"
    assert snapshot.object_type_hint == "mug_proxy"
    assert snapshot.task_name_hint == "shelf_pick"
    assert snapshot.object_position == (0.45, -3.42, 1.0)
    assert snapshot.metadata["fixture_positions"]["coffee_machine"] == [0.27, -3.42, 1.09]


def test_snapshot_can_be_projected_to_local_pick_place_scenario():
    snapshot = RoboCasaSnapshot(
        env_name="PickPlaceCoffee",
        robot="PandaOmron",
        behavior="counter_to_machine",
        seed=7,
        object_position=(0.40, -3.10, 0.06),
        target_position=(0.55, -3.45, 0.22),
        object_type_hint="mug_proxy",
        task_name_hint="shelf_place",
        metadata={},
    )

    scenario = scenario_from_robocasa_snapshot(snapshot, center_xy=(0.50, 0.0))

    assert isinstance(scenario, PickPlaceScenario)
    assert scenario.object_type == "mug_proxy"
    assert scenario.task_name == "shelf_place"
    assert scenario.object_description.startswith("RoboCasa-derived")
    assert abs(scenario.cube_start[0] - 0.50) < 0.25
    assert abs(scenario.place_target[1]) < 0.25


class FakeModel:
    def __init__(self):
        self.body_ids = {"obj_main": 0, "counter_main": 1, "coffee_main": 2}

    def body_name2id(self, name):
        return self.body_ids[name]


class FakeData:
    body_xpos = [
        [0.45, -3.42, 1.00],
        [0.32, -2.10, 0.46],
        [0.27, -3.42, 1.09],
    ]


class FakeSim:
    model = FakeModel()
    data = FakeData()


class FakeFixture:
    def __init__(self, root_body, size):
        self.root_body = root_body
        self.size = size


class FakeObject:
    root_body = "obj_main"


class FakeEnvironment:
    behavior = "machine_to_counter"
    sim = FakeSim()
    objects = {"obj": FakeObject()}
    counter = FakeFixture("counter_main", [2.8, 0.65, 0.92])
    coffee_machine = FakeFixture("coffee_main", [0.13, 0.45, 0.33])


def test_live_environment_snapshot_uses_counter_top_not_counter_center():
    snapshot = extract_snapshot_from_environment(
        FakeEnvironment(),
        env_name="PickPlaceCoffee",
        robot="PandaOmron",
        behavior="machine_to_counter",
        seed=42,
    )

    assert snapshot.target_position is not None
    assert snapshot.target_position[0] == 0.32
    assert snapshot.target_position[1] == -2.10
    assert snapshot.target_position[2] > 0.90
    assert snapshot.metadata["target_source"] == "counter_top"
