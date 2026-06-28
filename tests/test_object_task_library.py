from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from morphtamp_x_v2.cli import main, parser
from morphtamp_x_v2.objects import OBJECT_TYPES, object_grasp_width, object_spec
from morphtamp_x_v2.scene_builder import build_pick_place_scene
from morphtamp_x_v2.tasks import TASK_TYPES, make_scenario


PANDA_LIKE_XML = """
<mujoco model="panda_like">
  <worldbody>
    <body name="panda_hand"/>
  </worldbody>
</mujoco>
"""


def test_object_and_task_libraries_expose_expected_entries():
    assert {
        "cube",
        "sphere",
        "cylinder",
        "plate",
        "mug_proxy",
        "bowl_proxy",
        "capsule",
        "tall_box",
        "flat_box",
        "ring",
    }.issubset(set(OBJECT_TYPES))
    assert {
        "tabletop_easy",
        "long_transfer",
        "high_to_low",
        "low_to_high",
        "over_barrier",
        "narrow_slot",
        "shelf_pick",
        "shelf_place",
        "far_corner",
        "under_bridge",
        "around_wall",
        "precision_drop",
        "near_to_far_reach",
        "far_to_near_retract",
        "folded_transfer",
        "compound_shelf_barrier",
        "diagonal_reach_around",
    }.issubset(set(TASK_TYPES))
    assert object_spec("sphere").geom_type == "sphere"
    assert make_scenario(object_type="mug_proxy", task_name="shelf_place").support_blocks
    assert make_scenario(object_type="cube", task_name="folded_transfer").motion_style == "fold_then_extend"


def test_object_closed_widths_do_not_visually_interpenetrate_objects():
    for name in OBJECT_TYPES:
        spec = object_spec(name)
        assert spec.closed_width >= object_grasp_width(spec) + 0.004
        if name == "cube":
            assert spec.closed_width >= object_grasp_width(spec) + 0.020
        assert spec.open_width > spec.closed_width
        assert spec.open_width <= 0.080


def test_scene_builder_uses_selected_object_geometry_and_obstacle(tmp_path):
    panda = tmp_path / "panda.xml"
    panda.write_text(PANDA_LIKE_XML, encoding="utf-8")
    output = tmp_path / "scene.xml"
    scenario = make_scenario(object_type="sphere", task_name="over_barrier")

    build_pick_place_scene(panda_xml=panda, scenario=scenario, output_xml=output)

    root = ET.parse(output).getroot()
    object_geom = next(item for item in root.iter("geom") if item.attrib.get("name") == "v2_object_geom")
    obstacle = next(item for item in root.iter("geom") if item.attrib.get("name") == "v2_obstacle")
    assert object_geom.attrib["type"] == "sphere"
    assert obstacle.attrib["type"] == "box"
    assert float(obstacle.attrib["pos"].split()[2]) > scenario.table_center[2]


def test_scene_builder_adds_support_blocks_separately_from_obstacles(tmp_path):
    panda = tmp_path / "panda.xml"
    panda.write_text(PANDA_LIKE_XML, encoding="utf-8")
    output = tmp_path / "scene.xml"
    scenario = make_scenario(object_type="mug_proxy", task_name="shelf_place")

    build_pick_place_scene(panda_xml=panda, scenario=scenario, output_xml=output)

    root = ET.parse(output).getroot()
    support = next(item for item in root.iter("geom") if item.attrib.get("name") == "v2_support_0")
    obstacle_names = {item.attrib.get("name") for item in root.iter("geom")}
    assert support.attrib["type"] == "box"
    assert "v2_obstacle" not in obstacle_names


def test_over_barrier_obstacle_is_between_start_and_target_without_covering_pick_or_place():
    scenario = make_scenario(object_type="cube", task_name="over_barrier")
    assert scenario.obstacle_center is not None
    assert scenario.obstacle_size is not None
    obstacle_y = scenario.obstacle_center[1]
    half_y = scenario.obstacle_size[1]

    assert scenario.cube_start[1] < obstacle_y < scenario.place_target[1]
    assert abs(scenario.cube_start[1] - obstacle_y) > half_y + 0.12
    assert abs(scenario.place_target[1] - obstacle_y) > half_y + 0.12


def test_narrow_slot_target_is_not_inside_obstacle_volume():
    scenario = make_scenario(object_type="cube", task_name="narrow_slot")
    assert scenario.obstacle_center is not None
    assert scenario.obstacle_size is not None
    center = scenario.obstacle_center
    half = scenario.obstacle_size
    target = scenario.place_target

    inside_obstacle = (
        abs(target[0] - center[0]) <= half[0]
        and abs(target[1] - center[1]) <= half[1]
        and abs(target[2] - center[2]) <= half[2]
    )

    assert inside_obstacle is False


def _object_xy_half_extents(object_type: str) -> tuple[float, float]:
    spec = object_spec(object_type)
    if spec.geom_type == "box":
        return float(spec.size[0]), float(spec.size[1])
    if spec.geom_type in {"sphere", "cylinder"}:
        return float(spec.size[0]), float(spec.size[0])
    if spec.geom_type == "capsule":
        return float(spec.size[1] + spec.size[0]), float(spec.size[0])
    raise AssertionError(f"unsupported object geom {spec.geom_type!r}")


def test_supported_start_and_target_object_volumes_do_not_overlap_obstacles():
    for object_type in OBJECT_TYPES:
        half_x, half_y = _object_xy_half_extents(object_type)
        for task_name in TASK_TYPES:
            scenario = make_scenario(object_type=object_type, task_name=task_name)
            if scenario.obstacle_center is None or scenario.obstacle_size is None:
                continue
            obstacle = scenario.obstacle_center
            obstacle_half = scenario.obstacle_size
            for label, point in {
                "start": scenario.cube_start,
                "target": scenario.place_target,
            }.items():
                xy_overlaps = (
                    abs(point[0] - obstacle[0]) < half_x + obstacle_half[0] + 0.002
                    and abs(point[1] - obstacle[1]) < half_y + obstacle_half[1] + 0.002
                )
                assert not xy_overlaps, (
                    object_type,
                    task_name,
                    label,
                    point,
                    obstacle,
                )


def test_diagonal_reach_around_target_is_supported_by_table_not_floating():
    scenario = make_scenario(object_type="sphere", task_name="diagonal_reach_around")
    radius = object_spec("sphere").size[0]
    table_top_z = scenario.table_center[2] + 0.5 * scenario.table_size[2]

    assert scenario.place_target[2] == table_top_z + radius


def test_complex_tasks_cover_distinct_reach_directions():
    near_to_far = make_scenario(object_type="cube", task_name="near_to_far_reach")
    far_to_near = make_scenario(object_type="cube", task_name="far_to_near_retract")
    compound = make_scenario(object_type="cube", task_name="compound_shelf_barrier")

    assert near_to_far.place_target[0] > near_to_far.cube_start[0]
    assert near_to_far.motion_style == "extend"
    assert far_to_near.place_target[0] < far_to_near.cube_start[0]
    assert far_to_near.motion_style == "retract"
    assert compound.obstacle_center is not None
    assert compound.cube_start[2] != compound.place_target[2]
    assert compound.motion_style == "compound"


def test_cli_accepts_object_task_and_benchmark_options():
    run = parser().parse_args(
        [
            "run",
            "--object",
            "mug_proxy",
            "--task",
            "shelf_place",
            "--grasp-strategy",
            "cup_high_body_pinch",
            "--full-candidate-limit",
            "1",
            "--output-dir",
            "out",
        ]
    )
    benchmark = parser().parse_args(
        [
            "benchmark",
            "--objects",
            "cube",
            "sphere",
            "mug_proxy",
            "--tasks",
            "tabletop_easy",
            "over_barrier",
            "shelf_pick",
            "--panda-xml",
            "panda.xml",
            "--auto-fit-panda",
            "--full-candidate-limit",
            "3",
            "--output",
            "benchmark.json",
        ]
    )

    assert run.object_type == "mug_proxy"
    assert run.task == "shelf_place"
    assert run.grasp_strategy == "cup_high_body_pinch"
    assert run.full_candidate_limit == 1
    assert benchmark.command == "benchmark"
    assert benchmark.objects == ["cube", "sphere", "mug_proxy"]
    assert benchmark.panda_xml.name == "panda.xml"
    assert benchmark.auto_fit_panda is True
    assert benchmark.full_candidate_limit == 3


def test_cli_view_and_browse_default_to_smooth_but_slow_playback():
    view = parser().parse_args(
        [
            "view",
            "--xml",
            "scene.xml",
            "--replay",
            "replay.json",
        ]
    )
    browse = parser().parse_args(
        [
            "browse",
            "--panda-xml",
            "panda.xml",
        ]
    )

    assert view.fps == 60.0
    assert view.playback_speed == 0.55
    assert browse.fps == 60.0
    assert browse.playback_speed == 0.55


def test_cli_lists_objects_and_tasks(capsys):
    assert main(["list-objects"]) == 0
    objects = json.loads(capsys.readouterr().out)
    assert "mug_proxy" in objects["objects"]

    assert main(["list-tasks"]) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert "shelf_place" in tasks["tasks"]


def test_cli_check_tasks_reports_task_health(capsys):
    assert main(["check-tasks"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["checked_cases"] == len(OBJECT_TYPES) * len(TASK_TYPES)


def test_cli_benchmark_writes_object_task_matrix(tmp_path):
    output = tmp_path / "benchmark.json"

    assert (
        main(
            [
                "benchmark",
                "--objects",
                "cube",
                "sphere",
                "--tasks",
                "tabletop_easy",
                "over_barrier",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["total_runs"] == 4
    assert {row["object_type"] for row in payload["results"]} == {"cube", "sphere"}
    assert {row["task_name"] for row in payload["results"]} == {"tabletop_easy", "over_barrier"}
