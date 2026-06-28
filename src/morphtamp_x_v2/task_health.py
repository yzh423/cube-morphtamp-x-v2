from __future__ import annotations

from typing import Any

from .models import PickPlaceScenario
from .objects import OBJECT_TYPES
from .tasks import TASK_TYPES, make_scenario


def _table_top_z(scenario: PickPlaceScenario) -> float:
    return float(scenario.table_center[2] + 0.5 * scenario.table_size[2])


def _object_half_height(scenario: PickPlaceScenario) -> float:
    if scenario.object_geom_type == "box":
        return float(scenario.object_size[2])
    if scenario.object_geom_type == "sphere":
        return float(scenario.object_size[0])
    if scenario.object_geom_type in {"cylinder", "capsule"}:
        return float(scenario.object_size[1])
    return 0.0


def _point_inside_obstacle(
    point: tuple[float, float, float],
    scenario: PickPlaceScenario,
) -> bool:
    if scenario.obstacle_center is None or scenario.obstacle_size is None:
        return False
    center = scenario.obstacle_center
    half = scenario.obstacle_size
    return (
        abs(float(point[0]) - float(center[0])) <= float(half[0])
        and abs(float(point[1]) - float(center[1])) <= float(half[1])
        and abs(float(point[2]) - float(center[2])) <= float(half[2])
    )


def _point_on_table_xy(point: tuple[float, float, float], scenario: PickPlaceScenario) -> bool:
    return (
        abs(float(point[0]) - float(scenario.table_center[0])) <= 0.5 * float(scenario.table_size[0])
        and abs(float(point[1]) - float(scenario.table_center[1])) <= 0.5 * float(scenario.table_size[1])
    )


def _is_supported_by_table(point: tuple[float, float, float], scenario: PickPlaceScenario) -> bool:
    expected_z = _table_top_z(scenario) + _object_half_height(scenario)
    return _point_on_table_xy(point, scenario) and abs(float(point[2]) - expected_z) <= 1e-9


def _is_supported_by_obstacle_top(point: tuple[float, float, float], scenario: PickPlaceScenario) -> bool:
    if scenario.obstacle_center is None or scenario.obstacle_size is None:
        return False
    center = scenario.obstacle_center
    half = scenario.obstacle_size
    expected_z = float(center[2]) + float(half[2]) + _object_half_height(scenario)
    return (
        abs(float(point[0]) - float(center[0])) <= float(half[0])
        and abs(float(point[1]) - float(center[1])) <= float(half[1])
        and abs(float(point[2]) - expected_z) <= 1e-9
    )


def _is_supported_by_support_block(point: tuple[float, float, float], scenario: PickPlaceScenario) -> bool:
    half_height = _object_half_height(scenario)
    for center, half in scenario.support_blocks:
        expected_z = float(center[2]) + float(half[2]) + half_height
        if (
            abs(float(point[0]) - float(center[0])) <= float(half[0])
            and abs(float(point[1]) - float(center[1])) <= float(half[1])
            and abs(float(point[2]) - expected_z) <= 1e-9
        ):
            return True
    return False


def _is_supported(point: tuple[float, float, float], scenario: PickPlaceScenario) -> bool:
    return (
        _is_supported_by_table(point, scenario)
        or _is_supported_by_obstacle_top(point, scenario)
        or _is_supported_by_support_block(point, scenario)
    )


def check_scenario(scenario: PickPlaceScenario) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for label, point in (("start", scenario.cube_start), ("target", scenario.place_target)):
        if not _is_supported(point, scenario):
            failures.append(
                {
                    "code": f"{label}_not_supported",
                    "point": list(point),
                    "table_top_z": _table_top_z(scenario),
                }
            )
        if _point_inside_obstacle(point, scenario):
            failures.append(
                {
                    "code": f"{label}_inside_obstacle",
                    "point": list(point),
                    "obstacle_center": None
                    if scenario.obstacle_center is None
                    else list(scenario.obstacle_center),
                    "obstacle_size": None if scenario.obstacle_size is None else list(scenario.obstacle_size),
                }
            )
    return failures


def check_all_tasks(
    *,
    objects: tuple[str, ...] = OBJECT_TYPES,
    tasks: tuple[str, ...] = TASK_TYPES,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    for object_type in objects:
        for task_name in tasks:
            checked += 1
            scenario = make_scenario(object_type=object_type, task_name=task_name)
            for failure in check_scenario(scenario):
                failures.append(
                    {
                        "object_type": object_type,
                        "task_name": task_name,
                        **failure,
                    }
                )
    return {
        "schema_version": 1,
        "checked_cases": checked,
        "passed": len(failures) == 0,
        "failures": failures,
    }
