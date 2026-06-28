from __future__ import annotations

from morphtamp_x_v2.objects import OBJECT_TYPES
from morphtamp_x_v2.task_health import check_all_tasks, check_scenario
from morphtamp_x_v2.tasks import TASK_TYPES, make_scenario


def test_all_task_object_pairs_pass_basic_scene_health_checks():
    report = check_all_tasks(objects=OBJECT_TYPES, tasks=TASK_TYPES)

    assert report["schema_version"] == 1
    assert report["checked_cases"] == len(OBJECT_TYPES) * len(TASK_TYPES)
    assert report["passed"] is True
    assert report["failures"] == []


def test_health_check_rejects_floating_tabletop_target():
    scenario = make_scenario(object_type="sphere", task_name="diagonal_reach_around")
    floating = type(scenario)(
        **{
            **scenario.__dict__,
            "place_target": (
                scenario.place_target[0],
                scenario.place_target[1],
                scenario.place_target[2] + 0.12,
            ),
        }
    )

    failures = check_scenario(floating)

    assert any(item["code"] == "target_not_supported" for item in failures)


def test_health_check_rejects_target_inside_obstacle():
    scenario = make_scenario(object_type="cube", task_name="narrow_slot")
    assert scenario.obstacle_center is not None
    inside = type(scenario)(
        **{
            **scenario.__dict__,
            "place_target": scenario.obstacle_center,
        }
    )

    failures = check_scenario(inside)

    assert any(item["code"] == "target_inside_obstacle" for item in failures)
