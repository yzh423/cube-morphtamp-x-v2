from __future__ import annotations

import numpy as np

from .grasp_strategies import GraspStrategy, strategy_for_object
from .models import Phase, PickPlaceScenario


def _add(
    point: tuple[float, float, float],
    delta: tuple[float, float, float],
) -> tuple[float, float, float]:
    value = np.asarray(point, dtype=float) + np.asarray(delta, dtype=float)
    return (float(value[0]), float(value[1]), float(value[2]))


def plan_pick_place(
    scenario: PickPlaceScenario,
    *,
    grasp_strategy: GraspStrategy | None = None,
) -> tuple[Phase, ...]:
    """Create a physically ordered pick-place phase sequence.

    The TCP target is the gripper pinch center. For a cube, grasp and place are
    at the cube center; pregrasp/release are above the cube center.
    """

    start = scenario.cube_start
    target = scenario.place_target
    strategy = grasp_strategy or strategy_for_object(scenario.object_type)
    grasp_offset = tuple(
        float(base) + float(strategy_delta)
        for base, strategy_delta in zip(scenario.grasp_tcp_offset, strategy.tcp_offset)
    )

    def phase(
        name: str,
        tcp_position: tuple[float, float, float],
        gripper_width: float,
        object_attached: bool,
        description: str,
        *,
        uses_grasp_strategy: bool = False,
    ) -> Phase:
        return Phase(
            name,
            tcp_position,
            gripper_width,
            object_attached,
            description,
            strategy.name if uses_grasp_strategy else None,
            strategy.approach_axis if uses_grasp_strategy else None,
            strategy.tcp_quat if uses_grasp_strategy else (1.0, 0.0, 0.0, 0.0),
            bool(strategy.requires_orientation) if uses_grasp_strategy else False,
            float(strategy.orientation_tolerance) if uses_grasp_strategy else 3.141592653589793,
        )

    def tcp_for_object(point: tuple[float, float, float]) -> tuple[float, float, float]:
        return _add(point, grasp_offset)

    grasp_tcp = tcp_for_object(start)
    place_tcp = tcp_for_object(target)
    midpoint = tuple((np.asarray(start, dtype=float) + np.asarray(target, dtype=float)) * 0.5)
    base_transfer_z = max(start[2], target[2]) + scenario.lift_height
    if scenario.obstacle_center is not None and scenario.obstacle_size is not None:
        obstacle_top = float(scenario.obstacle_center[2] + scenario.obstacle_size[2])
        # TCP is the pinch center, not the lowest point of the hand. Keep a
        # conservative wrist / gripper clearance over physical barriers.
        base_transfer_z = max(base_transfer_z, obstacle_top + 0.26)
    home_z = base_transfer_z + grasp_offset[2] + 0.12
    home = (float(midpoint[0]), float(midpoint[1]), float(home_z))
    lift = tcp_for_object((float(start[0]), float(start[1]), float(base_transfer_z)))
    transport = tcp_for_object((float(target[0]), float(target[1]), float(base_transfer_z)))
    descend = _add(place_tcp, (0.0, 0.0, scenario.release_clearance))
    phases = [
        phase("home", home, scenario.open_width, False, "neutral safe pose above the transfer corridor"),
        phase(
            "pregrasp",
            _add(grasp_tcp, (0.0, 0.0, scenario.pregrasp_clearance)),
            scenario.open_width,
            False,
            f"{strategy.description}; approach while object remains static",
            uses_grasp_strategy=True,
        ),
        phase(
            "grasp",
            grasp_tcp,
            scenario.open_width,
            False,
            f"move pinch center to {strategy.name} pose before closing",
            uses_grasp_strategy=True,
        ),
        phase(
            "close_gripper",
            grasp_tcp,
            scenario.closed_width,
            False,
            "close fingers while object is still on support",
            uses_grasp_strategy=True,
        ),
        phase(
            "attach",
            grasp_tcp,
            scenario.closed_width,
            True,
            "activate weld after gripper closure",
            uses_grasp_strategy=True,
        ),
        phase(
            "lift",
            lift,
            scenario.closed_width,
            True,
            "lift attached object vertically while preserving grasp orientation",
            uses_grasp_strategy=True,
        ),
    ]
    motion_style = scenario.motion_style
    if motion_style in {"fold_then_extend", "compound"}:
        fold_x = max(0.28, min(start[0], target[0]) - 0.10)
        phases.append(
            phase(
                "fold_near_body",
                (float(fold_x), float(midpoint[1]), float(base_transfer_z + 0.03)),
                scenario.closed_width,
                True,
                "fold elbow near the robot body before re-extending",
                uses_grasp_strategy=True,
            )
        )
    if motion_style == "retract":
        retract_x = max(0.28, min(start[0], target[0]) - 0.08)
        phases.append(
            phase(
                "retract_near",
                (float(retract_x), float(midpoint[1]), float(base_transfer_z)),
                scenario.closed_width,
                True,
                "retract toward the robot base before final placement",
                uses_grasp_strategy=True,
            )
        )
    if motion_style == "compound":
        phases.append(
            phase(
                "high_clearance",
                (float(midpoint[0]), float(midpoint[1]), float(base_transfer_z + 0.08)),
                scenario.closed_width,
                True,
                "extra clearance waypoint for compound obstacle and height changes",
                uses_grasp_strategy=True,
            )
        )
    if motion_style in {"extend", "fold_then_extend", "compound"}:
        extend_x = min(0.72, max(start[0], target[0]) + 0.06)
        phases.append(
            phase(
                "extend_far",
                (float(extend_x), float(midpoint[1]), float(base_transfer_z)),
                scenario.closed_width,
                True,
                "deliberately extend toward the far workspace before target approach",
                uses_grasp_strategy=True,
            )
        )
    if scenario.obstacle_center is not None and scenario.obstacle_size is not None:
        obstacle = scenario.obstacle_center
        half = scenario.obstacle_size
        wall_like = half[1] > 2.0 * half[0]
        if wall_like:
            direction = 1.0 if target[0] >= obstacle[0] else -1.0
            bypass_x = float(obstacle[0] + direction * (half[0] + 0.16))
            bypass_y = float(0.5 * (start[1] + target[1]))
            phases.append(
                phase(
                    "bypass_obstacle",
                    (bypass_x, bypass_y, float(base_transfer_z)),
                    scenario.closed_width,
                    True,
                    "side bypass waypoint for wall-like obstacle",
                    uses_grasp_strategy=True,
                )
            )
    phases.extend(
        [
        phase(
            "transport",
            transport,
            scenario.closed_width,
            True,
            "carry object above target while preserving grasp orientation",
            uses_grasp_strategy=True,
        ),
        phase(
            "descend",
            descend,
            scenario.closed_width,
            True,
            "lower while still holding the object with the selected grasp",
            uses_grasp_strategy=True,
        ),
        phase(
            "place",
            place_tcp,
            scenario.closed_width,
            True,
            "place object center at target support height with the selected grasp",
            uses_grasp_strategy=True,
        ),
        phase(
            "open_gripper",
            place_tcp,
            scenario.open_width,
            True,
            "open fingers before releasing weld while keeping TCP pose stable",
            uses_grasp_strategy=True,
        ),
        phase(
            "release",
            place_tcp,
            scenario.open_width,
            False,
            "disable weld; object remains at target",
            uses_grasp_strategy=True,
        ),
        phase("return_home", home, scenario.open_width, False, "retreat after release"),
        ]
    )
    return tuple(phases)
