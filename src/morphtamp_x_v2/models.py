from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PickPlaceScenario:
    task_name: str = "cube_pick_place_v2"
    object_type: str = "cube"
    object_geom_type: str = "box"
    object_size: tuple[float, ...] = (0.02, 0.02, 0.02)
    object_mass: float = 0.10
    cube_size: float = 0.04
    cube_start: tuple[float, float, float] = (0.42, -0.12, 0.06)
    place_target: tuple[float, float, float] = (0.48, 0.12, 0.06)
    table_center: tuple[float, float, float] = (0.45, 0.0, 0.02)
    table_size: tuple[float, float, float] = (0.80, 0.60, 0.04)
    lift_height: float = 0.18
    pregrasp_clearance: float = 0.12
    release_clearance: float = 0.10
    open_width: float = 0.085
    closed_width: float = 0.036
    grasp_tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obstacle_center: tuple[float, float, float] | None = None
    obstacle_size: tuple[float, float, float] | None = None
    support_blocks: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        ...,
    ] = ()
    object_description: str = "box object; side pinch with flat finger pads"
    task_description: str = "single-object pick and place"
    motion_style: str = "direct"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "object_type": self.object_type,
            "object_geom_type": self.object_geom_type,
            "object_size": list(self.object_size),
            "object_mass": self.object_mass,
            "cube_size": self.cube_size,
            "cube_start": list(self.cube_start),
            "place_target": list(self.place_target),
            "table_center": list(self.table_center),
            "table_size": list(self.table_size),
            "lift_height": self.lift_height,
            "pregrasp_clearance": self.pregrasp_clearance,
            "release_clearance": self.release_clearance,
            "open_width": self.open_width,
            "closed_width": self.closed_width,
            "grasp_tcp_offset": list(self.grasp_tcp_offset),
            "obstacle_center": None if self.obstacle_center is None else list(self.obstacle_center),
            "obstacle_size": None if self.obstacle_size is None else list(self.obstacle_size),
            "support_blocks": [
                {"center": list(center), "size": list(size)}
                for center, size in self.support_blocks
            ],
            "object_description": self.object_description,
            "task_description": self.task_description,
            "motion_style": self.motion_style,
        }


@dataclass(frozen=True)
class Phase:
    name: str
    tcp_position: tuple[float, float, float]
    gripper_width: float
    object_attached: bool
    description: str
    grasp_strategy: str | None = None
    approach_axis: str | None = None
    tcp_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    orientation_required: bool = False
    orientation_tolerance: float = 3.141592653589793
    orientation_mode: str = "target"
    orientation_weight: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tcp_position": list(self.tcp_position),
            "tcp_quat": list(self.tcp_quat),
            "gripper_width": self.gripper_width,
            "object_attached": self.object_attached,
            "description": self.description,
            "grasp_strategy": self.grasp_strategy,
            "approach_axis": self.approach_axis,
            "orientation_required": self.orientation_required,
            "orientation_tolerance": self.orientation_tolerance,
            "orientation_mode": self.orientation_mode,
            "orientation_weight": self.orientation_weight,
        }


@dataclass(frozen=True)
class ReplayFrame:
    phase_name: str
    tcp_position: tuple[float, float, float]
    tcp_quat: tuple[float, float, float, float]
    object_position: tuple[float, float, float]
    object_quat: tuple[float, float, float, float]
    gripper_width: float
    object_attached: bool
    weld_active: bool
    progress: float
    orientation_required: bool = False
    orientation_tolerance: float = 3.141592653589793
    orientation_mode: str = "target"
    orientation_weight: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "tcp_position": list(self.tcp_position),
            "tcp_quat": list(self.tcp_quat),
            "object_position": list(self.object_position),
            "object_quat": list(self.object_quat),
            "gripper_width": self.gripper_width,
            "object_attached": self.object_attached,
            "weld_active": self.weld_active,
            "progress": self.progress,
            "orientation_required": self.orientation_required,
            "orientation_tolerance": self.orientation_tolerance,
            "orientation_mode": self.orientation_mode,
            "orientation_weight": self.orientation_weight,
        }
