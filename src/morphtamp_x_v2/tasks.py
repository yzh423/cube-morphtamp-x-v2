from __future__ import annotations

from dataclasses import dataclass

from .models import PickPlaceScenario
from .objects import object_half_height, object_spec


@dataclass(frozen=True)
class TaskSpec:
    name: str
    start_xy: tuple[float, float]
    target_xy: tuple[float, float]
    start_surface_z: float
    target_surface_z: float
    lift_height: float
    table_center: tuple[float, float, float]
    table_size: tuple[float, float, float]
    obstacle_center: tuple[float, float, float] | None = None
    obstacle_size: tuple[float, float, float] | None = None
    support_blocks: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        ...,
    ] = ()
    description: str = ""
    motion_style: str = "direct"


TASK_SPECS: dict[str, TaskSpec] = {
    "tabletop_easy": TaskSpec(
        name="tabletop_easy",
        start_xy=(0.42, -0.08),
        target_xy=(0.48, 0.08),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.17,
        table_center=(0.45, 0.0, 0.02),
        table_size=(0.80, 0.55, 0.04),
        description="short unobstructed tabletop transfer",
    ),
    "long_transfer": TaskSpec(
        name="long_transfer",
        start_xy=(0.36, -0.18),
        target_xy=(0.56, 0.18),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.20,
        table_center=(0.46, 0.0, 0.02),
        table_size=(0.95, 0.70, 0.04),
        description="larger lateral displacement to stress reach and smoothness",
    ),
    "high_to_low": TaskSpec(
        name="high_to_low",
        start_xy=(0.40, -0.10),
        target_xy=(0.50, 0.12),
        start_surface_z=0.16,
        target_surface_z=0.04,
        lift_height=0.18,
        table_center=(0.45, 0.0, 0.02),
        table_size=(0.90, 0.65, 0.04),
        support_blocks=(((0.40, -0.10, 0.08), (0.08, 0.08, 0.08)),),
        description="pick from elevated block and place on lower table",
    ),
    "low_to_high": TaskSpec(
        name="low_to_high",
        start_xy=(0.40, -0.12),
        target_xy=(0.52, 0.10),
        start_surface_z=0.04,
        target_surface_z=0.16,
        lift_height=0.24,
        table_center=(0.46, 0.0, 0.02),
        table_size=(0.92, 0.65, 0.04),
        support_blocks=(((0.52, 0.10, 0.08), (0.08, 0.08, 0.08)),),
        description="place onto elevated block/shelf proxy",
    ),
    "over_barrier": TaskSpec(
        name="over_barrier",
        start_xy=(0.40, -0.22),
        target_xy=(0.52, 0.22),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.28,
        table_center=(0.46, 0.0, 0.02),
        table_size=(0.92, 0.82, 0.04),
        obstacle_center=(0.46, 0.0, 0.13),
        obstacle_size=(0.16, 0.018, 0.09),
        description="carry object over a mid-height obstacle between start and target",
    ),
    "narrow_slot": TaskSpec(
        name="narrow_slot",
        start_xy=(0.40, -0.12),
        target_xy=(0.48, 0.13),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.20,
        table_center=(0.44, 0.0, 0.02),
        table_size=(0.80, 0.60, 0.04),
        obstacle_center=(0.46, 0.00, 0.09),
        obstacle_size=(0.018, 0.070, 0.07),
        description="target near a narrow slot proxy; emphasizes placement accuracy",
    ),
    "shelf_pick": TaskSpec(
        name="shelf_pick",
        start_xy=(0.38, -0.16),
        target_xy=(0.50, 0.12),
        start_surface_z=0.22,
        target_surface_z=0.04,
        lift_height=0.18,
        table_center=(0.44, 0.0, 0.02),
        table_size=(0.88, 0.70, 0.04),
        support_blocks=(((0.38, -0.16, 0.11), (0.11, 0.10, 0.11)),),
        description="pick object from a raised shelf block and place on table",
    ),
    "shelf_place": TaskSpec(
        name="shelf_place",
        start_xy=(0.40, -0.14),
        target_xy=(0.52, 0.16),
        start_surface_z=0.04,
        target_surface_z=0.22,
        lift_height=0.28,
        table_center=(0.46, 0.0, 0.02),
        table_size=(0.92, 0.78, 0.04),
        support_blocks=(((0.52, 0.16, 0.11), (0.11, 0.10, 0.11)),),
        description="place object onto an elevated shelf block",
    ),
    "far_corner": TaskSpec(
        name="far_corner",
        start_xy=(0.34, -0.24),
        target_xy=(0.62, 0.24),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.22,
        table_center=(0.48, 0.0, 0.02),
        table_size=(1.05, 0.90, 0.04),
        description="large diagonal transfer to stress reachability and joint limits",
    ),
    "under_bridge": TaskSpec(
        name="under_bridge",
        start_xy=(0.40, -0.20),
        target_xy=(0.52, 0.20),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.11,
        table_center=(0.46, 0.0, 0.02),
        table_size=(0.92, 0.82, 0.04),
        obstacle_center=(0.46, 0.0, 0.22),
        obstacle_size=(0.18, 0.018, 0.018),
        description="move below a low bridge; tests clearance control",
    ),
    "around_wall": TaskSpec(
        name="around_wall",
        start_xy=(0.38, -0.20),
        target_xy=(0.56, 0.20),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.26,
        table_center=(0.47, 0.0, 0.02),
        table_size=(0.96, 0.84, 0.04),
        obstacle_center=(0.47, 0.0, 0.11),
        obstacle_size=(0.025, 0.16, 0.07),
        description="transfer around/over a vertical wall proxy",
    ),
    "precision_drop": TaskSpec(
        name="precision_drop",
        start_xy=(0.40, -0.12),
        target_xy=(0.50, 0.08),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.16,
        table_center=(0.45, 0.0, 0.02),
        table_size=(0.82, 0.58, 0.04),
        description="small target pad; emphasizes placement precision",
    ),
    "near_to_far_reach": TaskSpec(
        name="near_to_far_reach",
        start_xy=(0.34, -0.10),
        target_xy=(0.64, 0.10),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.22,
        table_center=(0.49, 0.0, 0.02),
        table_size=(1.10, 0.70, 0.04),
        description="near-body pick followed by a deliberate far reach extension",
        motion_style="extend",
    ),
    "far_to_near_retract": TaskSpec(
        name="far_to_near_retract",
        start_xy=(0.64, -0.10),
        target_xy=(0.34, 0.10),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.22,
        table_center=(0.49, 0.0, 0.02),
        table_size=(1.10, 0.70, 0.04),
        description="far pick followed by retraction toward the robot base",
        motion_style="retract",
    ),
    "folded_transfer": TaskSpec(
        name="folded_transfer",
        start_xy=(0.58, -0.18),
        target_xy=(0.56, 0.20),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.24,
        table_center=(0.50, 0.0, 0.02),
        table_size=(1.00, 0.80, 0.04),
        obstacle_center=(0.46, 0.00, 0.12),
        obstacle_size=(0.050, 0.18, 0.08),
        description="force a visible fold-near-body waypoint before extending back out",
        motion_style="fold_then_extend",
    ),
    "compound_shelf_barrier": TaskSpec(
        name="compound_shelf_barrier",
        start_xy=(0.38, -0.22),
        target_xy=(0.60, 0.22),
        start_surface_z=0.16,
        target_surface_z=0.25,
        lift_height=0.30,
        table_center=(0.49, 0.0, 0.02),
        table_size=(1.08, 0.90, 0.04),
        obstacle_center=(0.49, 0.0, 0.20),
        obstacle_size=(0.18, 0.022, 0.14),
        support_blocks=(
            ((0.38, -0.22, 0.08), (0.09, 0.09, 0.08)),
            ((0.60, 0.22, 0.125), (0.10, 0.10, 0.125)),
        ),
        description="compound high shelf transfer over a solid mid-height barrier",
        motion_style="compound",
    ),
    "diagonal_reach_around": TaskSpec(
        name="diagonal_reach_around",
        start_xy=(0.35, -0.25),
        target_xy=(0.65, 0.25),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.28,
        table_center=(0.50, 0.0, 0.02),
        table_size=(1.10, 0.92, 0.04),
        obstacle_center=(0.50, 0.0, 0.13),
        obstacle_size=(0.035, 0.20, 0.09),
        description="diagonal transfer around a vertical wall while placing back on the table",
        motion_style="compound",
    ),
    "reach_boundary_far": TaskSpec(
        name="reach_boundary_far",
        start_xy=(0.32, -0.28),
        target_xy=(0.70, 0.28),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.26,
        table_center=(0.51, 0.0, 0.02),
        table_size=(1.15, 1.02, 0.04),
        description="far boundary transfer designed to expose insufficient reach margin",
        motion_style="extend",
    ),
    "tight_under_bridge": TaskSpec(
        name="tight_under_bridge",
        start_xy=(0.40, -0.22),
        target_xy=(0.54, 0.22),
        start_surface_z=0.04,
        target_surface_z=0.04,
        lift_height=0.10,
        table_center=(0.47, 0.0, 0.02),
        table_size=(0.95, 0.86, 0.04),
        obstacle_center=(0.47, 0.0, 0.18),
        obstacle_size=(0.20, 0.018, 0.018),
        description="low-clearance bridge task for collision and clearance stress testing",
        motion_style="direct",
    ),
    "high_shelf_barrier": TaskSpec(
        name="high_shelf_barrier",
        start_xy=(0.36, -0.24),
        target_xy=(0.62, 0.24),
        start_surface_z=0.04,
        target_surface_z=0.30,
        lift_height=0.34,
        table_center=(0.50, 0.0, 0.02),
        table_size=(1.12, 0.95, 0.04),
        obstacle_center=(0.50, 0.0, 0.18),
        obstacle_size=(0.16, 0.02, 0.12),
        support_blocks=(((0.62, 0.24, 0.15), (0.10, 0.10, 0.15)),),
        description="high shelf placement with a midline barrier for combined reach and clearance stress",
        motion_style="compound",
    ),
}

TASK_TYPES = tuple(TASK_SPECS)


def task_spec(name: str) -> TaskSpec:
    try:
        return TASK_SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown task {name!r}; choose from {', '.join(TASK_TYPES)}") from error


def make_scenario(
    *,
    object_type: str = "cube",
    task_name: str = "tabletop_easy",
    center_xy: tuple[float, float] | None = None,
) -> PickPlaceScenario:
    obj = object_spec(object_type)
    task = task_spec(task_name)
    half_height = object_half_height(obj)
    dx = 0.0 if center_xy is None else float(center_xy[0]) - 0.5 * (task.start_xy[0] + task.target_xy[0])
    dy = 0.0 if center_xy is None else float(center_xy[1]) - 0.5 * (task.start_xy[1] + task.target_xy[1])

    def xy(point: tuple[float, float]) -> tuple[float, float]:
        return (float(point[0] + dx), float(point[1] + dy))

    start_xy = xy(task.start_xy)
    target_xy = xy(task.target_xy)
    table_center = (
        float(task.table_center[0] + dx),
        float(task.table_center[1] + dy),
        float(task.table_center[2]),
    )
    obstacle_center = None
    if task.obstacle_center is not None:
        obstacle_center = (
            float(task.obstacle_center[0] + dx),
            float(task.obstacle_center[1] + dy),
            float(task.obstacle_center[2]),
        )
    support_blocks = tuple(
        (
            (
                float(center[0] + dx),
                float(center[1] + dy),
                float(center[2]),
            ),
            (
                float(size[0]),
                float(size[1]),
                float(size[2]),
            ),
        )
        for center, size in task.support_blocks
    )
    return PickPlaceScenario(
        task_name=task.name,
        object_type=obj.name,
        object_geom_type=obj.geom_type,
        object_size=obj.size,
        object_mass=obj.mass,
        cube_size=max(obj.size) * 2.0,
        cube_start=(start_xy[0], start_xy[1], float(task.start_surface_z + half_height)),
        place_target=(target_xy[0], target_xy[1], float(task.target_surface_z + half_height)),
        table_center=table_center,
        table_size=task.table_size,
        lift_height=task.lift_height,
        pregrasp_clearance=obj.grasp_clearance,
        release_clearance=obj.release_clearance,
        open_width=obj.open_width,
        closed_width=obj.closed_width,
        grasp_tcp_offset=obj.grasp_tcp_offset,
        obstacle_center=obstacle_center,
        obstacle_size=task.obstacle_size,
        support_blocks=support_blocks,
        object_description=obj.description,
        task_description=task.description,
        motion_style=task.motion_style,
    )
