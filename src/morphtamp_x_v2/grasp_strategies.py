from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraspStrategy:
    name: str
    approach_axis: str
    tcp_offset: tuple[float, float, float]
    hold_orientation: str
    description: str
    tcp_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    requires_orientation: bool = False
    orientation_tolerance: float = 3.141592653589793
    orientation_weight: float = 0.0


_DEFAULT = GraspStrategy(
    name="generic_center_pinch",
    approach_axis="top",
    tcp_offset=(0.0, 0.0, 0.0),
    hold_orientation="upright",
    description="generic center pinch for compact unknown objects",
)


_STRATEGY_CANDIDATES: dict[str, tuple[GraspStrategy, ...]] = {
    "cube": (
        GraspStrategy(
        name="box_side_pinch",
        approach_axis="side_y",
        tcp_offset=(0.0, 0.0, 0.0),
        hold_orientation="upright",
        description="flat parallel-jaw side pinch through the box center",
    ),
        GraspStrategy(
            name="box_top_center_pinch",
            approach_axis="top",
            tcp_offset=(0.0, 0.0, 0.012),
            hold_orientation="upright",
            description="top-biased center pinch for cluttered side approaches",
        ),
        GraspStrategy(
            name="box_x_side_pinch",
            approach_axis="side_x",
            tcp_offset=(0.0, 0.0, 0.0),
            hold_orientation="upright",
            description="alternate side pinch across the x axis",
        ),
        GraspStrategy(
            name="box_side_pinch_y_neg",
            approach_axis="side_y",
            tcp_offset=(0.0, -0.004, 0.0),
            hold_orientation="upright",
            description="side pinch with negative-y millimeter TCP calibration candidate",
        ),
        GraspStrategy(
            name="box_side_pinch_y_pos",
            approach_axis="side_y",
            tcp_offset=(0.0, 0.004, 0.0),
            hold_orientation="upright",
            description="side pinch with positive-y millimeter TCP calibration candidate",
        ),
    ),
    "tall_box": (
        GraspStrategy(
        name="tall_box_midbody_pinch",
        approach_axis="side_y",
        tcp_offset=(0.0, 0.0, 0.0),
        hold_orientation="upright",
        description="mid-body side pinch to avoid tipping tall rectangular objects",
    ),
        GraspStrategy(
            name="tall_box_high_body_pinch",
            approach_axis="side_y",
            tcp_offset=(0.0, 0.0, 0.018),
            hold_orientation="upright",
            description="higher side pinch for obstacle clearance around tall boxes",
        ),
    ),
    "flat_box": (
        GraspStrategy(
        name="flat_box_edge_pinch",
        approach_axis="edge",
        tcp_offset=(0.0, -0.018, 0.004),
        hold_orientation="edge_stable",
        description="slight edge offset for low-profile flat objects",
    ),
        GraspStrategy(
            name="flat_box_center_top_pinch",
            approach_axis="top",
            tcp_offset=(0.0, 0.0, 0.0),
            hold_orientation="edge_stable",
            description="center-stable top pinch for low-profile flat objects",
        ),
    ),
    "sphere": (
        GraspStrategy(
        name="spherical_center_pinch",
        approach_axis="side_y",
        tcp_offset=(0.0, 0.0, 0.0),
        hold_orientation="free",
        description="center pinch for round objects with symmetric contact",
    ),
        GraspStrategy(
            name="spherical_high_pinch",
            approach_axis="top",
            tcp_offset=(0.0, 0.0, 0.010),
            hold_orientation="free",
            description="slightly high pinch to improve support clearance for spheres",
        ),
        GraspStrategy(
            name="spherical_x_side_pinch",
            approach_axis="side_x",
            tcp_offset=(0.0, 0.0, 0.0),
            hold_orientation="free",
            description="alternate side pinch for sphere transfers",
        ),
    ),
    "bowl_proxy": (
        GraspStrategy(
        name="bowl_broad_side_pinch",
        approach_axis="side_y",
        tcp_offset=(0.0, 0.0, 0.004),
        hold_orientation="upright",
        description="broad side pinch biased slightly above the bowl center",
    ),
        GraspStrategy(
            name="bowl_center_top_pinch",
            approach_axis="top",
            tcp_offset=(0.0, 0.0, 0.010),
            hold_orientation="upright",
            description="top-biased stabilizing pinch for bowl proxies",
        ),
    ),
    "cylinder": (
        GraspStrategy(
        name="cylinder_body_side_pinch",
        approach_axis="side_y",
        tcp_offset=(0.0, 0.0, 0.0),
        hold_orientation="upright",
        description="side pinch on an upright cylindrical body",
    ),
        GraspStrategy(
            name="cylinder_high_body_pinch",
            approach_axis="side_y",
            tcp_offset=(0.0, 0.0, 0.014),
            hold_orientation="upright",
            description="higher cylindrical body grasp for shelf and obstacle tasks",
        ),
    ),
    "mug_proxy": (
        GraspStrategy(
        name="cup_body_side_pinch",
        approach_axis="side_y",
        tcp_offset=(0.0, 0.0, 0.006),
        hold_orientation="upright",
        description="side pinch on the cup body, slightly above center",
    ),
        GraspStrategy(
            name="cup_high_body_pinch",
            approach_axis="side_y",
            tcp_offset=(0.0, 0.0, 0.018),
            hold_orientation="upright",
            description="higher body pinch for cup-like objects in clutter",
        ),
        GraspStrategy(
            name="cup_x_side_pinch",
            approach_axis="side_x",
            tcp_offset=(0.0, 0.0, 0.006),
            hold_orientation="upright",
            description="alternate cup side pinch across x axis",
        ),
    ),
    "capsule": (
        GraspStrategy(
        name="long_axis_side_pinch",
        approach_axis="side_x",
        tcp_offset=(0.0, 0.0, 0.0),
        hold_orientation="upright",
        description="side pinch across the short axis of an elongated capsule",
    ),
        GraspStrategy(
            name="capsule_center_side_y_pinch",
            approach_axis="side_y",
            tcp_offset=(0.0, 0.0, 0.0),
            hold_orientation="upright",
            description="alternate capsule side pinch across y axis",
        ),
    ),
    "plate": (
        GraspStrategy(
        name="plate_center_stabilized_pinch",
        approach_axis="top",
        tcp_offset=(0.0, 0.0, 0.0),
        hold_orientation="edge_stable",
        description=(
            "center-stabilized pinch for a thin plate proxy; true edge grasp "
            "is deferred until orientation-constrained IK is enabled"
        ),
    ),
        GraspStrategy(
            name="plate_edge_pinch_6d",
            approach_axis="edge",
            tcp_offset=(0.0, -0.028, 0.004),
            hold_orientation="edge_stable",
            description="orientation-constrained edge pinch near the rim of a thin plate proxy",
            tcp_quat=(0.7071067811865476, 0.7071067811865475, 0.0, 0.0),
            requires_orientation=True,
            orientation_tolerance=0.35,
            orientation_weight=0.45,
        ),
        GraspStrategy(
            name="plate_clearance_top_pinch",
            approach_axis="top",
            tcp_offset=(0.0, 0.0, 0.006),
            hold_orientation="edge_stable",
            description="top-biased plate pinch for shelf or support clearance",
        ),
    ),
    "ring": (
        GraspStrategy(
        name="ring_edge_pinch",
        approach_axis="edge",
        tcp_offset=(0.0, -0.020, 0.004),
        hold_orientation="edge_stable",
        description="edge pinch on a thin ring proxy",
    ),
        GraspStrategy(
            name="ring_center_top_pinch",
            approach_axis="top",
            tcp_offset=(0.0, 0.0, 0.004),
            hold_orientation="edge_stable",
            description="center top pinch for ring proxy when edge grasp is not feasible",
        ),
    ),
}


def strategies_for_object(object_type: str) -> tuple[GraspStrategy, ...]:
    return _STRATEGY_CANDIDATES.get(object_type, (_DEFAULT,))


def strategy_for_object(object_type: str) -> GraspStrategy:
    return strategies_for_object(object_type)[0]
