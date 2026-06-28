from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    geom_type: str
    size: tuple[float, ...]
    mass: float
    open_width: float
    closed_width: float
    grasp_clearance: float
    release_clearance: float
    grasp_tcp_offset: tuple[float, float, float]
    material_rgba: str
    description: str


OBJECT_SPECS: dict[str, ObjectSpec] = {
    "cube": ObjectSpec(
        name="cube",
        geom_type="box",
        size=(0.02, 0.02, 0.02),
        mass=0.10,
        open_width=0.078,
        closed_width=0.064,
        grasp_clearance=0.12,
        release_clearance=0.10,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.05 0.35 1 1",
        description="box object; side pinch with flat finger pads",
    ),
    "sphere": ObjectSpec(
        name="sphere",
        geom_type="sphere",
        size=(0.022,),
        mass=0.08,
        open_width=0.078,
        closed_width=0.050,
        grasp_clearance=0.13,
        release_clearance=0.11,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.05 0.75 0.35 1",
        description="round object; symmetric center pinch",
    ),
    "cylinder": ObjectSpec(
        name="cylinder",
        geom_type="cylinder",
        size=(0.018, 0.035),
        mass=0.12,
        open_width=0.076,
        closed_width=0.042,
        grasp_clearance=0.13,
        release_clearance=0.10,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.95 0.55 0.08 1",
        description="upright cylinder/cup proxy; side pinch near mid-height",
    ),
    "plate": ObjectSpec(
        name="plate",
        geom_type="cylinder",
        size=(0.032, 0.006),
        mass=0.10,
        open_width=0.078,
        closed_width=0.070,
        grasp_clearance=0.10,
        release_clearance=0.08,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.65 0.55 0.95 1",
        description="thin plate/tray proxy; edge-sensitive shallow grasp",
    ),
    "mug_proxy": ObjectSpec(
        name="mug_proxy",
        geom_type="cylinder",
        size=(0.024, 0.040),
        mass=0.14,
        open_width=0.078,
        closed_width=0.058,
        grasp_clearance=0.14,
        release_clearance=0.11,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.85 0.30 0.12 1",
        description="mug/cup body proxy; side grasp on cylindrical body",
    ),
    "bowl_proxy": ObjectSpec(
        name="bowl_proxy",
        geom_type="sphere",
        size=(0.030,),
        mass=0.12,
        open_width=0.078,
        closed_width=0.068,
        grasp_clearance=0.13,
        release_clearance=0.10,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.80 0.65 0.25 1",
        description="small bowl proxy; broad symmetric side grasp",
    ),
    "capsule": ObjectSpec(
        name="capsule",
        geom_type="capsule",
        size=(0.016, 0.050),
        mass=0.10,
        open_width=0.078,
        closed_width=0.044,
        grasp_clearance=0.13,
        release_clearance=0.10,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.25 0.75 0.95 1",
        description="capsule/handle proxy; elongated side grasp",
    ),
    "tall_box": ObjectSpec(
        name="tall_box",
        geom_type="box",
        size=(0.018, 0.018, 0.050),
        mass=0.12,
        open_width=0.078,
        closed_width=0.058,
        grasp_clearance=0.16,
        release_clearance=0.12,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.60 0.30 0.90 1",
        description="tall box proxy; tests height clearance and wrist posture",
    ),
    "flat_box": ObjectSpec(
        name="flat_box",
        geom_type="box",
        size=(0.040, 0.026, 0.008),
        mass=0.08,
        open_width=0.078,
        closed_width=0.064,
        grasp_clearance=0.10,
        release_clearance=0.08,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.20 0.85 0.85 1",
        description="flat object proxy; tests low-profile pick and accurate placement",
    ),
    "ring": ObjectSpec(
        name="ring",
        geom_type="cylinder",
        size=(0.026, 0.010),
        mass=0.06,
        open_width=0.078,
        closed_width=0.060,
        grasp_clearance=0.11,
        release_clearance=0.09,
        grasp_tcp_offset=(0.0, 0.0, 0.0),
        material_rgba="0.95 0.85 0.10 1",
        description="ring proxy represented as a thin cylinder; tests small target transfer",
    ),
}

OBJECT_TYPES = tuple(OBJECT_SPECS)


def object_spec(name: str) -> ObjectSpec:
    try:
        return OBJECT_SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown object type {name!r}; choose from {', '.join(OBJECT_TYPES)}") from error


def object_half_height(spec: ObjectSpec) -> float:
    if spec.geom_type == "box":
        return float(spec.size[2])
    if spec.geom_type == "sphere":
        return float(spec.size[0])
    if spec.geom_type in {"cylinder", "capsule"}:
        return float(spec.size[1])
    raise ValueError(f"unsupported geom type {spec.geom_type!r}")


def object_grasp_width(spec: ObjectSpec) -> float:
    if spec.geom_type == "box":
        return 2.0 * float(spec.size[1])
    if spec.geom_type == "sphere":
        return 2.0 * float(spec.size[0])
    if spec.geom_type in {"cylinder", "capsule"}:
        return 2.0 * float(spec.size[0])
    raise ValueError(f"unsupported geom type {spec.geom_type!r}")
