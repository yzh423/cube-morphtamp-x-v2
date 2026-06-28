from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .objects import OBJECT_TYPES


DEVELOPMENT_TASKS = ("tabletop_easy", "over_barrier", "precision_drop")
FIXED_TASKS = (
    "tabletop_easy",
    "long_transfer",
    "high_to_low",
    "low_to_high",
    "over_barrier",
    "narrow_slot",
    "shelf_pick",
    "shelf_place",
    "far_corner",
    "around_wall",
    "folded_transfer",
)
HELDOUT_TASKS = ("compound_shelf_barrier", "diagonal_reach_around", "under_bridge")


@dataclass(frozen=True)
class BenchmarkProtocol:
    name: str
    objects: tuple[str, ...]
    tasks: tuple[str, ...]
    auto_fit_panda: bool
    description: str
    evidence_role: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objects": list(self.objects),
            "tasks": list(self.tasks),
            "auto_fit_panda": self.auto_fit_panda,
            "description": self.description,
            "evidence_role": self.evidence_role,
        }


PROTOCOLS: dict[str, BenchmarkProtocol] = {
    "development_auto_fit": BenchmarkProtocol(
        name="development_auto_fit",
        objects=OBJECT_TYPES,
        tasks=DEVELOPMENT_TASKS,
        auto_fit_panda=True,
        description="developer smoke protocol; may auto-fit task coordinates to the Panda workspace",
        evidence_role="debugging_only",
    ),
    "fixed": BenchmarkProtocol(
        name="fixed",
        objects=OBJECT_TYPES,
        tasks=FIXED_TASKS,
        auto_fit_panda=False,
        description="fixed-coordinate benchmark protocol for main object-task evidence",
        evidence_role="main_evidence",
    ),
    "heldout_fixed": BenchmarkProtocol(
        name="heldout_fixed",
        objects=OBJECT_TYPES,
        tasks=HELDOUT_TASKS,
        auto_fit_panda=False,
        description="fixed held-out stress protocol; reserve for final validation after tuning",
        evidence_role="heldout_evidence",
    ),
}

PROTOCOL_TYPES = tuple(PROTOCOLS)


def benchmark_protocol(name: str) -> BenchmarkProtocol:
    try:
        return PROTOCOLS[name]
    except KeyError as error:
        raise ValueError(f"unknown benchmark protocol {name!r}; choices={', '.join(PROTOCOL_TYPES)}") from error


def resolve_protocol_inputs(
    *,
    protocol_name: str | None,
    objects: list[str] | tuple[str, ...] | None,
    tasks: list[str] | tuple[str, ...] | None,
    auto_fit_panda: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], bool, dict[str, Any] | None]:
    if protocol_name is None:
        return (
            tuple(OBJECT_TYPES if objects is None else objects),
            tuple(FIXED_TASKS if tasks is None else tasks),
            bool(auto_fit_panda),
            None,
        )
    protocol = benchmark_protocol(protocol_name)
    resolved_objects = tuple(protocol.objects if objects is None else objects)
    resolved_tasks = tuple(protocol.tasks if tasks is None else tasks)
    return resolved_objects, resolved_tasks, bool(protocol.auto_fit_panda), protocol.to_json_dict()
