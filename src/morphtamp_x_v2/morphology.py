from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ArmDesign:
    name: str
    upper_scale: float
    forearm_scale: float
    wrist_scale: float
    base_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    description: str = "equivalent Panda morphology proxy"

    @property
    def morphology_cost(self) -> float:
        # Approximate Panda segment budget used only for design ranking. It is
        # intentionally separate from the MuJoCo robot body so we do not claim a
        # physically manufactured redesign unless a scaled MJCF validator is run.
        upper = 0.333 * self.upper_scale
        forearm = 0.316 * self.forearm_scale
        wrist = 0.211 * self.wrist_scale
        base = 0.15 * (abs(self.base_shift[0]) + abs(self.base_shift[1]))
        return upper + forearm + wrist + base

    @property
    def reach_proxy(self) -> float:
        return 0.86 + 0.333 * (self.upper_scale - 1.0) + 0.316 * (self.forearm_scale - 1.0) + 0.211 * (self.wrist_scale - 1.0)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "upper_scale": self.upper_scale,
            "forearm_scale": self.forearm_scale,
            "wrist_scale": self.wrist_scale,
            "base_shift": list(self.base_shift),
            "morphology_cost": self.morphology_cost,
            "reach_proxy": self.reach_proxy,
            "description": self.description,
        }


ARM_DESIGNS: dict[str, ArmDesign] = {
    "short_arm": ArmDesign("short_arm", 0.82, 0.82, 0.90, description="low-cost compact arm"),
    "nominal_panda": ArmDesign("nominal_panda", 1.0, 1.0, 1.0, description="reference Franka/Panda dimensions"),
    "long_forearm": ArmDesign("long_forearm", 1.0, 1.18, 1.0, description="forearm-biased reach extension"),
    "long_wrist": ArmDesign("long_wrist", 1.0, 1.0, 1.22, description="wrist-biased dexterity extension"),
    "high_reach_arm": ArmDesign("high_reach_arm", 1.12, 1.12, 1.05, description="longer arm for shelf / high tasks"),
    "compact_base_shift": ArmDesign(
        "compact_base_shift",
        0.92,
        0.92,
        0.95,
        base_shift=(0.05, 0.0, 0.0),
        description="compact arm with small base placement change",
    ),
}

ARM_DESIGN_TYPES = tuple(ARM_DESIGNS)


def arm_design(name: str) -> ArmDesign:
    try:
        return ARM_DESIGNS[name]
    except KeyError as error:
        raise ValueError(f"unknown arm design {name!r}; choices={', '.join(ARM_DESIGN_TYPES)}") from error


def pareto_designs(rows: Iterable[dict[str, Any]]) -> list[str]:
    candidates = list(rows)
    pareto: list[str] = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            other_better_or_equal = (
                other["success_rate"] >= row["success_rate"]
                and other["morphology_cost"] <= row["morphology_cost"]
                and other["mean_path_length"] <= row["mean_path_length"]
            )
            other_strictly_better = (
                other["success_rate"] > row["success_rate"]
                or other["morphology_cost"] < row["morphology_cost"]
                or other["mean_path_length"] < row["mean_path_length"]
            )
            if other_better_or_equal and other_strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(str(row["arm_design"]))
    return pareto
