from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
import json

from .models import PickPlaceScenario
from .objects import object_half_height, object_spec
from .tasks import make_scenario


@dataclass(frozen=True)
class RoboCasaSnapshot:
    env_name: str
    robot: str
    behavior: str
    seed: int | None
    object_position: tuple[float, float, float] | None
    target_position: tuple[float, float, float] | None
    object_type_hint: str
    task_name_hint: str
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "env_name": self.env_name,
            "robot": self.robot,
            "behavior": self.behavior,
            "seed": self.seed,
            "object_position": None if self.object_position is None else list(self.object_position),
            "target_position": None if self.target_position is None else list(self.target_position),
            "object_type_hint": self.object_type_hint,
            "task_name_hint": self.task_name_hint,
            "metadata": self.metadata,
        }


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        values = tuple(float(item) for item in value)
    except TypeError:
        return None
    if len(values) != 3:
        return None
    return values


def task_hint_for_robocasa(env_name: str, behavior: str | None = None) -> dict[str, Any]:
    normalized_env = str(env_name)
    normalized_behavior = str(behavior or "")
    if normalized_env == "PickPlaceCoffee":
        if normalized_behavior == "counter_to_machine":
            return {
                "supported": True,
                "object_type": "mug_proxy",
                "task_name": "shelf_place",
                "reason": "RoboCasa coffee task: counter object is placed under machine dispenser.",
            }
        return {
            "supported": True,
            "object_type": "mug_proxy",
            "task_name": "shelf_pick",
            "reason": "RoboCasa coffee task: machine object is moved back to counter.",
        }
    return {
        "supported": False,
        "object_type": "cube",
        "task_name": "tabletop_easy",
        "reason": f"No explicit RoboCasa mapping for {normalized_env!r}; using conservative fallback.",
    }


def extract_snapshot_from_mapping(payload: Mapping[str, Any]) -> RoboCasaSnapshot:
    env_name = str(payload.get("env_name", "PickPlaceCoffee"))
    behavior = str(payload.get("behavior", "machine_to_counter"))
    hint = task_hint_for_robocasa(env_name, behavior)
    return RoboCasaSnapshot(
        env_name=env_name,
        robot=str(payload.get("robot", "PandaOmron")),
        behavior=behavior,
        seed=None if payload.get("seed") is None else int(payload["seed"]),
        object_position=_vector3(payload.get("object_position")),
        target_position=_vector3(payload.get("target_position")),
        object_type_hint=str(payload.get("object_type_hint", hint["object_type"])),
        task_name_hint=str(payload.get("task_name_hint", hint["task_name"])),
        metadata={
            "source": "mapping",
            "hint": hint,
            "fixture_positions": dict(payload.get("fixture_positions", {})),
            "raw_keys": sorted(str(key) for key in payload.keys()),
        },
    )


def scenario_from_robocasa_snapshot(
    snapshot: RoboCasaSnapshot,
    *,
    center_xy: tuple[float, float] | None = None,
) -> PickPlaceScenario:
    """Project a RoboCasa semantic snapshot into the local MorphTAMP task library.

    RoboCasa kitchen coordinates are scene-layout specific and often far from the
    Panda base frame used by the compact benchmark. The projection therefore
    keeps the semantic object/task choice while using the validated local task
    geometry. This is deliberate: it lets RoboCasa provide task semantics without
    claiming full RoboCasa dynamics execution.
    """

    scenario = make_scenario(
        object_type=snapshot.object_type_hint,
        task_name=snapshot.task_name_hint,
        center_xy=center_xy,
    )
    description = (
        "RoboCasa-derived "
        f"{snapshot.env_name}/{snapshot.behavior} semantic projection; "
        f"source object={snapshot.object_position}, target={snapshot.target_position}."
    )
    return replace(
        scenario,
        object_description=f"{description} {scenario.object_description}",
        task_description=f"{description} {scenario.task_description}",
    )


def _body_position(environment: Any, body_name: str) -> tuple[float, float, float] | None:
    sim = getattr(environment, "sim", None)
    if sim is None:
        return None
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None
    body_id = None
    if hasattr(model, "body_name2id"):
        try:
            body_id = model.body_name2id(body_name)
        except Exception:
            body_id = None
    if body_id is None and hasattr(model, "body"):
        try:
            body_id = model.body(body_name).id
        except Exception:
            body_id = None
    if body_id is None:
        return None
    body_xpos = getattr(data, "body_xpos", None)
    if body_xpos is None:
        return None
    return _vector3(body_xpos[body_id])


def _site_position(environment: Any, site_name: str) -> tuple[float, float, float] | None:
    sim = getattr(environment, "sim", None)
    if sim is None:
        return None
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None
    site_id = None
    if hasattr(model, "site_name2id"):
        try:
            site_id = model.site_name2id(site_name)
        except Exception:
            site_id = None
    if site_id is None and hasattr(model, "site"):
        try:
            site_id = model.site(site_name).id
        except Exception:
            site_id = None
    if site_id is None:
        return None
    site_xpos = getattr(data, "site_xpos", None)
    if site_xpos is None:
        return None
    return _vector3(site_xpos[site_id])


def _root_body(candidate: Any, fallback: str) -> str:
    if candidate is None:
        return fallback
    if isinstance(candidate, str):
        return "obj_main" if candidate == "obj" else candidate
    return str(getattr(candidate, "root_body", fallback))


def _fixture_size(fixture: Any) -> tuple[float, float, float] | None:
    if fixture is None:
        return None
    size = getattr(fixture, "size", None)
    if size is None:
        width = getattr(fixture, "width", None)
        depth = getattr(fixture, "depth", None)
        height = getattr(fixture, "height", None)
        if width is not None and depth is not None and height is not None:
            size = (width, depth, height)
    return _vector3(size)


def _counter_top_target(
    *,
    counter_position: tuple[float, float, float] | None,
    counter_size: tuple[float, float, float] | None,
    object_type_hint: str,
) -> tuple[float, float, float] | None:
    if counter_position is None or counter_size is None:
        return counter_position
    object_height = object_half_height(object_spec(object_type_hint))
    return (
        float(counter_position[0]),
        float(counter_position[1]),
        float(counter_position[2] + 0.5 * counter_size[2] + object_height),
    )


def _task_object(environment: Any) -> Any:
    objects = getattr(environment, "objects", None)
    if isinstance(objects, Mapping):
        return objects.get("obj") or next(iter(objects.values()), None)
    if isinstance(objects, (list, tuple)):
        for item in objects:
            if not isinstance(item, str):
                return item
        return objects[0] if objects else None
    return getattr(environment, "obj", None)


def extract_snapshot_from_environment(
    environment: Any,
    *,
    env_name: str,
    robot: str,
    behavior: str | None = None,
    seed: int | None = None,
) -> RoboCasaSnapshot:
    behavior = str(behavior or getattr(environment, "behavior", "machine_to_counter"))
    hint = task_hint_for_robocasa(env_name, behavior)
    object_type_hint = str(hint["object_type"])
    task_object = _task_object(environment)
    object_body = _root_body(task_object, "obj_main")
    object_position = _body_position(environment, object_body)

    fixture_positions: dict[str, list[float] | None] = {}
    fixture_sizes: dict[str, list[float] | None] = {}
    for fixture_name in ("coffee_machine", "counter"):
        fixture = getattr(environment, fixture_name, None)
        body_name = _root_body(fixture, f"{fixture_name}_main")
        position = _body_position(environment, body_name)
        fixture_positions[fixture_name] = None if position is None else list(position)
        size = _fixture_size(fixture)
        fixture_sizes[fixture_name] = None if size is None else list(size)

    coffee_site = _site_position(environment, "coffee_machine_left_group_receptacle_place_site")
    counter_position = _vector3(fixture_positions.get("counter"))
    counter_size = _vector3(fixture_sizes.get("counter"))
    if behavior == "counter_to_machine":
        target_position = coffee_site or _vector3(fixture_positions.get("coffee_machine"))
        target_source = "coffee_receptacle_site" if coffee_site is not None else "coffee_machine_body"
    else:
        target_position = _counter_top_target(
            counter_position=counter_position,
            counter_size=counter_size,
            object_type_hint=object_type_hint,
        )
        target_source = "counter_top" if counter_size is not None else "counter_body"

    return RoboCasaSnapshot(
        env_name=env_name,
        robot=robot,
        behavior=behavior,
        seed=seed,
        object_position=object_position,
        target_position=target_position,
        object_type_hint=object_type_hint,
        task_name_hint=hint["task_name"],
        metadata={
            "source": "robocasa_live_environment",
            "hint": hint,
            "object_body": object_body,
            "fixture_positions": fixture_positions,
            "fixture_sizes": fixture_sizes,
            "target_source": target_source,
            "coffee_receptacle_site": None if coffee_site is None else list(coffee_site),
            "evidence_scope": (
                "RoboCasa semantic snapshot projected into MorphTAMP-X local "
                "MuJoCo/Panda task geometry; not full RoboCasa policy execution."
            ),
        },
    )


def create_live_robocasa_snapshot(
    *,
    env_name: str = "PickPlaceCoffee",
    robot: str = "PandaOmron",
    behavior: str = "machine_to_counter",
    seed: int | None = 42,
) -> RoboCasaSnapshot:
    try:
        import robocasa  # noqa: F401
        import robosuite as suite
    except Exception as error:  # pragma: no cover - depends on optional WSL env
        raise RuntimeError(
            "RoboCasa / robosuite are not importable in this Python environment. "
            "Run this command inside the WSL robocasa conda environment."
        ) from error

    environment = suite.make(
        env_name=env_name,
        robots=[robot],
        behavior=behavior,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=seed,
    )
    try:
        environment.reset()
        return extract_snapshot_from_environment(
            environment,
            env_name=env_name,
            robot=robot,
            behavior=behavior,
            seed=seed,
        )
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def write_robocasa_snapshot_files(
    snapshot: RoboCasaSnapshot,
    *,
    output_json: str | Path,
    output_scenario: str | Path | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "snapshot": snapshot.to_json_dict(),
        "projected_scenario": scenario_from_robocasa_snapshot(snapshot).to_json_dict(),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if output_scenario is not None:
        output_scenario = Path(output_scenario)
        output_scenario.parent.mkdir(parents=True, exist_ok=True)
        output_scenario.write_text(
            json.dumps(payload["projected_scenario"], indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return payload
