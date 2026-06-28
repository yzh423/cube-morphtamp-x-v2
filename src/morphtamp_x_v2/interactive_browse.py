from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Literal

from .models import PickPlaceScenario
from .panda_ik import auto_fit_panda_scenario, solve_joint_replay
from .planner import plan_pick_place
from .replay import build_replay_frames
from .scene_builder import build_pick_place_scene
from .tasks import make_scenario


BrowseAction = Literal["reload", "replay", "help", "quit", "none"]


@dataclass
class BrowseState:
    objects: tuple[str, ...]
    tasks: tuple[str, ...]
    object_index: int = 0
    task_index: int = 0

    @property
    def current(self) -> tuple[str, str]:
        return self.objects[self.object_index], self.tasks[self.task_index]

    def apply_key(self, key: int) -> BrowseAction:
        # GLFW special keys used by mujoco.viewer.
        if key == 256:  # Escape
            return "quit"
        if key == 265:  # Up
            self.task_index = (self.task_index + 1) % len(self.tasks)
            return "reload"
        if key == 264:  # Down
            self.task_index = (self.task_index - 1) % len(self.tasks)
            return "reload"
        if key == 262:  # Right
            self.object_index = (self.object_index + 1) % len(self.objects)
            return "reload"
        if key == 263:  # Left
            self.object_index = (self.object_index - 1) % len(self.objects)
            return "reload"
        if key < 0 or key > 255:
            return "none"
        char = chr(key).lower()
        if char == "q":
            return "quit"
        if char == "h":
            return "help"
        if char == "r":
            return "replay"
        if char in {"n", "]"}:
            self.task_index = (self.task_index + 1) % len(self.tasks)
            return "reload"
        if char in {"p", "["}:
            self.task_index = (self.task_index - 1) % len(self.tasks)
            return "reload"
        if char == "o":
            self.object_index = (self.object_index + 1) % len(self.objects)
            return "reload"
        if char == "i":
            self.object_index = (self.object_index - 1) % len(self.objects)
            return "reload"
        return "none"


@dataclass(frozen=True)
class BrowseCase:
    object_type: str
    task_name: str
    xml_path: Path
    replay_path: Path


def _case_dir(output_dir: Path, object_type: str, task_name: str) -> Path:
    safe = f"{object_type}_{task_name}".replace("/", "_").replace("\\", "_")
    return output_dir / safe


def build_browse_case(
    *,
    panda_xml: Path,
    output_dir: Path,
    object_type: str,
    task_name: str,
    auto_fit_panda: bool = False,
    position_tolerance: float = 0.035,
    scenario_builder: Callable[[str, str], PickPlaceScenario] | None = None,
    phase_builder: Callable[[Any], Any] = plan_pick_place,
    frame_builder: Callable[[Any, Any], Any] = build_replay_frames,
    scene_builder: Callable[..., Any] = build_pick_place_scene,
    replay_solver: Callable[..., Any] = solve_joint_replay,
) -> BrowseCase:
    case_dir = _case_dir(output_dir, object_type, task_name)
    case_dir.mkdir(parents=True, exist_ok=True)
    xml_path = case_dir / "scene.xml"
    replay_path = case_dir / "replay.json"
    if scenario_builder is None:
        if auto_fit_panda:
            scenario = auto_fit_panda_scenario(
                panda_xml,
                object_type=object_type,
                task_name=task_name,
            )
        else:
            scenario = make_scenario(object_type=object_type, task_name=task_name)
    else:
        scenario = scenario_builder(object_type, task_name)
    phases = phase_builder(scenario)
    frames = frame_builder(scenario, phases)
    scene = scene_builder(panda_xml=panda_xml, scenario=scenario, output_xml=xml_path)
    joint_replay = replay_solver(xml_path, frames, tolerance=position_tolerance)
    payload = {
        "schema_version": 1,
        "case_id": f"{object_type}:{task_name}",
        "object_type": object_type,
        "task_name": task_name,
        "scenario": scenario.to_json_dict(),
        "scene": scene.to_json_dict(),
        "phases": [phase.to_json_dict() for phase in phases],
        "frames": [frame.to_json_dict() for frame in frames],
        "joint_replay": joint_replay.to_json_dict(),
    }
    replay_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    current_payload = {
        "schema_version": 1,
        "case_id": f"{object_type}:{task_name}",
        "object_type": object_type,
        "task_name": task_name,
        "xml_path": str(xml_path),
        "replay_path": str(replay_path),
        "success": bool(getattr(joint_replay, "success", False)),
    }
    (output_dir / "current_case.json").write_text(
        json.dumps(current_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return BrowseCase(
        object_type=object_type,
        task_name=task_name,
        xml_path=xml_path,
        replay_path=replay_path,
    )


def help_text() -> str:
    return (
        "MorphTAMP-X browser keys: "
        "n/] next task, p/[ previous task, o next object, i previous object, "
        "r replay current, h help, q/Esc quit"
    )


def case_banner(object_type: str, task_name: str) -> str:
    return (
        "\n"
        "============================================================\n"
        f" CURRENT CASE  object={object_type}   task={task_name}\n"
        "============================================================"
    )


def launch_browse_viewer(
    *,
    panda_xml: Path,
    output_dir: Path,
    objects: tuple[str, ...],
    tasks: tuple[str, ...],
    auto_fit_panda: bool = False,
    position_tolerance: float = 0.035,
    fps: float = 60.0,
    playback_speed: float = 0.55,
) -> None:
    try:
        import mujoco.viewer
    except ModuleNotFoundError as error:
        raise RuntimeError("browse viewer requires mujoco") from error
    from .viewer import launch_viewer

    state = BrowseState(objects=objects, tasks=tasks)
    print(help_text())
    while True:
        object_type, task_name = state.current
        case = build_browse_case(
            panda_xml=panda_xml,
            output_dir=output_dir,
            object_type=object_type,
            task_name=task_name,
            auto_fit_panda=auto_fit_panda,
            position_tolerance=position_tolerance,
        )
        print(case_banner(object_type, task_name))
        print(f"current_case: {output_dir / 'current_case.json'}")
        requested: dict[str, BrowseAction] = {"action": "none"}

        def key_callback(key: int) -> None:
            action = state.apply_key(key)
            if action == "help":
                print(help_text())
            elif action in {"reload", "quit", "replay"}:
                requested["action"] = action

        launch_viewer(
            case.xml_path,
            case.replay_path,
            fps=fps,
            playback_speed=playback_speed,
            key_callback=key_callback,
            stop_requested=lambda: requested["action"] in {"reload", "quit", "replay"},
        )
        if requested["action"] == "quit":
            return
        # reload and replay both restart the passive viewer loop. Replay keeps
        # the same object/task; reload uses the new BrowseState indices.
