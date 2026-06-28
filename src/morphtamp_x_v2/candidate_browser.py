from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal


CandidateAction = Literal["reload", "replay", "help", "quit", "none"]


@dataclass(frozen=True)
class CandidateReplay:
    index: int
    strategy_name: str
    selected: bool
    success: bool
    xml_path: Path
    replay_path: Path


@dataclass
class CandidateSwitchState:
    candidates: tuple[CandidateReplay, ...]
    candidate_index: int = 0

    @property
    def current(self) -> CandidateReplay:
        return self.candidates[self.candidate_index]

    def apply_key(self, key: int) -> CandidateAction:
        if key == 256:  # Escape
            return "quit"
        if key in {262, 265}:  # right / up
            self.candidate_index = (self.candidate_index + 1) % len(self.candidates)
            return "reload"
        if key in {263, 264}:  # left / down
            self.candidate_index = (self.candidate_index - 1) % len(self.candidates)
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
        if char in {"n", "]", "d"}:
            self.candidate_index = (self.candidate_index + 1) % len(self.candidates)
            return "reload"
        if char in {"p", "[", "a"}:
            self.candidate_index = (self.candidate_index - 1) % len(self.candidates)
            return "reload"
        if char.isdigit() and char != "0":
            index = int(char) - 1
            if 0 <= index < len(self.candidates):
                self.candidate_index = index
                return "reload"
        return "none"


def load_candidate_replays(manifest_path: str | Path) -> tuple[tuple[CandidateReplay, ...], int]:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent.parent
    candidates = []
    selected_index = 0
    for offset, row in enumerate(payload.get("candidates", ())):
        xml = row.get("scene_xml")
        replay = row.get("replay_json")
        if not xml or not replay:
            continue
        candidate = CandidateReplay(
            index=int(row.get("index", offset + 1)),
            strategy_name=str(row.get("strategy_name", f"candidate_{offset + 1}")),
            selected=bool(row.get("selected", False)),
            success=bool(row.get("success", False)),
            xml_path=base_dir / str(xml),
            replay_path=base_dir / str(replay),
        )
        if candidate.selected:
            selected_index = len(candidates)
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"candidate manifest contains no playable candidates: {manifest_path}")
    return tuple(candidates), selected_index


def help_text() -> str:
    return (
        "Candidate viewer keys: n/d/] next candidate, p/a/[ previous candidate, "
        "1-9 jump to candidate, r replay current, h help, q/Esc quit"
    )


def candidate_banner(candidate: CandidateReplay, count: int) -> str:
    status = "selected" if candidate.selected else ("success" if candidate.success else "failed")
    return (
        "\n"
        "============================================================\n"
        f" CANDIDATE {candidate.index}/{count}  {candidate.strategy_name}  [{status}]\n"
        "============================================================"
    )


def launch_candidate_viewer(
    manifest_path: str | Path,
    *,
    fps: float = 60.0,
    playback_speed: float = 0.55,
) -> None:
    from .viewer import launch_viewer

    candidates, selected_index = load_candidate_replays(manifest_path)
    state = CandidateSwitchState(candidates=candidates, candidate_index=selected_index)
    print(help_text())
    while True:
        candidate = state.current
        print(candidate_banner(candidate, len(candidates)))
        print(f"scene : {candidate.xml_path}")
        print(f"replay: {candidate.replay_path}")
        requested: dict[str, CandidateAction] = {"action": "none"}

        def key_callback(key: int) -> None:
            action = state.apply_key(key)
            if action == "help":
                print(help_text())
            elif action in {"reload", "quit", "replay"}:
                requested["action"] = action

        launch_viewer(
            candidate.xml_path,
            candidate.replay_path,
            fps=fps,
            playback_speed=playback_speed,
            key_callback=key_callback,
            stop_requested=lambda: requested["action"] in {"reload", "quit", "replay"},
        )
        if requested["action"] == "quit":
            return
