from __future__ import annotations

import json

from morphtamp_x_v2.cli import parser
from morphtamp_x_v2.interactive_browse import BrowseState, build_browse_case


def test_browse_state_switches_tasks_and_objects_with_wrapping():
    state = BrowseState(objects=("cube", "sphere"), tasks=("easy", "hard", "shelf"))

    assert state.current == ("cube", "easy")

    assert state.apply_key(ord("n")) == "reload"
    assert state.current == ("cube", "hard")

    assert state.apply_key(ord("]")) == "reload"
    assert state.current == ("cube", "shelf")

    assert state.apply_key(ord("n")) == "reload"
    assert state.current == ("cube", "easy")

    assert state.apply_key(ord("o")) == "reload"
    assert state.current == ("sphere", "easy")

    assert state.apply_key(ord("i")) == "reload"
    assert state.current == ("cube", "easy")

    assert state.apply_key(ord("r")) == "replay"
    assert state.apply_key(ord("h")) == "help"
    assert state.apply_key(265) == "reload"
    assert state.current == ("cube", "hard")
    assert state.apply_key(264) == "reload"
    assert state.current == ("cube", "easy")
    assert state.apply_key(262) == "reload"
    assert state.current == ("sphere", "easy")
    assert state.apply_key(263) == "reload"
    assert state.current == ("cube", "easy")
    assert state.apply_key(999) == "none"
    assert state.apply_key(256) == "quit"


def test_cli_accepts_browse_options():
    args = parser().parse_args(
        [
            "browse",
            "--panda-xml",
            "panda.xml",
            "--objects",
            "cube",
            "mug_proxy",
            "--tasks",
            "tabletop_easy",
            "shelf_place",
            "--auto-fit-panda",
            "--output-dir",
            "results/browser",
        ]
    )

    assert args.command == "browse"
    assert args.objects == ["cube", "mug_proxy"]
    assert args.tasks == ["tabletop_easy", "shelf_place"]
    assert args.auto_fit_panda is True


def test_build_browse_case_uses_injected_builders(tmp_path):
    def scenario_builder(object_type: str, task_name: str):
        class Scenario:
            def to_json_dict(self):
                return {"object_type": object_type, "task_name": task_name}

        return Scenario()

    def phase_builder(_scenario):
        class Phase:
            def to_json_dict(self):
                return {"name": "home"}

        return (Phase(),)

    def frame_builder(_scenario, _phases):
        return ()

    def scene_builder(*, panda_xml, scenario, output_xml):
        output_xml.write_text("<mujoco/>", encoding="utf-8")

        class Scene:
            def to_json_dict(self):
                return {"xml": str(output_xml)}

        return Scene()

    replay_inputs = {}

    def replay_solver(xml_path, _frames, *, tolerance):
        replay_inputs["xml_path"] = xml_path

        class Replay:
            success = True

            def to_json_dict(self):
                return {"success": True, "joint_names": [], "frames": []}

        return Replay()

    case = build_browse_case(
        panda_xml=tmp_path / "panda.xml",
        output_dir=tmp_path,
        object_type="cube",
        task_name="tabletop_easy",
        scenario_builder=scenario_builder,
        phase_builder=phase_builder,
        frame_builder=frame_builder,
        scene_builder=scene_builder,
        replay_solver=replay_solver,
    )

    assert case.xml_path.exists()
    assert case.replay_path.exists()
    assert replay_inputs["xml_path"] == case.xml_path
    assert (tmp_path / "current_case.json").exists()
    payload = json.loads(case.replay_path.read_text(encoding="utf-8"))
    assert payload["scenario"]["object_type"] == "cube"
    current = json.loads((tmp_path / "current_case.json").read_text(encoding="utf-8"))
    assert current["object_type"] == "cube"
    assert current["task_name"] == "tabletop_easy"
