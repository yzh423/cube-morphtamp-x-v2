# Cube MorphTAMP-X v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a clean MuJoCo/Franka Panda pick-place vertical slice with realistic grasp attach, carry, release, metrics, and viewer support.

**Architecture:** v2 is intentionally small and explicit. `models.py` defines the task/replay contracts, `scene_builder.py` creates one integrated MuJoCo XML, `planner.py` creates semantic pick-place phases, `replay.py` creates physically plausible attach/release frames, and `validator.py` runs metrics. Old `cube_morphtamp_x` remains untouched except as a reference.

**Tech Stack:** Python 3.10+, NumPy, MuJoCo Python when available, pytest.

## Global Constraints

- Keep v2 non-destructive: do not delete or rewrite `cube_morphtamp_x`.
- First vertical slice supports one cube, one table, one target, and one Panda/Franka XML.
- Object is static until grasp closure.
- Weld/attachment starts only after grasp closure begins.
- Object remains attached through lift, transport, descend, and early release.
- Weld/attachment is released only after gripper opens.
- Metrics must report success, object final error, lift height, and release stability.

---

### Task 1: Minimal v2 package

**Files:**
- Create: `cube_morphtamp_x_v2/src/morphtamp_x_v2/models.py`
- Create: `cube_morphtamp_x_v2/src/morphtamp_x_v2/planner.py`
- Create: `cube_morphtamp_x_v2/tests/test_planner.py`

**Interfaces:**
- Produces: `PickPlaceScenario`, `Phase`, `plan_pick_place(scenario) -> tuple[Phase, ...]`.

- [x] Write failing tests for phase ordering and static object/grasp semantics.
- [x] Implement dataclasses and planner.
- [x] Run tests.

### Task 2: Replay attach/release logic

**Files:**
- Create: `cube_morphtamp_x_v2/src/morphtamp_x_v2/replay.py`
- Create: `cube_morphtamp_x_v2/tests/test_replay.py`

**Interfaces:**
- Consumes: `PickPlaceScenario`, `Phase`.
- Produces: `ReplayFrame`, `build_replay_frames(scenario, phases)`.

- [x] Write failing tests for no object snap before grasp, rigid carry during transport, release only after gripper opens.
- [x] Implement deterministic replay frames.
- [x] Run tests.

### Task 3: MuJoCo scene builder

**Files:**
- Create: `cube_morphtamp_x_v2/src/morphtamp_x_v2/scene_builder.py`
- Create: `cube_morphtamp_x_v2/tests/test_scene_builder.py`

**Interfaces:**
- Produces: `build_pick_place_scene(panda_xml, scenario, output_xml) -> SceneBuildResult`.

- [x] Write XML structure tests for table, cube, target marker, freejoint, and equality weld.
- [x] Implement XML builder using copied/expanded Panda XML.
- [x] Run tests.

### Task 4: CLI and metrics

**Files:**
- Create: `cube_morphtamp_x_v2/src/morphtamp_x_v2/validator.py`
- Create: `cube_morphtamp_x_v2/src/morphtamp_x_v2/cli.py`
- Create: `cube_morphtamp_x_v2/tests/test_validator_cli.py`

**Interfaces:**
- Produces: `evaluate_replay(scenario, frames) -> dict`, CLI commands `plan`, `build-scene`, `run`.

- [x] Write tests for metrics and CLI JSON output.
- [x] Implement validator and CLI.
- [x] Run full tests.
