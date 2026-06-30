from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import PickPlaceScenario, ReplayFrame
from .pose_math import IDENTITY_QUAT, quat_angle_error, quat_error_vector
from .tasks import make_scenario


DEFAULT_FINGERTIP_EXTENSION = 0.024


@dataclass(frozen=True)
class PandaJointFrame:
    phase_name: str
    q: tuple[float, ...]
    gripper_width: float
    object_position: tuple[float, float, float]
    object_attached: bool
    weld_active: bool
    tcp_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    position_error: float
    target_quat: tuple[float, float, float, float] = IDENTITY_QUAT
    tcp_quat: tuple[float, float, float, float] = IDENTITY_QUAT
    orientation_error: float = 0.0
    collision_count: int = 0
    joint_margin: float = 0.0
    condition_number: float | None = None
    min_singular_value: float | None = None
    object_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "q": list(self.q),
            "gripper_width": self.gripper_width,
            "object_position": list(self.object_position),
            "object_attached": self.object_attached,
            "weld_active": self.weld_active,
            "tcp_position": list(self.tcp_position),
            "target_position": list(self.target_position),
            "position_error": self.position_error,
            "target_quat": list(self.target_quat),
            "tcp_quat": list(self.tcp_quat),
            "orientation_error": self.orientation_error,
            "collision_count": self.collision_count,
            "joint_margin": self.joint_margin,
            "condition_number": self.condition_number,
            "min_singular_value": self.min_singular_value,
            "object_quat": list(self.object_quat),
        }


@dataclass(frozen=True)
class PandaIKReplay:
    success: bool
    joint_names: tuple[str, ...]
    eef_body: str | None
    eef_site: str | None
    max_position_error: float
    max_orientation_error: float
    joint_path_length: float
    max_joint_step: float
    energy_proxy: float
    smoothness_proxy: float
    frames: tuple[PandaJointFrame, ...]
    failure_reasons: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "joint_names": list(self.joint_names),
            "eef_body": self.eef_body,
            "eef_site": self.eef_site,
            "max_position_error": self.max_position_error,
            "max_orientation_error": self.max_orientation_error,
            "joint_path_length": self.joint_path_length,
            "max_joint_step": self.max_joint_step,
            "energy_proxy": self.energy_proxy,
            "smoothness_proxy": self.smoothness_proxy,
            "failure_reasons": list(self.failure_reasons),
            "frames": [frame.to_json_dict() for frame in self.frames],
        }


def _tuple3(vector: np.ndarray) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def joint_motion_costs(q_sequence: tuple[tuple[float, ...], ...]) -> dict[str, float]:
    if len(q_sequence) < 2:
        return {
            "joint_path_length": 0.0,
            "max_joint_step": 0.0,
            "energy_proxy": 0.0,
            "smoothness_proxy": 0.0,
        }
    q_arrays = [np.asarray(q, dtype=float) for q in q_sequence]
    deltas = [q_arrays[index] - q_arrays[index - 1] for index in range(1, len(q_arrays))]
    step_norms = [float(np.linalg.norm(delta)) for delta in deltas]
    accelerations = [
        deltas[index] - deltas[index - 1]
        for index in range(1, len(deltas))
    ]
    return {
        "joint_path_length": float(sum(step_norms)),
        "max_joint_step": float(max(step_norms)),
        "energy_proxy": float(sum(float(np.dot(delta, delta)) for delta in deltas)),
        "smoothness_proxy": float(sum(float(np.dot(accel, accel)) for accel in accelerations)),
    }


class MuJoCoPandaIK:
    def __init__(self, panda_xml: str | Path) -> None:
        try:
            import mujoco
        except ModuleNotFoundError as error:
            raise RuntimeError("Panda IK requires the mujoco Python package") from error
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(panda_xml))
        self.data = mujoco.MjData(self.model)
        self.joint_names = self._detect_arm_joint_names()
        if len(self.joint_names) != 7:
            raise ValueError(f"expected 7 Panda arm joints, found {self.joint_names}")
        self.joint_ids = tuple(
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            for name in self.joint_names
        )
        self.qpos_addresses = np.asarray([self.model.jnt_qposadr[j] for j in self.joint_ids], dtype=int)
        self.dof_addresses = np.asarray([self.model.jnt_dofadr[j] for j in self.joint_ids], dtype=int)
        self.ranges = np.asarray([self.model.jnt_range[j] for j in self.joint_ids], dtype=float)
        self.eef_site, self.eef_body = self._detect_eef()
        self.site_id = None
        self.body_id = None
        if self.eef_site is not None:
            self.site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.eef_site))
        elif self.eef_body is not None:
            self.body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.eef_body))
        else:
            raise ValueError("could not detect Panda end-effector site or body")
        self.gripper_qpos_addresses = self._detect_gripper_qpos_addresses()
        self.gripper_body_ids = self._detect_gripper_body_ids()
        self.gripper_geom_ids = self._detect_gripper_geom_ids()
        self.object_qpos_address = self._detect_object_qpos_address()
        self.object_qvel_address = self._detect_object_qvel_address()
        self.mujoco.mj_forward(self.model, self.data)
        self.fingertip_extension = DEFAULT_FINGERTIP_EXTENSION

    @property
    def neutral_q(self) -> np.ndarray:
        q = np.zeros(7, dtype=float)
        return np.clip(q, self.ranges[:, 0], self.ranges[:, 1])

    @property
    def ready_q(self) -> np.ndarray:
        q = np.asarray([0.0, -0.45, 0.0, -2.10, 0.0, 1.65, 0.785], dtype=float)
        return np.clip(q, self.ranges[:, 0], self.ranges[:, 1])

    @property
    def center_q(self) -> np.ndarray:
        return 0.5 * (self.ranges[:, 0] + self.ranges[:, 1])

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_addresses] = np.asarray(q, dtype=float)
        self.mujoco.mj_forward(self.model, self.data)

    def set_gripper_width(self, width: float) -> None:
        if len(self.gripper_qpos_addresses):
            self.data.qpos[self.gripper_qpos_addresses] = float(width) / float(len(self.gripper_qpos_addresses))
            self.mujoco.mj_forward(self.model, self.data)

    def set_object_position(
        self,
        position: tuple[float, float, float] | None,
        quat: tuple[float, float, float, float] | None = None,
    ) -> None:
        if position is None or self.object_qpos_address is None:
            return
        if quat is None:
            quat = (1.0, 0.0, 0.0, 0.0)
        address = int(self.object_qpos_address)
        self.data.qpos[address : address + 7] = np.asarray(
            [position[0], position[1], position[2], quat[0], quat[1], quat[2], quat[3]],
            dtype=float,
        )
        if self.object_qvel_address is not None:
            self.data.qvel[int(self.object_qvel_address) : int(self.object_qvel_address) + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def eef_position(self) -> np.ndarray:
        if self.site_id is not None:
            return np.asarray(self.data.site_xpos[self.site_id], dtype=float).copy()
        assert self.body_id is not None
        return np.asarray(self.data.xpos[self.body_id], dtype=float).copy()

    def task_position(self) -> np.ndarray:
        if len(self.gripper_geom_ids) >= 2:
            base = np.mean(np.asarray(self.data.geom_xpos[self.gripper_geom_ids], dtype=float), axis=0)
            if self.eef_body is not None:
                hand = self.eef_position()
                direction = base - hand
                norm = float(np.linalg.norm(direction))
                if norm > 1e-9:
                    base = base + self.fingertip_extension * direction / norm
            return base.copy()
        if len(self.gripper_body_ids) >= 2:
            base = np.mean(np.asarray(self.data.xpos[self.gripper_body_ids], dtype=float), axis=0)
            if self.eef_body is not None:
                hand = self.eef_position()
                direction = base - hand
                norm = float(np.linalg.norm(direction))
                if norm > 1e-9:
                    base = base + self.fingertip_extension * direction / norm
            return base.copy()
        return self.eef_position()

    def task_quat(self) -> tuple[float, float, float, float]:
        if self.site_id is not None:
            quat = np.zeros(4, dtype=float)
            mat = np.asarray(self.data.site_xmat[self.site_id], dtype=float)
            self.mujoco.mju_mat2Quat(quat, mat)
            return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        assert self.body_id is not None
        quat = np.asarray(self.data.xquat[self.body_id], dtype=float)
        return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        if len(self.gripper_geom_ids) >= 2:
            jacobians = []
            for geom_id in self.gripper_geom_ids:
                self.mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, int(geom_id))
                jacobians.append(jacp[:, self.dof_addresses].copy())
            return np.mean(np.asarray(jacobians, dtype=float), axis=0)
        if len(self.gripper_body_ids) >= 2:
            jacobians = []
            for body_id in self.gripper_body_ids:
                self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, int(body_id))
                jacobians.append(jacp[:, self.dof_addresses].copy())
            return np.mean(np.asarray(jacobians, dtype=float), axis=0)
        if self.site_id is not None:
            self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, int(self.site_id))
        else:
            assert self.body_id is not None
            self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, int(self.body_id))
        return jacp[:, self.dof_addresses]

    def rotational_jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        if len(self.gripper_geom_ids) >= 2:
            jacobians = []
            for geom_id in self.gripper_geom_ids:
                self.mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, int(geom_id))
                jacobians.append(jacr[:, self.dof_addresses].copy())
            return np.mean(np.asarray(jacobians, dtype=float), axis=0)
        if len(self.gripper_body_ids) >= 2:
            jacobians = []
            for body_id in self.gripper_body_ids:
                self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, int(body_id))
                jacobians.append(jacr[:, self.dof_addresses].copy())
            return np.mean(np.asarray(jacobians, dtype=float), axis=0)
        if self.site_id is not None:
            self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, int(self.site_id))
        else:
            assert self.body_id is not None
            self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, int(self.body_id))
        return jacr[:, self.dof_addresses]

    def joint_margin(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=float)
        lower = q - self.ranges[:, 0]
        upper = self.ranges[:, 1] - q
        return float(np.min(np.minimum(lower, upper)))

    def condition_number(self) -> float | None:
        jacobian = self.jacobian()
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if len(singular_values) == 0 or float(np.min(singular_values)) <= 1e-12:
            return None
        return float(np.max(singular_values) / np.min(singular_values))

    def min_singular_value(self) -> float | None:
        singular_values = np.linalg.svd(self.jacobian(), compute_uv=False)
        if len(singular_values) == 0:
            return None
        return float(np.min(singular_values))

    def collision_violation_count(self) -> int:
        count = 0
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            if float(contact.dist) > 0.001:
                continue
            names = self._contact_names(int(contact.geom1), int(contact.geom2))
            if self._is_hard_collision(names):
                count += 1
        return count

    def _contact_names(self, geom1: int, geom2: int) -> set[str]:
        names: set[str] = set()
        for geom_id in (geom1, geom2):
            geom_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name:
                names.add(str(geom_name))
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name:
                names.add(str(body_name))
        return names

    def _is_hard_collision(self, names: set[str]) -> bool:
        lowered = " ".join(names).lower()
        is_object = "v2_cube" in names or "v2_object_geom" in names
        is_gripper = any(token in lowered for token in ("finger", "gripper", "hand"))
        is_support = "v2_table" in names or any(name.startswith("v2_support_") for name in names)
        if "v2_obstacle" in names:
            allowed = {"world", "v2_table", "v2_place_target", "v2_obstacle"}
            allowed.update(name for name in names if name.startswith("v2_support_"))
            return any(name not in allowed for name in names)
        if is_support:
            # Object-support contact is physical support. Gripper/object contact
            # is part of grasp. Robot/support contact is not allowed.
            return not is_object
        if is_object and is_gripper:
            return False
        return False

    def solve(
        self,
        target: tuple[float, float, float],
        *,
        initial_q: np.ndarray,
        tolerance: float = 0.03,
        max_iterations: int = 160,
        damping: float = 1e-3,
        gripper_width: float | None = None,
        object_position: tuple[float, float, float] | None = None,
        object_quat: tuple[float, float, float, float] | None = None,
        target_quat: tuple[float, float, float, float] | None = None,
        orientation_weight: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        best_q = np.asarray(initial_q, dtype=float).copy()
        best_error = float("inf")
        best_orientation_error = float("inf")
        best_score = float("inf")
        target_quat = target_quat if target_quat is not None else None
        orientation_weight = max(float(orientation_weight), 0.0)
        for seed in self._seeds(initial_q):
            q = np.clip(seed.copy(), self.ranges[:, 0], self.ranges[:, 1])
            for _ in range(max_iterations):
                self.set_q(q)
                if gripper_width is not None:
                    self.set_gripper_width(gripper_width)
                self.set_object_position(object_position, object_quat)
                residual = np.asarray(target, dtype=float) - self.task_position()
                error = float(np.linalg.norm(residual))
                orientation_error = 0.0
                residual_for_solve = residual
                jacobian = self.jacobian()
                if target_quat is not None and orientation_weight > 0.0:
                    current_quat = self.task_quat()
                    orientation_error = quat_angle_error(current_quat, target_quat)
                    orientation_residual = orientation_weight * quat_error_vector(current_quat, target_quat)
                    residual_for_solve = np.concatenate([residual, orientation_residual])
                    jacobian = np.vstack([jacobian, orientation_weight * self.rotational_jacobian()])
                collisions = self.collision_violation_count()
                margin = self.joint_margin(q)
                score = (
                    max(0.0, error - tolerance) * 100.0
                    + error
                    + orientation_weight * orientation_error
                    + 1000.0 * collisions
                    + 0.01 / max(margin, 1e-3)
                    + 0.04 * float(np.linalg.norm(q - self.ready_q))
                    + 0.02 * float(np.linalg.norm(q - np.asarray(initial_q, dtype=float)))
                )
                if score < best_score:
                    best_score = score
                    best_error = error
                    best_orientation_error = orientation_error
                    best_q = q.copy()
                system = jacobian @ jacobian.T + (damping**2) * np.eye(jacobian.shape[0])
                try:
                    delta = jacobian.T @ np.linalg.solve(system, residual_for_solve)
                except np.linalg.LinAlgError:
                    break
                q = np.clip(q + delta, self.ranges[:, 0], self.ranges[:, 1])
                if float(np.linalg.norm(delta)) < 1e-9:
                    break
        self.set_q(best_q)
        if target_quat is None or orientation_weight <= 0.0:
            best_orientation_error = 0.0
        return best_q.copy(), best_error, best_orientation_error

    def _seeds(self, initial_q: np.ndarray) -> tuple[np.ndarray, ...]:
        q0 = np.asarray(initial_q, dtype=float)
        seeds = [q0, self.ready_q, self.neutral_q, self.center_q, 0.5 * (q0 + self.ready_q)]
        for joint2 in (-1.0, -0.6, 0.2, 0.8):
            for joint3 in (-1.0, 0.0, 1.0):
                seed = self.center_q.copy()
                seed[1] = np.clip(joint2, self.ranges[1, 0], self.ranges[1, 1])
                seed[2] = np.clip(joint3, self.ranges[2, 0], self.ranges[2, 1])
                seeds.append(seed)
        for joint5 in (-1.2, 0.0, 1.2):
            for joint7 in (-1.5, 0.0, 1.5):
                seed = self.ready_q.copy()
                seed[4] = np.clip(joint5, self.ranges[4, 0], self.ranges[4, 1])
                seed[6] = np.clip(joint7, self.ranges[6, 0], self.ranges[6, 1])
                seeds.append(seed)
        unique: list[np.ndarray] = []
        for seed in seeds:
            seed = np.clip(seed, self.ranges[:, 0], self.ranges[:, 1])
            if not any(np.allclose(seed, old) for old in unique):
                unique.append(seed)
        return tuple(unique)

    def _detect_object_qpos_address(self) -> int | None:
        joint_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "v2_cube_freejoint"))
        if joint_id < 0:
            return None
        return int(self.model.jnt_qposadr[joint_id])

    def _detect_object_qvel_address(self) -> int | None:
        joint_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "v2_cube_freejoint"))
        if joint_id < 0:
            return None
        return int(self.model.jnt_dofadr[joint_id])

    def _detect_arm_joint_names(self) -> tuple[str, ...]:
        available = []
        for joint_id in range(int(self.model.njnt)):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                available.append(str(name))
        for prefix in ("joint", "panda_joint"):
            candidate = tuple(f"{prefix}{index}" for index in range(1, 8))
            if all(name in available for name in candidate):
                return candidate
        return tuple(name for name in available if "finger" not in name.lower() and "gripper" not in name.lower())[:7]

    def _detect_eef(self) -> tuple[str | None, str | None]:
        site_names = [
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_SITE, site_id)
            for site_id in range(int(self.model.nsite))
        ]
        for token in ("tcp", "grip", "ee", "hand"):
            for name in site_names:
                if name and token in str(name).lower():
                    return str(name), None
        body_names = [
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(int(self.model.nbody))
        ]
        for token in ("hand", "attachment", "link7"):
            for name in body_names:
                if name and token in str(name).lower():
                    return None, str(name)
        return None, None

    def _detect_gripper_qpos_addresses(self) -> np.ndarray:
        addresses: list[int] = []
        arm_ids = set(self.joint_ids)
        for joint_id in range(int(self.model.njnt)):
            if joint_id in arm_ids:
                continue
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            if "finger" in name.lower() or "gripper" in name.lower():
                addresses.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(addresses, dtype=int)

    def _detect_gripper_body_ids(self) -> np.ndarray:
        body_ids: list[int] = []
        for body_id in range(1, int(self.model.nbody)):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            lowered = name.lower()
            if "finger" in lowered or "gripper" in lowered:
                body_ids.append(int(body_id))
        return np.asarray(body_ids[:2], dtype=int)

    def _detect_gripper_geom_ids(self) -> np.ndarray:
        geoms_by_body: dict[int, list[int]] = {}
        gripper_bodies = [int(item) for item in self.gripper_body_ids]
        for geom_id in range(int(self.model.ngeom)):
            body_id = int(self.model.geom_bodyid[geom_id])
            geom_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            lowered = f"{geom_name} {body_name}".lower()
            if body_id in gripper_bodies or "finger" in lowered or "gripper" in lowered:
                geoms_by_body.setdefault(body_id, []).append(int(geom_id))

        selected: list[int] = []
        for body_id in gripper_bodies:
            candidates = geoms_by_body.get(body_id, [])
            if not candidates:
                continue
            selected.append(self._representative_finger_geom(candidates))
        if len(selected) >= 2:
            return np.asarray(selected[:2], dtype=int)

        for body_id, candidates in geoms_by_body.items():
            if body_id in {int(self.model.geom_bodyid[item]) for item in selected}:
                continue
            selected.append(self._representative_finger_geom(candidates))
            if len(selected) >= 2:
                break
        return np.asarray(selected[:2], dtype=int)

    def _representative_finger_geom(self, candidates: list[int]) -> int:
        def score(geom_id: int) -> tuple[int, float]:
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            lowered = name.lower()
            preferred = int(any(token in lowered for token in ("pad", "finger", "collision")))
            return preferred, float(np.linalg.norm(np.asarray(self.model.geom_pos[geom_id], dtype=float)))

        return max(candidates, key=score)


def auto_fit_panda_scenario(
    panda_xml: str | Path,
    *,
    lateral_offset: float = 0.08,
    vertical_drop: float = 0.12,
    object_type: str = "cube",
    task_name: str = "tabletop_easy",
) -> PickPlaceScenario:
    ik = MuJoCoPandaIK(panda_xml)
    ik.set_q(ik.ready_q)
    tcp = ik.task_position()
    scenario = make_scenario(
        object_type=object_type,
        task_name=task_name,
        center_xy=(float(tcp[0]), float(tcp[1])),
    )
    z_shift = float(tcp[2] - vertical_drop - scenario.cube_start[2])
    table_center = (scenario.table_center[0], scenario.table_center[1], scenario.table_center[2] + z_shift)
    obstacle_center = None if scenario.obstacle_center is None else (
        scenario.obstacle_center[0],
        scenario.obstacle_center[1],
        scenario.obstacle_center[2] + z_shift,
    )
    support_blocks = tuple(
        (
            (center[0], center[1], center[2] + z_shift),
            half_size,
        )
        for center, half_size in scenario.support_blocks
    )
    return PickPlaceScenario(
        **{
            **scenario.__dict__,
            "task_name": f"auto_fit_panda_{task_name}",
            "cube_start": (
                scenario.cube_start[0],
                scenario.cube_start[1],
                scenario.cube_start[2] + z_shift,
            ),
            "place_target": (
                scenario.place_target[0],
                scenario.place_target[1],
                scenario.place_target[2] + z_shift,
            ),
            "table_center": table_center,
            "obstacle_center": obstacle_center,
            "support_blocks": support_blocks,
            "table_size": (
                max(0.28, min(scenario.table_size[0], 0.38)),
                max(0.22, min(scenario.table_size[1], 0.75, 2.0 * lateral_offset + 0.55)),
                scenario.table_size[2],
            ),
        }
    )


def solve_joint_replay(
    panda_xml: str | Path,
    frames: tuple[ReplayFrame, ...],
    *,
    tolerance: float = 0.035,
    orientation_tolerance: float = 3.141592653589793,
) -> PandaIKReplay:
    ik = MuJoCoPandaIK(panda_xml)
    q = ik.ready_q
    solved: list[PandaJointFrame] = []
    failures: list[str] = []
    max_error = 0.0
    max_orientation_error = 0.0
    hold_attach_quat: tuple[float, float, float, float] | None = None
    for frame in frames:
        frame_orientation_required = bool(getattr(frame, "orientation_required", False))
        frame_orientation_tolerance = float(
            getattr(frame, "orientation_tolerance", orientation_tolerance)
        )
        frame_orientation_mode = str(getattr(frame, "orientation_mode", "target"))
        frame_orientation_weight = max(float(getattr(frame, "orientation_weight", 0.0)), 0.0)
        effective_orientation_tolerance = min(
            float(orientation_tolerance),
            frame_orientation_tolerance,
        )
        target_quat = frame.tcp_quat
        if frame_orientation_required and frame_orientation_mode == "hold_attach":
            target_quat = hold_attach_quat
        elif frame_orientation_mode == "hold_attach":
            target_quat = hold_attach_quat
        orientation_weight = (
            frame_orientation_weight
            if frame_orientation_weight > 0.0 and target_quat is not None
            else 0.35
            if (
                (frame_orientation_required or float(orientation_tolerance) < 3.0)
                and effective_orientation_tolerance < 3.0
                and target_quat is not None
            )
            else 0.0
        )
        q, error, orientation_error = ik.solve(
            frame.tcp_position,
            initial_q=q,
            tolerance=tolerance,
            gripper_width=frame.gripper_width,
            object_position=frame.object_position,
            object_quat=frame.object_quat,
            target_quat=target_quat,
            orientation_weight=orientation_weight,
        )
        max_error = max(max_error, error)
        max_orientation_error = max(max_orientation_error, orientation_error)
        ik.set_q(q)
        ik.set_gripper_width(frame.gripper_width)
        ik.set_object_position(frame.object_position, frame.object_quat)
        tcp = ik.task_position()
        tcp_quat = ik.task_quat()
        if (
            frame_orientation_mode == "hold_attach"
            and hold_attach_quat is None
        ):
            hold_attach_quat = tcp_quat
        collision_count = ik.collision_violation_count()
        if error > tolerance:
            failures.append(f"{frame.phase_name}:ik_error:{error:.6g}")
        if (
            (frame_orientation_required or float(orientation_tolerance) < 3.0)
            and orientation_error > effective_orientation_tolerance
        ):
            failures.append(f"{frame.phase_name}:orientation_error:{orientation_error:.6g}")
        if collision_count > 0:
            failures.append(f"{frame.phase_name}:collision:{collision_count}")
        solved.append(
            PandaJointFrame(
                phase_name=frame.phase_name,
                q=tuple(float(item) for item in q),
                gripper_width=frame.gripper_width,
                object_position=frame.object_position,
                object_attached=frame.object_attached,
                weld_active=frame.weld_active,
                tcp_position=_tuple3(tcp),
                target_position=frame.tcp_position,
                position_error=float(error),
                target_quat=frame.tcp_quat,
                tcp_quat=tcp_quat,
                orientation_error=float(orientation_error),
                collision_count=int(collision_count),
                joint_margin=ik.joint_margin(q),
                condition_number=ik.condition_number(),
                min_singular_value=ik.min_singular_value(),
                object_quat=frame.object_quat,
            )
        )
    costs = joint_motion_costs(tuple(frame.q for frame in solved))
    return PandaIKReplay(
        success=not failures,
        joint_names=ik.joint_names,
        eef_body=ik.eef_body,
        eef_site=ik.eef_site,
        max_position_error=float(max_error),
        max_orientation_error=float(max_orientation_error),
        joint_path_length=costs["joint_path_length"],
        max_joint_step=costs["max_joint_step"],
        energy_proxy=costs["energy_proxy"],
        smoothness_proxy=costs["smoothness_proxy"],
        frames=tuple(solved),
        failure_reasons=tuple(failures),
    )
