#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BFM_ZERO_DEPLOY_ROOT = REPO_ROOT / "thirdparties" / "BFM-Zero-deploy"
BFM_ZERO_TRAIN_ROOT = REPO_ROOT / "thirdparties" / "BFM-Zero"
BFM_ZERO_VENV_PYTHON = BFM_ZERO_TRAIN_ROOT / ".venv" / "bin" / "python"

DEFAULT_FALLEN_Z_M = 0.45
DEFAULT_RECOVERY_Z_M = 0.75
DEFAULT_HORIZON_S = 4.0
DEFAULT_INDUCED_HORIZON_S = 8.0
DEFAULT_INVALID_ROOT_Z_MIN_M = -0.1
DEFAULT_INVALID_ROOT_Z_MAX_M = 1.5
DEFAULT_WRENCH_FORCE_MIN_N = 600.0
DEFAULT_WRENCH_FORCE_MAX_N = 1800.0
DEFAULT_WRENCH_TORQUE_MIN_NM = 200.0
DEFAULT_WRENCH_TORQUE_MAX_NM = 600.0
DEFAULT_WRENCH_DURATION_S = 0.5
DEFAULT_LINEAR_VELOCITY_MIN_MPS = 3.0
DEFAULT_LINEAR_VELOCITY_MAX_MPS = 6.0
DEFAULT_ANGULAR_VELOCITY_MIN_RPS = 4.0
DEFAULT_ANGULAR_VELOCITY_MAX_RPS = 8.0
DEFAULT_LINEAR_VELOCITY_UP_COS_MIN = -0.8
DEFAULT_LINEAR_VELOCITY_UP_COS_MAX = 0.0
DEFAULT_GOAL_KEY = "dance1_subject3_505"
DEFAULT_FALL_GOALS = [
    "fallAndGetUp1_subject4_2230",
    "fallAndGetUp1_subject4_3350",
]

BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]
EXTEND_PARENT_NAME = "torso_link"
EXTEND_OFFSET = (0.0, 0.0, 0.35)

_RUNTIME: SimpleNamespace | None = None


def ensure_runtime_python() -> None:
    if os.environ.get("BFM_ZERO_EVAL_REEXEC") == "1":
        return
    if not BFM_ZERO_VENV_PYTHON.is_file():
        return
    current = Path(sys.executable).absolute()
    target = BFM_ZERO_VENV_PYTHON.absolute()
    if current == target:
        return
    env = os.environ.copy()
    env["BFM_ZERO_EVAL_REEXEC"] = "1"
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("OMP_NUM_THREADS", "1")
    os.execve(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def runtime() -> SimpleNamespace:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import joblib
    import mujoco
    import numpy as np
    import torch
    import yaml
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError:
        imageio = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        matplotlib = None
        plt = None

    for path in (BFM_ZERO_DEPLOY_ROOT, BFM_ZERO_TRAIN_ROOT):
        sys.path.insert(0, str(path))

    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from humanoidverse.utils.torch_utils import (
        calc_heading_quat_inv,
        quat_mul as hv_quat_mul,
        quat_rotate,
        quat_rotate_inverse,
        quat_to_tan_norm,
    )
    from sim_env.utils.disturbance import DisturbanceController
    from sim_env.utils.elastic_band import ElasticBand
    from sim_env.utils.simulation_bridge import SimulationBridge
    from utils.strings import resolve_matching_names_values

    _RUNTIME = SimpleNamespace(
        DisturbanceController=DisturbanceController,
        ElasticBand=ElasticBand,
        imageio=imageio,
        joblib=joblib,
        matplotlib=matplotlib,
        mujoco=mujoco,
        np=np,
        plt=plt,
        torch=torch,
        yaml=yaml,
        SimulationBridge=SimulationBridge,
        calc_heading_quat_inv=calc_heading_quat_inv,
        hv_quat_mul=hv_quat_mul,
        quat_rotate=quat_rotate,
        quat_rotate_inverse=quat_rotate_inverse,
        quat_to_tan_norm=quat_to_tan_norm,
        load_model_from_checkpoint_dir=load_model_from_checkpoint_dir,
        resolve_matching_names_values=resolve_matching_names_values,
    )
    return _RUNTIME


def repo_rel(path: Path) -> str:
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path.relative_to(REPO_ROOT))


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    rt = runtime()
    with path.open("r") as handle:
        return rt.yaml.load(handle, Loader=rt.yaml.FullLoader)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    rt = runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        rt.yaml.safe_dump(payload, handle, sort_keys=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1\n")


def wait_for_path(path: Path, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {label}: {path}")


def read_log_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


class ManagedProcess:
    def __init__(self, cmd: list[str], cwd: Path, log_path: Path, *, env: dict[str, str] | None = None, use_pty: bool = False):
        self.cmd = cmd
        self.cwd = cwd
        self.log_path = log_path
        self.env = env
        self.use_pty = use_pty
        self.proc: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._log_f: Any = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_pty:
            import fcntl
            import pty

            master_fd, slave_fd = pty.openpty()
            flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
            fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self._log_f = self.log_path.open("wb")
            self.proc = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                preexec_fn=os.setsid,
            )
            os.close(slave_fd)
            self._master_fd = master_fd
            return

        self._log_f = self.log_path.open("wb")
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=self._log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    def poll(self) -> int | None:
        self._drain_pty()
        return None if self.proc is None else self.proc.poll()

    def send(self, text: str) -> None:
        if self._master_fd is None:
            raise RuntimeError("process was not started with a PTY")
        os.write(self._master_fd, text.encode())
        self._drain_pty()

    def wait_for_marker(self, markers: str | list[str], timeout_s: float) -> str:
        marker_list = [markers] if isinstance(markers, str) else markers
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.poll()
            text = read_log_text(self.log_path)
            for marker in marker_list:
                if marker in text:
                    return marker
            code = self.poll()
            if code is not None:
                raise RuntimeError(f"process exited before marker {marker_list}: {code}")
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for log marker {marker_list}")

    def _drain_pty(self) -> None:
        if self._master_fd is None or self._log_f is None:
            return
        while True:
            try:
                data = os.read(self._master_fd, 65536)
            except BlockingIOError:
                break
            except OSError:
                break
            if not data:
                break
            self._log_f.write(data)
            self._log_f.flush()

    def terminate(self, timeout_s: float = 10.0) -> None:
        self._drain_pty()
        if self.proc is not None and self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=5.0)
        self._drain_pty()
        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._master_fd)
            self._master_fd = None
        if self._log_f is not None:
            self._log_f.close()
            self._log_f = None


def parse_goal_key(goal_key: str) -> tuple[str, int]:
    motion_key, frame_str = goal_key.rsplit("_", 1)
    return motion_key, int(frame_str)


def wxyz_to_xyzw(rt: SimpleNamespace, quat_wxyz: Any) -> Any:
    return rt.np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=rt.np.float32)


def xyzw_to_wxyz(rt: SimpleNamespace, quat_xyzw: Any) -> Any:
    return rt.np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=rt.np.float32)


def euler_xyz_to_wxyz(rt: SimpleNamespace, roll: float, pitch: float, yaw: float) -> Any:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return rt.np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=rt.np.float32,
    )


def compute_humanoid_observations_max_local(
    rt: SimpleNamespace,
    body_pos: Any,
    body_rot: Any,
    body_vel: Any,
    body_ang_vel: Any,
    local_root_obs: bool,
    root_height_obs: bool,
) -> dict[str, Any]:
    obs_dict: dict[str, Any] = {}
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    root_h = root_pos[:, 2:3]
    heading_rot_inv = rt.calc_heading_quat_inv(root_rot, w_last=True)

    if root_height_obs:
        obs_dict["root_height"] = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2).repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(-1, heading_rot_inv_expand.shape[-1])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(-1, local_body_pos.shape[-1])
    flat_local_body_pos = rt.quat_rotate(flat_heading_rot_inv, flat_local_body_pos, w_last=True)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], -1)
    local_body_pos = local_body_pos[..., 3:]

    flat_body_rot = body_rot.reshape(-1, body_rot.shape[-1])
    flat_local_body_rot = rt.hv_quat_mul(flat_heading_rot_inv, flat_body_rot, w_last=True)
    flat_local_body_rot_obs = rt.quat_to_tan_norm(flat_local_body_rot, w_last=True)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], -1)

    if not local_root_obs:
        root_rot_obs = rt.quat_to_tan_norm(root_rot, w_last=True)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(-1, body_vel.shape[-1])
    flat_local_body_vel = rt.quat_rotate(flat_heading_rot_inv, flat_body_vel, w_last=True)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], -1)

    flat_body_ang_vel = body_ang_vel.reshape(-1, body_ang_vel.shape[-1])
    flat_local_body_ang_vel = rt.quat_rotate(flat_heading_rot_inv, flat_body_ang_vel, w_last=True)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], -1)

    obs_dict["local_body_pos"] = local_body_pos
    obs_dict["local_body_rot"] = local_body_rot_obs
    obs_dict["local_body_vel"] = local_body_vel
    obs_dict["local_body_ang_vel"] = local_body_ang_vel
    return obs_dict


def build_joint_mappings(rt: SimpleNamespace, model: Any, joint_names: list[str]) -> tuple[Any, Any, Any]:
    qpos_ids = []
    qvel_ids = []
    actuator_ids = []
    for joint_name in joint_names:
        joint_id = rt.mujoco.mj_name2id(model, rt.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"Joint {joint_name} not found in MuJoCo model")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        qvel_ids.append(int(model.jnt_dofadr[joint_id]))
        actuator_name = joint_name.replace("_joint", "")
        actuator_id = rt.mujoco.mj_name2id(model, rt.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise KeyError(f"Actuator {actuator_name} not found in MuJoCo model")
        actuator_ids.append(int(actuator_id))
    return (
        rt.np.asarray(qpos_ids, dtype=rt.np.int64),
        rt.np.asarray(qvel_ids, dtype=rt.np.int64),
        rt.np.asarray(actuator_ids, dtype=rt.np.int64),
    )


def build_body_ids(rt: SimpleNamespace, model: Any) -> tuple[list[int], int, int]:
    body_ids: list[int] = []
    pelvis_body_id: int | None = None
    extend_parent_body_id: int | None = None
    for body_name in BODY_NAMES:
        body_id = rt.mujoco.mj_name2id(model, rt.mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise KeyError(f"Body {body_name} not found in MuJoCo model")
        body_ids.append(int(body_id))
        if body_name == "pelvis":
            pelvis_body_id = int(body_id)
        if body_name == EXTEND_PARENT_NAME:
            extend_parent_body_id = int(body_id)
    if pelvis_body_id is None or extend_parent_body_id is None:
        raise KeyError("Required body IDs missing from MuJoCo model")
    return body_ids, pelvis_body_id, extend_parent_body_id


def resolve_joint_array(rt: SimpleNamespace, config_values: dict[str, Any], joint_names: list[str]) -> Any:
    joint_indices, _, values = rt.resolve_matching_names_values(
        config_values,
        joint_names,
        preserve_order=True,
        strict=False,
    )
    output = rt.np.zeros(len(joint_names), dtype=rt.np.float32)
    output[rt.np.asarray(joint_indices, dtype=rt.np.int64)] = rt.np.asarray(values, dtype=rt.np.float32)
    return output


def extract_body_state(rt: SimpleNamespace, model: Any, data: Any, body_ids: list[int], extend_parent_body_id: int) -> tuple[Any, Any, Any, Any]:
    np = rt.np
    body_count = len(body_ids) + 1
    body_pos = np.zeros((body_count, 3), dtype=np.float32)
    body_rot = np.zeros((body_count, 4), dtype=np.float32)
    body_vel = np.zeros((body_count, 3), dtype=np.float32)
    body_ang_vel = np.zeros((body_count, 3), dtype=np.float32)
    velocity = np.zeros(6, dtype=np.float64)
    offset = np.asarray(EXTEND_OFFSET, dtype=np.float32)
    for idx, body_id in enumerate(body_ids):
        body_pos[idx] = data.xpos[body_id]
        body_rot[idx] = wxyz_to_xyzw(rt, data.xquat[body_id])
        rt.mujoco.mj_objectVelocity(model, data, rt.mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        body_ang_vel[idx] = velocity[0:3]
        body_vel[idx] = velocity[3:6]
    parent_idx = len(body_ids)
    parent_rot_matrix = data.xmat[extend_parent_body_id].reshape(3, 3)
    offset_world = parent_rot_matrix @ offset
    body_pos[parent_idx] = data.xpos[extend_parent_body_id] + offset_world
    body_rot[parent_idx] = wxyz_to_xyzw(rt, data.xquat[extend_parent_body_id])
    rt.mujoco.mj_objectVelocity(model, data, rt.mujoco.mjtObj.mjOBJ_BODY, extend_parent_body_id, velocity, 0)
    parent_ang_vel = velocity[0:3].astype(np.float32)
    parent_lin_vel = velocity[3:6].astype(np.float32)
    body_ang_vel[parent_idx] = parent_ang_vel
    body_vel[parent_idx] = parent_lin_vel + np.cross(parent_ang_vel, offset_world.astype(np.float32))
    return body_pos, body_rot, body_vel, body_ang_vel


def build_backward_obs(rt: SimpleNamespace, body_pos: Any, body_rot: Any, body_vel: Any, body_ang_vel: Any, joint_pos: Any, joint_vel: Any) -> dict[str, Any]:
    torch = rt.torch
    body_pos_t = torch.from_numpy(body_pos[None, ...]).to(device="cpu", dtype=torch.float32)
    body_rot_t = torch.from_numpy(body_rot[None, ...]).to(device="cpu", dtype=torch.float32)
    body_vel_t = torch.from_numpy(body_vel[None, ...]).to(device="cpu", dtype=torch.float32)
    body_ang_vel_t = torch.from_numpy(body_ang_vel[None, ...]).to(device="cpu", dtype=torch.float32)
    max_local_self = compute_humanoid_observations_max_local(
        rt,
        body_pos_t,
        body_rot_t,
        body_vel_t,
        body_ang_vel_t,
        local_root_obs=True,
        root_height_obs=True,
    )
    privileged_state = torch.cat([value for value in max_local_self.values()], dim=-1)
    base_quat = body_rot_t[:, 0]
    gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    projected_gravity = rt.quat_rotate_inverse(base_quat, gravity.repeat(privileged_state.shape[0], 1), w_last=True)
    state = torch.cat(
        [
            torch.from_numpy(joint_pos[None, ...]).to(dtype=torch.float32),
            torch.from_numpy(joint_vel[None, ...]).to(dtype=torch.float32),
            projected_gravity,
            body_ang_vel_t[:, 0],
        ],
        dim=-1,
    )
    return {"state": state, "privileged_state": privileged_state}


def cosine_similarity(rt: SimpleNamespace, a: Any, b: Any) -> float:
    denom = float(rt.np.linalg.norm(a) * rt.np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(rt.np.clip(rt.np.dot(a, b) / denom, -1.0, 1.0))


def speed_summary(rt: SimpleNamespace, model: Any, data: Any, pelvis_body_id: int, joint_qvel_ids: Any) -> dict[str, float]:
    rt.mujoco.mj_forward(model, data)
    root_vel = rt.np.asarray(data.cvel[pelvis_body_id, 3:6], dtype=rt.np.float32)
    root_ang_vel = rt.np.asarray(data.cvel[pelvis_body_id, 0:3], dtype=rt.np.float32)
    joint_vel = rt.np.asarray(data.qvel[joint_qvel_ids], dtype=rt.np.float32)
    joint_vel_rms = float(rt.np.sqrt(rt.np.mean(joint_vel * joint_vel))) if joint_vel.size else 0.0
    return {
        "root_lin_speed_mps": float(rt.np.linalg.norm(root_vel)),
        "root_ang_speed_rps": float(rt.np.linalg.norm(root_ang_vel)),
        "joint_vel_rms": joint_vel_rms,
    }


def is_near_static(
    summary: dict[str, float],
    max_root_speed_mps: float,
    max_ang_speed_rps: float,
    max_joint_speed_rms: float,
) -> bool:
    return (
        float(summary["root_lin_speed_mps"]) <= max_root_speed_mps
        and float(summary["root_ang_speed_rps"]) <= max_ang_speed_rps
        and float(summary["joint_vel_rms"]) <= max_joint_speed_rms
    )


def stability_sample(
    rt: SimpleNamespace,
    data: Any,
    pelvis_body_id: int,
    joint_qvel_ids: Any,
) -> dict[str, Any]:
    root_pos = rt.np.asarray(data.xpos[pelvis_body_id], dtype=rt.np.float32)
    root_vel = rt.np.asarray(data.cvel[pelvis_body_id, 3:6], dtype=rt.np.float32)
    root_ang_vel = rt.np.asarray(data.cvel[pelvis_body_id, 0:3], dtype=rt.np.float32)
    joint_vel = rt.np.asarray(data.qvel[joint_qvel_ids], dtype=rt.np.float32)
    joint_vel_rms = float(rt.np.sqrt(rt.np.mean(joint_vel * joint_vel))) if joint_vel.size else 0.0
    return {
        "root_pos": root_pos.copy(),
        "root_speed": float(rt.np.linalg.norm(root_vel)),
        "root_ang_speed": float(rt.np.linalg.norm(root_ang_vel)),
        "joint_vel_rms": joint_vel_rms,
    }


def history_is_stable(
    rt: SimpleNamespace,
    history: deque[dict[str, Any]],
    stable_root_speed_mps: float,
    stable_ang_speed_rps: float,
    stable_joint_speed_rms: float,
    stable_root_span_m: float,
) -> bool:
    if not history:
        return False
    root_span = float(rt.np.linalg.norm(history[-1]["root_pos"] - history[0]["root_pos"]))
    return (
        max(sample["root_speed"] for sample in history) <= stable_root_speed_mps
        and max(sample["root_ang_speed"] for sample in history) <= stable_ang_speed_rps
        and max(sample["joint_vel_rms"] for sample in history) <= stable_joint_speed_rms
        and root_span <= stable_root_span_m
    )


def contact_summary_for_current_state(rt: SimpleNamespace, model: Any, data: Any) -> dict[str, Any]:
    self_pairs: set[tuple[str, str]] = set()
    floor_bodies: set[str] = set()
    self_contact_count = 0
    floor_contact_count = 0
    for idx in range(int(data.ncon)):
        contact = data.contact[idx]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 == body2:
            continue
        name1 = str(model.body(body1).name)
        name2 = str(model.body(body2).name)
        if body1 == 0 or body2 == 0:
            floor_contact_count += 1
            floor_bodies.add(name2 if body1 == 0 else name1)
            continue
        self_contact_count += 1
        self_pairs.add(tuple(sorted((name1, name2))))
    floor_contact_bodies = sorted(body for body in floor_bodies if body and body != "world")
    self_contact_pairs = [" <-> ".join(pair) for pair in sorted(self_pairs)]
    return {
        "has_floor_contact": bool(floor_contact_count > 0),
        "floor_contact_count": floor_contact_count,
        "floor_contact_bodies": floor_contact_bodies,
        "floor_contact_body_count": len(floor_contact_bodies),
        "has_self_contact": bool(self_contact_count > 0),
        "self_contact_count": self_contact_count,
        "self_contact_pairs": self_contact_pairs,
        "self_contact_pair_count": len(self_contact_pairs),
    }


def write_trajectory_npz(path: Path, time_s: Any, qpos: Any, qvel: Any, root_z: Any) -> None:
    rt = runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    rt.np.savez_compressed(
        path,
        time_s=rt.np.asarray(time_s, dtype=rt.np.float32),
        qpos=rt.np.asarray(qpos, dtype=rt.np.float32),
        qvel=rt.np.asarray(qvel, dtype=rt.np.float32),
        root_z=rt.np.asarray(root_z, dtype=rt.np.float32),
    )


def settle_passively(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    settle_steps: int,
    pelvis_body_id: int | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    data.ctrl[:] = 0.0
    qpos_log = [data.qpos.copy()]
    qvel_log = [data.qvel.copy()]
    root_z_log = [float(data.xpos[pelvis_body_id, 2])] if pelvis_body_id is not None else []
    for _ in range(settle_steps):
        rt.mujoco.mj_step(model, data)
        if not rt.np.all(rt.np.isfinite(data.qpos)) or not rt.np.all(rt.np.isfinite(data.qvel)):
            return False, None
        qpos_log.append(data.qpos.copy())
        qvel_log.append(data.qvel.copy())
        if pelvis_body_id is not None:
            root_z_log.append(float(data.xpos[pelvis_body_id, 2]))
    trajectory = None
    if pelvis_body_id is not None:
        sim_dt = float(model.opt.timestep)
        trajectory = {
            "time_s": rt.np.arange(len(qpos_log), dtype=rt.np.float32) * sim_dt,
            "qpos": rt.np.asarray(qpos_log, dtype=rt.np.float32),
            "qvel": rt.np.asarray(qvel_log, dtype=rt.np.float32),
            "root_z": rt.np.asarray(root_z_log, dtype=rt.np.float32),
        }
    return True, trajectory


def settle_until_stable(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    pelvis_body_id: int,
    joint_qvel_ids: Any,
    timeout_s: float,
    stable_dwell_s: float,
    stable_root_speed_mps: float,
    stable_ang_speed_rps: float,
    stable_joint_speed_rms: float,
    stable_root_span_m: float,
) -> tuple[bool, dict[str, Any] | None, dict[str, float] | None]:
    sim_dt = float(model.opt.timestep)
    max_steps = max(1, int(math.ceil(timeout_s / sim_dt)))
    dwell_steps = max(1, int(round(stable_dwell_s / sim_dt)))
    history: deque[dict[str, Any]] = deque(maxlen=dwell_steps)
    qpos_log = [data.qpos.copy()]
    qvel_log = [data.qvel.copy()]
    root_z_log = [float(data.xpos[pelvis_body_id, 2])]
    history.append(stability_sample(rt, data, pelvis_body_id, joint_qvel_ids))
    data.ctrl[:] = 0.0
    for _ in range(max_steps):
        rt.mujoco.mj_step(model, data)
        if not rt.np.all(rt.np.isfinite(data.qpos)) or not rt.np.all(rt.np.isfinite(data.qvel)):
            return False, None, None
        qpos_log.append(data.qpos.copy())
        qvel_log.append(data.qvel.copy())
        root_z_log.append(float(data.xpos[pelvis_body_id, 2]))
        history.append(stability_sample(rt, data, pelvis_body_id, joint_qvel_ids))
        if len(history) == dwell_steps and history_is_stable(
            rt,
            history,
            stable_root_speed_mps,
            stable_ang_speed_rps,
            stable_joint_speed_rms,
            stable_root_span_m,
        ):
            trajectory = {
                "time_s": rt.np.arange(len(qpos_log), dtype=rt.np.float32) * sim_dt,
                "qpos": rt.np.asarray(qpos_log, dtype=rt.np.float32),
                "qvel": rt.np.asarray(qvel_log, dtype=rt.np.float32),
                "root_z": rt.np.asarray(root_z_log, dtype=rt.np.float32),
            }
            return True, trajectory, speed_summary(rt, model, data, pelvis_body_id, joint_qvel_ids)
    trajectory = {
        "time_s": rt.np.arange(len(qpos_log), dtype=rt.np.float32) * sim_dt,
        "qpos": rt.np.asarray(qpos_log, dtype=rt.np.float32),
        "qvel": rt.np.asarray(qvel_log, dtype=rt.np.float32),
        "root_z": rt.np.asarray(root_z_log, dtype=rt.np.float32),
    }
    return False, trajectory, speed_summary(rt, model, data, pelvis_body_id, joint_qvel_ids)


def sample_fallen_state(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    rng: Any,
    joint_qpos_ids: Any,
    joint_qvel_ids: Any,
    joint_lower: Any,
    joint_upper: Any,
    pelvis_body_id: int,
    fallen_z: float,
    max_attempts: int,
    settle_timeout_s: float,
    stable_dwell_s: float,
    stable_root_speed_mps: float,
    stable_ang_speed_rps: float,
    stable_joint_speed_rms: float,
    stable_root_span_m: float,
) -> dict[str, Any]:
    joint_span = joint_upper - joint_lower
    rejection_counts = {
        "raw_floor_contact": 0,
        "raw_self_contact": 0,
        "settle_invalid": 0,
        "not_fallen_after_settle": 0,
        "settle_timeout": 0,
    }
    for attempt in range(1, max_attempts + 1):
        rt.mujoco.mj_resetData(model, data)
        data.qpos[0:2] = rng.uniform(-0.15, 0.15, size=2)
        data.qpos[2] = rng.uniform(0.55, 0.9)
        data.qpos[3:7] = euler_xyz_to_wxyz(
            rt,
            float(rng.uniform(-math.pi, math.pi)),
            float(rng.uniform(-math.pi, math.pi)),
            float(rng.uniform(-math.pi, math.pi)),
        )
        data.qpos[joint_qpos_ids] = rt.np.clip(rng.uniform(joint_lower, joint_upper), joint_lower, joint_upper)
        data.qvel[:] = 0.0
        data.qvel[0:3] = rng.normal(0.0, 0.05, size=3)
        data.qvel[3:6] = rng.normal(0.0, 0.2, size=3)
        data.qvel[joint_qvel_ids] = rng.normal(0.0, rt.np.maximum(0.1, 0.05 * joint_span), size=joint_qvel_ids.shape[0])
        rt.mujoco.mj_forward(model, data)
        raw_contact = contact_summary_for_current_state(rt, model, data)
        if raw_contact["has_floor_contact"]:
            rejection_counts["raw_floor_contact"] += 1
            continue
        if raw_contact["has_self_contact"]:
            rejection_counts["raw_self_contact"] += 1
            continue
        settled_ok, settle_trajectory, settled_speed = settle_until_stable(
            rt,
            model,
            data,
            pelvis_body_id,
            joint_qvel_ids,
            settle_timeout_s,
            stable_dwell_s,
            stable_root_speed_mps,
            stable_ang_speed_rps,
            stable_joint_speed_rms,
            stable_root_span_m,
        )
        if settle_trajectory is None:
            rejection_counts["settle_invalid"] += 1
            continue
        if not settled_ok or settled_speed is None:
            rejection_counts["settle_timeout"] += 1
            continue
        root_z = float(data.xpos[pelvis_body_id, 2])
        if root_z >= fallen_z:
            rejection_counts["not_fallen_after_settle"] += 1
            continue
        return {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "root_z_m": root_z,
            "sample_attempts": attempt,
            "raw_contact": raw_contact,
            "settled_speed": settled_speed,
            "settle_trajectory": settle_trajectory,
            "rejection_counts": rejection_counts,
        }
    raise RuntimeError(f"Failed to sample a fallen state after {max_attempts} attempts")


def motion_frame_to_state(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    motion_key: str,
    frame_idx: int,
    joint_qpos_ids: Any,
    joint_qvel_ids: Any,
    pelvis_body_id: int,
    settle_steps: int,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any] | None]:
    motion_data = rt.joblib.load(BFM_ZERO_TRAIN_ROOT / "humanoidverse" / "data" / "lafan_29dof.pkl")
    if motion_key not in motion_data:
        raise KeyError(f"Motion {motion_key} missing from lafan_29dof.pkl")
    entry = motion_data[motion_key]
    if frame_idx < 0 or frame_idx >= int(entry["dof"].shape[0]):
        raise IndexError(f"Frame {frame_idx} outside motion length {entry['dof'].shape[0]}")
    rt.mujoco.mj_resetData(model, data)
    data.qpos[0:3] = entry["root_trans_offset"][frame_idx]
    data.qpos[3:7] = xyzw_to_wxyz(rt, entry["root_rot"][frame_idx])
    data.qpos[joint_qpos_ids] = entry["dof"][frame_idx]
    data.qvel[:] = 0.0
    if frame_idx > 0 and frame_idx + 1 < int(entry["dof"].shape[0]):
        dt = 1.0 / float(entry.get("fps", 50))
        data.qvel[0:3] = (entry["root_trans_offset"][frame_idx + 1] - entry["root_trans_offset"][frame_idx - 1]) / (2.0 * dt)
        if len(joint_qvel_ids) > 0:
            data.qvel[joint_qvel_ids] = (entry["dof"][frame_idx + 1] - entry["dof"][frame_idx - 1]) / (2.0 * dt)
    rt.mujoco.mj_forward(model, data)
    raw_contact = contact_summary_for_current_state(rt, model, data)
    _, settle_trajectory = settle_passively(rt, model, data, settle_steps, pelvis_body_id)
    rt.mujoco.mj_forward(model, data)
    return data.qpos.copy(), data.qvel.copy(), raw_contact, settle_trajectory


def load_target_body_positions(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    goal_key: str,
    joint_qpos_ids: Any,
    body_ids: list[int],
    extend_parent_body_id: int,
) -> Any:
    motion_key, frame_idx = parse_goal_key(goal_key)
    pelvis_body_id = model.body("pelvis").id
    qpos, _, _, _ = motion_frame_to_state(
        rt,
        model,
        data,
        motion_key,
        frame_idx,
        joint_qpos_ids,
        rt.np.asarray([], dtype=rt.np.int64),
        pelvis_body_id,
        0,
    )
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    rt.mujoco.mj_forward(model, data)
    target_body_pos, _, _, _ = extract_body_state(rt, model, data, body_ids, extend_parent_body_id)
    return target_body_pos


def build_renderer(rt: SimpleNamespace, model: Any, width: int, height: int, pelvis_body_id: int) -> tuple[Any, Any]:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
    renderer = rt.mujoco.Renderer(model, width=width, height=height)
    camera = rt.mujoco.MjvCamera()
    rt.mujoco.mjv_defaultCamera(camera)
    camera.type = rt.mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = pelvis_body_id
    camera.distance = 2.75
    camera.azimuth = 135.0
    camera.elevation = -20.0
    return renderer, camera


def render_frame(rt: SimpleNamespace, renderer: Any, camera: Any, model: Any, data: Any, pelvis_body_id: int, qpos: Any) -> Any:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    rt.mujoco.mj_forward(model, data)
    # Seed tracking-camera lookat from the current pelvis pose so frame 0 starts centered on the robot.
    camera.lookat[:] = data.xpos[pelvis_body_id]
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def write_plot(rt: SimpleNamespace, curves: Any, time_s: Any, title: str, ylabel: str, output_path: Path) -> None:
    if rt.plt is None:
        raise ModuleNotFoundError("matplotlib is required for stage3 plot generation")
    fig, ax = rt.plt.subplots(figsize=(12, 6))
    for curve in curves:
        ax.plot(time_s, curve, color="#1f77b4", alpha=0.12, linewidth=1.0)
    ax.plot(time_s, rt.np.nanmean(curves, axis=0), color="#111111", linewidth=2.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    rt.plt.close(fig)


def tile_video(rt: SimpleNamespace, frames: Any, fps: int, output_path: Path) -> tuple[int, int, int, int]:
    if rt.imageio is None:
        raise ModuleNotFoundError("imageio is required for video export")
    num_runs, frame_count, tile_height, tile_width, _ = frames.shape
    grid_cols = math.ceil(math.sqrt(num_runs))
    grid_rows = math.ceil(num_runs / grid_cols)
    output_width = grid_cols * tile_width
    output_height = grid_rows * tile_height
    writer = rt.imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=None)
    for frame_idx in range(frame_count):
        canvas = rt.np.zeros((output_height, output_width, 3), dtype=rt.np.uint8)
        for run_idx in range(num_runs):
            row, col = divmod(run_idx, grid_cols)
            y0 = row * tile_height
            x0 = col * tile_width
            canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = frames[run_idx, frame_idx]
        writer.append_data(canvas)
    writer.close()
    return grid_rows, grid_cols, output_width, output_height


@dataclass
class EvalLayout:
    run_id: str
    root_log_dir: Path
    root_artifact_dir: Path
    stage1_dir: Path
    stage2_dir: Path
    stage3_dir: Path


def build_layout(run_id: str) -> EvalLayout:
    root_log_dir = REPO_ROOT / "logs" / "bfm-zero-fallen-recovery" / run_id
    root_artifact_dir = REPO_ROOT / "artifacts" / "bfm-zero-fallen-recovery" / run_id
    stage1_dir = root_log_dir / "stage1"
    stage2_dir = root_log_dir / "stage2"
    stage3_dir = root_artifact_dir / "stage3"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage3_dir.mkdir(parents=True, exist_ok=True)
    return EvalLayout(run_id, root_log_dir, root_artifact_dir, stage1_dir, stage2_dir, stage3_dir)


def sim_runner_cmd(mode: str, extra_args: list[str]) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "_sim_runner", "--mode", mode, *extra_args]


def stage1_manifest_path(stage1_dir: Path) -> Path:
    return stage1_dir / "initial_states_manifest.json"


def stage1_summary_path(stage1_dir: Path) -> Path:
    return stage1_dir / "summary.json"


def stage2_summary_path(stage2_dir: Path) -> Path:
    return stage2_dir / "summary.json"


def stage3_summary_path(stage3_dir: Path) -> Path:
    return stage3_dir / "summary.json"


def write_stage_state_npz(path: Path, qpos: Any, qvel: Any, root_z: float) -> None:
    rt = runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    rt.np.savez_compressed(path, qpos=qpos, qvel=qvel, root_z=rt.np.asarray(root_z, dtype=rt.np.float32))


def load_stage_state_npz(path: Path) -> tuple[Any, Any, float]:
    rt = runtime()
    data = rt.np.load(path)
    return data["qpos"], data["qvel"], float(data["root_z"])


def disturbance_config_from_args(
    args: argparse.Namespace,
    command_path: Path,
    status_path: Path,
    *,
    seed: int,
    method_override: str | None = None,
    scale_multiplier: float | None = None,
) -> dict[str, Any]:
    method = method_override or args.disturbance_method_override
    scale = float(args.disturbance_scale_multiplier if scale_multiplier is None else scale_multiplier)
    return {
        "ENABLED": True,
        "COMMAND_FILE": str(command_path),
        "STATUS_FILE": str(status_path),
        "TRIGGER_KEY": "f",
        "METHOD_OVERRIDE": method,
        "SCALE_MULTIPLIER": scale,
        "RNG_SEED": int(seed),
        "WRENCH_FORCE_MIN_N": float(args.wrench_force_min_n),
        "WRENCH_FORCE_MAX_N": float(args.wrench_force_max_n),
        "WRENCH_TORQUE_MIN_NM": float(args.wrench_torque_min_nm),
        "WRENCH_TORQUE_MAX_NM": float(args.wrench_torque_max_nm),
        "WRENCH_DURATION_S": float(args.wrench_duration_s),
        "LINEAR_VELOCITY_MIN_MPS": float(args.linear_velocity_min_mps),
        "LINEAR_VELOCITY_MAX_MPS": float(args.linear_velocity_max_mps),
        "ANGULAR_VELOCITY_MIN_RPS": float(args.angular_velocity_min_rps),
        "ANGULAR_VELOCITY_MAX_RPS": float(args.angular_velocity_max_rps),
        "LINEAR_VELOCITY_UP_COS_MIN": float(DEFAULT_LINEAR_VELOCITY_UP_COS_MIN),
        "LINEAR_VELOCITY_UP_COS_MAX": float(DEFAULT_LINEAR_VELOCITY_UP_COS_MAX),
        "STATUS_WRITE_INTERVAL_S": 0.1,
        "POST_START_MIN_WAIT_S": float(args.post_start_min_wait_s),
        "STATIC_DWELL_S": float(args.stable_dwell_s),
        "STATIC_ROOT_SPEED_MPS": float(args.stable_root_speed_mps),
        "STATIC_ANG_SPEED_RPS": float(args.stable_ang_speed_rps),
        "STATIC_JOINT_SPEED_RMS": float(args.stable_joint_speed_rms),
        "STATIC_ROOT_SPAN_M": float(args.stable_root_span_m),
    }


def write_runtime_configs(
    run_dir: Path,
    low_state_port: int,
    low_cmd_port: int,
    *,
    disable_elastic_band: bool,
    disturbance_config: dict[str, Any] | None = None,
    elastic_band_initial_length_steps: int = 0,
    auto_release_on_first_lowcmd: bool = False,
) -> tuple[Path, Path]:
    robot_config = load_yaml(BFM_ZERO_DEPLOY_ROOT / "config" / "robot" / "g1.yaml")
    scene_config = load_yaml(BFM_ZERO_DEPLOY_ROOT / "config" / "scene" / "g1_29dof.yaml")
    robot_scene = Path(str(scene_config["ROBOT_SCENE"]))
    if not robot_scene.is_absolute():
        scene_config["ROBOT_SCENE"] = str((BFM_ZERO_DEPLOY_ROOT / robot_scene).resolve())
    robot_config["LOW_STATE_PORT"] = int(low_state_port)
    robot_config["LOW_CMD_PORT"] = int(low_cmd_port)
    robot_config["USE_JOYSTICK"] = False
    if disable_elastic_band:
        scene_config["ENABLE_ELASTIC_BAND"] = False
    scene_config["ELASTIC_BAND_INITIAL_LENGTH_STEPS"] = int(elastic_band_initial_length_steps)
    scene_config["ELASTIC_BAND_AUTO_RELEASE_ON_FIRST_LOWCMD"] = bool(auto_release_on_first_lowcmd)
    if disturbance_config is not None:
        scene_config["DISTURBANCE_CONFIG"] = disturbance_config
    robot_config_path = run_dir / "robot_runtime.yaml"
    scene_config_path = run_dir / "scene_runtime.yaml"
    write_yaml(robot_config_path, robot_config)
    write_yaml(scene_config_path, scene_config)
    return robot_config_path, scene_config_path


def write_keyboard_event_request(command_path: Path, request_id: int, key: str = "f") -> None:
    write_json(
        command_path,
        {
            "created_at": now_iso(),
            "request_id": int(request_id),
            "event": "keyboard",
            "key": key,
            "source": "stage2_orchestrator",
        },
    )


def wait_for_status_value(status_path: Path, key: str, expected: Any, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if status_path.exists():
            try:
                last_payload = load_json(status_path)
            except json.JSONDecodeError:
                last_payload = {}
            if last_payload.get(key) == expected:
                return last_payload
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for status {key}={expected!r}: {status_path}")


def selected_goal_order() -> list[str]:
    config = load_yaml(BFM_ZERO_DEPLOY_ROOT / "config" / "exp" / "goal" / "goal.yaml")
    goals = config.get("selected_goals")
    if not isinstance(goals, list) or not goals:
        raise ValueError("config/exp/goal/goal.yaml has no selected_goals list")
    return [str(goal) for goal in goals]


def advance_policy_to_goal(policy: ManagedProcess, goal_key: str, timeout_s: float) -> int:
    order = selected_goal_order()
    if goal_key not in order:
        raise KeyError(f"{goal_key} missing from config/exp/goal/goal.yaml")
    target_index = order.index(goal_key)
    for idx in range(target_index):
        next_goal = order[idx + 1]
        policy.send("n")
        policy.wait_for_marker(f"Switch to goal {next_goal}", timeout_s)
    return target_index


def launch_deployer(run_dir: Path, robot_config_path: Path, timeout_s: float) -> ManagedProcess:
    cmd = [
        sys.executable,
        "rl_policy/bfm_zero.py",
        "--robot_config",
        str(robot_config_path),
        "--policy_config",
        "config/policy/motivo_newG1.yaml",
        "--model_path",
        "./model/exported/FBcprAuxModel.onnx",
        "--task",
        "config/exp/goal/goal.yaml",
    ]
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [str(BFM_ZERO_DEPLOY_ROOT)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    proc = ManagedProcess(cmd, BFM_ZERO_DEPLOY_ROOT, run_dir / "deployer_stdout.log", env=env, use_pty=True)
    proc.start()
    proc.wait_for_marker(["Using keyboard", "task_type=goal"], timeout_s)
    return proc


def launch_sim_capture(
    run_dir: Path,
    mode: str,
    robot_config_path: Path,
    scene_config_path: Path,
    extra_args: list[str],
) -> ManagedProcess:
    cmd = sim_runner_cmd(
        mode,
        [
            "--robot-config",
            str(robot_config_path),
            "--scene-config",
            str(scene_config_path),
            *extra_args,
        ],
    )
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    proc = ManagedProcess(cmd, REPO_ROOT, run_dir / "simulator_stdout.log", env=env, use_pty=False)
    proc.start()
    return proc


def synthetic_source_state_from_trajectory(
    run_dir: Path,
    trajectory_path: Path,
    run_idx: int,
    goal_key: str,
) -> dict[str, Any]:
    rt = runtime()
    traj = rt.np.load(trajectory_path)
    qpos0 = rt.np.asarray(traj["qpos"][0], dtype=rt.np.float32)
    qvel0 = rt.np.asarray(traj["qvel"][0], dtype=rt.np.float32)
    root_z0 = float(rt.np.asarray(traj["root_z"], dtype=rt.np.float32)[0])
    state_id = f"induced_state_{run_idx:03d}"
    state_path = run_dir / f"{state_id}.npz"
    write_stage_state_npz(state_path, qpos0, qvel0, root_z0)
    return {
        "state_id": state_id,
        "index": run_idx,
        "source": "sim-induced",
        "goal_key": goal_key,
        "root_z_m": root_z0,
        "state_path": repo_rel(state_path),
        "settle_trajectory_path": None,
    }


def run_stage2_replay(args: argparse.Namespace, manifest: dict[str, Any], layout: EvalLayout) -> dict[str, Any]:
    stage2_dir = layout.stage2_dir
    stage2_runs_dir = stage2_dir / "runs"
    stage2_runs_dir.mkdir(parents=True, exist_ok=True)
    states: list[dict[str, Any]] = list(manifest["states"])
    if not states:
        raise ValueError("Stage 1 manifest has no states")

    run_summaries: list[dict[str, Any]] = []
    success_count = 0

    for run_idx, state in enumerate(states):
        run_dir = stage2_runs_dir / f"run_{run_idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ready_file = run_dir / "sim_ready.json"
        start_flag = run_dir / "start_eval.flag"
        done_file = run_dir / "eval_done.json"
        trajectory_path = run_dir / "trajectory.npz"
        summary_path = run_dir / "summary.json"
        state_path = REPO_ROOT / str(state["state_path"])
        low_state_port = args.port_base + run_idx * 2
        low_cmd_port = args.port_base + run_idx * 2 + 1
        robot_config_path, scene_config_path = write_runtime_configs(
            run_dir,
            low_state_port,
            low_cmd_port,
            disable_elastic_band=True,
        )

        sim = launch_sim_capture(
            run_dir,
            "replay",
            robot_config_path,
            scene_config_path,
            [
                "--ready-file",
                str(ready_file),
                "--start-flag",
                str(start_flag),
                "--done-file",
                str(done_file),
                "--summary-path",
                str(summary_path),
                "--trajectory-path",
                str(trajectory_path),
                "--initial-state",
                str(state_path),
                "--fallen-z",
                f"{args.fallen_z:.6f}",
                "--recovery-z",
                f"{args.recovery_z:.6f}",
                "--horizon-s",
                f"{args.horizon_s:.6f}",
                "--goal-key",
                args.goal_key,
                "--source-state-id",
                str(state["state_id"]),
            ],
        )
        deployer: ManagedProcess | None = None
        try:
            wait_for_path(ready_file, 10.0, "simulator ready file")
            deployer = launch_deployer(run_dir, robot_config_path, args.policy_ready_timeout_s)
            goal_index = advance_policy_to_goal(deployer, args.goal_key, args.policy_ready_timeout_s)
            touch(start_flag)
            deployer.send("]")
            deployer.wait_for_marker(f"Switch to goal={args.goal_key}", args.policy_ready_timeout_s)
            wait_for_path(done_file, args.horizon_s + 20.0, f"stage2 done file for run {run_idx}")
        finally:
            if deployer is not None:
                deployer.terminate()
            sim.terminate()

        run_summary = load_json(summary_path)
        run_summary["goal_selection_index"] = goal_index
        run_summary["source_state"] = state
        run_summary["trajectory_path"] = repo_rel(trajectory_path)
        run_summary["summary_path"] = repo_rel(summary_path)
        run_summary["simulator_log"] = repo_rel(run_dir / "simulator_stdout.log")
        run_summary["deployer_log"] = repo_rel(run_dir / "deployer_stdout.log")
        write_json(summary_path, run_summary)
        run_summaries.append(run_summary)
        success_count += int(bool(run_summary["success"]))

    return {
        "run_id": str(manifest["run_id"]),
        "created_at": now_iso(),
        "stage": "stage2",
        "mode": "replay_saved_state",
        "stage1_manifest": repo_rel(Path(args.stage1_manifest)),
        "goal_key": args.goal_key,
        "seed": int(manifest["seed"]),
        "fallen_z_threshold_m": args.fallen_z,
        "recovery_z_threshold_m": args.recovery_z,
        "horizon_s": args.horizon_s,
        "num_runs": len(run_summaries),
        "success_count": success_count,
        "success_rate": float(success_count / len(run_summaries)),
        "run_summaries": [repo_rel(stage2_runs_dir / f"run_{idx:03d}" / "summary.json") for idx in range(len(run_summaries))],
    }


def run_induced_attempt(
    args: argparse.Namespace,
    run_dir: Path,
    run_idx: int,
    attempt_idx: int,
    *,
    method_override: str | None = None,
    scale_multiplier: float | None = None,
) -> dict[str, Any]:
    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_idx:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    ready_file = attempt_dir / "sim_ready.json"
    start_flag = attempt_dir / "start_eval.flag"
    done_file = attempt_dir / "eval_done.json"
    summary_path = attempt_dir / "summary.json"
    trajectory_path = attempt_dir / "trajectory.npz"
    command_path = attempt_dir / "sim_command.json"
    status_path = attempt_dir / "sim_status.json"
    attempt_slot = run_idx * max(int(args.max_attempts_per_run), 1) + attempt_idx
    low_state_port = args.port_base + attempt_slot * 2
    low_cmd_port = args.port_base + attempt_slot * 2 + 1
    attempt_seed = int(args.seed + run_idx * 1000 + attempt_idx)
    disturbance_config = disturbance_config_from_args(
        args,
        command_path,
        status_path,
        seed=attempt_seed,
        method_override=method_override,
        scale_multiplier=scale_multiplier,
    )
    robot_config_path, scene_config_path = write_runtime_configs(
        attempt_dir,
        low_state_port,
        low_cmd_port,
        disable_elastic_band=False,
        disturbance_config=disturbance_config,
        elastic_band_initial_length_steps=3,
        auto_release_on_first_lowcmd=True,
    )
    sim = launch_sim_capture(
        attempt_dir,
        "induce",
        robot_config_path,
        scene_config_path,
        [
            "--ready-file",
            str(ready_file),
            "--start-flag",
            str(start_flag),
            "--done-file",
            str(done_file),
            "--summary-path",
            str(summary_path),
            "--trajectory-path",
            str(trajectory_path),
            "--fallen-z",
            f"{args.fallen_z:.6f}",
            "--recovery-z",
            f"{args.recovery_z:.6f}",
            "--horizon-s",
            f"{args.horizon_s:.6f}",
            "--goal-key",
            args.goal_key,
            "--source-state-id",
            f"induced_run_{run_idx:03d}",
            "--timeout-s",
            f"{args.static_ready_timeout_s:.6f}",
        ],
    )
    deployer: ManagedProcess | None = None
    try:
        wait_for_path(ready_file, 10.0, "simulator ready file")
        deployer = launch_deployer(attempt_dir, robot_config_path, args.policy_ready_timeout_s)
        goal_index = advance_policy_to_goal(deployer, args.goal_key, args.policy_ready_timeout_s)
        deployer.send("]")
        deployer.wait_for_marker(f"Switch to goal={args.goal_key}", args.policy_ready_timeout_s)
        touch(start_flag)
        wait_for_status_value(status_path, "ready_for_disturbance", True, args.static_ready_timeout_s)
        write_keyboard_event_request(command_path, 1, "f")
        wait_for_path(done_file, args.horizon_s + args.static_ready_timeout_s + 20.0, f"stage2 done file for run {run_idx} attempt {attempt_idx}")
    finally:
        if deployer is not None:
            deployer.terminate()
        sim.terminate()

    run_summary = load_json(summary_path)
    run_summary["goal_selection_index"] = goal_index
    run_summary["attempt_index"] = attempt_idx
    run_summary["attempt_seed"] = attempt_seed
    run_summary["trajectory_path"] = repo_rel(trajectory_path)
    run_summary["summary_path"] = repo_rel(summary_path)
    run_summary["simulator_log"] = repo_rel(attempt_dir / "simulator_stdout.log")
    run_summary["deployer_log"] = repo_rel(attempt_dir / "deployer_stdout.log")
    write_json(summary_path, run_summary)
    return run_summary


def run_stage2_induced(
    args: argparse.Namespace,
    layout: EvalLayout,
    *,
    method_override: str | None = None,
    scale_multiplier: float | None = None,
    require_accepted_runs: bool = True,
) -> dict[str, Any]:
    stage2_dir = layout.stage2_dir
    stage2_runs_dir = stage2_dir / "runs"
    stage2_runs_dir.mkdir(parents=True, exist_ok=True)
    requested_num_runs = int(args.num_runs)
    if requested_num_runs <= 0:
        raise ValueError("--num-runs must be positive in induced-fall mode")

    run_summaries: list[dict[str, Any]] = []
    success_count = 0
    perturbation_success_count = 0

    for run_idx in range(requested_num_runs):
        if method_override is not None:
            scheduled_method = method_override
        elif args.disturbance_method_override != "mixed":
            scheduled_method = args.disturbance_method_override
        else:
            scheduled_method = "wrench" if run_idx % 2 == 0 else "velocity_delta"
        run_dir = stage2_runs_dir / f"run_{run_idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        accepted_summary: dict[str, Any] | None = None
        attempt_summary_paths: list[str] = []

        for attempt_idx in range(int(args.max_attempts_per_run)):
            try:
                attempt_summary = run_induced_attempt(
                    args,
                    run_dir,
                    run_idx,
                    attempt_idx,
                    method_override=scheduled_method,
                    scale_multiplier=scale_multiplier,
                )
                attempt_summary_paths.append(str(attempt_summary["summary_path"]))
            except Exception as exc:
                error_path = run_dir / "attempts" / f"attempt_{attempt_idx:03d}" / "orchestrator_error.json"
                write_json(
                    error_path,
                    {
                        "created_at": now_iso(),
                        "run_index": run_idx,
                        "attempt_index": attempt_idx,
                        "error": str(exc),
                    },
                )
                attempt_summary_paths.append(repo_rel(error_path))
                attempt_summary = {
                    "perturbation_success": False,
                    "success": False,
                    "error": str(exc),
                    "summary_path": repo_rel(error_path),
                }

            if bool(attempt_summary.get("perturbation_success")):
                accepted_summary = dict(attempt_summary)
                perturbation_success_count += 1
                break
            if not require_accepted_runs:
                accepted_summary = dict(attempt_summary)
                break

        if accepted_summary is None:
            raise RuntimeError(
                f"Run {run_idx} failed to produce an accepted perturbation within {args.max_attempts_per_run} attempts"
            )

        trajectory_rel = accepted_summary.get("trajectory_path")
        trajectory_path = REPO_ROOT / str(trajectory_rel) if trajectory_rel else None
        if trajectory_path is not None and trajectory_path.exists():
            source_state = synthetic_source_state_from_trajectory(run_dir, trajectory_path, run_idx, args.goal_key)
        else:
            source_state = {
                "state_id": f"induced_state_{run_idx:03d}",
                "index": run_idx,
                "source": "sim-induced",
                "goal_key": args.goal_key,
                "root_z_m": float("nan"),
                "state_path": None,
                "settle_trajectory_path": None,
            }
        final_summary = dict(accepted_summary)
        final_summary["source_state"] = source_state
        final_summary["attempt_count"] = 1 + int(final_summary.get("attempt_index", 0))
        final_summary["attempt_summary_paths"] = attempt_summary_paths
        final_summary["scheduled_disturbance_method"] = scheduled_method
        final_summary["summary_path"] = repo_rel(run_dir / "summary.json")
        write_json(run_dir / "summary.json", final_summary)
        run_summaries.append(final_summary)
        success_count += int(bool(final_summary["success"]))

    summary: dict[str, Any] = {
        "run_id": layout.run_id,
        "created_at": now_iso(),
        "stage": "stage2",
        "mode": "induce_fall_in_simulator",
        "goal_key": args.goal_key,
        "seed": int(args.seed),
        "fallen_z_threshold_m": args.fallen_z,
        "recovery_z_threshold_m": args.recovery_z,
        "horizon_s": args.horizon_s,
        "num_runs": len(run_summaries),
        "requested_num_runs": requested_num_runs,
        "accepted_num_runs": len(run_summaries),
        "success_count": success_count,
        "success_rate": float(success_count / len(run_summaries)),
        "perturbation_success_count": perturbation_success_count,
        "run_summaries": [repo_rel(stage2_runs_dir / f"run_{idx:03d}" / "summary.json") for idx in range(len(run_summaries))],
    }
    if method_override is not None:
        summary["disturbance_method_override"] = method_override
    elif args.disturbance_method_override == "mixed":
        summary["disturbance_method_schedule"] = "alternating_wrench_velocity_delta"
    else:
        summary["disturbance_method_override"] = args.disturbance_method_override
    if scale_multiplier is not None:
        summary["disturbance_scale_multiplier"] = float(scale_multiplier)
    return summary


def try_direct_goal_state(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    joint_qpos_ids: Any,
    joint_qvel_ids: Any,
    pelvis_body_id: int,
    goal_key: str,
    fallen_z: float,
    settle_steps: int,
) -> tuple[Any, Any, float, str, dict[str, Any], dict[str, Any] | None] | None:
    motion_key, frame_idx = parse_goal_key(goal_key)
    try:
        qpos, qvel, raw_contact, settle_trajectory = motion_frame_to_state(
            rt,
            model,
            data,
            motion_key,
            frame_idx,
            joint_qpos_ids,
            joint_qvel_ids,
            pelvis_body_id,
            settle_steps,
        )
    except Exception as exc:
        return None if isinstance(exc, Exception) else None
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    rt.mujoco.mj_forward(model, data)
    root_z = float(data.xpos[pelvis_body_id, 2])
    if root_z >= fallen_z:
        return None
    return qpos, qvel, root_z, "lafan_29dof frame extraction + passive settle", raw_contact, settle_trajectory


def stage1(args: argparse.Namespace) -> int:
    rt = runtime()
    layout = build_layout(args.run_id or f"{now_stamp()}_stage1_seed{args.seed}")
    stage1_dir = layout.stage1_dir
    states_dir = stage1_dir / "states"
    settle_logs_dir = stage1_dir / "settle_trajectories"
    states_dir.mkdir(parents=True, exist_ok=True)
    settle_logs_dir.mkdir(parents=True, exist_ok=True)

    model = rt.mujoco.MjModel.from_xml_path(str(BFM_ZERO_DEPLOY_ROOT / "data" / "robots" / "g1" / "scene_29dof_freebase.xml"))
    data = rt.mujoco.MjData(model)
    policy_config = load_yaml(BFM_ZERO_DEPLOY_ROOT / "config" / "policy" / "motivo_newG1.yaml")
    robot_config = load_yaml(BFM_ZERO_DEPLOY_ROOT / "config" / "robot" / "g1.yaml")
    joint_names = [str(name) for name in policy_config["isaac_joint_names"]]
    joint_qpos_ids, joint_qvel_ids, _ = build_joint_mappings(rt, model, joint_names)
    _, pelvis_body_id, _ = build_body_ids(rt, model)
    joint_lower = resolve_joint_array(rt, robot_config["joint_pos_lower_limit"], joint_names)
    joint_upper = resolve_joint_array(rt, robot_config["joint_pos_upper_limit"], joint_names)
    rng = rt.np.random.default_rng(args.seed)

    direct_extraction: dict[str, dict[str, Any]] = {}
    manifest_states: list[dict[str, Any]] = []
    rejection_counts = {
        "raw_floor_contact": 0,
        "raw_self_contact": 0,
        "settle_invalid": 0,
        "not_fallen_after_settle": 0,
        "settle_timeout": 0,
    }

    for random_index in range(args.random_count):
        sample = sample_fallen_state(
            rt,
            model,
            data,
            rng,
            joint_qpos_ids,
            joint_qvel_ids,
            joint_lower,
            joint_upper,
            pelvis_body_id,
            args.fallen_z,
            args.max_sample_attempts,
            args.capture_timeout_s,
            args.stable_dwell_s,
            args.stable_root_speed_mps,
            args.stable_ang_speed_rps,
            args.stable_joint_speed_rms,
            args.stable_root_span_m,
        )
        for key, value in sample["rejection_counts"].items():
            rejection_counts[key] += int(value)
        state_id = f"state_{len(manifest_states):03d}"
        state_path = states_dir / f"{state_id}.npz"
        settle_log_path = settle_logs_dir / f"{state_id}.npz"
        write_stage_state_npz(state_path, sample["qpos"], sample["qvel"], sample["root_z_m"])
        write_trajectory_npz(
            settle_log_path,
            sample["settle_trajectory"]["time_s"],
            sample["settle_trajectory"]["qpos"],
            sample["settle_trajectory"]["qvel"],
            sample["settle_trajectory"]["root_z"],
        )
        manifest_states.append(
            {
                "state_id": state_id,
                "index": len(manifest_states),
                "source": "random",
                "goal_key": None,
                "root_z_m": sample["root_z_m"],
                "seed": args.seed,
                "sample_attempts": sample["sample_attempts"],
                "state_path": repo_rel(state_path),
                "settle_trajectory_path": repo_rel(settle_log_path),
                "raw_contact": sample["raw_contact"],
                "settled_speed": sample["settled_speed"],
                "acceptance": {
                    "fallen_z_threshold_m": args.fallen_z,
                    "settle_timeout_s": args.capture_timeout_s,
                    "stable_dwell_s": args.stable_dwell_s,
                    "stable_root_speed_mps": args.stable_root_speed_mps,
                    "stable_ang_speed_rps": args.stable_ang_speed_rps,
                    "stable_joint_speed_rms": args.stable_joint_speed_rms,
                    "stable_root_span_m": args.stable_root_span_m,
                    "max_sample_attempts": args.max_sample_attempts,
                },
            }
        )

    for goal_offset, goal_key in enumerate(args.goal_keys):
        direct = try_direct_goal_state(
            rt,
            model,
            data,
            joint_qpos_ids,
            joint_qvel_ids,
            pelvis_body_id,
            goal_key,
            args.fallen_z,
            args.settle_steps,
        )
        if direct is None:
            raise RuntimeError(f"Goal-derived state for {goal_key} could not be reproduced by direct frame extraction")
        qpos, qvel, root_z, reason, raw_contact, settle_trajectory = direct
        direct_extraction[goal_key] = {"supported": True, "reason": reason}
        trajectory_path = settle_logs_dir / f"goal_{goal_offset:02d}_{goal_key}.npz"
        if settle_trajectory is not None:
            write_trajectory_npz(
                trajectory_path,
                settle_trajectory["time_s"],
                settle_trajectory["qpos"],
                settle_trajectory["qvel"],
                settle_trajectory["root_z"],
            )
        if root_z >= args.fallen_z:
            raise RuntimeError(f"Goal-derived state for {goal_key} has root_z={root_z:.4f} >= fallen threshold {args.fallen_z:.4f}")
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        settled_speed = speed_summary(rt, model, data, pelvis_body_id, joint_qvel_ids)
        state_id = f"state_{len(manifest_states):03d}"
        state_path = states_dir / f"{state_id}.npz"
        write_stage_state_npz(state_path, qpos, qvel, root_z)
        final_settle_log_path = settle_logs_dir / f"{state_id}.npz"
        trajectory_path.replace(final_settle_log_path)
        manifest_states.append(
            {
                "state_id": state_id,
                "index": len(manifest_states),
                "source": "goal-derived",
                "goal_key": goal_key,
                "root_z_m": root_z,
                "seed": args.seed,
                "sample_attempts": None,
                "state_path": repo_rel(state_path),
                "settle_trajectory_path": repo_rel(final_settle_log_path),
                "raw_contact": raw_contact,
                "settled_speed": settled_speed,
                "acceptance": {
                    "fallen_z_threshold_m": args.fallen_z,
                    "capture_reason": reason,
                },
            }
        )

    if len(manifest_states) != args.random_count + len(args.goal_keys):
        raise RuntimeError("Stage 1 did not produce the expected number of states")
    if not all(float(row["root_z_m"]) < args.fallen_z for row in manifest_states):
        raise RuntimeError("Stage 1 produced a non-fallen state")

    manifest = {
        "run_id": layout.run_id,
        "created_at": now_iso(),
        "stage": "stage1",
        "state_count": len(manifest_states),
        "random_count": args.random_count,
        "goal_derived_count": len(args.goal_keys),
        "seed": args.seed,
        "fallen_z_threshold_m": args.fallen_z,
        "goal_keys": list(args.goal_keys),
        "direct_extraction": direct_extraction,
        "stability_thresholds": {
            "stable_dwell_s": args.stable_dwell_s,
            "stable_root_speed_mps": args.stable_root_speed_mps,
            "stable_ang_speed_rps": args.stable_ang_speed_rps,
            "stable_joint_speed_rms": args.stable_joint_speed_rms,
            "stable_root_span_m": args.stable_root_span_m,
        },
        "rejection_counts": rejection_counts,
        "states_dir": repo_rel(states_dir),
        "settle_trajectories_dir": repo_rel(settle_logs_dir),
        "states": manifest_states,
    }
    manifest_path = stage1_manifest_path(stage1_dir)
    write_json(manifest_path, manifest)
    stage1_summary = {
        "run_id": layout.run_id,
        "created_at": now_iso(),
        "stage": "stage1",
        "stage1_manifest": repo_rel(manifest_path),
        "state_count": len(manifest_states),
        "random_count": args.random_count,
        "goal_derived_count": len(args.goal_keys),
        "fallen_z_threshold_m": args.fallen_z,
        "stability_thresholds": manifest["stability_thresholds"],
        "rejection_counts": rejection_counts,
        "states": [
            {
                "state_id": row["state_id"],
                "source": row["source"],
                "state_path": row["state_path"],
                "settle_trajectory_path": row["settle_trajectory_path"],
            }
            for row in manifest_states
        ],
    }
    write_json(stage1_summary_path(stage1_dir), stage1_summary)
    print(
        json.dumps(
            {
                "stage1_manifest": repo_rel(manifest_path),
                "stage1_summary": repo_rel(stage1_summary_path(stage1_dir)),
                "state_count": len(manifest_states),
            },
            indent=2,
        )
    )
    return 0


def stage2(args: argparse.Namespace) -> int:
    if args.induce_fall_in_simulator:
        if float(args.horizon_s) == DEFAULT_HORIZON_S:
            args.horizon_s = DEFAULT_INDUCED_HORIZON_S
        run_id = args.run_id or f"{now_stamp()}_stage2_induce_seed{args.seed}"
        layout = build_layout(run_id)
        summary = run_stage2_induced(args, layout)
    else:
        if not args.stage1_manifest:
            raise ValueError("--stage1-manifest is required unless --induce-fall-in-simulator is enabled")
        manifest = load_json(Path(args.stage1_manifest))
        layout = build_layout(str(manifest["run_id"]))
        summary = run_stage2_replay(args, manifest, layout)

    summary_path = stage2_summary_path(layout.stage2_dir)
    write_json(summary_path, summary)

    run_summaries = [load_json(REPO_ROOT / str(path)) for path in summary["run_summaries"]]
    if len(run_summaries) == 100 and int(summary["success_count"]) == 0:
        investigation_path = layout.stage2_dir / "zero_success_investigation.json"
        max_final_window = max(float(row.get("final_window_mean_root_z_m", float("nan"))) for row in run_summaries)
        max_any_height = max(float(row.get("max_root_z_m", float("nan"))) for row in run_summaries)
        write_json(
            investigation_path,
            {
                "run_id": str(summary["run_id"]),
                "created_at": now_iso(),
                "reason": "all_100_runs_failed_recovery_threshold",
                "goal_key": args.goal_key,
                "max_final_window_mean_root_z_m": max_final_window,
                "max_any_root_z_m": max_any_height,
                "stage2_summary": repo_rel(summary_path),
                "run_summaries": summary["run_summaries"],
            },
        )
        raise SystemExit(
            f"Stage 2 produced zero successful recoveries across 100 runs. "
            f"Readonly investigation written to {repo_rel(investigation_path)}."
        )

    print(
        json.dumps(
            {
                "stage2_summary": repo_rel(summary_path),
                "mode": summary["mode"],
                "success_count": int(summary["success_count"]),
                "num_runs": int(summary["num_runs"]),
            },
            indent=2,
        )
    )
    return 0


def tune_disturbance(args: argparse.Namespace) -> int:
    base_run_id = args.run_id or f"{now_stamp()}_tune_seed{args.seed}"
    base_layout = build_layout(base_run_id)
    case_rows: list[dict[str, Any]] = []
    original_max_attempts = int(args.max_attempts_per_run)
    args.max_attempts_per_run = 1
    try:
        for method in ("wrench", "velocity_delta"):
            for scale_multiplier in (1.0, 0.1):
                case_run_id = f"{base_run_id}_{method}_{'nominal' if scale_multiplier == 1.0 else 'x0p1'}"
                case_layout = build_layout(case_run_id)
                case_summary = run_stage2_induced(
                    args,
                    case_layout,
                    method_override=method,
                    scale_multiplier=scale_multiplier,
                    require_accepted_runs=False,
                )
                case_summary_path = stage2_summary_path(case_layout.stage2_dir)
                write_json(case_summary_path, case_summary)
                perturbation_success_rate = float(case_summary["perturbation_success_count"] / max(int(case_summary["num_runs"]), 1))
                case_rows.append(
                    {
                        "method": method,
                        "scale_multiplier": scale_multiplier,
                        "num_runs": int(case_summary["num_runs"]),
                        "perturbation_success_count": int(case_summary["perturbation_success_count"]),
                        "perturbation_success_rate": perturbation_success_rate,
                        "passes_0p1_gate": None if scale_multiplier != 0.1 else bool(perturbation_success_rate < args.tuning_gate_max_success_rate),
                        "stage2_summary": repo_rel(case_summary_path),
                    }
                )
    finally:
        args.max_attempts_per_run = original_max_attempts

    summary_path = base_layout.root_artifact_dir / "tuning_summary.json"
    write_json(
        summary_path,
        {
            "run_id": base_run_id,
            "created_at": now_iso(),
            "goal_key": args.goal_key,
            "seed": int(args.seed),
            "num_runs_per_case": int(args.num_runs),
            "gate_threshold_success_rate": float(args.tuning_gate_max_success_rate),
            "cases": case_rows,
        },
    )
    print(json.dumps({"tuning_summary": repo_rel(summary_path), "cases": case_rows}, indent=2))
    return 0


def stage3(args: argparse.Namespace) -> int:
    rt = runtime()
    stage2_dir = Path(args.stage2_dir)
    summary = load_json(stage2_summary_path(stage2_dir))
    run_id = str(summary["run_id"])
    layout = build_layout(run_id)
    stage3_dir = layout.stage3_dir
    stage1_manifest_rel = summary.get("stage1_manifest")
    stage1_manifest = load_json(REPO_ROOT / str(stage1_manifest_rel)) if stage1_manifest_rel else None

    run_summary_paths = [REPO_ROOT / str(path) for path in summary["run_summaries"]]
    run_summaries = [load_json(path) for path in run_summary_paths]
    if not run_summaries:
        raise ValueError("Stage 2 summary has no run summaries")

    model = rt.mujoco.MjModel.from_xml_path(str(BFM_ZERO_DEPLOY_ROOT / "data" / "robots" / "g1" / "scene_29dof_freebase.xml"))
    data = rt.mujoco.MjData(model)
    policy_config = load_yaml(BFM_ZERO_DEPLOY_ROOT / "config" / "policy" / "motivo_newG1.yaml")
    joint_names = [str(name) for name in policy_config["isaac_joint_names"]]
    joint_qpos_ids, joint_qvel_ids, _ = build_joint_mappings(rt, model, joint_names)
    body_ids, pelvis_body_id, extend_parent_body_id = build_body_ids(rt, model)
    target_body_pos = load_target_body_positions(rt, model, data, str(summary["goal_key"]), joint_qpos_ids, body_ids, extend_parent_body_id)
    goal_latents = rt.joblib.load(BFM_ZERO_DEPLOY_ROOT / "model" / "goal_inference" / "goal_reaching.pkl")
    target_goal_latent = rt.np.asarray(goal_latents[str(summary["goal_key"])], dtype=rt.np.float32).reshape(-1)
    latent_model = rt.load_model_from_checkpoint_dir(str(BFM_ZERO_DEPLOY_ROOT / "model" / "checkpoint"), device="cpu")
    latent_model.eval()

    trajectories = [rt.np.load(REPO_ROOT / str(run_summary["trajectory_path"])) for run_summary in run_summaries]
    time_vectors = [rt.np.asarray(traj["time_s"], dtype=rt.np.float32) for traj in trajectories]
    max_steps = max(int(time_vec.shape[0]) for time_vec in time_vectors)
    reference_idx = max(range(len(time_vectors)), key=lambda idx: int(time_vectors[idx].shape[0]))
    time_s = time_vectors[reference_idx]
    num_runs = len(run_summaries)
    rollout_steps = int(max_steps)

    mpjpe_mm = rt.np.full((num_runs, rollout_steps), rt.np.nan, dtype=rt.np.float32)
    latent_cosine = rt.np.full((num_runs, rollout_steps), rt.np.nan, dtype=rt.np.float32)
    root_z = rt.np.full((num_runs, rollout_steps), rt.np.nan, dtype=rt.np.float32)
    success = rt.np.zeros(num_runs, dtype=bool)
    trajectory_lengths = rt.np.zeros(num_runs, dtype=rt.np.int32)

    for run_idx, (run_summary, traj) in enumerate(zip(run_summaries, trajectories)):
        qpos = rt.np.asarray(traj["qpos"], dtype=rt.np.float32)
        qvel = rt.np.asarray(traj["qvel"], dtype=rt.np.float32)
        run_steps = int(qpos.shape[0])
        if qvel.shape[0] != run_steps:
            raise ValueError(f"Run {run_idx} qvel length mismatch: {qvel.shape[0]} vs {run_steps}")
        trajectory_lengths[run_idx] = run_steps
        for step_idx in range(run_steps):
            data.qpos[:] = qpos[step_idx]
            data.qvel[:] = qvel[step_idx]
            rt.mujoco.mj_forward(model, data)
            body_pos, body_rot, body_vel, body_ang_vel = extract_body_state(rt, model, data, body_ids, extend_parent_body_id)
            backward_obs = build_backward_obs(
                rt,
                body_pos,
                body_rot,
                body_vel,
                body_ang_vel,
                data.qpos[joint_qpos_ids].astype(rt.np.float32),
                data.qvel[joint_qvel_ids].astype(rt.np.float32),
            )
            z = latent_model.project_z(latent_model.backward_map(backward_obs)).cpu().numpy().reshape(-1)
            mpjpe_mm[run_idx, step_idx] = float(rt.np.linalg.norm(body_pos - target_body_pos, axis=-1).mean() * 1000.0)
            latent_cosine[run_idx, step_idx] = cosine_similarity(rt, z, target_goal_latent)
            root_z[run_idx, step_idx] = float(body_pos[0, 2])
        success[run_idx] = bool(run_summary["success"])

    metrics_path = stage3_dir / "metrics.npz"
    rt.np.savez_compressed(
        metrics_path,
        time_s=time_s,
        mpjpe_mm=mpjpe_mm,
        latent_cosine=latent_cosine,
        root_z=root_z,
        success=success,
        trajectory_lengths=trajectory_lengths,
    )

    success_rate = float(success.mean())
    title_suffix = (
        f"N={num_runs}, seed={manifest_seed_from_stage2(summary)}, "
        f"fallen_z<{summary['fallen_z_threshold_m']:.2f} m, "
        f"recover_z>{summary['recovery_z_threshold_m']:.2f} m, "
        f"horizon={summary['horizon_s']:.1f} s, success={success_rate:.1%}"
    )
    mpjpe_plot = stage3_dir / "mpjpe_lines.png"
    latent_plot = stage3_dir / "latent_cosine_lines.png"
    root_z_plot = stage3_dir / "base_height_lines.png"
    write_plot(rt, mpjpe_mm, time_s, f"BFM-Zero Fallen-Recovery MPJPE ({title_suffix})", "MPJPE (mm)", mpjpe_plot)
    write_plot(rt, latent_cosine, time_s, f"BFM-Zero Fallen-Recovery Latent Cosine ({title_suffix})", "Cosine Similarity", latent_plot)
    write_plot(rt, root_z, time_s, f"BFM-Zero Fallen-Recovery Base Height ({title_suffix})", "Pelvis Height (m)", root_z_plot)

    tiled_video = stage3_dir / "tiled_runs.mp4"
    grid_rows, grid_cols, video_width, video_height = render_tiled_video_from_paths(
        rt,
        model,
        data,
        pelvis_body_id,
        [REPO_ROOT / str(row["trajectory_path"]) for row in run_summaries],
        args.video_fps,
        args.render_width,
        args.render_height,
        tiled_video,
    )
    stage1_tiled_video = stage3_dir / "tiled_stage1_settle.mp4"
    stage1_grid_rows: int | None = None
    stage1_grid_cols: int | None = None
    stage1_video_width: int | None = None
    stage1_video_height: int | None = None
    if stage1_manifest is not None:
        stage1_grid_rows, stage1_grid_cols, stage1_video_width, stage1_video_height = render_tiled_video_from_paths(
            rt,
            model,
            data,
            pelvis_body_id,
            [REPO_ROOT / str(row["settle_trajectory_path"]) for row in stage1_manifest["states"]],
            args.video_fps,
            args.render_width,
            args.render_height,
            stage1_tiled_video,
        )

    latent_min = float(rt.np.nanmin(latent_cosine))
    latent_max = float(rt.np.nanmax(latent_cosine))
    failed_runs = [row for row in run_summaries if not bool(row["success"])]
    failed_recovery_index = None
    failed_recovery_videos: list[str] = []
    failed_settle_index = None
    failed_settle_videos: list[str] = []
    if failed_runs:
        failed_recovery_index, failed_recovery_videos = render_failed_video_group(
            rt,
            model,
            data,
            pelvis_body_id,
            failed_runs,
            lambda run_summary: REPO_ROOT / str(run_summary["trajectory_path"]),
            args.video_fps,
            args.failed_render_width,
            args.failed_render_height,
            layout.stage3_dir,
            "failed_videos_index.json",
            lambda run_name, state_id: f"{run_name}_{state_id}_1080p.mp4",
        )
        if stage1_manifest is not None:
            failed_settle_index, failed_settle_videos = render_failed_video_group(
                rt,
                model,
                data,
                pelvis_body_id,
                failed_runs,
                lambda run_summary: REPO_ROOT / str(run_summary["source_state"]["settle_trajectory_path"]),
                args.video_fps,
                args.failed_render_width,
                args.failed_render_height,
                layout.stage3_dir,
                "failed_settle_videos_index.json",
                lambda run_name, state_id: f"{run_name}_{state_id}_settle_1080p.mp4",
            )
    summary_payload = {
        "run_id": run_id,
        "created_at": now_iso(),
        "stage": "stage3",
        "stage2_summary": repo_rel(stage2_summary_path(stage2_dir)),
        "num_runs": num_runs,
        "goal_key": summary["goal_key"],
        "seed": manifest_seed_from_stage2(summary),
        "fallen_z_threshold_m": summary["fallen_z_threshold_m"],
        "recovery_z_threshold_m": summary["recovery_z_threshold_m"],
        "horizon_s": summary["horizon_s"],
        "success_count": int(success.sum()),
        "success_rate": success_rate,
        "video_grid_rows": grid_rows,
        "video_grid_cols": grid_cols,
        "video_width": video_width,
        "video_height": video_height,
        "stage1_video_grid_rows": stage1_grid_rows,
        "stage1_video_grid_cols": stage1_grid_cols,
        "stage1_video_width": stage1_video_width,
        "stage1_video_height": stage1_video_height,
        "sanity": {
            "mpjpe_recorded_finite": bool(rt.np.all(rt.np.isfinite(mpjpe_mm[~rt.np.isnan(mpjpe_mm)]))),
            "latent_cosine_recorded_finite": bool(rt.np.all(rt.np.isfinite(latent_cosine[~rt.np.isnan(latent_cosine)]))),
            "latent_cosine_in_range": bool(latent_min >= -1.0001 and latent_max <= 1.0001),
            "root_z_recorded_finite": bool(rt.np.all(rt.np.isfinite(root_z[~rt.np.isnan(root_z)]))),
            "variable_length_trajectories": bool(len({int(v) for v in trajectory_lengths.tolist()}) > 1),
            "full_grid_1920x1080": bool(num_runs == 100 and video_width == 1920 and video_height == 1080),
            "full_stage1_grid_1920x1080": bool(
                stage1_video_width is not None
                and stage1_video_height is not None
                and num_runs == 100
                and stage1_video_width == 1920
                and stage1_video_height == 1080
            ),
        },
        "artifacts": {
            "metrics_npz": repo_rel(metrics_path),
            "summary_json": repo_rel(stage3_summary_path(stage3_dir)),
            "mpjpe_plot": repo_rel(mpjpe_plot),
            "latent_plot": repo_rel(latent_plot),
            "base_height_plot": repo_rel(root_z_plot),
            "tiled_video": repo_rel(tiled_video),
        },
    }
    if stage1_manifest_rel:
        summary_payload["stage1_manifest"] = repo_rel(REPO_ROOT / str(stage1_manifest_rel))
        summary_payload["stage1_summary"] = repo_rel(stage1_summary_path((REPO_ROOT / str(stage1_manifest_rel)).parent))
        summary_payload["artifacts"]["stage1_tiled_settle_video"] = repo_rel(stage1_tiled_video)
    if failed_recovery_index is not None:
        summary_payload["artifacts"]["failed_videos_index"] = repo_rel(failed_recovery_index)
        summary_payload["artifacts"]["failed_videos_1080p"] = failed_recovery_videos
    if failed_settle_index is not None:
        summary_payload["artifacts"]["failed_settle_videos_index"] = repo_rel(failed_settle_index)
        summary_payload["artifacts"]["failed_settle_videos_1080p"] = failed_settle_videos
    write_json(stage3_summary_path(stage3_dir), summary_payload)
    print(
        json.dumps(
            {
                "stage3_summary": repo_rel(stage3_summary_path(stage3_dir)),
                "video": repo_rel(tiled_video),
                "stage1_video": repo_rel(stage1_tiled_video) if stage1_manifest_rel else None,
            },
            indent=2,
        )
    )
    return 0


def manifest_seed_from_stage2(stage2_summary: dict[str, Any]) -> int:
    if stage2_summary.get("stage1_manifest"):
        manifest = load_json(REPO_ROOT / str(stage2_summary["stage1_manifest"]))
        return int(manifest["seed"])
    return int(stage2_summary.get("seed", 0))


def load_stage2_context(stage2_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None, EvalLayout, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summary = load_json(stage2_summary_path(stage2_dir))
    run_id = str(summary["run_id"])
    layout = build_layout(run_id)
    run_summaries = [load_json(REPO_ROOT / str(path)) for path in summary["run_summaries"]]
    manifest = load_json(REPO_ROOT / str(summary["stage1_manifest"])) if summary.get("stage1_manifest") else None
    if manifest is not None:
        states_by_id = {str(row["state_id"]): row for row in manifest["states"]}
    else:
        states_by_id = {
            str(row["source_state"]["state_id"]): dict(row["source_state"])
            for row in run_summaries
            if isinstance(row.get("source_state"), dict) and row["source_state"].get("state_id")
        }
    return summary, manifest, layout, run_summaries, states_by_id


def build_model_context(rt: SimpleNamespace) -> dict[str, Any]:
    model = rt.mujoco.MjModel.from_xml_path(str(BFM_ZERO_DEPLOY_ROOT / "data" / "robots" / "g1" / "scene_29dof_freebase.xml"))
    data = rt.mujoco.MjData(model)
    policy_config = load_yaml(BFM_ZERO_DEPLOY_ROOT / "config" / "policy" / "motivo_newG1.yaml")
    robot_config = load_yaml(BFM_ZERO_DEPLOY_ROOT / "config" / "robot" / "g1.yaml")
    joint_names = [str(name) for name in policy_config["isaac_joint_names"]]
    joint_qpos_ids, joint_qvel_ids, _ = build_joint_mappings(rt, model, joint_names)
    body_ids, pelvis_body_id, extend_parent_body_id = build_body_ids(rt, model)
    default_dof_angles = resolve_joint_array(rt, policy_config["default_joint_pos"], joint_names)
    left_right_pairs = [
        (name, name.replace("left_", "right_"))
        for name in joint_names
        if name.startswith("left_") and name.replace("left_", "right_") in joint_names
    ]
    pair_indices = [(joint_names.index(left), joint_names.index(right), left, right) for left, right in left_right_pairs]
    leg_joint_indices = [
        idx
        for idx, name in enumerate(joint_names)
        if any(token in name for token in ("hip_", "knee_", "ankle_"))
    ]
    arm_joint_indices = [
        idx
        for idx, name in enumerate(joint_names)
        if any(token in name for token in ("shoulder_", "elbow_", "wrist_"))
    ]
    waist_joint_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("waist_")]
    return {
        "model": model,
        "data": data,
        "joint_names": joint_names,
        "joint_qpos_ids": joint_qpos_ids,
        "joint_qvel_ids": joint_qvel_ids,
        "body_ids": body_ids,
        "pelvis_body_id": pelvis_body_id,
        "extend_parent_body_id": extend_parent_body_id,
        "default_dof_angles": default_dof_angles,
        "pair_indices": pair_indices,
        "leg_joint_indices": leg_joint_indices,
        "arm_joint_indices": arm_joint_indices,
        "waist_joint_indices": waist_joint_indices,
    }


def quat_to_rpy_wxyz(quat_wxyz: Any) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_rotate_vector_wxyz(rt: SimpleNamespace, quat_wxyz: Any, vec: Any) -> Any:
    q = rt.np.asarray(quat_wxyz, dtype=rt.np.float64)
    v = rt.np.asarray(vec, dtype=rt.np.float64)
    q_vec = q[1:]
    uv = rt.np.cross(q_vec, v)
    uuv = rt.np.cross(q_vec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


def classify_root_orientation(rt: SimpleNamespace, quat_wxyz: Any) -> str:
    body_up_world = quat_rotate_vector_wxyz(rt, quat_wxyz, [0.0, 0.0, 1.0])
    axis = int(rt.np.argmax(rt.np.abs(body_up_world)))
    sign = 1.0 if body_up_world[axis] >= 0.0 else -1.0
    if axis == 2:
        return "upright" if sign > 0.0 else "upside_down"
    if axis == 0:
        return "prone" if sign > 0.0 else "supine"
    return "left_side" if sign > 0.0 else "right_side"


def contact_features_for_state(rt: SimpleNamespace, model: Any, data: Any) -> dict[str, Any]:
    return contact_summary_for_current_state(rt, model, data)


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def safe_std(rt: SimpleNamespace, values: Any) -> float:
    arr = rt.np.asarray(values, dtype=rt.np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def cohen_d(rt: SimpleNamespace, a: Any, b: Any) -> float:
    a = rt.np.asarray(a, dtype=rt.np.float64)
    b = rt.np.asarray(b, dtype=rt.np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    var_a = a.var(ddof=1) if a.size > 1 else 0.0
    var_b = b.var(ddof=1) if b.size > 1 else 0.0
    pooled_num = (a.size - 1) * var_a + (b.size - 1) * var_b
    pooled_den = max(a.size + b.size - 2, 1)
    pooled = math.sqrt(max(pooled_num / pooled_den, 0.0))
    if pooled == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def point_biserial(rt: SimpleNamespace, values: Any, labels: Any) -> float:
    x = rt.np.asarray(values, dtype=rt.np.float64)
    y = rt.np.asarray(labels, dtype=rt.np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size or rt.np.std(x) == 0.0 or rt.np.std(y) == 0.0:
        return 0.0
    return float(rt.np.corrcoef(x, y)[0, 1])


def render_video_from_qpos(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    renderer: Any,
    camera: Any,
    pelvis_body_id: int,
    qpos: Any,
    sample_indices: Any,
    fps: int,
    output_path: Path,
) -> None:
    if rt.imageio is None:
        raise ModuleNotFoundError("imageio is required for video export")
    writer = rt.imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=None)
    try:
        for sample_idx in sample_indices:
            frame = render_frame(rt, renderer, camera, model, data, pelvis_body_id, qpos[int(sample_idx)])
            writer.append_data(frame)
    finally:
        writer.close()


def render_tiled_video_from_paths(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    pelvis_body_id: int,
    trajectory_paths: list[Path],
    fps: int,
    render_width: int,
    render_height: int,
    output_path: Path,
) -> tuple[int, int, int, int]:
    trajectories = [rt.np.load(path) for path in trajectory_paths]
    if not trajectories:
        raise ValueError("No trajectories were provided for tiled rendering")
    max_duration_s = max(float(rt.np.asarray(traj["time_s"], dtype=rt.np.float32)[-1]) for traj in trajectories)
    render_times = rt.np.arange(0.0, max_duration_s + 1e-9, 1.0 / fps, dtype=rt.np.float32)
    renderer, camera = build_renderer(rt, model, render_width, render_height, pelvis_body_id)
    frames = rt.np.zeros((len(trajectories), len(render_times), render_height, render_width, 3), dtype=rt.np.uint8)
    try:
        for run_idx, traj in enumerate(trajectories):
            time_s = rt.np.asarray(traj["time_s"], dtype=rt.np.float32)
            qpos = rt.np.asarray(traj["qpos"], dtype=rt.np.float32)
            sample_indices = rt.np.clip(rt.np.searchsorted(time_s, render_times, side="left"), 0, len(time_s) - 1)
            for frame_idx, sample_idx in enumerate(sample_indices):
                frames[run_idx, frame_idx] = render_frame(rt, renderer, camera, model, data, pelvis_body_id, qpos[int(sample_idx)])
    finally:
        renderer.close()
    return tile_video(rt, frames, fps, output_path)


def render_failed_video_group(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    pelvis_body_id: int,
    failed_runs: list[dict[str, Any]],
    trajectory_path_for_run: Any,
    fps: int,
    render_width: int,
    render_height: int,
    output_dir: Path,
    index_name: str,
    file_name_builder: Any,
) -> tuple[Path, list[str]]:
    renderer, camera = build_renderer(rt, model, render_width, render_height, pelvis_body_id)
    output_paths: list[str] = []
    try:
        for run_summary in failed_runs:
            run_name = Path(str(run_summary["summary_path"])).parent.name
            state_id = str(run_summary["source_state"]["state_id"])
            output_path = output_dir / str(file_name_builder(run_name, state_id))
            traj = rt.np.load(trajectory_path_for_run(run_summary))
            time_s = rt.np.asarray(traj["time_s"], dtype=rt.np.float32)
            qpos = rt.np.asarray(traj["qpos"], dtype=rt.np.float32)
            render_times = rt.np.arange(0.0, float(time_s[-1]) + 1e-9, 1.0 / fps, dtype=rt.np.float32)
            render_indices = rt.np.clip(rt.np.searchsorted(time_s, render_times, side="left"), 0, len(time_s) - 1)
            render_video_from_qpos(rt, model, data, renderer, camera, pelvis_body_id, qpos, render_indices, fps, output_path)
            output_paths.append(repo_rel(output_path))
    finally:
        renderer.close()

    index_path = output_dir / index_name
    write_json(
        index_path,
        {
            "created_at": now_iso(),
            "video_fps": fps,
            "render_width": render_width,
            "render_height": render_height,
            "failed_videos": output_paths,
        },
    )
    return index_path, output_paths


def render_failed_videos(args: argparse.Namespace) -> int:
    rt = runtime()
    stage2_dir = Path(args.stage2_dir)
    summary, _, layout, run_summaries, _ = load_stage2_context(stage2_dir)
    failed_runs = [row for row in run_summaries if not bool(row["success"])]
    if not failed_runs:
        raise ValueError("Stage 2 summary contains no failed runs")

    ctx = build_model_context(rt)
    model = ctx["model"]
    data = ctx["data"]
    index_path, output_paths = render_failed_video_group(
        rt,
        model,
        data,
        int(ctx["pelvis_body_id"]),
        failed_runs,
        lambda run_summary: REPO_ROOT / str(run_summary["trajectory_path"]),
        args.video_fps,
        args.render_width,
        args.render_height,
        layout.stage3_dir,
        "failed_videos_index.json",
        lambda run_name, state_id: f"{run_name}_{state_id}_1080p.mp4",
    )
    summary_path = stage3_summary_path(layout.stage3_dir)
    if summary_path.exists():
        stage3_summary = load_json(summary_path)
        stage3_summary.setdefault("artifacts", {})
        stage3_summary["artifacts"]["failed_videos_index"] = repo_rel(index_path)
        stage3_summary["artifacts"]["failed_videos_1080p"] = output_paths
        write_json(summary_path, stage3_summary)
    print(json.dumps({"failed_video_count": len(output_paths), "failed_videos_index": repo_rel(index_path)}, indent=2))
    return 0


def analyze_failures(args: argparse.Namespace) -> int:
    rt = runtime()
    stage2_dir = Path(args.stage2_dir)
    summary, manifest, layout, run_summaries, states_by_id = load_stage2_context(stage2_dir)
    ctx = build_model_context(rt)
    model = ctx["model"]
    data = ctx["data"]
    joint_names: list[str] = ctx["joint_names"]
    joint_qpos_ids = ctx["joint_qpos_ids"]
    joint_qvel_ids = ctx["joint_qvel_ids"]
    default_dof_angles = ctx["default_dof_angles"]
    pair_indices = ctx["pair_indices"]
    leg_joint_indices = ctx["leg_joint_indices"]
    arm_joint_indices = ctx["arm_joint_indices"]
    waist_joint_indices = ctx["waist_joint_indices"]

    rows: list[dict[str, Any]] = []
    joint_pos_matrix = rt.np.zeros((len(run_summaries), len(joint_names)), dtype=rt.np.float64)
    joint_vel_matrix = rt.np.zeros((len(run_summaries), len(joint_names)), dtype=rt.np.float64)
    fail_mask = rt.np.zeros(len(run_summaries), dtype=bool)

    for idx, run_summary in enumerate(run_summaries):
        state_id = str(run_summary["source_state"]["state_id"])
        state_row = states_by_id[state_id]
        qpos0, qvel0, root_z0 = load_stage_state_npz(REPO_ROOT / str(state_row["state_path"]))
        joint_pos = rt.np.asarray(qpos0[joint_qpos_ids], dtype=rt.np.float64)
        joint_vel = rt.np.asarray(qvel0[joint_qvel_ids], dtype=rt.np.float64)
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        rt.mujoco.mj_forward(model, data)
        contacts = contact_features_for_state(rt, model, data)
        roll, pitch, yaw = quat_to_rpy_wxyz(qpos0[3:7])
        orientation = classify_root_orientation(rt, qpos0[3:7])

        leg_asymmetry = safe_mean([abs(joint_pos[left] - joint_pos[right]) for left, right, _, _ in pair_indices if left in leg_joint_indices])
        arm_asymmetry = safe_mean([abs(joint_pos[left] - joint_pos[right]) for left, right, _, _ in pair_indices if left in arm_joint_indices])
        row = {
            "run_name": Path(str(run_summary["summary_path"])).parent.name,
            "run_index": idx,
            "state_id": state_id,
            "state_source": str(state_row["source"]),
            "goal_key": state_row.get("goal_key"),
            "success": bool(run_summary["success"]),
            "failure": not bool(run_summary["success"]),
            "final_window_mean_root_z_m": float(run_summary["final_window_mean_root_z_m"]),
            "max_root_z_m": float(run_summary["max_root_z_m"]),
            "initial_root_z_m": float(root_z0),
            "root_xy_radius_m": float(rt.np.linalg.norm(qpos0[0:2])),
            "root_roll_deg": math.degrees(roll),
            "root_pitch_deg": math.degrees(pitch),
            "root_yaw_deg": math.degrees(yaw),
            "root_tilt_deg": math.degrees(math.sqrt(roll * roll + pitch * pitch)),
            "root_lin_speed_mps": float(rt.np.linalg.norm(qvel0[0:3])),
            "root_ang_speed_rps": float(rt.np.linalg.norm(qvel0[3:6])),
            "joint_vel_rms": float(rt.np.sqrt(rt.np.mean(joint_vel * joint_vel))),
            "joint_vel_abs_max": float(rt.np.max(rt.np.abs(joint_vel))),
            "joint_abs_mean_rad": float(rt.np.mean(rt.np.abs(joint_pos))),
            "joint_abs_max_rad": float(rt.np.max(rt.np.abs(joint_pos))),
            "joint_pos_l2_from_default": float(rt.np.linalg.norm(joint_pos - default_dof_angles)),
            "leg_joint_abs_mean_rad": float(rt.np.mean(rt.np.abs(joint_pos[leg_joint_indices]))),
            "arm_joint_abs_mean_rad": float(rt.np.mean(rt.np.abs(joint_pos[arm_joint_indices]))),
            "waist_abs_sum_rad": float(rt.np.sum(rt.np.abs(joint_pos[waist_joint_indices]))),
            "hip_pitch_abs_sum_rad": float(abs(joint_pos[joint_names.index("left_hip_pitch_joint")]) + abs(joint_pos[joint_names.index("right_hip_pitch_joint")])),
            "knee_flex_sum_rad": float(joint_pos[joint_names.index("left_knee_joint")] + joint_pos[joint_names.index("right_knee_joint")]),
            "ankle_pitch_abs_sum_rad": float(abs(joint_pos[joint_names.index("left_ankle_pitch_joint")]) + abs(joint_pos[joint_names.index("right_ankle_pitch_joint")])),
            "leg_asymmetry_l1_rad": float(leg_asymmetry),
            "arm_asymmetry_l1_rad": float(arm_asymmetry),
            "orientation_bin": orientation,
            **contacts,
        }
        rows.append(row)
        joint_pos_matrix[idx] = joint_pos
        joint_vel_matrix[idx] = joint_vel
        fail_mask[idx] = row["failure"]

    fail_rows = [row for row in rows if row["failure"]]
    success_rows = [row for row in rows if not row["failure"]]
    if not fail_rows:
        raise ValueError("No failed runs available for analysis")
    labels = fail_mask.astype(rt.np.float64)

    scalar_feature_names = [
        "initial_root_z_m",
        "root_xy_radius_m",
        "root_roll_deg",
        "root_pitch_deg",
        "root_yaw_deg",
        "root_tilt_deg",
        "root_lin_speed_mps",
        "root_ang_speed_rps",
        "joint_vel_rms",
        "joint_vel_abs_max",
        "joint_abs_mean_rad",
        "joint_abs_max_rad",
        "joint_pos_l2_from_default",
        "leg_joint_abs_mean_rad",
        "arm_joint_abs_mean_rad",
        "waist_abs_sum_rad",
        "hip_pitch_abs_sum_rad",
        "knee_flex_sum_rad",
        "ankle_pitch_abs_sum_rad",
        "leg_asymmetry_l1_rad",
        "arm_asymmetry_l1_rad",
        "self_contact_pair_count",
        "floor_contact_body_count",
    ]
    scalar_rankings = []
    for name in scalar_feature_names:
        fail_vals = [float(row[name]) for row in fail_rows]
        succ_vals = [float(row[name]) for row in success_rows]
        all_vals = [float(row[name]) for row in rows]
        scalar_rankings.append(
            {
                "feature": name,
                "fail_mean": safe_mean(fail_vals),
                "success_mean": safe_mean(succ_vals),
                "mean_diff": safe_mean(fail_vals) - safe_mean(succ_vals),
                "fail_std": safe_std(rt, fail_vals),
                "success_std": safe_std(rt, succ_vals),
                "cohen_d": cohen_d(rt, fail_vals, succ_vals),
                "point_biserial": point_biserial(rt, all_vals, labels),
            }
        )
    scalar_rankings.sort(key=lambda row: abs(float(row["cohen_d"])), reverse=True)

    joint_rankings = []
    success_joint_mean = joint_pos_matrix[~fail_mask].mean(axis=0)
    success_joint_std = joint_pos_matrix[~fail_mask].std(axis=0, ddof=1)
    success_joint_std[success_joint_std == 0.0] = 1e-6
    for idx, name in enumerate(joint_names):
        fail_vals = joint_pos_matrix[fail_mask, idx]
        succ_vals = joint_pos_matrix[~fail_mask, idx]
        joint_rankings.append(
            {
                "joint": name,
                "fail_mean_rad": float(fail_vals.mean()),
                "success_mean_rad": float(succ_vals.mean()),
                "mean_diff_rad": float(fail_vals.mean() - succ_vals.mean()),
                "cohen_d": cohen_d(rt, fail_vals, succ_vals),
                "point_biserial": point_biserial(rt, joint_pos_matrix[:, idx], labels),
            }
        )
    joint_rankings.sort(key=lambda row: abs(float(row["cohen_d"])), reverse=True)

    joint_velocity_rankings = []
    for idx, name in enumerate(joint_names):
        fail_vals = joint_vel_matrix[fail_mask, idx]
        succ_vals = joint_vel_matrix[~fail_mask, idx]
        joint_velocity_rankings.append(
            {
                "joint": name,
                "fail_mean_rad_s": float(fail_vals.mean()),
                "success_mean_rad_s": float(succ_vals.mean()),
                "mean_diff_rad_s": float(fail_vals.mean() - succ_vals.mean()),
                "cohen_d": cohen_d(rt, fail_vals, succ_vals),
                "point_biserial": point_biserial(rt, joint_vel_matrix[:, idx], labels),
            }
        )
    joint_velocity_rankings.sort(key=lambda row: abs(float(row["cohen_d"])), reverse=True)

    orientation_stats: list[dict[str, Any]] = []
    for orientation in sorted({str(row["orientation_bin"]) for row in rows}):
        total = sum(1 for row in rows if row["orientation_bin"] == orientation)
        failed = sum(1 for row in fail_rows if row["orientation_bin"] == orientation)
        orientation_stats.append(
            {
                "orientation": orientation,
                "count": total,
                "failed": failed,
                "failure_rate": float(failed / total) if total else float("nan"),
            }
        )
    orientation_stats.sort(key=lambda row: (float(row["failure_rate"]), row["count"]), reverse=True)

    contact_pair_stats: list[dict[str, Any]] = []
    all_pairs = sorted({pair for row in rows for pair in row["self_contact_pairs"]})
    for pair in all_pairs:
        total = sum(1 for row in rows if pair in row["self_contact_pairs"])
        failed = sum(1 for row in fail_rows if pair in row["self_contact_pairs"])
        if total == 0:
            continue
        contact_pair_stats.append(
            {
                "pair": pair,
                "count": total,
                "failed": failed,
                "failure_rate": float(failed / total),
            }
        )
    contact_pair_stats.sort(key=lambda row: (row["failure_rate"], row["failed"], row["count"]), reverse=True)

    failure_notes = []
    for row in fail_rows:
        run_idx = int(row["run_index"])
        zscores = rt.np.abs((joint_pos_matrix[run_idx] - success_joint_mean) / success_joint_std)
        top_joint_ids = rt.np.argsort(zscores)[::-1][:3]
        top_joints = [
            {
                "joint": joint_names[int(joint_id)],
                "value_rad": float(joint_pos_matrix[run_idx, int(joint_id)]),
                "success_mean_rad": float(success_joint_mean[int(joint_id)]),
                "z_score": float(zscores[int(joint_id)]),
            }
            for joint_id in top_joint_ids
        ]
        severity = "near_recovery" if float(row["final_window_mean_root_z_m"]) > 0.6 else "grounded"
        failure_notes.append(
            {
                "run_name": row["run_name"],
                "state_id": row["state_id"],
                "severity": severity,
                "final_window_mean_root_z_m": row["final_window_mean_root_z_m"],
                "max_root_z_m": row["max_root_z_m"],
                "orientation_bin": row["orientation_bin"],
                "self_contact_pairs": row["self_contact_pairs"],
                "floor_contact_bodies": row["floor_contact_bodies"],
                "top_joint_outliers": top_joints,
            }
        )

    analysis_json = layout.stage3_dir / "failure_analysis.json"
    analysis_payload = {
        "run_id": str(summary["run_id"]),
        "created_at": now_iso(),
        "stage2_summary": repo_rel(stage2_summary_path(stage2_dir)),
        "stage3_summary": repo_rel(stage3_summary_path(layout.stage3_dir)),
        "num_runs": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(fail_rows),
        "scalar_rankings": scalar_rankings,
        "joint_rankings": joint_rankings,
        "joint_velocity_rankings": joint_velocity_rankings,
        "orientation_stats": orientation_stats,
        "contact_pair_stats": contact_pair_stats,
        "failure_notes": failure_notes,
    }
    write_json(analysis_json, analysis_payload)

    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    failed_video_index = layout.stage3_dir / "failed_videos_index.json"
    failed_settle_video_index = layout.stage3_dir / "failed_settle_videos_index.json"
    failed_video_paths = []
    failed_settle_video_paths = []
    if failed_video_index.exists():
        failed_video_paths = list(load_json(failed_video_index).get("failed_videos", []))
    if failed_settle_video_index.exists():
        failed_settle_video_paths = list(load_json(failed_settle_video_index).get("failed_videos", []))

    top_scalar = scalar_rankings[:8]
    top_joints = joint_rankings[:8]
    top_joint_vels = joint_velocity_rankings[:5]
    top_contact_pairs = [row for row in contact_pair_stats if row["failed"] > 0][:8]
    report_lines = [
        "# BFM-Zero Fallen-Recovery Failure Analysis",
        "",
        f"Generated: `{now_iso()}`",
        "",
        "## Scope",
        "",
        f"- Run set: `{summary['run_id']}`",
        f"- Stage 2 summary: `{repo_rel(stage2_summary_path(stage2_dir))}`",
        f"- Stage 3 summary: `{repo_rel(stage3_summary_path(layout.stage3_dir))}`",
        f"- Population: `{len(rows)}` total runs, `{len(success_rows)}` successes, `{len(fail_rows)}` failures",
        f"- Recovery criterion: final `0.5 s` mean root height `> {summary['recovery_z_threshold_m']:.2f} m`",
        "",
        "## Failed Run Inventory",
        "",
        "| Run | State | Source | Final mean root z (m) | Max root z (m) | Orientation | Self-contact pairs |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for note in failure_notes:
        report_lines.append(
            f"| `{note['run_name']}` | `{note['state_id']}` | `{states_by_id[note['state_id']]['source']}` | "
            f"{note['final_window_mean_root_z_m']:.3f} | {note['max_root_z_m']:.3f} | `{note['orientation_bin']}` | "
            f"{'; '.join(note['self_contact_pairs']) if note['self_contact_pairs'] else 'none'} |"
        )
    report_lines.extend(
        [
            "",
            "## Failed Video Exports",
            "",
            f"- Tiled video: `{repo_rel(layout.stage3_dir / 'tiled_runs.mp4')}`",
            f"- Stage 1 tiled passive-settle video: `{repo_rel(layout.stage3_dir / 'tiled_stage1_settle.mp4')}`" if summary.get("stage1_manifest") else "- Stage 1 tiled passive-settle video: not available for induced-fall runs",
            f"- Failed-video index: `{repo_rel(failed_video_index) if failed_video_index.exists() else 'not generated yet'}`",
            f"- Failed settle-video index: `{repo_rel(failed_settle_video_index) if failed_settle_video_index.exists() else 'not generated yet'}`" if summary.get("stage1_manifest") else "- Failed settle-video index: not available without Stage 1 settle trajectories",
            "",
            "Failed recovery videos:",
        ]
    )
    report_lines.extend([f"- `{path}`" for path in failed_video_paths] or ["- No failed-video exports were present when this report was generated."])
    report_lines.extend(["", "Failed passive-settle videos:"])
    if summary.get("stage1_manifest"):
        report_lines.extend([f"- `{path}`" for path in failed_settle_video_paths] or ["- No failed settle-video exports were present when this report was generated."])
    else:
        report_lines.append("- No Stage 1 settle trajectories exist for induced-fall runs.")

    report_lines.extend(
        [
            "",
            "## Strongest Scalar Separators",
            "",
            "| Feature | Failure mean | Success mean | Mean diff | Cohen d | Point-biserial r |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_scalar:
        report_lines.append(
            f"| `{row['feature']}` | {row['fail_mean']:.3f} | {row['success_mean']:.3f} | {row['mean_diff']:.3f} | "
            f"{row['cohen_d']:.2f} | {row['point_biserial']:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Joint Position Outliers",
            "",
            "| Joint | Failure mean (rad) | Success mean (rad) | Mean diff (rad) | Cohen d | r |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_joints:
        report_lines.append(
            f"| `{row['joint']}` | {row['fail_mean_rad']:.3f} | {row['success_mean_rad']:.3f} | {row['mean_diff_rad']:.3f} | "
            f"{row['cohen_d']:.2f} | {row['point_biserial']:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Joint Velocity Outliers",
            "",
            "| Joint | Failure mean (rad/s) | Success mean (rad/s) | Mean diff (rad/s) | Cohen d | r |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_joint_vels:
        report_lines.append(
            f"| `{row['joint']}` | {row['fail_mean_rad_s']:.3f} | {row['success_mean_rad_s']:.3f} | {row['mean_diff_rad_s']:.3f} | "
            f"{row['cohen_d']:.2f} | {row['point_biserial']:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Orientation Bins",
            "",
            "Orientation labels are approximate axis-based bins inferred from the floating-base quaternion, assuming the torso body frame uses `+x` forward, `+y` left, `+z` up.",
            "",
            "| Orientation | Count | Failures | Failure rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in orientation_stats:
        report_lines.append(f"| `{row['orientation']}` | {row['count']} | {row['failed']} | {row['failure_rate']:.2%} |")

    report_lines.extend(
        [
            "",
            "## Self-Collision Signatures",
            "",
            "| Body pair | Count | Failures | Failure rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_contact_pairs:
        report_lines.append(f"| `{row['pair']}` | {row['count']} | {row['failed']} | {row['failure_rate']:.2%} |")

    report_lines.extend(
        [
            "",
            "## Per-Failure Notes",
            "",
        ]
    )
    for note in failure_notes:
        report_lines.append(f"### `{note['run_name']}` / `{note['state_id']}`")
        report_lines.append("")
        report_lines.append(f"- Severity: `{note['severity']}`")
        report_lines.append(f"- Final mean root height: `{note['final_window_mean_root_z_m']:.3f} m`")
        report_lines.append(f"- Max root height during rollout: `{note['max_root_z_m']:.3f} m`")
        report_lines.append(f"- Orientation bin: `{note['orientation_bin']}`")
        report_lines.append(f"- Self-contact pairs: `{'; '.join(note['self_contact_pairs']) if note['self_contact_pairs'] else 'none'}`")
        report_lines.append(f"- Floor-contact bodies: `{'; '.join(note['floor_contact_bodies']) if note['floor_contact_bodies'] else 'none'}`")
        report_lines.append("- Strongest joint-position deviations vs success set:")
        for joint in note["top_joint_outliers"]:
            report_lines.append(
                f"  - `{joint['joint']}` = `{joint['value_rad']:.3f} rad` "
                f"(success mean `{joint['success_mean_rad']:.3f} rad`, z-score `{joint['z_score']:.2f}`)"
            )
        report_lines.append("")

    report_lines.extend(
        [
            "## Best-Effort Findings",
            "",
            f"- The failed set is small (`{len(fail_rows)}` runs), so the strongest signals should be treated as ranking cues rather than hard causal proof.",
            f"- The single strongest pose-level separator is `{top_scalar[0]['feature']}` with Cohen d `{top_scalar[0]['cohen_d']:.2f}`.",
            f"- The single strongest joint-position separator is `{top_joints[0]['joint']}` with Cohen d `{top_joints[0]['cohen_d']:.2f}`.",
            f"- The most failure-loaded self-contact pair is `{top_contact_pairs[0]['pair']}` at `{top_contact_pairs[0]['failure_rate']:.2%}` failure rate when present." if top_contact_pairs else "- No self-contact pair appeared often enough to stand out over the success set.",
            f"- `{sum(1 for note in failure_notes if note['severity'] == 'near_recovery')}` of the `{len(failure_notes)}` failures were near misses; the rest stayed substantially below the recovery threshold.",
            "",
            "## Caveats",
            "",
            "- This analysis is intentionally best effort and observational. It ranks initial-state traits associated with failure but does not isolate policy causality from all confounders.",
            "- Contact features were computed from the Stage 1 saved state at rollout start, not from the entire rollout.",
            f"- Machine-readable details are available in `{repo_rel(analysis_json)}`.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n")

    stage3_summary = load_json(stage3_summary_path(layout.stage3_dir))
    stage3_summary.setdefault("artifacts", {})
    stage3_summary["artifacts"]["failure_analysis_json"] = repo_rel(analysis_json)
    stage3_summary["artifacts"]["failure_analysis_report"] = repo_rel(report_path)
    write_json(stage3_summary_path(layout.stage3_dir), stage3_summary)
    print(json.dumps({"failure_analysis_report": repo_rel(report_path), "failure_analysis_json": repo_rel(analysis_json)}, indent=2))
    return 0


def sim_runner(args: argparse.Namespace) -> int:
    rt = runtime()
    robot_config = load_yaml(Path(args.robot_config))
    scene_config = load_yaml(Path(args.scene_config))
    if args.mode == "replay":
        scene_config["ENABLE_ELASTIC_BAND"] = False
    model = rt.mujoco.MjModel.from_xml_path(str(scene_config["ROBOT_SCENE"]))
    model.opt.timestep = float(scene_config["SIMULATE_DT"])
    data = rt.mujoco.MjData(model)
    if args.initial_state:
        qpos0, qvel0, _ = load_stage_state_npz(Path(args.initial_state))
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        data.ctrl[:] = 0.0
        rt.mujoco.mj_forward(model, data)

    sim_bridge = rt.SimulationBridge(model, data, robot_config, scene_config)
    pelvis_body_id = model.body("pelvis").id
    joint_qvel_ids = rt.np.asarray(sim_bridge.qvel_adrs, dtype=rt.np.int64)
    disturbance_controller = rt.DisturbanceController(model, data, scene_config)
    elastic_band = None
    band_attached_link = None
    elastic_band_auto_released = False
    if scene_config.get("ENABLE_ELASTIC_BAND", False):
        elastic_band = rt.ElasticBand()
        elastic_band.length += 0.1 * int(scene_config.get("ELASTIC_BAND_INITIAL_LENGTH_STEPS", 0))
        if "h1" in robot_config["ROBOT_TYPE"] or "g1" in robot_config["ROBOT_TYPE"]:
            band_attached_link = model.body("torso_link").id
        else:
            band_attached_link = model.body("base_link").id

    ready_payload = {
        "created_at": now_iso(),
        "mode": args.mode,
        "robot_config": str(args.robot_config),
        "scene_config": str(args.scene_config),
        "sim_dt": float(model.opt.timestep),
    }
    write_json(Path(args.ready_file), ready_payload)

    sim_dt = float(model.opt.timestep)
    next_step = time.perf_counter()

    def sim_step() -> tuple[bool, dict[str, Any] | None]:
        nonlocal elastic_band_auto_released
        sim_bridge.publish_low_state()
        if (
            elastic_band is not None
            and scene_config.get("ELASTIC_BAND_AUTO_RELEASE_ON_FIRST_LOWCMD", False)
            and sim_bridge.has_received_command
            and not elastic_band_auto_released
        ):
            elastic_band.enable = False
        if band_attached_link is not None:
            data.xfrc_applied[band_attached_link, :3] = 0.0
        if elastic_band is not None and elastic_band.enable and band_attached_link is not None:
            pos = data.xpos[band_attached_link]
            lin_vel = data.cvel[band_attached_link, 3:6]
            data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(pos, lin_vel)
        triggered_now = disturbance_controller.step()
        trigger_sample = None
        if triggered_now:
            trigger_sample = {
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "root_z": float(data.xpos[pelvis_body_id, 2]),
            }
        sim_bridge.compute_torques()
        data.ctrl[:] = sim_bridge.torques
        rt.mujoco.mj_step(model, data)
        return triggered_now, trigger_sample

    if not args.trajectory_path or not args.summary_path or not args.done_file:
        raise ValueError(f"{args.mode} mode requires --trajectory-path, --summary-path, and --done-file")

    if args.mode == "replay":
        steps = int(round(args.horizon_s / sim_dt))
        record_count = steps + 1
        qpos_log = rt.np.zeros((record_count, model.nq), dtype=rt.np.float32)
        qvel_log = rt.np.zeros((record_count, model.nv), dtype=rt.np.float32)
        root_z_log = rt.np.zeros(record_count, dtype=rt.np.float32)
        time_log = rt.np.arange(record_count, dtype=rt.np.float32) * sim_dt
        started = False
        record_idx = 0
        while True:
            if not started and Path(args.start_flag).exists():
                started = True
                record_idx = 0
            if started and record_idx < record_count:
                qpos_log[record_idx] = data.qpos
                qvel_log[record_idx] = data.qvel
                root_z_log[record_idx] = float(data.xpos[pelvis_body_id, 2])
                record_idx += 1
                if record_idx >= record_count:
                    break
            sim_step()
            next_step += sim_dt
            time.sleep(max(0.0, next_step - time.perf_counter()))

        trajectory_path = Path(args.trajectory_path)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        rt.np.savez_compressed(
            trajectory_path,
            time_s=time_log,
            qpos=qpos_log,
            qvel=qvel_log,
            root_z=root_z_log,
        )
        final_window_steps = max(1, int(round(0.5 / sim_dt)))
        final_window_mean = float(root_z_log[-final_window_steps:].mean())
        success = bool(final_window_mean > args.recovery_z)
        write_json(
            Path(args.summary_path),
            {
                "created_at": now_iso(),
                "mode": "replay",
                "goal_key": args.goal_key,
                "source_state_id": args.source_state_id,
                "trajectory_path": str(trajectory_path),
                "fallen_z_threshold_m": args.fallen_z,
                "recovery_z_threshold_m": args.recovery_z,
                "horizon_s": args.horizon_s,
                "sim_dt": sim_dt,
                "steps": int(record_count),
                "success": success,
                "final_window_mean_root_z_m": final_window_mean,
                "max_root_z_m": float(root_z_log.max()),
            },
        )
        write_json(Path(args.done_file), {"created_at": now_iso(), "summary_path": str(args.summary_path)})
        return 0

    if args.mode != "induce":
        raise ValueError(f"Unsupported sim runner mode {args.mode}")

    disturbance_cfg = dict(scene_config.get("DISTURBANCE_CONFIG") or {})
    static_dwell_s = float(disturbance_cfg.get("STATIC_DWELL_S", 0.25))
    dwell_steps = max(1, int(round(static_dwell_s / sim_dt)))
    history: deque[dict[str, Any]] = deque(maxlen=dwell_steps)
    static_root_speed_mps = float(disturbance_cfg.get("STATIC_ROOT_SPEED_MPS", 0.15))
    static_ang_speed_rps = float(disturbance_cfg.get("STATIC_ANG_SPEED_RPS", 0.75))
    static_joint_speed_rms = float(disturbance_cfg.get("STATIC_JOINT_SPEED_RMS", 0.6))
    static_root_span_m = float(disturbance_cfg.get("STATIC_ROOT_SPAN_M", 0.03))
    post_start_min_wait_s = float(disturbance_cfg.get("POST_START_MIN_WAIT_S", 0.5))

    started = False
    ready_for_disturbance = False
    observation_started = False
    perturbation_success = False
    start_sim_time_s: float | None = None
    observation_start_time_s: float | None = None
    invalid_root_z_min_m = float(scene_config.get("INVALID_ROOT_Z_MIN_M", DEFAULT_INVALID_ROOT_Z_MIN_M))
    invalid_root_z_max_m = float(scene_config.get("INVALID_ROOT_Z_MAX_M", DEFAULT_INVALID_ROOT_Z_MAX_M))
    qpos_log: list[Any] = []
    qvel_log: list[Any] = []
    root_z_log: list[float] = []
    time_log: list[float] = []
    timeout_deadline_s: float | None = None
    failure_reason = ""

    while True:
        current_speed = speed_summary(rt, model, data, pelvis_body_id, joint_qvel_ids)
        if not started and Path(args.start_flag).exists():
            started = True
            start_sim_time_s = float(data.time)
            timeout_deadline_s = time.monotonic() + args.timeout_s
            history.clear()

        if started and not observation_started:
            history.append(stability_sample(rt, data, pelvis_body_id, joint_qvel_ids))
            enough_wait = start_sim_time_s is not None and float(data.time) - start_sim_time_s >= post_start_min_wait_s
            ready_for_disturbance = bool(
                sim_bridge.has_received_command
                and enough_wait
                and len(history) == dwell_steps
                and history_is_stable(
                    rt,
                    history,
                    static_root_speed_mps,
                    static_ang_speed_rps,
                    static_joint_speed_rms,
                    static_root_span_m,
                )
            )
        disturbance_controller.maybe_write_status(
            {
                "created_at": now_iso(),
                "mode": args.mode,
                "started": started,
                "has_received_lowcmd": bool(sim_bridge.has_received_command),
                "ready_for_disturbance": ready_for_disturbance,
                "static_window_ok": ready_for_disturbance,
                "root_lin_speed_mps": current_speed["root_lin_speed_mps"],
                "root_ang_speed_rps": current_speed["root_ang_speed_rps"],
                "joint_vel_rms": current_speed["joint_vel_rms"],
                "perturbation_success": perturbation_success,
            },
            force=not started or ready_for_disturbance or observation_started,
        )

        if started and not observation_started and timeout_deadline_s is not None and time.monotonic() >= timeout_deadline_s:
            failure_reason = "timed_out_waiting_for_disturbance"
            break

        triggered_now, trigger_sample = sim_step()
        if not rt.np.all(rt.np.isfinite(data.qpos)) or not rt.np.all(rt.np.isfinite(data.qvel)):
            failure_reason = "simulation_invalid"
            break
        if (
            elastic_band is not None
            and scene_config.get("ELASTIC_BAND_AUTO_RELEASE_ON_FIRST_LOWCMD", False)
            and sim_bridge.has_received_command
            and not elastic_band_auto_released
        ):
            elastic_band.enable = False
            elastic_band_auto_released = True
        next_step += sim_dt
        time.sleep(max(0.0, next_step - time.perf_counter()))

        if triggered_now and not observation_started and trigger_sample is not None:
            observation_started = True
            observation_start_time_s = float(disturbance_controller.last_event.sim_time_s if disturbance_controller.last_event else data.time)
            qpos_log.append(trigger_sample["qpos"])
            qvel_log.append(trigger_sample["qvel"])
            root_z_log.append(float(trigger_sample["root_z"]))
            time_log.append(0.0)
            perturbation_success = bool(trigger_sample["root_z"] < args.fallen_z)

        if observation_started and observation_start_time_s is not None:
            elapsed_s = float(data.time) - observation_start_time_s
            qpos_log.append(data.qpos.copy())
            qvel_log.append(data.qvel.copy())
            root_z_value = float(data.xpos[pelvis_body_id, 2])
            root_z_log.append(root_z_value)
            time_log.append(max(0.0, elapsed_s))
            if elapsed_s <= 1.0 + 1e-9 and root_z_value < args.fallen_z:
                perturbation_success = True
            if root_z_value < invalid_root_z_min_m or root_z_value > invalid_root_z_max_m:
                failure_reason = "simulation_unrealistic_root_height"
                break
            if elapsed_s >= args.horizon_s - 1e-9:
                break

    trajectory_path = Path(args.trajectory_path)
    summary_payload = {
        "created_at": now_iso(),
        "mode": "induce",
        "goal_key": args.goal_key,
        "source_state_id": args.source_state_id,
        "trajectory_path": str(trajectory_path),
        "fallen_z_threshold_m": args.fallen_z,
        "recovery_z_threshold_m": args.recovery_z,
        "horizon_s": args.horizon_s,
        "sim_dt": sim_dt,
        "success": False,
        "perturbation_success": perturbation_success,
        "failure_reason": failure_reason,
        "disturbance_method": None if disturbance_controller.last_event is None else disturbance_controller.last_event.method,
        "disturbance_event": None if disturbance_controller.last_event is None else disturbance_controller.last_event.to_dict(),
        "final_window_mean_root_z_m": float("nan"),
        "max_root_z_m": float("nan"),
        "min_root_z_first_1s_m": float("nan"),
        "elastic_band_auto_released": elastic_band_auto_released,
    }
    if qpos_log:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        qpos_arr = rt.np.asarray(qpos_log, dtype=rt.np.float32)
        qvel_arr = rt.np.asarray(qvel_log, dtype=rt.np.float32)
        root_z_arr = rt.np.asarray(root_z_log, dtype=rt.np.float32)
        time_arr = rt.np.asarray(time_log, dtype=rt.np.float32)
        rt.np.savez_compressed(
            trajectory_path,
            time_s=time_arr,
            qpos=qpos_arr,
            qvel=qvel_arr,
            root_z=root_z_arr,
        )
        final_window_mask = time_arr >= max(0.0, float(time_arr[-1]) - 0.5)
        final_window_mean = float(root_z_arr[final_window_mask].mean())
        first_window_mask = time_arr <= 1.0 + 1e-9
        min_root_z_first_1s = float(root_z_arr[first_window_mask].min()) if first_window_mask.any() else float("nan")
        summary_payload["steps"] = int(qpos_arr.shape[0])
        summary_payload["success"] = bool(not failure_reason and final_window_mean > args.recovery_z)
        summary_payload["final_window_mean_root_z_m"] = final_window_mean
        summary_payload["max_root_z_m"] = float(root_z_arr.max())
        summary_payload["min_root_z_first_1s_m"] = min_root_z_first_1s
        summary_payload["perturbation_success"] = bool(min_root_z_first_1s < args.fallen_z)

    write_json(Path(args.summary_path), summary_payload)
    disturbance_controller.maybe_write_status(
        {
            "created_at": now_iso(),
            "mode": args.mode,
            "started": started,
            "has_received_lowcmd": bool(sim_bridge.has_received_command),
            "ready_for_disturbance": ready_for_disturbance,
            "static_window_ok": ready_for_disturbance,
            "perturbation_success": bool(summary_payload["perturbation_success"]),
        },
        force=True,
    )
    write_json(Path(args.done_file), {"created_at": now_iso(), "summary_path": str(args.summary_path)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BFM-Zero fallen-recovery evaluation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("stage1", help="Generate static fallen initial states")
    p1.add_argument("--run-id", type=str, default=None)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--random-count", type=int, default=98)
    p1.add_argument("--goal-keys", nargs="+", default=list(DEFAULT_FALL_GOALS))
    p1.add_argument("--fallen-z", type=float, default=DEFAULT_FALLEN_Z_M)
    p1.add_argument("--settle-steps", type=int, default=150)
    p1.add_argument("--max-sample-attempts", type=int, default=1000)
    p1.add_argument("--capture-timeout-s", type=float, default=30.0)
    p1.add_argument("--stable-dwell-s", type=float, default=0.25)
    p1.add_argument("--stable-root-speed-mps", type=float, default=0.15)
    p1.add_argument("--stable-ang-speed-rps", type=float, default=0.75)
    p1.add_argument("--stable-joint-speed-rms", type=float, default=0.6)
    p1.add_argument("--stable-root-span-m", type=float, default=0.03)
    p1.add_argument("--port-base", type=int, default=26000)

    p2 = subparsers.add_parser("stage2", help="Replay saved states or induce falls with isolated simulator + deployer runs")
    p2.add_argument("--stage1-manifest", type=str, default=None)
    p2.add_argument("--run-id", type=str, default=None)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--goal-key", type=str, default=DEFAULT_GOAL_KEY)
    p2.add_argument("--fallen-z", type=float, default=DEFAULT_FALLEN_Z_M)
    p2.add_argument("--recovery-z", type=float, default=DEFAULT_RECOVERY_Z_M)
    p2.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    p2.add_argument("--port-base", type=int, default=28000)
    p2.add_argument("--policy-ready-timeout-s", type=float, default=30.0)
    p2.add_argument("--induce-fall-in-simulator", action="store_true")
    p2.add_argument("--num-runs", type=int, default=100)
    p2.add_argument("--max-attempts-per-run", type=int, default=10)
    p2.add_argument("--static-ready-timeout-s", type=float, default=45.0)
    p2.add_argument("--post-start-min-wait-s", type=float, default=0.5)
    p2.add_argument("--stable-dwell-s", type=float, default=0.25)
    p2.add_argument("--stable-root-speed-mps", type=float, default=0.15)
    p2.add_argument("--stable-ang-speed-rps", type=float, default=0.75)
    p2.add_argument("--stable-joint-speed-rms", type=float, default=0.6)
    p2.add_argument("--stable-root-span-m", type=float, default=0.03)
    p2.add_argument("--disturbance-method-override", choices=["mixed", "wrench", "velocity_delta"], default="velocity_delta")
    p2.add_argument("--disturbance-scale-multiplier", type=float, default=1.0)
    p2.add_argument("--wrench-force-min-n", type=float, default=DEFAULT_WRENCH_FORCE_MIN_N)
    p2.add_argument("--wrench-force-max-n", type=float, default=DEFAULT_WRENCH_FORCE_MAX_N)
    p2.add_argument("--wrench-torque-min-nm", type=float, default=DEFAULT_WRENCH_TORQUE_MIN_NM)
    p2.add_argument("--wrench-torque-max-nm", type=float, default=DEFAULT_WRENCH_TORQUE_MAX_NM)
    p2.add_argument("--wrench-duration-s", type=float, default=DEFAULT_WRENCH_DURATION_S)
    p2.add_argument("--linear-velocity-min-mps", type=float, default=DEFAULT_LINEAR_VELOCITY_MIN_MPS)
    p2.add_argument("--linear-velocity-max-mps", type=float, default=DEFAULT_LINEAR_VELOCITY_MAX_MPS)
    p2.add_argument("--angular-velocity-min-rps", type=float, default=DEFAULT_ANGULAR_VELOCITY_MIN_RPS)
    p2.add_argument("--angular-velocity-max-rps", type=float, default=DEFAULT_ANGULAR_VELOCITY_MAX_RPS)

    p2t = subparsers.add_parser("tune-disturbance", help="Run nominal-vs-0.1x disturbance tuning sweeps")
    p2t.add_argument("--run-id", type=str, default=None)
    p2t.add_argument("--seed", type=int, default=0)
    p2t.add_argument("--goal-key", type=str, default=DEFAULT_GOAL_KEY)
    p2t.add_argument("--fallen-z", type=float, default=DEFAULT_FALLEN_Z_M)
    p2t.add_argument("--recovery-z", type=float, default=DEFAULT_RECOVERY_Z_M)
    p2t.add_argument("--horizon-s", type=float, default=DEFAULT_INDUCED_HORIZON_S)
    p2t.add_argument("--port-base", type=int, default=32000)
    p2t.add_argument("--policy-ready-timeout-s", type=float, default=30.0)
    p2t.add_argument("--num-runs", type=int, default=20)
    p2t.add_argument("--max-attempts-per-run", type=int, default=1)
    p2t.add_argument("--static-ready-timeout-s", type=float, default=45.0)
    p2t.add_argument("--post-start-min-wait-s", type=float, default=0.5)
    p2t.add_argument("--stable-dwell-s", type=float, default=0.25)
    p2t.add_argument("--stable-root-speed-mps", type=float, default=0.15)
    p2t.add_argument("--stable-ang-speed-rps", type=float, default=0.75)
    p2t.add_argument("--stable-joint-speed-rms", type=float, default=0.6)
    p2t.add_argument("--stable-root-span-m", type=float, default=0.03)
    p2t.add_argument("--disturbance-method-override", choices=["mixed", "wrench", "velocity_delta"], default="velocity_delta")
    p2t.add_argument("--disturbance-scale-multiplier", type=float, default=1.0)
    p2t.add_argument("--wrench-force-min-n", type=float, default=DEFAULT_WRENCH_FORCE_MIN_N)
    p2t.add_argument("--wrench-force-max-n", type=float, default=DEFAULT_WRENCH_FORCE_MAX_N)
    p2t.add_argument("--wrench-torque-min-nm", type=float, default=DEFAULT_WRENCH_TORQUE_MIN_NM)
    p2t.add_argument("--wrench-torque-max-nm", type=float, default=DEFAULT_WRENCH_TORQUE_MAX_NM)
    p2t.add_argument("--wrench-duration-s", type=float, default=DEFAULT_WRENCH_DURATION_S)
    p2t.add_argument("--linear-velocity-min-mps", type=float, default=DEFAULT_LINEAR_VELOCITY_MIN_MPS)
    p2t.add_argument("--linear-velocity-max-mps", type=float, default=DEFAULT_LINEAR_VELOCITY_MAX_MPS)
    p2t.add_argument("--angular-velocity-min-rps", type=float, default=DEFAULT_ANGULAR_VELOCITY_MIN_RPS)
    p2t.add_argument("--angular-velocity-max-rps", type=float, default=DEFAULT_ANGULAR_VELOCITY_MAX_RPS)
    p2t.add_argument("--tuning-gate-max-success-rate", type=float, default=0.5)

    p3 = subparsers.add_parser("stage3", help="Post-process stage2 logs into metrics, plots, and tiled video")
    p3.add_argument("--stage2-dir", type=str, required=True)
    p3.add_argument("--video-fps", type=int, default=25)
    p3.add_argument("--render-width", type=int, default=192)
    p3.add_argument("--render-height", type=int, default=108)
    p3.add_argument("--failed-render-width", type=int, default=1920)
    p3.add_argument("--failed-render-height", type=int, default=1080)

    p4 = subparsers.add_parser("render-failed-videos", help="Render one 1080p replay video per failed Stage 2 run")
    p4.add_argument("--stage2-dir", type=str, required=True)
    p4.add_argument("--video-fps", type=int, default=25)
    p4.add_argument("--render-width", type=int, default=1920)
    p4.add_argument("--render-height", type=int, default=1080)

    p5 = subparsers.add_parser("analyze-failures", help="Analyze which initial states are associated with failed recoveries")
    p5.add_argument("--stage2-dir", type=str, required=True)
    p5.add_argument("--report-path", type=str, default="BFM_ZERO_FALLEN_RECOVERY_FAILURE_ANALYSIS.md")

    ps = subparsers.add_parser("_sim_runner", help=argparse.SUPPRESS)
    ps.add_argument("--mode", choices=["replay", "induce"], required=True)
    ps.add_argument("--robot-config", type=str, required=True)
    ps.add_argument("--scene-config", type=str, required=True)
    ps.add_argument("--ready-file", type=str, required=True)
    ps.add_argument("--start-flag", type=str, required=True)
    ps.add_argument("--initial-state", type=str, default=None)
    ps.add_argument("--done-file", type=str, default=None)
    ps.add_argument("--summary-path", type=str, default=None)
    ps.add_argument("--trajectory-path", type=str, default=None)
    ps.add_argument("--fallen-z", type=float, default=DEFAULT_FALLEN_Z_M)
    ps.add_argument("--recovery-z", type=float, default=DEFAULT_RECOVERY_Z_M)
    ps.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    ps.add_argument("--goal-key", type=str, default=DEFAULT_GOAL_KEY)
    ps.add_argument("--source-state-id", type=str, default="")
    ps.add_argument("--timeout-s", type=float, default=30.0)

    return parser


def main() -> int:
    ensure_runtime_python()
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "stage1":
        return stage1(args)
    if args.command == "stage2":
        return stage2(args)
    if args.command == "tune-disturbance":
        return tune_disturbance(args)
    if args.command == "stage3":
        return stage3(args)
    if args.command == "render-failed-videos":
        return render_failed_videos(args)
    if args.command == "analyze-failures":
        return analyze_failures(args)
    if args.command == "_sim_runner":
        return sim_runner(args)
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
