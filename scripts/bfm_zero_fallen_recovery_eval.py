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
    current = Path(sys.executable).resolve()
    target = BFM_ZERO_VENV_PYTHON.resolve()
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

    import imageio.v2 as imageio
    import joblib
    import matplotlib
    import mujoco
    import numpy as np
    import torch
    import yaml

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    from sim_env.utils.simulation_bridge import SimulationBridge
    from utils.strings import resolve_matching_names_values

    _RUNTIME = SimpleNamespace(
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


def settle_passively(rt: SimpleNamespace, model: Any, data: Any, settle_steps: int) -> bool:
    data.ctrl[:] = 0.0
    for _ in range(settle_steps):
        rt.mujoco.mj_step(model, data)
        if not rt.np.all(rt.np.isfinite(data.qpos)) or not rt.np.all(rt.np.isfinite(data.qvel)):
            return False
    return True


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
    settle_steps: int,
    max_attempts: int,
) -> tuple[Any, Any, float, int]:
    joint_span = joint_upper - joint_lower
    for attempt in range(1, max_attempts + 1):
        rt.mujoco.mj_resetData(model, data)
        data.qpos[0:2] = rng.uniform(-0.15, 0.15, size=2)
        data.qpos[2] = rng.uniform(0.15, 0.4)
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
        if not settle_passively(rt, model, data, settle_steps):
            continue
        root_z = float(data.xpos[pelvis_body_id, 2])
        if root_z < fallen_z:
            return data.qpos.copy(), data.qvel.copy(), root_z, attempt
    raise RuntimeError(f"Failed to sample a fallen state after {max_attempts} attempts")


def motion_frame_to_state(
    rt: SimpleNamespace,
    model: Any,
    data: Any,
    motion_key: str,
    frame_idx: int,
    joint_qpos_ids: Any,
    joint_qvel_ids: Any,
    settle_steps: int,
) -> tuple[Any, Any]:
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
    settle_passively(rt, model, data, settle_steps)
    rt.mujoco.mj_forward(model, data)
    return data.qpos.copy(), data.qvel.copy()


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
    qpos, _ = motion_frame_to_state(rt, model, data, motion_key, frame_idx, joint_qpos_ids, rt.np.asarray([], dtype=rt.np.int64), 0)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    rt.mujoco.mj_forward(model, data)
    target_body_pos, _, _, _ = extract_body_state(rt, model, data, body_ids, extend_parent_body_id)
    return target_body_pos


def build_renderer(rt: SimpleNamespace, model: Any, width: int, height: int, pelvis_body_id: int) -> tuple[Any, Any]:
    renderer = rt.mujoco.Renderer(model, width=width, height=height)
    camera = rt.mujoco.MjvCamera()
    camera.type = rt.mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = pelvis_body_id
    camera.distance = 2.75
    camera.azimuth = 135.0
    camera.elevation = -20.0
    return renderer, camera


def render_frame(rt: SimpleNamespace, renderer: Any, camera: Any, model: Any, data: Any, qpos: Any) -> Any:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    rt.mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def write_plot(rt: SimpleNamespace, curves: Any, time_s: Any, title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = rt.plt.subplots(figsize=(12, 6))
    for curve in curves:
        ax.plot(time_s, curve, color="#1f77b4", alpha=0.12, linewidth=1.0)
    ax.plot(time_s, curves.mean(axis=0), color="#111111", linewidth=2.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    rt.plt.close(fig)


def tile_video(rt: SimpleNamespace, frames: Any, fps: int, output_path: Path) -> tuple[int, int, int, int]:
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


def write_runtime_configs(run_dir: Path, low_state_port: int, low_cmd_port: int, disable_elastic_band: bool) -> tuple[Path, Path]:
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
    robot_config_path = run_dir / "robot_runtime.yaml"
    scene_config_path = run_dir / "scene_runtime.yaml"
    write_yaml(robot_config_path, robot_config)
    write_yaml(scene_config_path, scene_config)
    return robot_config_path, scene_config_path


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
) -> tuple[Any, Any, float, str] | None:
    motion_key, frame_idx = parse_goal_key(goal_key)
    try:
        qpos, qvel = motion_frame_to_state(rt, model, data, motion_key, frame_idx, joint_qpos_ids, joint_qvel_ids, settle_steps)
    except Exception as exc:
        return None if isinstance(exc, Exception) else None
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    rt.mujoco.mj_forward(model, data)
    root_z = float(data.xpos[pelvis_body_id, 2])
    if root_z >= fallen_z:
        return None
    return qpos, qvel, root_z, "lafan_29dof frame extraction + passive settle"


def capture_goal_state_via_process(
    goal_key: str,
    run_dir: Path,
    fallen_z: float,
    settle_dwell_s: float,
    stable_root_speed_mps: float,
    stable_ang_speed_rps: float,
    stable_joint_speed_rms: float,
    stable_root_span_m: float,
    timeout_s: float,
    low_state_port: int,
    low_cmd_port: int,
) -> tuple[Any, Any, float, str]:
    ready_file = run_dir / "sim_ready.json"
    start_flag = run_dir / "capture_start.flag"
    snapshot_path = run_dir / "captured_goal_state.npz"
    summary_path = run_dir / "captured_goal_state_summary.json"
    robot_config_path, scene_config_path = write_runtime_configs(run_dir, low_state_port, low_cmd_port, True)
    sim = launch_sim_capture(
        run_dir,
        "capture-fallen",
        robot_config_path,
        scene_config_path,
        [
            "--ready-file",
            str(ready_file),
            "--start-flag",
            str(start_flag),
            "--snapshot-path",
            str(snapshot_path),
            "--summary-path",
            str(summary_path),
            "--fallen-z",
            f"{fallen_z:.6f}",
            "--stable-dwell-s",
            f"{settle_dwell_s:.6f}",
            "--stable-root-speed-mps",
            f"{stable_root_speed_mps:.6f}",
            "--stable-ang-speed-rps",
            f"{stable_ang_speed_rps:.6f}",
            "--stable-joint-speed-rms",
            f"{stable_joint_speed_rms:.6f}",
            "--stable-root-span-m",
            f"{stable_root_span_m:.6f}",
            "--timeout-s",
            f"{timeout_s:.6f}",
        ],
    )
    deployer: ManagedProcess | None = None
    try:
        wait_for_path(ready_file, 10.0, "simulator ready file")
        deployer = launch_deployer(run_dir, robot_config_path, 30.0)
        advance_policy_to_goal(deployer, goal_key, 10.0)
        touch(start_flag)
        deployer.send("]")
        deployer.wait_for_marker(f"Switch to goal={goal_key}", 10.0)
        wait_for_path(snapshot_path, timeout_s + 10.0, f"captured goal state for {goal_key}")
        qpos, qvel, root_z = load_stage_state_npz(snapshot_path)
        return qpos, qvel, root_z, "policy rollout capture"
    finally:
        if deployer is not None:
            deployer.terminate()
        sim.terminate()


def stage1(args: argparse.Namespace) -> int:
    rt = runtime()
    layout = build_layout(args.run_id or f"{now_stamp()}_stage1_seed{args.seed}")
    stage1_dir = layout.stage1_dir
    states_dir = stage1_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

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

    for random_index in range(args.random_count):
        qpos, qvel, root_z, attempts = sample_fallen_state(
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
            args.settle_steps,
            args.max_sample_attempts,
        )
        state_id = f"state_{len(manifest_states):03d}"
        state_path = states_dir / f"{state_id}.npz"
        write_stage_state_npz(state_path, qpos, qvel, root_z)
        manifest_states.append(
            {
                "state_id": state_id,
                "index": len(manifest_states),
                "source": "random",
                "goal_key": None,
                "root_z_m": root_z,
                "seed": args.seed,
                "sample_attempts": attempts,
                "state_path": repo_rel(state_path),
                "acceptance": {
                    "fallen_z_threshold_m": args.fallen_z,
                    "settle_steps": args.settle_steps,
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
        if direct is not None:
            qpos, qvel, root_z, reason = direct
            direct_extraction[goal_key] = {"supported": True, "reason": reason}
        else:
            direct_extraction[goal_key] = {"supported": False, "reason": "direct frame extraction did not yield a fallen settled state"}
            qpos, qvel, root_z, reason = capture_goal_state_via_process(
                goal_key=goal_key,
                run_dir=stage1_dir / f"goal_capture_{goal_offset:02d}_{goal_key}",
                fallen_z=args.fallen_z,
                settle_dwell_s=args.stable_dwell_s,
                stable_root_speed_mps=args.stable_root_speed_mps,
                stable_ang_speed_rps=args.stable_ang_speed_rps,
                stable_joint_speed_rms=args.stable_joint_speed_rms,
                stable_root_span_m=args.stable_root_span_m,
                timeout_s=args.capture_timeout_s,
                low_state_port=args.port_base + goal_offset * 2,
                low_cmd_port=args.port_base + goal_offset * 2 + 1,
            )
        if root_z >= args.fallen_z:
            raise RuntimeError(f"Goal-derived state for {goal_key} has root_z={root_z:.4f} >= fallen threshold {args.fallen_z:.4f}")
        state_id = f"state_{len(manifest_states):03d}"
        state_path = states_dir / f"{state_id}.npz"
        write_stage_state_npz(state_path, qpos, qvel, root_z)
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
        "states_dir": repo_rel(states_dir),
        "states": manifest_states,
    }
    manifest_path = stage1_manifest_path(stage1_dir)
    write_json(manifest_path, manifest)
    print(json.dumps({"stage1_manifest": repo_rel(manifest_path), "state_count": len(manifest_states)}, indent=2))
    return 0


def stage2(args: argparse.Namespace) -> int:
    manifest = load_json(Path(args.stage1_manifest))
    run_id = str(manifest["run_id"])
    layout = build_layout(run_id)
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
        robot_config_path, scene_config_path = write_runtime_configs(run_dir, low_state_port, low_cmd_port, True)

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

    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "stage": "stage2",
        "stage1_manifest": repo_rel(Path(args.stage1_manifest)),
        "goal_key": args.goal_key,
        "fallen_z_threshold_m": args.fallen_z,
        "recovery_z_threshold_m": args.recovery_z,
        "horizon_s": args.horizon_s,
        "num_runs": len(run_summaries),
        "success_count": success_count,
        "success_rate": float(success_count / len(run_summaries)),
        "run_summaries": [repo_rel(stage2_runs_dir / f"run_{idx:03d}" / "summary.json") for idx in range(len(run_summaries))],
    }
    summary_path = stage2_summary_path(stage2_dir)
    write_json(summary_path, summary)

    if len(run_summaries) == 100 and success_count == 0:
        investigation_path = stage2_dir / "zero_success_investigation.json"
        max_final_window = max(float(row.get("final_window_mean_root_z_m", float("nan"))) for row in run_summaries)
        max_any_height = max(float(row.get("max_root_z_m", float("nan"))) for row in run_summaries)
        write_json(
            investigation_path,
            {
                "run_id": run_id,
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

    print(json.dumps({"stage2_summary": repo_rel(summary_path), "success_count": success_count, "num_runs": len(run_summaries)}, indent=2))
    return 0


def stage3(args: argparse.Namespace) -> int:
    rt = runtime()
    stage2_dir = Path(args.stage2_dir)
    summary = load_json(stage2_summary_path(stage2_dir))
    run_id = str(summary["run_id"])
    layout = build_layout(run_id)
    stage3_dir = layout.stage3_dir

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

    first_traj = rt.np.load(REPO_ROOT / str(run_summaries[0]["trajectory_path"]))
    time_s = rt.np.asarray(first_traj["time_s"], dtype=rt.np.float32)
    num_runs = len(run_summaries)
    rollout_steps = int(time_s.shape[0])

    mpjpe_mm = rt.np.zeros((num_runs, rollout_steps), dtype=rt.np.float32)
    latent_cosine = rt.np.zeros((num_runs, rollout_steps), dtype=rt.np.float32)
    root_z = rt.np.zeros((num_runs, rollout_steps), dtype=rt.np.float32)
    success = rt.np.zeros(num_runs, dtype=bool)

    for run_idx, run_summary in enumerate(run_summaries):
        traj = rt.np.load(REPO_ROOT / str(run_summary["trajectory_path"]))
        qpos = rt.np.asarray(traj["qpos"], dtype=rt.np.float32)
        qvel = rt.np.asarray(traj["qvel"], dtype=rt.np.float32)
        if qpos.shape[0] != rollout_steps:
            raise ValueError(f"Run {run_idx} trajectory length mismatch: {qpos.shape[0]} vs {rollout_steps}")
        for step_idx in range(rollout_steps):
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
    rt.np.savez_compressed(metrics_path, time_s=time_s, mpjpe_mm=mpjpe_mm, latent_cosine=latent_cosine, root_z=root_z, success=success)

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

    render_times = rt.np.arange(0.0, float(summary["horizon_s"]) + 1e-9, 1.0 / args.video_fps, dtype=rt.np.float32)
    render_indices = rt.np.clip(rt.np.searchsorted(time_s, render_times, side="left"), 0, len(time_s) - 1)
    renderer, camera = build_renderer(rt, model, args.render_width, args.render_height, pelvis_body_id)
    frames = rt.np.zeros((num_runs, len(render_indices), args.render_height, args.render_width, 3), dtype=rt.np.uint8)
    for run_idx, run_summary in enumerate(run_summaries):
        traj = rt.np.load(REPO_ROOT / str(run_summary["trajectory_path"]))
        qpos = rt.np.asarray(traj["qpos"], dtype=rt.np.float32)
        for frame_idx, sample_idx in enumerate(render_indices):
            frames[run_idx, frame_idx] = render_frame(rt, renderer, camera, model, data, qpos[int(sample_idx)])
    renderer.close()
    tiled_video = stage3_dir / "tiled_runs.mp4"
    grid_rows, grid_cols, video_width, video_height = tile_video(rt, frames, args.video_fps, tiled_video)

    latent_min = float(rt.np.nanmin(latent_cosine))
    latent_max = float(rt.np.nanmax(latent_cosine))
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
        "sanity": {
            "mpjpe_finite": bool(rt.np.all(rt.np.isfinite(mpjpe_mm))),
            "latent_cosine_finite": bool(rt.np.all(rt.np.isfinite(latent_cosine))),
            "latent_cosine_in_range": bool(latent_min >= -1.0001 and latent_max <= 1.0001),
            "root_z_finite": bool(rt.np.all(rt.np.isfinite(root_z))),
            "full_grid_1920x1080": bool(num_runs == 100 and video_width == 1920 and video_height == 1080),
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
    write_json(stage3_summary_path(stage3_dir), summary_payload)
    print(json.dumps({"stage3_summary": repo_rel(stage3_summary_path(stage3_dir)), "video": repo_rel(tiled_video)}, indent=2))
    return 0


def manifest_seed_from_stage2(stage2_summary: dict[str, Any]) -> int:
    manifest = load_json(REPO_ROOT / str(stage2_summary["stage1_manifest"]))
    return int(manifest["seed"])


def sim_runner(args: argparse.Namespace) -> int:
    rt = runtime()
    robot_config = load_yaml(Path(args.robot_config))
    scene_config = load_yaml(Path(args.scene_config))
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

    def sim_step() -> None:
        sim_bridge.publish_low_state()
        sim_bridge.compute_torques()
        data.ctrl[:] = sim_bridge.torques
        rt.mujoco.mj_step(model, data)

    if args.mode == "capture-fallen":
        if not args.snapshot_path or not args.summary_path:
            raise ValueError("capture-fallen mode requires --snapshot-path and --summary-path")
        dwell_steps = max(1, int(round(args.stable_dwell_s / sim_dt)))
        history: deque[dict[str, Any]] = deque(maxlen=dwell_steps)
        started = False
        start_wall: float | None = None
        while True:
            if not started and Path(args.start_flag).exists():
                started = True
                start_wall = time.monotonic()
            if started and start_wall is not None and time.monotonic() - start_wall > args.timeout_s:
                write_json(
                    Path(args.summary_path),
                    {
                        "created_at": now_iso(),
                        "status": "timeout",
                        "timeout_s": args.timeout_s,
                        "fallen_z_threshold_m": args.fallen_z,
                    },
                )
                return 1
            root_pos = rt.np.asarray(data.xpos[pelvis_body_id], dtype=rt.np.float32)
            root_vel = rt.np.asarray(data.cvel[pelvis_body_id, 3:6], dtype=rt.np.float32)
            root_ang_vel = rt.np.asarray(data.cvel[pelvis_body_id, 0:3], dtype=rt.np.float32)
            joint_vel = rt.np.asarray(data.qvel[6:], dtype=rt.np.float32)
            if started:
                history.append(
                    {
                        "root_pos": root_pos.copy(),
                        "root_speed": float(rt.np.linalg.norm(root_vel)),
                        "root_ang_speed": float(rt.np.linalg.norm(root_ang_vel)),
                        "joint_vel_rms": float(rt.np.sqrt(rt.np.mean(joint_vel * joint_vel))),
                    }
                )
                if len(history) == dwell_steps and all(sample["root_pos"][2] < args.fallen_z for sample in history):
                    root_span = rt.np.linalg.norm(history[-1]["root_pos"] - history[0]["root_pos"])
                    if (
                        max(sample["root_speed"] for sample in history) <= args.stable_root_speed_mps
                        and max(sample["root_ang_speed"] for sample in history) <= args.stable_ang_speed_rps
                        and max(sample["joint_vel_rms"] for sample in history) <= args.stable_joint_speed_rms
                        and float(root_span) <= args.stable_root_span_m
                    ):
                        snapshot_path = Path(args.snapshot_path)
                        write_stage_state_npz(snapshot_path, data.qpos.copy(), data.qvel.copy(), float(root_pos[2]))
                        write_json(
                            Path(args.summary_path),
                            {
                                "created_at": now_iso(),
                                "status": "captured",
                                "snapshot_path": str(snapshot_path),
                                "root_z_m": float(root_pos[2]),
                                "stable_dwell_s": args.stable_dwell_s,
                                "stable_root_span_m": args.stable_root_span_m,
                            },
                        )
                        return 0
            sim_step()
            next_step += sim_dt
            time.sleep(max(0.0, next_step - time.perf_counter()))

    if args.mode != "replay":
        raise ValueError(f"Unsupported sim runner mode {args.mode}")
    if not args.trajectory_path or not args.summary_path or not args.done_file:
        raise ValueError("replay mode requires --trajectory-path, --summary-path, and --done-file")

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

    p2 = subparsers.add_parser("stage2", help="Replay saved states with isolated simulator + deployer runs")
    p2.add_argument("--stage1-manifest", type=str, required=True)
    p2.add_argument("--goal-key", type=str, default=DEFAULT_GOAL_KEY)
    p2.add_argument("--fallen-z", type=float, default=DEFAULT_FALLEN_Z_M)
    p2.add_argument("--recovery-z", type=float, default=DEFAULT_RECOVERY_Z_M)
    p2.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    p2.add_argument("--port-base", type=int, default=28000)
    p2.add_argument("--policy-ready-timeout-s", type=float, default=30.0)

    p3 = subparsers.add_parser("stage3", help="Post-process stage2 logs into metrics, plots, and tiled video")
    p3.add_argument("--stage2-dir", type=str, required=True)
    p3.add_argument("--video-fps", type=int, default=25)
    p3.add_argument("--render-width", type=int, default=192)
    p3.add_argument("--render-height", type=int, default=108)

    ps = subparsers.add_parser("_sim_runner", help=argparse.SUPPRESS)
    ps.add_argument("--mode", choices=["capture-fallen", "replay"], required=True)
    ps.add_argument("--robot-config", type=str, required=True)
    ps.add_argument("--scene-config", type=str, required=True)
    ps.add_argument("--ready-file", type=str, required=True)
    ps.add_argument("--start-flag", type=str, required=True)
    ps.add_argument("--initial-state", type=str, default=None)
    ps.add_argument("--snapshot-path", type=str, default=None)
    ps.add_argument("--done-file", type=str, default=None)
    ps.add_argument("--summary-path", type=str, default=None)
    ps.add_argument("--trajectory-path", type=str, default=None)
    ps.add_argument("--fallen-z", type=float, default=DEFAULT_FALLEN_Z_M)
    ps.add_argument("--recovery-z", type=float, default=DEFAULT_RECOVERY_Z_M)
    ps.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    ps.add_argument("--goal-key", type=str, default=DEFAULT_GOAL_KEY)
    ps.add_argument("--source-state-id", type=str, default="")
    ps.add_argument("--timeout-s", type=float, default=30.0)
    ps.add_argument("--stable-dwell-s", type=float, default=0.25)
    ps.add_argument("--stable-root-speed-mps", type=float, default=0.15)
    ps.add_argument("--stable-ang-speed-rps", type=float, default=0.75)
    ps.add_argument("--stable-joint-speed-rms", type=float, default=0.6)
    ps.add_argument("--stable-root-span-m", type=float, default=0.03)

    return parser


def main() -> int:
    ensure_runtime_python()
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "stage1":
        return stage1(args)
    if args.command == "stage2":
        return stage2(args)
    if args.command == "stage3":
        return stage3(args)
    if args.command == "_sim_runner":
        return sim_runner(args)
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
