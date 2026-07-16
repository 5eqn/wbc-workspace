from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .config import HeraclesConfig
from .model import HeraclesPlanner
from .rotations import quat_xyzw_to_rot6d, rot6d_to_quat_xyzw


class VelocityModel(Protocol):
    def __call__(
        self,
        trajectory: np.ndarray,
        state: np.ndarray,
        time_value: np.ndarray,
        duration: np.ndarray,
    ) -> np.ndarray: ...


class TorchVelocityModel:
    def __init__(self, model: HeraclesPlanner, device: torch.device):
        self.model = model.eval().to(device)
        self.device = device

    def __call__(
        self,
        trajectory: np.ndarray,
        state: np.ndarray,
        time_value: np.ndarray,
        duration: np.ndarray,
    ) -> np.ndarray:
        with torch.no_grad():
            inputs = [
                torch.from_numpy(value).to(self.device)
                for value in (trajectory, state, time_value, duration)
            ]
            return self.model(*inputs).cpu().numpy()


class OnnxVelocityModel:
    def __init__(self, path: Path, use_cuda: bool = True):
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda and "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(path), providers=providers)

    def __call__(
        self,
        trajectory: np.ndarray,
        state: np.ndarray,
        time_value: np.ndarray,
        duration: np.ndarray,
    ) -> np.ndarray:
        return self.session.run(
            ["velocity"],
            {
                "trajectory": trajectory,
                "state": state,
                "flow_time": time_value,
                "segment_duration": duration,
            },
        )[0]


class PlannerRuntime:
    def __init__(
        self,
        velocity_model: VelocityModel,
        normalization: Path,
        config: HeraclesConfig | None = None,
        seed: int | None = None,
    ):
        if config is None:
            config = HeraclesConfig()
        self.model = velocity_model
        stats = np.load(normalization)
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)
        self.config = config
        self.rng = np.random.default_rng(config.seed if seed is None else seed)

    def plan_keyframes(
        self,
        joint_pos: np.ndarray,
        root_quat_xyzw: np.ndarray,
        original_joint_prefix: np.ndarray,
        original_root_prefix_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = np.concatenate(
            (np.asarray(joint_pos, dtype=np.float32), quat_xyzw_to_rot6d(root_quat_xyzw[None])[0])
        )
        prefix_indices = np.linspace(0, len(original_joint_prefix) - 1, self.config.keyframes)
        frame_axis = np.arange(len(original_joint_prefix))
        warm_joints = CubicSpline(frame_axis, original_joint_prefix, axis=0)(prefix_indices)
        key_times = prefix_indices / self.config.data_hz
        warm_quat = Slerp(
            frame_axis / self.config.data_hz,
            Rotation.from_quat(original_root_prefix_xyzw),
        )(key_times).as_quat()
        warm_state = np.concatenate((warm_joints, quat_xyzw_to_rot6d(warm_quat)), axis=-1)
        warm_residual = (warm_state - state).astype(np.float32)
        warm_residual[0] = 0.0
        warm_normalized = warm_residual / self.std
        noise = self.rng.standard_normal(warm_normalized.shape, dtype=np.float32)
        noise[0] = 0.0
        value = (
            self.config.warm_start_t * noise + (1.0 - self.config.warm_start_t) * warm_normalized
        )[None]
        normalized_state = ((state - self.mean) / self.std)[None].astype(np.float32)
        duration = np.asarray([self.config.horizon_s], dtype=np.float32)
        times = np.linspace(
            self.config.warm_start_t, 0.0, self.config.euler_steps + 1, dtype=np.float32
        )
        for current, following in zip(times[:-1], times[1:], strict=True):
            flow_time = np.asarray([current], dtype=np.float32)
            velocity = self.model(value, normalized_state, flow_time, duration)
            value = value + np.float32(following - current) * velocity
            value[:, 0] = 0.0
        residual = value[0] * self.std
        generated = residual + state
        generated[:, self.config.joint_dim :] = quat_xyzw_to_rot6d(
            rot6d_to_quat_xyzw(generated[:, self.config.joint_dim :])
        )
        return generated, residual, warm_residual

    def interpolate_prefix(self, keyframes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key_time = np.linspace(0.0, self.config.horizon_s, self.config.keyframes)
        frame_time = (
            np.arange(round(self.config.horizon_s * self.config.data_hz) + 1, dtype=np.float64)
            / self.config.data_hz
        )
        joints = CubicSpline(key_time, keyframes[:, : self.config.joint_dim], axis=0)(frame_time)
        quat = rot6d_to_quat_xyzw(keyframes[:, self.config.joint_dim :])
        root = Slerp(key_time, Rotation.from_quat(quat))(frame_time).as_quat()
        return joints.astype(np.float32), root.astype(np.float32)

    def build_sonic_window(
        self,
        generated_joint_prefix: np.ndarray,
        generated_root_prefix_xyzw: np.ndarray,
        original_joint_window: np.ndarray,
        original_root_window_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prefix_frames = len(generated_joint_prefix)
        joints = np.concatenate((generated_joint_prefix, original_joint_window[prefix_frames:]))
        roots = np.concatenate(
            (generated_root_prefix_xyzw, original_root_window_xyzw[prefix_frames:])
        )
        velocities = np.gradient(joints, 1.0 / self.config.data_hz, axis=0).astype(np.float32)
        return joints.astype(np.float32), roots.astype(np.float32), velocities


def load_torch_checkpoint(path: Path, device: torch.device) -> HeraclesPlanner:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = HeraclesConfig(**checkpoint["config"])
    model = HeraclesPlanner(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def export_onnx(checkpoint: Path, output: Path, parity_output: Path) -> dict[str, float | bool]:
    device = torch.device("cpu")
    model = load_torch_checkpoint(checkpoint, device)
    config = model.config
    generator = torch.Generator(device=device).manual_seed(config.seed)
    trajectory = torch.randn(
        1, config.keyframes, config.state_dim, generator=generator, device=device
    )
    state = torch.randn(1, config.state_dim, generator=generator, device=device)
    flow_time = torch.tensor([0.5], device=device)
    duration = torch.tensor([config.horizon_s], device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (trajectory, state, flow_time, duration),
        output,
        input_names=["trajectory", "state", "flow_time", "segment_duration"],
        output_names=["velocity"],
        dynamic_axes={"trajectory": {0: "batch"}, "state": {0: "batch"}, "velocity": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    with torch.no_grad():
        torch_output = model(trajectory, state, flow_time, duration).cpu().numpy()
    onnx_output = OnnxVelocityModel(output, use_cuda=False)(
        trajectory.cpu().numpy(),
        state.cpu().numpy(),
        flow_time.cpu().numpy(),
        duration.cpu().numpy(),
    )
    maximum_error = float(np.max(np.abs(torch_output - onnx_output)))
    report = {"maximum_absolute_error": maximum_error, "passed": maximum_error <= 1e-5}
    parity_output.parent.mkdir(parents=True, exist_ok=True)
    parity_output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"PyTorch/ONNX parity failed: {maximum_error}")
    return report


def benchmark_inference(
    model_path: Path, normalization: Path, output: Path, iterations: int = 200
) -> dict[str, float | int | bool]:
    config = HeraclesConfig()
    runtime = PlannerRuntime(OnnxVelocityModel(model_path), normalization, config)
    joint = np.zeros(config.joint_dim, dtype=np.float32)
    quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    frames = round(config.horizon_s * config.data_hz) + 1
    original_joints = np.zeros((frames, config.joint_dim), dtype=np.float32)
    original_roots = np.repeat(quat[None], frames, axis=0)
    for _ in range(10):
        runtime.plan_keyframes(joint, quat, original_joints, original_roots)
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        runtime.plan_keyframes(joint, quat, original_joints, original_roots)
        timings.append(time.perf_counter() - started)
    p99 = float(np.quantile(timings, 0.99))
    report = {
        "iterations": iterations,
        "mean_ms": float(np.mean(timings) * 1000.0),
        "p99_ms": p99 * 1000.0,
        "required_period_ms": 1000.0 / config.replan_hz,
        "sustained_25hz": p99 <= 1.0 / config.replan_hz,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["sustained_25hz"]:
        raise RuntimeError(f"25 Hz inference gate failed: p99={report['p99_ms']:.2f} ms")
    return report
