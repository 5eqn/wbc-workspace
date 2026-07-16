from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import HeraclesConfig
from .inference import OnnxVelocityModel, PlannerRuntime
from .zmq_protocol import PosePublisher, SimStateSubscriber, append_planner_log


def _window(value: np.ndarray, start: int, frames: int) -> np.ndarray:
    indices = np.minimum(np.arange(start, start + frames), len(value) - 1)
    return value[indices]


def serve(
    model: Path,
    normalization: Path,
    motion: Path,
    state_endpoint: str,
    pose_endpoint: str,
    log_path: Path,
    max_seconds: float | None = None,
) -> None:
    config = HeraclesConfig()
    runtime = PlannerRuntime(OnnxVelocityModel(model), normalization, config)
    reference = np.load(motion)
    reference_joint = reference["joint_pos"]
    reference_root = reference["root_quat_xyzw"]
    subscriber = SimStateSubscriber(state_endpoint)
    publisher = PosePublisher(pose_endpoint)
    window_frames = round(0.9 * config.data_hz) + 1
    prefix_frames = round(config.horizon_s * config.data_hz) + 1
    started = time.monotonic()
    next_plan = started
    plan_index = 0
    try:
        while max_seconds is None or time.monotonic() - started < max_seconds:
            fields = subscriber.receive()
            now = time.monotonic()
            if now < next_plan:
                continue
            source_frame = min(
                int(np.asarray(fields["frame_index"]).reshape(-1)[-1]), len(reference_joint) - 1
            )
            original_joint = _window(reference_joint, source_frame, window_frames)
            original_root = _window(reference_root, source_frame, window_frames)
            measured_joint = np.asarray(fields["joint_pos"], dtype=np.float32).reshape(-1, 29)[-1]
            root_wxyz = np.asarray(fields["body_quat_w"], dtype=np.float32).reshape(-1, 4)[-1]
            root_xyzw = root_wxyz[[1, 2, 3, 0]]
            keyframes, residual, warm = runtime.plan_keyframes(
                measured_joint,
                root_xyzw,
                original_joint[:prefix_frames],
                original_root[:prefix_frames],
            )
            generated_joint, generated_root = runtime.interpolate_prefix(keyframes)
            joint_window, root_window, velocity_window = runtime.build_sonic_window(
                generated_joint, generated_root, original_joint, original_root
            )
            frame_index = np.arange(source_frame, source_frame + window_frames, dtype=np.int64)
            publisher.publish(joint_window, velocity_window, root_window, frame_index)
            append_planner_log(
                log_path,
                {
                    "plan_index": plan_index,
                    "monotonic_s": now - started,
                    "source_frame": source_frame,
                    "measured_joint_pos": measured_joint.tolist(),
                    "measured_root_quat_xyzw": root_xyzw.tolist(),
                    "generated_keyframes": keyframes.tolist(),
                    "generated_residual": residual.tolist(),
                    "directional_warm_start": warm.tolist(),
                    "published_joint_pos": joint_window.tolist(),
                    "published_joint_vel": velocity_window.tolist(),
                    "published_root_quat_xyzw": root_window.tolist(),
                },
            )
            plan_index += 1
            next_plan += 1.0 / config.replan_hz
            if next_plan < now:
                next_plan = now
    finally:
        subscriber.close()
        publisher.close()
