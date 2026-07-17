#!/usr/bin/env python3
"""Generate the static BFM-Zero fallen-recovery latent inspector dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

import bfm_zero_fallen_recovery_eval as recovery
from bfm_zero_trial_sources import aligned_fast_actions, load_fast_source, load_stage2_source


REPO_ROOT = Path(__file__).resolve().parents[1]
FPS = 25
FALLEN_Z_M = 0.45
RECOVERY_Z_M = 0.75
RECOVERY_DWELL_S = 0.5
LATENT_DIM = 256
DISCOUNT = 0.98
DISCRIMINATOR_EPS = 1e-7
COLORS = {
    "prone": "#4C78A8",
    "supine": "#E45756",
    "lateral": "#59A14F",
    "other": "#8B93A1",
}
RUN_SPECS = (
    {
        "id": "oldmodel-fullstage1-bfm-zero-20260716a",
        "label": "Old model · random initial states · 100",
        "stage3_summary": "artifacts/bfm-zero-fallen-recovery/oldmodel-fullstage1-bfm-zero-20260716a/stage3/summary.json",
        "video": "artifacts/bfm-zero-fallen-recovery/oldmodel-fullstage1-bfm-zero-20260716a/stage3/tiled_runs.mp4",
        "frames": 101,
    },
    {
        "id": "oldmodel-induced-bfm-zero-20260716b-velocity-default",
        "label": "Old model · induced fall · 100",
        "stage3_summary": "artifacts/bfm-zero-fallen-recovery/oldmodel-induced-bfm-zero-20260716b-velocity-default/stage3/summary.json",
        "video": "artifacts/bfm-zero-fallen-recovery/oldmodel-induced-bfm-zero-20260716b-velocity-default/stage3/tiled_runs.mp4",
        "frames": 201,
    },
    {
        "id": "lafan1-60m-step192m-128x1-20260717",
        "label": "LAFAN1 60M · 192M transitions · induced fall · 100",
        "stage3_summary": "artifacts/bfm-zero-fast-induced/lafan1-60m-step192m-128x1-20260717/stage3/summary.json",
        "video": "artifacts/bfm-zero-fast-induced/lafan1-60m-step192m-128x1-20260717/stage3/tiled_runs.mp4",
        "frames": 201,
    },
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def probe_video(path: Path, expected_frames: int) -> None:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ]
    probe = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    if (int(probe["width"]), int(probe["height"])) != (1920, 1080):
        raise ValueError(f"{path}: expected 1920x1080, got {probe['width']}x{probe['height']}")
    numerator, denominator = (int(value) for value in probe["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    if abs(fps - FPS) > 1e-9 or int(probe["nb_frames"]) != expected_frames:
        raise ValueError(
            f"{path}: expected {expected_frames} frames at {FPS} FPS, "
            f"got {probe['nb_frames']} frames at {fps:g} FPS"
        )
    expected_duration = expected_frames / FPS
    if abs(float(probe["duration"]) - expected_duration) > 1e-6:
        raise ValueError(
            f"{path}: expected encoded duration {expected_duration:g} s, got {probe['duration']} s"
        )


def load_stage3_source(spec: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    summary_path = REPO_ROOT / spec["stage3_summary"]
    summary = load_json(summary_path)
    if summary.get("source_kind") == "fast_replay_v2":
        source = load_fast_source(REPO_ROOT / summary["source_path"], REPO_ROOT)
    else:
        stage2_summary = REPO_ROOT / summary.get("stage2_summary", "")
        source = load_stage2_source(stage2_summary.parent, REPO_ROOT)
    selected = [trial.attempt_id for trial in source.trials]
    if summary.get("selected_attempt_ids", selected) != selected:
        raise ValueError(f"{summary_path}: selected attempts differ from the validated source")
    if len(source.trials) != 100:
        raise ValueError(f"{summary_path}: expected exactly 100 selected trials")
    return summary, source


def orientation_from_qpos(qpos: np.ndarray) -> str:
    quat = np.asarray(qpos[3:7], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(quat).all() or norm < 1e-8:
        return "other"
    w, x, y, z = quat / norm
    up_world = np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]
    )
    if not np.isfinite(up_world).all():
        return "other"
    axis = int(np.argmax(np.abs(up_world)))
    if axis == 0:
        return "prone" if up_world[0] > 0.0 else "supine"
    if axis == 1:
        return "lateral"
    return "other"


def recovery_end(root_z: np.ndarray, start: int) -> int:
    window = int(math.ceil(RECOVERY_DWELL_S * FPS))
    above = root_z > RECOVERY_Z_M
    run_length = 0
    for index in range(start, len(root_z)):
        run_length = run_length + 1 if above[index] else 0
        if run_length >= window:
            return index
    return len(root_z) - 1


def move_observation(observation: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device) for key, value in observation.items()}


def discounted_ground_truth(signal: np.ndarray, *, next_step_reward: bool) -> np.ndarray:
    """Return an infinite discounted sum, holding the final signal forever."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError(f"invalid ground-truth signal: {signal.shape}")
    result = np.empty_like(signal)
    result[-1] = signal[-1] / (1.0 - DISCOUNT)
    for step in range(len(signal) - 2, -1, -1):
        reward_index = step + 1 if next_step_reward else step
        result[step] = signal[reward_index] + DISCOUNT * result[step + 1]
    expected = signal[1:] if next_step_reward else signal[:-1]
    if not np.allclose(result[:-1], expected + DISCOUNT * result[1:], atol=1e-10, rtol=1e-12):
        raise ValueError("discounted ground-truth recurrence validation failed")
    if not np.isclose(result[-1], signal[-1] / (1.0 - DISCOUNT), atol=1e-10, rtol=1e-12):
        raise ValueError("discounted ground-truth tail validation failed")
    return result


def extract_run_latents(
    rt: Any,
    latent_model: Any,
    device: torch.device,
    model: Any,
    data: Any,
    joint_qpos_ids: np.ndarray,
    joint_qvel_ids: np.ndarray,
    body_ids: list[int],
    extend_parent_body_id: int,
    trajectory: Any,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    render_times = np.arange(frame_count, dtype=np.float32) / FPS
    time_s = np.asarray(trajectory["time_s"], dtype=np.float32)
    qpos = np.asarray(trajectory["qpos"], dtype=np.float32)
    qvel = np.asarray(trajectory["qvel"], dtype=np.float32)
    root_z_source = np.asarray(trajectory["root_z"], dtype=np.float32)
    if qpos.shape != (len(time_s), 36) or qvel.shape != (len(time_s), 35):
        raise ValueError(f"unexpected trajectory shapes: time={time_s.shape}, qpos={qpos.shape}, qvel={qvel.shape}")
    sample_indices = np.clip(np.searchsorted(time_s, render_times, side="left"), 0, len(time_s) - 1)
    observations: dict[str, list[torch.Tensor]] = {"state": [], "privileged_state": []}
    for sample_index in sample_indices:
        data.qpos[:] = qpos[int(sample_index)]
        data.qvel[:] = qvel[int(sample_index)]
        rt.mujoco.mj_forward(model, data)
        body_state = recovery.extract_body_state(rt, model, data, body_ids, extend_parent_body_id)
        observation = recovery.build_backward_obs(
            rt,
            *body_state,
            data.qpos[joint_qpos_ids].astype(np.float32),
            data.qvel[joint_qvel_ids].astype(np.float32),
        )
        for key in observations:
            observations[key].append(observation[key])
    batch = move_observation({key: torch.cat(value, dim=0) for key, value in observations.items()}, device)
    with torch.inference_mode():
        latent = latent_model.project_z(latent_model.backward_map(batch))
    latent_np = latent.detach().cpu().numpy().astype(np.float32)
    if latent_np.shape != (frame_count, LATENT_DIM) or not np.isfinite(latent_np).all():
        raise ValueError(f"invalid latent batch shape/values: {latent_np.shape}")
    return latent_np, root_z_source[sample_indices]


def extract_model_diagnostics(
    rt: Any,
    latent_model: Any,
    device: torch.device,
    model: Any,
    data: Any,
    joint_qpos_ids: np.ndarray,
    joint_qvel_ids: np.ndarray,
    body_ids: list[int],
    extend_parent_body_id: int,
    trajectory: Any,
    telemetry: Any,
    goal_z_np: np.ndarray,
    frame_count: int,
) -> dict[str, np.ndarray]:
    required = {
        "simulator_tick", "sim_time_s", "state", "last_action", "history_actor",
        "goal_z", "onnx_input", "raw_actor_output", "clipped_action",
    }
    if not required.issubset(telemetry.files):
        raise ValueError(f"telemetry missing {sorted(required - set(telemetry.files))}")
    sim_time = np.asarray(telemetry["sim_time_s"], dtype=np.float64)
    ticks = np.asarray(telemetry["simulator_tick"], dtype=np.int64)
    trajectory_sim_time = np.asarray(trajectory["sim_time_s"], dtype=np.float64)
    in_rollout = (sim_time >= trajectory_sim_time[0] - 1e-9) & (sim_time <= trajectory_sim_time[-1] + 1e-9)
    indices = np.flatnonzero(in_rollout)
    if not len(indices):
        raise ValueError("telemetry does not overlap trajectory sim_time_s")
    sim_time = sim_time[indices]
    ticks = ticks[indices]
    if np.any(np.diff(ticks) < 0):
        raise ValueError("telemetry simulator ticks are not monotonic")
    if sim_time[0] > trajectory_sim_time[0] + 0.03 or sim_time[-1] < trajectory_sim_time[-1] - 0.03:
        raise ValueError("telemetry does not cover the complete trajectory")

    replay_input = np.asarray(telemetry["onnx_input"], dtype=np.float32)[indices]
    if replay_input.shape[1:] != (721,):
        raise ValueError(f"expected ONNX input [N,721], got {replay_input.shape}")
    if not np.allclose(replay_input[:, 465:], goal_z_np[None], atol=1e-6, rtol=0.0):
        raise ValueError("rollout goal z differs from pinned goal context")

    qpos = np.asarray(trajectory["qpos"], dtype=np.float32)
    qvel = np.asarray(trajectory["qvel"], dtype=np.float32)
    trajectory_indices = np.clip(np.searchsorted(trajectory_sim_time, sim_time, side="left"), 0, len(qpos) - 1)
    privileged = []
    for sample_index in trajectory_indices:
        data.qpos[:] = qpos[int(sample_index)]
        data.qvel[:] = qvel[int(sample_index)]
        rt.mujoco.mj_forward(model, data)
        backward_obs = recovery.build_backward_obs(
            rt, *recovery.extract_body_state(rt, model, data, body_ids, extend_parent_body_id),
            data.qpos[joint_qpos_ids].astype(np.float32),
            data.qvel[joint_qvel_ids].astype(np.float32),
        )
        privileged.append(backward_obs["privileged_state"])
    observation = {
        "state": torch.from_numpy(np.asarray(telemetry["state"], dtype=np.float32)[indices]),
        "privileged_state": torch.cat(privileged, dim=0),
        "last_action": torch.from_numpy(np.asarray(telemetry["last_action"], dtype=np.float32)[indices]),
        "history_actor": torch.from_numpy(np.asarray(telemetry["history_actor"], dtype=np.float32)[indices]),
    }
    observation = move_observation(observation, device)
    action = torch.from_numpy(np.asarray(telemetry["clipped_action"], dtype=np.float32)[indices]).to(device)
    goal_z = torch.from_numpy(np.broadcast_to(goal_z_np, (len(indices), LATENT_DIM)).copy()).to(device)
    with torch.inference_mode():
        backward = latent_model.backward_map(observation)
        forward = latent_model.forward_map(observation, goal_z, action)
        discriminator = latent_model.discriminator(observation, goal_z)
        critic = latent_model.critic(observation, goal_z, action)
    b = backward.detach().cpu().numpy().astype(np.float64)
    f = np.moveaxis(forward.detach().cpu().numpy().astype(np.float64), 0, 1)
    d = discriminator.detach().cpu().numpy().reshape(-1).astype(np.float64)
    qd = np.moveaxis(critic.detach().cpu().numpy().astype(np.float64), 0, 1).reshape(len(indices), 2)
    backward_dot = np.sum(b * goal_z_np[None, :], axis=-1)
    forward_dot = np.sum(f * goal_z_np[None, None, :], axis=-1)
    discriminator_reward = np.log(np.clip(d, DISCRIMINATOR_EPS, 1.0 - DISCRIMINATOR_EPS)) - np.log1p(
        -np.clip(d, DISCRIMINATOR_EPS, 1.0 - DISCRIMINATOR_EPS)
    )
    forward_ground_truth = discounted_ground_truth(backward_dot, next_step_reward=True)
    discriminator_ground_truth = discounted_ground_truth(discriminator_reward, next_step_reward=False)
    frame_times = np.arange(frame_count, dtype=np.float64) / FPS
    control_time = sim_time - trajectory_sim_time[0]
    frame_indices = np.clip(np.searchsorted(control_time, frame_times, side="left"), 0, len(control_time) - 1)
    outputs = {
        "backward_dot_z": backward_dot[frame_indices],
        "forward_dot_z": forward_dot[frame_indices],
        "forward_dot_z_ground_truth": forward_ground_truth[frame_indices],
        "discriminator_probability": d[frame_indices],
        "discriminator_q": qd[frame_indices],
        "discriminator_q_ground_truth": discriminator_ground_truth[frame_indices],
    }
    if outputs["backward_dot_z"].shape != (frame_count,) or outputs["discriminator_probability"].shape != (frame_count,):
        raise ValueError("B/D frame shape validation failed")
    if outputs["forward_dot_z"].shape != (frame_count, 2) or outputs["discriminator_q"].shape != (frame_count, 2):
        raise ValueError("F/QD frame shape validation failed")
    if outputs["forward_dot_z_ground_truth"].shape != (frame_count,) or outputs["discriminator_q_ground_truth"].shape != (frame_count,):
        raise ValueError("F/QD ground-truth frame shape validation failed")
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise ValueError("non-finite model diagnostic")
    if np.any((outputs["discriminator_probability"] < 0) | (outputs["discriminator_probability"] > 1)):
        raise ValueError("discriminator range validation failed")
    return outputs


def extract_fast_model_data(
    latent_model: Any,
    device: torch.device,
    trial: Any,
    goal_z_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    observation = move_observation(
        {key: torch.from_numpy(np.asarray(value, dtype=np.float32)) for key, value in trial.observations.items()},
        device,
    )
    action_np = aligned_fast_actions(trial)
    action = torch.from_numpy(action_np).to(device)
    goal_z = torch.from_numpy(np.broadcast_to(goal_z_np, (401, LATENT_DIM)).copy()).to(device)
    with torch.inference_mode():
        backward = latent_model.backward_map(observation)
        latent = latent_model.project_z(backward)
        forward = latent_model.forward_map(observation, goal_z, action)
        discriminator = latent_model.discriminator(observation, goal_z)
        critic = latent_model.critic(observation, goal_z, action)
    latent_np = latent.detach().cpu().numpy().astype(np.float32)
    b = backward.detach().cpu().numpy().astype(np.float64)
    f = np.moveaxis(forward.detach().cpu().numpy().astype(np.float64), 0, 1)
    d = discriminator.detach().cpu().numpy().reshape(-1).astype(np.float64)
    qd = np.moveaxis(critic.detach().cpu().numpy().astype(np.float64), 0, 1).reshape(401, 2)
    if latent_np.shape != (401, LATENT_DIM):
        raise ValueError(f"{trial.run_id}: invalid fast latent shape {latent_np.shape}")
    backward_dot = np.sum(b * goal_z_np[None, :], axis=-1)
    forward_dot = np.sum(f * goal_z_np[None, None, :], axis=-1)
    clipped_d = np.clip(d, DISCRIMINATOR_EPS, 1.0 - DISCRIMINATOR_EPS)
    discriminator_reward = np.log(clipped_d) - np.log1p(-clipped_d)
    display = np.arange(0, 401, 2)
    outputs = {
        "backward_dot_z": backward_dot[display],
        "forward_dot_z": forward_dot[display],
        "forward_dot_z_ground_truth": discounted_ground_truth(
            backward_dot, next_step_reward=True
        )[display],
        "discriminator_probability": d[display],
        "discriminator_q": qd[display],
        "discriminator_q_ground_truth": discounted_ground_truth(
            discriminator_reward, next_step_reward=False
        )[display],
    }
    if latent_np.shape[0] != 401 or action_np.shape[0] != 401:
        raise ValueError(f"{trial.run_id}: fast 401-state/action alignment failed")
    if not np.isfinite(latent_np).all() or not all(np.isfinite(value).all() for value in outputs.values()):
        raise ValueError(f"{trial.run_id}: non-finite fast model output")
    return latent_np[display], np.asarray(trial.qpos, dtype=np.float32)[display, 2], outputs


def fit_projection(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    fit_rows = []
    for record in records:
        fallen_frame = record["fallen_frame"]
        if fallen_frame is not None:
            end = recovery_end(record["root_z"], fallen_frame)
            fit_rows.append(record["latent"][fallen_frame : end + 1])
    if not fit_rows:
        raise ValueError("no fallen intervals available for PCA")
    fit = np.concatenate(fit_rows, axis=0).astype(np.float64)
    mean = fit.mean(axis=0)
    _, _, components = np.linalg.svd(fit - mean, full_matrices=False)
    components = components[:3]
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            component *= -1.0
    return mean, components


def project_record(record: dict[str, Any], mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    scores = (record["latent"].astype(np.float64) - mean) @ components.T
    invalid = np.flatnonzero(~np.isfinite(scores).all(axis=1) | (np.abs(scores) > 16.0).any(axis=1))
    if len(invalid):
        raise ValueError(f"non-finite or out-of-cube PCA score for {record['run_id']} frame {int(invalid[0])}")
    return scores


def generate(device_arg: str, output: Path) -> None:
    rt = recovery.runtime()
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    scene = recovery.BFM_ZERO_DEPLOY_ROOT / "data/robots/g1/scene_29dof_freebase.xml"
    model = rt.mujoco.MjModel.from_xml_path(str(scene))
    data = rt.mujoco.MjData(model)
    policy = recovery.load_yaml(recovery.BFM_ZERO_DEPLOY_ROOT / "config/policy/motivo_newG1.yaml")
    joint_names = [str(name) for name in policy["isaac_joint_names"]]
    joint_qpos_ids, joint_qvel_ids, _ = recovery.build_joint_mappings(rt, model, joint_names)
    body_ids, _, extend_parent_body_id = recovery.build_body_ids(rt, model)
    groups: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        probe_video(REPO_ROOT / spec["video"], int(spec["frames"]))
        stage3_summary, source = load_stage3_source(spec)
        model_metadata = dict(source.model)
        checkpoint_path = Path(model_metadata["checkpoint_path"])
        if source.goal_z is None:
            goal_context = rt.joblib.load(Path(model_metadata["goal_context_path"]))
            goal_z = np.asarray(goal_context[source.goal_key], dtype=np.float32).reshape(-1)
        else:
            goal_z = np.asarray(source.goal_z, dtype=np.float32).reshape(-1)
        if goal_z.shape != (LATENT_DIM,) or abs(float(np.linalg.norm(goal_z)) - 16.0) > 1e-4:
            raise ValueError(f"{spec['id']}: invalid goal z: shape={goal_z.shape}, norm={np.linalg.norm(goal_z)}")
        goal_hash = hashlib.sha256(goal_z.tobytes()).hexdigest()
        if stage3_summary.get("goal_z_sha256", goal_hash) != goal_hash:
            raise ValueError(f"{spec['id']}: Stage 3 goal latent hash mismatch")
        latent_model = rt.load_model_from_checkpoint_dir(str(checkpoint_path), device=str(device))
        latent_model.eval()
        group_records = []
        print(f"Extracting {spec['id']} on {device}...", flush=True)
        for index, trial in enumerate(source.trials):
            trajectory = {
                "time_s": trial.time_s,
                "sim_time_s": trial.metadata.get("sim_time_s", trial.time_s),
                "qpos": trial.qpos,
                "qvel": trial.qvel,
                "root_z": np.asarray(trial.qpos)[:, 2],
            }
            if source.source_kind == "fast_replay_v2":
                latent, root_z, diagnostics = extract_fast_model_data(
                    latent_model, device, trial, goal_z
                )
            else:
                latent, root_z = extract_run_latents(
                    rt, latent_model, device, model, data, joint_qpos_ids, joint_qvel_ids,
                    body_ids, extend_parent_body_id, trajectory, int(spec["frames"]),
                )
                with np.load(Path(trial.metadata["telemetry_path"])) as telemetry:
                    diagnostics = extract_model_diagnostics(
                        rt, latent_model, device, model, data, joint_qpos_ids, joint_qvel_ids,
                        body_ids, extend_parent_body_id, trajectory, telemetry, goal_z, int(spec["frames"]),
                    )
            fallen = np.flatnonzero(root_z < FALLEN_Z_M)
            fallen_frame = int(fallen[0]) if len(fallen) else None
            raw_time_s = np.asarray(trial.time_s, dtype=np.float32)
            raw_root_z = np.asarray(trial.qpos, dtype=np.float32)[:, 2]
            raw_qpos = np.asarray(trial.qpos, dtype=np.float32)
            first_second = np.flatnonzero(
                (raw_time_s <= raw_time_s[0] + 1.0 + 1e-8) & np.isfinite(raw_root_z)
            )
            lowest_index = int(first_second[np.argmin(raw_root_z[first_second])]) if len(first_second) else None
            orientation = orientation_from_qpos(raw_qpos[lowest_index]) if lowest_index is not None else "other"
            record = {
                "index": index, "run_id": trial.run_id, "attempt_id": trial.attempt_id,
                "orientation": orientation,
                "fallen_frame": fallen_frame, "latent": latent, "root_z": root_z,
                **diagnostics,
            }
            group_records.append(record)
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/100", flush=True)
        mean, components = fit_projection(group_records)
        groups.append(
            {
                "spec": spec,
                "records": group_records,
                "mean": mean,
                "components": components,
                "source": source,
                "model": model_metadata,
                "goal_hash": goal_hash,
            }
        )

    output_runs = []
    for group in groups:
        spec = group["spec"]
        trials = []
        counts = {name: 0 for name in COLORS}
        for record in group["records"]:
            xyz = project_record(record, group["mean"], group["components"])
            counts[record["orientation"]] += 1
            trials.append(
                {
                    "index": record["index"],
                    "attempt_id": record["attempt_id"],
                    "run_id": record["run_id"],
                    "orientation": record["orientation"],
                    "color": COLORS[record["orientation"]],
                    "fallen_frame": record["fallen_frame"],
                    "xyz": xyz.round(7).tolist(),
                    "backward_dot_z": record["backward_dot_z"].round(7).tolist(),
                    "forward_dot_z": record["forward_dot_z"].round(7).tolist(),
                    "forward_dot_z_ground_truth": record["forward_dot_z_ground_truth"].round(7).tolist(),
                    "discriminator_probability": record["discriminator_probability"].round(7).tolist(),
                    "discriminator_q": record["discriminator_q"].round(7).tolist(),
                    "discriminator_q_ground_truth": record["discriminator_q_ground_truth"].round(7).tolist(),
                }
            )
        output_runs.append(
            {
                "id": spec["id"], "label": spec["label"],
                "video_url": f"/{spec['video']}", "fps": FPS,
                "duration_s": (int(spec["frames"]) - 1) / FPS,
                "frames": int(spec["frames"]), "orientation_counts": counts,
                "no_fall_count": sum(trial["fallen_frame"] is None for trial in trials),
                "source_kind": group["source"].source_kind,
                "model": group["model"],
                "goal_key": group["source"].goal_key,
                "goal_z_sha256": group["goal_hash"],
                "projection": {
                    "method": "dataset-local centered PCA via NumPy SVD; raw component scores",
                    "latent_dim": LATENT_DIM,
                    "output_dim": 3,
                    "axes": ["X (PC1)", "Y (PC2)", "Z (PC3)"],
                    "cube_bounds": [-16, 16],
                    "ticks": [-16, -8, 0, 8, 16],
                    "mean": group["mean"].round(9).tolist(),
                    "components": group["components"].round(9).tolist(),
                    "fit_rule": "first root_z < 0.45 m through recovery dwell end; otherwise trajectory end",
                },
                "trials": trials,
            }
        )
    payload = {
        "model_diagnostics": {
            "model_frequency_hz": 50,
            "display_frequency_hz": FPS,
            "discount": DISCOUNT,
            "backward_score": "raw backward_map(observation) dot pinned goal latent z",
            "forward_score": "each forward_map(observation, z, action) head dot pinned goal latent z",
            "forward_ground_truth": "G_F[t] = B(s[t+1]) dot z + 0.98 G_F[t+1]",
            "discriminator_reward": "r_D[t] = log(D[t]) - log(1 - D[t]), with D clipped to [1e-7, 1 - 1e-7]",
            "discriminator_ground_truth": "G_D[t] = r_D[t] + 0.98 G_D[t+1]",
            "tail_rule": "raw B dot z and D probability hold their final observed values forever",
            "ensemble": "F and QD heads are parallel estimators from one training run, not separate seeds",
        },
        "projection": {
            "method": "independent centered PCA per dataset",
            "cube_bounds": [-16, 16],
            "cross_dataset_comparable": False,
        },
        "runs": output_runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "device": str(device), "runs": [
        {"id": run["id"], "frames": run["frames"], "orientation_counts": run["orientation_counts"], "no_fall_count": run["no_fall_count"]}
        for run in output_runs
    ]}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts/bfm-zero-latent-inspector/data.json")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    generate(args.device, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
