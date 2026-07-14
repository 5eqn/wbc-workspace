#!/usr/bin/env python3
"""Generate the static BFM-Zero fallen-recovery latent inspector dataset."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

import bfm_zero_fallen_recovery_eval as recovery


REPO_ROOT = Path(__file__).resolve().parents[1]
FPS = 25
FALLEN_Z_M = 0.45
RECOVERY_Z_M = 0.75
RECOVERY_DWELL_S = 0.5
LATENT_DIM = 256
COLORS = {
    "prone": "#4C78A8",
    "supine": "#E45756",
    "lateral": "#59A14F",
    "other": "#8B93A1",
}
RUN_SPECS = (
    {
        "id": "newmodel-fullstage1-bfm-zero-20260714a",
        "label": "Random initial states · 100",
        "summary": "logs/bfm-zero-fallen-recovery/newmodel-fullstage1-bfm-zero-20260714a/stage2/summary.json",
        "video": "artifacts/bfm-zero-fallen-recovery/newmodel-fullstage1-bfm-zero-20260714a/stage3/tiled_runs.mp4",
        "frames": 101,
    },
    {
        "id": "newmodel-induced-bfm-zero-20260714b-velocity-default",
        "label": "Induced fall · 100",
        "summary": "logs/bfm-zero-fallen-recovery/newmodel-induced-bfm-zero-20260714b-velocity-default/stage2/summary.json",
        "video": "artifacts/bfm-zero-fallen-recovery/newmodel-induced-bfm-zero-20260714b-velocity-default/stage3/tiled_runs.mp4",
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


def validate_and_load_runs(spec: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any], Any, Any]]]:
    summary_path = REPO_ROOT / spec["summary"]
    summary = load_json(summary_path)
    paths = [REPO_ROOT / str(path) for path in summary.get("run_summaries", [])]
    if len(paths) != 100:
        raise ValueError(f"{summary_path}: expected 100 run summaries, got {len(paths)}")
    loaded = []
    for index, path in enumerate(paths):
        expected_id = f"run_{index:03d}"
        if path.parent.name != expected_id:
            raise ValueError(f"{summary_path}: entry {index} is {path.parent.name}, expected {expected_id}")
        run_summary = load_json(path)
        trajectory_path = REPO_ROOT / str(run_summary["trajectory_path"])
        trajectory = np.load(trajectory_path)
        required = {"time_s", "qpos", "qvel", "root_z"}
        if not required.issubset(trajectory.files):
            raise ValueError(f"{trajectory_path}: missing {sorted(required - set(trajectory.files))}")
        telemetry_path = REPO_ROOT / str(run_summary["telemetry_path"])
        telemetry = np.load(telemetry_path)
        loaded.append((expected_id, run_summary, trajectory, telemetry))
    return summary, loaded


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
    gamma: float = 0.98,
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
    future = np.zeros_like(b)
    for step in range(len(b) - 2, -1, -1):
        future[step] = b[step + 1] + gamma * future[step + 1]
    recurrence_error = np.max(np.abs(future[:-1] - b[1:] - gamma * future[1:])) if len(b) > 1 else 0.0
    if recurrence_error > 1e-5:
        raise ValueError(f"future recurrence error {recurrence_error}")
    l2 = np.linalg.norm(f - future[:, None, :], axis=-1)
    dot = np.sum(f * goal_z_np[None, None, :], axis=-1)
    frame_times = np.arange(frame_count, dtype=np.float64) / FPS
    control_time = sim_time - trajectory_sim_time[0]
    frame_indices = np.clip(np.searchsorted(control_time, frame_times, side="left"), 0, len(control_time) - 1)
    outputs = {
        "forward_future_l2": l2[frame_indices],
        "forward_dot_z": dot[frame_indices],
        "discriminator_probability": d[frame_indices],
        "discriminator_q": qd[frame_indices],
    }
    if outputs["forward_future_l2"].shape != (frame_count, 2) or outputs["discriminator_q"].shape != (frame_count, 2):
        raise ValueError("F/QD frame shape validation failed")
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise ValueError("non-finite model diagnostic")
    if np.any(outputs["forward_future_l2"] < 0) or np.any((outputs["discriminator_probability"] < 0) | (outputs["discriminator_probability"] > 1)):
        raise ValueError("L2 or discriminator range validation failed")
    return outputs


def anomaly_metrics(latent: np.ndarray, fallen_frame: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dot = np.sum(latent[1:] * latent[:-1], axis=1, dtype=np.float64) / float(LATENT_DIM)
    angles = np.zeros(len(latent), dtype=np.float64)
    angles[1:] = np.arccos(np.clip(dot, -1.0, 1.0))
    speed = angles * FPS
    cumulative = np.zeros(len(latent), dtype=np.float64)
    tortuosity = np.ones(len(latent), dtype=np.float64)
    if fallen_frame is not None:
        cumulative[fallen_frame:] = np.cumsum(angles[fallen_frame:])
        origin = latent[fallen_frame].astype(np.float64)
        direct_dot = latent[fallen_frame:].astype(np.float64) @ origin / float(LATENT_DIM)
        direct = np.arccos(np.clip(direct_dot, -1.0, 1.0))
        tortuosity[fallen_frame:] = cumulative[fallen_frame:] / np.maximum(direct, 1e-4)
        tortuosity[fallen_frame] = 1.0
    for name, values in (("angular speed", speed), ("cumulative path", cumulative), ("tortuosity", tortuosity)):
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"non-finite or negative {name}")
    return speed, cumulative, tortuosity


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
    norms = np.linalg.norm(scores, axis=1)
    invalid = np.flatnonzero(~np.isfinite(scores).all(axis=1) | ~np.isfinite(norms) | (norms < 1e-8))
    if len(invalid):
        raise ValueError(f"invalid 3D PCA score for {record['run_id']} frame {int(invalid[0])}")
    xyz = scores / norms[:, None]
    if float(np.max(np.abs(np.linalg.norm(xyz, axis=1) - 1.0))) >= 1e-5:
        raise ValueError(f"unit sphere projection failed for {record['run_id']}")
    return xyz


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
    first_summary = load_json(REPO_ROOT / RUN_SPECS[0]["summary"])
    model_metadata = dict(first_summary["model"])
    checkpoint_path = Path(model_metadata["checkpoint_path"])
    goal_context_path = Path(model_metadata["goal_context_path"])
    goal_context = rt.joblib.load(goal_context_path)
    goal_z = np.asarray(goal_context[recovery.DEFAULT_GOAL_KEY], dtype=np.float32).reshape(-1)
    if goal_z.shape != (LATENT_DIM,) or abs(float(np.linalg.norm(goal_z)) - 16.0) > 1e-4:
        raise ValueError(f"invalid pinned goal z: shape={goal_z.shape}, norm={np.linalg.norm(goal_z)}")
    latent_model = rt.load_model_from_checkpoint_dir(str(checkpoint_path), device=str(device))
    latent_model.eval()

    records: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        probe_video(REPO_ROOT / spec["video"], int(spec["frames"]))
        summary, loaded = validate_and_load_runs(spec)
        if summary["model"]["checkpoint_sha256"] != model_metadata["checkpoint_sha256"] or summary["model"]["actor_onnx_sha256"] != model_metadata["actor_onnx_sha256"]:
            raise ValueError(f"{spec['id']} uses different model artifacts")
        group_records = []
        print(f"Extracting {spec['id']} on {device}...", flush=True)
        for index, (run_id, _summary, trajectory, telemetry) in enumerate(loaded):
            latent, root_z = extract_run_latents(
                rt, latent_model, device, model, data, joint_qpos_ids, joint_qvel_ids,
                body_ids, extend_parent_body_id, trajectory, int(spec["frames"]),
            )
            fallen = np.flatnonzero(root_z < FALLEN_Z_M)
            fallen_frame = int(fallen[0]) if len(fallen) else None
            raw_time_s = np.asarray(trajectory["time_s"], dtype=np.float32)
            raw_root_z = np.asarray(trajectory["root_z"], dtype=np.float32)
            raw_qpos = np.asarray(trajectory["qpos"], dtype=np.float32)
            first_second = np.flatnonzero(
                (raw_time_s <= raw_time_s[0] + 1.0 + 1e-8) & np.isfinite(raw_root_z)
            )
            lowest_index = int(first_second[np.argmin(raw_root_z[first_second])]) if len(first_second) else None
            orientation = orientation_from_qpos(raw_qpos[lowest_index]) if lowest_index is not None else "other"
            speed, cumulative, tortuosity = anomaly_metrics(latent, fallen_frame)
            diagnostics = extract_model_diagnostics(
                rt, latent_model, device, model, data, joint_qpos_ids, joint_qvel_ids,
                body_ids, extend_parent_body_id, trajectory, telemetry, goal_z, int(spec["frames"]),
            )
            record = {
                "index": index, "run_id": run_id, "orientation": orientation,
                "fallen_frame": fallen_frame, "latent": latent, "root_z": root_z,
                "angular_speed": speed, "cumulative": cumulative, "tortuosity": tortuosity,
                **diagnostics,
            }
            records.append(record)
            group_records.append(record)
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/100", flush=True)
        groups.append({"spec": spec, "records": group_records})

    mean, components = fit_projection(records)
    output_runs = []
    for group in groups:
        spec = group["spec"]
        trials = []
        counts = {name: 0 for name in COLORS}
        for record in group["records"]:
            xyz = project_record(record, mean, components)
            counts[record["orientation"]] += 1
            trials.append(
                {
                    "index": record["index"],
                    "run_id": record["run_id"],
                    "orientation": record["orientation"],
                    "color": COLORS[record["orientation"]],
                    "fallen_frame": record["fallen_frame"],
                    "xyz": xyz.round(7).tolist(),
                    "angular_speed_rad_s": record["angular_speed"].round(7).tolist(),
                    "cumulative_path_rad": record["cumulative"].round(7).tolist(),
                    "tortuosity": record["tortuosity"].round(7).tolist(),
                    "forward_future_l2": record["forward_future_l2"].round(7).tolist(),
                    "forward_dot_z": record["forward_dot_z"].round(7).tolist(),
                    "discriminator_probability": record["discriminator_probability"].round(7).tolist(),
                    "discriminator_q": record["discriminator_q"].round(7).tolist(),
                }
            )
        output_runs.append(
            {
                "id": spec["id"], "label": spec["label"],
                "video_url": f"/{spec['video']}", "fps": FPS,
                "duration_s": (int(spec["frames"]) - 1) / FPS,
                "frames": int(spec["frames"]), "orientation_counts": counts,
                "no_fall_count": sum(trial["fallen_frame"] is None for trial in trials),
                "trials": trials,
            }
        )
    payload = {
        "model_diagnostics": {
            **model_metadata,
            "gamma": 0.98,
            "model_frequency_hz": 50,
            "display_frequency_hz": FPS,
            "future_target": "G_t = B_(t+1) + 0.98 G_(t+1), with G_last = 0 and finite-rollout truncation",
            "ensemble": "F and QD heads are parallel estimators from one training run, not separate seeds",
        },
        "projection": {
            "method": "shared centered PCA via NumPy SVD, then per-point unit normalization",
            "latent_dim": LATENT_DIM, "output_dim": 3,
            "fit_rule": "first root_z < 0.45 m through first 0.5 s sustained root_z > 0.75 m window end; otherwise trajectory end",
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
