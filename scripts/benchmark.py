#!/usr/bin/env python3
"""Root-owned benchmark utilities for the WBC comparison.

This file is intentionally flat under scripts/. It may use run-sonic as design
reference, but it does not import or execute code from thirdparties/run-sonic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import numpy as np


MOTIONS = [
    "dance_chicken_c03_neutral2s",
    "dance_chicken_c04_neutral2s",
    "dance_heart111_c01_neutral2s",
    "dance_phony_c01_neutral2s",
    "dance_phony_c03_neutral2s",
    "dance_phony_c04_neutral2s",
    "forward_lunge_R_001__A359_M_neutral2s",
    "macarena_001__A545_M_neutral2s",
    "squat_001__A359_neutral2s",
    "walking_quip_360_R_002__A428_neutral2s",
]

POLICIES = ["sonic", "holomotion"]
ROOT = Path(__file__).resolve().parents[1]
RELEASE_EVENT_LOGS = ["sequence_events.csv", "holomotion_sequence_events.csv"]
CONTROL_EVENT = "control_state_observed"
RELEASE_REQUEST_EVENT = "release_file_touched"
RELEASE_CONFIRMED_EVENT = "support_release_confirmed"
PLAYBACK_EVENTS = ["sent_key_T", "motion_start_observed", "motion_playing_observed"]
HOLOMOTION_MIN_TAG = "v1.3.0"
SONIC_DEPLOY = ROOT / "thirdparties" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
HOLO_DEPLOY = ROOT / "thirdparties" / "HoloMotion" / "deployment" / "unitree_g1_ros2_29dof"


def read_csv_matrix(path: Path, skip_prefix_cols: int = 0) -> np.ndarray:
    import numpy as np

    rows: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            vals: list[float] = []
            for value in row[skip_prefix_cols:]:
                if value == "":
                    continue
                try:
                    vals.append(float(value))
                except ValueError:
                    vals = []
                    break
            if vals:
                rows.append(vals)
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    width = min(len(row) for row in rows)
    return np.asarray([row[:width] for row in rows], dtype=np.float64)


def load_reference(policy: str, motion: str) -> tuple[np.ndarray, float]:
    import numpy as np

    if policy == "sonic":
        q = read_csv_matrix(ROOT / "assets" / "motions" / "sonic_motions" / motion / "joint_pos.csv")
        return q[:, :29], 50.0
    path = ROOT / "assets" / "motions" / "holomotion_motions" / f"{motion}_holomotion.npz"
    data = np.load(path)
    for key in ("ref_dof_pos", "dof_pos"):
        if key in data:
            q = np.asarray(data[key], dtype=np.float64)
            return q[:, :29], 50.0
    raise KeyError(f"{path}: no HoloMotion joint-position array")


def load_tracked(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    if run_dir.suffix == ".npz":
        data = np.load(run_dir, allow_pickle=False)
        if "q" not in data:
            raise KeyError(f"{run_dir}: missing q array")
        q = np.asarray(data["q"], dtype=np.float64)
        if "timestamps" in data:
            timestamps = np.asarray(data["timestamps"], dtype=np.float64)
            times = timestamps - timestamps[0]
        else:
            times = np.arange(q.shape[0], dtype=np.float64) / 50.0
        return times, q[:, :29]

    lowcmd = run_dir / "lowcmd.csv"
    sonic_q = run_dir / "csv" / "q.csv"
    holo_log = run_dir / "holomotion_policy_log.csv"
    if lowcmd.exists():
        header = lowcmd.read_text().splitlines()[0].split(",")
        q_cols = [i for i, name in enumerate(header) if name.startswith("q_") or name.startswith("measured_q_")]
        t_col = header.index("time_s") if "time_s" in header else 0
        rows = []
        times = []
        with lowcmd.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    times.append(float(row.get("time_s") or row.get(header[t_col]) or len(times) / 50.0))
                    rows.append([float(row[header[i]]) for i in q_cols[:29]])
                except (KeyError, ValueError):
                    continue
        if rows:
            return np.asarray(times), np.asarray(rows, dtype=np.float64)
    if sonic_q.exists():
        q = read_csv_matrix(sonic_q, skip_prefix_cols=1)
        return np.arange(q.shape[0], dtype=np.float64) / 50.0, q[:, :29]
    if holo_log.exists():
        rows = []
        times = []
        with holo_log.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    times.append(float(row.get("phase_time_s") or row.get("sim_time_s") or len(times) / 50.0))
                    rows.append([float(row[f"measured_q_{i}"]) for i in range(29)])
                except (KeyError, ValueError):
                    continue
        if rows:
            return np.asarray(times), np.asarray(rows, dtype=np.float64)
    raise FileNotFoundError(f"{run_dir}: no supported tracked joint log")


def interpolate_ref(ref: np.ndarray, ref_hz: float, query_t: np.ndarray, delay_s: float) -> np.ndarray:
    import numpy as np

    ref_t = np.arange(ref.shape[0], dtype=np.float64) / ref_hz
    phase_t = np.clip(query_t - delay_s, ref_t[0], ref_t[-1])
    cols = [np.interp(phase_t, ref_t, ref[:, j]) for j in range(ref.shape[1])]
    return np.stack(cols, axis=1)


def aligned_rmse(ref: np.ndarray, ref_hz: float, tracked_t: np.ndarray, tracked_q: np.ndarray) -> dict:
    import numpy as np

    n = min(tracked_q.shape[1], ref.shape[1], 29)
    tracked_q = tracked_q[:, :n]
    best = None
    for delay in np.arange(-0.2, 0.2001, 0.01):
        rq = interpolate_ref(ref[:, :n], ref_hz, tracked_t, float(delay))
        err = tracked_q - rq
        rmse = np.sqrt(np.mean(err * err, axis=0))
        mean = float(np.mean(rmse))
        if best is None or mean < best["mean_joint_rmse_rad"]:
            best = {
                "tracking_delay_s": round(float(delay), 4),
                "mean_joint_rmse_rad": mean,
                "joint_rmse_rad": rmse.tolist(),
                "samples": int(tracked_q.shape[0]),
            }
    assert best is not None
    return best


def expected_run_dir(logs: Path, policy: str, motion: str) -> Path:
    run_dir = logs / policy / motion
    if run_dir.exists():
        return run_dir
    flat_log = logs / f"{policy}_{motion}.npz"
    if flat_log.exists():
        return flat_log
    return run_dir


def log_tag(policy: str, motion: str) -> str:
    return f"{policy}_{motion}"


def event_time(row: dict[str, str]) -> float:
    value = row.get("monotonic_s") or row.get("wall_time_s") or row.get("sim_time_s")
    if value in (None, ""):
        raise ValueError("event row has no usable time column")
    return float(value)


def event_sim_time(row: dict[str, str]) -> float:
    value = row.get("sim_time_s") or row.get("phase_time_s") or row.get("time_s")
    if value in (None, ""):
        return event_time(row)
    return float(value)


def find_event_log(run_dir: Path) -> Path:
    if run_dir.is_file():
        event_dir = run_dir.parent / run_dir.stem.replace("_", "/", 1)
        for name in RELEASE_EVENT_LOGS:
            path = event_dir / name
            if path.exists():
                return path
        for name in RELEASE_EVENT_LOGS:
            path = run_dir.with_name(f"{run_dir.stem}_{name}")
            if path.exists():
                return path
    for name in RELEASE_EVENT_LOGS:
        path = run_dir / name
        if path.exists():
            return path
    names = ", ".join(RELEASE_EVENT_LOGS)
    raise FileNotFoundError(f"{run_dir}: missing release-order event log ({names})")


def load_events(run_dir: Path) -> list[dict[str, str]]:
    path = find_event_log(run_dir)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: empty event log")
    for row in rows:
        row["_event_log"] = str(path)
    return rows


def first_event(events: list[dict[str, str]], event: str) -> dict[str, str]:
    for row in events:
        if row.get("event") == event:
            return row
    raise ValueError(f"missing event {event}")


def validate_release_order(run_dir: Path) -> dict:
    events = load_events(run_dir)
    control = first_event(events, CONTROL_EVENT)
    release_request = first_event(events, RELEASE_REQUEST_EVENT)
    release_confirmed = first_event(events, RELEASE_CONFIRMED_EVENT)
    playback = None
    for row in events:
        if row.get("event") in PLAYBACK_EVENTS:
            playback = row
            break
    if playback is None:
        raise ValueError(f"missing playback event ({', '.join(PLAYBACK_EVENTS)})")

    control_t = event_time(control)
    release_request_t = event_time(release_request)
    release_confirmed_t = event_time(release_confirmed)
    playback_t = event_time(playback)
    if not control_t <= release_request_t <= release_confirmed_t <= playback_t:
        raise ValueError(
            "invalid release order: expected CONTROL <= release request <= "
            "release confirmed <= playback"
        )
    support_active = str(release_confirmed.get("support_active", "")).strip()
    if support_active not in {"0", "0.0", "false", "False"}:
        raise ValueError(
            "release confirmation did not show support_active=0 "
            f"(got {support_active!r})"
        )
    return {
        "passed_release_gate": True,
        "event_log": release_confirmed.get("_event_log"),
        "control_event_s": control_t,
        "release_request_s": release_request_t,
        "release_confirmed_s": release_confirmed_t,
        "playback_event": playback.get("event"),
        "playback_event_s": playback_t,
    }


def tracking_window(run_dir: Path, tracked_t, tracked_q):
    import numpy as np

    try:
        events = load_events(run_dir)
        playback = None
        for row in events:
            if row.get("event") in PLAYBACK_EVENTS:
                playback = row
                break
        if playback is None:
            return tracked_t, tracked_q, None
        playback_t = event_sim_time(playback)
        if tracked_t[0] <= playback_t <= tracked_t[-1]:
            keep = tracked_t >= playback_t
            return tracked_t[keep] - playback_t, tracked_q[keep], playback_t
    except Exception:
        pass
    return tracked_t, tracked_q, None


def timing_proof(run_dir: Path) -> dict:
    import numpy as np

    if run_dir.suffix != ".npz":
        t, _ = load_tracked(run_dir)
        tag = str(run_dir)
    else:
        data = np.load(run_dir, allow_pickle=False)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        t = timestamps - timestamps[0]
        tag = str(data["tag"]) if "tag" in data else run_dir.stem
    dt = np.diff(t)
    wall_duration = float(t[-1] - t[0]) if t.size else 0.0
    sim_duration = round(wall_duration / 0.02) * 0.02
    return {
        "tag": tag,
        "phase_dt_s": float(np.mean(dt)) if dt.size else 0.0,
        "phase_dt_std_s": float(np.std(dt)) if dt.size else 0.0,
        "wall_duration_s": wall_duration,
        "sim_duration_s": sim_duration,
        "phase_eq_wall": bool(abs(wall_duration - sim_duration) <= 0.05),
    }


def report(args: argparse.Namespace) -> int:
    logs = Path(args.logs)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    timing_rows = []
    for motion in MOTIONS:
        row = {"motion": motion}
        for policy in POLICIES:
            run_dir = expected_run_dir(logs, policy, motion)
            try:
                ref, hz = load_reference(policy, motion)
                t, q = load_tracked(run_dir)
                t_eval, q_eval, playback_offset = tracking_window(run_dir, t, q)
                metrics = aligned_rmse(ref, hz, t_eval, q_eval)
                if playback_offset is not None:
                    metrics["playback_offset_s"] = playback_offset
                try:
                    metrics.update(validate_release_order(run_dir))
                except Exception as exc:
                    metrics["passed_release_gate"] = False
                    metrics["release_gate_error"] = str(exc)
                    failures.append(f"{policy}/{motion}: {exc}")
                metrics["passed_rmse_gate"] = metrics["mean_joint_rmse_rad"] < 0.2
                row[policy] = metrics
                if not metrics["passed_rmse_gate"]:
                    failures.append(f"{policy}/{motion}: RMSE {metrics['mean_joint_rmse_rad']:.4f}")
            except Exception as exc:
                row[policy] = {"error": str(exc), "passed_rmse_gate": False}
                failures.append(f"{policy}/{motion}: {exc}")
            try:
                timing_rows.append(timing_proof(run_dir))
            except Exception as exc:
                failures.append(f"{policy}/{motion}: timing proof unavailable: {exc}")
        rows.append(row)
    all_rmse_passed = all(
        row[policy].get("passed_rmse_gate") is True
        for row in rows
        for policy in POLICIES
    )
    all_release_gates_passed = all(
        row[policy].get("passed_release_gate") is True
        for row in rows
        for policy in POLICIES
    )
    summary = {
        "motions": MOTIONS,
        "policies": POLICIES,
        "results": rows,
        "all_rmse_passed": all_rmse_passed,
        "all_release_gates_passed": all_release_gates_passed,
        "failures": failures,
    }
    (artifacts / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (artifacts / "phase_time_proof.json").write_text(json.dumps(timing_rows, indent=2) + "\n")
    write_metric_csvs(artifacts, summary)
    failures.extend(make_all_comparison_videos(logs, artifacts, rows))
    write_markdown(artifacts / "report.md", summary)
    return 0 if not failures else 1


def write_metric_csvs(artifacts: Path, summary: dict) -> None:
    with (artifacts / "rmse_summary.csv").open("w", newline="") as f:
        fieldnames = [
            "tag",
            "policy",
            "motion",
            "tracking_delay_s",
            "mean_joint_rmse_rad",
            "samples",
            "passed_rmse_gate",
            "passed_release_gate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["results"]:
            for policy in POLICIES:
                item = row[policy]
                writer.writerow({
                    "tag": log_tag(policy, row["motion"]),
                    "policy": policy,
                    "motion": row["motion"],
                    "tracking_delay_s": item.get("tracking_delay_s", ""),
                    "mean_joint_rmse_rad": item.get("mean_joint_rmse_rad", ""),
                    "samples": item.get("samples", ""),
                    "passed_rmse_gate": item.get("passed_rmse_gate", False),
                    "passed_release_gate": item.get("passed_release_gate", False),
                })
    with (artifacts / "per_joint_rmse.csv").open("w", newline="") as f:
        fieldnames = ["tag", "policy", "motion", "metric", *[f"joint_{i}" for i in range(29)]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["results"]:
            for policy in POLICIES:
                item = row[policy]
                if "joint_rmse_rad" not in item:
                    continue
                values = {
                    "tag": log_tag(policy, row["motion"]),
                    "policy": policy,
                    "motion": row["motion"],
                    "metric": "rmse_rad",
                }
                for i, value in enumerate(item["joint_rmse_rad"][:29]):
                    values[f"joint_{i}"] = value
                writer.writerow(values)


def make_all_comparison_videos(logs: Path, artifacts: Path, rows: list[dict]) -> list[str]:
    failures = []
    videos = []
    for row in rows:
        motion = row["motion"]
        try:
            path = make_comparison_video(logs, artifacts, motion, row)
            videos.append({"motion": motion, "path": str(path)})
        except Exception as exc:
            failures.append(f"{motion}: comparison video unavailable: {exc}")
    (artifacts / "comparison_videos.json").write_text(json.dumps(videos, indent=2) + "\n")
    return failures


def make_comparison_video(logs: Path, artifacts: Path, motion: str, row: dict) -> Path:
    import imageio.v2 as imageio
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_holo, hz_holo = load_reference("holomotion", motion)
    ref_sonic, hz_sonic = load_reference("sonic", motion)
    policy_data = []
    for policy, ref, hz in [
        ("holomotion", ref_holo, hz_holo),
        ("sonic", ref_sonic, hz_sonic),
    ]:
        run_dir = expected_run_dir(logs, policy, motion)
        t, q = load_tracked(run_dir)
        t_eval, q_eval, _ = tracking_window(run_dir, t, q)
        delay = row[policy].get("tracking_delay_s", 0.0)
        ref_eval = interpolate_ref(ref, hz, t_eval, float(delay))
        policy_data.append((policy, t_eval, q_eval[:, :6], ref_eval[:, :6]))

    max_t = max(float(data[1][-1]) for data in policy_data if data[1].size)
    frame_times = np.linspace(0.0, max_t, min(240, max(2, int(max_t * 10))))
    out = artifacts / f"{motion}_comparison.mp4"
    with imageio.get_writer(out, fps=10, codec="libx264", quality=7) as writer:
        for now in frame_times:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.12), dpi=100, sharey=True)
            for ax, (policy, t, q, ref_q) in zip(axes, policy_data):
                end = max(1, int(np.searchsorted(t, now)))
                for joint in range(q.shape[1]):
                    ax.plot(t[:end], ref_q[:end, joint], "--", linewidth=1.0, alpha=0.45)
                    ax.plot(t[:end], q[:end, joint], linewidth=1.2, alpha=0.85)
                item = row[policy]
                rmse = item.get("mean_joint_rmse_rad", float("nan"))
                delay = item.get("tracking_delay_s", float("nan"))
                release = "release pass" if item.get("passed_release_gate") else "release missing/fail"
                ax.set_title(f"{policy}  RMSE {rmse:.3f}  delay {delay:.3f}s\n{release}")
                ax.set_xlabel("phase time (s)")
                ax.set_xlim(0, max_t)
                ax.set_ylim(-3.2, 3.2)
                ax.grid(True, alpha=0.25)
            axes[0].set_ylabel("joint angle rad, first 6 DOF")
            fig.suptitle(f"{motion}  tracked vs reference ghost overlay")
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
            plt.close(fig)
    return out


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# WBC Benchmark Report",
        "",
        "| Motion | SONIC RMSE | SONIC Delay | SONIC Release | HoloMotion RMSE | HoloMotion Delay | HoloMotion Release |",
        "| --- | ---: | ---: | :---: | ---: | ---: | :---: |",
    ]
    for row in summary["results"]:
        vals = []
        for policy in POLICIES:
            item = row[policy]
            vals.extend([
                f"{item.get('mean_joint_rmse_rad', float('nan')):.4f}" if "mean_joint_rmse_rad" in item else "missing",
                f"{item.get('tracking_delay_s', float('nan')):.3f}" if "tracking_delay_s" in item else "missing",
                "pass" if item.get("passed_release_gate") else "fail",
            ])
        lines.append(
            f"| {row['motion']} | {vals[0]} | {vals[1]} | {vals[2]} | "
            f"{vals[3]} | {vals[4]} | {vals[5]} |"
        )
    lines.extend([
        "",
        f"All RMSE gates passed: `{summary['all_rmse_passed']}`",
        f"All release-order gates passed: `{summary['all_release_gates_passed']}`",
        "",
    ])
    if summary["failures"]:
        lines.append("## Failures")
        lines.extend(f"- {failure}" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n")


def validate_assets(_: argparse.Namespace) -> int:
    missing = []
    for motion in MOTIONS:
        if not (ROOT / "assets" / "motions" / "sonic_motions" / motion / "joint_pos.csv").exists():
            missing.append(f"sonic {motion}")
        if not (ROOT / "assets" / "motions" / "holomotion_motions" / f"{motion}_holomotion.npz").exists():
            missing.append(f"holomotion {motion}")
    required = [
        ROOT / "assets" / "GEAR-SONIC" / "model_decoder.onnx",
        ROOT / "assets" / "GEAR-SONIC" / "model_encoder.onnx",
        ROOT / "assets" / "GEAR-SONIC" / "observation_config.yaml",
        ROOT / "assets" / "HoloMotion_models" / "HoloMotion_motion_tracking_model" / "exported" / "motion_tracking_model.onnx",
    ]
    missing.extend(str(path) for path in required if not path.exists())
    print(json.dumps({"ok": not missing, "missing": missing}, indent=2))
    return 0 if not missing else 1


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def prepare_stock_assets(_: argparse.Namespace) -> int:
    gear_assets = ROOT / "assets" / "GEAR-SONIC"
    sonic_policy = SONIC_DEPLOY / "policy" / "release"
    sonic_policy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gear_assets / "model_decoder.onnx", sonic_policy / "model_decoder.onnx")
    shutil.copy2(gear_assets / "model_encoder.onnx", sonic_policy / "model_encoder.onnx")
    shutil.copy2(gear_assets / "observation_config.yaml", sonic_policy / "observation_config.yaml")
    copy_tree_contents(ROOT / "assets" / "motions" / "sonic_motions", SONIC_DEPLOY / "reference" / "benchmark")

    holo_models = ROOT / "assets" / "HoloMotion_models"
    holo_model_dst = HOLO_DEPLOY / "src" / "models"
    copy_tree_contents(holo_models / "HoloMotion_motion_tracking_model", holo_model_dst / "motion_tracking_model")
    copy_tree_contents(holo_models / "HoloMotion_velocity_tracking_model", holo_model_dst / "velocity_tracking_model")
    copy_tree_contents(ROOT / "assets" / "motions" / "holomotion_motions", HOLO_DEPLOY / "src" / "motion_data")

    staged = {
        "sonic_policy": str(sonic_policy),
        "sonic_motion_data": str(SONIC_DEPLOY / "reference" / "benchmark"),
        "holomotion_models": str(holo_model_dst),
        "holomotion_motion_data": str(HOLO_DEPLOY / "src" / "motion_data"),
    }
    print(json.dumps({"ok": True, "staged": staged}, indent=2))
    return 0


def git_output(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={cwd}", *args], cwd=cwd, text=True
    ).strip()


def validate_deploy(_: argparse.Namespace) -> int:
    checks: list[dict[str, str | bool]] = []
    warnings: list[dict[str, str | bool]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    def warn(name: str, ok: bool, detail: str = "") -> None:
        warnings.append({"name": name, "ok": ok, "detail": detail})

    holo = ROOT / "thirdparties" / "HoloMotion"
    sonic = ROOT / "thirdparties" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
    try:
        version = git_output(["describe", "--tags", "--always"], holo)
    except Exception as exc:
        version = f"unavailable: {exc}"
    add("holomotion_v1_3_lineage", version.startswith(HOLOMOTION_MIN_TAG), version)
    add(
        "holomotion_stock_launcher",
        (holo / "deployment" / "unitree_g1_ros2_29dof" / "launch_holomotion_29dof_docker.sh").exists(),
        "launch_holomotion_29dof_docker.sh",
    )
    doc = holo / "docs" / "realworld_deployment.md"
    doc_text = doc.read_text(errors="replace") if doc.exists() else ""
    add("holomotion_offline_motion_docs", "Offline Motion" in doc_text, str(doc))
    add("holomotion_motion_mode_docs", "Press `B` to enter motion tracking" in doc_text, str(doc))
    add(
        "sonic_stock_deploy",
        (sonic / "deploy.sh").exists() and (sonic / ".justfile").exists(),
        "deploy.sh + .justfile",
    )
    deploy_text = (sonic / "deploy.sh").read_text(errors="replace") if (sonic / "deploy.sh").exists() else ""
    add("sonic_stock_binary_recipe", "g1_deploy_onnx_ref" in deploy_text, "g1_deploy_onnx_ref")
    add(
        "sonic_benchmark_assets_staged",
        (sonic / "policy" / "release" / "model_decoder.onnx").exists()
        and (sonic / "policy" / "release" / "model_encoder.onnx").exists()
        and (sonic / "reference" / "benchmark" / MOTIONS[0] / "joint_pos.csv").exists(),
        "run scripts/run.sh prepare-assets",
    )
    planner = sonic / "planner" / "target_vel" / "V2" / "planner_sonic.onnx"
    warn("sonic_optional_velocity_planner_present", planner.exists(), str(planner))
    add(
        "holomotion_assets_staged",
        (holo / "deployment" / "unitree_g1_ros2_29dof" / "src" / "models" / "motion_tracking_model" / "exported" / "motion_tracking_model.onnx").exists()
        and (holo / "deployment" / "unitree_g1_ros2_29dof" / "src" / "models" / "velocity_tracking_model" / "exported" / "velocity_model.onnx").exists()
        and (holo / "deployment" / "unitree_g1_ros2_29dof" / "src" / "motion_data" / f"{MOTIONS[0]}_holomotion.npz").exists(),
        "run scripts/run.sh prepare-assets",
    )
    add(
        "run_sonic_reference_only",
        (ROOT / "thirdparties" / "run-sonic").exists(),
        "present for read-only reference; not used by this script",
    )

    failures = [item for item in checks if not item["ok"]]
    print(json.dumps({"ok": not failures, "checks": checks, "warnings": warnings}, indent=2))
    return 0 if not failures else 1


def docker_check(image: str, script: str) -> dict[str, object]:
    cmd = ["docker", "run", "--rm", image, "bash", "-lc", script]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "image": image,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def validate_images(_: argparse.Namespace) -> int:
    checks = [
        docker_check(
            "wbc-unitree_mujoco",
            "python3 - <<'PY'\n"
            "import imageio, mujoco, numpy, scipy\n"
            "from pathlib import Path\n"
            "assert Path('/workspace/unitree_mujoco/simulate').exists()\n"
            "PY\n"
            "command -v ffmpeg >/dev/null",
        ),
        docker_check(
            "wbc-gear-sonic",
            "check_path() { test -e \"$1\" || { echo \"missing: $1\"; exit 1; }; }\n"
            "check_cmd() { command -v \"$1\" >/dev/null || { echo \"missing command: $1\"; exit 1; }; }\n"
            "check_path /workspace/GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh\n"
            "check_path /opt/ros/humble/setup.bash\n"
            "check_path /opt/onnxruntime/lib\n"
            "check_path /usr/include/x86_64-linux-gnu/NvInfer.h\n"
            "ldconfig -p | grep -q libnvinfer\n"
            "ldconfig -p | grep -q libnvonnxparser\n"
            "check_cmd colcon",
        ),
        docker_check(
            "wbc-holomotion",
            "check_path() { test -e \"$1\" || { echo \"missing: $1\"; exit 1; }; }\n"
            "check_cmd() { command -v \"$1\" >/dev/null || { echo \"missing command: $1\"; exit 1; }; }\n"
            "check_path /workspace/HoloMotion/deployment/unitree_g1_ros2_29dof/launch_holomotion_29dof_docker.sh\n"
            "check_path /opt/conda/etc/profile.d/conda.sh\n"
            "check_path /opt/conda/envs/holomotion_deploy\n"
            "check_path /opt/ros/humble/setup.sh\n"
            "check_path /opt/unitree_ros2/setup.sh\n"
            "check_path /opt/unitree_ros2/cyclonedds_ws/install/setup.bash\n"
            "check_path /workspace/HoloMotion/deployment/unitree_g1_ros2_29dof/install/setup.bash\n"
            "check_path /opt/cyclonedds/install/lib\n"
            "check_cmd colcon\n"
            "test \"$PROFILE_PYTHON\" = /usr/bin/python3\n"
            "conda run -n holomotion_deploy python - <<'PY'\n"
            "import zmq\n"
            "PY",
        ),
    ]
    failures = [item for item in checks if not item["ok"]]
    print(json.dumps({"ok": not failures, "checks": checks}, indent=2))
    return 0 if not failures else 1


def smoke_holomotion(_: argparse.Namespace) -> int:
    script = (
        "cd /workspace/HoloMotion/deployment/unitree_g1_ros2_29dof && "
        "timeout 20s ./launch_holomotion_29dof_docker.sh "
        "--profile launch_profiles/x86_64_docker.yaml "
        "--set runtime.unitree_setup=/opt/unitree_ros2/cyclonedds_ws/install/setup.bash "
        "--set robot.network_interface=lo "
        "--set policy.inference_backend=onnx"
    )
    result = docker_check("wbc-holomotion", script)
    combined = "\n".join(
        str(result.get(key, "")) for key in ("stdout", "stderr")
    )
    markers = [
        "Entered ZERO_TORQUE state",
        "latest_obs subscriber ready",
        "Dual policies loaded successfully",
        "Loaded 10 motion clips successfully",
        "Policy node setup completed successfully",
    ]
    missing = [marker for marker in markers if marker not in combined]
    ok = result["returncode"] in {0, 124} and not missing
    payload = {
        "ok": ok,
        "image": result["image"],
        "returncode": result["returncode"],
        "missing_markers": missing,
    }
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


def smoke_sonic_build(_: argparse.Namespace) -> int:
    script = (
        "cd /workspace/GR00T-WholeBodyControl/gear_sonic_deploy && "
        "source /opt/ros/humble/setup.bash && "
        "source scripts/setup_env.sh && "
        "rm -rf build && "
        "cmake -S . -B build "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON "
        "-DTensorRT_FIND_COMPONENTS='nvinfer;nvinfer_plugin;nvonnxparser' && "
        "cmake --build build --target g1_deploy_onnx_ref -j$(nproc) && "
        "test -x target/release/g1_deploy_onnx_ref"
    )
    result = docker_check("wbc-gear-sonic", script)
    combined = "\n".join(
        str(result.get(key, "")) for key in ("stdout", "stderr")
    )
    markers = [
        "Found TensorRT",
        "found components: nvinfer nvinfer_plugin nvonnxparser",
        "Built target g1_deploy_onnx_ref",
    ]
    missing = [marker for marker in markers if marker not in combined]
    ok = result["returncode"] == 0 and not missing
    payload = {
        "ok": ok,
        "image": result["image"],
        "returncode": result["returncode"],
        "missing_markers": missing,
    }
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


def docker(args: argparse.Namespace) -> int:
    image = f"wbc-{args.image}"
    dockerfile = ROOT / "docker" / f"{args.image}.Dockerfile"
    cmd = ["docker", "build", "-f", str(dockerfile), "-t", image, str(ROOT)]
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate-assets").set_defaults(func=validate_assets)
    sub.add_parser("prepare-assets").set_defaults(func=prepare_stock_assets)
    sub.add_parser("validate-deploy").set_defaults(func=validate_deploy)
    sub.add_parser("validate-images").set_defaults(func=validate_images)
    sub.add_parser("smoke-holomotion").set_defaults(func=smoke_holomotion)
    sub.add_parser("smoke-sonic-build").set_defaults(func=smoke_sonic_build)
    p = sub.add_parser("report")
    p.add_argument("--logs", default=str(ROOT / "logs"))
    p.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    p.set_defaults(func=report)
    p = sub.add_parser("docker-build")
    p.add_argument("image", choices=["gear-sonic", "holomotion", "unitree_mujoco"])
    p.set_defaults(func=docker)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
