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
    return logs / policy / motion


def report(args: argparse.Namespace) -> int:
    logs = Path(args.logs)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for motion in MOTIONS:
        row = {"motion": motion}
        for policy in POLICIES:
            run_dir = expected_run_dir(logs, policy, motion)
            try:
                ref, hz = load_reference(policy, motion)
                t, q = load_tracked(run_dir)
                metrics = aligned_rmse(ref, hz, t, q)
                metrics["passed_rmse_gate"] = metrics["mean_joint_rmse_rad"] < 0.2
                row[policy] = metrics
                if not metrics["passed_rmse_gate"]:
                    failures.append(f"{policy}/{motion}: RMSE {metrics['mean_joint_rmse_rad']:.4f}")
            except Exception as exc:
                row[policy] = {"error": str(exc), "passed_rmse_gate": False}
                failures.append(f"{policy}/{motion}: {exc}")
        rows.append(row)
    summary = {
        "motions": MOTIONS,
        "policies": POLICIES,
        "results": rows,
        "all_rmse_passed": not failures,
        "failures": failures,
    }
    (artifacts / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(artifacts / "report.md", summary)
    return 0 if not failures else 1


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# WBC Benchmark Report",
        "",
        "| Motion | SONIC RMSE | SONIC Delay | HoloMotion RMSE | HoloMotion Delay |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["results"]:
        vals = []
        for policy in POLICIES:
            item = row[policy]
            vals.extend([
                f"{item.get('mean_joint_rmse_rad', float('nan')):.4f}" if "mean_joint_rmse_rad" in item else "missing",
                f"{item.get('tracking_delay_s', float('nan')):.3f}" if "tracking_delay_s" in item else "missing",
            ])
        lines.append(f"| {row['motion']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    lines.extend(["", f"All RMSE gates passed: `{summary['all_rmse_passed']}`", ""])
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
