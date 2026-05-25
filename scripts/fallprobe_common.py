#!/usr/bin/env python3
"""Helpers for SONIC fall probes."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmark_common import RELEASE_CONFIRMED_EVENT, event_sim_time, expected_run_dir, first_event, load_events
from benchmark_report import aligned_rmse, load_reference, load_replay, load_tracked, tracking_window


def measure_attempt(root: Path, motion: str, fall_base_z: float) -> dict[str, float | bool]:
    run_dir = expected_run_dir(root / "logs", "sonic", motion)
    ref, hz = load_reference("sonic", motion)
    tracked_t, tracked_q = load_tracked(run_dir)
    eval_t, eval_q, _ = tracking_window(run_dir, tracked_t, tracked_q)
    rmse = aligned_rmse(ref, hz, eval_t, eval_q)

    events = load_events(run_dir)
    release_t = event_sim_time(first_event(events, RELEASE_CONFIRMED_EVENT))
    replay = load_replay(run_dir)
    released = replay["sim_time_s"] >= release_t
    base_z_min = float(replay["base_z"][released].min())
    return {
        "rmse": float(rmse["mean_joint_rmse_rad"]),
        "delay": float(rmse["tracking_delay_s"]),
        "base_z_min": base_z_min,
        "fell": base_z_min < fall_base_z,
    }


def _read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_summary(
    out_dir: Path,
    attempt: int,
    motion: str,
    ok: bool,
    fell: bool,
    rmse: object,
    delay: object,
    base_z_min: object,
    error: str,
) -> dict[str, float | int]:
    path = out_dir / "summary.csv"
    rows = _read_summary(path)
    total_attempts = sum(1 for row in rows if row.get("ok") == "true") + (1 if ok else 0)
    motion_attempts = sum(1 for row in rows if row.get("motion") == motion and row.get("ok") == "true") + (
        1 if ok else 0
    )
    motion_falls = sum(1 for row in rows if row.get("motion") == motion and row.get("fell") == "true") + (
        1 if ok and fell else 0
    )
    fall_rate = float(motion_falls / motion_attempts) if motion_attempts else 0.0
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                attempt,
                motion,
                str(ok).lower(),
                str(fell).lower(),
                fall_rate if ok else "",
                total_attempts,
                motion_attempts if ok else "",
                motion_falls if ok else "",
                rmse,
                delay,
                base_z_min,
                error,
            ]
        )
    return {
        "fall_rate": fall_rate,
        "total_attempts": total_attempts,
        "motion_attempts": motion_attempts,
        "motion_falls": motion_falls,
    }
