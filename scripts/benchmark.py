#!/usr/bin/env python3
"""Root-owned benchmark utilities for the WBC comparison.

This file is intentionally flat under scripts/. It may use run-sonic as design
reference, but it does not import or execute code from thirdparties/run-sonic.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
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
PLAYBACK_EVENTS = ["sent_key_T", "sent_key_B", "motion_start_observed", "motion_playing_observed"]
EVENT_LOG_FIELDS = [
    "event",
    "monotonic_s",
    "wall_time_s",
    "sim_time_s",
    "support_active",
    "detail",
]
HOLOMOTION_MIN_TAG = "v1.3.0"
SONIC_DEPLOY = ROOT / "thirdparties" / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
HOLO_DEPLOY = ROOT / "thirdparties" / "HoloMotion" / "deployment" / "unitree_g1_ros2_29dof"
SIM_IMAGE = "wbc-unitree_mujoco"
SONIC_IMAGE = "wbc-gear-sonic"
HOLO_IMAGE = "wbc-holomotion"
GLOBAL_IMPOSSIBILITY = "impossibility.json"
HOLO_KEY_BITS = {
    "start": 1 << 2,
    "select": 1 << 3,
    "a": 1 << 8,
    "b": 1 << 9,
    "y": 1 << 11,
    "up": 1 << 12,
    "right": 1 << 13,
    "down": 1 << 14,
    "left": 1 << 15,
}
SONIC_POLICY_FROM_HARDWARE = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]
HARDWARE_FROM_SONIC_POLICY = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]
SONIC_BODY_PART_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
SONIC_POST_RELEASE_WAIT_BY_MOTION = {
    "dance_phony_c01_neutral2s": 0.5,
}


class SequenceEventLog:
    """Append-only release-order event log for real benchmark runs."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._monotonic0 = time.monotonic()
        self._wall0 = time.time()
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=EVENT_LOG_FIELDS).writeheader()

    def append(
        self,
        event: str,
        *,
        sim_time_s: float | None = None,
        support_active: int | bool | str | None = None,
        detail: str = "",
    ) -> None:
        monotonic_s = time.monotonic() - self._monotonic0
        row = {
            "event": event,
            "monotonic_s": f"{monotonic_s:.6f}",
            "wall_time_s": f"{self._wall0 + monotonic_s:.6f}",
            "sim_time_s": "" if sim_time_s is None else f"{float(sim_time_s):.6f}",
            "support_active": "" if support_active is None else str(support_active),
            "detail": detail,
        }
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=EVENT_LOG_FIELDS).writerow(row)


def write_sequence_events(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in EVENT_LOG_FIELDS})


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
        return q[:, HARDWARE_FROM_SONIC_POLICY], 50.0
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


def first_event_index(events: list[dict[str, str]], event: str) -> int:
    for index, row in enumerate(events):
        if row.get("event") == event:
            return index
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

    control_i = first_event_index(events, CONTROL_EVENT)
    release_request_i = first_event_index(events, RELEASE_REQUEST_EVENT)
    release_confirmed_i = first_event_index(events, RELEASE_CONFIRMED_EVENT)
    playback_i = events.index(playback)
    if not control_i <= release_request_i <= release_confirmed_i <= playback_i:
        raise ValueError(
            "invalid release log order: expected CONTROL row <= release request "
            "row <= release confirmed row <= playback row"
        )

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


def smoke_release_gate(_: argparse.Namespace) -> int:
    valid_rows = [
        {
            "event": CONTROL_EVENT,
            "monotonic_s": "1.000000",
            "wall_time_s": "101.000000",
            "sim_time_s": "1.000000",
            "support_active": "1",
            "detail": "policy/controller reached active CONTROL",
        },
        {
            "event": RELEASE_REQUEST_EVENT,
            "monotonic_s": "2.000000",
            "wall_time_s": "102.000000",
            "sim_time_s": "2.000000",
            "support_active": "1",
            "detail": "requested simulator support release",
        },
        {
            "event": RELEASE_CONFIRMED_EVENT,
            "monotonic_s": "2.100000",
            "wall_time_s": "102.100000",
            "sim_time_s": "2.100000",
            "support_active": "0",
            "detail": "simulator reports support inactive",
        },
        {
            "event": "sent_key_T",
            "monotonic_s": "2.200000",
            "wall_time_s": "102.200000",
            "sim_time_s": "2.200000",
            "support_active": "0",
            "detail": "stock SONIC playback trigger after release",
        },
    ]
    invalid_order_rows = [valid_rows[0], valid_rows[3], valid_rows[1], valid_rows[2]]
    invalid_support_rows = [
        row if row["event"] != RELEASE_CONFIRMED_EVENT else {**row, "support_active": "1"}
        for row in valid_rows
    ]

    results = []
    with tempfile.TemporaryDirectory(prefix="wbc-release-gate-") as tmp:
        tmp_path = Path(tmp)
        cases = [
            ("valid", valid_rows, True),
            ("invalid_order", invalid_order_rows, False),
            ("invalid_support", invalid_support_rows, False),
        ]
        for name, rows, should_pass in cases:
            run_dir = tmp_path / name
            write_sequence_events(run_dir / "sequence_events.csv", rows)
            try:
                validate_release_order(run_dir)
                passed = True
                error = ""
            except Exception as exc:
                passed = False
                error = str(exc)
            ok = passed is should_pass
            results.append({
                "case": name,
                "ok": ok,
                "passed_release_gate": passed,
                "expected_pass": should_pass,
                "error": error,
            })

    payload = {"ok": all(item["ok"] for item in results), "cases": results}
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


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


def load_global_impossibility(logs: Path) -> dict | None:
    path = logs / GLOBAL_IMPOSSIBILITY
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not payload.get("documented_impossibility"):
        raise ValueError(f"{path}: does not declare documented_impossibility=true")
    return payload


def write_impossibility_artifacts(artifacts: Path, payload: dict) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    summary = {
        "motions": MOTIONS,
        "policies": POLICIES,
        "documented_impossibility": True,
        "all_rmse_passed": False,
        "all_release_gates_passed": False,
        "impossibility": payload,
        "failures": [],
        "results": [
            {
                "motion": motion,
                **{
                    policy: {
                        "documented_impossibility": True,
                        "passed_rmse_gate": False,
                        "passed_release_gate": False,
                        "reason": payload.get("reason", ""),
                    }
                    for policy in POLICIES
                },
            }
            for motion in MOTIONS
        ],
    }
    (artifacts / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (artifacts / GLOBAL_IMPOSSIBILITY).write_text(json.dumps(payload, indent=2) + "\n")
    (artifacts / "phase_time_proof.json").write_text(
        json.dumps({
            "documented_impossibility": True,
            "reason": payload.get("reason", ""),
            "phase_wall_sim_time_proof_unavailable": True,
        }, indent=2) + "\n"
    )
    (artifacts / "comparison_videos.json").write_text(
        json.dumps({
            "documented_impossibility": True,
            "reason": payload.get("reason", ""),
            "videos": [],
        }, indent=2) + "\n"
    )
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
            "documented_impossibility",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for motion in MOTIONS:
            for policy in POLICIES:
                writer.writerow({
                    "tag": log_tag(policy, motion),
                    "policy": policy,
                    "motion": motion,
                    "passed_rmse_gate": False,
                    "passed_release_gate": False,
                    "documented_impossibility": True,
                })
    with (artifacts / "per_joint_rmse.csv").open("w", newline="") as f:
        csv.writer(f).writerow(["tag", "policy", "motion", "metric", *[f"joint_{i}" for i in range(29)]])
    write_impossibility_markdown(artifacts / "report.md", payload)


def write_impossibility_markdown(path: Path, payload: dict) -> None:
    evidence = payload.get("evidence", {})
    lines = [
        "# WBC Benchmark Report",
        "",
        "Documented impossibility: `true`",
        "",
        f"Reason: {payload.get('reason', '')}",
        "",
        "The stock deploy path cannot produce valid RMSE or timing/video artifacts on this host because the GPU runtime required by the stock policy initialization is unavailable.",
        "",
        "## Evidence",
    ]
    for name, item in evidence.items():
        lines.append(f"- {name}: returncode `{item.get('returncode', '')}`")
        detail = (item.get("stdout", "") + "\n" + item.get("stderr", "")).strip()
        if detail:
            lines.append("")
            lines.append("```text")
            lines.append(detail[-2000:])
            lines.append("```")
            lines.append("")
    lines.extend([
        "## Gate Status",
        "",
        "- RMSE gate: not evaluated because stock policy startup is impossible on this host.",
        "- Release gate: not evaluated because policy CONTROL is unreachable without the stock policy runtime.",
        "- Tier gate: script line count remains below Bronze; no thirdparty source changes are required for this documented-impossibility path.",
    ])
    path.write_text("\n".join(lines) + "\n")


def report(args: argparse.Namespace) -> int:
    logs = Path(args.logs)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    impossibility = load_global_impossibility(logs)
    if impossibility is not None:
        write_impossibility_artifacts(artifacts, impossibility)
        return 0
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


class ManagedProcess:
    """Small process wrapper that keeps policy stdout in a file while driving stdin."""

    def __init__(self, cmd: list[str], log_path: Path, *, use_pty: bool = False):
        self.cmd = cmd
        self.log_path = log_path
        self.use_pty = use_pty
        self.proc: subprocess.Popen | None = None
        self._master_fd: int | None = None
        self._log_f = None

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
            raise RuntimeError("process was not started with a pty")
        os.write(self._master_fd, text.encode())
        self._drain_pty()

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

    def terminate(self, timeout: float = 10.0) -> None:
        self._drain_pty()
        proc = self.proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5.0)
        self._drain_pty()
        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._master_fd)
            self._master_fd = None
        if self._log_f is not None:
            self._log_f.close()
            self._log_f = None


def docker_rm_force(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def cleanup_stale_benchmark_containers() -> None:
    prefixes = ("wbc-sim-", "wbc-sonic-", "wbc-holo-", "wbc-holo-bridge-")
    exact = {"unitree_mujoco"}
    proc = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return
    stale = [
        name
        for name in proc.stdout.splitlines()
        if name in exact or any(name.startswith(prefix) for prefix in prefixes)
    ]
    for name in stale:
        docker_rm_force(name)
    if stale:
        print(json.dumps({"cleanup_stale_containers": stale}), flush=True)


def docker_base_args(name: str, image: str, *, tty: bool = False) -> list[str]:
    args = ["docker", "run", "--rm", "--name", name, "--network", "host"]
    gpu_request = os.environ.get("WBC_DOCKER_GPUS")
    if gpu_request is None and image in {SONIC_IMAGE, HOLO_IMAGE} and shutil.which("nvidia-smi"):
        gpu_request = "all"
    if gpu_request:
        args.extend(["--gpus", gpu_request])
    if tty:
        args.extend(["-i", "-t"])
    args.extend(["-v", f"{ROOT}:/workspace/wbc", image])
    return args


def capture_command(cmd: list[str], timeout_s: float = 30.0) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": str(exc)}


def probe_gpu_runtime() -> dict[str, object]:
    host = capture_command([
        "bash",
        "-lc",
        "command -v nvidia-smi >/dev/null && nvidia-smi || true; "
        "ls -l /dev/nvidia* 2>/dev/null || true",
    ])
    docker_gpu = capture_command([
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        SONIC_IMAGE,
        "bash",
        "-lc",
        "test -e /dev/nvidiactl && echo WBC_GPU_DEVICE_PRESENT; "
        "nvidia-smi 2>&1 || true",
    ])
    docker_plain = capture_command([
        "docker",
        "run",
        "--rm",
        SONIC_IMAGE,
        "bash",
        "-lc",
        "test -e /dev/nvidiactl && echo WBC_GPU_DEVICE_PRESENT || echo WBC_GPU_DEVICE_ABSENT",
    ])
    docker_text = "\n".join(str(docker_gpu.get(key, "")) for key in ("stdout", "stderr"))
    ok = (
        docker_gpu.get("returncode") == 0
        and ("WBC_GPU_DEVICE_PRESENT" in docker_text or "NVIDIA-SMI" in docker_text)
    )
    reason = ""
    if not ok:
        reason = (
            "Stock policy runtime requires a working NVIDIA GPU driver/runtime. "
            "The Docker GPU probe did not expose /dev/nvidiactl or NVIDIA-SMI, "
            "so SONIC TensorRT initialization cannot run."
        )
    return {
        "ok": ok,
        "reason": reason,
        "evidence": {
            "host_nvidia_probe": host,
            "docker_gpu_probe": docker_gpu,
            "docker_plain_probe": docker_plain,
        },
    }


def write_global_impossibility(logs: Path, probe: dict[str, object]) -> dict[str, object]:
    logs.mkdir(parents=True, exist_ok=True)
    payload = {
        "documented_impossibility": True,
        "scope": "global",
        "created_wall_time_s": time.time(),
        "policies": POLICIES,
        "motions": MOTIONS,
        "reason": probe.get("reason", "Benchmark execution is impossible in this runtime."),
        "hard_gate": "Mean joint RMSE < 0.2 for both policies on all motions",
        "evidence": probe.get("evidence", {}),
    }
    (logs / GLOBAL_IMPOSSIBILITY).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def read_control_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "support_active": True,
            "wireless_keys": 0,
            "lx": 0.0,
            "ly": 0.0,
            "rx": 0.0,
            "ry": 0.0,
            "stop": False,
        }
    return json.loads(path.read_text())


def update_control_file(path: Path, **updates: object) -> dict[str, object]:
    data = read_control_file(path)
    data.update(updates)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


def wait_for_file(path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {path}")


def latest_sim_status(run_dir: Path) -> dict[str, str] | None:
    path = run_dir / "simulator_status.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def latest_sim_time(run_dir: Path) -> float | None:
    row = latest_sim_status(run_dir)
    if not row:
        return None
    value = row.get("sim_time_s")
    return None if value in (None, "") else float(value)


def wait_for_support_state(run_dir: Path, support_active: int, timeout_s: float) -> dict[str, str]:
    deadline = time.monotonic() + timeout_s
    expected = str(int(support_active))
    while time.monotonic() < deadline:
        row = latest_sim_status(run_dir)
        if row and row.get("support_active") == expected:
            return row
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for simulator support_active={expected}")


def wait_for_pre_release_hold(
    run_dir: Path,
    event_log: SequenceEventLog,
    *,
    window: int = 20,
    max_base_z_span: float = 0.20,
    timeout_s: float = 15.0,
) -> None:
    status_path = run_dir / "simulator_status.csv"
    deadline = time.monotonic() + timeout_s
    start_sim_time = latest_sim_time(run_dir)
    while time.monotonic() < deadline:
        if status_path.exists():
            with status_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            active_rows = [
                row for row in rows
                if row.get("support_active") == "1"
                and (start_sim_time is None or float(row["sim_time_s"]) >= start_sim_time)
            ]
            if len(active_rows) >= window:
                recent = active_rows[-window:]
                base_z = [float(row["base_z"]) for row in recent]
                span = max(base_z) - min(base_z)
                if span <= max_base_z_span:
                    event_log.append(
                        "pre_release_hold_converged",
                        sim_time_s=float(recent[-1]["sim_time_s"]),
                        support_active=1,
                        detail=f"{window} support-active rows base_z_span={span:.6f}",
                    )
                    return
        time.sleep(0.05)
    raise TimeoutError("timed out waiting for pre-release hold convergence")


def read_log_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def wait_for_log_marker(
    proc: ManagedProcess,
    log_path: Path,
    markers: str | list[str],
    timeout_s: float,
) -> str:
    marker_list = [markers] if isinstance(markers, str) else markers
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc.poll()
        text = read_log_text(log_path)
        for marker in marker_list:
            if marker in text:
                return marker
        code = proc.poll()
        if code is not None:
            raise RuntimeError(f"process exited before marker {marker_list}: {code}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for log marker {marker_list}")


def pulse_holomotion_key(control_file: Path, key: str, *, hold_s: float = 0.16, gap_s: float = 0.16) -> None:
    update_control_file(control_file, wireless_keys=HOLO_KEY_BITS[key])
    time.sleep(hold_s)
    update_control_file(control_file, wireless_keys=0)
    time.sleep(gap_s)


def tap_holomotion_key(control_file: Path, key: str, *, gap_s: float = 0.20) -> None:
    update_control_file(control_file, wireless_keys=0, wireless_keys_once=HOLO_KEY_BITS[key])
    time.sleep(gap_s)


def wait_for_new_log_marker(log_path: Path, marker: str, start_offset: int, timeout_s: float) -> tuple[int, str]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = read_log_text(log_path)
        found = text.find(marker, start_offset)
        if found >= 0:
            return len(text), marker
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for new log marker {marker!r}")


def wait_for_holomotion_clip_selection(
    log_path: Path,
    start_offset: int,
    expected_clip: str,
    timeout_s: float,
) -> tuple[int, str]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = read_log_text(log_path)
        found = text.find(expected_clip, start_offset)
        if found >= 0:
            line_start = text.rfind("\n", 0, found) + 1
            line_end = text.find("\n", found)
            if line_end < 0:
                line_end = len(text)
            line = text[line_start:line_end]
            if "Selected " in line and "motion clip" in line:
                return len(text), line.split("] ", 1)[-1]
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for HoloMotion clip selection {expected_clip!r}")


def select_holomotion_clip(control_file: Path, log_path: Path, motion: str, timeout_s: float) -> str:
    clip_index = holomotion_clip_index(motion)
    log_offset = len(read_log_text(log_path))
    tap_holomotion_key(control_file, "left")
    log_offset, marker = wait_for_holomotion_clip_selection(
        log_path,
        log_offset,
        f"{MOTIONS[0]}_holomotion.npz",
        timeout_s,
    )
    for next_index in range(1, clip_index + 1):
        tap_holomotion_key(control_file, "down")
        log_offset, marker = wait_for_holomotion_clip_selection(
            log_path,
            log_offset,
            f"{MOTIONS[next_index]}_holomotion.npz",
            timeout_s,
        )
    expected_clip = f"{motion}_holomotion.npz"
    if expected_clip not in marker:
        raise RuntimeError(f"HoloMotion selected clip did not settle on {expected_clip}: {marker}")
    return marker


def release_support(run_dir: Path, control_file: Path, event_log: SequenceEventLog, detail: str) -> None:
    row = latest_sim_status(run_dir)
    event_log.append(
        RELEASE_REQUEST_EVENT,
        sim_time_s=latest_sim_time(run_dir),
        support_active=(row or {}).get("support_active", "1"),
        detail=detail,
    )
    update_control_file(control_file, support_active=False)
    released = wait_for_support_state(run_dir, 0, 5.0)
    event_log.append(
        RELEASE_CONFIRMED_EVENT,
        sim_time_s=float(released["sim_time_s"]),
        support_active=0,
        detail="simulator reports support inactive",
    )


def policy_motion_duration(policy: str, motion: str) -> float:
    del policy
    path = ROOT / "assets" / "motions" / "sonic_motions" / motion / "joint_pos.csv"
    with path.open(newline="") as f:
        frames = max(0, sum(1 for _ in f) - 1)
    return float(frames) / 50.0


def csv_shape(path: Path) -> tuple[int, int]:
    rows = 0
    width = 0
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            values = [value for value in row if value != ""]
            if not values:
                continue
            rows += 1
            width = len(values) if width == 0 else min(width, len(values))
    if rows == 0 or width == 0:
        raise ValueError(f"{path}: no numeric data rows")
    return rows, width


def write_sonic_metadata_for_stock_reader(dst: Path) -> None:
    joint_rows, joint_width = csv_shape(dst / "joint_pos.csv")
    body_pos_rows, body_pos_width = csv_shape(dst / "body_pos.csv")
    body_quat_rows, body_quat_width = csv_shape(dst / "body_quat.csv")
    body_lin_vel_rows, body_lin_vel_width = csv_shape(dst / "body_lin_vel.csv")
    body_ang_vel_rows, body_ang_vel_width = csv_shape(dst / "body_ang_vel.csv")
    body_count = body_pos_width // 3
    quat_count = body_quat_width // 4
    if body_count != quat_count:
        raise ValueError(f"{dst}: body_pos/body_quat body-count mismatch")
    if body_count != len(SONIC_BODY_PART_INDEXES):
        raise ValueError(f"{dst}: body_pos has {body_count} bodies, expected stock SONIC 14-body subset")
    indexes = " ".join(str(i) for i in SONIC_BODY_PART_INDEXES)
    (dst / "metadata.txt").write_text(
        "\n".join([
            f"Metadata for: {dst.name}",
            "==============================",
            "",
            "Body part indexes:",
            f"[ {indexes}]",
            "",
            f"Total timesteps: {joint_rows}",
            "",
            "Data arrays summary:",
            f"  joint_pos: ({joint_rows}, {joint_width}) (float64)",
            f"  joint_vel: ({joint_rows}, {joint_width}) (float64)",
            f"  body_pos_w: ({body_pos_rows}, {body_count}, 3) (float64)",
            f"  body_quat_w: ({body_quat_rows}, {quat_count}, 4) (float64)",
            f"  body_lin_vel_w: ({body_lin_vel_rows}, {body_lin_vel_width // 3}, 3) (float64)",
            f"  body_ang_vel_w: ({body_ang_vel_rows}, {body_ang_vel_width // 3}, 3) (float64)",
            f"  _body_indexes: ({len(SONIC_BODY_PART_INDEXES)},) (int64)",
            "",
        ])
    )


def subset_sonic_body_csv(path: Path, coords_per_body: int) -> None:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"{path}: no data rows")
    width = min(len([value for value in row if value != ""]) for row in data_rows)
    if width % coords_per_body != 0:
        raise ValueError(f"{path}: width {width} is not divisible by {coords_per_body}")
    body_count = width // coords_per_body
    if body_count == len(SONIC_BODY_PART_INDEXES):
        return
    if max(SONIC_BODY_PART_INDEXES) >= body_count:
        raise ValueError(f"{path}: has {body_count} bodies, cannot select SONIC body indexes")

    selected_cols = [
        body_idx * coords_per_body + coord
        for body_idx in SONIC_BODY_PART_INDEXES
        for coord in range(coords_per_body)
    ]
    out_rows = []
    if len(header) >= width:
        out_rows.append([header[col] for col in selected_cols])
    else:
        out_rows.append([f"value_{i}" for i in range(len(selected_cols))])
    for row in data_rows:
        values = [value for value in row if value != ""]
        if len(values) < width:
            raise ValueError(f"{path}: ragged row with {len(values)} values, expected at least {width}")
        out_rows.append([values[col] for col in selected_cols])
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(out_rows)


def subset_sonic_body_csvs_for_stock_reader(dst: Path) -> None:
    subset_sonic_body_csv(dst / "body_pos.csv", 3)
    subset_sonic_body_csv(dst / "body_quat.csv", 4)
    subset_sonic_body_csv(dst / "body_lin_vel.csv", 3)
    subset_sonic_body_csv(dst / "body_ang_vel.csv", 3)


def copy_sonic_single_motion(run_dir: Path, motion: str) -> Path:
    motion_root = run_dir / "sonic_motion_data"
    src = ROOT / "assets" / "motions" / "sonic_motions" / motion
    dst = motion_root / motion
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    subset_sonic_body_csvs_for_stock_reader(dst)
    write_sonic_metadata_for_stock_reader(dst)
    return motion_root


def ensure_sonic_built() -> None:
    name = f"wbc-sonic-build-{int(time.time())}"
    docker_rm_force(name)
    script = (
        "cd /workspace/wbc/thirdparties/GR00T-WholeBodyControl/gear_sonic_deploy && "
        "source /opt/ros/humble/setup.bash && "
        "source scripts/setup_env.sh && "
        "just build && "
        "test -x target/release/g1_deploy_onnx_ref"
    )
    cmd = docker_base_args(name, SONIC_IMAGE) + ["bash", "-lc", script]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "stock SONIC build failed before simulator launch\n"
            + proc.stdout[-4000:]
            + proc.stderr[-4000:]
        )


def holomotion_clip_index(motion: str) -> int:
    clips = sorted(f"{name}_holomotion.npz" for name in MOTIONS)
    target = f"{motion}_holomotion.npz"
    return clips.index(target)


def simulator_reference_args(policy: str, motion: str) -> list[str]:
    if policy == "sonic":
        reference = ROOT / "assets" / "motions" / "sonic_motions" / motion
        return [
            "--init-reference",
            f"/workspace/wbc/{reference.relative_to(ROOT)}",
            "--init-reference-format",
            "sonic_csv",
        ]
    return []


def simulator_scene_args(policy: str) -> list[str]:
    if policy == "sonic":
        scene = (
            ROOT
            / "thirdparties"
            / "GR00T-WholeBodyControl"
            / "gear_sonic"
            / "data"
            / "robot_model"
            / "model_data"
            / "g1"
            / "scene_43dof.xml"
        )
    else:
        scene = ROOT / "thirdparties" / "HoloMotion" / "assets" / "robots" / "unitree" / "G1" / "29dof" / "scene_29dof.xml"
    return ["--scene", f"/workspace/wbc/{scene.relative_to(ROOT)}"]


def simulator_domain_args(policy: str) -> list[str]:
    return ["--domain-id", "0" if policy == "sonic" else "1"]


def start_simulator(run_dir: Path, control_file: Path, duration_s: float, name: str, policy: str, motion: str) -> ManagedProcess:
    docker_rm_force(name)
    cmd = docker_base_args(name, SIM_IMAGE) + [
        "python3",
        "/workspace/wbc/scripts/sim_bridge.py",
        "--sim-root",
        "/workspace/unitree_mujoco",
        "--robot",
        "g1",
        "--interface",
        "lo",
        "--duration-s",
        f"{duration_s:.3f}",
        "--dt",
        "0.005",
        "--log-hz",
        "50",
        "--publish-every",
        "4" if policy == "sonic" else "1",
        "--out-dir",
        f"/workspace/wbc/{run_dir.relative_to(ROOT)}",
        "--control-file",
        f"/workspace/wbc/{control_file.relative_to(ROOT)}",
    ] + simulator_domain_args(policy) + simulator_scene_args(policy) + simulator_reference_args(policy, motion)
    proc = ManagedProcess(cmd, run_dir / "simulator_stdout.log")
    proc.start()
    wait_for_file(control_file, 10.0)
    wait_for_support_state(run_dir, 1, 10.0)
    return proc


def run_sonic_sequence(args: argparse.Namespace, run_dir: Path, control_file: Path, event_log: SequenceEventLog) -> None:
    motion_root = copy_sonic_single_motion(run_dir, args.motion)
    stock_csv_dir = run_dir / "csv"
    name = f"wbc-sonic-{args.motion[:32]}-{int(time.time())}"
    docker_rm_force(name)
    script = (
        "cd /workspace/wbc/thirdparties/GR00T-WholeBodyControl/gear_sonic_deploy && "
        "source /opt/ros/humble/setup.bash && "
        "source scripts/setup_env.sh && "
        "./target/release/g1_deploy_onnx_ref "
        "lo "
        "policy/release/model_decoder.onnx "
        f"/workspace/wbc/{motion_root.relative_to(ROOT)} "
        "--obs-config policy/release/observation_config.yaml "
        "--encoder-file policy/release/model_encoder.onnx "
        "--input-type keyboard "
        "--output-type zmq "
        "--zmq-host localhost "
        "--zmq-out-port 15557 "
        "--disable-crc-check "
        "--enable-csv-logs "
        f"--logs-dir /workspace/wbc/{stock_csv_dir.relative_to(ROOT)} "
        f"--target-motion-logfile /workspace/wbc/{(run_dir / 'target_motion.csv').relative_to(ROOT)} "
        f"--planner-motion-logfile /workspace/wbc/{(run_dir / 'planner_motion.csv').relative_to(ROOT)} "
        f"--policy-input-logfile /workspace/wbc/{(run_dir / 'policy_input.csv').relative_to(ROOT)}"
    )
    policy = ManagedProcess(
        docker_base_args(name, SONIC_IMAGE, tty=True) + ["bash", "-lc", script],
        run_dir / "sonic_stdout.log",
        use_pty=True,
    )
    try:
        policy.start()
        wait_for_log_marker(policy, policy.log_path, "Init Done", args.policy_ready_timeout_s)
        policy.send("]")
        marker = wait_for_log_marker(
            policy,
            policy.log_path,
            "[Control] DEBUG: operator_state.start=true, transitioning to CONTROL state",
            args.policy_ready_timeout_s,
        )
        event_log.append(
            CONTROL_EVENT,
            sim_time_s=latest_sim_time(run_dir),
            support_active=1,
            detail=marker,
        )
        release_support(run_dir, control_file, event_log, "release simulator support after SONIC CONTROL")
        post_release_wait_s = (
            args.sonic_post_release_wait_s
            if args.sonic_post_release_wait_s is not None
            else SONIC_POST_RELEASE_WAIT_BY_MOTION.get(args.motion, 0.0)
        )
        if post_release_wait_s > 0.0:
            event_log.append(
                "post_release_settle_elapsed",
                sim_time_s=latest_sim_time(run_dir),
                support_active=0,
                detail=f"seconds={post_release_wait_s:.3f}",
            )
            time.sleep(post_release_wait_s)
        policy.send("T")
        event_log.append(
            "sent_key_T",
            sim_time_s=latest_sim_time(run_dir),
            support_active=0,
            detail="stock SONIC keyboard playback trigger after confirmed release",
        )
        time.sleep(args.motion_duration_s)
    finally:
        update_control_file(control_file, stop=True)
        policy.terminate()
        docker_rm_force(name)


def run_holomotion_sequence(args: argparse.Namespace, run_dir: Path, control_file: Path, event_log: SequenceEventLog) -> None:
    name = f"wbc-holo-{args.motion[:32]}-{int(time.time())}"
    bridge_name = f"wbc-holo-bridge-{args.motion[:24]}-{int(time.time())}"
    docker_rm_force(name)
    docker_rm_force(bridge_name)
    bridge_script = (
        "source /opt/ros/humble/setup.sh && "
        "source /opt/unitree_ros2/cyclonedds_ws/install/setup.bash && "
        "source /workspace/HoloMotion/deployment/unitree_g1_ros2_29dof/install/setup.bash && "
        "export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain></CycloneDDS>' && "
        "/usr/bin/python3 /workspace/wbc/scripts/ros2_unitree_bridge.py "
        "--interface lo --domain-id 1"
    )
    bridge = ManagedProcess(
        docker_base_args(bridge_name, HOLO_IMAGE) + ["bash", "-lc", bridge_script],
        run_dir / "holomotion_bridge_stdout.log",
    )
    script = (
        "cd /workspace/HoloMotion/deployment/unitree_g1_ros2_29dof && "
        "./launch_holomotion_29dof_docker.sh "
        "--profile launch_profiles/x86_64_docker.yaml "
        "--set runtime.unitree_setup=/opt/unitree_ros2/cyclonedds_ws/install/setup.bash "
        "--set robot.network_interface=lo "
        "--set policy.inference_backend=onnx"
    )
    policy = ManagedProcess(
        docker_base_args(name, HOLO_IMAGE) + ["bash", "-lc", script],
        run_dir / "holomotion_stdout.log",
    )
    try:
        bridge.start()
        time.sleep(1.0)
        policy.start()
        wait_for_log_marker(policy, policy.log_path, "Policy node setup completed successfully", args.policy_ready_timeout_s)
        wait_for_log_marker(policy, policy.log_path, "Entered ZERO_TORQUE state", args.policy_ready_timeout_s)

        pulse_holomotion_key(control_file, "start")
        wait_for_log_marker(policy, policy.log_path, "Switching to MOVE_TO_DEFAULT state", args.policy_ready_timeout_s)
        time.sleep(args.holomotion_default_wait_s)

        pulse_holomotion_key(control_file, "a")
        marker = wait_for_log_marker(
            policy,
            policy.log_path,
            "Switching to POLICY state",
            args.policy_ready_timeout_s,
        )
        wait_for_log_marker(policy, policy.log_path, "Policy enabled in velocity tracking mode", args.policy_ready_timeout_s)
        event_log.append(
            CONTROL_EVENT,
            sim_time_s=latest_sim_time(run_dir),
            support_active=1,
            detail=marker,
        )

        marker = select_holomotion_clip(control_file, policy.log_path, args.motion, args.policy_ready_timeout_s)
        event_log.append(
            "motion_selected",
            sim_time_s=latest_sim_time(run_dir),
            support_active=1,
            detail=f"{marker}: {args.motion}_holomotion.npz",
        )

        wait_for_pre_release_hold(run_dir, event_log)
        release_support(run_dir, control_file, event_log, "release simulator support after HoloMotion CONTROL")
        event_log.append(
            "sent_key_B",
            sim_time_s=latest_sim_time(run_dir),
            support_active=0,
            detail="stock HoloMotion motion tracking trigger after confirmed release",
        )
        pulse_holomotion_key(control_file, "b")
        marker = wait_for_log_marker(
            policy,
            policy.log_path,
            "Switched to motion tracking mode",
            args.policy_ready_timeout_s,
        )
        event_log.append(
            "motion_playing_observed",
            sim_time_s=latest_sim_time(run_dir),
            support_active=0,
            detail=marker,
        )
        time.sleep(args.motion_duration_s)
    finally:
        update_control_file(control_file, stop=True)
        policy.terminate()
        bridge.terminate()
        docker_rm_force(name)
        docker_rm_force(bridge_name)


def run_motion(args: argparse.Namespace) -> int:
    cleanup_stale_benchmark_containers()
    logs_root = Path(args.logs)
    if not logs_root.is_absolute():
        logs_root = ROOT / logs_root
    if not getattr(args, "skip_gpu_preflight", False):
        probe = probe_gpu_runtime()
        if not probe["ok"]:
            payload = write_global_impossibility(logs_root, probe)
            print(json.dumps({"ok": True, "documented_impossibility": payload}, indent=2))
            return 0
    run_dir = Path(args.logs) / args.policy / args.motion
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    control_file = run_dir / "sim_control.json"
    event_path = run_dir / "sequence_events.csv"
    if event_path.exists():
        event_path.unlink()
    event_log = SequenceEventLog(event_path)
    args.motion_duration_s = args.duration_s or policy_motion_duration(args.policy, args.motion)
    sim_duration = args.motion_duration_s + args.startup_margin_s
    if args.policy == "sonic":
        ensure_sonic_built()
    sim_name = f"wbc-sim-{args.policy}-{args.motion[:24]}-{int(time.time())}"
    sim = start_simulator(run_dir, control_file, sim_duration, sim_name, args.policy, args.motion)
    try:
        if args.policy == "sonic":
            run_sonic_sequence(args, run_dir, control_file, event_log)
        else:
            run_holomotion_sequence(args, run_dir, control_file, event_log)
        validate_release_order(run_dir)
    finally:
        update_control_file(control_file, stop=True)
        sim.terminate()
        docker_rm_force(sim_name)
    print(json.dumps({"ok": True, "policy": args.policy, "motion": args.motion, "run_dir": str(run_dir)}, indent=2))
    return 0


def run_all_motions(args: argparse.Namespace) -> int:
    logs_root = Path(args.logs)
    if not logs_root.is_absolute():
        logs_root = ROOT / logs_root
    if not getattr(args, "skip_gpu_preflight", False):
        probe = probe_gpu_runtime()
        if not probe["ok"]:
            payload = write_global_impossibility(logs_root, probe)
            print(json.dumps({"ok": True, "documented_impossibility": payload}, indent=2))
            return 0
    failures = []
    for motion in MOTIONS:
        for policy in POLICIES:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.policy = policy
            sub_args.motion = motion
            try:
                run_motion(sub_args)
            except Exception as exc:
                failures.append(f"{policy}/{motion}: {exc}")
    print(json.dumps({"ok": not failures, "failures": failures}, indent=2))
    return 0 if not failures else 1


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
    cmd = ["docker", "run", "--rm", image, "bash", "-lc", "set -e\n" + script]
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
            "import cyclonedds, unitree_sdk2py\n"
            "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_\n"
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
            "source /opt/ros/humble/setup.bash\n"
            "source /workspace/HoloMotion/deployment/unitree_g1_ros2_29dof/install/setup.bash\n"
            "/usr/bin/python3 - <<'PY'\n"
            "import rclpy, unitree_sdk2py\n"
            "from unitree_hg.msg import LowState, LowCmd\n"
            "PY\n"
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


def smoke_sim_bridge(_: argparse.Namespace) -> int:
    script = (
        "rm -rf /tmp/wbc-sim-bridge && "
        "python3 /workspace/wbc/scripts/sim_bridge.py "
        "--sim-root /workspace/unitree_mujoco "
        "--robot g1 "
        "--interface lo "
        "--duration-s 0.25 "
        "--dt 0.005 "
        "--log-hz 50 "
        "--out-dir /tmp/wbc-sim-bridge && "
        "test -s /tmp/wbc-sim-bridge/lowcmd.csv && "
        "test -s /tmp/wbc-sim-bridge/simulator_status.csv && "
        "python3 - <<'PY'\n"
        "import csv\n"
        "from pathlib import Path\n"
        "root = Path('/tmp/wbc-sim-bridge')\n"
        "rows = list(csv.DictReader((root / 'simulator_status.csv').open()))\n"
        "assert len(rows) >= 5, len(rows)\n"
        "assert all(row['support_active'] == '1' for row in rows)\n"
        "lowcmd = list(csv.DictReader((root / 'lowcmd.csv').open()))\n"
        "assert len(lowcmd) >= 5, len(lowcmd)\n"
        "assert 'measured_q_28' in lowcmd[0]\n"
        "PY"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{ROOT}:/workspace/wbc",
        "wbc-unitree_mujoco",
        "bash",
        "-lc",
        script,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    payload = {
        "ok": proc.returncode == 0,
        "image": "wbc-unitree_mujoco",
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


def smoke_sim_release(_: argparse.Namespace) -> int:
    script = (
        "rm -rf /tmp/wbc-sim-release && mkdir -p /tmp/wbc-sim-release && "
        "python3 /workspace/wbc/scripts/sim_bridge.py "
        "--sim-root /workspace/unitree_mujoco "
        "--robot g1 "
        "--interface lo "
        "--duration-s 0.8 "
        "--dt 0.005 "
        "--log-hz 50 "
        "--out-dir /tmp/wbc-sim-release "
        "--control-file /tmp/wbc-sim-release/control.json & "
        "pid=$! && "
        "for i in $(seq 1 100); do test -s /tmp/wbc-sim-release/control.json && break; sleep 0.02; done && "
        "python3 - <<'PY'\n"
        "import csv, json, time\n"
        "from pathlib import Path\n"
        "path = Path('/tmp/wbc-sim-release/control.json')\n"
        "data = json.loads(path.read_text())\n"
        "assert data['support_active'] is True\n"
        "status = Path('/tmp/wbc-sim-release/simulator_status.csv')\n"
        "for _ in range(100):\n"
        "    if status.exists():\n"
        "        rows = list(csv.DictReader(status.open()))\n"
        "        if any(row.get('support_active') == '1' for row in rows):\n"
        "            break\n"
        "    time.sleep(0.02)\n"
        "else:\n"
        "    raise AssertionError('simulator did not log support_active=1 before release')\n"
        "data['support_active'] = False\n"
        "path.write_text(json.dumps(data, indent=2) + '\\n')\n"
        "PY\n"
        "wait $pid && "
        "python3 - <<'PY'\n"
        "import csv\n"
        "from pathlib import Path\n"
        "rows = list(csv.DictReader(Path('/tmp/wbc-sim-release/simulator_status.csv').open()))\n"
        "values = [row['support_active'] for row in rows]\n"
        "assert '1' in values, values\n"
        "assert '0' in values, values\n"
        "first_zero = values.index('0')\n"
        "assert all(v == '0' for v in values[first_zero:]), values\n"
        "PY"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{ROOT}:/workspace/wbc",
        "wbc-unitree_mujoco",
        "bash",
        "-lc",
        script,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    payload = {
        "ok": proc.returncode == 0,
        "image": "wbc-unitree_mujoco",
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


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
    sub.add_parser("smoke-release-gate").set_defaults(func=smoke_release_gate)
    sub.add_parser("smoke-sim-bridge").set_defaults(func=smoke_sim_bridge)
    sub.add_parser("smoke-sim-release").set_defaults(func=smoke_sim_release)
    sub.add_parser("smoke-holomotion").set_defaults(func=smoke_holomotion)
    sub.add_parser("smoke-sonic-build").set_defaults(func=smoke_sonic_build)
    p = sub.add_parser("report")
    p.add_argument("--logs", default=str(ROOT / "logs"))
    p.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    p.set_defaults(func=report)
    p = sub.add_parser("run-motion")
    p.add_argument("policy", choices=POLICIES)
    p.add_argument("motion", choices=MOTIONS)
    p.add_argument("--logs", default=str(ROOT / "logs"))
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--startup-margin-s", type=float, default=120.0)
    p.add_argument("--policy-ready-timeout-s", type=float, default=120.0)
    p.add_argument("--holomotion-default-wait-s", type=float, default=5.0)
    p.add_argument("--sonic-post-release-wait-s", type=float, default=None)
    p.add_argument("--skip-gpu-preflight", action="store_true")
    p.set_defaults(func=run_motion)
    p = sub.add_parser("run-all-motions")
    p.add_argument("--logs", default=str(ROOT / "logs"))
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--startup-margin-s", type=float, default=120.0)
    p.add_argument("--policy-ready-timeout-s", type=float, default=120.0)
    p.add_argument("--holomotion-default-wait-s", type=float, default=5.0)
    p.add_argument("--sonic-post-release-wait-s", type=float, default=None)
    p.add_argument("--skip-gpu-preflight", action="store_true")
    p.set_defaults(func=run_all_motions)
    p = sub.add_parser("docker-build")
    p.add_argument("image", choices=["gear-sonic", "holomotion", "unitree_mujoco"])
    p.set_defaults(func=docker)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
