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
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

try:
    from sim_bridge import BODY_JOINT_NAMES
except ModuleNotFoundError:
    from scripts.sim_bridge import BODY_JOINT_NAMES

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
COMMON_SIM_SCENE = (
    ROOT
    / "thirdparties"
    / "GR00T-WholeBodyControl"
    / "decoupled_wbc"
    / "control"
    / "robot_model"
    / "model_data"
    / "g1"
    / "scene_29dof.xml"
)
SIM_IMAGE = "wbc-unitree_mujoco"
SONIC_IMAGE = "wbc-gear-sonic"
HOLO_IMAGE = "wbc-holomotion"
GLOBAL_IMPOSSIBILITY = "impossibility.json"
REPORT_OUTPUT_FILES = [
    "metrics_summary.json",
    "phase_time_proof.json",
    "rmse_summary.csv",
    "per_joint_rmse.csv",
    "release_validation.json",
    "single_robot_interface.json",
    "comparison_videos.json",
    "report.md",
    GLOBAL_IMPOSSIBILITY,
]
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
SONIC_POST_RELEASE_WAIT_S = 2.0
HOLOMOTION_POST_RELEASE_VELOCITY_HOLD_S = 5.0

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


def write_sequence_events(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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
        next(reader, None)
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


def load_status_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "simulator_status.csv"
    if not path.exists():
        raise FileNotFoundError(f"{run_dir}: missing simulator_status.csv")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: empty simulator status log")
    return rows


def first_status_time(rows: list[dict[str, str]], predicate) -> float | None:
    for row in rows:
        if predicate(row):
            return float(row["sim_time_s"])
    return None


def status_window(rows: list[dict[str, str]], start_s: float, end_s: float) -> list[dict[str, str]]:
    return [row for row in rows if start_s <= float(row["sim_time_s"]) <= end_s]


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


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


def wait_for_cmd_active_state(run_dir: Path, cmd_active: int, timeout_s: float) -> dict[str, str]:
    deadline = time.monotonic() + timeout_s
    expected = str(int(cmd_active))
    while time.monotonic() < deadline:
        row = latest_sim_status(run_dir)
        if row and row.get("cmd_active") == expected:
            return row
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for simulator cmd_active={expected}")


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


def wait_for_motion_window(sim: ManagedProcess, duration_s: float, event_log: SequenceEventLog, run_dir: Path) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        code = sim.poll()
        if code is not None:
            event_log.append(
                "simulator_ended_during_motion",
                sim_time_s=latest_sim_time(run_dir),
                detail=f"simulator process exited with code {code}",
            )
            return
        time.sleep(min(0.10, max(0.0, deadline - time.monotonic())))


def wait_for_sim_window(sim: ManagedProcess, duration_s: float) -> int | None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        code = sim.poll()
        if code is not None:
            return code
        time.sleep(min(0.10, max(0.0, deadline - time.monotonic())))
    return sim.poll()


def policy_motion_duration(policy: str, motion: str) -> float:
    del policy
    path = ROOT / "assets" / "motions" / "sonic_motions" / motion / "joint_pos.csv"
    with path.open(newline="") as f:
        frames = max(0, sum(1 for _ in f) - 1)
    return float(frames) / 50.0


def logged_scene_path(replay: dict) -> Path:
    scene = str(replay["summary"].get("scene", ""))
    if not scene:
        raise ValueError(f"{replay['path']}: sim_bridge_summary.json has no scene")
    if scene.startswith("/workspace/wbc/"):
        return ROOT / scene.removeprefix("/workspace/wbc/")
    return Path(scene)


def git_output(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={cwd}", *args], cwd=cwd, text=True
    ).strip()


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
