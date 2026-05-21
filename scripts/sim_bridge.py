#!/usr/bin/env python3
"""Headless Unitree MuJoCo bridge for benchmark orchestration.

This script is benchmark-owned glue, kept flat under scripts/. It reuses the
upstream Unitree MuJoCo Python bridge for DDS LowState/LowCmd transport and
adds only headless stepping, root elastic-band control, and benchmark logs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct
import sys
import time
from typing import Any


DEFAULT_SIM_ROOT = Path("/workspace/unitree_mujoco")


def load_upstream_bridge(sim_root: Path, robot: str):
    bridge_dir = sim_root / "simulate_python"
    sys.path.insert(0, str(bridge_dir))
    import config  # type: ignore

    config.ROBOT = robot
    config.USE_JOYSTICK = 0
    config.PRINT_SCENE_INFORMATION = False
    config.ENABLE_ELASTIC_BAND = True

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore
    from unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge  # type: ignore

    return ChannelFactoryInitialize, ElasticBand, UnitreeSdk2Bridge


def set_wireless_remote(low_state: Any, keys: int, lx: float, ly: float, rx: float, ry: float) -> None:
    data = bytearray(40)
    data[2] = keys & 0xFF
    data[3] = (keys >> 8) & 0xFF
    data[4:8] = struct.pack("f", float(lx))
    data[8:12] = struct.pack("f", float(rx))
    data[12:16] = struct.pack("f", float(ry))
    data[20:24] = struct.pack("f", float(ly))
    for index, value in enumerate(data):
        low_state.wireless_remote[index] = value


def read_control(path: Path, previous: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return previous
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return previous
    if not isinstance(raw, dict):
        return previous
    control = dict(previous)
    for key in ("support_active", "wireless_keys", "lx", "ly", "rx", "ry", "stop"):
        if key in raw:
            control[key] = raw[key]
    return control


def write_control(path: Path, control: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(control, indent=2) + "\n")


def open_logs(out_dir: Path, num_motor: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    status_f = (out_dir / "simulator_status.csv").open("w", newline="")
    lowcmd_f = (out_dir / "lowcmd.csv").open("w", newline="")

    status_writer = csv.DictWriter(
        status_f,
        fieldnames=[
            "monotonic_s",
            "wall_time_s",
            "sim_time_s",
            "support_active",
            "wireless_keys",
            "base_z",
        ],
    )
    status_writer.writeheader()

    lowcmd_fields = (
        ["time_s", "sim_time_s"]
        + [f"measured_q_{i}" for i in range(num_motor)]
        + [f"measured_dq_{i}" for i in range(num_motor)]
        + [f"tau_est_{i}" for i in range(num_motor)]
    )
    lowcmd_writer = csv.DictWriter(lowcmd_f, fieldnames=lowcmd_fields)
    lowcmd_writer.writeheader()
    return status_f, lowcmd_f, status_writer, lowcmd_writer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", default=str(DEFAULT_SIM_ROOT))
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--scene", default="")
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--log-hz", type=float, default=50.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--control-file", default="")
    parser.add_argument("--support-active", type=int, default=1)
    args = parser.parse_args()

    import mujoco

    sim_root = Path(args.sim_root)
    scene = Path(args.scene) if args.scene else sim_root / "unitree_robots" / args.robot / "scene.xml"
    out_dir = Path(args.out_dir)
    control_path = Path(args.control_file) if args.control_file else out_dir / "sim_control.json"
    control = {
        "support_active": bool(args.support_active),
        "wireless_keys": 0,
        "lx": 0.0,
        "ly": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "stop": False,
    }
    write_control(control_path, control)

    ChannelFactoryInitialize, ElasticBand, UnitreeSdk2Bridge = load_upstream_bridge(sim_root, args.robot)
    ChannelFactoryInitialize(args.domain_id, args.interface)

    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    bridge = UnitreeSdk2Bridge(model, data)
    elastic_band = ElasticBand()
    band_body = model.body("torso_link").id if args.robot in {"g1", "h1"} else model.body("base_link").id

    status_f, lowcmd_f, status_writer, lowcmd_writer = open_logs(out_dir, model.nu)
    started_wall = time.time()
    started_mono = time.monotonic()
    next_log_t = 0.0
    log_dt = 1.0 / args.log_hz

    try:
        while True:
            elapsed = time.monotonic() - started_mono
            if elapsed >= args.duration_s:
                break
            control = read_control(control_path, control)
            if bool(control.get("stop", False)):
                break

            keys = int(control.get("wireless_keys", 0))
            support_active = bool(control.get("support_active", True))
            set_wireless_remote(
                bridge.low_state,
                keys,
                float(control.get("lx", 0.0)),
                float(control.get("ly", 0.0)),
                float(control.get("rx", 0.0)),
                float(control.get("ry", 0.0)),
            )

            data.xfrc_applied[:] = 0.0
            if support_active:
                data.xfrc_applied[band_body, :3] = elastic_band.Advance(data.qpos[:3], data.qvel[:3])
            mujoco.mj_step(model, data)

            if data.time + 1e-9 >= next_log_t:
                now_wall = started_wall + elapsed
                status_writer.writerow({
                    "monotonic_s": f"{elapsed:.6f}",
                    "wall_time_s": f"{now_wall:.6f}",
                    "sim_time_s": f"{data.time:.6f}",
                    "support_active": int(support_active),
                    "wireless_keys": keys,
                    "base_z": f"{float(data.qpos[2]):.6f}",
                })
                row = {"time_s": f"{elapsed:.6f}", "sim_time_s": f"{data.time:.6f}"}
                for i in range(model.nu):
                    row[f"measured_q_{i}"] = f"{float(data.sensordata[i]):.9f}"
                    row[f"measured_dq_{i}"] = f"{float(data.sensordata[i + model.nu]):.9f}"
                    row[f"tau_est_{i}"] = f"{float(data.sensordata[i + 2 * model.nu]):.9f}"
                lowcmd_writer.writerow(row)
                next_log_t += log_dt

            sleep_s = args.dt - (time.monotonic() - started_mono - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        status_f.close()
        lowcmd_f.close()
        if hasattr(bridge.lowStateThread, "Stop"):
            bridge.lowStateThread.Stop()
        if hasattr(bridge.HighStateThread, "Stop"):
            bridge.HighStateThread.Stop()
        if hasattr(bridge.WirelessControllerThread, "Stop"):
            bridge.WirelessControllerThread.Stop()

    (out_dir / "sim_bridge_summary.json").write_text(
        json.dumps({
            "scene": str(scene),
            "robot": args.robot,
            "duration_s": args.duration_s,
            "dt": args.dt,
            "log_hz": args.log_hz,
            "control_file": str(control_path),
        }, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
