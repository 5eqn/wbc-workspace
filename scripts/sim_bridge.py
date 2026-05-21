#!/usr/bin/env python3
"""Headless Unitree MuJoCo bridge for benchmark orchestration.

This script is benchmark-owned glue, kept flat under scripts/. It reuses the
upstream Unitree MuJoCo Python bridge for DDS LowState/LowCmd transport and
adds only headless stepping, root-only pre-control support, and benchmark logs.
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
BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
DEFAULT_ANGLES = [
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
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


def read_csv_frame(path: Path, frame: int, width: int) -> list[float]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row_idx, row in enumerate(reader):
            if row_idx != frame:
                continue
            values = [float(value) for value in row if value != ""]
            if len(values) < width:
                raise ValueError(f"{path} frame {frame} has {len(values)} values, expected {width}")
            return values[:width]
    raise ValueError(f"{path} does not contain frame {frame}")


def first_body_frame(array: Any, frame: int, width: int):
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 3:
        values = values[frame, 0, :]
    elif values.ndim == 2:
        values = values[frame, :]
    else:
        values = values.reshape(values.shape[0], -1)[frame, :]
    if values.size < width:
        raise ValueError(f"reference array has {values.size} values, expected {width}")
    return values[:width]


def snap_root_to_ground(model: Any, data: Any, mujoco: Any, ground_clearance: float) -> None:
    robot_geom_z = [
        float(data.geom_xpos[i, 2])
        for i in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) != "floor"
    ]
    if robot_geom_z:
        data.qpos[2] -= min(robot_geom_z) - ground_clearance
        mujoco.mj_forward(model, data)


def set_named_joint_qpos(model: Any, data: Any, mujoco: Any, q: list[float]) -> None:
    for i, joint_name in enumerate(BODY_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Missing joint in model: {joint_name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(q[i])


def apply_initial_reference(
    model: Any,
    data: Any,
    mujoco: Any,
    reference: Path,
    ref_format: str,
    frame: int,
    ground_clearance: float,
) -> dict[str, Any]:
    import numpy as np

    if ref_format == "sonic_csv":
        joint_policy = read_csv_frame(reference / "joint_pos.csv", frame, 29)
        joint_hw = [joint_policy[i] for i in HARDWARE_FROM_SONIC_POLICY]
        root_pos = read_csv_frame(reference / "body_pos.csv", frame, 42)[:3]
        root_quat = read_csv_frame(reference / "body_quat.csv", frame, 56)[:4]
    elif ref_format == "holomotion_npz":
        loaded = np.load(reference, allow_pickle=False)
        joint_hw = np.asarray(loaded["ref_dof_pos"], dtype=np.float64)[frame, :29].tolist()
        root_pos = first_body_frame(loaded["ref_global_translation"], frame, 3).tolist()
        root_quat = first_body_frame(loaded["ref_global_rotation_quat"], frame, 4).tolist()
    else:
        raise ValueError(f"unsupported reference format: {ref_format}")

    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat
    set_named_joint_qpos(model, data, mujoco, joint_hw)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    snap_root_to_ground(model, data, mujoco, ground_clearance)
    return {
        "reference": str(reference),
        "reference_format": ref_format,
        "reference_frame": frame,
        "root_z": float(data.qpos[2]),
    }


def load_upstream_bridge(sim_root: Path, robot: str):
    bridge_dir = sim_root / "simulate_python"
    sys.path.insert(0, str(bridge_dir))
    import config  # type: ignore

    config.ROBOT = robot
    config.USE_JOYSTICK = 0
    config.PRINT_SCENE_INFORMATION = False
    config.ENABLE_ELASTIC_BAND = True

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge  # type: ignore

    return ChannelFactoryInitialize, UnitreeSdk2Bridge


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
    for key in ("support_active", "wireless_keys", "wireless_keys_once", "lx", "ly", "rx", "ry", "stop"):
        if key in raw:
            control[key] = raw[key]
    return control


def write_control(path: Path, control: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(control, indent=2) + "\n")
    path.chmod(0o666)


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
    parser.add_argument("--support-height", type=float, default=0.75)
    parser.add_argument("--init-reference", default="")
    parser.add_argument("--init-reference-format", choices=["sonic_csv", "holomotion_npz"], default="sonic_csv")
    parser.add_argument("--init-reference-frame", type=int, default=0)
    parser.add_argument("--ground-clearance", type=float, default=0.001)
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

    ChannelFactoryInitialize, UnitreeSdk2Bridge = load_upstream_bridge(sim_root, args.robot)
    ChannelFactoryInitialize(args.domain_id, args.interface)

    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    data.qpos[0:3] = [0.0, 0.0, float(args.support_height)]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    set_named_joint_qpos(model, data, mujoco, DEFAULT_ANGLES)
    mujoco.mj_forward(model, data)
    init_summary = None
    if args.init_reference:
        init_summary = apply_initial_reference(
            model,
            data,
            mujoco,
            Path(args.init_reference),
            args.init_reference_format,
            args.init_reference_frame,
            args.ground_clearance,
        )
        print(
            "[sim_bridge] initialized support pose from "
            f"{init_summary['reference']} frame={args.init_reference_frame} "
            f"root_z={init_summary['root_z']:.3f}",
            flush=True,
        )
    bridge = UnitreeSdk2Bridge(model, data)
    if hasattr(bridge.lowStateThread, "Stop"):
        bridge.lowStateThread.Stop()
    support_root_qpos = data.qpos[:7].copy()
    support_root_qvel = data.qvel[:6].copy()

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

            one_shot_keys = int(control.pop("wireless_keys_once", 0) or 0)
            if one_shot_keys:
                write_control(control_path, control)
            keys = one_shot_keys or int(control.get("wireless_keys", 0))
            support_active = bool(control.get("support_active", True))
            set_wireless_remote(
                bridge.low_state,
                keys,
                float(control.get("lx", 0.0)),
                float(control.get("ly", 0.0)),
                float(control.get("rx", 0.0)),
                float(control.get("ry", 0.0)),
            )
            bridge.PublishLowState()

            if support_active:
                data.qpos[:7] = support_root_qpos
                data.qvel[:6] = support_root_qvel
                mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            if support_active:
                data.qpos[:7] = support_root_qpos
                data.qvel[:6] = support_root_qvel
                mujoco.mj_forward(model, data)

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
                status_f.flush()
                lowcmd_f.flush()
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
            "support_height": args.support_height,
            "init_reference": init_summary,
            "control_file": str(control_path),
        }, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
