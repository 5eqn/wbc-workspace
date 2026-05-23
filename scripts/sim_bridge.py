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
HOLD_KP = 120.0
HOLD_KD = 6.0
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
NEUTRAL_JOINT_Q = [
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


def set_named_joint_qpos(model: Any, data: Any, mujoco: Any, q: list[float]) -> None:
    for i, joint_name in enumerate(BODY_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Missing joint in model: {joint_name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(q[i])


def load_upstream_bridge(sim_root: Path, robot: str):
    bridge_dir = sim_root / "simulate_python"
    sys.path.insert(0, str(bridge_dir))
    import config  # type: ignore

    config.ROBOT = robot
    config.USE_JOYSTICK = 0
    config.PRINT_SCENE_INFORMATION = False
    config.ENABLE_ELASTIC_BAND = True

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # type: ignore
    from unitree_sdk2py.idl.default import (  # type: ignore
        unitree_go_msg_dds__SportModeState_,
        unitree_hg_msg_dds__BmsState_,
        unitree_hg_msg_dds__IMUState_,
        unitree_hg_msg_dds__LowCmd_,
        unitree_hg_msg_dds__LowState_,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_  # type: ignore
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_, IMUState_, LowCmd_, LowState_  # type: ignore

    return {
        "ChannelFactoryInitialize": ChannelFactoryInitialize,
        "ChannelPublisher": ChannelPublisher,
        "ChannelSubscriber": ChannelSubscriber,
        "LowCmd": LowCmd_,
        "LowState": LowState_,
        "SportModeState": SportModeState_,
        "BmsState": BmsState_,
        "IMUState": IMUState_,
        "LowCmdDefault": unitree_hg_msg_dds__LowCmd_,
        "LowStateDefault": unitree_hg_msg_dds__LowState_,
        "SportModeStateDefault": unitree_go_msg_dds__SportModeState_,
        "BmsStateDefault": unitree_hg_msg_dds__BmsState_,
        "IMUStateDefault": unitree_hg_msg_dds__IMUState_,
    }


def quat_to_rpy_wxyz(q: Any) -> tuple[float, float, float]:
    import math

    qw, qx, qy, qz = [float(v) for v in q]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class UnitreeG1Bridge:
    def __init__(self, model: Any, data: Any, mujoco: Any, sdk: dict[str, Any]):
        import threading

        self.model = model
        self.data = data
        self.mujoco = mujoco
        self.num_motor = len(BODY_JOINT_NAMES)
        self.body_qpos_adrs: list[int] = []
        self.body_qvel_adrs: list[int] = []
        self.body_actuator_ids: list[int] = []
        for joint_name in BODY_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                joint_name.removesuffix("_joint"),
            )
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"Missing G1 body joint/actuator in scene: {joint_name}")
            self.body_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
            self.body_qvel_adrs.append(int(model.jnt_dofadr[joint_id]))
            self.body_actuator_ids.append(actuator_id)

        self.latest_cmd = sdk["LowCmdDefault"]()
        self.received_cmd = False
        self.cmd_count = 0
        self.last_cmd_monotonic: float | None = None
        self.cmd_lock = threading.Lock()
        self.low_state = sdk["LowStateDefault"]()
        self.low_state.mode_machine = 5

        self.low_state_pub = sdk["ChannelPublisher"]("rt/lowstate", sdk["LowState"])
        self.low_state_pub.Init()
        self.low_cmd_sub = sdk["ChannelSubscriber"]("rt/lowcmd", sdk["LowCmd"])
        self.low_cmd_sub.Init(self.low_cmd_handler, 10)

        self.sport_state = sdk["SportModeStateDefault"]()
        self.sport_state_pub = sdk["ChannelPublisher"]("rt/sportmodestate", sdk["SportModeState"])
        self.sport_state_pub.Init()
        self.secondary_imu = sdk["IMUStateDefault"]()
        self.secondary_imu_pub = sdk["ChannelPublisher"]("rt/secondary_imu", sdk["IMUState"])
        self.secondary_imu_pub.Init()
        self.bms_state = sdk["BmsStateDefault"]()
        self.bms_state.soc = 100
        self.bms_pub = sdk["ChannelPublisher"]("rt/lf/bmsstate", sdk["BmsState"])
        self.bms_pub.Init()
        self.torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    def body_q(self):
        import numpy as np

        return np.asarray([self.data.qpos[adr] for adr in self.body_qpos_adrs], dtype=np.float64)

    def body_dq(self):
        import numpy as np

        return np.asarray([self.data.qvel[adr] for adr in self.body_qvel_adrs], dtype=np.float64)

    def body_tau_est(self):
        import numpy as np

        return np.asarray([self.data.actuator_force[i] for i in self.body_actuator_ids], dtype=np.float64)

    def low_cmd_handler(self, msg: Any) -> None:
        with self.cmd_lock:
            self.latest_cmd = msg
            self.received_cmd = True
            self.cmd_count += 1
            self.last_cmd_monotonic = time.monotonic()

    def apply_control(self, precontrol_hold: bool, cmd_timeout_s: float) -> bool:
        q = self.body_q()
        dq = self.body_dq()
        self.data.ctrl[:] = 0.0
        with self.cmd_lock:
            cmd = self.latest_cmd
            received = self.received_cmd
            last_cmd_monotonic = self.last_cmd_monotonic
        cmd_active = (
            received
            and last_cmd_monotonic is not None
            and time.monotonic() - last_cmd_monotonic <= cmd_timeout_s
        )
        if cmd_active:
            for i in range(self.num_motor):
                motor = cmd.motor_cmd[i]
                self.data.ctrl[self.body_actuator_ids[i]] = (
                    motor.tau + motor.kp * (motor.q - q[i]) + motor.kd * (motor.dq - dq[i])
                )
        elif precontrol_hold:
            for i in range(self.num_motor):
                self.data.ctrl[self.body_actuator_ids[i]] = HOLD_KP * (NEUTRAL_JOINT_Q[i] - q[i]) - HOLD_KD * dq[i]
        return cmd_active

    def publish(self) -> None:
        import numpy as np

        body_q = self.body_q()
        body_dq = self.body_dq()
        body_tau = self.body_tau_est()
        for i in range(self.num_motor):
            self.low_state.motor_state[i].q = float(body_q[i])
            self.low_state.motor_state[i].dq = float(body_dq[i])
            self.low_state.motor_state[i].tau_est = float(body_tau[i])

        base_quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        self.low_state.imu_state.quaternion[:] = [float(v) for v in base_quat]
        self.low_state.imu_state.rpy[:] = list(quat_to_rpy_wxyz(base_quat))
        self.low_state.imu_state.gyroscope[:] = [float(v) for v in self.data.qvel[3:6]]
        self.low_state.imu_state.accelerometer[:] = [float(v) for v in self.data.qacc[0:3]]
        self.low_state.tick = int(round(float(self.data.time) * 1000.0))
        self.low_state_pub.Write(self.low_state)

        self.sport_state.position[:] = [float(v) for v in self.data.qpos[0:3]]
        self.sport_state.velocity[:] = [float(v) for v in self.data.qvel[0:3]]
        self.sport_state_pub.Write(self.sport_state)
        self.bms_pub.Write(self.bms_state)

        if self.torso_body_id >= 0:
            torso_quat = np.asarray(self.data.xquat[self.torso_body_id], dtype=np.float64)
            torso_vel = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_objectVelocity(
                self.model,
                self.data,
                self.mujoco.mjtObj.mjOBJ_BODY,
                self.torso_body_id,
                torso_vel,
                1,
            )
            self.secondary_imu.quaternion[:] = [float(v) for v in torso_quat]
            self.secondary_imu.rpy[:] = list(quat_to_rpy_wxyz(torso_quat))
            self.secondary_imu.gyroscope[:] = [float(v) for v in torso_vel[0:3]]
            self.secondary_imu.accelerometer[:] = [0.0, 0.0, 0.0]
            self.secondary_imu_pub.Write(self.secondary_imu)


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


def open_logs(out_dir: Path, num_motor: int, nq: int, nv: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    status_f = (out_dir / "simulator_status.csv").open("w", newline="")
    lowcmd_f = (out_dir / "lowcmd.csv").open("w", newline="")
    replay_f = (out_dir / "replay.csv").open("w", newline="")

    status_writer = csv.DictWriter(
        status_f,
        fieldnames=[
            "monotonic_s",
            "wall_time_s",
            "sim_time_s",
            "support_active",
            "cmd_received",
            "cmd_active",
            "cmd_count",
            "cmd_age_s",
            "cmd_q_rms",
            "cmd_kp_mean",
            "cmd_target_error_rms",
            "ctrl_rms",
            "wireless_keys",
            "base_z",
        ],
    )
    status_writer.writeheader()

    lowcmd_fields = (
        ["time_s", "sim_time_s"]
        + [f"cmd_q_{i}" for i in range(num_motor)]
        + [f"cmd_dq_{i}" for i in range(num_motor)]
        + [f"cmd_tau_{i}" for i in range(num_motor)]
        + [f"cmd_kp_{i}" for i in range(num_motor)]
        + [f"cmd_kd_{i}" for i in range(num_motor)]
        + [f"ctrl_{i}" for i in range(num_motor)]
        + [f"measured_q_{i}" for i in range(num_motor)]
        + [f"measured_dq_{i}" for i in range(num_motor)]
        + [f"tau_est_{i}" for i in range(num_motor)]
    )
    lowcmd_writer = csv.DictWriter(lowcmd_f, fieldnames=lowcmd_fields)
    lowcmd_writer.writeheader()

    replay_fields = (
        [
            "time_s",
            "wall_time_s",
            "sim_time_s",
            "support_active",
            "cmd_received",
            "cmd_active",
            "cmd_count",
            "wireless_keys",
            "base_z",
        ]
        + [f"qpos_{i}" for i in range(nq)]
        + [f"qvel_{i}" for i in range(nv)]
        + [f"measured_q_{i}" for i in range(num_motor)]
    )
    replay_writer = csv.DictWriter(replay_f, fieldnames=replay_fields)
    replay_writer.writeheader()
    return status_f, lowcmd_f, replay_f, status_writer, lowcmd_writer, replay_writer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", default=str(DEFAULT_SIM_ROOT))
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--scene", default="")
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--log-hz", type=float, default=50.0)
    parser.add_argument("--publish-every", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--control-file", default="")
    parser.add_argument("--support-active", type=int, default=1)
    parser.add_argument("--support-height", type=float, default=0.75)
    parser.add_argument("--fall-stop-base-z", type=float, default=0.25)
    parser.add_argument("--fall-stop-hold-s", type=float, default=0.20)
    parser.add_argument("--cmd-timeout-s", type=float, default=0.25)
    args = parser.parse_args()
    args.publish_every = max(1, args.publish_every)

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

    sdk = load_upstream_bridge(sim_root, args.robot)
    sdk["ChannelFactoryInitialize"](args.domain_id, args.interface)

    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    data.qpos[0:3] = [0.0, 0.0, float(args.support_height)]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    set_named_joint_qpos(model, data, mujoco, NEUTRAL_JOINT_Q)
    mujoco.mj_forward(model, data)
    bridge = UnitreeG1Bridge(model, data, mujoco, sdk)
    support_root_qpos = data.qpos[:7].copy()
    support_root_qvel = data.qvel[:6].copy()

    status_f, lowcmd_f, replay_f, status_writer, lowcmd_writer, replay_writer = open_logs(
        out_dir,
        model.nu,
        model.nq,
        model.nv,
    )
    started_mono = time.monotonic()
    sync_mono = started_mono
    sync_sim = float(data.time)
    next_log_t = 0.0
    log_dt = 1.0 / args.log_hz
    step = 0
    fall_below_since: float | None = None
    fall_stop = None

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
            cmd_active = bridge.apply_control(
                precontrol_hold=support_active,
                cmd_timeout_s=float(args.cmd_timeout_s),
            )

            if support_active:
                data.qpos[:7] = support_root_qpos
                data.qvel[:6] = support_root_qvel
                mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            if support_active:
                data.qpos[:7] = support_root_qpos
                data.qvel[:6] = support_root_qvel
                mujoco.mj_forward(model, data)
                fall_below_since = None
            if step % args.publish_every == 0:
                set_wireless_remote(
                    bridge.low_state,
                    keys,
                    float(control.get("lx", 0.0)),
                    float(control.get("ly", 0.0)),
                    float(control.get("rx", 0.0)),
                    float(control.get("ry", 0.0)),
                )
                bridge.publish()

            if data.time + 1e-9 >= next_log_t:
                now_mono = time.monotonic()
                elapsed = now_mono - started_mono
                now_wall = time.time()
                cmd_count = 0
                cmd_age_s = ""
                cmd_q_rms = ""
                cmd_kp_mean = ""
                cmd_target_error_rms = ""
                with bridge.cmd_lock:
                    cmd_received = bridge.received_cmd
                    cmd_count = bridge.cmd_count
                    if bridge.last_cmd_monotonic is not None:
                        cmd_age_s = f"{now_mono - bridge.last_cmd_monotonic:.6f}"
                    if bridge.received_cmd:
                        cmd_q = [float(bridge.latest_cmd.motor_cmd[i].q) for i in range(bridge.num_motor)]
                        cmd_kp = [float(bridge.latest_cmd.motor_cmd[i].kp) for i in range(bridge.num_motor)]
                        measured_q = bridge.body_q()
                        cmd_q_rms = f"{float((sum(v * v for v in cmd_q) / bridge.num_motor) ** 0.5):.6f}"
                        cmd_kp_mean = f"{float(sum(cmd_kp) / bridge.num_motor):.6f}"
                        cmd_target_error_rms = (
                            f"{float((sum((cmd_q[i] - measured_q[i]) ** 2 for i in range(bridge.num_motor)) / bridge.num_motor) ** 0.5):.6f}"
                        )
                status_writer.writerow({
                    "monotonic_s": f"{elapsed:.6f}",
                    "wall_time_s": f"{now_wall:.6f}",
                    "sim_time_s": f"{data.time:.6f}",
                    "support_active": int(support_active),
                    "cmd_received": int(cmd_received),
                    "cmd_active": int(cmd_active),
                    "cmd_count": cmd_count,
                    "cmd_age_s": cmd_age_s,
                    "cmd_q_rms": cmd_q_rms,
                    "cmd_kp_mean": cmd_kp_mean,
                    "cmd_target_error_rms": cmd_target_error_rms,
                    "ctrl_rms": f"{float((data.ctrl @ data.ctrl / max(1, data.ctrl.size)) ** 0.5):.6f}",
                    "wireless_keys": keys,
                    "base_z": f"{float(data.qpos[2]):.6f}",
                })
                row = {"time_s": f"{elapsed:.6f}", "sim_time_s": f"{data.time:.6f}"}
                with bridge.cmd_lock:
                    cmd = bridge.latest_cmd if bridge.received_cmd else None
                measured_q = bridge.body_q()
                measured_dq = bridge.body_dq()
                tau_est = bridge.body_tau_est()
                for i in range(model.nu):
                    if i < bridge.num_motor and cmd is not None:
                        motor = cmd.motor_cmd[i]
                        row[f"cmd_q_{i}"] = f"{float(motor.q):.9f}"
                        row[f"cmd_dq_{i}"] = f"{float(motor.dq):.9f}"
                        row[f"cmd_tau_{i}"] = f"{float(motor.tau):.9f}"
                        row[f"cmd_kp_{i}"] = f"{float(motor.kp):.9f}"
                        row[f"cmd_kd_{i}"] = f"{float(motor.kd):.9f}"
                    else:
                        row[f"cmd_q_{i}"] = ""
                        row[f"cmd_dq_{i}"] = ""
                        row[f"cmd_tau_{i}"] = ""
                        row[f"cmd_kp_{i}"] = ""
                        row[f"cmd_kd_{i}"] = ""
                    if i < bridge.num_motor:
                        actuator_id = bridge.body_actuator_ids[i]
                        row[f"ctrl_{i}"] = f"{float(data.ctrl[actuator_id]):.9f}"
                        row[f"measured_q_{i}"] = f"{float(measured_q[i]):.9f}"
                        row[f"measured_dq_{i}"] = f"{float(measured_dq[i]):.9f}"
                        row[f"tau_est_{i}"] = f"{float(tau_est[i]):.9f}"
                    else:
                        row[f"ctrl_{i}"] = ""
                        row[f"measured_q_{i}"] = ""
                        row[f"measured_dq_{i}"] = ""
                        row[f"tau_est_{i}"] = ""
                lowcmd_writer.writerow(row)

                replay_row = {
                    "time_s": f"{elapsed:.6f}",
                    "wall_time_s": f"{now_wall:.6f}",
                    "sim_time_s": f"{data.time:.6f}",
                    "support_active": int(support_active),
                    "cmd_received": int(cmd_received),
                    "cmd_active": int(cmd_active),
                    "cmd_count": cmd_count,
                    "wireless_keys": keys,
                    "base_z": f"{float(data.qpos[2]):.6f}",
                }
                for i in range(model.nq):
                    replay_row[f"qpos_{i}"] = f"{float(data.qpos[i]):.9f}"
                for i in range(model.nv):
                    replay_row[f"qvel_{i}"] = f"{float(data.qvel[i]):.9f}"
                for i in range(bridge.num_motor):
                    replay_row[f"measured_q_{i}"] = f"{float(measured_q[i]):.9f}"
                replay_writer.writerow(replay_row)
                status_f.flush()
                lowcmd_f.flush()
                replay_f.flush()
                next_log_t += log_dt

            if (
                not support_active
                and args.fall_stop_base_z > 0.0
                and float(data.qpos[2]) < args.fall_stop_base_z
            ):
                if fall_below_since is None:
                    fall_below_since = float(data.time)
                elif float(data.time) - fall_below_since >= args.fall_stop_hold_s:
                    fall_stop = {
                        "sim_time_s": float(data.time),
                        "base_z": float(data.qpos[2]),
                        "threshold": float(args.fall_stop_base_z),
                        "hold_s": float(args.fall_stop_hold_s),
                    }
                    print(
                        "[sim_bridge] early stop: base_z "
                        f"{fall_stop['base_z']:.3f} below {fall_stop['threshold']:.3f} "
                        f"for {fall_stop['hold_s']:.3f}s",
                        flush=True,
                    )
                    break
            else:
                fall_below_since = None

            sleep_s = sync_mono + (float(data.time) - sync_sim) - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            step += 1
    finally:
        status_f.close()
        lowcmd_f.close()
        replay_f.close()

    (out_dir / "sim_bridge_summary.json").write_text(
        json.dumps({
            "scene": str(scene),
            "robot": args.robot,
            "duration_s": args.duration_s,
            "dt": args.dt,
            "log_hz": args.log_hz,
            "publish_every": args.publish_every,
            "support_height": args.support_height,
            "neutral_joint_q": NEUTRAL_JOINT_Q,
            "hold_kp": HOLD_KP,
            "hold_kd": HOLD_KD,
            "fall_stop": fall_stop,
            "control_file": str(control_path),
            "cmd_timeout_s": args.cmd_timeout_s,
            "replay_log": "replay.csv",
            "qpos_columns": [f"qpos_{i}" for i in range(model.nq)],
            "qvel_columns": [f"qvel_{i}" for i in range(model.nv)],
            "body_joint_names": BODY_JOINT_NAMES,
            "body_qpos_addresses": bridge.body_qpos_adrs,
        }, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
