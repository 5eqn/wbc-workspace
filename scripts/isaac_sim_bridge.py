#!/usr/bin/env python3
"""Headless Unitree G1 Isaac Sim endpoint for benchmark orchestration.

This bridge mirrors scripts/sim_bridge.py's DDS and log contract, using the
Unitree IsaacLab G1 USD asset by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from sim_bridge import (
    BODY_JOINT_NAMES,
    HOLD_KD,
    HOLD_KP,
    NEUTRAL_JOINT_Q,
    load_unitree_sdk2py,
    open_logs,
    quat_to_rpy_wxyz,
    read_control,
    set_wireless_remote,
    write_control,
)


DEFAULT_MJCF_SCENE = (
    "/workspace/wbc/thirdparties/GR00T-WholeBodyControl/decoupled_wbc/"
    "control/robot_model/model_data/g1/scene_29dof.xml"
)
DEFAULT_URDF_SCENE = (
    "/workspace/wbc/thirdparties/GR00T-WholeBodyControl/decoupled_wbc/"
    "control/robot_model/model_data/g1/g1_29dof.urdf"
)
DEFAULT_UNITREE_ISAACLAB_SCENE = (
    "/workspace/wbc/thirdparties/unitree_sim_isaaclab/assets/robots/"
    "g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd"
)
DEFAULT_SCENE = DEFAULT_UNITREE_ISAACLAB_SCENE
DEFAULT_TASK = "Isaac-G1-29DOF"


class UnitreeG1IsaacBridge:
    def __init__(self, robot: Any, np: Any, sdk: dict[str, Any]):
        import threading

        self.robot = robot
        self.np = np
        self.num_motor = len(BODY_JOINT_NAMES)
        self.dof_names = list(robot.dof_names)
        self.body_dof_ids = [int(robot.get_dof_index(name)) for name in BODY_JOINT_NAMES]
        self.body_dof_ids_np = np.asarray(self.body_dof_ids, dtype=np.int32)
        self.neutral_body_q = np.asarray(NEUTRAL_JOINT_Q, dtype=np.float32)

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
        self.last_effort = np.zeros(self.num_motor, dtype=np.float32)

    def low_cmd_handler(self, msg: Any) -> None:
        with self.cmd_lock:
            self.latest_cmd = msg
            self.received_cmd = True
            self.cmd_count += 1
            self.last_cmd_monotonic = time.monotonic()

    def body_q(self):
        return self.robot.get_joint_positions(joint_indices=self.body_dof_ids_np)

    def body_dq(self):
        return self.robot.get_joint_velocities(joint_indices=self.body_dof_ids_np)

    def body_tau_est(self):
        try:
            efforts = self.robot.get_measured_joint_efforts(joint_indices=self.body_dof_ids_np)
            if efforts is not None:
                return efforts
        except Exception:
            pass
        return self.last_effort

    def world_vector_to_body(self, quat, vec):
        np = self.np
        qw, qx, qy, qz = [float(v) for v in quat]
        rot = np.asarray([
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ], dtype=np.float32)
        return rot.T @ np.asarray(vec, dtype=np.float32)

    def command_active(self, cmd_timeout_s: float) -> bool:
        with self.cmd_lock:
            return (
                self.received_cmd
                and self.last_cmd_monotonic is not None
                and time.monotonic() - self.last_cmd_monotonic <= cmd_timeout_s
            )

    def apply_control(self, precontrol_hold: bool, cmd_timeout_s: float, ArticulationAction: Any) -> bool:
        np = self.np
        q = self.body_q()
        dq = self.body_dq()
        effort = np.zeros(self.num_motor, dtype=np.float32)
        with self.cmd_lock:
            cmd = self.latest_cmd
            active = (
                self.received_cmd
                and self.last_cmd_monotonic is not None
                and time.monotonic() - self.last_cmd_monotonic <= cmd_timeout_s
            )
        if active:
            for i in range(self.num_motor):
                motor = cmd.motor_cmd[i]
                effort[i] = float(motor.tau + motor.kp * (motor.q - q[i]) + motor.kd * (motor.dq - dq[i]))
        elif precontrol_hold:
            effort[:] = HOLD_KP * (self.neutral_body_q - q) - HOLD_KD * dq
        self.last_effort = effort
        self.robot.apply_action(ArticulationAction(joint_efforts=effort, joint_indices=self.body_dof_ids_np))
        return active

    def publish(self, sim_time_s: float) -> None:
        body_q = self.body_q()
        body_dq = self.body_dq()
        body_tau = self.body_tau_est()
        for i in range(self.num_motor):
            self.low_state.motor_state[i].q = float(body_q[i])
            self.low_state.motor_state[i].dq = float(body_dq[i])
            self.low_state.motor_state[i].tau_est = float(body_tau[i])

        pos, quat = self.robot.get_world_pose()
        lin_vel = self.robot.get_linear_velocity()
        ang_vel = self.robot.get_angular_velocity()
        imu_gyro = self.world_vector_to_body(quat, ang_vel)
        self.low_state.imu_state.quaternion[:] = [float(v) for v in quat]
        self.low_state.imu_state.rpy[:] = list(quat_to_rpy_wxyz(quat))
        self.low_state.imu_state.gyroscope[:] = [float(v) for v in imu_gyro]
        self.low_state.imu_state.accelerometer[:] = [0.0, 0.0, 0.0]
        self.low_state.tick = int(round(sim_time_s * 1000.0))
        self.low_state_pub.Write(self.low_state)

        self.sport_state.position[:] = [float(v) for v in pos]
        self.sport_state.velocity[:] = [float(v) for v in lin_vel]
        self.sport_state_pub.Write(self.sport_state)
        self.secondary_imu.quaternion[:] = [float(v) for v in quat]
        self.secondary_imu.rpy[:] = list(quat_to_rpy_wxyz(quat))
        self.secondary_imu.gyroscope[:] = [float(v) for v in imu_gyro]
        self.secondary_imu.accelerometer[:] = [0.0, 0.0, 0.0]
        self.secondary_imu_pub.Write(self.secondary_imu)
        self.bms_pub.Write(self.bms_state)


def set_import_config(import_config: Any, name: str, value: Any) -> None:
    setter = getattr(import_config, f"set_{name}", None)
    if setter is not None:
        setter(value)
        return
    if hasattr(import_config, name):
        setattr(import_config, name, value)


def import_mjcf_scene(scene_path: str, prim_path: str, app: Any) -> None:
    from isaacsim.core.utils.extensions import enable_extension
    import omni.kit.commands

    enable_extension("isaacsim.asset.importer.mjcf")
    app.update()
    ok, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
    if not ok:
        raise RuntimeError("failed to create MJCF import config")
    import_config.set_fix_base(False)
    import_config.set_import_inertia_tensor(True)
    import_config.set_create_physics_scene(True)
    import_config.set_self_collision(False)
    ok, _ = omni.kit.commands.execute(
        "MJCFCreateAsset",
        mjcf_path=str(Path(scene_path).resolve()),
        import_config=import_config,
        prim_path=prim_path,
    )
    if not ok:
        raise RuntimeError(f"failed to import MJCF scene: {scene_path}")
    app.update()


def import_urdf_scene(scene_path: str, app: Any) -> None:
    from isaacsim.core.utils.extensions import enable_extension
    import omni.kit.commands

    enable_extension("isaacsim.asset.importer.urdf")
    app.update()
    ok, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not ok:
        raise RuntimeError("failed to create URDF import config")
    set_import_config(import_config, "fix_base", False)
    set_import_config(import_config, "merge_fixed_joints", True)
    set_import_config(import_config, "import_inertia_tensor", True)
    set_import_config(import_config, "self_collision", False)
    ok, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(Path(scene_path).resolve()),
        import_config=import_config,
    )
    if not ok:
        raise RuntimeError(f"failed to import URDF scene: {scene_path}")
    app.update()


def import_usd_scene(scene_path: str, prim_path: str) -> None:
    from isaacsim.core.utils.stage import add_reference_to_stage

    scene = Path(scene_path)
    if not scene.exists():
        raise FileNotFoundError(f"Isaac USD scene not found: {scene_path}")
    add_reference_to_stage(usd_path=str(scene.resolve()), prim_path=prim_path)


def find_articulation_root(fallback: str) -> str:
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdPhysics

    stage = get_current_stage()
    candidates = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if candidates:
        return sorted(candidates, key=lambda path: (path.count("/"), path))[0]
    return fallback


def configure_contact_physics(world: Any) -> int:
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    physics = world.get_physics_context()
    physics.set_friction_offset_threshold(0.01)
    physics.set_friction_correlation_distance(0.00625)

    material = PhysicsMaterial(
        prim_path="/World/Physics_Materials/g1_contact",
        static_friction=2.5,
        dynamic_friction=2.5,
        restitution=0.0,
    )
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
    physx_material.CreateFrictionCombineModeAttr().Set("max")
    physx_material.CreateRestitutionCombineModeAttr().Set("min")

    bound = 0
    stage = get_current_stage()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material.material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            bound += 1
    return bound


def configure_articulation_physics(robot: Any) -> dict[str, int | None]:
    solver_position_iterations = 8
    solver_velocity_iterations = 4
    applied = {
        "solver_position_iteration_count": None,
        "solver_velocity_iteration_count": None,
    }
    try:
        robot.set_solver_position_iteration_count(solver_position_iterations)
        applied["solver_position_iteration_count"] = int(robot.get_solver_position_iteration_count())
    except Exception:
        pass
    try:
        robot.set_solver_velocity_iteration_count(solver_velocity_iterations)
        applied["solver_velocity_iteration_count"] = int(robot.get_solver_velocity_iteration_count())
    except Exception:
        pass
    return applied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--log-hz", type=float, default=50.0)
    parser.add_argument("--publish-every", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--control-file", default="")
    parser.add_argument("--support-active", type=int, default=1)
    parser.add_argument("--support-height", type=float, default=0.80)
    parser.add_argument("--fall-stop-base-z", type=float, default=0.25)
    parser.add_argument("--fall-stop-hold-s", type=float, default=0.20)
    parser.add_argument("--cmd-timeout-s", type=float, default=0.25)
    parser.add_argument("--no-render", action="store_true", default=True)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.publish_every = max(1, args.publish_every)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    sdk = load_unitree_sdk2py()
    sdk["ChannelFactoryInitialize"](args.domain_id, args.interface)

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

    world = None
    try:
        scene_suffix = Path(args.scene).suffix.lower()
        if scene_suffix == ".urdf":
            import_urdf_scene(args.scene, simulation_app)
            articulation_path = find_articulation_root("/g1_29dof/pelvis")
        elif scene_suffix in {".usd", ".usda", ".usdc"}:
            import_usd_scene(args.scene, "/g1")
            articulation_path = find_articulation_root("/g1")
        else:
            import_mjcf_scene(args.scene, "/g1", simulation_app)
            articulation_path = "/g1/pelvis/pelvis"
        world = World(physics_dt=float(args.dt), rendering_dt=max(float(args.dt), 0.02), stage_units_in_meters=1.0)
        if scene_suffix in {".urdf", ".usd", ".usda", ".usdc"}:
            world.scene.add_default_ground_plane()
        robot = SingleArticulation(articulation_path, name="g1")
        world.scene.add(robot)
        world.reset()
        contact_material_bindings = configure_contact_physics(world)
        articulation_physics = configure_articulation_physics(robot)
        bridge = UnitreeG1IsaacBridge(robot, np, sdk)

        support_pos = np.asarray([0.0, 0.0, float(args.support_height)], dtype=np.float32)
        support_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        zero3 = np.zeros(3, dtype=np.float32)
        robot.set_world_pose(position=support_pos, orientation=support_quat)
        robot.set_linear_velocity(zero3)
        robot.set_angular_velocity(zero3)
        robot.set_joint_positions(bridge.neutral_body_q, joint_indices=bridge.body_dof_ids_np)
        robot.set_joint_velocities(np.zeros(bridge.num_motor, dtype=np.float32), joint_indices=bridge.body_dof_ids_np)

        status_f, lowcmd_f, replay_f, status_writer, lowcmd_writer, replay_writer = open_logs(
            out_dir,
            bridge.num_motor,
            7 + bridge.num_motor,
            6 + bridge.num_motor,
        )
        started_mono = time.monotonic()
        sync_mono = started_mono
        sync_sim = 0.0
        next_log_t = 0.0
        log_dt = 1.0 / args.log_hz
        step = 0
        fall_below_since: float | None = None
        fall_stop = None

        try:
            while simulation_app.is_running():
                elapsed = time.monotonic() - started_mono
                sim_time_s = step * args.dt
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
                cmd_active = bridge.apply_control(support_active, float(args.cmd_timeout_s), ArticulationAction)
                if support_active:
                    robot.set_world_pose(position=support_pos, orientation=support_quat)
                    robot.set_linear_velocity(zero3)
                    robot.set_angular_velocity(zero3)
                    fall_below_since = None

                world.step(render=False)
                if support_active:
                    robot.set_world_pose(position=support_pos, orientation=support_quat)
                    robot.set_linear_velocity(zero3)
                    robot.set_angular_velocity(zero3)

                if step % args.publish_every == 0:
                    set_wireless_remote(
                        bridge.low_state,
                        keys,
                        float(control.get("lx", 0.0)),
                        float(control.get("ly", 0.0)),
                        float(control.get("rx", 0.0)),
                        float(control.get("ry", 0.0)),
                    )
                    bridge.publish(sim_time_s)

                pos, quat = robot.get_world_pose()
                lin_vel = robot.get_linear_velocity()
                ang_vel = robot.get_angular_velocity()
                base_z = float(pos[2])
                if sim_time_s + 1e-9 >= next_log_t:
                    now_mono = time.monotonic()
                    now_wall = time.time()
                    with bridge.cmd_lock:
                        cmd_received = bridge.received_cmd
                        cmd_count = bridge.cmd_count
                        cmd = bridge.latest_cmd if bridge.received_cmd else None
                        cmd_age_s = (
                            ""
                            if bridge.last_cmd_monotonic is None
                            else f"{now_mono - bridge.last_cmd_monotonic:.6f}"
                        )
                    measured_q = bridge.body_q()
                    measured_dq = bridge.body_dq()
                    tau_est = bridge.body_tau_est()
                    if cmd is not None:
                        cmd_q = [float(cmd.motor_cmd[i].q) for i in range(bridge.num_motor)]
                        cmd_kp = [float(cmd.motor_cmd[i].kp) for i in range(bridge.num_motor)]
                        cmd_q_rms = f"{float((sum(v * v for v in cmd_q) / bridge.num_motor) ** 0.5):.6f}"
                        cmd_kp_mean = f"{float(sum(cmd_kp) / bridge.num_motor):.6f}"
                        cmd_target_error_rms = (
                            f"{float((sum((cmd_q[i] - measured_q[i]) ** 2 for i in range(bridge.num_motor)) / bridge.num_motor) ** 0.5):.6f}"
                        )
                    else:
                        cmd_q_rms = ""
                        cmd_kp_mean = ""
                        cmd_target_error_rms = ""
                    ctrl_rms = f"{float(np.sqrt(np.mean(bridge.last_effort * bridge.last_effort))):.6f}"
                    status_writer.writerow({
                        "monotonic_s": f"{now_mono - started_mono:.6f}",
                        "wall_time_s": f"{now_wall:.6f}",
                        "sim_time_s": f"{sim_time_s:.6f}",
                        "support_active": int(support_active),
                        "cmd_received": int(cmd_received),
                        "cmd_active": int(cmd_active),
                        "cmd_count": cmd_count,
                        "cmd_age_s": cmd_age_s,
                        "cmd_q_rms": cmd_q_rms,
                        "cmd_kp_mean": cmd_kp_mean,
                        "cmd_target_error_rms": cmd_target_error_rms,
                        "ctrl_rms": ctrl_rms,
                        "wireless_keys": keys,
                        "base_z": f"{base_z:.6f}",
                    })

                    row = {"time_s": f"{now_mono - started_mono:.6f}", "sim_time_s": f"{sim_time_s:.6f}"}
                    for i in range(bridge.num_motor):
                        if cmd is not None:
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
                        row[f"ctrl_{i}"] = f"{float(bridge.last_effort[i]):.9f}"
                        row[f"measured_q_{i}"] = f"{float(measured_q[i]):.9f}"
                        row[f"measured_dq_{i}"] = f"{float(measured_dq[i]):.9f}"
                        row[f"tau_est_{i}"] = f"{float(tau_est[i]):.9f}"
                    lowcmd_writer.writerow(row)

                    qpos = [*pos.tolist(), *quat.tolist(), *measured_q.tolist()]
                    qvel = [*lin_vel.tolist(), *ang_vel.tolist(), *measured_dq.tolist()]
                    replay_row = {
                        "time_s": f"{now_mono - started_mono:.6f}",
                        "wall_time_s": f"{now_wall:.6f}",
                        "sim_time_s": f"{sim_time_s:.6f}",
                        "support_active": int(support_active),
                        "cmd_received": int(cmd_received),
                        "cmd_active": int(cmd_active),
                        "cmd_count": cmd_count,
                        "wireless_keys": keys,
                        "base_z": f"{base_z:.6f}",
                    }
                    for i, value in enumerate(qpos):
                        replay_row[f"qpos_{i}"] = f"{float(value):.9f}"
                    for i, value in enumerate(qvel):
                        replay_row[f"qvel_{i}"] = f"{float(value):.9f}"
                    for i in range(bridge.num_motor):
                        replay_row[f"measured_q_{i}"] = f"{float(measured_q[i]):.9f}"
                    replay_writer.writerow(replay_row)
                    status_f.flush()
                    lowcmd_f.flush()
                    replay_f.flush()
                    next_log_t += log_dt

                if not support_active and args.fall_stop_base_z > 0.0 and base_z < args.fall_stop_base_z:
                    if fall_below_since is None:
                        fall_below_since = sim_time_s
                    elif sim_time_s - fall_below_since >= args.fall_stop_hold_s:
                        fall_stop = {
                            "sim_time_s": sim_time_s,
                            "base_z": base_z,
                            "threshold": float(args.fall_stop_base_z),
                            "hold_s": float(args.fall_stop_hold_s),
                        }
                        print(
                            "[isaac_sim_bridge] early stop: base_z "
                            f"{base_z:.3f} below {args.fall_stop_base_z:.3f} "
                            f"for {args.fall_stop_hold_s:.3f}s",
                            flush=True,
                        )
                        break
                else:
                    fall_below_since = None

                sleep_s = sync_mono + (((step + 1) * args.dt) - sync_sim) - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                step += 1
        finally:
            status_f.close()
            lowcmd_f.close()
            replay_f.close()

        (out_dir / "sim_bridge_summary.json").write_text(
            json.dumps({
                "backend": "isaac",
                "task": args.task,
                "scene": args.scene,
                "import_scene": args.scene,
                "articulation_path": articulation_path,
                "duration_s": args.duration_s,
                "dt": args.dt,
                "log_hz": args.log_hz,
                "publish_every": args.publish_every,
                "support_height": args.support_height,
                "neutral_joint_q": NEUTRAL_JOINT_Q,
                "hold_kp": HOLD_KP,
                "hold_kd": HOLD_KD,
                "contact_material_bindings": contact_material_bindings,
                "articulation_physics": articulation_physics,
                "fall_stop": fall_stop,
                "control_file": str(control_path),
                "cmd_timeout_s": args.cmd_timeout_s,
                "replay_log": "replay.csv",
                "qpos_columns": [f"qpos_{i}" for i in range(7 + bridge.num_motor)],
                "qvel_columns": [f"qvel_{i}" for i in range(6 + bridge.num_motor)],
                "body_joint_names": BODY_JOINT_NAMES,
                "body_qpos_addresses": [7 + i for i in range(bridge.num_motor)],
                "isaac_dof_names": bridge.dof_names,
                "isaac_body_dof_ids": bridge.body_dof_ids,
            }, indent=2) + "\n"
        )
    finally:
        if world is not None:
            world.clear()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
