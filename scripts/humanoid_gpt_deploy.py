#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HGPT_REPO = ROOT / "thirdparties" / "Humanoid-GPT"


def _prepend_runtime_libs() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    prefix = Path(conda_prefix)
    extra_dirs: list[str] = []
    extra_dirs.extend(
        str(path)
        for path in sorted(prefix.glob("lib/python*/site-packages/nvidia/*/lib"))
        if path.is_dir()
    )
    extra_dirs.extend(
        str(path)
        for path in sorted(prefix.glob("lib/python*/site-packages/tensorrt_libs"))
        if path.is_dir()
    )
    if not extra_dirs:
        return
    existing = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    merged: list[str] = []
    for entry in extra_dirs + existing:
        if entry not in merged:
            merged.append(entry)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--freq", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-type", default="mlp")
    parser.add_argument("--onnx-walk", default="storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx")
    parser.add_argument("--onnx-track", default="storage/ckpts/pns_wo_priv216.onnx")
    parser.add_argument("--convert-xml-path", default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    _prepend_runtime_libs()
    if str(HGPT_REPO) not in sys.path:
        sys.path.insert(0, str(HGPT_REPO))
    os.chdir(HGPT_REPO)

    import mujoco
    import numpy as np
    from jax import tree_util as jtu
    from loop_rate_limiters import RateLimiter
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from deploy.constants import DEFAULT_QPOS as DEFAULT_QPOS_JOINT, KDs_walking, KPs_walking
    from deploy.play_track import load_offline_motions
    from deploy.real_robot import KeyMap, LowLevelControlG1
    from deploy.walk_policy import WalkPolicy
    from tracking import constants as consts
    from tracking.infer_utils import G1TrackInferFn, g1_infer_env_config
    from tracking.policy import Args as PolicyArgs, get_policy_onnx

    ctrl_dt = 1.0 / args.freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)
    policy_args = PolicyArgs(
        load_path=args.onnx_track,
        policy_type=args.policy_type,
        device=args.device,
    )
    track_policy = get_policy_onnx(policy_args, use_trt=True, strict_trt=True)
    walk_policy = WalkPolicy(args.onnx_walk)

    convert_xml_path = args.convert_xml_path or str(consts.TRACK_XML)
    convert_model = mujoco.MjModel.from_xml_path(convert_xml_path)
    ref_motions = load_offline_motions(args.motion_file, convert_model, args.freq)
    if len(ref_motions) != 1:
        raise ValueError(f"expected exactly one offline motion from {args.motion_file}, got {len(ref_motions)}")
    ref_traj = ref_motions[0]["data"]
    traj_name = ref_motions[0]["filename"]

    ChannelFactoryInitialize(args.domain_id, args.interface)
    low_ctrl = LowLevelControlG1(ctrl_dt=ctrl_dt, debug=args.debug)

    phantom_model = mujoco.MjModel.from_xml_path(str(consts.ROOT_PATH / "scene_mjx_track.xml"))
    phantom_model.opt.timestep = 0.001
    infer_fn = G1TrackInferFn(env_cfg, phantom_model, track_policy, privileged=False)

    running = True

    def _stop(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"HGPT_READY motion={traj_name} providers={track_policy.onnx_model.get_providers()}", flush=True)
    print("HGPT_WAITING_FOR_START", flush=True)
    while running and low_ctrl.remote.button[KeyMap.start] != 1:
        low_ctrl.set_motor_damping()
        time.sleep(ctrl_dt)

    if not running:
        low_ctrl.set_motor_damping()
        return 1

    low_ctrl.move_to_default_pos(duration=2.0, dt=ctrl_dt)
    print("HGPT_DEFAULT_READY", flush=True)

    while running and low_ctrl.remote.button[KeyMap.A] != 1:
        low_ctrl.step(DEFAULT_QPOS_JOINT, consts.KPs, consts.KDs)
        time.sleep(ctrl_dt)

    if not running:
        low_ctrl.set_motor_damping()
        return 1

    print("HGPT_CONTROL_READY", flush=True)
    rate = RateLimiter(frequency=args.freq, warn=False)
    cmd_vel = np.zeros(3, dtype=np.float32)
    tracking_active = False
    track_step = 0
    printed_start = False
    while running:
        if low_ctrl.remote.button[KeyMap.select] == 1:
            break
        root_quat, root_gyro, jnt_qpos, jnt_qvel = low_ctrl.get_sensor_state()
        if not tracking_active and low_ctrl.remote.button[KeyMap.B] == 1:
            tracking_active = True
            infer_fn.info["last_action"][:] = 0
            track_step = 0
        if tracking_active:
            traj_len = len(ref_traj["qpos"])
            ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
            next_step = min(track_step + 1, traj_len - 1)
            ref_next = jtu.tree_map(lambda x: x[next_step][None], ref_traj)
            motor_targets = infer_fn.infer_onnx_real(
                root_quat,
                root_gyro,
                jnt_qpos,
                jnt_qvel,
                {"ref_curr": ref_curr, "ref_next": ref_next},
            )
            low_ctrl.step(np.asarray(motor_targets).flatten(), consts.KPs, consts.KDs)
            if not printed_start:
                print(f"HGPT_MOTION_TRACKING_START motion={traj_name}", flush=True)
                printed_start = True
            if track_step < traj_len - 1:
                track_step += 1
            elif printed_start:
                print(f"HGPT_MOTION_TRACKING_COMPLETE motion={traj_name}", flush=True)
                printed_start = False
        else:
            motor_targets = walk_policy.infer(root_quat, root_gyro, jnt_qpos, jnt_qvel, cmd_vel)
            low_ctrl.step(motor_targets, KPs_walking, KDs_walking)
        rate.sleep()

    low_ctrl.set_motor_damping()
    print("HGPT_EXIT", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
