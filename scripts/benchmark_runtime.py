#!/usr/bin/env python3
from __future__ import annotations

from benchmark_common import *
from benchmark_report import validate_release_validation_logs

def pulse_holomotion_key(control_file: Path, key: str, *, hold_s: float = 0.16, gap_s: float = 0.16) -> None:
    update_control_file(control_file, wireless_keys=HOLO_KEY_BITS[key])
    time.sleep(hold_s)
    update_control_file(control_file, wireless_keys=0)
    time.sleep(gap_s)


def tap_holomotion_key(control_file: Path, key: str, *, gap_s: float = 0.20) -> None:
    update_control_file(control_file, wireless_keys=0, wireless_keys_once=HOLO_KEY_BITS[key])
    time.sleep(gap_s)


def pulse_remote_key(control_file: Path, key: str, *, hold_s: float = 0.16, gap_s: float = 0.16) -> None:
    pulse_holomotion_key(control_file, key, hold_s=hold_s, gap_s=gap_s)


def tap_remote_key(control_file: Path, key: str, *, gap_s: float = 0.20) -> None:
    tap_holomotion_key(control_file, key, gap_s=gap_s)


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

def wait_for_sim_window(sim: ManagedProcess, duration_s: float) -> int | None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        code = sim.poll()
        if code is not None:
            return code
        time.sleep(min(0.10, max(0.0, deadline - time.monotonic())))
    return sim.poll()

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
        "if [ -f build/CMakeCache.txt ] && "
        "! grep -q '/workspace/wbc/thirdparties/GR00T-WholeBodyControl/gear_sonic_deploy' build/CMakeCache.txt; "
        "then rm -rf build; fi && "
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
    del policy, motion
    return []


def policy_scene_path(policy: str) -> Path:
    del policy
    return COMMON_SIM_SCENE

def simulator_scene_args(policy: str) -> list[str]:
    scene = policy_scene_path(policy)
    return ["--scene", f"/workspace/wbc/{scene.relative_to(ROOT)}"]


def simulator_domain_args(policy: str) -> list[str]:
    return ["--domain-id", "0" if policy == "sonic" else "1"]


def start_simulator(
    run_dir: Path,
    control_file: Path,
    duration_s: float,
    name: str,
    policy: str,
    motion: str,
    support_height: float,
    fall_stop_base_z: float,
    backend: str = "mujoco",
) -> ManagedProcess:
    docker_rm_force(name)
    if backend == "mujoco":
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
            "--support-height",
            f"{support_height:.3f}",
            "--fall-stop-base-z",
            f"{fall_stop_base_z:.3f}",
            "--out-dir",
            f"/workspace/wbc/{run_dir.relative_to(ROOT)}",
            "--control-file",
            f"/workspace/wbc/{control_file.relative_to(ROOT)}",
        ] + simulator_domain_args(policy) + simulator_scene_args(policy) + simulator_reference_args(policy, motion)
    elif backend == "isaac":
        cmd = docker_base_args(name, ISAAC_SIM_IMAGE) + [
            "conda",
            "run",
            "-n",
            "unitree_isaacsim",
            "python",
            "/workspace/wbc/scripts/isaac_sim_bridge.py",
            "--task",
            "Isaac-Move-Cylinder-G129-Dex1-Wholebody",
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
            "--support-height",
            f"{support_height:.3f}",
            "--fall-stop-base-z",
            f"{fall_stop_base_z:.3f}",
            "--out-dir",
            f"/workspace/wbc/{run_dir.relative_to(ROOT)}",
            "--control-file",
            f"/workspace/wbc/{control_file.relative_to(ROOT)}",
            "--headless",
        ] + simulator_domain_args(policy) + simulator_reference_args(policy, motion)
    else:
        raise ValueError(f"unsupported simulator backend: {backend}")
    proc = ManagedProcess(cmd, run_dir / "simulator_stdout.log")
    proc.start()
    ready_timeout_s = 60.0 if backend == "isaac" else 10.0
    wait_for_file(control_file, ready_timeout_s)
    wait_for_support_state(run_dir, 1, ready_timeout_s)
    return proc


def run_sonic_sequence(
    args: argparse.Namespace,
    run_dir: Path,
    control_file: Path,
    event_log: SequenceEventLog,
    sim: ManagedProcess,
) -> None:
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
        post_release_wait_s = args.sonic_post_release_wait_s
        if post_release_wait_s < SONIC_POST_RELEASE_WAIT_S:
            raise ValueError(
                f"SONIC post-release wait must be at least {SONIC_POST_RELEASE_WAIT_S:.1f}s "
                f"(got {post_release_wait_s:.3f}s)"
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
        wait_for_motion_window(sim, args.motion_duration_s, event_log, run_dir)
    finally:
        update_control_file(control_file, stop=True)
        policy.terminate()
        docker_rm_force(name)


def run_holomotion_sequence(
    args: argparse.Namespace,
    run_dir: Path,
    control_file: Path,
    event_log: SequenceEventLog,
    sim: ManagedProcess,
) -> None:
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
        time.sleep(HOLOMOTION_POST_RELEASE_VELOCITY_HOLD_S)
        event_log.append(
            "post_release_velocity_hold_elapsed",
            sim_time_s=latest_sim_time(run_dir),
            support_active=0,
            detail=f"seconds={HOLOMOTION_POST_RELEASE_VELOCITY_HOLD_S:.3f}",
        )
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
        wait_for_motion_window(sim, args.motion_duration_s, event_log, run_dir)
    finally:
        update_control_file(control_file, stop=True)
        policy.terminate()
        bridge.terminate()
        docker_rm_force(name)
        docker_rm_force(bridge_name)


def humanoid_gpt_motion_file(motion: str) -> Path:
    return HUMANOID_GPT_TRACK_DIR / f"{motion}.npz"


def run_humanoid_gpt_sequence(
    args: argparse.Namespace,
    run_dir: Path,
    control_file: Path,
    event_log: SequenceEventLog,
    sim: ManagedProcess,
) -> None:
    motion_file = humanoid_gpt_motion_file(args.motion)
    if not motion_file.exists():
        raise FileNotFoundError(f"missing translated Humanoid-GPT motion: {motion_file}")
    hgpt_prefix = Path.home() / "miniconda3" / "envs" / HUMANOID_GPT_ENV
    runtime_lib_dirs = sorted(
        str(path)
        for path in hgpt_prefix.glob("lib/python3.12/site-packages/nvidia/*/lib")
        if path.is_dir()
    )
    trt_dir = hgpt_prefix / "lib" / "python3.12" / "site-packages" / "tensorrt_libs"
    if trt_dir.is_dir():
        runtime_lib_dirs.append(str(trt_dir))
    launch_env = ""
    if runtime_lib_dirs:
        quoted = ":".join(runtime_lib_dirs)
        launch_env = (
            f"export LD_LIBRARY_PATH='{quoted}'"
            "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
        )
    policy = ManagedProcess(
        [
            "bash",
            "-lc",
            (
                f"export CYCLONEDDS_HOME=/home/seqn/cyclonedds/install; "
                f"{launch_env}"
                f"conda run --no-capture-output -n {HUMANOID_GPT_ENV} "
                f"python -u {ROOT / 'scripts' / 'humanoid_gpt_deploy.py'} "
                f"--motion-file {motion_file} --interface lo --domain-id 1"
            ),
        ],
        run_dir / "humanoid_gpt_stdout.log",
    )
    try:
        policy.start()
        wait_for_log_marker(policy, policy.log_path, "HGPT_WAITING_FOR_START", args.policy_ready_timeout_s)

        pulse_remote_key(control_file, "start")
        wait_for_log_marker(policy, policy.log_path, "HGPT_DEFAULT_READY", args.policy_ready_timeout_s)

        pulse_remote_key(control_file, "a")
        marker = wait_for_log_marker(
            policy,
            policy.log_path,
            "HGPT_CONTROL_READY",
            args.policy_ready_timeout_s,
        )
        event_log.append(
            CONTROL_EVENT,
            sim_time_s=latest_sim_time(run_dir),
            support_active=1,
            detail=marker,
        )
        event_log.append(
            "motion_selected",
            sim_time_s=latest_sim_time(run_dir),
            support_active=1,
            detail=str(motion_file.relative_to(ROOT)),
        )

        wait_for_pre_release_hold(run_dir, event_log)
        release_support(run_dir, control_file, event_log, "release simulator support after Humanoid-GPT CONTROL")
        if args.humanoid_gpt_post_release_wait_s > 0.0:
            time.sleep(args.humanoid_gpt_post_release_wait_s)
            event_log.append(
                "post_release_hold_elapsed",
                sim_time_s=latest_sim_time(run_dir),
                support_active=0,
                detail=f"seconds={args.humanoid_gpt_post_release_wait_s:.3f}",
            )

        event_log.append(
            "sent_key_B",
            sim_time_s=latest_sim_time(run_dir),
            support_active=0,
            detail="Humanoid-GPT offline motion tracking trigger after confirmed release",
        )
        pulse_remote_key(control_file, "b")
        marker = wait_for_log_marker(
            policy,
            policy.log_path,
            "HGPT_MOTION_TRACKING_START",
            args.policy_ready_timeout_s,
        )
        event_log.append(
            "motion_playing_observed",
            sim_time_s=latest_sim_time(run_dir),
            support_active=0,
            detail=marker,
        )
        wait_for_motion_window(sim, args.motion_duration_s, event_log, run_dir)
    finally:
        update_control_file(control_file, stop=True)
        policy.terminate()


def clean_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def run_release_no_control_case(args: argparse.Namespace, root: Path) -> None:
    run_dir = root / "no_control_direct_release"
    clean_run_dir(run_dir)
    control_file = run_dir / "sim_control.json"
    event_log = SequenceEventLog(run_dir / "sequence_events.csv")
    sim_name = f"wbc-sim-release-no-control-{int(time.time())}"
    sim = start_simulator(
        run_dir,
        control_file,
        4.0,
        sim_name,
        "holomotion",
        MOTIONS[0],
        args.support_height,
        args.fall_stop_base_z,
        getattr(args, "backend", "mujoco"),
    )
    try:
        event_log.append(
            "release_validation_case_start",
            sim_time_s=latest_sim_time(run_dir),
            support_active=1,
            detail="no-control direct release must fall within 2s",
        )
        release_support(run_dir, control_file, event_log, "direct release with no active controller")
        wait_for_sim_window(sim, 2.5)
        event_log.append(
            "release_validation_case_end",
            sim_time_s=latest_sim_time(run_dir),
            support_active=(latest_sim_status(run_dir) or {}).get("support_active", ""),
            detail="no-control direct release window complete",
        )
    finally:
        update_control_file(control_file, stop=True)
        sim.terminate()
        docker_rm_force(sim_name)


def run_release_sonic_control_case(args: argparse.Namespace, root: Path) -> None:
    motion = args.motion or MOTIONS[0]
    run_dir = root / "sonic_control_release_stop"
    clean_run_dir(run_dir)
    control_file = run_dir / "sim_control.json"
    event_log = SequenceEventLog(run_dir / "sequence_events.csv")
    ensure_sonic_built()
    sim_name = f"wbc-sim-release-sonic-{int(time.time())}"
    sim = start_simulator(
        run_dir,
        control_file,
        args.policy_ready_timeout_s + 12.0,
        sim_name,
        "sonic",
        motion,
        args.support_height,
        args.fall_stop_base_z,
        getattr(args, "backend", "mujoco"),
    )
    policy_name = f"wbc-sonic-release-{int(time.time())}"
    policy = None
    try:
        motion_root = copy_sonic_single_motion(run_dir, motion)
        stock_csv_dir = run_dir / "csv"
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
            f"--logs-dir /workspace/wbc/{stock_csv_dir.relative_to(ROOT)}"
        )
        policy = ManagedProcess(
            docker_base_args(policy_name, SONIC_IMAGE, tty=True) + ["bash", "-lc", script],
            run_dir / "sonic_stdout.log",
            use_pty=True,
        )
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
        code = wait_for_sim_window(sim, 5.25)
        if code is not None:
            event_log.append(
                "simulator_ended_before_stable_control_window",
                sim_time_s=latest_sim_time(run_dir),
                support_active=0,
                detail=f"simulator exited with code {code}",
            )
            return
        event_log.append(
            "stable_released_control_elapsed",
            sim_time_s=latest_sim_time(run_dir),
            support_active=0,
            detail="released CONTROL remained standing for more than 5s",
        )
        policy.terminate(timeout=2.0)
        docker_rm_force(policy_name)
        policy = None
        stopped = wait_for_cmd_active_state(run_dir, 0, 2.0)
        event_log.append(
            "control_stopped",
            sim_time_s=float(stopped["sim_time_s"]),
            support_active=0,
            detail="stock SONIC policy process terminated; simulator continues with cmd timeout",
        )
        wait_for_sim_window(sim, 2.5)
        event_log.append(
            "post_control_stop_window_elapsed",
            sim_time_s=latest_sim_time(run_dir),
            support_active=0,
            detail="control-stop fall window complete",
        )
    finally:
        update_control_file(control_file, stop=True)
        if policy is not None:
            policy.terminate()
        sim.terminate()
        docker_rm_force(policy_name)
        docker_rm_force(sim_name)


def run_release_validation(args: argparse.Namespace) -> int:
    cleanup_stale_benchmark_containers()
    logs_root = Path(args.logs)
    if not logs_root.is_absolute():
        logs_root = ROOT / logs_root
    root = logs_root / "release_validation"
    if not getattr(args, "skip_gpu_preflight", False):
        probe = probe_gpu_runtime()
        if not probe["ok"]:
            raise RuntimeError(str(probe.get("reason") or "GPU runtime preflight failed"))
    execution_failures = []
    try:
        run_release_no_control_case(args, root)
    except Exception as exc:
        execution_failures.append(f"no_control_direct_release execution failed: {exc}")
    try:
        run_release_sonic_control_case(args, root)
    except Exception as exc:
        execution_failures.append(f"sonic_control_release_stop execution failed: {exc}")
    proof = validate_release_validation_logs(logs_root)
    if execution_failures:
        proof["passed"] = False
        proof["failures"].extend(execution_failures)
        (root / "execution_failures.json").write_text(json.dumps(execution_failures, indent=2) + "\n")
    print(json.dumps({"ok": proof["passed"], "proof": proof}, indent=2))
    return 0 if proof["passed"] else 1


def run_motion(args: argparse.Namespace) -> int:
    cleanup_stale_benchmark_containers()
    logs_root = Path(args.logs)
    if not logs_root.is_absolute():
        logs_root = ROOT / logs_root
    if not getattr(args, "skip_gpu_preflight", False):
        probe = probe_gpu_runtime()
        if not probe["ok"]:
            raise RuntimeError(str(probe.get("reason") or "GPU runtime preflight failed"))
    run_dir = Path(args.logs) / args.policy / args.motion
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    clean_run_dir(run_dir)
    control_file = run_dir / "sim_control.json"
    event_path = run_dir / "sequence_events.csv"
    event_log = SequenceEventLog(event_path)
    args.motion_duration_s = args.duration_s or policy_motion_duration(args.policy, args.motion)
    sim_duration = args.motion_duration_s + args.startup_margin_s
    if args.policy == "sonic":
        ensure_sonic_built()
    sim_name = f"wbc-sim-{args.policy}-{args.motion[:24]}-{int(time.time())}"
    sim = start_simulator(
        run_dir,
        control_file,
        sim_duration,
        sim_name,
        args.policy,
        args.motion,
        args.support_height,
        args.fall_stop_base_z,
        args.backend,
    )
    try:
        if args.policy == "sonic":
            run_sonic_sequence(args, run_dir, control_file, event_log, sim)
        elif args.policy == "humanoid-gpt":
            run_humanoid_gpt_sequence(args, run_dir, control_file, event_log, sim)
        else:
            run_holomotion_sequence(args, run_dir, control_file, event_log, sim)
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
            raise RuntimeError(str(probe.get("reason") or "GPU runtime preflight failed"))
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
