#!/usr/bin/env python3
from __future__ import annotations

from benchmark_common import *

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
        docker_check(
            "wbc-unitree_isaacsim",
            "check_path() { test -e \"$1\" || { echo \"missing: $1\"; exit 1; }; }\n"
            "check_path /home/code/IsaacLab/isaaclab.sh\n"
            "check_path /home/code/unitree_sim_isaaclab/tasks/__init__.py\n"
            "conda run -n unitree_isaacsim python - <<'PY'\n"
            "import isaacsim, isaaclab, tasks, unitree_sdk2py\n"
            "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_\n"
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
        "test -s /tmp/wbc-sim-bridge/replay.csv && "
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
        "replay = list(csv.DictReader((root / 'replay.csv').open()))\n"
        "assert len(replay) >= 5, len(replay)\n"
        "assert 'qpos_0' in replay[0] and 'qpos_6' in replay[0]\n"
        "assert 'measured_q_28' in replay[0]\n"
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
