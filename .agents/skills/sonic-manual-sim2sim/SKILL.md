---
name: sonic-manual-sim2sim
description: Guide a manual SONIC Sim2Sim run for this repository when the user wants the benchmark-owned MuJoCo DDS bridge on the host against the benchmark-owned `GR00T-WholeBodyControl` G1 29-DOF scene, a visible host MuJoCo window, and stock interactive Docker SONIC driving SONIC's in-tree motion library. Use for by-hand validation or demos; do not use for the automated `scripts/benchmark.py run-motion` path.
---

# SONIC Manual Sim2Sim

## Overview

Use this skill when the user wants the simplest manual SONIC path in this repo:

- host-side `scripts/sim_bridge.py` against the benchmark-owned `GR00T-WholeBodyControl` G1 29-DOF scene
- a visible interactive MuJoCo window on the host with `--viewer`
- stock SONIC deploy inside `wbc-gear-sonic`
- SONIC's in-tree `reference/benchmark` motion library so `N` / `P` works directly

This workflow intentionally does not stage per-motion run directories, benchmark wrappers, or scoring steps. It is just:

1. Start MuJoCo on the host.
2. Start SONIC in the container.
3. Operate SONIC manually.

## Assumptions

- `wbc-gear-sonic:latest` exists locally.
- `conda run -n go2-mjlab python -c "import mujoco, unitree_sdk2py, cyclonedds"` succeeds.
- `thirdparties/GR00T-WholeBodyControl/gear_sonic_deploy/reference/benchmark` already contains the motion set the user wants to browse with `N` / `P`.
- Old `g1_deploy_onnx_ref` or `scripts/sim_bridge.py` processes are not still running.

## Workflow

### 1. Start MuJoCo on the host

```bash
cd /home/seqn/wbc-workspace

export DISPLAY=:1  # change if your host X server uses a different display
export RUN_DIR="$PWD/logs/manual-sonic-host-unitree/live"
mkdir -p "$RUN_DIR"

conda run --no-capture-output -n go2-mjlab python scripts/sim_bridge.py \
  --sim-root thirdparties/unitree_mujoco \
  --scene thirdparties/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/scene_29dof.xml \
  --robot g1 \
  --interface lo \
  --domain-id 0 \
  --duration-s 3000 \
  --dt 0.005 \
  --log-hz 50 \
  --publish-every 4 \
  --support-height 0.75 \
  --viewer \
  --viewer-fps 60 \
  --out-dir "$RUN_DIR" \
  --control-file "$RUN_DIR/sim_control.json"
```

`--viewer` makes this repo-owned bridge open MuJoCo's interactive Simulate window, including the built-in UI and reset controls. When `--viewer` is enabled, the bridge disables early fall stop so the window reset button can be used after a fall. If the window does not appear, the command fails immediately when `DISPLAY` is missing or unreachable.

### 2. Start SONIC deploy in the container

```bash
cd /home/seqn/wbc-workspace

export RUN_DIR="$PWD/logs/manual-sonic-host-unitree/live"

docker run --rm --name sonic-manual --network host --gpus all -it \
  -v "$PWD:/workspace/wbc" \
  wbc-gear-sonic:latest \
  bash -lc '
    cd /workspace/wbc/thirdparties/GR00T-WholeBodyControl/gear_sonic_deploy &&
    source /opt/ros/humble/setup.bash &&
    source scripts/setup_env.sh &&
    ./target/release/g1_deploy_onnx_ref \
      lo \
      policy/release/model_decoder.onnx \
      /workspace/wbc/thirdparties/GR00T-WholeBodyControl/gear_sonic_deploy/reference/benchmark \
      --obs-config policy/release/observation_config.yaml \
      --encoder-file policy/release/model_encoder.onnx \
      --input-type keyboard \
      --output-type zmq \
      --zmq-host localhost \
      --zmq-out-port 15557 \
      --disable-crc-check \
      --enable-csv-logs \
      --logs-dir /workspace/wbc/logs/manual-sonic-host-unitree/live/csv \
      --target-motion-logfile /workspace/wbc/logs/manual-sonic-host-unitree/live/target_motion.csv \
      --planner-motion-logfile /workspace/wbc/logs/manual-sonic-host-unitree/live/planner_motion.csv \
      --policy-input-logfile /workspace/wbc/logs/manual-sonic-host-unitree/live/policy_input.csv
  '
```

The motion root is now SONIC's own `reference/benchmark` directory, so the full library stays inside SONIC and manual `N` / `P` selection works directly.

## Manual SONIC Guide

1. Wait for `Init Done`.
2. Press `N` / `P` until the console shows the motion you want.
3. Press `]` to enter `CONTROL`.
4. Wait for `transitioning to CONTROL state`.
5. Release host support:

```bash
cd /home/seqn/wbc-workspace
export RUN_DIR="$PWD/logs/manual-sonic-host-unitree/live"

python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_DIR"]) / "sim_control.json"
data = json.loads(path.read_text())
data["support_active"] = False
path.write_text(json.dumps(data, indent=2) + "\n")
PY
```

6. Wait about 2 seconds.
7. Press `T` to play.
8. Use `R` to replay the current motion.
9. Use `O` to stop SONIC and exit.

Useful keys:

- `N` / `P`: next / previous motion
- `]`: enter `CONTROL`
- `T`: play the selected motion
- `R`: replay the selected motion
- `O`: stop and exit

If the user wants a clean restart on a different motion, stop both processes and rerun the two command groups above.
