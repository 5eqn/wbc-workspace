#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:-mujoco}"
LOGS="logs/${BACKEND}-backend-sim2sim"
ARTIFACTS="artifacts/${BACKEND}-backend-sim2sim"
mkdir -p "$ROOT_DIR/$ARTIFACTS"

if [[ "$BACKEND" == "isaac" ]]; then
  docker run --rm --gpus all \
    -e ACCEPT_EULA=Y \
    -e OMNI_KIT_ACCEPT_EULA=Y \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e OMNI_KIT_DISABLE_STARTUP=1 \
    -v "$ROOT_DIR:/workspace/wbc" \
    -w /workspace/wbc \
    wbc-unitree_isaacsim \
    conda run -n unitree_isaacsim python scripts/benchmark.py report --backend "$BACKEND" --logs "$LOGS" --artifacts "$ARTIFACTS"
else
  docker run --rm \
    -e MUJOCO_GL=osmesa \
    -v "$ROOT_DIR:/workspace/wbc" \
    -w /workspace/wbc \
    wbc-unitree_mujoco \
    python3 scripts/benchmark.py report --backend "$BACKEND" --logs "$LOGS" --artifacts "$ARTIFACTS"
fi
