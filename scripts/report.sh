#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT_DIR/artifacts"

docker run --rm \
  -v "$ROOT_DIR:/workspace/wbc" \
  -w /workspace/wbc \
  wbc-unitree_mujoco \
  python3 scripts/benchmark.py report --logs logs --artifacts artifacts
