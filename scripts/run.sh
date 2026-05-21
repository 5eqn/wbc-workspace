#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  echo "Usage: $0 [build|validate-assets|prepare-assets|validate-deploy|validate-images|smoke-holomotion|smoke-sonic-build|all]"
}

cmd="${1:-all}"
case "$cmd" in
  build)
    "$PYTHON_BIN" "$ROOT_DIR/scripts/benchmark.py" docker-build unitree_mujoco
    "$PYTHON_BIN" "$ROOT_DIR/scripts/benchmark.py" docker-build gear-sonic
    "$PYTHON_BIN" "$ROOT_DIR/scripts/benchmark.py" docker-build holomotion
    ;;
  validate-assets)
    docker run --rm -v "$ROOT_DIR:/workspace/wbc" -w /workspace/wbc wbc-unitree_mujoco \
      python3 scripts/benchmark.py validate-assets
    ;;
  prepare-assets)
    docker run --rm -v "$ROOT_DIR:/workspace/wbc" -w /workspace/wbc wbc-unitree_mujoco \
      python3 scripts/benchmark.py prepare-assets
    ;;
  validate-deploy)
    docker run --rm -v "$ROOT_DIR:/workspace/wbc" -w /workspace/wbc wbc-unitree_mujoco \
      python3 scripts/benchmark.py validate-deploy
    ;;
  validate-images)
    "$PYTHON_BIN" "$ROOT_DIR/scripts/benchmark.py" validate-images
    ;;
  smoke-holomotion)
    "$PYTHON_BIN" "$ROOT_DIR/scripts/benchmark.py" smoke-holomotion
    ;;
  smoke-sonic-build)
    "$PYTHON_BIN" "$ROOT_DIR/scripts/benchmark.py" smoke-sonic-build
    ;;
  all)
    "$0" validate-assets
    "$0" prepare-assets
    "$0" build
    "$0" validate-deploy
    "$0" validate-images
    "$0" smoke-sonic-build
    "$0" smoke-holomotion
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
