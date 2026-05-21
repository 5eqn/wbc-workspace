#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  echo "Usage: $0 [build|validate-assets|all]"
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
  all)
    "$0" build
    "$0" validate-assets
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
