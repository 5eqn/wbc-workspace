#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE_DIR="${OUT_DIR:-/tmp/wbc-sonic-fallprobe}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  PROBE_DIR="$1"
  shift
fi
extra_mount=()
if [[ "$PROBE_DIR" = /* && "$PROBE_DIR" != /tmp/* && "$PROBE_DIR" != "$ROOT_DIR"* ]]; then
  extra_mount=(-v "$PROBE_DIR:$PROBE_DIR")
fi

docker run --rm \
  -e MUJOCO_GL=osmesa \
  -v "$ROOT_DIR:/workspace/wbc" \
  -v /tmp:/tmp \
  "${extra_mount[@]}" \
  -w /workspace/wbc \
  wbc-unitree_mujoco \
  python3 scripts/fallprobe_render_failures.py "$PROBE_DIR" "$@"
