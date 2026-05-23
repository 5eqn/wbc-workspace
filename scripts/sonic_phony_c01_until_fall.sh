#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION="dance_phony_c01_neutral2s"
OUT_DIR="${OUT_DIR:-/tmp/wbc-sonic-phony-c01-until-fall}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/.tmp_sonic_phony_c01_until_fall}"
FALL_BASE_Z="${FALL_BASE_Z:-0.25}"
MAX_RUNS="${MAX_RUNS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<EOF
Usage: OUT_DIR=/tmp/wbc-sonic-phony-c01-until-fall MAX_RUNS=0 $0

Runs SONIC on $MOTION until the released replay falls.

Environment:
  OUT_DIR      Final per-iteration logs and summaries. Default: $OUT_DIR
  WORK_DIR     Transient repo-local logs directory required by Docker mounts.
               Default: $WORK_DIR
  FALL_BASE_Z  Fall threshold on released replay base_z. Default: $FALL_BASE_Z
  MAX_RUNS     0 means infinite. Positive value stops after that many passes.
  PYTHON_BIN   Python interpreter for post-run metric extraction.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

mkdir -p "$OUT_DIR"
rm -rf "$WORK_DIR"
printf 'iteration,ok,fell,rmse,delay,base_z_min,error\n' > "$OUT_DIR/summary.csv"

iteration=0
while :; do
  iteration=$((iteration + 1))
  iter_name="$(printf 'run_%04d' "$iteration")"
  iter_out="$OUT_DIR/$iter_name"
  rm -rf "$WORK_DIR" "$iter_out"
  mkdir -p "$WORK_DIR" "$iter_out"

  echo "[sonic-phony-c01] iteration $iteration start"
  if ! "$ROOT_DIR/scripts/run.sh" run-motion sonic "$MOTION" --logs "$WORK_DIR/logs" \
      > "$iter_out/run_motion.stdout" 2> "$iter_out/run_motion.stderr"; then
    error="$(tail -c 1000 "$iter_out/run_motion.stderr" | tr '\n,' '  ')"
    [[ -d "$WORK_DIR/logs" ]] && mv "$WORK_DIR/logs" "$iter_out/logs"
    printf '%d,false,false,,,,%s\n' "$iteration" "$error" >> "$OUT_DIR/summary.csv"
    echo "[sonic-phony-c01] iteration $iteration run failed; logs: $iter_out" >&2
    exit 2
  fi

  mv "$WORK_DIR/logs" "$iter_out/logs"
  PYTHONPATH="$ROOT_DIR/scripts" "$PYTHON_BIN" - "$iter_out" "$MOTION" "$FALL_BASE_Z" <<'PY' > "$iter_out/metrics.env"
import sys
from pathlib import Path

from benchmark_common import (
    RELEASE_CONFIRMED_EVENT,
    event_sim_time,
    expected_run_dir,
    first_event,
    load_events,
)
from benchmark_report import aligned_rmse, load_reference, load_replay, load_tracked, tracking_window

root = Path(sys.argv[1])
motion = sys.argv[2]
fall_base_z = float(sys.argv[3])
run_dir = expected_run_dir(root / "logs", "sonic", motion)

ref, hz = load_reference("sonic", motion)
t, q = load_tracked(run_dir)
t_eval, q_eval, _ = tracking_window(run_dir, t, q)
metrics = aligned_rmse(ref, hz, t_eval, q_eval)

events = load_events(run_dir)
release_t = event_sim_time(first_event(events, RELEASE_CONFIRMED_EVENT))
replay = load_replay(run_dir)
released = replay["sim_time_s"] >= release_t
base_z_min = float(replay["base_z"][released].min())
fell = base_z_min < fall_base_z

print(f"rmse={metrics['mean_joint_rmse_rad']}")
print(f"delay={metrics['tracking_delay_s']}")
print(f"base_z_min={base_z_min}")
print(f"fell={'true' if fell else 'false'}")
PY

  # shellcheck disable=SC1090
  source "$iter_out/metrics.env"
  printf '%d,true,%s,%s,%s,%s,\n' \
    "$iteration" "$fell" "$rmse" "$delay" "$base_z_min" >> "$OUT_DIR/summary.csv"
  echo "[sonic-phony-c01] iteration $iteration ok fell=$fell rmse=$rmse delay=$delay base_z_min=$base_z_min"

  if [[ "$fell" == "true" ]]; then
    echo "[sonic-phony-c01] fall reproduced on iteration $iteration; logs: $iter_out"
    exit 0
  fi

  if [[ "$MAX_RUNS" -gt 0 && "$iteration" -ge "$MAX_RUNS" ]]; then
    echo "[sonic-phony-c01] reached MAX_RUNS=$MAX_RUNS without fall; summary: $OUT_DIR/summary.csv"
    exit 1
  fi
done
