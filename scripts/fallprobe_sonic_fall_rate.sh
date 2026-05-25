#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/wbc-sonic-fallprobe}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/.tmp_fallprobe_sonic}"
FALL_BASE_Z="${FALL_BASE_Z:-0.25}"
MAX_RUNS="${MAX_RUNS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<EOF
Usage: $0 [CLIP_NUM ...] [--clip CLIP_NUM ...]

Runs SONIC repeatedly and reports the observed fall chance per selected motion.
Clip numbers are 1-10 in benchmark MOTIONS order. With no clip, clip 4 is used.

Examples:
  $0 4
  $0 --clip 1 4 7
  OUT_DIR=/tmp/wbc-fallprobe MAX_RUNS=30 $0 --clip 4

Environment:
  OUT_DIR      Final per-attempt logs and summaries. Default: $OUT_DIR
  WORK_DIR     Transient repo-local logs directory required by Docker mounts.
               Default: $WORK_DIR
  FALL_BASE_Z  Fall threshold on released replay base_z. Default: $FALL_BASE_Z
  MAX_RUNS     0 means infinite. Positive value stops after total attempts.
  PYTHON_BIN   Python interpreter for post-run metric extraction.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

clip_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clip)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        clip_args+=("$1")
        shift
      done
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        clip_args+=("$1")
        shift
      done
      ;;
    --*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      clip_args+=("$1")
      shift
      ;;
  esac
done
if [[ ${#clip_args[@]} -eq 0 ]]; then
  clip_args=(4)
fi

motion_list="$(
  PYTHONPATH="$ROOT_DIR/scripts" "$PYTHON_BIN" - "${clip_args[@]}" <<'PY'
import sys
from benchmark_common import MOTIONS

seen = set()
selected = []
for raw in sys.argv[1:]:
    try:
        index = int(raw)
    except ValueError as exc:
        raise SystemExit(f"clip must be an integer 1-10, got {raw!r}") from exc
    if not 1 <= index <= len(MOTIONS):
        raise SystemExit(f"clip must be in 1-{len(MOTIONS)}, got {index}")
    motion = MOTIONS[index - 1]
    if motion not in seen:
        seen.add(motion)
        selected.append(motion)
for motion in selected:
    print(motion)
PY
)"
mapfile -t motions <<< "$motion_list"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

mkdir -p "$OUT_DIR"
rm -rf "$WORK_DIR"
printf 'attempt,motion,ok,fell,fall_rate,total_attempts,motion_attempts,motion_falls,rmse,delay,base_z_min,error\n' > "$OUT_DIR/summary.csv"
printf 'attempt,motion,run_dir,rmse,delay,base_z_min\n' > "$OUT_DIR/failed_attempts.csv"

echo "[fallprobe] output: $OUT_DIR"
echo "[fallprobe] clips: ${motions[*]}"

attempt=0
while :; do
  for motion in "${motions[@]}"; do
    attempt=$((attempt + 1))
    iter_name="$(printf 'attempt_%05d_%s' "$attempt" "$motion")"
    iter_out="$OUT_DIR/$iter_name"
    rm -rf "$WORK_DIR" "$iter_out"
    mkdir -p "$WORK_DIR" "$iter_out"

    echo "[fallprobe] attempt $attempt motion=$motion start"
    if ! "$ROOT_DIR/scripts/run.sh" run-motion sonic "$motion" --logs "$WORK_DIR/logs" \
        > "$iter_out/run_motion.stdout" 2> "$iter_out/run_motion.stderr"; then
      error="$(tail -c 1000 "$iter_out/run_motion.stderr" | tr '\n,' '  ')"
      [[ -d "$WORK_DIR/logs" ]] && mv "$WORK_DIR/logs" "$iter_out/logs"
      PYTHONPATH="$ROOT_DIR/scripts" "$PYTHON_BIN" - "$OUT_DIR" "$attempt" "$motion" false false "" "" "" "$error" <<'PY'
import sys
from pathlib import Path

from fallprobe_common import append_summary

append_summary(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4] == "true",
               sys.argv[5] == "true", sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])
PY
      echo "[fallprobe] attempt $attempt motion=$motion run failed; logs: $iter_out" >&2
      if [[ "$MAX_RUNS" -gt 0 && "$attempt" -ge "$MAX_RUNS" ]]; then
        exit 2
      fi
      continue
    fi

    mv "$WORK_DIR/logs" "$iter_out/logs"
    PYTHONPATH="$ROOT_DIR/scripts" "$PYTHON_BIN" - "$iter_out" "$motion" "$FALL_BASE_Z" "$OUT_DIR" "$attempt" <<'PY' > "$iter_out/metrics.env"
import sys
from pathlib import Path

from fallprobe_common import append_summary, measure_attempt

root = Path(sys.argv[1])
motion = sys.argv[2]
fall_base_z = float(sys.argv[3])
out_dir = Path(sys.argv[4])
attempt = int(sys.argv[5])

metrics = measure_attempt(root, motion, fall_base_z)
stats = append_summary(
    out_dir,
    attempt,
    motion,
    True,
    bool(metrics["fell"]),
    metrics["rmse"],
    metrics["delay"],
    metrics["base_z_min"],
    "",
)
if metrics["fell"]:
    with (out_dir / "failed_attempts.csv").open("a", newline="") as f:
        f.write(
            f"{attempt},{motion},{root},{metrics['rmse']},"
            f"{metrics['delay']},{metrics['base_z_min']}\n"
        )

print(f"rmse={metrics['rmse']}")
print(f"delay={metrics['delay']}")
print(f"base_z_min={metrics['base_z_min']}")
print(f"fell={'true' if metrics['fell'] else 'false'}")
print(f"fall_rate={stats['fall_rate']}")
print(f"motion_attempts={stats['motion_attempts']}")
print(f"motion_falls={stats['motion_falls']}")
PY

    # shellcheck disable=SC1090
    source "$iter_out/metrics.env"
    echo "[fallprobe] attempt $attempt motion=$motion fell=$fell fall_rate=$fall_rate ($motion_falls/$motion_attempts) rmse=$rmse delay=$delay base_z_min=$base_z_min"

    if [[ "$MAX_RUNS" -gt 0 && "$attempt" -ge "$MAX_RUNS" ]]; then
      echo "[fallprobe] reached MAX_RUNS=$MAX_RUNS; summary: $OUT_DIR/summary.csv"
      exit 0
    fi
  done
done
