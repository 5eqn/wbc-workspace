---
name: heracles-sonic-reproduction
description: Preprocess, train, export, attach, and evaluate the repo-owned 35D Heracles planner with SONIC. Use when Codex needs to reproduce the LAFAN1 planner, inspect its 72-hour training gate, run ONNX parity or 25 Hz checks, serve SONIC protocol v1 references, or summarize paired Heracles-vs-SONIC trials.
---

# Heracles SONIC Reproduction

## Prepare

Read `scripts/heracles_planner/README.md` and the latest Heracles era in `STATE.md`. Preserve the
read-only dataset and third-party checkouts.

```bash
uv sync --project scripts/heracles_planner --frozen
uv run --project scripts/heracles_planner heracles-planner preprocess
uv run --project scripts/heracles_planner pytest -q scripts/heracles_planner/tests \
  -o cache_dir=scripts/heracles_planner/.pytest_cache
```

Verify `logs/heracles-planner/data/manifest.json` contains 40 motions and the fixed 26/7/7 split.

## Gate training

```bash
uv run --project scripts/heracles_planner heracles-planner benchmark-training --steps 20
```

Read `artifacts/heracles-planner/training_duration_gate.json`. Do not run full training when
`full_training_allowed` is false. Require human direction before changing any reported or chosen
parameter.

## Train and export

Only after the duration gate passes:

```bash
uv run --project scripts/heracles_planner heracles-planner train
uv run --project scripts/heracles_planner heracles-planner export \
  --checkpoint artifacts/heracles-planner/checkpoints/best.pt
uv run --project scripts/heracles_planner heracles-planner benchmark-inference \
  --model artifacts/heracles-planner/heracles.onnx
```

Require passing deterministic ONNX parity and sustained 25 Hz before attachment.

## Inspect checkpoints

When human review is requested before attachment, generate the deterministic four-lag best/last
pose videos directly from the PyTorch checkpoints:

```bash
uv run --project scripts/heracles_planner heracles-planner debug-videos
```

Open `artifacts/heracles-planner/debug-videos/index.html`. Require four videos per checkpoint and
verify `manifest.json` reports 50 FPS with decoded frame count equal to the selected source length.
This offline review does not replace final ONNX parity, rate, or integrated tracking gates.

## Attach and evaluate

Enable the simulator's opt-in `--state-zmq-port 15558` feed. Run the host `serve` command against
the chosen preprocessed motion and start SONIC in protocol-v1 ZMQ input mode. Preserve paired raw
states, references, actions, generated trajectories, disturbance vectors, failures, and seeds.

Use `heracles-planner summarize-evaluation` for metrics and deterministic comparison-video
selection. Do not claim integrated outcomes or create placeholder videos when upstream gates have
not passed.
