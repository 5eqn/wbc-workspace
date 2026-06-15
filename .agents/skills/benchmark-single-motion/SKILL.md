---
name: benchmark-single-motion
description: Run one policy-motion pair on one backend and inspect the resulting logs for this repository. Use when Codex needs a focused debug run, a quick backend check, or a single-motion evidence pass for `sonic`, `holomotion`, or `humanoid-gpt`.
---

# Benchmark Single Motion

## Run

Choose one policy, one motion, and one backend, then run:

```bash
python scripts/benchmark.py run-motion <policy> <motion> --backend <mujoco|isaac>
```

Valid policy names come from `scripts/benchmark_common.py` and currently include:

- `sonic`
- `holomotion`
- `humanoid-gpt`

## Inspect

Review the run directory:

```text
logs/<backend>-backend-sim2sim/<policy>/<motion>/
```

Look for:

- `sequence_events.csv`
- `simulator_status.csv`
- `replay.csv`
- `sim_control.json`
- policy stdout logs

## Finish

Treat the run as complete when the command succeeds, the run directory exists, and release ordering is valid for the selected policy-motion pair.
