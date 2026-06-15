---
name: humanoid-gpt-sim2sim
description: Run Humanoid-GPT inside the shared benchmark runtime path for this repository and regenerate shared MuJoCo report artifacts. Use when Codex needs a single Humanoid-GPT MuJoCo motion run, a full Humanoid-GPT sweep, or refreshed shared backend reports after translation output is ready.
---

# Humanoid-GPT Sim2Sim

## Prepare

Use the translation skill first, because benchmark runtime reads Humanoid-GPT motions from:

```text
logs/humanoid-gpt-translation/converted_from_holomotion/
```

Prepare the host environment expected by runtime:

- conda environment `h-gpt`
- `CYCLONEDDS_HOME=/home/seqn/cyclonedds/install`

## Run

Run one motion:

```bash
python scripts/benchmark.py run-motion humanoid-gpt <motion> --backend mujoco
```

Run the full MuJoCo Humanoid-GPT sweep:

```bash
python scripts/benchmark.py run-all-motions --backend mujoco
bash scripts/report.sh mujoco
```

## Inspect

Review raw runs under:

```text
logs/mujoco-backend-sim2sim/humanoid-gpt/<motion>/
```

Review refreshed shared report artifacts under:

```text
artifacts/mujoco-backend-sim2sim/
```

## Finish

Treat the run as complete when the Humanoid-GPT run directories appear in the shared MuJoCo log tree and the regenerated MuJoCo report succeeds.
