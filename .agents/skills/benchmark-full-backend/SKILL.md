---
name: benchmark-full-backend
description: Run a full backend benchmark sweep and generate shared report artifacts for this repository. Use when Codex needs release validation, all-motion backend runs, or final MuJoCo or Isaac report output for the shared benchmark pipeline.
---

# Benchmark Full Backend

## Run

For MuJoCo:

```bash
bash scripts/run.sh run-release-validation
bash scripts/run.sh run-all-motions
bash scripts/report.sh mujoco
```

For Isaac:

```bash
bash scripts/run.sh run-release-validation --backend isaac
bash scripts/run.sh run-all-motions --backend isaac
bash scripts/report.sh isaac
```

## Inspect

Check raw logs under:

- `logs/mujoco-backend-sim2sim/`
- `logs/isaac-backend-sim2sim/`

Check report artifacts under:

- `artifacts/mujoco-backend-sim2sim/`
- `artifacts/isaac-backend-sim2sim/`

Key report outputs include:

- `metrics_summary.json`
- `phase_time_proof.json`
- `rmse_summary.csv`
- `per_joint_rmse.csv`
- `release_validation.json`
- `single_robot_interface.json`
- `comparison_videos.json`
- `report.md`

## Finish

Treat the run as complete when release validation, all-motion runs, and report generation succeed for the requested backend and the expected report files are present.
