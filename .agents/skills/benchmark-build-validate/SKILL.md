---
name: benchmark-build-validate
description: Build benchmark container images, validate deploy state, and run smoke checks for this repository. Use when Codex needs to prepare the Docker environment, verify image health, or run the standard smoke suite before deeper benchmark work.
---

# Benchmark Build Validate

## Run

Build the images that match the requested backend:

```bash
bash scripts/run.sh build
bash scripts/run.sh build-isaac
```

Run validation and smoke checks:

```bash
bash scripts/run.sh validate-deploy
bash scripts/run.sh validate-images
bash scripts/run.sh smoke-release-gate
bash scripts/run.sh smoke-sim-bridge
bash scripts/run.sh smoke-sim-release
bash scripts/run.sh smoke-sonic-build
bash scripts/run.sh smoke-holomotion
```

## Inspect

Review smoke outputs under:

- `logs/isaac-smoke/`
- `logs/isaac-urdf-smoke/`
- `logs/isaac-usd-smoke/`
- backend-specific temporary log trees created by the checks

## Finish

Treat the run as complete when the requested build, validation, and smoke commands succeed and the expected smoke logs appear.
