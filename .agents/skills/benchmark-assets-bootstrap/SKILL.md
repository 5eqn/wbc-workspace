---
name: benchmark-assets-bootstrap
description: Bootstrap benchmark assets, refresh pinned third-party checkouts, and prepare motion and model inputs for this repository. Use when Codex needs to populate or refresh `assets/`, fetch dependencies into `thirdparties/`, validate raw assets, or prepare stock motion assets before builds or runtime checks.
---

# Benchmark Assets Bootstrap

## Run

Use these commands in order:

```bash
git submodule update --init --recursive
bash scripts/download.sh
bash scripts/run.sh validate-assets
bash scripts/run.sh prepare-assets
```

## Inspect

Check for these paths after the run:

- `assets/HoloMotion_models/`
- `assets/GEAR-SONIC/`
- `assets/motions/`
- required roots under `thirdparties/`

## Finish

Treat the run as complete when asset validation succeeds and the required motion and model directories exist.
