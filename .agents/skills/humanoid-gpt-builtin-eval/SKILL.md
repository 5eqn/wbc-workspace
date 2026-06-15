---
name: humanoid-gpt-builtin-eval
description: Run official Humanoid-GPT evaluation and render output videos for translated benchmark motions in this repository. Use when Codex needs native tracking metrics, rendered evaluation videos, or refreshed artifacts from translated SONIC or HoloMotion motion sets.
---

# Humanoid-GPT Builtin Eval

## Run

Run evaluation on translated HoloMotion motions:

```bash
python scripts/humanoid_gpt_builtin_eval.py --source holomotion --device cuda:0 --workers 1
```

Run evaluation on translated SONIC motions:

```bash
python scripts/humanoid_gpt_builtin_eval.py --source sonic --device cuda:0 --workers 1
```

Use the translation skill first so the converted motion files already exist.

## Inspect

Review logs under:

- `logs/humanoid-gpt-builtin-eval/native_metrics/`
- `logs/humanoid-gpt-builtin-eval/videos/`

Review summary artifacts under:

- `artifacts/humanoid-gpt-builtin-eval/metrics_summary.json`
- `artifacts/humanoid-gpt-builtin-eval/per_motion_metrics.csv`
- `artifacts/humanoid-gpt-builtin-eval/videos.json`

## Finish

Treat the run as complete when the metrics summary, per-motion CSV, and video index appear for the requested translated source set.
