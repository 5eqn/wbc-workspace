---
name: humanoid-gpt-translation
description: Translate SONIC and HoloMotion benchmark motions into Humanoid-GPT format and verify equivalence for this repository. Use when Codex needs to produce translated `.npz` files, run the translation gate, inspect numeric or FK mismatches, or refresh translation artifacts before Humanoid-GPT evaluation or Sim2Sim runs.
---

# Humanoid-GPT Translation

## Run

Run the full translation gate:

```bash
python scripts/humanoid_gpt_translation.py
```

Run a narrower pass for selected motions:

```bash
python scripts/humanoid_gpt_translation.py --motions <motion-a> <motion-b>
```

## Inspect

Review translated motions and checks under:

- `logs/humanoid-gpt-translation/converted_from_sonic/`
- `logs/humanoid-gpt-translation/converted_from_holomotion/`
- `logs/humanoid-gpt-translation/equivalence_checks/`

Review summary artifacts under:

- `artifacts/humanoid-gpt-translation/translation_equivalence.json`
- `artifacts/humanoid-gpt-translation/per_motion_diffs.csv`
- `artifacts/humanoid-gpt-translation/joint_mapping_report.md`

## Finish

Treat the run as complete when `translation_equivalence.json` exists and reports a passing gate for the selected motion set.
