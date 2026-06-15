# STATE

## Eras

| Datetime | Codename | Description |
| --- | --- | --- |
| 2026-06-15T16:38:34+08:00 | INIT | Initial inferred repository era captured after the repo already contained benchmark scripts, Dockerfiles, assets, logs, artifacts, Humanoid-GPT integration work, and collaborator instruction files. |
| 2026-06-15T16:50:32+08:00 | REPO-SKILLS | Added repo-level Codex skills under `.agents/skills`, rewrote `AGENTS.md` around skills, and converted `STATE.md` to eras, files, and skills. |
| 2026-06-15T16:55:06+08:00 | CLEAN-WORKTREE | Added vendored ignore rules for local benchmark and model assets, advanced the related submodule gitlinks, and prepared the parent repository for a clean committed state. |

## Files

| Path | Era | Description |
| --- | --- | --- |
| `AGENTS.md` | REPO-SKILLS | Root repository instructions centered on goals, state tracking, and repo-level skills. |
| `STATE.md` | REPO-SKILLS | Repository state tracker with eras, files, and skills. |
| `CLAUDE.md` | INIT | Alternate collaborator instruction file. |
| `GOAL.md` | INIT | Active Humanoid-GPT Sim2Sim goal and verification surface. |
| `HOLOMOTION_V1_3_ARCHITECTURE_IMAGE_PROOF.md` | INIT | HoloMotion architecture audit note. |
| `HOLOMOTION_V1_3_MOTION_TRACKING_PARAMETER_COUNT.md` | INIT | HoloMotion parameter-count audit note. |
| `docker/*.Dockerfile` | INIT | Docker build definitions for MuJoCo, Isaac Sim, SONIC, and HoloMotion environments. |
| `scripts/*.sh` | INIT | Shell entrypoints for download, run, report, and fallprobe tasks. |
| `scripts/benchmark*.py` | INIT | Benchmark CLI, shared constants, runtime orchestration, validation checks, and report generation. |
| `scripts/*bridge*.py` | INIT | MuJoCo, Isaac, and ROS2 bridge helpers. |
| `scripts/fallprobe_*.py` | INIT | Fall-probing analysis helpers. |
| `scripts/humanoid_gpt_*.py` | INIT | Humanoid-GPT translation, built-in evaluation, and deploy entrypoints. |
| `.agents/skills/benchmark-assets-bootstrap/*` | REPO-SKILLS | Repo skill for asset bootstrap and pinned third-party refresh. |
| `.agents/skills/benchmark-build-validate/*` | REPO-SKILLS | Repo skill for image builds, validation, and smoke checks. |
| `.agents/skills/benchmark-single-motion/*` | REPO-SKILLS | Repo skill for single policy-motion backend runs. |
| `.agents/skills/benchmark-full-backend/*` | REPO-SKILLS | Repo skill for backend sweeps and report generation. |
| `.agents/skills/humanoid-gpt-translation/*` | REPO-SKILLS | Repo skill for Humanoid-GPT translation and equivalence checks. |
| `.agents/skills/humanoid-gpt-builtin-eval/*` | REPO-SKILLS | Repo skill for official Humanoid-GPT evaluation and video rendering. |
| `.agents/skills/humanoid-gpt-sim2sim/*` | REPO-SKILLS | Repo skill for Humanoid-GPT runs inside the shared benchmark path. |
| `thirdparties/GR00T-WholeBodyControl/.gitignore` | CLEAN-WORKTREE | Vendored ignore rules for local benchmark reference assets. |
| `thirdparties/HoloMotion/.gitignore` | CLEAN-WORKTREE | Vendored ignore rules for local deployment launch and model assets. |
| `artifacts/humanoid-gpt-translation/*` | INIT | Translation-equivalence artifacts and joint-mapping reports. |
| `artifacts/humanoid-gpt-builtin-eval/*` | INIT | Humanoid-GPT built-in evaluation summaries and video indexes. |
| `artifacts/mujoco-backend-sim2sim/*` | INIT | MuJoCo backend reports and comparison videos. |
| `artifacts/isaac-backend-sim2sim/*` | INIT | Isaac backend reports and comparison videos. |
| `logs/humanoid-gpt-translation/*` | INIT | Saved translated motions and per-motion equivalence logs. |
| `logs/humanoid-gpt-builtin-eval/*` | INIT | Native Humanoid-GPT evaluation logs and rendered videos. |
| `logs/mujoco-backend-sim2sim/*` | INIT | Raw MuJoCo benchmark runs for SONIC, HoloMotion, Humanoid-GPT, and release validation. |
| `logs/isaac-backend-sim2sim/*` | INIT | Raw Isaac benchmark runs for SONIC, HoloMotion, and release validation. |
| `logs/isaac-smoke/*` | INIT | Isaac simulator smoke outputs. |
| `logs/isaac-urdf-smoke/*` | INIT | Isaac URDF smoke outputs. |
| `logs/isaac-usd-smoke/*` | INIT | Isaac USD smoke outputs. |
| `logs/quick_unitree_sdk2py/*` | INIT | Quick Unitree SDK2 Python smoke outputs. |

## Skills

| Skill | Path | Era | Trigger | Purpose |
| --- | --- | --- | --- | --- |
| `benchmark-assets-bootstrap` | `.agents/skills/benchmark-assets-bootstrap` | REPO-SKILLS | Asset bootstrap, refresh, or preparation requests | Populate `assets/`, refresh pinned third-party checkouts, validate raw assets, and prepare stock motion assets. |
| `benchmark-build-validate` | `.agents/skills/benchmark-build-validate` | REPO-SKILLS | Build, validate, or smoke-check requests | Build benchmark images, validate deploy state, and run smoke checks. |
| `benchmark-single-motion` | `.agents/skills/benchmark-single-motion` | REPO-SKILLS | Single run, debug run, or one-motion inspection requests | Run one policy-motion pair on one backend and inspect the resulting log tree. |
| `benchmark-full-backend` | `.agents/skills/benchmark-full-backend` | REPO-SKILLS | Full benchmark sweep or backend report requests | Run release validation, sweep all motions, and generate backend report artifacts. |
| `humanoid-gpt-translation` | `.agents/skills/humanoid-gpt-translation` | REPO-SKILLS | Humanoid-GPT translation or equivalence requests | Translate SONIC and HoloMotion benchmark motions into Humanoid-GPT format and verify equivalence. |
| `humanoid-gpt-builtin-eval` | `.agents/skills/humanoid-gpt-builtin-eval` | REPO-SKILLS | Official Humanoid-GPT evaluation requests | Run official Humanoid-GPT evaluation and render output videos for translated motions. |
| `humanoid-gpt-sim2sim` | `.agents/skills/humanoid-gpt-sim2sim` | REPO-SKILLS | Shared MuJoCo Humanoid-GPT benchmark requests | Run Humanoid-GPT inside the shared benchmark runtime path and regenerate shared backend reports. |
