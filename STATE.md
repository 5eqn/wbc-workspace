# STATE

## Eras

| Datetime | Codename | Description |
| --- | --- | --- |
| 2026-06-15T16:38:34+08:00 | INIT | Initial inferred repository era captured after the repo already contained benchmark scripts, Dockerfiles, assets, logs, artifacts, Humanoid-GPT integration work, and collaborator instruction files. |
| 2026-06-15T16:50:32+08:00 | REPO-SKILLS | Added repo-level Codex skills under `.agents/skills`, rewrote `AGENTS.md` around skills, and converted `STATE.md` to eras, files, and skills. |
| 2026-06-15T16:55:06+08:00 | CLEAN-WORKTREE | Added vendored ignore rules for local benchmark and model assets, advanced the related submodule gitlinks, and prepared the parent repository for a clean committed state. |
| 2026-06-15T17:30:00+08:00 | HOST-SONIC-SPLIT | Verified that `go2-mjlab` can run the host-side MuJoCo DDS bridge directly, and confirmed the split path where interactive SONIC runs in `wbc-gear-sonic` over `--network host`; noted that simulator duration must exceed SONIC TensorRT init time. |
| 2026-06-15T17:38:00+08:00 | SONIC-MANUAL-SKILL | Added a repo-level skill for guiding manual SONIC Sim2Sim runs with host-side MuJoCo plus interactive Docker SONIC, including the release-order workflow and sim-bridge rationale. |
| 2026-06-15T18:19:52+08:00 | HOST-UNITREE-MANUAL-VALIDATION | Reworked the SONIC manual skill around benchmark-style single-motion staging plus the host `unitree_mujoco` 29-DOF scene, and recorded the validated manual sweep outcome: `8/10` RMSE-only, `0/10` no-fall. |
| 2026-06-16T09:31:57+08:00 | SONIC-AUTO-RERUN | Re-ran one automated SONIC MuJoCo benchmark motion on the current tree and confirmed the benchmark path still passes on `dance_chicken_c03_neutral2s`. |
| 2026-06-16T09:48:13+08:00 | SONIC-MANUAL-VIEWER-FLOW | Added optional host MuJoCo viewer support to `scripts/sim_bridge.py` and simplified the manual SONIC skill to a two-command workflow that uses SONIC's in-tree `reference/benchmark` motion library for manual `N/P` selection. |
| 2026-06-16T10:37:34+08:00 | SONIC-MANUAL-INTERACTIVE-WINDOW | Changed `scripts/sim_bridge.py --viewer` to open MuJoCo's full interactive Simulate window via the bridge-owned loop, and disabled fall early-stop while that window mode is active so manual reset can be used. |
| 2026-06-16T11:09:14+08:00 | SONIC-MANUAL-BENCHMARK-SCENE | Switched the manual SONIC host workflow to the same benchmark-owned `GR00T-WholeBodyControl` G1 29-DOF scene used by the automated path, instead of the `unitree_mujoco` host scene. |

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
| `scripts/sim_bridge.py` | SONIC-MANUAL-INTERACTIVE-WINDOW | Host-side MuJoCo DDS bridge for split SONIC manual-control runs, with optional interactive Simulate window launch, explicit `DISPLAY` failure reporting, and viewer-mode fall-stop suppression. |
| `scripts/fallprobe_*.py` | INIT | Fall-probing analysis helpers. |
| `scripts/humanoid_gpt_*.py` | INIT | Humanoid-GPT translation, built-in evaluation, and deploy entrypoints. |
| `thirdparties/run-sonic/GR00T-WholeBodyControl-main/verification/MANUAL_TEST_GUIDE.md` | HOST-SONIC-SPLIT | Manual SONIC/HoloMotion shared-simulator guide that documents the interactive `N/P`, `]`, and `T` control flow and support-release ordering. |
| `thirdparties/GR00T-WholeBodyControl/gear_sonic_deploy/reference/benchmark/*` | SONIC-MANUAL-VIEWER-FLOW | SONIC's in-tree multi-motion reference library used by the simplified manual `N/P` browsing workflow. |
| `.agents/skills/sonic-manual-sim2sim/*` | SONIC-MANUAL-BENCHMARK-SCENE | Repo skill for the simplified manual SONIC workflow: host interactive-window bridge on the benchmark-owned G1 29-DOF scene, interactive Docker SONIC, and in-tree `reference/benchmark` motion browsing. |
| `logs/mujoco-backend-sim2sim/sonic/dance_chicken_c03_neutral2s/*` | SONIC-AUTO-RERUN | Current-tree automated SONIC MuJoCo rerun proving the benchmark path still passes one representative motion with valid release order, RMSE, and no fall stop. |
| `logs/manual-sonic-host-unitree-sweep/*` | HOST-UNITREE-MANUAL-VALIDATION | Per-motion host-side manual SONIC sweep logs against the `unitree_mujoco` scene, including simulator, deploy, and scoring inputs. |
| `artifacts/manual-sonic-host-unitree-sweep/*` | HOST-UNITREE-MANUAL-VALIDATION | Aggregated summary proving the current host-unitree manual path reaches `8/10` RMSE-only and `0/10` full no-fall passes. |
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
| `sonic-manual-sim2sim` | `.agents/skills/sonic-manual-sim2sim` | SONIC-MANUAL-BENCHMARK-SCENE | Manual SONIC Sim2Sim, host MuJoCo split-run, or interactive SONIC guidance requests | Guide the simplified host interactive-window plus Docker-SONIC workflow against the benchmark-owned `GR00T-WholeBodyControl` G1 29-DOF scene using SONIC's in-tree `reference/benchmark` motion library and the manual `N/P`, `]`, `T`, `R`, and `O` controls. |
| `humanoid-gpt-translation` | `.agents/skills/humanoid-gpt-translation` | REPO-SKILLS | Humanoid-GPT translation or equivalence requests | Translate SONIC and HoloMotion benchmark motions into Humanoid-GPT format and verify equivalence. |
| `humanoid-gpt-builtin-eval` | `.agents/skills/humanoid-gpt-builtin-eval` | REPO-SKILLS | Official Humanoid-GPT evaluation requests | Run official Humanoid-GPT evaluation and render output videos for translated motions. |
| `humanoid-gpt-sim2sim` | `.agents/skills/humanoid-gpt-sim2sim` | REPO-SKILLS | Shared MuJoCo Humanoid-GPT benchmark requests | Run Humanoid-GPT inside the shared benchmark runtime path and regenerate shared backend reports. |
