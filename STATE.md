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
| 2026-07-08T21:02:00+08:00 | BFM-ZERO-FALLEN-RECOVERY | Added a repo-owned three-stage fallen-recovery workflow around `thirdparties/BFM-Zero-deploy`: Stage 1 static fallen-state generation, Stage 2 isolated simulator+deployer replays with keyboard-driven goal selection, Stage 3 offline metrics and tiled video generation, plus failure-only 1080p replay exports and a tracked initial-state failure analysis report. |
| 2026-07-09T07:04:33+08:00 | BFM-ZERO-STAGE1-STRICT-SAMPLING | Tightened the repo-owned BFM-Zero fallen-recovery workflow so Stage 1 random states must start free of raw ground/self contact, then settle until stable before fallen-state acceptance; Stage 1 now records passive-settle trajectories for all accepted states, Stage 3 renders tiled settle videos plus full-resolution failed settle videos, and the two goal-derived states remain on the original direct frame-extraction path. |
| 2026-07-09T09:23:10+08:00 | BFM-ZERO-STRICT-RUN-A | Completed the first full strict-sampling fallen-recovery sweep as `fullstage1-bfm-zero-20260709a`: `98` accepted random states plus `2` unchanged goal-derived states, `92/100` Stage 2 successes, refreshed failure analysis, tiled Stage 1/Stage 2 videos, and paired full-resolution failed recovery plus failed passive-settle exports. |
| 2026-07-09T11:37:02+08:00 | BFM-ZERO-IN-SIM-DISTURBANCE | Added an opt-in in-simulator fall-induction path for BFM-Zero recovery evaluation: shared deploy-side disturbance and lowcmd tracking hooks, elastic-band preload plus first-lowcmd auto-release, Stage 2 accepted-perturbation retries from standing, optional nominal-vs-0.1x disturbance tuning sweeps, and Stage 3 auto-skip of Stage 1 settle artifacts when no Stage 1 manifest exists. |
| 2026-07-09T14:38:04+08:00 | BFM-ZERO-INDUCED-WRENCH3X-RUN | Raised the induced-mode wrench defaults to `3x` (`1800-5400 N`, `600-1800 Nm`), fixed Stage 3 to handle variable-length induced trajectories, and completed the 100-run induced sweep `induced-full-20260709b-wrench3x`: `100/100` accepted perturbations, `51/100` recoveries overall, with wrench runs overwhelmingly failing and velocity-delta runs fully recovering. |
| 2026-07-09T15:29:50+08:00 | BFM-ZERO-INDUCED-WRENCH1X-DEFAULT | Restored the induced-mode wrench defaults to the original `1x` range (`600-1800 N`, `200-600 Nm`) and reran the 100-run mixed induced sweep as `induced-full-20260709c-wrench1x-default`: `100/100` accepted perturbations, `77/100` recoveries overall, with wrench at `29/50` and velocity-delta at `48/50`. |
| 2026-07-09T15:56:00+08:00 | BFM-ZERO-STAGE3-TRACKING-CAMERA | Switched the BFM-Zero Stage 3 offline renderer back to MuJoCo tracking-camera mode and seeded the initial look-at from the pelvis pose so the first rendered frame starts centered on the robot. |
| 2026-07-09T17:14:57+08:00 | BFM-ZERO-VELOCITY-ONLY-D | Switched the induced defaults to velocity-only with nominal linear `3-6 m/s`, angular `4-8 rad/s`, and an up-cosine gate of `[-0.8, 0.0]`, then completed `induced-full-20260709d-velocity-default` with `100/100` accepted perturbations, `98/100` recoveries, and a verified Stage 3 frame-0 on-robot camera start. |

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
| `scripts/bfm_zero_fallen_recovery_eval.py` | BFM-ZERO-VELOCITY-ONLY-D | Repo-owned BFM-Zero fallen-recovery CLI with both the original saved-state replay path and an opt-in standing-start in-simulator disturbance path, velocity-only induced defaults at nominal linear `3-6 m/s` and angular `4-8 rad/s`, an up-cosine gate for linear kicks, accepted-perturbation retries, disturbance tuning sweeps, Stage 3 auto-discovery of optional Stage 1 settle artifacts, variable-length induced trajectory handling in Stage 3, and a pelvis-seeded tracking camera for Stage 3 renders so frame 0 starts on-robot. |
| `thirdparties/BFM-Zero-deploy/sshkeyboard.py` | BFM-ZERO-FALLEN-RECOVERY | Local PTY-friendly keyboard shim that lets the unmodified BFM-Zero deployer consume scripted `n` and `]` key events during automated stage replays. |
| `thirdparties/BFM-Zero-deploy/sim_env/utils/disturbance.py` | BFM-ZERO-VELOCITY-ONLY-D | Shared BFM-Zero-deploy disturbance helper for sampled root-body wrench or linear-plus-angular velocity perturbations, keyboard/control-file triggering, status export, and the velocity-only default sampler with the constrained linear kick vertical component used by the `20260709d` induced evaluation. |
| `thirdparties/BFM-Zero-deploy/sim_env/utils/simulation_bridge.py` | BFM-ZERO-IN-SIM-DISTURBANCE | Deploy-side MuJoCo bridge now tracking first and total lowcmd reception so elastic-band auto-release and induced-fall readiness can key off real deploy traffic. |
| `thirdparties/BFM-Zero-deploy/sim_env/base_sim.py` | BFM-ZERO-IN-SIM-DISTURBANCE | Viewer-backed BFM-Zero deploy simulator now wiring the shared disturbance helper, `F` keyboard triggering, elastic-band preload, and first-lowcmd auto-release while preserving the default behavior when disturbance mode is off. |
| `BFM_ZERO_FALLEN_RECOVERY_FAILURE_ANALYSIS.md` | BFM-ZERO-STRICT-RUN-A | Tracked best-effort report for the `fullstage1-bfm-zero-20260709a` strict-sampling sweep, including failed recovery and failed passive-settle video indexes. |
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
| `logs/bfm-zero-fallen-recovery/*` | BFM-ZERO-VELOCITY-ONLY-D | Stage 1 manifests, settled state files, strict-sampling replay logs, and induced-mode Stage 2 run directories including the mixed comparison sweeps plus the velocity-only `induced-full-20260709d-velocity-default` run. |
| `artifacts/bfm-zero-fallen-recovery/*` | BFM-ZERO-VELOCITY-ONLY-D | Stage 3 metrics, plots, tiled videos, failed-run 1080p exports, tuning summaries, and the induced-mode comparison artifacts including the velocity-only `induced-full-20260709d-velocity-default` sweep and its verified frame-0 camera centering. |

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
