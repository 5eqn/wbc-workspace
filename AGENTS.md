# WBC Benchmark — HoloMotion vs GEAR-SONIC on Unitree G1 29DOF

## Pre-existing assets

`assets/` contains pre-downloaded artifacts (Docker build context, do not re-download):

| Path | Content |
|------|---------|
| `assets/motions.zip` | 10 motion clips in two formats: `sonic_motions/<name>/` (CSV: joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel, metadata.txt, info.txt) and `holomotion_motions/<name>_holomotion.npz` |
| `assets/GEAR-SONIC/` | Pre-downloaded ONNX models: `model_encoder.onnx` (~50MB), `model_decoder.onnx` (~40MB), `observation_config.yaml` |
| `assets/HoloMotion_models/` | Pre-downloaded ONNX models: `HoloMotion_motion_tracking_model/exported/motion_tracking_model.onnx`, `HoloMotion_velocity_tracking_model/exported/velocity_model.onnx` |

Motions are already format-converted. Each policy gets its expected format: SONIC reads CSV from `sonic_motions/`, HoloMotion reads NPZ from `holomotion_motions/`.

## Pre-existing thirdparties

`thirdparties/` contains upstream repos (Docker build context):

| Path | Version | Key paths |
|------|---------|-----------|
| `thirdparties/HoloMotion/` | v1.3.0 | `deployment/unitree_g1_ros2_29dof/` — Docker launch scripts. Conda env: PyTorch 2.3.1, CUDA 12.1, onnxruntime-gpu, MuJoCo. `environments/` — conda env YAMLs + pip requirements. |
| `thirdparties/GR00T-WholeBodyControl/` | latest main | `gear_sonic_deploy/docker/Dockerfile.ros2` — CUDA 12.4.1 + ROS2 Humble. `gear_sonic_deploy/` — deployment framework. `install_scripts/` — uv-based install (install_mujoco_sim.sh, install_ros.sh, etc.). |
| `thirdparties/unitree_mujoco/` | latest | `simulate/` — C++ MuJoCo simulator (recommended). `simulate_python/` — Python simulator. `unitree_robots/` — MJCF robot descriptions (G1 29DOF supported). Uses DDS/unitree_sdk2 for low-level motor I/O (`LowCmd`, `LowState`). |

All three repos already have Docker infrastructure and/or MuJoCo-based simulation. Pre-existing scripts should be reused — prefer them over writing new code.

## Rating tiers

| Tier | Stock code changed (lines in thirdparties/) | Script code (lines in scripts/, excl. download.sh) |
|------|---------------------------------------------|-----------------------------------------------------|
| **Gold** | <100 | ≤ 1000 |
| **Silver** | <400 | ≤ 4000 |
| **Bronze** | <1000 | ≤ 10000 |

`run.sh` and `report.sh` count toward script code lines. Extra helper scripts/deps also count. Only `download.sh` is excluded.

**Bronze is the hard gate.** If Bronze is impossible while meeting the RMSE < 0.2 hard gate, prove why and exit.

Results will be reviewed by Claude Opus 4.6 and a human expert in embodied intelligence. They prefer a clean project structure that runs elegantly, not something hard to maintain and reproduce. Everything runs inside Docker — anyone with Docker can reproduce.

## Hard gates

1. **Tracking RMSE gate.** SONIC must achieve mean joint RMSE < 0.2 on all 10 motions. HoloMotion v1.3 must achieve mean joint RMSE < 0.2 on at least 8 of 10 motions. RMSE is computed after aligning reference and tracked trajectories by the measured tracking delay. (Without delay alignment, RMSE is inflated by the time offset and does not reflect tracking quality.) For any remaining HoloMotion failures, first try stock-compatible stabilization changes, especially longer post-release stabilization time before switching to motion tracking, before claiming failure. If the gate is impossible, prove why (dependency breaks in Docker, malformed motion, stock deploy cannot track a clip, etc.).

2. **Dockerfiles stay as close as possible to the original environment** — follow each repo's official docs. Prefer installing dependencies and using pre-existing scripts from upstream repos over writing custom scripts. Robotic research debugging without vision is very hard; staying close to official repos/documents is the only safe path.

3. **Dockerfiles contain no scripts.** All scripts live in `scripts/` and are mounted into containers at runtime via `-v`. Dockerfiles are pure environment builds — no entrypoint logic, no control flow, no embedded scripts. Every line in a Dockerfile installs a dependency, copies a file from `thirdparties/`, or sets an environment variable. You should not inline scripts inside Dockerfile, and `printf` and here-doc is disabled.

4. **Reproducible from scratch.** Anyone who clones the repo recursively and runs `download.sh` must be able to reproduce all results by running `run.sh` then `report.sh`. No manual steps, no host pre-configuration beyond Docker + Python + bash.

5. **Bronze tier or better.** If even Bronze can't be reached while satisfying the RMSE gate, prove impossibility and exit.

6. **Host requires minimal dependencies.** No venv, no ROS2/DDS on the host. Only Docker, Python (standard library + common packages like numpy/matplotlib), and standard bash tools. All heavyweight runtimes (ROS2, DDS, MuJoCo, PyTorch, ONNX Runtime) live exclusively inside Docker containers.

7. **`thirdparties/run-sonic/` is read-only reference material.** You may inspect its docs and scripts to understand proven workflows, timing checks, joint-order handling, metrics, and video rendering, but benchmark execution must not depend on running code from `thirdparties/run-sonic/`, and you must not modify it. Any benchmark-owned implementation scripts must live flat in the root `scripts/` directory; do not add script subdirectories and do not place new executable glue under `thirdparties/`.

8. **Elastic-band/support release order is mandatory.** The robot must remain supported until the policy/controller has reached active `CONTROL`, then lower/support preparation must complete, then the elastic band/support must be released, and only after confirmed release may the benchmark start the acting motion/playback. Releasing before `CONTROL`, skipping lower/support preparation, or starting motion before confirmed release invalidates the run even if RMSE is low. Use `thirdparties/run-sonic/` only as read-only reference material for the proven no-fall release workflow and event-order checks.

9. **Use HoloMotion v1.3 and stock deploy/control paths.** HoloMotion must be the pre-existing `thirdparties/HoloMotion/` v1.3.0 checkout; other HoloMotion versions are not valid for this benchmark. SONIC must run through the stock `thirdparties/GR00T-WholeBodyControl/gear_sonic_deploy/` deploy path, and HoloMotion v1.3 must run through `thirdparties/HoloMotion/deployment/unitree_g1_ros2_29dof/` and its documented launch/control graph. Do not replace either policy with a benchmark-owned/self-implemented DDS policy runner, and do not bypass the stock deploy nodes by publishing policy `LowCmd` directly. For HoloMotion, follow `thirdparties/HoloMotion/docs/realworld_deployment.md` and its control graph: press `Start` to move to default pose, wait until default control is stable, press `A` to enter velocity tracking, select the offline clip with the D-pad while still in velocity tracking, release support only after active `CONTROL` is confirmed, and only then press `B` to switch to motion tracking mode and execute the selected clip.

10. After `report.sh` completes, `artifacts/` must contain:
   - Proof that phase time = wall time = sim time
   - SONIC mean joint RMSE < 0.2 for all 10 motions, and HoloMotion mean joint RMSE < 0.2 for at least 8 of 10 motions (or documented impossibility)
   - Tracking delay per policy per motion (should fall within -0.2s ~ 0.2s; positive preferred, prior estimate ~0.04s)
   - One valid MuJoCo-rerendered side-by-side comparison MP4 per motion
   - Raw MuJoCo playback logs sufficient to reproduce every video frame without inventing root pose, support state, joint order, or timing
   - Release validation proof covering the three required release tests

11. **Raw MuJoCo playback logging is a hard gate.** The simulator must log enough raw MuJoCo state during every run to replay the robot exactly as simulated: wall time, sim time, benchmark phase time, complete root pose (`qpos[0:7]`), robot joint positions in hardware/MJCF order, root/body velocities when available, support/elastic-band state, control-active state, command/control state, and event markers. `report.sh` must prove that simulation did not lag: phase time, wall time, and MuJoCo sim time must agree within a documented tolerance for each run. Do not reconstruct missing physical state in report code. If a video or metric needs root pose, support state, timing, or joint order, that information must come from raw logs or from the loaded MJCF by name/order, not from arbitrary constants or inferred placeholders.

12. **Release behavior proof is a hard gate.** Before counting motion-tracking results, the benchmark must run and report three release validation tests from a clean simulator/policy start: (a) no-control direct release must fall within 2 seconds after release; (b) active CONTROL plus lower/support preparation plus release must remain standing for more than 5 seconds after confirmed release; (c) after that stable released CONTROL case, removing/stopping control must make the robot fall within 2 seconds. These tests must be backed by raw MuJoCo logs and clear pass/fail artifacts. Event-order logs alone are not sufficient proof.

13. **Comparison videos are a hard gate and must be MuJoCo rerenders from raw playback logs, not plots or reconstructed poses.** `report.sh` must fail if any required comparison video is missing, unreadable, generated from plots/traces instead of MuJoCo robot rendering, uses reconstructed root/support state, uses the wrong joint order, or uses the wrong time interval. For each of the 10 motions, `artifacts/` must contain one side-by-side HoloMotion-vs-SONIC MP4 rendered by loading the robot MJCF in MuJoCo, replaying the raw logged MuJoCo root pose and measured robot joints for each policy, and drawing a reference ghost overlay from the corresponding reference motion. Metrics and event labels must be burned into the video. The rendered timeline starts at the pose where the robot is still hanging/supported in air, includes CONTROL entry, lower/support preparation, confirmed RELEASE, and only then acting-motion playback. The acting motion must start at the same video phase time for both policies; if one side reaches CONTROL/release later, that side may appear later or wait, but playback starts must align. The video ends at the acting motion's nominal end, so total video duration is the pre-playback interval plus the nominal clip duration. A joint-trace plot, matplotlib animation, slideshow, policy-only video, reference-only video, video without ghost overlay, video without metrics, video rendered from inferred root/base height, or video that starts at acting-motion playback does not satisfy this gate.

## Folder structure

```
wbc-benchmark/
├── docker/                   # Dockerfiles — pure env build, NO scripts, NO entrypoint logic
│   ├── holomotion.Dockerfile
│   ├── gear-sonic.Dockerfile
│   └── unitree_mujoco.Dockerfile
├── scripts/
│   ├── run.sh                # Entrypoint 1: build, deploy, run a policy+motion pair
│   ├── report.sh             # Entrypoint 2: generate all reports from collected logs
│   └── download.sh           # Asset downloader (excluded from script count)
├── assets/                   # Pre-formatted motion clips + model weights (build context)
├── thirdparties/             # Upstream repos (build context)
├── logs/                     # Raw runtime output per run
└── artifacts/                # Final reports & video covering all 10 motions
```

`scripts/` has no subdirectories. Extra dependency files live flat alongside run.sh and report.sh, and count toward the tier limit.

## Runtime hints

These are operational hints, not substitutes for the hard gates. Add new hints here whenever a run reveals useful, reproducible knowledge that helps future agents avoid repeating mistakes.

1. Treat one policy + one motion as the atomic validation unit. Do not assess the full batch until one motion is correct.
2. Restart both simulator and policy for every motion. Avoid persistent host-network simulator or policy processes that can contaminate DDS/ROS traffic between runs.
3. For a single-motion run, launch the simulator first and verify it is ready with minimal deviation from stock `unitree_mujoco` behavior, because the simulator is standing in for the real robot low-level interface.
4. Launch the stock policy second through its official deploy path, then wait until the policy model/control graph is fully loaded and ready. Start benchmark timing only after both simulator and policy are ready.
5. Use a fixed nominal timing sequence after readiness: motion selection, control entry, lower/support preparation, release, then motion start/playback.
6. Gold path: drive policy mode selection through simulated Xbox joystick/wireless-remote input, and drive lower/release through the simulator's intended GLFW/key input path. Silver path: use the least intrusive benchmark-owned control mechanism that still preserves stock deploy behavior.
7. For lower/release details, inspect `thirdparties/run-sonic/release_to_ground.sh` as read-only reference material and translate only the minimum necessary workflow into flat root `scripts/`.
8. Validate the control/lower/release workflow independently before motion tracking: direct release before control should fall; entering control, lowering, and releasing should stand for more than 5 seconds; stopping the control policy after stable release should fall within about 2 seconds.
9. Stop a test early when the robot has clearly fallen. A fallen run should be marked failed immediately rather than allowed to continue and contaminate timing or metrics.
10. HoloMotion v1.3 stock deploy still failed `dance_chicken_c03_neutral2s` after initializing the simulator from that HoloMotion NPZ frame 0; the run reached valid release order, then fell during motion tracking with RMSE around `0.421`. Root initialization alone is not the missing contract.
11. HoloMotion v1.3 stock deploy with simulator `--publish-every 4` missed the current one-shot D-pad clip-selection pulse and timed out before release. Keep HoloMotion at the current higher low-state publish cadence unless the selection transport is changed deliberately.
12. Do not use the ad-hoc HoloMotion `eval_mujoco_sim2sim.py` diagnostic as standalone impossibility proof yet. With the local headless CPU invocation and `assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml`, it fails `dance_chicken_c04_neutral2s` too (joint RMSE about `0.496`, base height below `0.25m` around frame 58), even though the stock deploy path tracks C04 under the RMSE gate in `unitree_mujoco`. Treat these diagnostics as a clue about invocation/model-context mismatch, not as authoritative evidence that a clip is impossible.
13. Fresh current-code full batch `logs_full_v3`/`artifacts_full_v3` clears all SONIC motions and all release-order gates. The remaining RMSE failures are HoloMotion `dance_chicken_c03_neutral2s` (about `0.412`, early-stopped after fall) and HoloMotion `dance_heart111_c01_neutral2s` (about `0.296`, early-stopped after fall).

## Hints (not requirements)

1. unitree_mujoco runs as shared simulator; HoloMotion and SONIC connect to it as control policies.
2. Event transport via ROS2 or DDS — use whatever the policy already listens to. These repos should not require source changes to run (preliminary tests indicate they work stock).
3. Timing events: SELECT_MOTION, CONTROL at 1s, RELEASE at 2s, START_POLICY at 3s, motion ends at nominal clip duration. This is an example, you just need to form such explicit timing, and the timing starts when everything is ready (simulation & policy both running).
4. Be lazy — search the web, check GitHub issues, find existing repos and Dockerfiles that can be reused. Only write custom code when necessary and you know exactly what you're doing. Error-prone to implement from scratch. Specifically for robot elastic band & releasing methods, follow strictly to recommended workflow in HoloMotion or SONIC. This work is basically reproducing Internet works and it's always good to follow pre-existing documentations.
5. Tiering is a general template. Some repos may need mandatory changes to run in Docker. But these two (HoloMotion, SONIC) should work without stock changes if preliminary tests are correct.
6. Each agent/session should focus on exactly one atomic, clear task. Each atomic task corresponds to exactly one git commit. No multi-tasking within a single commit — keep changes small, reviewable, and semantically coherent. For each atomic task, make a git commit.
7. Ideal structure: stock GR00T & HoloMotion deploy runs in the "lo" network interface with zero code changes. unitree_mujoco runs in its own Docker container with a lightweight data collector (sample at ≤50 Hz; do not over-collect — too much data prevents simulation time from keeping pace with wall time). `run.sh` is a thin orchestrator that sends key events or virtual joystick events to deploy/simulator containers. `report.sh` shells into the simulator container to generate video and metrics.
8. These guidelines describe what great looks like, but they are not assessment rules. Do not use hacky workarounds to technically satisfy a line count or tier while violating the spirit of the benchmark. Claude Opus 4.6 and a human expert in embodied intelligence will review the final result — adhere to sound engineering principles throughout. Do no deliberately compress code.
9. Use Docker build cache aggressively, for example if missing dependencies but previous heavy step passes, add a separate RUN later instead of changing previous RUN, merge & sanitize only when the full Dockerfile builds. Also always prefer to use China mirrors in http.
10. Motion clips are generated with `thirdparties/run-sonic`, specifically some verification scripts inside. but that repo is too large and contains lots of self-implemented stuff, and it's using simulation path inside GR00T repo instead of isolated unitree_mujoco, and it's for comparing HoloMotion v1.2 and SONIC. Now that HoloMotion is updated to v1.3, I think it's better to more strictly adhere to HoloMotion deploy guideline instead of making our own DDS deployer. Although `run-sonic` is actually runnable and proves these motions can be tracked under RMSE 0.2 rads, your task this time is essentially making it cleaner and runnable.
11. You have `docker` group. Avoid changing the host environment if possible, at most install some Python dependencies for host-running scripts.
