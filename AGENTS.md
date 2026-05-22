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

8. **Release behavior must be physically correct.** The robot starts in a simulator-supported floating pose: while `support_active=true`, the simulator holds the root pose fixed; release means switching `support_active=false`, after which the root is no longer pinned and MuJoCo dynamics take over. The robot must remain supported until the policy/controller has reached active `CONTROL`; only after confirmed release may the benchmark start the acting motion/playback. Prove this with three rigid physics tests from clean simulator/policy starts: (a) no-control direct release must fall within 2 seconds after release; (b) active `CONTROL` plus release must remain standing for more than 5 seconds after confirmed release; (c) after that stable released `CONTROL` case, removing/stopping control must make the robot fall within 2 seconds. These tests must be backed by raw simulator logs and event evidence. Event-order logs alone are not sufficient proof.

9. **Use HoloMotion v1.3 and stock policy deploy/control paths.** HoloMotion must be the pre-existing `thirdparties/HoloMotion/` v1.3.0 checkout; other HoloMotion versions are not valid for this benchmark. SONIC must run through the stock `thirdparties/GR00T-WholeBodyControl/gear_sonic_deploy/` deploy path, and HoloMotion v1.3 must run through `thirdparties/HoloMotion/deployment/unitree_g1_ros2_29dof/` and its documented launch/control graph. Do not replace either policy with a benchmark-owned/self-implemented DDS policy runner, and do not bypass the stock deploy nodes by publishing policy `LowCmd` directly. The simulator may be benchmark-owned/self-implemented if it remains physically faithful enough to satisfy the release tests, SONIC reaches the RMSE gate, and the robot does not fall after correct release. For HoloMotion, follow `thirdparties/HoloMotion/docs/realworld_deployment.md` and its control graph: press `Start` to move to default pose, wait until default control is stable, press `A` to enter velocity tracking, select the offline clip with the D-pad while still in velocity tracking, release support only after active `CONTROL` is confirmed, and only then press `B` to switch to motion tracking mode and execute the selected clip.

10. After `report.sh` completes, `artifacts/` must contain:
   - Proof that phase time = wall time = sim time
   - SONIC mean joint RMSE < 0.2 for all 10 motions, and HoloMotion mean joint RMSE < 0.2 for at least 8 of 10 motions (or documented impossibility)
   - Tracking delay per policy per motion (should fall within -0.2s ~ 0.2s; positive preferred, prior estimate ~0.04s)
   - One valid MuJoCo-rerendered side-by-side comparison MP4 per motion
   - Raw simulator playback logs sufficient to reproduce every video frame without inventing root pose, support state, joint order, or timing
   - Release validation proof covering the three required release tests

11. **Comparison videos must faithfully reflect simulation.** `report.sh` must fail if any required comparison video is missing, unreadable, generated from plots/traces instead of MuJoCo robot rendering, uses reconstructed root/support state, uses the wrong joint order, changes playback speed, or uses the wrong time interval. For each of the 10 motions, `artifacts/` must contain one side-by-side HoloMotion-vs-SONIC MP4 rendered by loading the robot MJCF in MuJoCo and replaying the raw logged root `qpos` and measured robot joints for each policy. Metrics and event labels must be burned into the video. The rendered timeline starts at the pose where the robot is still floating with simulator support active, makes the support-release moment visible, starts the acting motion at the same video phase time for both policies, and ends when the acting motion ends. Align the acting-motion start and end times across policies without changing speed; if one side reaches release later, that side may appear later or wait, but playback starts and ends must align. A joint-trace plot, matplotlib animation, slideshow, policy-only video, reference-only video, video without metrics, video rendered from inferred root/base height, video with incorrect joint order, or video that starts at acting-motion playback does not satisfy this gate.

12. **Single 29DOF robot interface, immediately swappable to the real robot.** HoloMotion and SONIC must both communicate only through the same Unitree G1 29DOF low-level robot contract used by a real robot: DDS/ROS2 low-level state/command topics, the same 29 body-joint order, the same control/release timing semantics, and no policy-specific simulator plant. The simulator is a stand-in for one physical G1 29DOF robot, not a different robot per policy. Benchmark code must be able to replace the simulator with a real 29DOF Unitree G1 endpoint without changing policy deploy code, joint order, policy mode sequence, or benchmark orchestration beyond selecting the real robot transport endpoint and disabling simulator-only support/release controls. Any 43DOF or hand-equipped model may be used only as read-only reference or visualization metadata; it must not define the active benchmark plant, low-level command surface, RMSE joint set, release proof, or real-robot compatibility claim.

## Current status

As of 2026-05-22, hard gates 8, 11, and 12 have a passing implementation path in the current code.

- `scripts/sim_bridge.py` is now a low-entropy, policy-agnostic Unitree G1 29DOF DDS/MuJoCo endpoint. It no longer reads motion files, applies SONIC/HoloMotion reference remaps, or uses clip-specific initial poses. It starts from one neutral 29DOF pose, uses root-only simulator support, and uses one simple pre-control hold before real `LowCmd` takes over.
- The only retained benchmark evidence directories are now `logs/` and `artifacts/`; development `logs_*` and `artifacts_*` directories were removed after promoting the passing run into the plain paths.
- Fresh evidence in `logs/` and `artifacts/` passed report checks: `all_rmse_passed=true`, `all_release_gates_passed=true`, `single_robot_interface_passed=true`, and `failures=[]`.
- SONIC passed all 10 motions under the RMSE gate in `artifacts/rmse_summary.csv`. HoloMotion passed 8/10; the remaining failures are the known out-of-distribution balance failures `dance_chicken_c03_neutral2s` and `dance_heart111_c01_neutral2s`.
- Hard gate 8 release proof passed in `artifacts/release_validation.json`: direct release fell after `0.32s`, released SONIC `CONTROL` remained standing beyond 5 seconds with stable base height, and stopping control caused fall after `0.32s`.
- Hard gate 11 video proof passed with 10 MuJoCo-rendered comparison MP4s in `artifacts/`.
- The observed SONIC `dance_phony_c01_neutral2s` paralysed/jitter failure was not reproduced in clean repeats. Simplified simulator repeats passed at about `0.137`, `0.120`, and `0.123` RMSE; the old simulator path also passed three repeats at about `0.130`, `0.120`, and `0.121`. Treat it as an occasional timing/contact sensitivity unless a reproducible trigger is found.
- One full-batch SONIC `dance_chicken_c04_neutral2s` run fell during motion and failed RMSE, but a clean rerun passed at `0.166` RMSE and the final report uses the clean raw logs. This suggests occasional instability rather than a deterministic policy or simulator mapping error.
- Agents must update this Current status section whenever they change the simulator/control path, replace retained evidence, discover a reproducible failure mode, or complete a gate-relevant verification. Do this before ending the task or committing the status-changing work.

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
3. For a single-motion run, launch the simulator first and verify it is ready and physically plausible, because the simulator is standing in for the real robot low-level interface. A benchmark-owned simulator is acceptable when it preserves the required low-level interface, release physics, timing, joint order, and RMSE behavior.
4. Launch the stock policy second through its official deploy path, then wait until the policy model/control graph is fully loaded and ready. Start benchmark timing only after both simulator and policy are ready.
5. Use a fixed nominal timing sequence after readiness: motion selection, control entry, release, then motion start/playback.
6. Gold path: drive policy mode selection through simulated Xbox joystick/wireless-remote input, and drive support release through the simulator's intended release control path. Silver path: use the least intrusive benchmark-owned control mechanism that still preserves stock policy deploy behavior.
7. For release details, inspect `thirdparties/run-sonic/release_to_ground.sh` as read-only reference material and translate only the minimum necessary workflow into flat root `scripts/`.
8. Validate the control/release workflow independently before motion tracking: direct release before control should fall; entering control and releasing should stand for more than 5 seconds; stopping the control policy after stable release should fall within about 2 seconds.
9. Stop a test early when the robot has clearly fallen. A fallen run should be marked failed immediately rather than allowed to continue and contaminate timing or metrics.
10. HoloMotion v1.3 stock deploy still failed `dance_chicken_c03_neutral2s` after initializing the simulator from that HoloMotion NPZ frame 0; the run reached valid release order, then fell during motion tracking with RMSE around `0.421`. Root initialization alone is not the missing contract.
11. HoloMotion v1.3 stock deploy with simulator `--publish-every 4` missed the current one-shot D-pad clip-selection pulse and timed out before release. Keep HoloMotion at the current higher low-state publish cadence unless the selection transport is changed deliberately.
12. Do not use the ad-hoc HoloMotion `eval_mujoco_sim2sim.py` diagnostic as standalone impossibility proof yet. With the local headless CPU invocation and `assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml`, it fails `dance_chicken_c04_neutral2s` too (joint RMSE about `0.496`, base height below `0.25m` around frame 58), even though the stock deploy path tracks C04 under the RMSE gate in `unitree_mujoco`. Treat these diagnostics as a clue about invocation/model-context mismatch, not as authoritative evidence that a clip is impossible.
13. The retained current-code full batch is `logs/` plus `artifacts/`. It clears all SONIC motions and all release-order gates; HoloMotion clears the required 8/10 RMSE motions, with the remaining out-of-distribution balance failures documented in the Current status section.
14. Keep the simulator low-entropy and policy-agnostic. It should behave like a simple headless `unitree_mujoco` low-level robot endpoint: MuJoCo scene + DDS `LowCmd`/`LowState` + raw replay logs + simulator-only support/release. Do not put SONIC/HoloMotion reference remaps, clip-specific initial poses, or policy-order constants inside the simulator. Initial support pose should be universal, and pre-control support/hold should be implemented with one simple mechanism (root pinning and, if needed before any `LowCmd`, a single obvious large PD hold), not many hidden motor-derived constants. If SONIC ever produces paralysed/jittering commands on a clip that should track, first suspect simulator/interface mismatch and reproduce across repeated clean starts before blaming the policy.

## Hints (not requirements)

1. The simulator may be stock `unitree_mujoco` or benchmark-owned, but it is the shared simulated robot that HoloMotion and SONIC connect to as stock control policies. It must preserve the low-level robot interface, timing, joint order, support/release behavior, and raw replay evidence needed by the hard gates.
2. Event transport via ROS2 or DDS — use whatever the policy already listens to. These repos should not require source changes to run (preliminary tests indicate they work stock).
3. Timing events: SELECT_MOTION, CONTROL at 1s, RELEASE at 2s, START_POLICY at 3s, motion ends at nominal clip duration. This is an example, you just need to form such explicit timing, and the timing starts when everything is ready (simulation & policy both running).
4. Be lazy — search the web, check GitHub issues, find existing repos and Dockerfiles that can be reused. Only write custom code when necessary and you know exactly what you're doing. Error-prone to implement from scratch. Specifically for robot support/release methods, follow strictly to recommended workflow in HoloMotion or SONIC. This work is basically reproducing Internet works and it's always good to follow pre-existing documentations.
5. Tiering is a general template. Some repos may need mandatory changes to run in Docker. But these two (HoloMotion, SONIC) should work without stock changes if preliminary tests are correct.
6. Each agent/session should focus on exactly one atomic, clear task. Each atomic task corresponds to exactly one git commit. No multi-tasking within a single commit — keep changes small, reviewable, and semantically coherent. For each atomic task, make a git commit.
7. Ideal structure: stock GR00T & HoloMotion deploy runs in the "lo" network interface with zero code changes. The simulator runs in its own Docker container with a lightweight data collector (sample at ≤50 Hz; do not over-collect — too much data prevents simulation time from keeping pace with wall time). `run.sh` is a thin orchestrator that sends key events or virtual joystick events to deploy/simulator containers. `report.sh` generates video and metrics from raw simulator logs.
8. These guidelines describe what great looks like, but they are not assessment rules. Do not use hacky workarounds to technically satisfy a line count or tier while violating the spirit of the benchmark. Claude Opus 4.6 and a human expert in embodied intelligence will review the final result — adhere to sound engineering principles throughout. Do no deliberately compress code.
9. Use Docker build cache aggressively, for example if missing dependencies but previous heavy step passes, add a separate RUN later instead of changing previous RUN, merge & sanitize only when the full Dockerfile builds. Also always prefer to use China mirrors in http.
10. Motion clips are generated with `thirdparties/run-sonic`, specifically some verification scripts inside. but that repo is too large and contains lots of self-implemented stuff, and it's using simulation path inside GR00T repo instead of isolated unitree_mujoco, and it's for comparing HoloMotion v1.2 and SONIC. Now that HoloMotion is updated to v1.3, I think it's better to more strictly adhere to HoloMotion deploy guideline instead of making our own DDS deployer. Although `run-sonic` is actually runnable and proves these motions can be tracked under RMSE 0.2 rads, your task this time is essentially making it cleaner and runnable.
11. You have `docker` group. Avoid changing the host environment if possible, at most install some Python dependencies for host-running scripts.
