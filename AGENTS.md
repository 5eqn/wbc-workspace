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

1. **Mean joint RMSE < 0.2** for both SONIC and HoloMotion on all 10 motions, **computed after aligning reference and tracked trajectories by the measured tracking delay**. (Without delay alignment, RMSE is inflated by the time offset and does not reflect tracking quality.) If impossible, prove why (dependency breaks in Docker, malformed motion, etc.).

2. **Dockerfiles stay as close as possible to the original environment** — follow each repo's official docs. Prefer installing dependencies and using pre-existing scripts from upstream repos over writing custom scripts. Robotic research debugging without vision is very hard; staying close to official repos/documents is the only safe path.

3. **Dockerfiles contain no scripts.** All scripts live in `scripts/` and are mounted into containers at runtime via `-v`. Dockerfiles are pure environment builds — no entrypoint logic, no control flow, no embedded scripts. Every line in a Dockerfile installs a dependency, copies a file from `thirdparties/`, or sets an environment variable. You should not inline scripts inside Dockerfile, and `printf` and here-doc is disabled.

4. **Reproducible from scratch.** Anyone who clones the repo recursively and runs `download.sh` must be able to reproduce all results by running `run.sh` then `report.sh`. No manual steps, no host pre-configuration beyond Docker + Python + bash.

5. **Bronze tier or better.** If even Bronze can't be reached while satisfying the RMSE gate, prove impossibility and exit.

6. **Host requires minimal dependencies.** No venv, no ROS2/DDS on the host. Only Docker, Python (standard library + common packages like numpy/matplotlib), and standard bash tools. All heavyweight runtimes (ROS2, DDS, MuJoCo, PyTorch, ONNX Runtime) live exclusively inside Docker containers.

7. **`thirdparties/run-sonic/` is read-only reference material.** You may inspect its docs and scripts to understand proven workflows, timing checks, joint-order handling, metrics, and video rendering, but benchmark execution must not depend on running code from `thirdparties/run-sonic/`, and you must not modify it. Any benchmark-owned implementation scripts must live flat in the root `scripts/` directory; do not add script subdirectories and do not place new executable glue under `thirdparties/`.

8. **Elastic-band/support release order is mandatory.** The robot must remain supported until the policy/controller has reached active `CONTROL`, then the elastic band/support must be released, and only after confirmed release may the benchmark start the acting motion/playback. Releasing before `CONTROL`, or starting motion before confirmed release, invalidates the run even if RMSE is low. Use `thirdparties/run-sonic/` only as read-only reference material for the proven no-fall release workflow and event-order checks.

9. **Use stock deploy paths, including HoloMotion v1.3.** HoloMotion must be the pre-existing `thirdparties/HoloMotion/` v1.3.0 checkout, and both SONIC and HoloMotion must run through their stock deployment/control paths inside Docker. Do not replace either policy with a benchmark-owned/self-implemented DDS policy runner. For HoloMotion, follow the v1.3 deployment documentation and launch/control graph, including the documented procedure for switching from the ready/default control state into motion tracking mode before playback.

10. After `report.sh` completes, `artifacts/` must contain:
   - Proof that phase time = wall time = sim time
   - Mean joint RMSE < 0.2 for both policies, all 10 motions (or documented impossibility)
   - Tracking delay per policy per motion (should fall within -0.2s ~ 0.2s; positive preferred, prior estimate ~0.04s)
   - Side-by-side comparison video (HoloMotion vs SONIC, reference ghost overlay, metrics burned in)

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
