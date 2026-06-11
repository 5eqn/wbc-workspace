# Unitree SDK Bridge And Isaac Sim Backend Goal

## Outcome

Migrate the existing Unitree simulator bridge path to depend only on
`unitree_sdk2py`, then add a pinned Isaac Sim backend with equivalent shared CLI
entrypoints, simulator bridge behavior, logs, reports, and artifacts.

SONIC and HoloMotion must each pass Isaac Sim validation independently on at
least 7 of 10 canonical motions. Existing MuJoCo/backend behavior must remain
intact and pre-existing sim2sim artifacts should be preserved under a
MuJoCo-specific artifact namespace.

The result should be reviewable by Claude Opus 4.6 and human embodied
intelligence experts.

## Assumptions

- Use the correct spelling: `isaac`, not `issac`.
- Pin a newer Isaac Sim version that works in this repo and runtime after
  compatibility testing.
- Fall detection, the 10-motion set, and RMSE definition are the same as the
  previous work in this workspace.
- RMSE threshold remains `< 0.2`.
- Pass criterion is independent per policy:
  - SONIC: at least 7 of 10 motions must not fall and must have RMSE `< 0.2`.
  - HoloMotion: at least 7 of 10 motions must not fall and must have RMSE
    `< 0.2`.
- Isaac Sim terms are pre-accepted by the user/environment. The implementation
  must not bypass licensing or interactive acceptance.
- Explicit third-party dependencies may be committed under `thirdparties/` or
  `assets/` as local build-cache inputs.
- Existing CLI entrypoints should be shared, not duplicated.

## Success Criteria

1. `sim_bridge.py` depends on `unitree_sdk2py` only.
2. No existing Dockerfile is changed for the Unitree SDK migration.
3. A quick compatibility test passes for SONIC and HoloMotion on 1 canonical
   motion.
4. Add `docker/unitree_isaacsim.Dockerfile`.
5. The Isaac Dockerfile uses China mirrors where practical.
6. Explicit Isaac/Unitree third-party dependencies are downloaded locally once
   and reused as build cache/build context.
7. Add an Isaac-specific sim bridge while keeping policy, motion, and reporting
   flow shared.
8. Run SONIC and HoloMotion across all 10 canonical motions in Isaac Sim.
9. Generate symmetric MuJoCo and Isaac artifact trees.
10. Isaac validation passes independently:
    - SONIC: `>= 7/10`
    - HoloMotion: `>= 7/10`

## Verification Surface

```text
wbc-workspace/
├── docker/
│   ├── <existing Dockerfiles unchanged>
│   └── unitree_isaacsim.Dockerfile
├── scripts/
│   ├── run.sh
│   ├── report.sh
│   └── download.sh
├── thirdparties/
│   └── <pinned downloaded dependencies/cache>
├── logs/
│   ├── mujoco-backend-sim2sim/
│   │   └── <pre-existing logs moved here>
│   └── isaac-backend-sim2sim/
│       ├── sonic/
│       └── holomotion/
└── artifacts/
    ├── mujoco-backend-sim2sim/
    │   └── <pre-existing artifacts moved here>
    └── isaac-backend-sim2sim/
        ├── reports/
        ├── metrics.csv
        ├── summary.json
        └── videos/
```

## Execution Plan

1. Inspect current bridge, CLI, report, logs, and previous artifact conventions.
   - Verify: identify canonical 10 motions, fall metric, RMSE metric, and
     current sim2sim artifact structure.
2. Modify `sim_bridge.py` to remove non-`unitree_sdk2py` dependency paths.
   - Verify: import/runtime smoke test inside the existing container without
     editing existing Dockerfiles.
3. Run quick MuJoCo/backend compatibility test.
   - Verify: SONIC and HoloMotion each run 1 motion and produce logs/report.
4. Move existing sim2sim artifacts/logs into `mujoco-backend-sim2sim/`.
   - Verify: old artifacts are preserved and report paths still work.
5. Add Isaac dependency download/cache flow.
   - Verify: `scripts/download.sh` fetches pinned third-party inputs once into
     committed/cacheable locations.
6. Add `docker/unitree_isaacsim.Dockerfile`.
   - Verify: Docker build succeeds using pinned versions and mirrors where
     practical.
7. Add Isaac-specific simulator bridge behind a shared CLI selector.
   - Verify: the same `scripts/run.sh` and `scripts/report.sh` can target the
     MuJoCo and Isaac backends.
8. Run full Isaac validation for SONIC and HoloMotion over 10 motions.
   - Verify: raw logs, videos, metrics, and summary reports are generated under
     `isaac-backend-sim2sim/`.
9. Evaluate pass/fail.
   - Verify: each policy has at least 7 motions with no fall and RMSE `< 0.2`.

## Constraints

- Do not change any existing Dockerfile for the Unitree SDK migration.
- Keep changes surgical: bridge dependency logic, new Isaac Dockerfile, new
  Isaac bridge, and minimal script/report wiring only.
- Maintain existing policy, motion, and report semantics unless a simulator
  boundary requires adaptation.
- Pin third-party versions or commits where possible.
- Avoid changing the host environment except lightweight tooling if unavoidable.
- Major runtime and validation should happen inside Docker.

## Blocked Stop Conditions

Stop and report if Isaac Sim cannot run due to GPU/runtime constraints, required
licensed assets are unavailable despite terms being pre-accepted, previous
metric definitions cannot be recovered, or the canonical 10-motion set cannot be
identified defensibly from the repo/history.
