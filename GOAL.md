# Humanoid-GPT Sim2Sim Evaluation Goal

## Outcome

Run Humanoid-GPT end to end in this workspace using the existing simulator as an
unchanged Dockerized stand-in for the real robot.

The work must prove three things in order:

1. SONIC and HoloMotion reference motions can each be translated into
   Humanoid-GPT reference format correctly.
2. Official Humanoid-GPT built-in evaluation runs on those translated
   references and produces native tracking metrics and video.
3. Humanoid-GPT then runs a separated Sim2Sim path in the same benchmark and
   artifact pipeline family as HoloMotion and SONIC, while the simulator
   remains a drop-in replacement for the real robot and no simulator code is
   modified.

If SONIC-derived and HoloMotion-derived translations of the same canonical
motion are not equivalent after defensible canonicalization, the goal is blocked
immediately. No further Sim2Sim claim is valid after a failed translation gate.

The completed result should be reviewable by Claude Opus 4.6 and human embodied
intelligence experts, and they should be satisfied that any failure is not due
to hidden simulator edits, incorrect joint order, or misunderstood motion
format.

## Assumptions

- Humanoid-GPT official install, inference, and deploy instructions are the
  source of truth for Humanoid-GPT-side runtime behavior.
- The simulator side already works and is not the debugging target of this
  task.
- Humanoid-GPT evaluation and deploy run on the host under a healthy CUDA
  environment.
- The simulator runs inside Docker and must stay byte-for-byte behaviorally
  unchanged.
- Canonical motions exist in both SONIC and HoloMotion forms and refer to the
  same intended motion content.
- "Same result" means equivalent translated motion after deterministic format
  normalization, not merely similar qualitative behavior.
- Quaternion sign ambiguity, frame-rate alignment, neutral-prefix handling, and
  source serialization differences must be normalized before comparing motion
  equivalence.
- A valid translation proof must compare both translated numeric motion fields
  and forward-kinematics results in Humanoid-GPT space.

## Success Criteria

1. A host Humanoid-GPT environment is prepared and verified without damaging the
   existing CUDA environment on host.
2. A deterministic translation path exists from SONIC reference motions into
   Humanoid-GPT reference format.
3. A deterministic translation path exists from HoloMotion reference motions
   into Humanoid-GPT reference format.
4. For every canonical motion selected for evaluation, the SONIC-derived and
   HoloMotion-derived Humanoid-GPT references pass the translation equivalence
   gate.
5. A translation-proof artifact exists with per-motion pass or fail, source
   metadata, joint mapping, frequency alignment, raw diff statistics,
   forward-kinematics or body-pose diff statistics, and spot-check videos.
6. Official Humanoid-GPT built-in evaluation runs on the translated references
   and emits native tracking metrics and rendered video.
7. Humanoid-GPT deploy code runs a separated MuJoCo Sim2Sim path through the
   existing Docker simulator without modifying simulator code.
8. The separated Sim2Sim path writes logs and reports in the same pipeline
   family as HoloMotion and SONIC.
9. The simulator remains a drop-in stand-in for the real robot boundary. Any
   failure is resolved on the Humanoid-GPT environment, deploy, or
   motion-translation side, not by editing simulator code.

## Translation Gate

The translation gate is the first hard verification barrier. It must pass
before any built-in Humanoid-GPT evaluation or separated Sim2Sim run counts as
valid.

For each canonical motion available from both SONIC and HoloMotion:

1. Translate the SONIC source to Humanoid-GPT format.
2. Translate the HoloMotion source to Humanoid-GPT format.
3. Normalize both outputs deterministically:
   - same target frame rate
   - same frame count or deterministic crop rule
   - same root-pose convention
   - same quaternion sign canonicalization rule
   - same neutral-prefix rule
   - same joint-name semantic mapping
4. Compare both translations in Humanoid-GPT space.

The motion passes the gate only if all of the following are true:

1. Joint semantics match exactly by documented mapping.
2. Frame alignment is defensible and deterministic.
3. Per-frame translated state is numerically equivalent up to serialization
   tolerance only.
4. Forward-kinematics or body-pose results in Humanoid-GPT model space are
   equivalent.
5. Spot-check visualization does not reveal semantic divergence masked by raw
   representation quirks.

If a motion fails the gate and no defensible correction is found, the goal is
blocked. Do not continue to built-in Humanoid-GPT evaluation or shared-pipeline
Sim2Sim.

## Verification Surface

```text
wbc-workspace/
├── GOAL.md
├── logs/
│   ├── humanoid-gpt-translation/
│   │   ├── converted_from_sonic/
│   │   ├── converted_from_holomotion/
│   │   └── equivalence_checks/
│   ├── humanoid-gpt-builtin-eval/
│   │   ├── native_metrics/
│   │   └── videos/
│   └── mujoco-backend-sim2sim/
│       └── humanoid-gpt/
│           └── <motion>/
│               ├── sequence_events.csv
│               ├── simulator_status.csv
│               ├── replay.csv
│               ├── sim_control.json
│               ├── simulator_stdout.log
│               └── humanoid_gpt_stdout.log
└── artifacts/
    ├── humanoid-gpt-translation/
    │   ├── translation_equivalence.json
    │   ├── per_motion_diffs.csv
    │   ├── joint_mapping_report.md
    │   └── spotcheck_videos.json
    ├── humanoid-gpt-builtin-eval/
    │   ├── metrics_summary.json
    │   ├── per_motion_metrics.csv
    │   └── videos.json
    └── mujoco-backend-sim2sim/
        ├── metrics_summary.json
        ├── phase_time_proof.json
        ├── rmse_summary.csv
        ├── per_joint_rmse.csv
        ├── release_validation.json
        ├── single_robot_interface.json
        ├── comparison_videos.json
        └── report.md
```

## Iteration Policy

### Iteration 1: Translation Proof And Built-In Humanoid-GPT Evaluation

1. Recover the exact source motion schemas for SONIC and HoloMotion.
   Verify: a documented mapping exists from each source format into
   Humanoid-GPT-required fields.
2. Convert a shared canonical motion from SONIC and HoloMotion independently.
   Verify: both conversions load in Humanoid-GPT format without schema hacks.
3. Run the translation gate on the shared canonical motion.
   Verify: the two translated results match after deterministic
   canonicalization.
4. Scale the translation gate to the full evaluation motion set.
   Verify: `translation_equivalence.json` records pass for every motion selected
   for evaluation.
5. Run official Humanoid-GPT built-in evaluation on the translated motions.
   Verify: native tracking metrics and rendered videos are generated under the
   Humanoid-GPT built-in evaluation artifact tree.

Retry policy for iteration 1:

- If host import, CUDA, JAX, MuJoCo, ONNX Runtime, or viewer runtime fails, fix
  only the Humanoid-GPT host environment and rerun the smallest smoke check.
- If translation mismatches, inspect source schema, joint naming, joint order,
  root convention, frequency conversion, quaternion handling, and neutral-prefix
  policy before trying again.
- If the translation gate still fails after defensible investigation, block the
  goal.

### Iteration 2: Separated Sim2Sim In Shared Pipeline

1. Reuse the existing Docker simulator as-is and connect Humanoid-GPT deploy
   side to it.
   Verify: simulator code and simulator runtime behavior remain unchanged.
2. Run Humanoid-GPT over the canonical motion set through the separated Sim2Sim
   path.
   Verify: raw logs, event ordering, replay, and policy stdout exist under the
   backend log tree.
3. Generate report artifacts in the same family as SONIC and HoloMotion.
   Verify: metrics summary, RMSE tables, phase-time proof, release validation,
   single-robot-interface proof, comparison videos, and report output are
   produced.
4. Confirm the drop-in real-robot contract.
   Verify: no simulator-side workaround was introduced; Humanoid-GPT deploy
   remains aligned with the same real-robot boundary that the simulator stands
   in for.

Retry policy for iteration 2:

- If separated Sim2Sim fails while the simulator remains healthy, treat the
  failure as a Humanoid-GPT deploy, environment, or motion-translation problem.
- Do not repair Humanoid-GPT integration by modifying simulator code.
- If a valid drop-in deploy path cannot be established without simulator edits,
  block the goal.

## Constraints

- Maintain a healthy CUDA environment on host.
- Preserve already existing code except for changes directly required to support
  Humanoid-GPT translation, Humanoid-GPT-side deploy, benchmark wiring, logs,
  and reports.
- Run the simulator part inside Docker.
- Do not change a single line of simulator code.
- Do not change simulator behavior to compensate for Humanoid-GPT deploy issues.
- Treat the simulator as the drop-in replacement part for the real robot.
- Humanoid-GPT deploy must work according to official instructions, not by
  inventing a simulator-only path that would not transfer to the real robot.
- Keep changes surgical and traceable to this goal.

## Boundaries

- Allowed work:
  - host Humanoid-GPT environment setup
  - motion translation code and proof artifacts
  - Humanoid-GPT-side deploy integration
  - benchmark and report wiring needed to add Humanoid-GPT to the existing
    pipeline family
  - logs and artifacts generation
- Protected work:
  - simulator implementation
  - simulator Docker runtime behavior
  - simulator-side code paths used as the real-robot stand-in

## Blocked Stop Conditions

Stop and report the goal as blocked if any of the following becomes true:

1. The host Humanoid-GPT environment cannot be made to run correctly without
   damaging or destabilizing the existing CUDA host environment.
2. Motion translation cannot be completed defensibly because source schemas,
   joint semantics, or Humanoid-GPT target semantics cannot be resolved with
   sufficient confidence.
3. SONIC-derived and HoloMotion-derived translations of the same canonical
   motion are not equivalent after deterministic canonicalization and
   investigation.
4. A working Humanoid-GPT deploy path cannot be established against the
   unchanged simulator boundary.
5. The only apparent way to make progress is to modify simulator code or
   simulator behavior.
6. Model capability is insufficient to complete a defensible motion translation
   or deploy integration under the constraints above.
