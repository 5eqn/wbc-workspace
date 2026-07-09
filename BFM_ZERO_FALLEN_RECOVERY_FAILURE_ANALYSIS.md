# BFM-Zero Fallen-Recovery Failure Analysis

Generated: `2026-07-09T07:21:24+08:00`

## Scope

- Run set: `fullstage1-bfm-zero-20260709a`
- Stage 2 summary: `logs/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage2/summary.json`
- Stage 3 summary: `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/summary.json`
- Population: `100` total runs, `92` successes, `8` failures
- Recovery criterion: final `0.5 s` mean root height `> 0.75 m`

## Failed Run Inventory

| Run | State | Source | Final mean root z (m) | Max root z (m) | Orientation | Self-contact pairs |
| --- | --- | --- | ---: | ---: | --- | --- |
| `run_008` | `state_008` | `random` | 0.060 | 0.358 | `supine` | dummy_lf_1 <-> left_rubber_hand; left_hip_roll_link <-> left_rubber_hand; left_knee_link <-> left_rubber_hand; left_rubber_hand <-> right_elbow_link; left_rubber_hand <-> right_wrist_roll_link; left_shoulder_yaw_link <-> torso_link; left_wrist_pitch_link <-> torso_link; pelvis <-> right_rubber_hand |
| `run_015` | `state_015` | `random` | 0.375 | 0.565 | `supine` | none |
| `run_019` | `state_019` | `random` | 0.059 | 0.123 | `right_side` | dummy_lf_3 <-> left_rubber_hand |
| `run_028` | `state_028` | `random` | 0.672 | 0.767 | `supine` | left_hip_roll_link <-> pelvis; left_wrist_roll_link <-> left_wrist_yaw_link; right_ankle_pitch_link <-> torso_link; right_ankle_roll_link <-> torso_link |
| `run_032` | `state_032` | `random` | 0.061 | 0.340 | `right_side` | dummy_lf_1 <-> pelvis; dummy_lf_3 <-> torso_link; left_wrist_roll_link <-> left_wrist_yaw_link |
| `run_051` | `state_051` | `random` | 0.163 | 0.188 | `supine` | dummy_lf_1 <-> pelvis; dummy_lf_3 <-> torso_link; left_knee_link <-> torso_link |
| `run_083` | `state_083` | `random` | 0.060 | 0.219 | `left_side` | left_shoulder_yaw_link <-> torso_link |
| `run_099` | `state_099` | `goal-derived` | 0.060 | 0.069 | `left_side` | none |

## Failed Video Exports

- Tiled video: `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/tiled_runs.mp4`
- Stage 1 tiled passive-settle video: `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/tiled_stage1_settle.mp4`
- Failed-video index: `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/failed_videos_index.json`
- Failed settle-video index: `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/failed_settle_videos_index.json`

Failed recovery videos:
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_008_state_008_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_015_state_015_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_019_state_019_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_028_state_028_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_032_state_032_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_051_state_051_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_083_state_083_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_099_state_099_1080p.mp4`

Failed passive-settle videos:
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_008_state_008_settle_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_015_state_015_settle_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_019_state_019_settle_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_028_state_028_settle_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_032_state_032_settle_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_051_state_051_settle_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_083_state_083_settle_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/run_099_state_099_settle_1080p.mp4`

## Strongest Scalar Separators

| Feature | Failure mean | Success mean | Mean diff | Cohen d | Point-biserial r |
| --- | ---: | ---: | ---: | ---: | ---: |
| `root_xy_radius_m` | 0.571 | 0.211 | 0.360 | 0.93 | 0.25 |
| `ankle_pitch_abs_sum_rad` | 1.419 | 1.051 | 0.368 | 0.85 | 0.23 |
| `joint_abs_max_rad` | 2.730 | 2.845 | -0.116 | -0.57 | -0.15 |
| `root_lin_speed_mps` | 0.038 | 0.011 | 0.027 | 0.53 | 0.14 |
| `joint_pos_l2_from_default` | 6.798 | 7.168 | -0.371 | -0.49 | -0.13 |
| `root_ang_speed_rps` | 0.278 | 0.103 | 0.175 | 0.42 | 0.11 |
| `leg_asymmetry_l1_rad` | 1.224 | 1.408 | -0.184 | -0.39 | -0.11 |
| `waist_abs_sum_rad` | 1.519 | 1.938 | -0.419 | -0.37 | -0.10 |

## Joint Position Outliers

| Joint | Failure mean (rad) | Success mean (rad) | Mean diff (rad) | Cohen d | r |
| --- | ---: | ---: | ---: | ---: | ---: |
| `right_ankle_pitch_joint` | -0.662 | -0.291 | -0.371 | -0.72 | -0.19 |
| `left_wrist_pitch_joint` | 0.575 | -0.109 | 0.684 | 0.72 | 0.19 |
| `left_shoulder_pitch_joint` | -0.723 | 0.294 | -1.017 | -0.59 | -0.16 |
| `left_elbow_joint` | 1.199 | 0.601 | 0.598 | 0.57 | 0.16 |
| `left_knee_joint` | 2.151 | 1.717 | 0.434 | 0.46 | 0.13 |
| `right_wrist_pitch_joint` | 0.593 | 0.105 | 0.488 | 0.45 | 0.12 |
| `right_hip_yaw_joint` | -0.677 | 0.057 | -0.734 | -0.44 | -0.12 |
| `right_shoulder_pitch_joint` | -1.008 | -0.207 | -0.801 | -0.41 | -0.11 |

## Joint Velocity Outliers

| Joint | Failure mean (rad/s) | Success mean (rad/s) | Mean diff (rad/s) | Cohen d | r |
| --- | ---: | ---: | ---: | ---: | ---: |
| `right_hip_pitch_joint` | 0.235 | 0.021 | 0.214 | 1.14 | 0.30 |
| `left_shoulder_roll_joint` | -0.170 | 0.009 | -0.179 | -0.90 | -0.24 |
| `right_shoulder_roll_joint` | 0.291 | 0.011 | 0.280 | 0.90 | 0.24 |
| `left_hip_pitch_joint` | 0.275 | -0.005 | 0.280 | 0.67 | 0.18 |
| `right_wrist_yaw_joint` | 0.185 | -0.086 | 0.272 | 0.55 | 0.15 |

## Orientation Bins

Orientation labels are approximate axis-based bins inferred from the floating-base quaternion, assuming the torso body frame uses `+x` forward, `+y` left, `+z` up.

| Orientation | Count | Failures | Failure rate |
| --- | ---: | ---: | ---: |
| `supine` | 29 | 4 | 13.79% |
| `left_side` | 18 | 2 | 11.11% |
| `right_side` | 29 | 2 | 6.90% |
| `prone` | 23 | 0 | 0.00% |
| `upside_down` | 1 | 0 | 0.00% |

## Self-Collision Signatures

| Body pair | Count | Failures | Failure rate |
| --- | ---: | ---: | ---: |
| `dummy_lf_3 <-> torso_link` | 2 | 2 | 100.00% |
| `dummy_lf_1 <-> left_rubber_hand` | 1 | 1 | 100.00% |
| `dummy_lf_3 <-> left_rubber_hand` | 1 | 1 | 100.00% |
| `left_knee_link <-> left_rubber_hand` | 1 | 1 | 100.00% |
| `left_rubber_hand <-> right_elbow_link` | 1 | 1 | 100.00% |
| `left_rubber_hand <-> right_wrist_roll_link` | 1 | 1 | 100.00% |
| `right_ankle_pitch_link <-> torso_link` | 1 | 1 | 100.00% |
| `right_ankle_roll_link <-> torso_link` | 1 | 1 | 100.00% |

## Per-Failure Notes

### `run_008` / `state_008`

- Severity: `grounded`
- Final mean root height: `0.060 m`
- Max root height during rollout: `0.358 m`
- Orientation bin: `supine`
- Self-contact pairs: `dummy_lf_1 <-> left_rubber_hand; left_hip_roll_link <-> left_rubber_hand; left_knee_link <-> left_rubber_hand; left_rubber_hand <-> right_elbow_link; left_rubber_hand <-> right_wrist_roll_link; left_shoulder_yaw_link <-> torso_link; left_wrist_pitch_link <-> torso_link; pelvis <-> right_rubber_hand`
- Floor-contact bodies: `dummy_lf_4; left_hip_pitch_link; left_hip_yaw_link; left_shoulder_yaw_link; right_hip_roll_link; right_hip_yaw_link; right_knee_link`
- Strongest joint-position deviations vs success set:
  - `waist_yaw_joint` = `2.531 rad` (success mean `-0.320 rad`, z-score `1.77`)
  - `right_hip_yaw_joint` = `-2.758 rad` (success mean `0.057 rad`, z-score `1.70`)
  - `waist_pitch_joint` = `0.521 rad` (success mean `0.014 rad`, z-score `1.44`)

### `run_015` / `state_015`

- Severity: `grounded`
- Final mean root height: `0.375 m`
- Max root height during rollout: `0.565 m`
- Orientation bin: `supine`
- Self-contact pairs: `none`
- Floor-contact bodies: `dummy_lf_4; left_rubber_hand; left_shoulder_yaw_link; left_wrist_pitch_link; right_hip_pitch_link; right_hip_yaw_link; right_rubber_hand; right_shoulder_yaw_link; right_wrist_pitch_link; torso_link`
- Strongest joint-position deviations vs success set:
  - `left_wrist_roll_joint` = `-1.973 rad` (success mean `0.017 rad`, z-score `1.81`)
  - `left_shoulder_yaw_joint` = `-2.475 rad` (success mean `0.080 rad`, z-score `1.58`)
  - `left_knee_joint` = `0.291 rad` (success mean `1.717 rad`, z-score `1.53`)

### `run_019` / `state_019`

- Severity: `grounded`
- Final mean root height: `0.059 m`
- Max root height during rollout: `0.123 m`
- Orientation bin: `right_side`
- Self-contact pairs: `dummy_lf_3 <-> left_rubber_hand`
- Floor-contact bodies: `dummy_lf_4; left_hip_yaw_link; left_wrist_pitch_link; right_ankle_roll_link; right_hip_roll_link; right_hip_yaw_link; right_rubber_hand; right_wrist_pitch_link; torso_link`
- Strongest joint-position deviations vs success set:
  - `left_hip_roll_joint` = `2.887 rad` (success mean `1.246 rad`, z-score `1.54`)
  - `right_shoulder_pitch_joint` = `-3.090 rad` (success mean `-0.207 rad`, z-score `1.48`)
  - `left_hip_pitch_joint` = `-2.533 rad` (success mean `0.235 rad`, z-score `1.47`)

### `run_028` / `state_028`

- Severity: `near_recovery`
- Final mean root height: `0.672 m`
- Max root height during rollout: `0.767 m`
- Orientation bin: `supine`
- Self-contact pairs: `left_hip_roll_link <-> pelvis; left_wrist_roll_link <-> left_wrist_yaw_link; right_ankle_pitch_link <-> torso_link; right_ankle_roll_link <-> torso_link`
- Floor-contact bodies: `dummy_lf_4; left_hip_yaw_link; left_rubber_hand; left_wrist_yaw_link; right_hip_roll_link; right_knee_link; right_rubber_hand; right_shoulder_yaw_link; torso_link`
- Strongest joint-position deviations vs success set:
  - `left_wrist_pitch_joint` = `1.580 rad` (success mean `-0.109 rad`, z-score `1.76`)
  - `right_wrist_roll_joint` = `-1.974 rad` (success mean `0.096 rad`, z-score `1.71`)
  - `left_ankle_pitch_joint` = `0.525 rad` (success mean `-0.344 rad`, z-score `1.69`)

### `run_032` / `state_032`

- Severity: `grounded`
- Final mean root height: `0.061 m`
- Max root height during rollout: `0.340 m`
- Orientation bin: `right_side`
- Self-contact pairs: `dummy_lf_1 <-> pelvis; dummy_lf_3 <-> torso_link; left_wrist_roll_link <-> left_wrist_yaw_link`
- Floor-contact bodies: `dummy_lf_2; dummy_lf_4; left_wrist_pitch_link; right_ankle_roll_link; right_hip_pitch_link; right_knee_link; right_rubber_hand; right_shoulder_yaw_link; right_wrist_pitch_link; torso_link`
- Strongest joint-position deviations vs success set:
  - `left_wrist_roll_joint` = `1.973 rad` (success mean `0.017 rad`, z-score `1.78`)
  - `left_wrist_pitch_joint` = `1.579 rad` (success mean `-0.109 rad`, z-score `1.76`)
  - `right_shoulder_pitch_joint` = `-3.049 rad` (success mean `-0.207 rad`, z-score `1.46`)

### `run_051` / `state_051`

- Severity: `grounded`
- Final mean root height: `0.163 m`
- Max root height during rollout: `0.188 m`
- Orientation bin: `supine`
- Self-contact pairs: `dummy_lf_1 <-> pelvis; dummy_lf_3 <-> torso_link; left_knee_link <-> torso_link`
- Floor-contact bodies: `left_hip_pitch_link; left_hip_roll_link; left_wrist_pitch_link; left_wrist_roll_link; pelvis; right_hip_yaw_link; right_knee_link; right_rubber_hand; torso_link`
- Strongest joint-position deviations vs success set:
  - `right_hip_yaw_joint` = `-2.604 rad` (success mean `0.057 rad`, z-score `1.61`)
  - `right_wrist_yaw_joint` = `1.618 rad` (success mean `-0.044 rad`, z-score `1.57`)
  - `right_shoulder_roll_joint` = `-2.253 rad` (success mean `-1.014 rad`, z-score `1.52`)

### `run_083` / `state_083`

- Severity: `grounded`
- Final mean root height: `0.060 m`
- Max root height during rollout: `0.219 m`
- Orientation bin: `left_side`
- Self-contact pairs: `left_shoulder_yaw_link <-> torso_link`
- Floor-contact bodies: `dummy_lf_3; left_hip_roll_link; left_knee_link; left_rubber_hand; left_wrist_pitch_link; left_wrist_yaw_link; right_ankle_roll_link; right_hip_pitch_link; right_knee_link; right_wrist_pitch_link`
- Strongest joint-position deviations vs success set:
  - `left_wrist_yaw_joint` = `1.556 rad` (success mean `-0.085 rad`, z-score `1.71`)
  - `left_elbow_joint` = `2.102 rad` (success mean `0.601 rad`, z-score `1.43`)
  - `right_hip_roll_joint` = `0.246 rad` (success mean `-1.209 rad`, z-score `1.40`)

### `run_099` / `state_099`

- Severity: `grounded`
- Final mean root height: `0.060 m`
- Max root height during rollout: `0.069 m`
- Orientation bin: `left_side`
- Self-contact pairs: `none`
- Floor-contact bodies: `dummy_lf_1; dummy_lf_2; pelvis; right_ankle_roll_link`
- Strongest joint-position deviations vs success set:
  - `left_ankle_pitch_joint` = `0.527 rad` (success mean `-0.344 rad`, z-score `1.70`)
  - `right_ankle_pitch_joint` = `0.527 rad` (success mean `-0.291 rad`, z-score `1.58`)
  - `waist_pitch_joint` = `0.522 rad` (success mean `0.014 rad`, z-score `1.44`)

## Best-Effort Findings

- The failed set is small (`8` runs), so the strongest signals should be treated as ranking cues rather than hard causal proof.
- The single strongest pose-level separator is `root_xy_radius_m` with Cohen d `0.93`.
- The single strongest joint-position separator is `right_ankle_pitch_joint` with Cohen d `-0.72`.
- The most failure-loaded self-contact pair is `dummy_lf_3 <-> torso_link` at `100.00%` failure rate when present.
- `1` of the `8` failures were near misses; the rest stayed substantially below the recovery threshold.

## Caveats

- This analysis is intentionally best effort and observational. It ranks initial-state traits associated with failure but does not isolate policy causality from all confounders.
- Contact features were computed from the Stage 1 saved state at rollout start, not from the entire rollout.
- Machine-readable details are available in `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260709a/stage3/failure_analysis.json`.
