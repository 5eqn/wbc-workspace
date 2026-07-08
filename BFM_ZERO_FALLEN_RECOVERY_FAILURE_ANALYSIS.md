# BFM-Zero Fallen-Recovery Failure Analysis

Generated: `2026-07-08T22:43:59+08:00`

## Scope

- Run set: `fullstage1-bfm-zero-20260708a`
- Stage 2 summary: `logs/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage2/summary.json`
- Stage 3 summary: `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/summary.json`
- Population: `100` total runs, `91` successes, `9` failures
- Recovery criterion: final `0.5 s` mean root height `> 0.75 m`

## Failed Run Inventory

| Run | State | Source | Final mean root z (m) | Max root z (m) | Orientation | Self-contact pairs |
| --- | --- | --- | ---: | ---: | --- | --- |
| `run_003` | `state_003` | `random` | 0.173 | 0.190 | `supine` | right_shoulder_roll_link <-> torso_link |
| `run_024` | `state_024` | `random` | 0.152 | 0.182 | `left_side` | left_elbow_link <-> right_shoulder_pitch_link; left_elbow_link <-> torso_link; left_shoulder_roll_link <-> torso_link; left_shoulder_yaw_link <-> torso_link; left_wrist_roll_link <-> right_shoulder_pitch_link |
| `run_035` | `state_035` | `random` | 0.201 | 0.299 | `upside_down` | none |
| `run_040` | `state_040` | `random` | 0.175 | 0.211 | `left_side` | left_shoulder_roll_link <-> torso_link; left_shoulder_yaw_link <-> right_rubber_hand; left_shoulder_yaw_link <-> torso_link; left_wrist_roll_link <-> left_wrist_yaw_link; right_rubber_hand <-> torso_link; right_shoulder_roll_link <-> torso_link |
| `run_056` | `state_056` | `random` | 0.061 | 0.185 | `left_side` | left_shoulder_roll_link <-> torso_link |
| `run_060` | `state_060` | `random` | 0.062 | 0.092 | `right_side` | none |
| `run_072` | `state_072` | `random` | 0.102 | 0.227 | `right_side` | left_wrist_roll_link <-> left_wrist_yaw_link; right_ankle_roll_link <-> torso_link; right_knee_link <-> torso_link; right_shoulder_roll_link <-> torso_link |
| `run_075` | `state_075` | `random` | 0.703 | 0.774 | `left_side` | right_wrist_roll_link <-> right_wrist_yaw_link |
| `run_099` | `state_099` | `goal-derived` | 0.061 | 0.067 | `left_side` | none |

## Failed Video Exports

- Tiled video: `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/tiled_runs.mp4`
- Failed-video index: `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/failed_videos_index.json`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/run_003_state_003_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/run_024_state_024_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/run_035_state_035_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/run_040_state_040_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/run_056_state_056_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/run_060_state_060_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/run_072_state_072_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/run_075_state_075_1080p.mp4`
- `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/run_099_state_099_1080p.mp4`

## Strongest Scalar Separators

| Feature | Failure mean | Success mean | Mean diff | Cohen d | Point-biserial r |
| --- | ---: | ---: | ---: | ---: | ---: |
| `joint_pos_l2_from_default` | 6.069 | 6.917 | -0.847 | -1.16 | -0.32 |
| `joint_abs_mean_rad` | 0.889 | 1.025 | -0.136 | -1.11 | -0.31 |
| `joint_abs_max_rad` | 2.594 | 2.792 | -0.198 | -0.87 | -0.24 |
| `leg_joint_abs_mean_rad` | 0.912 | 1.068 | -0.156 | -0.78 | -0.22 |
| `root_xy_radius_m` | 0.472 | 0.177 | 0.295 | 0.77 | 0.22 |
| `arm_joint_abs_mean_rad` | 0.926 | 1.068 | -0.142 | -0.71 | -0.20 |
| `arm_asymmetry_l1_rad` | 1.225 | 1.478 | -0.254 | -0.65 | -0.18 |
| `joint_vel_rms` | 2.870 | 3.313 | -0.442 | -0.44 | -0.13 |

## Joint Position Outliers

| Joint | Failure mean (rad) | Success mean (rad) | Mean diff (rad) | Cohen d | r |
| --- | ---: | ---: | ---: | ---: | ---: |
| `right_knee_joint` | 1.036 | 1.671 | -0.635 | -0.70 | -0.20 |
| `right_shoulder_yaw_joint` | 0.769 | -0.324 | 1.093 | 0.70 | 0.20 |
| `right_ankle_pitch_joint` | 0.050 | -0.266 | 0.316 | 0.69 | 0.20 |
| `left_hip_roll_joint` | 0.675 | 1.450 | -0.775 | -0.68 | -0.19 |
| `left_shoulder_roll_joint` | 0.236 | 0.793 | -0.557 | -0.63 | -0.18 |
| `left_ankle_roll_joint` | 0.119 | 0.020 | 0.100 | 0.52 | 0.15 |
| `right_hip_roll_joint` | -1.630 | -1.213 | -0.417 | -0.36 | -0.10 |
| `right_shoulder_pitch_joint` | 0.113 | -0.440 | 0.553 | 0.32 | 0.09 |

## Joint Velocity Outliers

| Joint | Failure mean (rad/s) | Success mean (rad/s) | Mean diff (rad/s) | Cohen d | r |
| --- | ---: | ---: | ---: | ---: | ---: |
| `right_ankle_roll_joint` | -0.653 | 0.116 | -0.769 | -0.49 | -0.14 |
| `right_hip_pitch_joint` | 1.092 | -0.391 | 1.482 | 0.38 | 0.11 |
| `right_wrist_roll_joint` | -0.703 | 0.161 | -0.864 | -0.38 | -0.11 |
| `left_shoulder_pitch_joint` | 1.671 | 0.204 | 1.467 | 0.37 | 0.11 |
| `waist_pitch_joint` | 0.724 | -0.244 | 0.968 | 0.35 | 0.10 |

## Orientation Bins

Orientation labels are approximate axis-based bins inferred from the floating-base quaternion, assuming the torso body frame uses `+x` forward, `+y` left, `+z` up.

| Orientation | Count | Failures | Failure rate |
| --- | ---: | ---: | ---: |
| `left_side` | 27 | 5 | 18.52% |
| `upside_down` | 6 | 1 | 16.67% |
| `right_side` | 20 | 2 | 10.00% |
| `supine` | 19 | 1 | 5.26% |
| `prone` | 19 | 0 | 0.00% |
| `upright` | 9 | 0 | 0.00% |

## Self-Collision Signatures

| Body pair | Count | Failures | Failure rate |
| --- | ---: | ---: | ---: |
| `left_elbow_link <-> right_shoulder_pitch_link` | 1 | 1 | 100.00% |
| `left_shoulder_yaw_link <-> right_rubber_hand` | 1 | 1 | 100.00% |
| `left_wrist_roll_link <-> right_shoulder_pitch_link` | 2 | 1 | 50.00% |
| `right_ankle_roll_link <-> torso_link` | 2 | 1 | 50.00% |
| `right_rubber_hand <-> torso_link` | 2 | 1 | 50.00% |
| `right_shoulder_roll_link <-> torso_link` | 8 | 3 | 37.50% |
| `right_knee_link <-> torso_link` | 3 | 1 | 33.33% |
| `left_elbow_link <-> torso_link` | 4 | 1 | 25.00% |

## Per-Failure Notes

### `run_003` / `state_003`

- Severity: `grounded`
- Final mean root height: `0.173 m`
- Max root height during rollout: `0.190 m`
- Orientation bin: `supine`
- Self-contact pairs: `right_shoulder_roll_link <-> torso_link`
- Floor-contact bodies: `left_shoulder_yaw_link; right_ankle_roll_link; right_hip_roll_link; right_hip_yaw_link; right_knee_link`
- Strongest joint-position deviations vs success set:
  - `waist_yaw_joint` = `-2.101 rad` (success mean `0.093 rad`, z-score `1.44`)
  - `left_knee_joint` = `2.728 rad` (success mean `1.356 rad`, z-score `1.43`)
  - `right_shoulder_pitch_joint` = `2.034 rad` (success mean `-0.440 rad`, z-score `1.40`)

### `run_024` / `state_024`

- Severity: `grounded`
- Final mean root height: `0.152 m`
- Max root height during rollout: `0.182 m`
- Orientation bin: `left_side`
- Self-contact pairs: `left_elbow_link <-> right_shoulder_pitch_link; left_elbow_link <-> torso_link; left_shoulder_roll_link <-> torso_link; left_shoulder_yaw_link <-> torso_link; left_wrist_roll_link <-> right_shoulder_pitch_link`
- Floor-contact bodies: `dummy_lf_1; left_hip_pitch_link; right_wrist_pitch_link; torso_link`
- Strongest joint-position deviations vs success set:
  - `left_shoulder_roll_joint` = `-1.366 rad` (success mean `0.793 rad`, z-score `2.59`)
  - `right_shoulder_roll_joint` = `-2.233 rad` (success mean `-0.776 rad`, z-score `2.00`)
  - `right_ankle_pitch_joint` = `0.524 rad` (success mean `-0.266 rad`, z-score `1.77`)

### `run_035` / `state_035`

- Severity: `grounded`
- Final mean root height: `0.201 m`
- Max root height during rollout: `0.299 m`
- Orientation bin: `upside_down`
- Self-contact pairs: `none`
- Floor-contact bodies: `dummy_lf_4; torso_link`
- Strongest joint-position deviations vs success set:
  - `left_hip_pitch_joint` = `2.893 rad` (success mean `-0.050 rad`, z-score `1.88`)
  - `right_shoulder_yaw_joint` = `2.624 rad` (success mean `-0.324 rad`, z-score `1.86`)
  - `waist_yaw_joint` = `-2.333 rad` (success mean `0.093 rad`, z-score `1.59`)

### `run_040` / `state_040`

- Severity: `grounded`
- Final mean root height: `0.175 m`
- Max root height during rollout: `0.211 m`
- Orientation bin: `left_side`
- Self-contact pairs: `left_shoulder_roll_link <-> torso_link; left_shoulder_yaw_link <-> right_rubber_hand; left_shoulder_yaw_link <-> torso_link; left_wrist_roll_link <-> left_wrist_yaw_link; right_rubber_hand <-> torso_link; right_shoulder_roll_link <-> torso_link`
- Floor-contact bodies: `left_shoulder_pitch_link; right_rubber_hand`
- Strongest joint-position deviations vs success set:
  - `left_shoulder_roll_joint` = `-2.034 rad` (success mean `0.793 rad`, z-score `3.39`)
  - `waist_yaw_joint` = `2.689 rad` (success mean `0.093 rad`, z-score `1.70`)
  - `waist_roll_joint` = `-0.527 rad` (success mean `0.026 rad`, z-score `1.45`)

### `run_056` / `state_056`

- Severity: `grounded`
- Final mean root height: `0.061 m`
- Max root height during rollout: `0.185 m`
- Orientation bin: `left_side`
- Self-contact pairs: `left_shoulder_roll_link <-> torso_link`
- Floor-contact bodies: `left_elbow_link; right_elbow_link; right_hip_pitch_link; right_shoulder_yaw_link; right_wrist_pitch_link; right_wrist_roll_link; right_wrist_yaw_link`
- Strongest joint-position deviations vs success set:
  - `right_knee_joint` = `-0.075 rad` (success mean `1.671 rad`, z-score `1.93`)
  - `right_hip_yaw_joint` = `-2.754 rad` (success mean `-0.068 rad`, z-score `1.62`)
  - `right_elbow_joint` = `-0.701 rad` (success mean `0.772 rad`, z-score `1.51`)

### `run_060` / `state_060`

- Severity: `grounded`
- Final mean root height: `0.062 m`
- Max root height during rollout: `0.092 m`
- Orientation bin: `right_side`
- Self-contact pairs: `none`
- Floor-contact bodies: `left_shoulder_pitch_link; pelvis; right_ankle_roll_link; torso_link`
- Strongest joint-position deviations vs success set:
  - `right_knee_joint` = `0.278 rad` (success mean `1.671 rad`, z-score `1.54`)
  - `right_wrist_yaw_joint` = `-1.463 rad` (success mean `-0.100 rad`, z-score `1.53`)
  - `right_ankle_pitch_joint` = `0.412 rad` (success mean `-0.266 rad`, z-score `1.52`)

### `run_072` / `state_072`

- Severity: `grounded`
- Final mean root height: `0.102 m`
- Max root height during rollout: `0.227 m`
- Orientation bin: `right_side`
- Self-contact pairs: `left_wrist_roll_link <-> left_wrist_yaw_link; right_ankle_roll_link <-> torso_link; right_knee_link <-> torso_link; right_shoulder_roll_link <-> torso_link`
- Floor-contact bodies: `dummy_lf_2; dummy_lf_4; left_knee_link; right_elbow_link`
- Strongest joint-position deviations vs success set:
  - `left_wrist_yaw_joint` = `1.568 rad` (success mean `-0.040 rad`, z-score `1.75`)
  - `right_ankle_pitch_joint` = `0.455 rad` (success mean `-0.266 rad`, z-score `1.61`)
  - `left_hip_pitch_joint` = `-2.512 rad` (success mean `-0.050 rad`, z-score `1.57`)

### `run_075` / `state_075`

- Severity: `near_recovery`
- Final mean root height: `0.703 m`
- Max root height during rollout: `0.774 m`
- Orientation bin: `left_side`
- Self-contact pairs: `right_wrist_roll_link <-> right_wrist_yaw_link`
- Floor-contact bodies: `dummy_lf_4; right_elbow_link; right_knee_link; right_shoulder_yaw_link`
- Strongest joint-position deviations vs success set:
  - `right_hip_pitch_joint` = `-2.545 rad` (success mean `0.342 rad`, z-score `1.81`)
  - `right_wrist_pitch_joint` = `-1.601 rad` (success mean `0.191 rad`, z-score `1.73`)
  - `left_wrist_pitch_joint` = `-1.498 rad` (success mean `0.169 rad`, z-score `1.70`)

### `run_099` / `state_099`

- Severity: `grounded`
- Final mean root height: `0.061 m`
- Max root height during rollout: `0.067 m`
- Orientation bin: `left_side`
- Self-contact pairs: `none`
- Floor-contact bodies: `dummy_lf_1; dummy_lf_2; pelvis; right_ankle_roll_link`
- Strongest joint-position deviations vs success set:
  - `right_ankle_pitch_joint` = `0.527 rad` (success mean `-0.266 rad`, z-score `1.77`)
  - `left_ankle_pitch_joint` = `0.527 rad` (success mean `-0.156 rad`, z-score `1.34`)
  - `waist_pitch_joint` = `0.522 rad` (success mean `0.005 rad`, z-score `1.29`)

## Best-Effort Findings

- The failed set is small (`9` runs), so the strongest signals should be treated as ranking cues rather than hard causal proof.
- The single strongest pose-level separator is `joint_pos_l2_from_default` with Cohen d `-1.16`.
- The single strongest joint-position separator is `right_knee_joint` with Cohen d `-0.70`.
- The most failure-loaded self-contact pair is `left_elbow_link <-> right_shoulder_pitch_link` at `100.00%` failure rate when present.
- `1` of the `9` failures were near misses; the rest stayed substantially below the recovery threshold.

## Caveats

- This analysis is intentionally best effort and observational. It ranks initial-state traits associated with failure but does not isolate policy causality from all confounders.
- Contact features were computed from the Stage 1 saved state at rollout start, not from the entire rollout.
- Machine-readable details are available in `artifacts/bfm-zero-fallen-recovery/fullstage1-bfm-zero-20260708a/stage3/failure_analysis.json`.
