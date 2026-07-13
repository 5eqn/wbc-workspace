# Minimal Heracles Planner Reproduction with SONIC

## Status

The host package, locked environment, dataset preprocessing, model, training gate, ONNX export,
runtime adapter, SONIC protocol-v1 encoder, opt-in simulator state feed, and paired-metric selection
code are implemented.

The complete RTX 3090 gate projects 23.50 hours for 4,000 epochs, including validation and a
conservative two checkpoint writes per epoch, so the required full run was started. It is resumable
and updates `artifacts/heracles-planner/checkpoints/training_status.json` after every epoch. At this
document update it had completed epoch 15 of 3,999 with a live ETA of about 15.5 hours.

A one-epoch preflight checkpoint passed deterministic CPU PyTorch/ONNX parity at a maximum absolute
error of `1.79e-6`. CUDA ONNX inference passed the 25 Hz gate at 5.53 ms mean and 6.54 ms p99. These
are pipeline smoke results, not final-model results. Until the full run, final export, and integrated
evaluation finish, there is no final ONNX model, paired statistic, or comparison video, and no
downstream outcome should be inferred.

Evidence:

- `logs/heracles-planner/data/manifest.json`: source hash, all 40 motions, splits, frame counts,
  conventions, and Jacobian-weight provenance.
- `logs/heracles-planner/data/normalization.npz`: train-only mean/std and loss weights.
- `artifacts/heracles-planner/training_duration_gate.json`: measured duration gate.
- `artifacts/heracles-planner/smoke-onnx-parity.json`: preflight export parity.
- `artifacts/heracles-planner/smoke-inference-25hz.json`: preflight runtime rate.
- `uv.lock`: exact host dependency resolution.

## Reproduction commands

```bash
uv sync --frozen
uv run pytest -q
uv run heracles-planner preprocess
uv run heracles-planner inspect-model \
  --output artifacts/heracles-planner/model_summary.json
uv run heracles-planner benchmark-training --steps 50
```

The training command resumes from `last.pt` when present:

```bash
uv run heracles-planner train
```

Only after an allowed full run produces `best.pt`:

```bash
uv run heracles-planner export \
  --checkpoint artifacts/heracles-planner/checkpoints/best.pt
uv run heracles-planner benchmark-inference \
  --model artifacts/heracles-planner/heracles.onnx
```

## Dataset and conventions

The read-only source is
`thirdparties/BFM-Zero/humanoidverse/data/lafan_29dof.pkl`. It is a Joblib archive with exactly 40
G1 motions at 30 Hz. It stores joints in the G1 29-DOF IsaacLab order and root quaternions in
`xyzw`; SONIC protocol v1 receives the same joint order and `wxyz` quaternions, so the adapter only
reorders quaternion components.

The fixed split is subject-based:

| Split | Subjects | Motions | Use |
| --- | --- | ---: | --- |
| Train | 1, 2, 3 | 26 | Optimization and normalization only |
| Validation | 4 | 7 | Checkpoint selection only |
| Test | 5 | 7 | Paired evaluation only |

`fallAndGetUp1_subject5` is the sole get-up test. The other six subject-5 motions are normal tests.
Joint positions and root translation are resampled by cubic splines. Root orientation is resampled
by quaternion slerp after normalization and sign unwrapping. Only missing fields, wrong shapes,
non-finite values, inconsistent lengths, wrong FPS, or invalid quaternion norms are rejected.

An epoch is one pass over every training start that has a complete 2.0-second source segment,
using starts every 0.04 seconds. This gives 137,508 training windows. Each visit samples a segment
duration log-uniformly in `[0.2, 2.0]` seconds and evaluates eight keyframes across that segment.

## Planner definition

The 35D state is 29 joint angles followed by root Rot6D (the first two rotation-matrix columns).
The generated value is eight 35D residual keyframes. Joint residuals are ordinary angle
differences. Rot6D residuals are representation-space differences; after adding them to the current
state, each result is Gram-Schmidt projected back onto SO(3). Token zero is identically zero in the
training path, target, model output, and Euler integration.

The conditional flow path is

```text
x_t = (1 - t) x_data + t x_noise
v_target = x_noise - x_data
```

Generation begins at `t=0.9` with `0.9 * noise + 0.1 * directional_prefix`, then takes five equal
backward-Euler steps to zero. The directional prefix is the next 0.2 seconds of the untouched
reference, expressed as residuals from measured state. Eight generated keyframes are expanded to
50 Hz with cubic joint interpolation and rotational slerp. Replanning is 25 Hz.

The model has 22,641,187 trainable parameters (1.1% below the approximate 22.9M target): six
AdaLN-Zero transformer blocks, width 512, four attention heads, MLP ratio 2, GELU, sinusoidal token
positions, and no dropout.

## Parameter provenance

“Reported” means specified as a Heracles setting in the reproduction request. “Inferred” means
required to make two reported requirements operational without adding a tunable choice.
“Unspecified-and-chosen” records the fixed value mandated for an omission or a necessary local
implementation choice.

| Parameter | Value | Provenance |
| --- | --- | --- |
| State | 29 joints + root Rot6D (35D) | Reported |
| Output | 8 residual keyframes over 0.2 s | Reported |
| Transformer | 6 blocks, 4 heads, width 512 | Reported |
| Parameter target | approximately 22.9M | Reported |
| Flow | conditional linear path and velocity target | Reported |
| Inpainting | zero residual at first token | Reported |
| Warm start | directional, `t=0.9` | Reported |
| Solver | 5 backward-Euler steps from 0.9 to 0 | Reported |
| Output interpolation | cubic joints, rotational slerp | Reported |
| Replanning | 25 Hz | Reported |
| Optimizer | AdamW | Reported |
| Batch size | 256 | Reported |
| Learning rate | `1e-4`, cosine decay | Reported |
| Weight decay | `1e-4` | Reported |
| Epochs | 4,000 | Reported |
| Window stride | 0.04 s | Unspecified-and-chosen |
| Segment length | log-uniform 0.2–2.0 s | Unspecified-and-chosen |
| Joint-state noise | Gaussian, sigma 0.05 rad | Unspecified-and-chosen |
| Root noise | SO(3) tangent Gaussian, sigma 0.02 rad | Unspecified-and-chosen |
| Jacobian FD step | `1e-3` rad | Unspecified-and-chosen |
| Jacobian weights | normalize mean to 1; floor 0.1 | Unspecified-and-chosen |
| MLP | ratio 2, GELU | Unspecified-and-chosen |
| Positions | sinusoidal | Unspecified-and-chosen |
| Dropout | none | Unspecified-and-chosen |
| Gradient clipping | global norm 1.0 | Unspecified-and-chosen |
| LR warmup | none | Unspecified-and-chosen |
| Seed | 42 | Unspecified-and-chosen |
| Normalization | training split only | Unspecified-and-chosen |
| Valid start rule | enough remaining data for the full 2.0 s maximum segment | Inferred |
| Duration conditioning | add fixed sinusoidal duration embedding to AdaLN condition | Inferred |
| Rot6D residual reconstruction | add then orthonormalize | Inferred |
| Warm-start direction | untouched original 0.2 s reference prefix | Inferred |
| SONIC reference window | generated frames 0–0.2 s, untouched source thereafter to 0.9 s | Reported |
| Derived velocity | 50 Hz numerical gradient of the assembled joint window | Inferred |
| Quaternion wire order | source/internal `xyzw`; SONIC field `body_quat_w` uses `wxyz` | Inferred from local formats |
| Jacobian model | repo SONIC G1 29-DOF MJCF at neutral configuration; all body positions | Unspecified-and-chosen |
| CUDA determinism | cuBLAS workspace `:4096:8`, math SDP only | Unspecified-and-chosen |
| Duration estimate statistic | median of 50 end-to-end train and validation steps after 5 warmups; median of 3 checkpoint writes | Unspecified-and-chosen |

## SONIC attachment

The default simulator and SONIC paths are unchanged. `scripts/sim_bridge.py` has two opt-in flags:
`--state-zmq-port` and `--state-zmq-topic`. When enabled, it publishes measured root quaternion,
29 joint positions, 29 joint velocities, and frame index in the same fixed-header packed format.
The host service subscribes to that feed and publishes SONIC protocol v1 at `pose`.

Example future launch parameters are state port 15558 and pose port 5556. SONIC must use its
documented ZMQ input mode. The service writes every measured state, generated keyframe/residual,
warm start, published reference, and derived velocity to JSONL.

No third-party source file was patched.

## Evaluation contract

`heracles-planner summarize-evaluation` consumes paired trial NPZ files and implements the required
metrics and deterministic top-three win/loss selection. Normal ranking is completion then lower
joint RMSE. Get-up and disturbed ranking is final-0.5-second stand-up success then lower joint RMSE.
The disturbed RMSE mask begins at exactly 0.5 seconds. The stand-up statistic uses mean absolute
root-height error below 0.3 m in the final 0.5 seconds.

The planned induced disturbance remains exactly one paired linear delta of 3–6 m/s and angular
delta of 4–8 rad/s at 0.5 seconds, with independent uniform sphere directions, only on motions at
least 2.5 seconds. Since full training is still active, no disturbance has been applied and no
evaluation statistic or video is claimed.
