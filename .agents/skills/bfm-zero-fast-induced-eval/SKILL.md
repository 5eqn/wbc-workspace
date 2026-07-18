---
name: bfm-zero-fast-induced-eval
description: Run and verify the repository's provider-aware headless, batched BFM-Zero or Legged Lab AMP velocity-induced fall evaluation, accepted-only replay collection, and Stage 3 export. Use for GPU smokes, resumable sweeps, dynamic ONNX validation, schema-v2 BFM or schema-v3 AMP replay inspection, model recomputation, tiled video, or BFM latent-inspector generation.
---

# Provider-Aware Fast Induced Evaluation

Run commands from the repository root. Treat `thirdparties/BFM-Zero`,
`thirdparties/BFM-Zero-deploy`, the existing recovery scripts, and
`/home/seqn/legged_lab` as read-only. The default provider is `bfm-zero`; select
the pinned 570-to-29 policy with `--model-provider legged-lab-amp`.

## Prepare and verify

1. Read `STATE.md` and inspect active GPU and deployer processes.
2. Sync and run host gates:

```bash
uv sync --project scripts/bfm_zero_fast_induced_eval --all-groups
uv run --project scripts/bfm_zero_fast_induced_eval ruff check scripts/bfm_zero_fast_induced_eval
uv run --project scripts/bfm_zero_fast_induced_eval pytest -q scripts/bfm_zero_fast_induced_eval/tests
uv run --project scripts/bfm_zero_fast_induced_eval bfm-zero-fast-induced-eval preflight
uv run --project scripts/bfm_zero_fast_induced_eval bfm-zero-fast-induced-eval patch-onnx
```

Require at least 22 GiB free, the known old-model hashes, CUDA ONNX Runtime,
exact batch-1 parity, and successful batch sizes 2 and 128. Stop if DDS, a
deployer, or the sequential BFM recovery evaluator is active.

For AMP, run preflight and patching with `--model-provider legged-lab-amp`.
Require the pinned ONNX, sibling TorchScript, and `params/env.yaml`, batch-1
TorchScript/ONNX parity, and CUDA batches 2 and 128. AMP owns its five-frame
term-major chronological history, yaw-local 6D rotation, six ordered key-body
positions, previous-action injection, and raw-action conversion. Joint position,
joint velocity, previous action, and ONNX output use the policy's Isaac
articulation-native joint order; only the simulator/replay boundary uses canonical
Unitree SDK order. Validate both orders independently.

The evaluator-local pose boundary is fixed: HumanoidVerse reset/source root
states are XYZW, Isaac Sim root-pose writes are WXYZ, and replay `qpos` is
MuJoCo WXYZ. The evaluator converts before the default-pose reset and every
full root-state write. It requires an upright-identity MuJoCo quaternion
immediately after reset and world-up Z greater than `0.9` after warmup and
after the kick. This complete contract is recorded in the manifest and is
resume-critical.

For a custom actor/checkpoint pair, pass all matching `--actor`, `--checkpoint`,
and `--goals` paths plus `--allow-custom-model`. Verify checkpoint-to-ONNX
parity separately before collection; the flag is an explicit bypass of the
pinned-old-model hash gate, not a pairing check.

## Run collection

Use a distinct run ID. First run a two-environment GPU smoke, then one complete
128-environment round:

```bash
uv run --project scripts/bfm_zero_fast_induced_eval bfm-zero-fast-induced-eval run \
  --run-id smoke-YYYYMMDD --num-envs 2 --rounds 1
uv run --project scripts/bfm_zero_fast_induced_eval bfm-zero-fast-induced-eval run \
  --run-id round-gate-YYYYMMDD --num-envs 128 --rounds 1
```

Do not start the default 100-round sweep during implementation unless the user
explicitly asks for the dataset. Resume a requested sweep by repeating the
same command and run ID; valid completed rounds are skipped and `.tmp` rounds
are regenerated.

Validated AMP evidence uses
`g1-amp-get-up-20260717-jointorderfix-smoke-2x1-20260718` (`2/2` accepted,
`1/2` recovered) and
`g1-amp-get-up-20260717-jointorderfix-128x3-20260718` (`383/384` accepted,
`158/383` recovered). Do not add rounds to the latter. The older similarly named
runs without `jointorderfix` fed canonical SDK-ordered joint terms to an
Isaac-native-order policy and are invalid, superseded contract evidence; retain
their artifacts but do not use their zero-recovery results.

## Inspect output

Require `manifest.json`, `summary.json`, and valid `round_NNN.h5` files under
`logs/bfm-zero-fast-induced/<run-id>/`. Confirm:

- all attempts have seed, disturbance, model hash, and metric metadata;
- only first-second induced falls are stored;
- schema version is 2; `qpos` is `(accepted, 401, 36)` in root XYZ/root WXYZ/29-joint
  order and `qvel` is `(accepted, 401, 35)` in root linear/root angular/29-joint order;
- every stored observation and full-state field has 401 frames while action/boundary fields
  have 400 transitions;
- tensors are finite and transition `t` indexes observation `t`, action `t`, and observation `t+1`;
- frame-0 root quaternions are finite, normalized, and have positive world-up Z; sampled
  `qpos` frames load and render in
  `thirdparties/BFM-Zero-deploy/data/robots/g1/scene_29dof_freebase.xml`;
- the summary reports attempted, accepted, discarded, recovered, conditional rate, Wilson interval, throughput, and wall time.

AMP uses schema v3 with `provider_input (accepted,401,570)`, common full state,
raw canonical action, and boundary fields. Provider identity/layout/history,
joint/body order, control/physics/pose contracts, and hashes are resume-critical.
The manifest records the irreducible HumanoidVerse 4/0 versus training 8/4
articulation solver-iteration difference. BFM schema-v2 remains unchanged.

Recompute model outputs on sampled stored frames with:

```bash
uv run --project scripts/bfm_zero_fast_induced_eval bfm-zero-fast-induced-eval \
  verify-recompute logs/bfm-zero-fast-induced/<run-id>/round_000.h5
```

Recomputation dispatches by manifest provider. AMP uses stored input with the
recorded dynamic ONNX plus sampled TorchScript parity; BFM retains checkpoint
actor/F/D/QD/Q recomputation.

## Export Stage 3 and inspector data

Do not rerun the simulator when valid schema-v2 replay already exists. Generate a fixed
100-trial Stage 3 export from the first accepted rows in numeric round/stored order:

```bash
thirdparties/BFM-Zero/.venv/bin/python scripts/bfm_zero_fallen_recovery_eval.py stage3 \
  --fast-run-dir logs/bfm-zero-fast-induced/<run-id>
```

Require at least 100 usable trials, schema version 2, finite exact shapes, existing model
artifacts, and recorded model hashes. The output under
`artifacts/bfm-zero-fast-induced/<run-id>/stage3/` must be a 10×10, 1920×1080,
25 FPS, 201-frame tiled video plus metrics, plots, and a summary containing source kind,
selected attempt IDs, goal hash, model hashes, and artifact paths. Per-failure videos are
not part of the fast default.

For AMP schema v3, this command exports recovery labels, pelvis-height,
world-up-Z, and action-norm metrics and plots, plus the deterministic first-100
10x10, 1920x1080, 25 FPS, 201-frame video. AMP is explicitly excluded from the
BFM latent inspector and B/F/D/QD/Q diagnostics because those interfaces do not
exist; do not fabricate latent data.

Regenerate the three-source static inspector with:

```bash
thirdparties/BFM-Zero/.venv/bin/python scripts/bfm_zero_latent_inspector.py --device cuda
```

The fast path must use stored observations/actions directly, hold action 399 for final
observation 400, calculate diagnostics and discounted ground truth at 50 Hz, and select
frames `0,2,…,400` for the 25 Hz UI. Keep the Stage 2 telemetry reconstruction path for
legacy datasets. Fit PCA independently per dataset, store its mean/components with that
dataset, and reject any projected value outside the fixed `[-16,16]³` cube.

Validate `data.json` has three 100-trial datasets, finite arrays, per-dataset projection
metadata, matching model/goal hashes, and video-aligned frame counts. Recompute sampled
fast B/F/D/QD/ground-truth values directly from HDF5 and checkpoint data. Finally, switch
through all three datasets in headless Chrome at 1920×1080, require 100 interactive cells,
synchronized timeline/plots/video, and no console or loading errors before refreshing the
browser proof.
