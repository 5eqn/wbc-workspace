# BFM-Zero Induced-Fall Smoke Guide

Run commands from the repository root. An induced-fall evaluation uses a
matched pair of model artifacts:

- `--model-checkpoint` supplies the encoder used to create the 256-D goal
  context.
- `--actor-onnx` supplies the actor used for closed-loop control.

Do not change only one of these arguments. A checkpoint and ONNX actor from
different releases can have compatible shapes while representing different
policies, so the command may run but its result is not a valid model
evaluation.

## Old model: two-run smoke test

The Stage 2 CLI currently defaults to the new release. Therefore, always pass
both old-model paths explicitly:

```bash
python scripts/bfm_zero_fallen_recovery_eval.py stage2 \
  --run-id smoke-oldmodel-induced-2run-YYYYMMDD \
  --seed 0 \
  --model-checkpoint thirdparties/BFM-Zero-deploy/model/checkpoint \
  --actor-onnx thirdparties/BFM-Zero-deploy/model/exported/FBcprAuxModel.onnx \
  --induce-fall-in-simulator \
  --num-runs 2 \
  --horizon-s 8 \
  --disturbance-method-override velocity_delta
```

The expected old-model artifact hashes are:

```text
checkpoint/model/model.safetensors  d8d07d7b83ad030286f44d55a48899dc42db4b48000132fffad063653a6eb49e
exported/FBcprAuxModel.onnx          209097902c45621eebab2edb81070c31895fcdd558f0cf6f7f5a360fd747ab74
```

The result is written to
`logs/bfm-zero-fallen-recovery/<run-id>/stage2/summary.json`. For the two-run
smoke gate, require all of the following:

- `accepted_num_runs == 2`
- `perturbation_success_count == 2`, proving both trials actually fell below
  the configured `fallen_z_threshold_m`
- `success_count >= 1` and `success_rate >= 0.5`

A recovery succeeds when the mean pelvis height over the final 0.5 seconds is
above `recovery_z_threshold_m` (default `0.75 m`). The established 100-run old
baseline is `98/100`; a two-run smoke is only a quick health check, not a new
rate estimate.

## Switching to the new model

Switch both model paths together:

```bash
python scripts/bfm_zero_fallen_recovery_eval.py stage2 \
  --run-id smoke-newmodel-induced-2run-YYYYMMDD \
  --seed 0 \
  --model-checkpoint "$HOME/BFM-Zero-Data/new_model_for_training_code_inference/checkpoint" \
  --actor-onnx "$HOME/BFM-Zero-Data/new_model_for_training_code_inference/exported/FBcprAuxModel.onnx" \
  --induce-fall-in-simulator \
  --num-runs 2 \
  --horizon-s 8 \
  --disturbance-method-override velocity_delta
```

This is the same evaluation workflow, but it is not a single-parameter model
swap. As recorded in
`artifacts/bfm-zero-new-model-diagnosis/report.json`, the published new ONNX
does not match its checkpoint, and the parity-correct re-export also failed
the existing closed-loop recovery smoke. Treat a new-model result as a model
validation result, not as evidence that the evaluator is misconfigured.
