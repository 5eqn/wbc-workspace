# BFM-Zero fast induced-fall evaluation

This isolated Python 3.10 project runs a matched BFM-Zero checkpoint and ONNX
actor in the repository's headless Isaac Sim environment. It evaluates 128
velocity-kick trials concurrently and stores only trials whose pelvis falls
below 0.45 m during the first second. Replay schema v2 stores 401 policy-cadence
observation, `qpos` (36-D), and `qvel` (35-D) frames aligned with 400 actions;
pelvis-height metrics remain sampled at the 200 Hz physics cadence.

The Isaac runtime is supplied by `thirdparties/BFM-Zero/.venv`; the uv project
owns the evaluator, lock file, host tests, and command line. Run from the
repository root:

```bash
uv sync --project scripts/bfm_zero_fast_induced_eval --all-groups
uv run --project scripts/bfm_zero_fast_induced_eval pytest scripts/bfm_zero_fast_induced_eval/tests
uv run --project scripts/bfm_zero_fast_induced_eval bfm-zero-fast-induced-eval preflight
uv run --project scripts/bfm_zero_fast_induced_eval bfm-zero-fast-induced-eval patch-onnx
uv run --project scripts/bfm_zero_fast_induced_eval bfm-zero-fast-induced-eval run --rounds 1
```

`run` automatically re-executes in the existing Isaac Python environment while
loading this project's source. Do not run a full 100-round collection as an
implementation smoke test. Outputs are written to
`logs/bfm-zero-fast-induced/<run-id>/` and completed round files are resumed.
