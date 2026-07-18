from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfm_zero_trial_sources import aligned_fast_actions, load_fast_source, load_stage2_source


def _fast_run(tmp_path: Path, attempts: int = 101, schema: int = 2) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    run = tmp_path / "fast"
    run.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    actor = tmp_path / "actor.onnx"
    actor.write_bytes(b"actor")
    goals = tmp_path / "goals.pkl"
    goals.write_bytes(b"goals")
    goal_z = np.zeros(256, dtype=np.float32)
    goal_z[0] = 16.0
    manifest = {
        "schema_version": schema,
        "run_id": "synthetic-fast",
        "config": {
            "goal_key": "goal",
            "seed": 7,
            "fallen_height_m": 0.45,
            "recovery_height_m": 0.75,
            "horizon_s": 8.0,
        },
        "paths": {"checkpoint": str(checkpoint), "actor": str(actor), "goals": str(goals)},
        "hashes": {
            "actor_sha256": hashlib.sha256(actor.read_bytes()).hexdigest(),
            "checkpoint_sha256": "checkpoint-hash",
        },
        "goal_z": goal_z.tolist(),
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "summary.json").write_text(json.dumps({"complete": True}))
    with h5py.File(run / "round_010.h5", "w") as handle:
        handle.attrs.update(
            schema_version=schema,
            round_index=10,
            complete=True,
            accepted_count=attempts,
            attempted_count=attempts,
        )
        for name, shape in {
            "state": (attempts, 401, 64),
            "privileged_state": (attempts, 401, 463),
            "last_action": (attempts, 401, 29),
            "history_actor": (attempts, 401, 372),
            "qpos": (attempts, 401, 36),
            "qvel": (attempts, 401, 35),
            "action": (attempts, 400, 29),
            "terminated": (attempts, 400),
            "truncated": (attempts, 400),
        }.items():
            dtype = np.bool_ if name in ("terminated", "truncated") else np.float32
            handle.create_dataset(name, shape=shape, dtype=dtype, fillvalue=0)
        group = handle.create_group("attempts")
        group.create_dataset("accepted", data=np.ones(attempts, dtype=np.bool_))
        group.create_dataset("recovered", data=np.arange(attempts) != 99)
        strings = h5py.string_dtype("utf-8")
        group.create_dataset(
            "metadata_json",
            data=[json.dumps({"attempt_index": index}) for index in range(attempts)],
            dtype=strings,
        )
    return run


def test_fast_source_selects_exact_first_100_and_aligns_actions(tmp_path) -> None:
    source = load_fast_source(_fast_run(tmp_path), tmp_path)
    assert source.source_kind == "fast_replay_v2"
    assert [trial.attempt_id for trial in source.trials] == list(range(100))
    assert len(source.trials) == 100
    assert source.trials[99].success is False
    aligned = aligned_fast_actions(source.trials[0])
    assert aligned.shape == (401, 29)
    np.testing.assert_array_equal(aligned[-1], aligned[-2])


def test_fast_source_rejects_fewer_than_100_and_schema_v1(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires at least 100"):
        load_fast_source(_fast_run(tmp_path / "short", attempts=99), tmp_path)
    legacy = _fast_run(tmp_path / "legacy", schema=1)
    with pytest.raises(ValueError, match="schema-v2"):
        load_fast_source(legacy, tmp_path)


@pytest.mark.parametrize("field", ["qpos", "state"])
def test_fast_source_rejects_shapes_and_nonfinite(tmp_path, field) -> None:
    run = _fast_run(tmp_path)
    with h5py.File(run / "round_010.h5", "r+") as handle:
        if field == "qpos":
            del handle[field]
            handle.create_dataset(field, shape=(101, 400, 36), dtype=np.float32)
        else:
            handle[field][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="shape|non-finite"):
        load_fast_source(run, tmp_path)


def test_stage2_adapter_preserves_order_and_first_100(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    goals = tmp_path / "goals.pkl"
    goals.write_bytes(b"goals")
    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    refs = []
    for index in range(101):
        run = tmp_path / f"run_{index:03d}"
        run.mkdir()
        trajectory = run / "trajectory.npz"
        np.savez(
            trajectory,
            time_s=np.array([0.0], dtype=np.float32),
            qpos=np.full((1, 36), index, dtype=np.float32),
            qvel=np.zeros((1, 35), dtype=np.float32),
            root_z=np.array([0.4], dtype=np.float32),
        )
        telemetry = run / "telemetry.npz"
        np.savez(telemetry, state=np.zeros((1, 64), dtype=np.float32))
        run_summary = run / "summary.json"
        run_summary.write_text(
            json.dumps(
                {
                    "trajectory_path": str(trajectory),
                    "telemetry_path": str(telemetry),
                    "success": index % 2 == 0,
                }
            )
        )
        refs.append(str(run_summary))
    (stage2 / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "stage2-source",
                "run_summaries": refs,
                "goal_key": "goal",
                "seed": 3,
                "fallen_z_threshold_m": 0.45,
                "recovery_z_threshold_m": 0.75,
                "horizon_s": 4.0,
                "model": {"checkpoint_path": str(checkpoint), "goal_context_path": str(goals)},
            }
        )
    )
    source = load_stage2_source(stage2, tmp_path)
    assert [trial.attempt_id for trial in source.trials] == list(range(100))
    assert source.trials[-1].qpos[0, 0] == 99


def test_amp_schema_v3_source_needs_no_bfm_goal_or_checkpoint(tmp_path) -> None:
    run = tmp_path / "amp"
    run.mkdir()
    actor = tmp_path / "policy.onnx"
    torchscript = tmp_path / "policy.pt"
    env_config = tmp_path / "env.yaml"
    for path, payload in ((actor, b"onnx"), (torchscript, b"torch"), (env_config, b"env")):
        path.write_bytes(payload)
    hashes = {
        "actor_sha256": hashlib.sha256(actor.read_bytes()).hexdigest(),
        "torchscript_sha256": hashlib.sha256(torchscript.read_bytes()).hexdigest(),
        "environment_config_sha256": hashlib.sha256(env_config.read_bytes()).hexdigest(),
    }
    manifest = {
        "schema_version": 3,
        "run_id": "amp-synthetic",
        "model_provider": {"identity": "legged-lab-amp", "version": 1},
        "config": {
            "seed": 0,
            "fallen_height_m": 0.45,
            "recovery_height_m": 0.75,
            "horizon_s": 8.0,
        },
        "paths": {
            "actor": str(actor),
            "torchscript": str(torchscript),
            "environment_config": str(env_config),
        },
        "hashes": hashes,
        "goal_z": None,
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "summary.json").write_text(json.dumps({"complete": True}))
    attempts = 101
    with h5py.File(run / "round_000.h5", "w") as handle:
        handle.attrs.update(
            schema_version=3,
            round_index=0,
            complete=True,
            accepted_count=attempts,
            attempted_count=attempts,
        )
        shapes = {
            "provider_input": (attempts, 401, 570),
            "qpos": (attempts, 401, 36),
            "qvel": (attempts, 401, 35),
            "action": (attempts, 400, 29),
            "terminated": (attempts, 400),
            "truncated": (attempts, 400),
        }
        for name, shape in shapes.items():
            dtype = np.bool_ if name in ("terminated", "truncated") else np.float32
            handle.create_dataset(name, shape=shape, dtype=dtype, fillvalue=0)
        group = handle.create_group("attempts")
        group.create_dataset("accepted", data=np.ones(attempts, dtype=np.bool_))
        group.create_dataset("recovered", data=np.ones(attempts, dtype=np.bool_))
        strings = h5py.string_dtype("utf-8")
        group.create_dataset(
            "metadata_json",
            data=[json.dumps({"attempt_index": index}) for index in range(attempts)],
            dtype=strings,
        )
    source = load_fast_source(run, tmp_path)
    assert source.source_kind == "fast_replay_v3_amp"
    assert source.goal_z is None and source.goal_key == ""
    assert [trial.attempt_id for trial in source.trials] == list(range(100))
    assert source.trials[0].observations["provider_input"].shape == (401, 570)
    assert "excluded" in source.model["latent_inspector"]
