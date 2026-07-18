from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from bfm_zero_fast_induced_eval.metrics import TrialMetrics
from bfm_zero_fast_induced_eval.schema import OBSERVATION_FIELDS, REPLAY_STATE_FIELDS
from bfm_zero_fast_induced_eval.storage import read_attempt_summaries, valid_round, write_round_atomic


def _round_data(attempts: int = 3) -> dict[str, np.ndarray]:
    data = {
        name: np.broadcast_to(
            np.arange(attempts, dtype=np.float32)[:, None, None], (attempts, 401, dim)
        ).copy()
        for name, dim in OBSERVATION_FIELDS.items()
    }
    data.update(
        {
            name: np.broadcast_to(
                np.arange(attempts, dtype=np.float32)[:, None, None], (attempts, 401, dim)
            ).copy()
            for name, dim in REPLAY_STATE_FIELDS.items()
        }
    )
    data.update(
        action=np.zeros((attempts, 400, 29), dtype=np.float32),
        terminated=np.zeros((attempts, 400), dtype=np.bool_),
        truncated=np.zeros((attempts, 400), dtype=np.bool_),
    )
    return data


def test_accepted_filtering_and_attempt_audit(tmp_path) -> None:
    path = tmp_path / "round_000.h5"
    mask = np.array([True, False, True])
    attempts = [{"seed": index} for index in range(3)]
    metrics = [TrialMetrics(bool(mask[index]), index == 2, 0.4 + index, 0.7 + index) for index in range(3)]
    write_round_atomic(path, 0, _round_data(), mask, attempts, metrics)
    assert valid_round(path, 0)
    with h5py.File(path, "r") as handle:
        assert handle["state"].shape == (2, 401, 64)
        assert handle["action"].shape == (2, 400, 29)
        assert handle["qpos"].shape == (2, 401, 36)
        assert handle["qvel"].shape == (2, 401, 35)
        assert handle["state"][0, 0, 0] == 0
        assert handle["state"][1, 0, 0] == 2
        assert handle["qpos"][1, 0, 0] == 2
        assert handle["attempts/metadata_json"].shape == (3,)
    rows = read_attempt_summaries(path)
    assert [row["accepted"] for row in rows] == [True, False, True]
    assert [row["recovered"] for row in rows] == [False, False, True]


def test_resume_rejects_incomplete_and_ignores_temporary(tmp_path) -> None:
    path = tmp_path / "round_004.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["complete"] = False
    assert not valid_round(path, 4)
    temporary = tmp_path / "round_004.h5.tmp"
    temporary.write_text(json.dumps({"partial": True}))
    path.unlink()
    mask = np.array([True, False, True])
    attempts = [{"seed": index} for index in range(3)]
    metrics = [TrialMetrics(bool(flag), False, 0.4, 0.8) for flag in mask]
    write_round_atomic(path, 4, _round_data(), mask, attempts, metrics)
    assert valid_round(path, 4)
    assert not temporary.exists()


def test_resume_rejects_schema_v1(tmp_path) -> None:
    path = tmp_path / "round_000.h5"
    mask = np.array([True, False, True])
    attempts = [{"seed": index} for index in range(3)]
    metrics = [TrialMetrics(bool(flag), False, 0.4, 0.8) for flag in mask]
    write_round_atomic(path, 0, _round_data(), mask, attempts, metrics)
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema_version"] = 1
    assert not valid_round(path, 0)


@pytest.mark.parametrize("field", ["qpos", "qvel"])
def test_rejects_nonfinite_replay_state(tmp_path, field) -> None:
    data = _round_data()
    data[field][0, 0, 0] = np.nan
    mask = np.array([True, False, True])
    attempts = [{"seed": index} for index in range(3)]
    metrics = [TrialMetrics(bool(flag), False, 0.4, 0.8) for flag in mask]
    with pytest.raises(ValueError, match=f"{field} contains non-finite values"):
        write_round_atomic(tmp_path / "round_000.h5", 0, data, mask, attempts, metrics)


def test_schema_v3_provider_input_storage(tmp_path) -> None:
    attempts = 2
    data = {
        "provider_input": np.zeros((attempts, 401, 570), dtype=np.float32),
        "qpos": np.zeros((attempts, 401, 36), dtype=np.float32),
        "qvel": np.zeros((attempts, 401, 35), dtype=np.float32),
        "action": np.zeros((attempts, 400, 29), dtype=np.float32),
        "terminated": np.zeros((attempts, 400), dtype=np.bool_),
        "truncated": np.zeros((attempts, 400), dtype=np.bool_),
    }
    mask = np.ones(attempts, dtype=np.bool_)
    metadata = [{"seed": index} for index in range(attempts)]
    metrics = [TrialMetrics(True, True, 0.4, 0.8) for _ in range(attempts)]
    path = tmp_path / "round_000.h5"
    write_round_atomic(
        path,
        0,
        data,
        mask,
        metadata,
        metrics,
        schema_version=3,
        observation_fields={"provider_input": 570},
    )
    assert valid_round(path, 0, schema_version=3, observation_fields={"provider_input": 570})
    assert not valid_round(path, 0)
