from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .metrics import TrialMetrics
from .schema import (
    ACTION_DIM,
    OBSERVATION_FIELDS,
    OBSERVATION_FRAMES,
    REPLAY_STATE_FIELDS,
    SCHEMA_VERSION,
    TRANSITIONS,
)


def _expected_shapes(accepted: int) -> dict[str, tuple[int, ...]]:
    result = {name: (accepted, OBSERVATION_FRAMES, dim) for name, dim in OBSERVATION_FIELDS.items()}
    result.update(
        {name: (accepted, OBSERVATION_FRAMES, dim) for name, dim in REPLAY_STATE_FIELDS.items()}
    )
    result.update(
        action=(accepted, TRANSITIONS, ACTION_DIM),
        terminated=(accepted, TRANSITIONS),
        truncated=(accepted, TRANSITIONS),
    )
    return result


def valid_round(path: Path, expected_round: int | None = None) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                return False
            if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                return False
            if expected_round is not None and int(handle.attrs["round_index"]) != expected_round:
                return False
            accepted = int(handle.attrs["accepted_count"])
            for name, shape in _expected_shapes(accepted).items():
                if name not in handle or handle[name].shape != shape:
                    return False
            attempted_shape = (int(handle.attrs["attempted_count"]),)
            return "attempts" in handle and handle["attempts/accepted"].shape == attempted_shape
    except (OSError, KeyError, ValueError):
        return False


def write_round_atomic(
    path: Path,
    round_index: int,
    trajectories: dict[str, np.ndarray],
    accepted_mask: np.ndarray,
    attempts: list[dict[str, Any]],
    metrics: list[TrialMetrics],
    rollout_wall_time_s: float = 0.0,
) -> None:
    accepted_mask = np.asarray(accepted_mask, dtype=np.bool_)
    accepted_count = int(accepted_mask.sum())
    if len(attempts) != accepted_mask.size or len(metrics) != accepted_mask.size:
        raise ValueError("attempt metadata, metrics, and mask lengths differ")
    expected = _expected_shapes(accepted_mask.size)
    for name, full_shape in expected.items():
        value = np.asarray(trajectories[name])
        if value.shape != full_shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {full_shape}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            schema_version=SCHEMA_VERSION,
            round_index=round_index,
            attempted_count=accepted_mask.size,
            accepted_count=accepted_count,
            complete=False,
            rollout_wall_time_s=float(rollout_wall_time_s),
        )
        for name, value in trajectories.items():
            filtered = np.asarray(value)[accepted_mask]
            handle.create_dataset(name, data=filtered, compression="gzip", compression_opts=4, shuffle=True)
        group = handle.create_group("attempts")
        string_dtype = h5py.string_dtype("utf-8")
        metadata = [json.dumps(row, sort_keys=True) for row in attempts]
        group.create_dataset("metadata_json", data=metadata, dtype=string_dtype)
        group.create_dataset("accepted", data=accepted_mask)
        group.create_dataset("recovered", data=np.asarray([row.recovered for row in metrics], dtype=np.bool_))
        group.create_dataset("minimum_first_second_m", data=[row.minimum_first_second_m for row in metrics])
        group.create_dataset(
            "final_half_second_mean_m", data=[row.final_half_second_mean_m for row in metrics]
        )
        handle.attrs["complete"] = True
        handle.flush()
    temporary.replace(path)


def read_attempt_summaries(path: Path) -> list[dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        metadata = [json.loads(item) for item in handle["attempts/metadata_json"].asstr()[:]]
        accepted = handle["attempts/accepted"][:]
        recovered = handle["attempts/recovered"][:]
        minima = handle["attempts/minimum_first_second_m"][:]
        final_means = handle["attempts/final_half_second_mean_m"][:]
    for index, row in enumerate(metadata):
        row["accepted"] = bool(accepted[index])
        row["recovered"] = bool(recovered[index])
        row["minimum_first_second_m"] = float(minima[index])
        row["final_half_second_mean_m"] = float(final_means[index])
    return metadata
