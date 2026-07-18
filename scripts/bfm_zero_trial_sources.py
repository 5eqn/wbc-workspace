"""Validated adapters for Stage 2 and provider-aware fast-evaluator trials."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRIAL_LIMIT = 100
FAST_SCHEMA_VERSION = 2
FAST_AMP_SCHEMA_VERSION = 3
FAST_STATE_FRAMES = 401
FAST_TRANSITIONS = 400
FAST_OBSERVATION_SHAPES = {
    "state": (FAST_STATE_FRAMES, 64),
    "privileged_state": (FAST_STATE_FRAMES, 463),
    "last_action": (FAST_STATE_FRAMES, 29),
    "history_actor": (FAST_STATE_FRAMES, 372),
}


@dataclass(frozen=True)
class Trial:
    source_id: str
    attempt_id: int
    run_id: str
    success: bool
    time_s: Any
    qpos: Any
    qvel: Any
    observations: dict[str, Any]
    actions: Any | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrialSource:
    source_id: str
    source_kind: str
    source_path: Path
    goal_key: str
    goal_z: Any | None
    model: dict[str, Any]
    seed: int
    fallen_z_threshold_m: float
    recovery_z_threshold_m: float
    horizon_s: float
    trials: tuple[Trial, ...]
    metadata: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path, repo_root: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(name: str, value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def load_stage2_source(
    stage2_dir: Path, repo_root: Path, limit: int = TRIAL_LIMIT
) -> TrialSource:
    import numpy as np

    stage2_dir = stage2_dir.resolve()
    summary_path = stage2_dir / "summary.json"
    summary = _load_json(summary_path)
    model = dict(summary.get("model") or {})
    for key in ("checkpoint_path", "goal_context_path"):
        if key not in model or not _resolve(model[key], repo_root).exists():
            raise FileNotFoundError(f"Stage 2 model artifact is missing: {key}")

    usable: list[Trial] = []
    for source_index, summary_ref in enumerate(summary.get("run_summaries", [])):
        run_summary_path = _resolve(summary_ref, repo_root)
        run_summary = _load_json(run_summary_path)
        trajectory_path = _resolve(run_summary["trajectory_path"], repo_root)
        telemetry_path = _resolve(run_summary["telemetry_path"], repo_root)
        if not telemetry_path.is_file():
            raise FileNotFoundError(telemetry_path)
        with np.load(trajectory_path) as trajectory:
            required = {"time_s", "qpos", "qvel", "root_z"}
            if not required.issubset(trajectory.files):
                raise ValueError(
                    f"{trajectory_path}: missing {sorted(required - set(trajectory.files))}"
                )
            time_s = _finite("time_s", trajectory["time_s"]).astype(
                np.float32, copy=True
            )
            sim_time_s = _finite(
                "sim_time_s",
                trajectory["sim_time_s"]
                if "sim_time_s" in trajectory.files
                else time_s,
            ).astype(np.float64, copy=True)
            qpos = _finite("qpos", trajectory["qpos"]).astype(np.float32, copy=True)
            qvel = _finite("qvel", trajectory["qvel"]).astype(np.float32, copy=True)
        if (
            time_s.ndim != 1
            or sim_time_s.shape != time_s.shape
            or qpos.shape != (len(time_s), 36)
            or qvel.shape != (len(time_s), 35)
        ):
            raise ValueError(
                f"{trajectory_path}: malformed Stage 2 shapes time={time_s.shape}, "
                f"qpos={qpos.shape}, qvel={qvel.shape}"
            )
        usable.append(
            Trial(
                source_id=str(summary["run_id"]),
                attempt_id=source_index,
                run_id=run_summary_path.parent.name,
                success=bool(run_summary["success"]),
                time_s=time_s,
                qpos=qpos,
                qvel=qvel,
                observations={},
                actions=None,
                metadata={
                    "run_summary": run_summary,
                    "run_summary_path": str(run_summary_path),
                    "trajectory_path": str(trajectory_path),
                    "telemetry_path": str(telemetry_path),
                    "sim_time_s": sim_time_s,
                },
            )
        )
        if len(usable) == limit:
            break
    if len(usable) < limit:
        raise ValueError(
            f"{summary_path}: requires at least {limit} usable trials, found {len(usable)}"
        )
    return TrialSource(
        source_id=str(summary["run_id"]),
        source_kind="stage2",
        source_path=stage2_dir,
        goal_key=str(summary["goal_key"]),
        goal_z=None,
        model=model,
        seed=int(summary.get("seed", 0)),
        fallen_z_threshold_m=float(summary["fallen_z_threshold_m"]),
        recovery_z_threshold_m=float(summary["recovery_z_threshold_m"]),
        horizon_s=float(summary["horizon_s"]),
        trials=tuple(usable),
        metadata={"summary": summary, "summary_path": str(summary_path)},
    )


def _numeric_rounds(run_dir: Path) -> list[tuple[int, Path]]:
    rounds: list[tuple[int, Path]] = []
    for path in run_dir.glob("round_*.h5"):
        match = re.fullmatch(r"round_(\d+)\.h5", path.name)
        if match:
            rounds.append((int(match.group(1)), path))
    return sorted(rounds)


def load_fast_source(
    fast_run_dir: Path, repo_root: Path, limit: int = TRIAL_LIMIT
) -> TrialSource:
    import h5py
    import numpy as np

    fast_run_dir = fast_run_dir.resolve()
    manifest_path = fast_run_dir / "manifest.json"
    summary_path = fast_run_dir / "summary.json"
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    schema_version = int(manifest.get("schema_version", -1))
    provider = dict(manifest.get("model_provider") or {})
    is_amp = (
        schema_version == FAST_AMP_SCHEMA_VERSION
        and provider.get("identity") == "legged-lab-amp"
    )
    if schema_version != FAST_SCHEMA_VERSION and not is_amp:
        raise ValueError(f"{manifest_path}: schema-v2 BFM or schema-v3 AMP is required")
    paths = dict(manifest.get("paths") or {})
    hashes = dict(manifest.get("hashes") or {})
    actor = _resolve(paths.get("actor", ""), repo_root)
    artifacts = [("actor", actor)]
    checkpoint = goals = torchscript = env_config = None
    if is_amp:
        torchscript = _resolve(paths.get("torchscript", ""), repo_root)
        env_config = _resolve(paths.get("environment_config", ""), repo_root)
        artifacts += [("torchscript", torchscript), ("environment_config", env_config)]
    else:
        checkpoint = _resolve(paths.get("checkpoint", ""), repo_root)
        goals = _resolve(paths.get("goals", ""), repo_root)
        artifacts += [("checkpoint", checkpoint), ("goals", goals)]
    for name, path in artifacts:
        if not path.exists():
            raise FileNotFoundError(f"Fast-run {name} artifact is missing: {path}")
    if hashes.get("actor_sha256") != _sha256(actor):
        raise ValueError("fast-run actor hash does not match the manifest")
    if is_amp:
        if hashes.get("torchscript_sha256") != _sha256(torchscript):
            raise ValueError("fast-run TorchScript hash does not match the manifest")
        if hashes.get("environment_config_sha256") != _sha256(env_config):
            raise ValueError(
                "fast-run environment config hash does not match the manifest"
            )

    goal_z = None
    if not is_amp:
        goal_z = _finite("goal_z", manifest.get("goal_z", [])).astype(np.float32)
        if goal_z.shape != (256,) or abs(float(np.linalg.norm(goal_z)) - 16.0) > 1e-4:
            raise ValueError(
                f"invalid fast-run goal latent: shape={goal_z.shape}, norm={np.linalg.norm(goal_z)}"
            )

    trials: list[Trial] = []
    selected_rows: list[dict[str, int]] = []
    rounds = _numeric_rounds(fast_run_dir)
    if not rounds:
        raise ValueError(f"{fast_run_dir}: no numeric round_*.h5 files")
    for round_index, round_path in rounds:
        with h5py.File(round_path, "r") as handle:
            if int(handle.attrs.get("schema_version", -1)) != schema_version:
                raise ValueError(f"{round_path}: schema mismatch")
            if not bool(handle.attrs.get("complete", False)):
                raise ValueError(f"{round_path}: incomplete round")
            if int(handle.attrs.get("round_index", -1)) != round_index:
                raise ValueError(f"{round_path}: round index mismatch")
            accepted = np.asarray(handle["attempts/accepted"][:], dtype=np.bool_)
            recovered = np.asarray(handle["attempts/recovered"][:], dtype=np.bool_)
            metadata = [
                json.loads(item) for item in handle["attempts/metadata_json"].asstr()[:]
            ]
            if accepted.shape != recovered.shape or len(metadata) != len(accepted):
                raise ValueError(f"{round_path}: malformed attempt metadata")
            accepted_attempts = np.flatnonzero(accepted)
            accepted_count = len(accepted_attempts)
            observation_shapes = (
                {"provider_input": (FAST_STATE_FRAMES, 570)}
                if is_amp
                else FAST_OBSERVATION_SHAPES
            )
            expected = {
                **observation_shapes,
                "qpos": (FAST_STATE_FRAMES, 36),
                "qvel": (FAST_STATE_FRAMES, 35),
                "action": (FAST_TRANSITIONS, 29),
            }
            for name, tail in expected.items():
                if name not in handle or handle[name].shape != (accepted_count, *tail):
                    actual = None if name not in handle else handle[name].shape
                    raise ValueError(
                        f"{round_path}: {name} shape {actual}, expected {(accepted_count, *tail)}"
                    )
            for name in ("terminated", "truncated"):
                if name not in handle or handle[name].shape != (
                    accepted_count,
                    FAST_TRANSITIONS,
                ):
                    actual = None if name not in handle else handle[name].shape
                    raise ValueError(
                        f"{round_path}: {name} shape {actual}, "
                        f"expected {(accepted_count, FAST_TRANSITIONS)}"
                    )
            for stored_row, attempt_row in enumerate(accepted_attempts):
                attempt = dict(metadata[int(attempt_row)])
                attempt_id = int(attempt.get("attempt_index", attempt_row))
                observations = {
                    name: _finite(
                        f"{round_path}:{name}[{stored_row}]", handle[name][stored_row]
                    ).astype(np.float32, copy=False)
                    for name in observation_shapes
                }
                qpos = _finite(
                    f"{round_path}:qpos[{stored_row}]", handle["qpos"][stored_row]
                ).astype(np.float32, copy=False)
                qvel = _finite(
                    f"{round_path}:qvel[{stored_row}]", handle["qvel"][stored_row]
                ).astype(np.float32, copy=False)
                action = _finite(
                    f"{round_path}:action[{stored_row}]", handle["action"][stored_row]
                ).astype(np.float32, copy=False)
                trials.append(
                    Trial(
                        source_id=str(manifest["run_id"]),
                        attempt_id=attempt_id,
                        run_id=f"attempt_{attempt_id:03d}",
                        success=bool(recovered[int(attempt_row)]),
                        time_s=np.arange(FAST_STATE_FRAMES, dtype=np.float32) / 50.0,
                        qpos=qpos,
                        qvel=qvel,
                        observations=observations,
                        actions=action,
                        metadata={
                            **attempt,
                            "round_index": round_index,
                            "attempt_row": int(attempt_row),
                            "stored_row": stored_row,
                            "round_path": str(round_path),
                        },
                    )
                )
                selected_rows.append(
                    {
                        "attempt_id": attempt_id,
                        "round_index": round_index,
                        "stored_row": stored_row,
                    }
                )
                if len(trials) == limit:
                    break
        if len(trials) == limit:
            break
    if len(trials) < limit:
        raise ValueError(
            f"{fast_run_dir}: requires at least {limit} usable trials, found {len(trials)}"
        )
    return TrialSource(
        source_id=str(manifest["run_id"]),
        source_kind="fast_replay_v3_amp" if is_amp else "fast_replay_v2",
        source_path=fast_run_dir,
        goal_key=str(manifest["config"].get("goal_key", "")) if not is_amp else "",
        goal_z=goal_z,
        model=(
            {
                "provider": "legged-lab-amp",
                "actor_onnx_path": str(actor),
                "actor_onnx_sha256": hashes.get("actor_sha256"),
                "torchscript_path": str(torchscript),
                "torchscript_sha256": hashes.get("torchscript_sha256"),
                "environment_config_path": str(env_config),
                "environment_config_sha256": hashes.get("environment_config_sha256"),
                "latent_inspector": "excluded: provider has no B/F/D/QD/Q checkpoint interfaces",
            }
            if is_amp
            else {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": hashes.get("checkpoint_sha256"),
                "actor_onnx_path": str(actor),
                "actor_onnx_sha256": hashes.get("actor_sha256"),
                "goal_context_path": str(goals),
            }
        ),
        seed=int(manifest["config"]["seed"]),
        fallen_z_threshold_m=float(manifest["config"]["fallen_height_m"]),
        recovery_z_threshold_m=float(manifest["config"]["recovery_height_m"]),
        horizon_s=float(manifest["config"]["horizon_s"]),
        trials=tuple(trials),
        metadata={
            "manifest": manifest,
            "summary": summary,
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "selected_rows": selected_rows,
        },
    )


def aligned_fast_actions(trial: Trial) -> Any:
    """Align 400 transition actions to 401 states by holding action 399."""
    import numpy as np

    if trial.actions is None or trial.actions.shape != (FAST_TRANSITIONS, 29):
        raise ValueError(f"{trial.run_id}: expected 400x29 fast actions")
    return np.concatenate((trial.actions, trial.actions[-1:]), axis=0)
