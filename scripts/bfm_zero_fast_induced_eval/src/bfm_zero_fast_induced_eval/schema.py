from __future__ import annotations

from collections import OrderedDict

import numpy as np

STATE_DIM = 64
PRIVILEGED_STATE_DIM = 463
LAST_ACTION_DIM = 29
HISTORY_ACTOR_DIM = 372
ACTION_DIM = 29
GOAL_DIM = 256
OBSERVATION_FRAMES = 401
TRANSITIONS = 400
SCHEMA_VERSION = 2
QPOS_DIM = 36
QVEL_DIM = 35

OBSERVATION_FIELDS = OrderedDict(
    state=STATE_DIM,
    privileged_state=PRIVILEGED_STATE_DIM,
    last_action=LAST_ACTION_DIM,
    history_actor=HISTORY_ACTOR_DIM,
)

REPLAY_STATE_FIELDS = OrderedDict(
    qpos=QPOS_DIM,
    qvel=QVEL_DIM,
)

# The environment's HistoryHandler sorts keys, then flattens four frames per key.
HISTORY_COMPONENTS = OrderedDict(
    actions=(0, 116),
    base_ang_vel=(116, 128),
    dof_pos=(128, 244),
    dof_vel=(244, 360),
    projected_gravity=(360, 372),
)


def actor_input(observation: dict[str, np.ndarray], z: np.ndarray) -> np.ndarray:
    arrays = [observation["state"], observation["last_action"], observation["history_actor"]]
    batch = arrays[0].shape[0]
    goal = np.asarray(z, dtype=np.float32).reshape(1, GOAL_DIM)
    goal = np.broadcast_to(goal, (batch, GOAL_DIM))
    result = np.concatenate([*arrays, goal], axis=-1).astype(np.float32, copy=False)
    if result.shape != (batch, 721):
        raise ValueError(f"actor input has shape {result.shape}, expected {(batch, 721)}")
    return result


def build_history_actor(previous_frames: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten four newest-first frames in the exact environment key order."""
    dimensions = {"actions": 29, "base_ang_vel": 3, "dof_pos": 29, "dof_vel": 29, "projected_gravity": 3}
    arrays = []
    batch = None
    for key in sorted(dimensions):
        value = np.asarray(previous_frames[key], dtype=np.float32)
        expected_tail = (4, dimensions[key])
        if value.ndim != 3 or value.shape[1:] != expected_tail:
            raise ValueError(
                f"{key} history has shape {value.shape}, "
                f"expected (batch, {expected_tail[0]}, {expected_tail[1]})"
            )
        batch = value.shape[0] if batch is None else batch
        if value.shape[0] != batch:
            raise ValueError("history components have different batch sizes")
        arrays.append(value.reshape(value.shape[0], -1))
    result = np.concatenate(arrays, axis=-1)
    if result.shape != (batch, HISTORY_ACTOR_DIM):
        raise ValueError(f"history actor has unexpected shape {result.shape}")
    return result


def validate_transition_alignment(
    last_action: np.ndarray,
    action: np.ndarray,
    qpos: np.ndarray | None = None,
    qvel: np.ndarray | None = None,
) -> None:
    """Check the observation-frame and transition-action axes used by replay indexing."""
    last_action = np.asarray(last_action)
    action = np.asarray(action)
    if last_action.shape[-2:] != (OBSERVATION_FRAMES, ACTION_DIM):
        raise ValueError("last_action must end in (401, 29)")
    if action.shape[-2:] != (TRANSITIONS, ACTION_DIM):
        raise ValueError("action must end in (400, 29)")
    for name, value, dimension in (
        ("qpos", qpos, QPOS_DIM),
        ("qvel", qvel, QVEL_DIM),
    ):
        if value is not None and np.asarray(value).shape[-2:] != (OBSERVATION_FRAMES, dimension):
            raise ValueError(f"{name} must end in ({OBSERVATION_FRAMES}, {dimension})")


def transition_at(trajectory: dict[str, np.ndarray], step: int) -> dict[str, object]:
    if not 0 <= step < TRANSITIONS:
        raise IndexError(step)
    return {
        "observation": {name: trajectory[name][step] for name in OBSERVATION_FIELDS},
        "action": trajectory["action"][step],
        "next_observation": {name: trajectory[name][step + 1] for name in OBSERVATION_FIELDS},
        "qpos": trajectory["qpos"][step],
        "qvel": trajectory["qvel"][step],
        "next_qpos": trajectory["qpos"][step + 1],
        "next_qvel": trajectory["qvel"][step + 1],
        "terminated": trajectory["terminated"][step],
        "truncated": trajectory["truncated"][step],
    }


def validate_observation(observation: dict[str, np.ndarray], num_envs: int) -> None:
    for name, dim in OBSERVATION_FIELDS.items():
        value = np.asarray(observation[name])
        if value.shape != (num_envs, dim):
            raise ValueError(f"{name} has shape {value.shape}, expected {(num_envs, dim)}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")


def raw_bytes_per_accepted_trial() -> int:
    observation_floats = OBSERVATION_FRAMES * sum(OBSERVATION_FIELDS.values())
    replay_state_floats = OBSERVATION_FRAMES * sum(REPLAY_STATE_FIELDS.values())
    action_floats = TRANSITIONS * ACTION_DIM
    boundary_bools = TRANSITIONS * 2
    return (observation_floats + replay_state_floats + action_floats) * 4 + boundary_bools


def estimated_raw_gib(num_trials: int) -> float:
    return raw_bytes_per_accepted_trial() * num_trials / 2**30
