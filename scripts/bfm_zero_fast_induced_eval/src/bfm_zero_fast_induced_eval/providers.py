from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import DEFAULT_ACTOR, DEFAULT_CHECKPOINT, DEFAULT_GOALS
from .schema import OBSERVATION_FIELDS, actor_input, validate_observation

AMP_RUN = Path("/home/seqn/legged_lab/logs/rsl_rl/g1_amp_get_up/2026-07-17_10-00-26")
DEFAULT_AMP_ACTOR = AMP_RUN / "exported/policy.onnx"
DEFAULT_AMP_TORCHSCRIPT = AMP_RUN / "exported/policy.pt"
DEFAULT_AMP_ENV_CONFIG = AMP_RUN / "params/env.yaml"

SDK_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
SDK_TO_POLICY = tuple(SDK_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES)
POLICY_TO_SDK = tuple(POLICY_JOINT_NAMES.index(name) for name in SDK_JOINT_NAMES)
KEY_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
)
AMP_DEFAULT_JOINT_POSITION = np.asarray(
    [
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        0,
        0,
        0,
        0.3,
        0.25,
        0,
        0.97,
        0.15,
        0,
        0,
        0.3,
        -0.25,
        0,
        0.97,
        -0.15,
        0,
        0,
    ],
    dtype=np.float32,
)
AMP_TERM_DIMS = {
    "base_ang_vel": 3,
    "root_local_rot_tan_norm": 6,
    "joint_pos": 29,
    "joint_vel": 29,
    "actions": 29,
    "key_body_pos_b": 18,
}


@dataclass(frozen=True)
class CanonicalSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    body_positions_w: np.ndarray
    body_names: tuple[str, ...]


class PolicyProvider(Protocol):
    name: str
    version: int
    input_dim: int
    output_dim: int
    schema_version: int
    observation_fields: dict[str, int]

    def reset(self, observation: dict[str, np.ndarray], snapshot: CanonicalSnapshot) -> None: ...
    def infer(
        self, observation: dict[str, np.ndarray], snapshot: CanonicalSnapshot, session: Any
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def metadata(self) -> dict[str, Any]: ...


def _quat_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def yaw_local_rotation_6d(root_quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = _quat_matrix_wxyz(root_quaternion_wxyz)
    yaw = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    inverse_yaw = np.zeros_like(matrix)
    inverse_yaw[..., 0, 0] = cosine
    inverse_yaw[..., 0, 1] = sine
    inverse_yaw[..., 1, 0] = -sine
    inverse_yaw[..., 1, 1] = cosine
    inverse_yaw[..., 2, 2] = 1
    local = inverse_yaw @ matrix
    return np.concatenate((local[..., :, 0], local[..., :, 2]), axis=-1).astype(np.float32)


def key_body_positions(snapshot: CanonicalSnapshot) -> np.ndarray:
    try:
        indices = [snapshot.body_names.index(name) for name in KEY_BODY_NAMES]
    except ValueError as error:
        raise ValueError(f"runtime body order is missing an AMP key body: {error}") from error
    relative = snapshot.body_positions_w[:, indices] - snapshot.qpos[:, None, :3]
    rotation = _quat_matrix_wxyz(snapshot.qpos[:, 3:7])
    local = np.einsum("bij,bkj->bki", np.swapaxes(rotation, -1, -2), relative)
    return local.reshape(len(relative), -1).astype(np.float32)


class BfmZeroProvider:
    name = "bfm-zero"
    version = 1
    input_dim = 721
    output_dim = 29
    schema_version = 2
    observation_fields = dict(OBSERVATION_FIELDS)

    def __init__(self, goal: np.ndarray):
        self.goal = np.asarray(goal, dtype=np.float32)

    def reset(self, observation: dict[str, np.ndarray], snapshot: CanonicalSnapshot) -> None:
        validate_observation(observation, len(snapshot.qpos))

    def infer(
        self, observation: dict[str, np.ndarray], snapshot: CanonicalSnapshot, session: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        validate_observation(observation, len(snapshot.qpos))
        model_input = actor_input(observation, self.goal)
        action = session.run(None, {session.get_inputs()[0].name: model_input})[0].astype(np.float32)
        return model_input, action

    def replay_frame(
        self, observation: dict[str, np.ndarray], model_input: np.ndarray
    ) -> dict[str, np.ndarray]:
        return {name: np.asarray(observation[name], dtype=np.float32) for name in OBSERVATION_FIELDS}

    def metadata(self) -> dict[str, Any]:
        return {
            "identity": self.name,
            "version": self.version,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_layout": ["state", "last_action", "history_actor", "goal_z"],
            "history_contract": "environment-managed four-frame newest-first history",
            "control_profile": "bfm-zero-default",
        }


class LeggedLabAmpProvider:
    name = "legged-lab-amp"
    version = 2
    input_dim = 570
    output_dim = 29
    schema_version = 3
    observation_fields = {"provider_input": 570}

    def __init__(self) -> None:
        self._history: dict[str, np.ndarray] | None = None
        self._previous_action: np.ndarray | None = None

    def _terms(self, snapshot: CanonicalSnapshot) -> dict[str, np.ndarray]:
        batch = len(snapshot.qpos)
        previous = self._previous_action
        if previous is None:
            previous = np.zeros((batch, 29), dtype=np.float32)
        return {
            "base_ang_vel": np.asarray(snapshot.qvel[:, 3:6], dtype=np.float32),
            "root_local_rot_tan_norm": yaw_local_rotation_6d(snapshot.qpos[:, 3:7]),
            "joint_pos": np.asarray(snapshot.qpos[:, 7:][:, SDK_TO_POLICY], dtype=np.float32),
            "joint_vel": np.asarray(snapshot.qvel[:, 6:][:, SDK_TO_POLICY], dtype=np.float32),
            "actions": previous,
            "key_body_pos_b": key_body_positions(snapshot),
        }

    def reset(self, observation: dict[str, np.ndarray], snapshot: CanonicalSnapshot) -> None:
        self._previous_action = np.zeros((len(snapshot.qpos), 29), dtype=np.float32)
        terms = self._terms(snapshot)
        self._history = {name: np.repeat(value[:, None, :], 5, axis=1) for name, value in terms.items()}

    def build_input(self, snapshot: CanonicalSnapshot, *, advance: bool = True) -> np.ndarray:
        terms = self._terms(snapshot)
        if self._history is None:
            self.reset({}, snapshot)
        elif advance:
            self._history = {
                name: np.concatenate((self._history[name][:, 1:], value[:, None, :]), axis=1)
                for name, value in terms.items()
            }
        assert self._history is not None
        result = np.concatenate(
            [self._history[name].reshape(len(snapshot.qpos), -1) for name in AMP_TERM_DIMS], axis=-1
        )
        if result.shape != (len(snapshot.qpos), self.input_dim) or not np.isfinite(result).all():
            raise ValueError(f"invalid AMP policy input {result.shape}")
        return result.astype(np.float32, copy=False)

    def infer(
        self, observation: dict[str, np.ndarray], snapshot: CanonicalSnapshot, session: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        model_input = self.build_input(snapshot)
        policy_action = session.run(None, {session.get_inputs()[0].name: model_input})[0].astype(np.float32)
        if policy_action.shape != (len(snapshot.qpos), 29) or not np.isfinite(policy_action).all():
            raise ValueError(f"invalid AMP action {policy_action.shape}")
        self._previous_action = policy_action.copy()
        return model_input, policy_action[:, POLICY_TO_SDK]

    def replay_frame(
        self, observation: dict[str, np.ndarray], model_input: np.ndarray
    ) -> dict[str, np.ndarray]:
        return {"provider_input": np.asarray(model_input, dtype=np.float32)}

    def metadata(self) -> dict[str, Any]:
        return {
            "identity": self.name,
            "version": self.version,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_layout": [
                {"term": name, "history": 5, "dimension": dim} for name, dim in AMP_TERM_DIMS.items()
            ],
            "history_contract": (
                "term-major, chronological oldest-to-newest; reset repeats first frame; "
                "previous action starts at zero"
            ),
            "canonical_joint_order": list(SDK_JOINT_NAMES),
            "policy_joint_order": list(POLICY_JOINT_NAMES),
            "key_body_order": list(KEY_BODY_NAMES),
            "default_joint_position": AMP_DEFAULT_JOINT_POSITION.tolist(),
            "control_profile": "default_joint_position + 0.25 * raw_action",
        }


def resolve_provider_artifacts(
    name: str, actor: Path | None, checkpoint: Path | None, goals: Path | None
) -> tuple[Path, Path | None, Path | None]:
    if name == "bfm-zero":
        return actor or DEFAULT_ACTOR, checkpoint or DEFAULT_CHECKPOINT, goals or DEFAULT_GOALS
    if name == "legged-lab-amp":
        return actor or DEFAULT_AMP_ACTOR, checkpoint, goals
    raise ValueError(f"unknown model provider {name!r}")


def action_to_target(raw_action: np.ndarray) -> np.ndarray:
    return AMP_DEFAULT_JOINT_POSITION + 0.25 * np.asarray(raw_action, dtype=np.float32)
