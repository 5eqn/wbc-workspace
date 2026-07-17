from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class EvalConfig:
    num_envs: int = 128
    rounds: int = 100
    horizon_s: float = 8.0
    policy_hz: int = 50
    physics_hz: int = 200
    warmup_s: float = 0.5
    seed: int = 0
    fallen_height_m: float = 0.45
    fall_window_s: float = 1.0
    recovery_height_m: float = 0.75
    recovery_window_s: float = 0.5
    linear_velocity_min_mps: float = 3.0
    linear_velocity_max_mps: float = 6.0
    angular_velocity_min_rps: float = 4.0
    angular_velocity_max_rps: float = 8.0
    linear_up_cos_min: float = -0.8
    linear_up_cos_max: float = 0.0
    goal_key: str = "dance1_subject3_505"
    minimum_free_gib: float = 22.0

    @property
    def policy_steps(self) -> int:
        return round(self.horizon_s * self.policy_hz)

    @property
    def observation_frames(self) -> int:
        return self.policy_steps + 1

    @property
    def physics_steps(self) -> int:
        return round(self.horizon_s * self.physics_hz)

    def validate(self) -> None:
        if self.num_envs < 1 or self.rounds < 1:
            raise ValueError("num_envs and rounds must be positive")
        if self.physics_hz % self.policy_hz:
            raise ValueError("physics_hz must be an integer multiple of policy_hz")
        if self.observation_frames != 401 or self.policy_steps != 400:
            raise ValueError("the replay contract requires exactly 401 observations and 400 actions")
        if self.fall_window_s > self.horizon_s or self.recovery_window_s > self.horizon_s:
            raise ValueError("metric windows must fit inside the rollout horizon")
        if not -1 <= self.linear_up_cos_min <= self.linear_up_cos_max <= 1:
            raise ValueError("linear up-cosine bounds must be ordered within [-1, 1]")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


REPO_ROOT = repository_root()
DEFAULT_ACTOR = REPO_ROOT / "thirdparties/BFM-Zero-deploy/model/exported/FBcprAuxModel.onnx"
DEFAULT_CHECKPOINT = REPO_ROOT / "thirdparties/BFM-Zero-deploy/model/checkpoint"
DEFAULT_GOALS = REPO_ROOT / "thirdparties/BFM-Zero-deploy/model/goal_inference/goal_reaching.pkl"
DEFAULT_MOTION = REPO_ROOT / "thirdparties/BFM-Zero/humanoidverse/data/lafan_29dof.pkl"
ISAAC_PYTHON = REPO_ROOT / "thirdparties/BFM-Zero/.venv/bin/python"
OUTPUT_ROOT = REPO_ROOT / "logs/bfm-zero-fast-induced"
EXPECTED_ACTOR_SHA256 = "209097902c45621eebab2edb81070c31895fcdd558f0cf6f7f5a360fd747ab74"
EXPECTED_MODEL_SHA256 = "d8d07d7b83ad030286f44d55a48899dc42db4b48000132fffad063653a6eb49e"
