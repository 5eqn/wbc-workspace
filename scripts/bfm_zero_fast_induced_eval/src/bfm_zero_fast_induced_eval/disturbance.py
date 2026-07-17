from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import EvalConfig


@dataclass(frozen=True)
class VelocityDisturbance:
    seed: int
    linear_velocity_delta: tuple[float, float, float]
    linear_velocity_magnitude: float
    angular_velocity_delta: tuple[float, float, float]
    angular_velocity_magnitude: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    while True:
        vector = rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            return vector / norm


def sample_velocity_disturbance(seed: int, config: EvalConfig) -> VelocityDisturbance:
    """Match the established deploy disturbance sampler byte-for-byte in draw order."""
    rng = np.random.default_rng(seed)
    for _ in range(1024):
        linear_direction = _unit_vector(rng)
        up_cos = float(linear_direction[2])
        if config.linear_up_cos_min <= up_cos <= config.linear_up_cos_max:
            linear_magnitude = float(
                rng.uniform(config.linear_velocity_min_mps, config.linear_velocity_max_mps)
            )
            linear = linear_direction * linear_magnitude
            break
    else:
        raise RuntimeError("failed to sample a linear velocity within the up-cosine bounds")
    angular_magnitude = float(rng.uniform(config.angular_velocity_min_rps, config.angular_velocity_max_rps))
    angular = _unit_vector(rng) * angular_magnitude
    return VelocityDisturbance(
        seed=seed,
        linear_velocity_delta=tuple(float(x) for x in linear.astype(np.float32)),
        linear_velocity_magnitude=linear_magnitude,
        angular_velocity_delta=tuple(float(x) for x in angular.astype(np.float32)),
        angular_velocity_magnitude=angular_magnitude,
    )


def attempt_seed(base_seed: int, attempt_index: int) -> int:
    return int(base_seed + attempt_index * 1000)
