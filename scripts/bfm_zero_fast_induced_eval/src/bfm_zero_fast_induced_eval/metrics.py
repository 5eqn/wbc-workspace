from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrialMetrics:
    induced_fall: bool
    recovered: bool
    minimum_first_second_m: float
    final_half_second_mean_m: float


def trial_metrics(
    pelvis_height_200hz: np.ndarray,
    *,
    physics_hz: int = 200,
    fallen_height_m: float = 0.45,
    recovery_height_m: float = 0.75,
) -> TrialMetrics:
    heights = np.asarray(pelvis_height_200hz, dtype=np.float32)
    if heights.ndim != 1 or heights.size < physics_hz:
        raise ValueError("pelvis height must be a one-dimensional array spanning at least one second")
    if not np.isfinite(heights).all():
        raise ValueError("pelvis height contains non-finite values")
    first_min = float(heights[:physics_hz].min())
    final_mean = float(heights[-round(0.5 * physics_hz) :].mean())
    # Compare in the sampled tensor's precision so an exact float32 threshold is a boundary, not a fall.
    fallen_threshold = float(np.float32(fallen_height_m))
    recovery_threshold = float(np.float32(recovery_height_m))
    induced = first_min < fallen_threshold
    return TrialMetrics(induced, induced and final_mean > recovery_threshold, first_min, final_mean)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)
