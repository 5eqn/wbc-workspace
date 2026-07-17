from __future__ import annotations

from typing import Any

import numpy as np

UPRIGHT_WORLD_UP_Z_THRESHOLD = 0.9
RESET_IDENTITY_ATOL = 1e-5

POSE_CONTRACT = {
    "reset_source_quaternion_order": "xyzw",
    "humanoidverse_root_state_quaternion_order": "xyzw",
    "isaac_root_pose_write_quaternion_order": "wxyz",
    "replay_qpos_quaternion_order": "wxyz",
    "reset_mujoco_quaternion": [1.0, 0.0, 0.0, 0.0],
    "reset_identity_atol": RESET_IDENTITY_ATOL,
    "upright_world_up_z_threshold": UPRIGHT_WORLD_UP_Z_THRESHOLD,
    "upright_gate_stages": ["post_warmup", "post_kick"],
}


def xyzw_to_wxyz(quaternion: Any) -> Any:
    """Reorder a final quaternion axis without changing its array/tensor type."""
    if quaternion.shape[-1] != 4:
        raise ValueError(f"quaternion must end in four components, got {quaternion.shape}")
    return quaternion[..., [3, 0, 1, 2]]


def world_up_z_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1] != 4 or not np.isfinite(value).all():
        raise ValueError(f"invalid WXYZ quaternion array with shape {value.shape}")
    norms = np.linalg.norm(value, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("WXYZ quaternion is not normalized")
    x = value[..., 1]
    y = value[..., 2]
    return 1.0 - 2.0 * (x * x + y * y)


def require_reset_identity(qpos: np.ndarray) -> None:
    quaternion = np.asarray(qpos, dtype=np.float64)[..., 3:7]
    identity = np.asarray(POSE_CONTRACT["reset_mujoco_quaternion"], dtype=np.float64)
    equivalent = np.minimum(
        np.linalg.norm(quaternion - identity, axis=-1),
        np.linalg.norm(quaternion + identity, axis=-1),
    )
    if not np.all(equivalent <= RESET_IDENTITY_ATOL):
        raise RuntimeError(
            "post-reset MuJoCo root quaternion is not upright identity; "
            f"maximum sign-invariant error is {float(equivalent.max()):.6g}"
        )


def require_upright(qpos: np.ndarray, stage: str) -> None:
    up_z = world_up_z_wxyz(np.asarray(qpos)[..., 3:7])
    if not np.all(up_z > UPRIGHT_WORLD_UP_Z_THRESHOLD):
        raise RuntimeError(
            f"{stage} root pose is not upright: minimum world-up Z is "
            f"{float(up_z.min()):.6g}, required > {UPRIGHT_WORLD_UP_Z_THRESHOLD}"
        )
