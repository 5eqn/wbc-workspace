import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("zero-norm quaternion")
    result = quat / norm
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0:
            result[index] *= -1
    return result


def resample_quaternion_xyzw(quat: np.ndarray, old_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_quat(normalize_quaternion_xyzw(quat))
    return Slerp(old_t, rotations)(new_t).as_quat().astype(np.float32)


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    matrices = Rotation.from_quat(normalize_quaternion_xyzw(quat)).as_matrix()
    return np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1).astype(np.float32)


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64)
    first = value[..., :3]
    second = value[..., 3:]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    matrix = np.stack((first, second, third), axis=-1)
    return Rotation.from_matrix(matrix).as_quat().astype(np.float32)


def torch_rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(rot6d[..., :3], dim=-1)
    second = rot6d[..., 3:]
    second = torch.nn.functional.normalize(
        second - (first * second).sum(dim=-1, keepdim=True) * first, dim=-1
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def torch_matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def perturb_rot6d(rot6d: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    tangent = (
        torch.randn(
            (*rot6d.shape[:-1], 3), device=rot6d.device, dtype=rot6d.dtype, generator=generator
        )
        * sigma
    )
    angle = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = tangent / angle
    skew = torch.zeros((*axis.shape[:-1], 3, 3), device=axis.device, dtype=axis.dtype)
    skew[..., 0, 1] = -axis[..., 2]
    skew[..., 0, 2] = axis[..., 1]
    skew[..., 1, 0] = axis[..., 2]
    skew[..., 1, 2] = -axis[..., 0]
    skew[..., 2, 0] = -axis[..., 1]
    skew[..., 2, 1] = axis[..., 0]
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(skew)
    delta = (
        identity
        + torch.sin(angle)[..., None] * skew
        + (1.0 - torch.cos(angle))[..., None] * (skew @ skew)
    )
    return torch_matrix_to_rot6d(delta @ torch_rot6d_to_matrix(rot6d))
