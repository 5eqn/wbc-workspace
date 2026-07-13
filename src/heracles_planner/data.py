from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

from .config import HeraclesConfig
from .rotations import quat_xyzw_to_rot6d, resample_quaternion_xyzw

SPLITS = {"train": (1, 2, 3), "validation": (4,), "test": (5,)}
REQUIRED_FIELDS = {
    "root_trans_offset": (None, 3),
    "dof": (None, 29),
    "root_rot": (None, 4),
}


@dataclass(frozen=True)
class Motion:
    name: str
    split: str
    joint_pos: np.ndarray
    root_quat_xyzw: np.ndarray
    root_pos: np.ndarray
    state: np.ndarray


def _subject(name: str) -> int:
    try:
        return int(name.rsplit("subject", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"motion has no subject suffix: {name}") from exc


def split_for_motion(name: str) -> str:
    subject = _subject(name)
    for split, subjects in SPLITS.items():
        if subject in subjects:
            return split
    raise ValueError(f"unsupported LAFAN1 subject in {name}")


def validate_raw_motion(name: str, raw: dict[str, Any], config: HeraclesConfig) -> None:
    if raw.get("fps") != config.source_hz:
        raise ValueError(f"{name}: expected {config.source_hz} Hz, got {raw.get('fps')}")
    lengths: set[int] = set()
    for field, expected_tail in REQUIRED_FIELDS.items():
        if field not in raw:
            raise ValueError(f"{name}: missing {field}")
        value = np.asarray(raw[field])
        if value.ndim != 2 or value.shape[1:] != expected_tail[1:]:
            raise ValueError(f"{name}: invalid {field} shape {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name}: non-finite {field}")
        lengths.add(len(value))
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError(f"{name}: inconsistent or empty arrays")
    quat_norm = np.linalg.norm(np.asarray(raw["root_rot"]), axis=-1)
    if np.max(np.abs(quat_norm - 1.0)) > 1e-3:
        raise ValueError(f"{name}: invalid root quaternion norm")


def resample_motion(name: str, raw: dict[str, Any], config: HeraclesConfig) -> Motion:
    validate_raw_motion(name, raw, config)
    frames = len(raw["dof"])
    old_t = np.arange(frames, dtype=np.float64) / config.source_hz
    new_frames = round(old_t[-1] * config.data_hz) + 1
    new_t = np.arange(new_frames, dtype=np.float64) / config.data_hz
    new_t[-1] = min(new_t[-1], old_t[-1])
    joints = CubicSpline(old_t, np.asarray(raw["dof"], dtype=np.float64), axis=0)(new_t)
    root_pos = CubicSpline(old_t, np.asarray(raw["root_trans_offset"], dtype=np.float64), axis=0)(
        new_t
    )
    root_quat = resample_quaternion_xyzw(np.asarray(raw["root_rot"]), old_t, new_t)
    state = np.concatenate((joints, quat_xyzw_to_rot6d(root_quat)), axis=-1).astype(np.float32)
    return Motion(
        name=name,
        split=split_for_motion(name),
        joint_pos=joints.astype(np.float32),
        root_quat_xyzw=root_quat,
        root_pos=root_pos.astype(np.float32),
        state=state,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def compute_jacobian_weights(
    model_path: Path, config: HeraclesConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if len(joint_ids) != config.joint_dim:
        raise ValueError(f"expected 29 hinge joints in {model_path}, found {len(joint_ids)}")
    qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    scores = np.zeros(config.joint_dim, dtype=np.float64)
    step = config.jacobian_fd_step
    for index, address in enumerate(qpos_addresses):
        initial = float(data.qpos[address])
        data.qpos[address] = initial + step
        mujoco.mj_forward(model, data)
        plus = data.xpos.copy()
        data.qpos[address] = initial - step
        mujoco.mj_forward(model, data)
        minus = data.xpos.copy()
        data.qpos[address] = initial
        scores[index] = np.linalg.norm((plus - minus) / (2.0 * step))
    scores /= scores.mean()
    scores = np.maximum(scores, config.jacobian_weight_floor).astype(np.float32)
    weights = np.concatenate((scores, np.ones(config.rotation_dim, dtype=np.float32)))
    return weights, {
        "model": str(model_path),
        "finite_difference_step_rad": step,
        "normalized_joint_weight_floor": config.jacobian_weight_floor,
        "joint_names": [model.joint(joint_id).name for joint_id in joint_ids],
    }


def preprocess_dataset(
    source: Path, output: Path, model_path: Path, config: HeraclesConfig
) -> dict[str, Any]:
    raw = joblib.load(source, mmap_mode="r")
    if not isinstance(raw, dict) or len(raw) != 40:
        raise ValueError(f"expected exactly 40 motions, found {type(raw).__name__}:{len(raw)}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_motions = []
    training_states = []
    for name in sorted(raw):
        motion = resample_motion(name, raw[name], config)
        np.savez_compressed(
            output / f"{name}.npz",
            joint_pos=motion.joint_pos,
            root_quat_xyzw=motion.root_quat_xyzw,
            root_pos=motion.root_pos,
            state=motion.state,
        )
        if motion.split == "train":
            training_states.append(motion.state)
        manifest_motions.append(
            {
                "name": name,
                "subject": _subject(name),
                "split": motion.split,
                "frames_50hz": len(motion.state),
                "duration_s": (len(motion.state) - 1) / config.data_hz,
                "get_up": name == "fallAndGetUp1_subject5",
            }
        )
    train = np.concatenate(training_states, axis=0).astype(np.float64)
    mean = train.mean(axis=0).astype(np.float32)
    std = train.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    weights, jacobian = compute_jacobian_weights(model_path, config)
    np.savez(output / "normalization.npz", mean=mean, std=std, loss_weights=weights)
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_quaternion_order": "xyzw",
        "output_quaternion_order": "xyzw",
        "joint_order": "G1 29-DOF IsaacLab order (identical in source and SONIC v1)",
        "source_hz": config.source_hz,
        "output_hz": config.data_hz,
        "split_subjects": {key: list(value) for key, value in SPLITS.items()},
        "normalization_split": "train",
        "jacobian_weighting": jacobian,
        "motions": manifest_motions,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class MotionWindowDataset(Dataset[tuple[np.ndarray, np.ndarray, np.float32]]):
    def __init__(self, root: Path, split: str, config: HeraclesConfig):
        self.root = root
        self.split = split
        self.config = config
        manifest = json.loads((root / "manifest.json").read_text())
        self.motions: list[np.ndarray] = []
        self.index: list[tuple[int, int]] = []
        max_frames = round(config.segment_max_s * config.data_hz)
        for row in manifest["motions"]:
            if row["split"] != split:
                continue
            state = np.load(root / f"{row['name']}.npz")["state"]
            motion_index = len(self.motions)
            self.motions.append(state)
            for start in range(0, len(state) - max_frames, config.stride_frames):
                self.index.append((motion_index, start))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.float32]:
        motion_index, start = self.index[index]
        rng = np.random.default_rng(self.config.seed + self.epoch * len(self) + index)
        log_duration = rng.uniform(
            math.log(self.config.segment_min_s), math.log(self.config.segment_max_s)
        )
        duration = float(math.exp(log_duration))
        offsets = np.linspace(0, duration * self.config.data_hz, self.config.keyframes)
        source = self.motions[motion_index]
        positions = start + offsets
        base = np.floor(positions).astype(np.int64)
        fraction = (positions - base).astype(np.float32)[:, None]
        indices = np.stack((base - 1, base, base + 1, base + 2), axis=1)
        indices = np.clip(indices, 0, len(source) - 1)
        p0, p1, p2, p3 = (source[indices[:, offset]] for offset in range(4))
        sampled = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * fraction
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * fraction**2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * fraction**3
        )
        sampled = sampled.astype(np.float32)
        residual = sampled - sampled[0]
        residual[0] = 0.0
        return sampled[0].copy(), residual, np.float32(duration)
