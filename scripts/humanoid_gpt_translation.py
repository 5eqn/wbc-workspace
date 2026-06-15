#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from benchmark_common import HARDWARE_FROM_SONIC_POLICY, MOTIONS, ROOT, SONIC_BODY_PART_INDEXES
from sim_bridge import BODY_JOINT_NAMES

HGPT_REPO = ROOT / "thirdparties" / "Humanoid-GPT"
if str(HGPT_REPO) not in sys.path:
    sys.path.insert(0, str(HGPT_REPO))

from tracking.constants import KPT_NAMES, TRACK_XML  # noqa: E402


LOG_ROOT = ROOT / "logs" / "humanoid-gpt-translation"
ARTIFACT_ROOT = ROOT / "artifacts" / "humanoid-gpt-translation"
KINEMATIC_TOL = 1e-5
NUMERIC_TOL = 1e-5


@dataclass(frozen=True)
class MotionArrays:
    motion: str
    source: str
    frequency: float
    qpos: np.ndarray
    qvel: np.ndarray
    root_pos: np.ndarray
    root_rot_wxyz: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    body_pos_subset: np.ndarray
    body_quat_subset_wxyz: np.ndarray
    body_lin_vel_subset: np.ndarray
    body_ang_vel_subset: np.ndarray
    metadata: dict[str, Any]


def _canon_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)


def _canon_wxyz_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    dot = np.sum(reference * candidate, axis=-1, keepdims=True)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    return candidate * sign


def _read_csv_matrix(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            values = [value for value in row if value != ""]
            if not values:
                continue
            rows.append([float(value) for value in values])
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    width = min(len(row) for row in rows)
    return np.asarray([row[:width] for row in rows], dtype=np.float32)


def _load_sonic_motion(motion: str) -> MotionArrays:
    motion_root = ROOT / "assets" / "motions" / "sonic_motions" / motion
    joint_pos = _read_csv_matrix(motion_root / "joint_pos.csv")
    joint_vel = _read_csv_matrix(motion_root / "joint_vel.csv")
    body_pos = _read_csv_matrix(motion_root / "body_pos.csv")
    body_quat = _read_csv_matrix(motion_root / "body_quat.csv")
    body_lin_vel = _read_csv_matrix(motion_root / "body_lin_vel.csv")
    body_ang_vel = _read_csv_matrix(motion_root / "body_ang_vel.csv")

    dof_pos = joint_pos[:, HARDWARE_FROM_SONIC_POLICY]
    dof_vel = joint_vel[:, HARDWARE_FROM_SONIC_POLICY]

    body_pos = body_pos.reshape(body_pos.shape[0], body_pos.shape[1] // 3, 3)
    body_quat = body_quat.reshape(body_quat.shape[0], body_quat.shape[1] // 4, 4)
    body_lin_vel = body_lin_vel.reshape(body_lin_vel.shape[0], body_lin_vel.shape[1] // 3, 3)
    body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] // 3, 3)

    body_width = int(body_pos.shape[1])
    if body_width == len(SONIC_BODY_PART_INDEXES):
        body_pos_subset = body_pos
        body_quat_subset = body_quat
        body_lin_vel_subset = body_lin_vel
        body_ang_vel_subset = body_ang_vel
        sonic_body_indexes = SONIC_BODY_PART_INDEXES
    elif body_width > max(SONIC_BODY_PART_INDEXES):
        body_pos_subset = body_pos[:, SONIC_BODY_PART_INDEXES]
        body_quat_subset = body_quat[:, SONIC_BODY_PART_INDEXES]
        body_lin_vel_subset = body_lin_vel[:, SONIC_BODY_PART_INDEXES]
        body_ang_vel_subset = body_ang_vel[:, SONIC_BODY_PART_INDEXES]
        sonic_body_indexes = SONIC_BODY_PART_INDEXES
    else:
        raise ValueError(f"{motion}: SONIC body arrays expose {body_width} bodies, cannot subset expected key bodies")

    root_pos = body_pos_subset[:, 0]
    root_rot_wxyz = body_quat_subset[:, 0]
    qpos = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1)
    qvel = np.concatenate([root_pos * 0.0, root_pos * 0.0, dof_vel], axis=1)

    metadata = {
        "motion": motion,
        "source": "sonic",
        "joint_order": "sonic_joint_pos_csv_remapped_via_HARDWARE_FROM_SONIC_POLICY",
        "joint_order_target": BODY_JOINT_NAMES,
        "body_subset_rule": (
            "use body CSV directly when width==14 bodies; otherwise subset full-width "
            f"body arrays by SONIC_BODY_PART_INDEXES={sonic_body_indexes}"
        ),
        "quaternion_convention": "wxyz",
        "frequency_hz": 50.0,
    }
    return MotionArrays(
        motion=motion,
        source="sonic",
        frequency=50.0,
        qpos=qpos,
        qvel=qvel,
        root_pos=root_pos,
        root_rot_wxyz=root_rot_wxyz,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        body_pos_subset=body_pos_subset,
        body_quat_subset_wxyz=body_quat_subset,
        body_lin_vel_subset=body_lin_vel_subset,
        body_ang_vel_subset=body_ang_vel_subset,
        metadata=metadata,
    )


def _load_holomotion_motion(motion: str) -> MotionArrays:
    path = ROOT / "assets" / "motions" / "holomotion_motions" / f"{motion}_holomotion.npz"
    data = np.load(path, allow_pickle=True)
    dof_pos = np.asarray(data["ref_dof_pos"], dtype=np.float32)[:, :29]
    dof_vel = np.asarray(data["ref_dof_vel"], dtype=np.float32)[:, :29]
    body_pos = np.asarray(data["ref_global_translation"], dtype=np.float32)
    body_quat_xyzw = np.asarray(data["ref_global_rotation_quat"], dtype=np.float32)
    body_lin_vel = np.asarray(data["ref_global_velocity"], dtype=np.float32)
    body_ang_vel = np.asarray(data["ref_global_angular_velocity"], dtype=np.float32)
    body_quat_wxyz = _canon_xyzw_to_wxyz(body_quat_xyzw)

    root_pos = body_pos[:, 0]
    root_rot_wxyz = body_quat_wxyz[:, 0]
    qpos = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1)
    qvel = np.concatenate([root_pos * 0.0, root_pos * 0.0, dof_vel], axis=1)

    meta_blob = str(data["metadata"]) if "metadata" in data else ""
    metadata = {
        "motion": motion,
        "source": "holomotion",
        "joint_order": "holomotion_ref_dof_pos_direct",
        "joint_order_target": BODY_JOINT_NAMES,
        "body_subset_rule": f"subset full 30-body arrays by SONIC_BODY_PART_INDEXES={SONIC_BODY_PART_INDEXES}",
        "quaternion_convention_input": "xyzw",
        "quaternion_convention_target": "wxyz",
        "frequency_hz": 50.0,
        "metadata_json": meta_blob,
    }
    return MotionArrays(
        motion=motion,
        source="holomotion",
        frequency=50.0,
        qpos=qpos,
        qvel=qvel,
        root_pos=root_pos,
        root_rot_wxyz=root_rot_wxyz,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        body_pos_subset=body_pos[:, SONIC_BODY_PART_INDEXES],
        body_quat_subset_wxyz=body_quat_wxyz[:, SONIC_BODY_PART_INDEXES],
        body_lin_vel_subset=body_lin_vel[:, SONIC_BODY_PART_INDEXES],
        body_ang_vel_subset=body_ang_vel[:, SONIC_BODY_PART_INDEXES],
        metadata=metadata,
    )


def _body_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = model.body(name).id
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = data.xmat[bid].reshape(3, 3)
    pose[:3, 3] = data.xpos[bid]
    return pose


def _fk_key_poses(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    out = np.zeros((qpos.shape[0], len(KPT_NAMES), 4, 4), dtype=np.float32)
    for idx in range(qpos.shape[0]):
        data.qpos[:] = qpos[idx]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for j, name in enumerate(KPT_NAMES):
            out[idx, j] = _body_pose(model, data, name)
    return out


def _compare_motion_pair(model: mujoco.MjModel, sonic: MotionArrays, holo: MotionArrays) -> dict[str, Any]:
    frame_count = min(len(sonic.qpos), len(holo.qpos))
    sonic_qpos = sonic.qpos[:frame_count]
    holo_qpos = holo.qpos[:frame_count]
    sonic_qvel = sonic.qvel[:frame_count]
    holo_qvel = holo.qvel[:frame_count]

    aligned_sonic_root_rot = _canon_wxyz_sign(holo.root_rot_wxyz[:frame_count], sonic.root_rot_wxyz[:frame_count])
    sonic_qpos_aligned = sonic_qpos.copy()
    sonic_qpos_aligned[:, 3:7] = aligned_sonic_root_rot

    qpos_abs = np.abs(sonic_qpos_aligned - holo_qpos)
    qvel_abs = np.abs(sonic_qvel - holo_qvel)

    sonic_body_quat_aligned = _canon_wxyz_sign(
        holo.body_quat_subset_wxyz[:frame_count],
        sonic.body_quat_subset_wxyz[:frame_count],
    )
    body_quat_abs = np.abs(sonic_body_quat_aligned - holo.body_quat_subset_wxyz[:frame_count])

    sonic_fk = _fk_key_poses(model, sonic_qpos_aligned)
    holo_fk = _fk_key_poses(model, holo_qpos)
    fk_abs = np.abs(sonic_fk - holo_fk)

    passed = bool(
        float(np.max(qpos_abs)) <= NUMERIC_TOL
        and float(np.max(qvel_abs)) <= NUMERIC_TOL
        and float(np.max(np.abs(sonic.body_pos_subset[:frame_count] - holo.body_pos_subset[:frame_count]))) <= NUMERIC_TOL
        and float(np.max(body_quat_abs)) <= NUMERIC_TOL
        and float(np.max(np.abs(sonic.body_lin_vel_subset[:frame_count] - holo.body_lin_vel_subset[:frame_count]))) <= NUMERIC_TOL
        and float(np.max(np.abs(sonic.body_ang_vel_subset[:frame_count] - holo.body_ang_vel_subset[:frame_count]))) <= NUMERIC_TOL
        and float(np.max(fk_abs)) <= KINEMATIC_TOL
    )

    return {
        "motion": sonic.motion,
        "passed": passed,
        "frame_count_compared": frame_count,
        "joint_order_target": BODY_JOINT_NAMES,
        "key_body_names": KPT_NAMES,
        "normalization": {
            "frequency_hz": 50.0,
            "frame_crop_rule": "min(len(sonic), len(holomotion))",
            "root_quaternion_rule": "compare in wxyz; flip SONIC signs to maximize per-frame dot with HoloMotion",
            "holomotion_quaternion_input": "xyzw",
            "sonic_quaternion_input": "wxyz",
            "neutral_prefix_rule": "none; compare published benchmark assets as-is",
        },
        "source_metadata": {
            "sonic": sonic.metadata,
            "holomotion": holo.metadata,
        },
        "numeric_diffs": {
            "max_abs_qpos": float(np.max(qpos_abs)),
            "mean_abs_qpos": float(np.mean(qpos_abs)),
            "max_abs_qvel": float(np.max(qvel_abs)),
            "mean_abs_qvel": float(np.mean(qvel_abs)),
            "max_abs_body_pos": float(np.max(np.abs(sonic.body_pos_subset[:frame_count] - holo.body_pos_subset[:frame_count]))),
            "max_abs_body_quat": float(np.max(body_quat_abs)),
            "max_abs_body_lin_vel": float(np.max(np.abs(sonic.body_lin_vel_subset[:frame_count] - holo.body_lin_vel_subset[:frame_count]))),
            "max_abs_body_ang_vel": float(np.max(np.abs(sonic.body_ang_vel_subset[:frame_count] - holo.body_ang_vel_subset[:frame_count]))),
        },
        "fk_diffs": {
            "max_abs_pose_matrix": float(np.max(fk_abs)),
            "mean_abs_pose_matrix": float(np.mean(fk_abs)),
        },
    }


def _save_translation_npz(arrays: MotionArrays, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arrays.motion}.npz"
    np.savez_compressed(
        out_path,
        qpos=arrays.qpos,
        qvel=arrays.qvel,
        root_pos=arrays.root_pos,
        root_rot=arrays.root_rot_wxyz,
        dof_pos=arrays.dof_pos,
        dof_vel=arrays.dof_vel,
        frequency=np.array(arrays.frequency, dtype=np.float32),
        body_pos_subset=arrays.body_pos_subset,
        body_quat_subset_wxyz=arrays.body_quat_subset_wxyz,
        body_lin_vel_subset=arrays.body_lin_vel_subset,
        body_ang_vel_subset=arrays.body_ang_vel_subset,
        metadata_json=np.array(json.dumps(arrays.metadata), dtype=f"<U{max(1, len(json.dumps(arrays.metadata)))}"),
    )
    return out_path


def _write_joint_mapping_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Humanoid-GPT Translation Joint Mapping",
        "",
        "Target joint order for translated Humanoid-GPT references:",
        "",
    ]
    for idx, name in enumerate(BODY_JOINT_NAMES):
        lines.append(f"- `{idx}`: `{name}`")
    lines.extend([
        "",
        "Mapping rules:",
        "",
        "- HoloMotion `ref_dof_pos` is already in the target 29-DOF benchmark order.",
        "- SONIC `joint_pos.csv` is converted to target order via `HARDWARE_FROM_SONIC_POLICY` from `scripts/benchmark_common.py`.",
        "- HoloMotion global quaternions are stored as `xyzw` and converted to `wxyz`.",
        "- SONIC global quaternions are already `wxyz`.",
        f"- Key-body comparison subset uses `SONIC_BODY_PART_INDEXES={SONIC_BODY_PART_INDEXES}`.",
    ])
    path.write_text("\n".join(lines) + "\n")


def _write_diff_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "motion",
        "passed",
        "frame_count_compared",
        "max_abs_qpos",
        "mean_abs_qpos",
        "max_abs_qvel",
        "mean_abs_qvel",
        "max_abs_body_pos",
        "max_abs_body_quat",
        "max_abs_body_lin_vel",
        "max_abs_body_ang_vel",
        "max_abs_fk_pose_matrix",
        "mean_abs_fk_pose_matrix",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "motion": row["motion"],
                "passed": row["passed"],
                "frame_count_compared": row["frame_count_compared"],
                "max_abs_qpos": row["numeric_diffs"]["max_abs_qpos"],
                "mean_abs_qpos": row["numeric_diffs"]["mean_abs_qpos"],
                "max_abs_qvel": row["numeric_diffs"]["max_abs_qvel"],
                "mean_abs_qvel": row["numeric_diffs"]["mean_abs_qvel"],
                "max_abs_body_pos": row["numeric_diffs"]["max_abs_body_pos"],
                "max_abs_body_quat": row["numeric_diffs"]["max_abs_body_quat"],
                "max_abs_body_lin_vel": row["numeric_diffs"]["max_abs_body_lin_vel"],
                "max_abs_body_ang_vel": row["numeric_diffs"]["max_abs_body_ang_vel"],
                "max_abs_fk_pose_matrix": row["fk_diffs"]["max_abs_pose_matrix"],
                "mean_abs_fk_pose_matrix": row["fk_diffs"]["mean_abs_pose_matrix"],
            })


def run_translation_gate(args: argparse.Namespace) -> int:
    motions = args.motions or MOTIONS
    sonic_out = LOG_ROOT / "converted_from_sonic"
    holo_out = LOG_ROOT / "converted_from_holomotion"
    eq_out = LOG_ROOT / "equivalence_checks"
    eq_out.mkdir(parents=True, exist_ok=True)

    track_xml = HGPT_REPO / TRACK_XML
    model = mujoco.MjModel.from_xml_path(str(track_xml))

    rows: list[dict[str, Any]] = []
    for motion in motions:
        sonic = _load_sonic_motion(motion)
        holo = _load_holomotion_motion(motion)
        sonic_path = _save_translation_npz(sonic, sonic_out)
        holo_path = _save_translation_npz(holo, holo_out)
        row = _compare_motion_pair(model, sonic, holo)
        row["translated_paths"] = {
            "sonic": str(sonic_path.relative_to(ROOT)),
            "holomotion": str(holo_path.relative_to(ROOT)),
        }
        (eq_out / f"{motion}.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)

    summary = {
        "passed": all(row["passed"] for row in rows),
        "motions": rows,
        "tolerances": {
            "numeric_tolerance": NUMERIC_TOL,
            "kinematic_tolerance": KINEMATIC_TOL,
        },
        "goal_gate": "Block if any motion fails translation equivalence after deterministic normalization",
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "translation_equivalence.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_diff_csv(ARTIFACT_ROOT / "per_motion_diffs.csv", rows)
    _write_joint_mapping_report(ARTIFACT_ROOT / "joint_mapping_report.md")
    (ARTIFACT_ROOT / "spotcheck_videos.json").write_text(
        json.dumps(
            {
                "status": "not_generated_yet",
                "reason": "translation gate currently verifies numeric and FK equivalence first",
                "next_step": "generate paired Humanoid-GPT-space visual spot checks if any mismatch appears or before final completion audit",
            },
            indent=2,
        )
        + "\n"
    )

    print(json.dumps({"ok": summary["passed"], "artifact": str((ARTIFACT_ROOT / 'translation_equivalence.json').relative_to(ROOT))}, indent=2))
    return 0 if summary["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate SONIC/HoloMotion motions into Humanoid-GPT format and verify equivalence.")
    parser.add_argument("--motions", nargs="*", default=None, choices=MOTIONS)
    args = parser.parse_args()
    return run_translation_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
