from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TrialMetrics:
    motion: str
    test_kind: str
    method: str
    seed: int
    completed: bool
    stand_up_success: bool | None
    joint_rmse_rad: float
    root_height_mae_m: float
    root_orientation_mae_rad: float
    failure_reason: str | None


def _orientation_error(reference_xyzw: np.ndarray, simulated_xyzw: np.ndarray) -> np.ndarray:
    relative = Rotation.from_quat(reference_xyzw).inv() * Rotation.from_quat(simulated_xyzw)
    return relative.magnitude()


def score_trial(path: Path) -> TrialMetrics:
    data = np.load(path, allow_pickle=False)
    time_s = data["time_s"]
    kind = str(data["test_kind"].item())
    if kind == "disturbed":
        primary_mask = time_s >= 0.5
    else:
        primary_mask = np.ones(len(time_s), dtype=bool)
    joint_error = data["simulated_joint_pos"] - data["reference_joint_pos"]
    joint_rmse = float(np.sqrt(np.mean(np.square(joint_error[primary_mask]))))
    height_error = np.abs(data["simulated_root_pos"][:, 2] - data["reference_root_pos"][:, 2])
    orientation_error = _orientation_error(
        data["reference_root_quat_xyzw"], data["simulated_root_quat_xyzw"]
    )
    final_mask = time_s >= time_s[-1] - 0.5
    stand_up = None
    if kind in {"get_up", "disturbed"}:
        stand_up = bool(np.mean(height_error[final_mask]) < 0.3)
    reason = str(data["failure_reason"].item()) if "failure_reason" in data else None
    if reason == "":
        reason = None
    return TrialMetrics(
        motion=str(data["motion"].item()),
        test_kind=kind,
        method=str(data["method"].item()),
        seed=int(data["seed"].item()),
        completed=bool(data["completed"].item()),
        stand_up_success=stand_up,
        joint_rmse_rad=joint_rmse,
        root_height_mae_m=float(np.mean(height_error)),
        root_orientation_mae_rad=float(np.mean(orientation_error)),
        failure_reason=reason,
    )


def _rank(metrics: TrialMetrics) -> tuple[int, float]:
    success = metrics.completed if metrics.test_kind == "normal" else bool(metrics.stand_up_success)
    return int(success), -metrics.joint_rmse_rad


def summarize_trials(trial_paths: list[Path], output: Path) -> dict[str, object]:
    metrics = [score_trial(path) for path in sorted(trial_paths)]
    aggregates: dict[str, dict[str, dict[str, float | int]]] = {}
    for kind in sorted({row.test_kind for row in metrics}):
        aggregates[kind] = {}
        for method in ("sonic", "ours"):
            rows = [row for row in metrics if row.test_kind == kind and row.method == method]
            if not rows:
                continue
            if kind == "normal":
                aggregates[kind][method] = {
                    "trials": len(rows),
                    "completion_rate": float(np.mean([row.completed for row in rows])),
                    "mean_joint_rmse_rad": float(np.mean([row.joint_rmse_rad for row in rows])),
                    "mean_root_height_mae_m": float(
                        np.mean([row.root_height_mae_m for row in rows])
                    ),
                    "mean_root_orientation_mae_rad": float(
                        np.mean([row.root_orientation_mae_rad for row in rows])
                    ),
                    "joint_rmse_gate_regressions": sum(row.joint_rmse_rad > 0.2 for row in rows),
                }
            else:
                aggregates[kind][method] = {
                    "trials": len(rows),
                    "stand_up_success_rate": float(
                        np.mean([bool(row.stand_up_success) for row in rows])
                    ),
                    "mean_joint_rmse_rad": float(np.mean([row.joint_rmse_rad for row in rows])),
                }
    paired: dict[tuple[str, str, int], dict[str, TrialMetrics]] = {}
    for row in metrics:
        paired.setdefault((row.motion, row.test_kind, row.seed), {})[row.method] = row
    selections: dict[str, dict[str, list[dict[str, object]]]] = {}
    for (motion, kind, seed), methods in paired.items():
        if set(methods) != {"sonic", "ours"}:
            raise ValueError(f"incomplete pair for {motion}/{kind}/{seed}")
        sonic, ours = methods["sonic"], methods["ours"]
        sonic_rank, ours_rank = _rank(sonic), _rank(ours)
        if sonic_rank == ours_rank:
            continue
        winner = "Ours > SONIC" if ours_rank > sonic_rank else "SONIC > Ours"
        success_margin = abs(ours_rank[0] - sonic_rank[0])
        rmse_margin = abs(ours.joint_rmse_rad - sonic.joint_rmse_rad)
        margin = success_margin * 1_000_000.0 + rmse_margin
        selections.setdefault(kind, {}).setdefault(winner, []).append(
            {"motion": motion, "seed": seed, "margin": margin}
        )
    for classes in selections.values():
        for winner, rows in classes.items():
            classes[winner] = sorted(
                rows, key=lambda row: (-float(row["margin"]), str(row["motion"]), int(row["seed"]))
            )[:3]
        for winner in ("Ours > SONIC", "SONIC > Ours"):
            classes.setdefault(winner, [])
    summary = {
        "trials": [asdict(row) for row in metrics],
        "primary_statistics": aggregates,
        "video_selections": selections,
        "joint_rmse_gate_rad": 0.2,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
