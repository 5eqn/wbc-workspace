#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmark_common import MOTIONS, ROOT


HGPT_REPO = ROOT / "thirdparties" / "Humanoid-GPT"
TRANSLATION_ROOT = ROOT / "logs" / "humanoid-gpt-translation"
LOG_ROOT = ROOT / "logs" / "humanoid-gpt-builtin-eval"
ARTIFACT_ROOT = ROOT / "artifacts" / "humanoid-gpt-builtin-eval"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _build_source_dir(source: str) -> Path:
    return TRANSLATION_ROOT / f"converted_from_{source}"


def _prepend_runtime_libs() -> list[str]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return []

    prefix = Path(conda_prefix)
    extra_dirs: list[Path] = []
    extra_dirs.extend(sorted(prefix.glob("lib/python*/site-packages/nvidia/*/lib")))
    extra_dirs.extend(sorted(prefix.glob("lib/python*/site-packages/tensorrt_libs")))

    new_entries = [str(path) for path in extra_dirs if path.is_dir()]
    if not new_entries:
        return []

    existing = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    merged: list[str] = []
    for entry in new_entries + existing:
        if entry not in merged:
            merged.append(entry)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
    return new_entries


def _load_hgpt_modules() -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    runtime_libs = _prepend_runtime_libs()

    if str(HGPT_REPO) not in sys.path:
        sys.path.insert(0, str(HGPT_REPO))

    os.chdir(HGPT_REPO)

    policy_mod = importlib.import_module("tracking.policy")
    consts_mod = importlib.import_module("tracking.constants")
    logger_mod = importlib.import_module("utils.logger")
    inference_mod = importlib.import_module("scripts.inference")
    eval_mod = importlib.import_module("scripts.eval_parallel")
    ort_mod = importlib.import_module("onnxruntime")
    jax_mod = importlib.import_module("jax")
    torch_mod = importlib.import_module("torch")

    return {
        "policy_mod": policy_mod,
        "consts_mod": consts_mod,
        "logger_mod": logger_mod,
        "inference_mod": inference_mod,
        "eval_mod": eval_mod,
        "ort_mod": ort_mod,
        "jax_mod": jax_mod,
        "torch_mod": torch_mod,
        "runtime_libs": runtime_libs,
    }


def _motion_files(source_dir: Path, motions: list[str] | None) -> list[Path]:
    selected = motions or MOTIONS
    files = [source_dir / f"{motion}.npz" for motion in selected]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing translated motions: {[str(path) for path in missing]}")
    return files


def _summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No metrics rows to summarize")

    def avg(key: str) -> float:
        return float(sum(float(row[key]) for row in rows) / len(rows))

    successful = sum(1 for row in rows if float(row["length_ratio"]) >= 1.0)
    return {
        "motion_count": len(rows),
        "successful_motion_count": successful,
        "success_rate": float(successful / len(rows)),
        "average_length_ratio": avg("length_ratio"),
        "average_kpt_pos_mae": avg("kpt_pos_mae"),
        "average_kpt_rot_mae": avg("kpt_rot_mae"),
        "average_joint_pos_mae": avg("joint_pos_mae"),
        "average_joint_vel_mae": avg("joint_vel_mae"),
        "average_root_pos_err_mm": avg("root_pos_err_mm"),
        "average_root_vel_err_mms": avg("root_vel_err_mms"),
        "average_root_yaw_err": avg("root_yaw_err"),
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "motion",
        "file_name",
        "traj_id",
        "length_ratio",
        "kpt_pos_mae",
        "kpt_rot_mae",
        "joint_pos_mae",
        "joint_vel_mae",
        "root_pos_err_mm",
        "root_vel_err_mms",
        "root_yaw_err",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "motion": Path(row["file_name"]).stem,
                "file_name": row["file_name"],
                "traj_id": row["traj_id"],
                "length_ratio": row["length_ratio"],
                "kpt_pos_mae": row["kpt_pos_mae"],
                "kpt_rot_mae": row["kpt_rot_mae"],
                "joint_pos_mae": row["joint_pos_mae"],
                "joint_vel_mae": row["joint_vel_mae"],
                "root_pos_err_mm": row["root_pos_err_mm"],
                "root_vel_err_mms": row["root_vel_err_mms"],
                "root_yaw_err": row["root_yaw_err"],
            })


def _run_eval_metrics(
    eval_mod: Any,
    policy: Any,
    motion_files: list[Path],
    load_path: Path,
    track_xml: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    eval_args = eval_mod.ParallelEvalArgs(
        load_path=str(load_path),
        mocap_path=str(motion_files[0].parent.resolve()),
        privileged=args.privileged,
        num_envs=1,
        device=args.device,
        convert=True,
        freq=args.freq,
        convert_xml_path=str(track_xml),
        workers=args.workers,
    )
    convert_mj_model = eval_mod.mujoco.MjModel.from_xml_path(str(track_xml))
    env_cfg = eval_mod.g1_infer_env_config(ctrl_dt=1 / args.freq)

    rows: list[dict[str, Any]] = []
    for traj_id, motion_file in enumerate(motion_files):
        raw_data = eval_mod._load_npz_with_qpos(motion_file)
        ref_traj = eval_mod._convert_traj_to_kpt(raw_data, convert_mj_model, args.freq)
        rows.append(
            eval_mod._evaluate_single_traj(
                traj_id=traj_id,
                ref_traj=ref_traj,
                file_name=motion_file.name,
                args=eval_args,
                env_cfg=env_cfg,
                policy=policy,
            )
        )
    return rows


def run_builtin_eval(args: argparse.Namespace) -> int:
    source_dir = _build_source_dir(args.source)
    motion_files = _motion_files(source_dir, args.motions)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    modules = _load_hgpt_modules()
    policy_mod = modules["policy_mod"]
    consts_mod = modules["consts_mod"]
    logger_mod = modules["logger_mod"]
    inference_mod = modules["inference_mod"]
    eval_mod = modules["eval_mod"]
    ort_mod = modules["ort_mod"]
    jax_mod = modules["jax_mod"]
    torch_mod = modules["torch_mod"]

    track_xml = (HGPT_REPO / consts_mod.TRACK_XML).resolve()
    load_path = (HGPT_REPO / args.load_path).resolve() if not Path(args.load_path).is_absolute() else Path(args.load_path)
    metrics_log_dir = LOG_ROOT / "native_metrics"
    videos_log_dir = LOG_ROOT / "videos"
    metrics_log_dir.mkdir(parents=True, exist_ok=True)
    videos_log_dir.mkdir(parents=True, exist_ok=True)

    policy_args = policy_mod.Args(load_path=str(load_path), device=args.device)
    policy = policy_mod.get_policy_onnx(policy_args)
    policy_providers = list(policy.onnx_model.get_providers())

    logger_mod.update_file_handler(metrics_log_dir / "eval_parallel.log")
    eval_metrics = _run_eval_metrics(
        eval_mod=eval_mod,
        policy=policy,
        motion_files=motion_files,
        load_path=load_path,
        track_xml=track_xml,
        args=args,
    )
    eval_metrics = sorted(eval_metrics, key=lambda row: row["traj_id"])

    (metrics_log_dir / "eval_parallel_metrics.json").write_text(json.dumps(eval_metrics, indent=2) + "\n")

    videos: list[dict[str, Any]] = []
    for motion_file in motion_files:
        motion = motion_file.stem
        video_path = videos_log_dir / f"{motion}.mp4"
        logger_mod.update_file_handler(videos_log_dir / f"{motion}.log")
        infer_args = inference_mod.InferenceArgs(
            load_path=str(load_path),
            mocap_path=str(motion_file.resolve()),
            privileged=args.privileged,
            video_path=str(video_path),
            headless=True,
            num_envs=1,
            device=args.device,
            convert=True,
            freq=args.freq,
            convert_xml_path=str(track_xml),
            show_ref_ghost=False,
        )
        inference_mod.main(infer_args)
        videos.append({
            "motion": motion,
            "source_npz": _rel(motion_file.resolve()),
            "video_path": _rel(video_path.resolve()),
            "log_path": _rel((videos_log_dir / f"{motion}.log").resolve()),
        })

    summary = {
        "source": args.source,
        "motion_set": [path.stem for path in motion_files],
        "load_path": _rel(load_path.resolve()),
        "track_xml": _rel(track_xml.resolve()),
        "device": args.device,
        "workers": args.workers,
        "privileged": args.privileged,
        "convert": True,
        "policy_providers": policy_providers,
        "available_ort_providers": list(ort_mod.get_available_providers()),
        "jax_devices": [str(device) for device in jax_mod.devices()],
        "torch_cuda_available": bool(torch_mod.cuda.is_available()),
        "torch_cuda_device_count": int(torch_mod.cuda.device_count()),
        "runtime_library_dirs": modules["runtime_libs"],
        "translation_equivalence_artifact": "artifacts/humanoid-gpt-translation/translation_equivalence.json",
        "metrics_source": "official Humanoid-GPT scripts.eval_parallel internals with --convert",
        "metrics_execution_mode": "single-process wrapper over upstream eval_parallel internals to preserve CUDA ONNX runtime stability",
        "video_source": "official Humanoid-GPT scripts.inference with --convert",
        "video_note": "Upstream inference applies EMA smoothing before conversion; videos are qualitative artifacts alongside eval_parallel metrics.",
    }
    summary.update(_summarize_metrics(eval_metrics))

    per_motion_csv = ARTIFACT_ROOT / "per_motion_metrics.csv"
    metrics_summary_json = ARTIFACT_ROOT / "metrics_summary.json"
    videos_json = ARTIFACT_ROOT / "videos.json"
    _write_metrics_csv(per_motion_csv, eval_metrics)
    metrics_summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    videos_json.write_text(json.dumps(videos, indent=2) + "\n")

    print(json.dumps({
        "ok": True,
        "metrics_summary": _rel(metrics_summary_json.resolve()),
        "per_motion_metrics": _rel(per_motion_csv.resolve()),
        "videos": _rel(videos_json.resolve()),
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official Humanoid-GPT built-in evaluation on translated benchmark motions.")
    parser.add_argument("--source", choices=["holomotion", "sonic"], default="holomotion")
    parser.add_argument("--motions", nargs="*", choices=MOTIONS, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--freq", type=int, default=50)
    parser.add_argument("--privileged", action="store_true", default=False)
    parser.add_argument("--load-path", default="storage/ckpts/pns_wo_priv216.onnx")
    args = parser.parse_args()
    return run_builtin_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
