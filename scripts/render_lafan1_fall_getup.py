#!/usr/bin/env python3
"""Render the six G1-retargeted LAFAN1 fall/get-up clips at source speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path.home() / "wm-workspace/datasets/lafan1_retargeting"
DEFAULT_SCENE = ROOT / "thirdparties/GR00T-WholeBodyControl/motionbricks/assets/skeletons/g1/scene_29dof.xml"


def csv_pose_to_mujoco(row: np.ndarray) -> np.ndarray:
    """Convert CSV xyzw free-joint quaternion to MuJoCo's wxyz convention."""
    return np.concatenate((row[:3], row[6:7], row[3:6], row[7:]))


def render_clip(csv_path: Path, output_path: Path, scene_path: Path, fps: int) -> dict[str, object]:
    poses = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if poses.ndim != 2 or poses.shape[1] != model.nq:
        raise ValueError(f"{csv_path}: expected (*, {model.nq}), got {poses.shape}")

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)
    pelvis_id = model.body("pelvis").id
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = pelvis_id
    camera.distance = 2.7
    camera.azimuth = 135.0
    camera.elevation = -18.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-movflags", "+faststart"],
    ) as writer:
        for row in poses:
            data.qpos[:] = csv_pose_to_mujoco(row)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.xpos[pelvis_id]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    renderer.close()

    return {
        "clip": csv_path.stem,
        "source": str(csv_path),
        "video": str(output_path),
        "frames": int(poses.shape[0]),
        "fps": fps,
        "duration_seconds": poses.shape[0] / fps,
        "resolution": "640x480",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lafan1-fall-getup-g1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="Render at most this many source frames per clip (smoke tests).")
    args = parser.parse_args()

    sources = sorted((args.dataset / "g1").glob("fallAndGetUp*.csv"))
    if len(sources) != 6:
        raise RuntimeError(f"Expected 6 fall/get-up clips, found {len(sources)}")

    results = []
    for source in sources:
        render_source = source
        if args.limit:
            data = np.loadtxt(source, delimiter=",", dtype=np.float64, max_rows=args.limit)
            temporary = args.output_dir / f".{source.stem}.smoke.csv"
            temporary.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(temporary, data, delimiter=",")
            render_source = temporary
        output = args.output_dir / f"{source.stem}_g1_1x.mp4"
        result = render_clip(render_source, output, args.scene, args.fps)
        result["clip"] = source.stem
        result["source"] = str(source)
        results.append(result)
        if render_source != source:
            render_source.unlink()
        print(f"rendered {output} ({result['frames']} frames)", flush=True)

    manifest = {
        "robot": "Unitree G1 29-DOF",
        "playback_speed": "1x",
        "source_fps": args.fps,
        "clips": results,
        "total_frames": sum(int(item["frames"]) for item in results),
        "total_duration_seconds": sum(float(item["duration_seconds"]) for item in results),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
