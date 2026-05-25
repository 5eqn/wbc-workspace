#!/usr/bin/env python3
"""Render MuJoCo reconstruction videos for failed SONIC fall-probe attempts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmark_common import ROOT, event_sim_time, load_events, logged_scene_path, policy_motion_duration
from benchmark_report import (
    load_replay,
    nearest_replay_index,
    overlay_text,
    playback_event,
    render_replay_image,
    validate_video_file,
)


def render_failure(row: dict[str, str], output_dir: Path) -> Path:
    import imageio.v2 as imageio
    import mujoco

    run_root = Path(row["run_dir"])
    run_dir = run_root / "logs" / "sonic" / row["motion"]
    replay = load_replay(run_dir)
    events = load_events(run_dir)
    playback_t = event_sim_time(playback_event(events))
    scene = logged_scene_path(replay)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.azimuth = 135.0
    camera.elevation = -16.0

    motion_duration = policy_motion_duration("sonic", row["motion"])
    start_t = max(float(replay["sim_time_s"][0]), playback_t - 1.0)
    end_t = min(float(replay["sim_time_s"][-1]), playback_t + motion_duration)
    fps = 20
    frame_count = max(2, int((end_t - start_t) * fps) + 1)
    out = output_dir / f"fallprobe_attempt_{int(row['attempt']):05d}_{row['motion']}.mp4"
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame in range(frame_count):
            sim_t = start_t + frame / fps
            idx = nearest_replay_index(replay, sim_t)
            image = render_replay_image(mujoco, model, data, renderer, camera, replay["qpos"][idx])
            phase_t = float(replay["sim_time_s"][idx]) - playback_t
            support = "support" if int(replay["support_active"][idx]) else "released"
            lines = [
                f"SONIC fallprobe | attempt {row['attempt']} | {row['motion']}",
                f"phase={phase_t:+.2f}s sim={replay['sim_time_s'][idx]:.2f}s {support}",
                f"RMSE={float(row['rmse']):.3f} delay={float(row['delay']):.3f}s base_z_min={float(row['base_z_min']):.3f}m",
            ]
            writer.append_data(overlay_text(image, lines))
    renderer.close()
    validate_video_file(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_dir", nargs="?", default="/tmp/wbc-sonic-fallprobe")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    output_dir = Path(args.output_dir) if args.output_dir else probe_dir / "failure_videos"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = probe_dir / "failed_attempts.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest}: run fallprobe_sonic_fall_rate.sh first")
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))

    videos = []
    for row in rows:
        videos.append({"attempt": int(row["attempt"]), "motion": row["motion"], "path": str(render_failure(row, output_dir))})
    (output_dir / "fallprobe_failure_videos.json").write_text(json.dumps(videos, indent=2) + "\n")
    print(json.dumps({"ok": True, "videos": videos}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
