from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from mujoco.egl import GLContext as EGLGLContext
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .config import HeraclesConfig
from .inference import TorchVelocityModel, load_torch_checkpoint
from .paths import REPO_ROOT
from .rotations import quat_xyzw_to_rot6d, rot6d_to_quat_xyzw

DEFAULT_DATA = REPO_ROOT / "logs/heracles-planner/data"
DEFAULT_MOTION_ROOT = REPO_ROOT / "assets/motions/sonic_motions"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "artifacts/heracles-planner/checkpoints"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/heracles-planner/debug-videos"
DEFAULT_SCENE = (
    REPO_ROOT
    / "thirdparties/GR00T-WholeBodyControl/motionbricks/assets/skeletons/g1/scene_29dof.xml"
)

BENCHMARK_MOTIONS = (
    "dance_chicken_c03_neutral2s",
    "dance_chicken_c04_neutral2s",
    "dance_heart111_c01_neutral2s",
    "dance_phony_c01_neutral2s",
    "dance_phony_c03_neutral2s",
    "dance_phony_c04_neutral2s",
    "forward_lunge_R_001__A359_M_neutral2s",
    "macarena_001__A545_M_neutral2s",
    "squat_001__A359_neutral2s",
    "walking_quip_360_R_002__A428_neutral2s",
)
GET_UP_MOTIONS = (
    "fallAndGetUp1_subject1",
    "fallAndGetUp1_subject4",
    "fallAndGetUp1_subject5",
    "fallAndGetUp2_subject2",
    "fallAndGetUp2_subject3",
    "fallAndGetUp3_subject1",
)
FEATURED_MOTIONS = (
    "dance_phony_c01_neutral2s",
    "squat_001__A359_neutral2s",
    "fallAndGetUp1_subject1",
    "fallAndGetUp1_subject4",
)
LAGS = (2, 5, 15, 50)
FUTURE_FRAMES = (2, 5, 10)
GET_UP_FRAMES = 601
FPS = 50
SEED = 42

# q in the benchmark CSV is in SONIC policy order. The planner and render model use hardware order.
HARDWARE_FROM_SONIC_POLICY = np.asarray(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class MotionReference:
    name: str
    joint_pos: np.ndarray
    root_quat_xyzw: np.ndarray
    root_pos: np.ndarray
    source_paths: tuple[Path, ...]
    kind: str

    @property
    def frames(self) -> int:
        return len(self.joint_pos)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_csv(path: Path) -> np.ndarray:
    value = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"{path}: expected a 2D CSV, got {value.shape}")
    return value


def load_reference(name: str, data_root: Path, motion_root: Path) -> MotionReference:
    if name in BENCHMARK_MOTIONS:
        root = motion_root / name
        joint_path = root / "joint_pos.csv"
        pos_path = root / "body_pos.csv"
        quat_path = root / "body_quat.csv"
        joint = _load_csv(joint_path)
        position = _load_csv(pos_path)
        quat_wxyz = _load_csv(quat_path)
        if joint.shape[1] != 29 or position.shape[1] < 3 or quat_wxyz.shape[1] < 4:
            raise ValueError(f"{name}: invalid benchmark reference shapes")
        if not (len(joint) == len(position) == len(quat_wxyz)):
            raise ValueError(f"{name}: inconsistent benchmark reference lengths")
        return MotionReference(
            name=name,
            joint_pos=joint[:, HARDWARE_FROM_SONIC_POLICY],
            root_quat_xyzw=quat_wxyz[:, [1, 2, 3, 0]],
            root_pos=position[:, :3],
            source_paths=(joint_path, pos_path, quat_path),
            kind="benchmark",
        )
    if name in GET_UP_MOTIONS:
        path = data_root / f"{name}.npz"
        with np.load(path) as data:
            if len(data["joint_pos"]) < GET_UP_FRAMES:
                raise ValueError(f"{path}: fewer than {GET_UP_FRAMES} frames")
            return MotionReference(
                name=name,
                joint_pos=data["joint_pos"][:GET_UP_FRAMES].astype(np.float32),
                root_quat_xyzw=data["root_quat_xyzw"][:GET_UP_FRAMES].astype(np.float32),
                root_pos=data["root_pos"][:GET_UP_FRAMES].astype(np.float32),
                source_paths=(path,),
                kind="get_up_12s",
            )
    raise ValueError(f"unknown featured motion: {name}")


def target_prefix_indices(frame: int, lag: int, frames: int, prefix_frames: int = 11) -> np.ndarray:
    return np.minimum(np.arange(frame + lag, frame + lag + prefix_frames), frames - 1)


def deterministic_noise(motion: str, frame: int, lag: int, config: HeraclesConfig) -> np.ndarray:
    key = f"{SEED}\0{motion}\0{frame}\0{lag}".encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    noise = np.random.default_rng(local_seed).standard_normal(
        (config.keyframes, config.state_dim), dtype=np.float32
    )
    noise[0] = 0.0
    return noise


def _warm_residual(
    current_state: np.ndarray,
    prefix_joint: np.ndarray,
    prefix_root: np.ndarray,
    config: HeraclesConfig,
) -> np.ndarray:
    frame_axis = np.arange(len(prefix_joint), dtype=np.float64)
    key_axis = np.linspace(0.0, len(prefix_joint) - 1, config.keyframes)
    warm_joint = CubicSpline(frame_axis, prefix_joint, axis=0)(key_axis)
    warm_root = Slerp(frame_axis, Rotation.from_quat(prefix_root))(key_axis).as_quat()
    warm_state = np.concatenate((warm_joint, quat_xyzw_to_rot6d(warm_root)), axis=-1)
    residual = (warm_state - current_state).astype(np.float32)
    residual[0] = 0.0
    return residual


def _interpolate_debug_frames(
    keyframes: np.ndarray, config: HeraclesConfig
) -> tuple[np.ndarray, np.ndarray]:
    key_time = np.linspace(0.0, config.horizon_s, config.keyframes)
    frame_time = np.asarray(FUTURE_FRAMES, dtype=np.float64) / config.data_hz
    joints = CubicSpline(key_time, keyframes[:, : config.joint_dim], axis=0)(frame_time)
    quat = rot6d_to_quat_xyzw(keyframes[:, config.joint_dim :])
    roots = Slerp(key_time, Rotation.from_quat(quat))(frame_time).as_quat()
    return joints.astype(np.float32), roots.astype(np.float32)


def predict_reference(
    checkpoint: Path,
    normalization: Path,
    reference: MotionReference,
    batch_frames: int = 32,
    limit_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("debug-video inference requires CUDA")
    device = torch.device("cuda")
    model = load_torch_checkpoint(checkpoint, device)
    config = model.config
    velocity_model = TorchVelocityModel(model, device)
    with np.load(normalization) as stats:
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)
    frames = reference.frames if limit_frames <= 0 else min(reference.frames, limit_frames)
    generated_joint = np.empty((frames, len(LAGS), len(FUTURE_FRAMES), 29), dtype=np.float32)
    generated_root = np.empty((frames, len(LAGS), len(FUTURE_FRAMES), 4), dtype=np.float32)

    for first in range(0, frames, batch_frames):
        last = min(frames, first + batch_frames)
        samples: list[tuple[int, int]] = [
            (frame, lag) for frame in range(first, last) for lag in LAGS
        ]
        states = []
        warm = []
        noise = []
        for frame, lag in samples:
            state = np.concatenate(
                (
                    reference.joint_pos[frame],
                    quat_xyzw_to_rot6d(reference.root_quat_xyzw[frame : frame + 1])[0],
                )
            ).astype(np.float32)
            indices = target_prefix_indices(frame, lag, reference.frames)
            states.append(state)
            warm.append(
                _warm_residual(
                    state,
                    reference.joint_pos[indices],
                    reference.root_quat_xyzw[indices],
                    config,
                )
            )
            noise.append(deterministic_noise(reference.name, frame, lag, config))
        state_batch = np.stack(states)
        warm_batch = np.stack(warm) / std
        value = config.warm_start_t * np.stack(noise) + (1.0 - config.warm_start_t) * warm_batch
        normalized_state = (state_batch - mean) / std
        duration = np.full(len(samples), config.horizon_s, dtype=np.float32)
        times = np.linspace(config.warm_start_t, 0.0, config.euler_steps + 1, dtype=np.float32)
        for current, following in zip(times[:-1], times[1:], strict=True):
            flow_time = np.full(len(samples), current, dtype=np.float32)
            velocity = velocity_model(
                value.astype(np.float32),
                normalized_state.astype(np.float32),
                flow_time,
                duration,
            )
            value += np.float32(following - current) * velocity
            value[:, 0] = 0.0
        residual = value * std
        keyframes = residual + state_batch[:, None]
        rotation_shape = keyframes[:, :, config.joint_dim :].shape
        projected_quat = rot6d_to_quat_xyzw(
            keyframes[:, :, config.joint_dim :].reshape(-1, config.rotation_dim)
        )
        keyframes[:, :, config.joint_dim :] = quat_xyzw_to_rot6d(projected_quat).reshape(
            rotation_shape
        )
        for sample_index, (frame, lag) in enumerate(samples):
            lag_index = LAGS.index(lag)
            joint, root = _interpolate_debug_frames(keyframes[sample_index], config)
            generated_joint[frame, lag_index] = joint
            generated_root[frame, lag_index] = root

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "epoch": int(checkpoint_data["epoch"]),
        "validation_loss": float(checkpoint_data["validation_loss"]),
        "frames": frames,
        "batch_frames": batch_frames,
        "seed": SEED,
        "lags": list(LAGS),
        "future_frames": list(FUTURE_FRAMES),
    }
    return generated_joint, generated_root, metadata


class PoseRenderer:
    tile_width = 256
    tile_height = 192
    header_height = 32
    columns = 5
    rows = 4

    def __init__(self, scene: Path):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        if self.model.nq != 36:
            raise ValueError(f"{scene}: expected nq=36, got {self.model.nq}")
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.tile_width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.tile_height)
        self.data = mujoco.MjData(self.model)
        self.gl_context = EGLGLContext(self.tile_width, self.tile_height)
        self.gl_context.make_current()
        self.renderer = mujoco.Renderer(
            self.model, width=self.tile_width, height=self.tile_height
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.75]
        self.camera.distance = 2.7
        self.camera.azimuth = 135.0
        self.camera.elevation = -18.0

    @property
    def width(self) -> int:
        return self.tile_width * self.columns

    @property
    def height(self) -> int:
        return (self.tile_height + self.header_height) * self.rows

    def render(
        self, root_pos: np.ndarray, root_quat_xyzw: np.ndarray, joints: np.ndarray
    ) -> np.ndarray:
        self.data.qpos[:3] = [0.0, 0.0, float(root_pos[2])]
        self.data.qpos[3:7] = root_quat_xyzw[[3, 0, 1, 2]]
        self.data.qpos[7:] = joints
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()
        self.gl_context.free()


def _ffmpeg_filter(checkpoint_label: str) -> str:
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    tile_w = PoseRenderer.tile_width
    row_h = PoseRenderer.tile_height + PoseRenderer.header_height
    labels = ("CURRENT T=%{n}", "APPLY +2", "INSPECT +5", "INSPECT +10")
    colors = ("#303030", "#176b3a", "#244a73", "#244a73", "#8a551e")
    filters = []
    for row, lag in enumerate(LAGS):
        for column in range(PoseRenderer.columns):
            x = column * tile_w
            y = row * row_h
            label = labels[column] if column < len(labels) else f"TARGET T+{lag}"
            if column == 0:
                label = f"{checkpoint_label}  L={lag}  {label}"
            filters.append(
                f"drawbox=x={x}:y={y}:w={tile_w}:h={PoseRenderer.header_height}:"
                f"color={colors[column]}:t=fill"
            )
            filters.append(
                f"drawtext=fontfile={font}:text='{label}':x={x + 7}:y={y + 8}:"
                "fontsize=13:fontcolor=white"
            )
    return ",".join(filters)


def render_video(
    output: Path,
    checkpoint_label: str,
    reference: MotionReference,
    generated_joint: np.ndarray,
    generated_root: np.ndarray,
    scene: Path,
) -> None:
    renderer = PoseRenderer(scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{renderer.width}x{renderer.height}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-vf",
        _ffmpeg_filter(checkpoint_label),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "21",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    frames = len(generated_joint)
    try:
        for frame in range(frames):
            canvas = np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)
            current = renderer.render(
                reference.root_pos[frame],
                reference.root_quat_xyzw[frame],
                reference.joint_pos[frame],
            )
            for lag_index, lag in enumerate(LAGS):
                y = lag_index * (renderer.tile_height + renderer.header_height)
                image_y = y + renderer.header_height
                canvas[image_y : image_y + renderer.tile_height, : renderer.tile_width] = current
                for future_index in range(len(FUTURE_FRAMES)):
                    future = renderer.render(
                        reference.root_pos[frame],
                        generated_root[frame, lag_index, future_index],
                        generated_joint[frame, lag_index, future_index],
                    )
                    x = (future_index + 1) * renderer.tile_width
                    canvas[
                        image_y : image_y + renderer.tile_height,
                        x : x + renderer.tile_width,
                    ] = future
                target = min(frame + lag, reference.frames - 1)
                target_image = renderer.render(
                    reference.root_pos[target],
                    reference.root_quat_xyzw[target],
                    reference.joint_pos[target],
                )
                x = (renderer.columns - 1) * renderer.tile_width
                canvas[
                    image_y : image_y + renderer.tile_height,
                    x : x + renderer.tile_width,
                ] = target_image
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        return_code = process.wait()
        renderer.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {output}")


def _probe_video(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
        "decoded_frames": int(stream["nb_read_frames"]),
        "sha256": _sha256(path),
    }


def _write_index(output: Path, rows: list[dict[str, object]]) -> None:
    by_motion: dict[str, dict[str, str]] = {}
    for row in rows:
        relative = Path(str(row["video"])).relative_to(output)
        by_motion.setdefault(str(row["motion"]), {})[str(row["checkpoint_label"])] = str(relative)
    table = []
    default_order = {motion: index for index, motion in enumerate(FEATURED_MOTIONS)}
    for motion in sorted(by_motion, key=lambda name: (default_order.get(name, 1_000), name)):
        videos = by_motion.get(motion, {})
        cells = [f"<td><strong>{html.escape(motion)}</strong></td>"]
        for label in ("best", "last"):
            if label in videos:
                src = html.escape(videos[label])
                cells.append(f'<td><video controls preload="metadata" src="{src}"></video></td>')
            else:
                cells.append("<td>not generated</td>")
        table.append("<tr>" + "".join(cells) + "</tr>")
    document = """<!doctype html>
<meta charset="utf-8">
<title>Heracles checkpoint debug videos</title>
<style>
body { font-family: sans-serif; margin: 1rem; background: #111; color: #eee; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #555; padding: .5rem; vertical-align: top; }
video { width: 100%; max-width: 720px; }
</style>
<h1>Heracles checkpoint debug videos</h1>
<p>Rows are target lags 2, 5, 15, and 50. Frame +2 would be applied;
frames +5 and +10 are inspection-only.</p>
<table><thead><tr><th>Motion</th><th>best.pt</th><th>last.pt</th></tr></thead><tbody>
""" + "\n".join(table) + "\n</tbody></table>\n"
    (output / "index.html").write_text(document)


def generate_debug_videos(
    checkpoints: list[Path],
    normalization: Path,
    data_root: Path,
    motion_root: Path,
    scene: Path,
    output: Path,
    motions: list[str] | None = None,
    limit_frames: int = 0,
    batch_frames: int = 32,
    overwrite: bool = False,
) -> dict[str, object]:
    selected = motions or list(FEATURED_MOTIONS)
    if len(set(selected)) != len(selected):
        raise ValueError("featured motion list contains duplicates")
    output.mkdir(parents=True, exist_ok=True)
    prior_rows: list[dict[str, object]] = []
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior_rows = list(json.loads(manifest_path.read_text()).get("videos", []))
    rows_by_key = {
        (str(row["checkpoint_label"]), str(row["motion"])): row for row in prior_rows
    }

    for checkpoint in checkpoints:
        label = checkpoint.stem
        if label not in {"best", "last"}:
            raise ValueError(f"checkpoint name must be best.pt or last.pt: {checkpoint}")
        for motion in selected:
            reference = load_reference(motion, data_root, motion_root)
            video_path = output / label / f"{motion}.mp4"
            prediction_path = output / label / f"{motion}.predictions.npz"
            if overwrite or not video_path.exists():
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                generated_joint, generated_root, checkpoint_metadata = predict_reference(
                    checkpoint,
                    normalization,
                    reference,
                    batch_frames=batch_frames,
                    limit_frames=limit_frames,
                )
                np.savez_compressed(
                    prediction_path,
                    generated_joint_pos=generated_joint,
                    generated_root_quat_xyzw=generated_root,
                    lags=np.asarray(LAGS, dtype=np.int64),
                    future_frames=np.asarray(FUTURE_FRAMES, dtype=np.int64),
                )
                render_video(
                    video_path,
                    label,
                    reference,
                    generated_joint,
                    generated_root,
                    scene,
                )
            else:
                checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
                checkpoint_metadata = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "epoch": int(checkpoint_data["epoch"]),
                    "validation_loss": float(checkpoint_data["validation_loss"]),
                    "frames": (
                        reference.frames
                        if limit_frames <= 0
                        else min(reference.frames, limit_frames)
                    ),
                    "batch_frames": batch_frames,
                    "seed": SEED,
                    "lags": list(LAGS),
                    "future_frames": list(FUTURE_FRAMES),
                }
            probe = _probe_video(video_path)
            expected_frames = int(checkpoint_metadata["frames"])
            if probe["decoded_frames"] != expected_frames or probe["fps"] != "50/1":
                raise RuntimeError(
                    f"{video_path}: expected {expected_frames} frames at 50 FPS, got {probe}"
                )
            source_hashes = {str(path): _sha256(path) for path in reference.source_paths}
            rows_by_key[(label, motion)] = {
                "checkpoint_label": label,
                "motion": motion,
                "kind": reference.kind,
                "source_frames": reference.frames,
                "video_frames": expected_frames,
                "video": str(video_path),
                "predictions": str(prediction_path),
                "source_sha256": source_hashes,
                **checkpoint_metadata,
                **probe,
            }
            rows = sorted(
                rows_by_key.values(),
                key=lambda row: (str(row["motion"]), str(row["checkpoint_label"])),
            )
            manifest = {
                "schema_version": 1,
                "description": "Offline four-lag Heracles checkpoint inspection",
                "fps": FPS,
                "lags": list(LAGS),
                "future_frames": list(FUTURE_FRAMES),
                "applied_future_frames": [2],
                "get_up_excerpt_frames": GET_UP_FRAMES,
                "noise_policy": (
                    "SHA256(seed=42, motion, source frame, lag); shared across checkpoints"
                ),
                "videos": rows,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            _write_index(output, rows)
            print(f"generated and verified {video_path} ({expected_frames} frames)", flush=True)
    return json.loads(manifest_path.read_text())
