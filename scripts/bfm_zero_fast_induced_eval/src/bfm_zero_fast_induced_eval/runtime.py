from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DEFAULT_ACTOR,
    DEFAULT_CHECKPOINT,
    DEFAULT_GOALS,
    DEFAULT_MOTION,
    EXPECTED_ACTOR_SHA256,
    EXPECTED_MODEL_SHA256,
    ISAAC_PYTHON,
    OUTPUT_ROOT,
    REPO_ROOT,
    EvalConfig,
)
from .disturbance import attempt_seed, sample_velocity_disturbance
from .metrics import trial_metrics, wilson_interval
from .onnx_batch import patch_dynamic_batch, sha256_file
from .pose import POSE_CONTRACT, require_reset_identity, require_upright, xyzw_to_wxyz
from .providers import (
    AMP_DEFAULT_JOINT_POSITION,
    DEFAULT_AMP_ENV_CONFIG,
    DEFAULT_AMP_TORCHSCRIPT,
    POLICY_JOINT_NAMES,
    SDK_JOINT_NAMES,
    BfmZeroProvider,
    CanonicalSnapshot,
    LeggedLabAmpProvider,
    PolicyProvider,
    action_to_target,
)
from .schema import (
    REPLAY_STATE_FIELDS,
    SCHEMA_VERSION,
    estimated_raw_gib,
)
from .storage import read_attempt_summaries, valid_round, write_round_atomic


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def directory_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def checkpoint_identity_hash(path: Path) -> str:
    """Hash the inference identity without reading optimizer, replay, or debug payloads."""
    import hashlib

    digest = hashlib.sha256()
    for relative in ("config.json", "init_kwargs.json", "model/model.safetensors"):
        child = path / relative
        if child.is_file():
            digest.update(relative.encode())
            digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def conflicting_processes() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True)
    conflicts = []
    own_pid = os.getpid()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if int(pid_text) == own_pid:
            continue
        lower = command.lower()
        process_markers = (
            "cyclonedds",
            "unitree_sdk2",
            "g1_deploy.py",
            "deploy_policy.py",
            "bfm_zero_fallen_recovery_eval.py stage2",
        )
        if any(token in lower for token in process_markers):
            conflicts.append(stripped)
    return conflicts


def preflight(
    output_root: Path,
    config: EvalConfig,
    actor: Path,
    checkpoint: Path | None,
    goals: Path | None,
    *,
    allow_custom_model: bool = False,
    model_provider: str = "bfm-zero",
) -> dict[str, Any]:
    config.validate()
    if model_provider == "bfm-zero":
        required = (actor, checkpoint, goals, DEFAULT_MOTION, ISAAC_PYTHON)
    elif model_provider == "legged-lab-amp":
        required = (actor, DEFAULT_AMP_TORCHSCRIPT, DEFAULT_AMP_ENV_CONFIG, DEFAULT_MOTION, ISAAC_PYTHON)
    else:
        raise ValueError(f"unknown model provider {model_provider!r}")
    missing = [str(path) for path in required if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"required paths are missing: {missing}")
    output_root.mkdir(parents=True, exist_ok=True)
    usage = os.statvfs(output_root)
    free_gib = usage.f_bavail * usage.f_frsize / 2**30
    if free_gib < config.minimum_free_gib:
        raise RuntimeError(f"only {free_gib:.2f} GiB free; {config.minimum_free_gib:.2f} GiB is required")
    conflicts = conflicting_processes()
    if conflicts:
        raise RuntimeError("DDS/deployer processes are active:\n" + "\n".join(conflicts))
    actor_hash = sha256_file(actor)
    if model_provider == "legged-lab-amp":
        return {
            "free_gib": free_gib,
            "maximum_raw_gib": estimated_raw_gib(config.num_envs * config.rounds, {"provider_input": 570}),
            "actor_sha256": actor_hash,
            "torchscript_sha256": sha256_file(DEFAULT_AMP_TORCHSCRIPT),
            "env_config_sha256": sha256_file(DEFAULT_AMP_ENV_CONFIG),
            "model_policy": "legged-lab-amp-export",
            "python": platform.python_version(),
            "checked_at": now_iso(),
        }
    assert checkpoint is not None
    model_path = checkpoint / "model/model.safetensors"
    model_hash = sha256_file(model_path)
    known_actor = actor_hash == EXPECTED_ACTOR_SHA256
    known_checkpoint = model_hash == EXPECTED_MODEL_SHA256
    if not allow_custom_model and not known_actor:
        raise RuntimeError(f"actor hash {actor_hash} is not the known-good old actor")
    if not allow_custom_model and not known_checkpoint:
        raise RuntimeError(f"checkpoint model hash {model_hash} is not the known-good old checkpoint")
    known_pair = known_actor and known_checkpoint
    checkpoint_hash = directory_hash(checkpoint) if known_pair else checkpoint_identity_hash(checkpoint)
    return {
        "free_gib": free_gib,
        "maximum_raw_gib": estimated_raw_gib(config.num_envs * config.rounds),
        "actor_sha256": actor_hash,
        "checkpoint_model_sha256": model_hash,
        "checkpoint_sha256": checkpoint_hash,
        "model_policy": "known-good-old" if known_pair else "explicit-custom",
        "python": platform.python_version(),
        "checked_at": now_iso(),
    }


def reexec_in_isaac() -> None:
    # Both venv interpreters symlink to /usr/bin/python3.10; compare venv paths, not targets.
    if Path(sys.executable).absolute() == ISAAC_PYTHON.absolute():
        return
    if os.environ.get("BFM_ZERO_FAST_ISAAC_REEXEC") == "1":
        raise RuntimeError(f"failed to re-execute with Isaac Python at {ISAAC_PYTHON}")
    environment = os.environ.copy()
    package_src = Path(__file__).resolve().parents[1]
    project_root = package_src.parent
    project_site = project_root / ".venv/lib/python3.10/site-packages"
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(package_src),
            str(project_site),
            str(REPO_ROOT / "thirdparties/BFM-Zero"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    environment["BFM_ZERO_FAST_ISAAC_REEXEC"] = "1"
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    command = [str(ISAAC_PYTHON), "-m", "bfm_zero_fast_induced_eval.cli", *sys.argv[1:]]
    os.execve(str(ISAAC_PYTHON), command, environment)


def _versions(ort: Any, torch: Any, h5py: Any) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "h5py": h5py.__version__,
    }


def _load_goal(path: Path, key: str) -> np.ndarray:
    import joblib

    goals = joblib.load(path)
    if key not in goals:
        raise KeyError(f"goal {key!r} is absent from {path}")
    value = np.asarray(goals[key], dtype=np.float32).reshape(-1)
    if value.shape != (256,) or not np.isfinite(value).all():
        raise ValueError(f"invalid goal latent with shape {value.shape}")
    return value


def _make_environment(config: EvalConfig, model_provider: str = "bfm-zero"):
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

    def hydra_dict(values: dict[str, float]) -> str:
        return "{" + ",".join(f"{key}:{value}" for key, value in values.items()) + "}"

    overrides = [
        "env.config.headless=true",
        "simulator=isaacsim",
        "robot=g1/g1_29dof_hard_waist",
        "robot.control.action_scale=0.25",
        "obs.root_height_obs=true",
        "env.config.lie_down_init=false",
        "env.config.lie_down_init_prob=0.0",
    ]
    if model_provider == "bfm-zero":
        overrides += ["robot.control.action_clip_value=5.0", "robot.control.normalize_action_to=5.0"]
    else:
        default_angles = dict(zip(SDK_JOINT_NAMES, AMP_DEFAULT_JOINT_POSITION.tolist(), strict=True))
        effort = [
            88,
            139,
            88,
            139,
            25,
            25,
            88,
            139,
            88,
            139,
            25,
            25,
            88,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            5,
            5,
            25,
            25,
            25,
            25,
            25,
            5,
            5,
        ]
        velocity = [
            32,
            20,
            32,
            20,
            37,
            37,
            32,
            20,
            32,
            20,
            37,
            37,
            32,
            37,
            37,
            37,
            37,
            37,
            37,
            37,
            22,
            22,
            37,
            37,
            37,
            37,
            37,
            22,
            22,
        ]
        stiffness = {
            "hip_yaw": 100.0,
            "hip_roll": 100.0,
            "hip_pitch": 100.0,
            "knee": 150.0,
            "ankle_pitch": 40.0,
            "ankle_roll": 40.0,
            "waist_yaw": 200.0,
            "waist_roll": 40.0,
            "waist_pitch": 40.0,
            "shoulder_pitch": 40.0,
            "shoulder_roll": 40.0,
            "shoulder_yaw": 40.0,
            "elbow": 40.0,
            "wrist_roll": 40.0,
            "wrist_pitch": 40.0,
            "wrist_yaw": 40.0,
        }
        damping = {
            "hip_yaw": 2.0,
            "hip_roll": 2.0,
            "hip_pitch": 2.0,
            "knee": 4.0,
            "ankle_pitch": 2.0,
            "ankle_roll": 2.0,
            "waist_yaw": 5.0,
            "waist_roll": 5.0,
            "waist_pitch": 5.0,
            "shoulder_pitch": 1.0,
            "shoulder_roll": 1.0,
            "shoulder_yaw": 1.0,
            "elbow": 1.0,
            "wrist_roll": 1.0,
            "wrist_pitch": 1.0,
            "wrist_yaw": 1.0,
        }
        overrides += [
            "robot.control.action_clip_value=1000000.0",
            "robot.control.normalize_action=false",
            "robot.control.normalize_action_to=1000000.0",
            "robot.control.action_rescale=false",
            f"robot.init_state.default_joint_angles={hydra_dict(default_angles)}",
            f"robot.dof_effort_limit_list={effort}",
            f"robot.dof_vel_limit_list={velocity}",
            f"robot.dof_armature_list={[0.01] * 29}",
            f"robot.dof_joint_friction_list={[0.0] * 29}",
            f"robot.control.stiffness={hydra_dict(stiffness)}",
            f"robot.control.damping={hydra_dict(damping)}",
        ]
    env_config = HumanoidVerseIsaacConfig(
        device="cuda:0",
        lafan_tail_path=str(DEFAULT_MOTION),
        enable_cameras=False,
        max_episode_length_s=10_000.0,
        disable_obs_noise=True,
        disable_domain_randomization=True,
        include_last_action=True,
        include_history_actor=True,
        root_height_obs=True,
        hydra_overrides=overrides,
    )
    wrapped, _ = env_config.build(config.num_envs)
    base = wrapped.base_env
    termination = base.config.termination
    forbidden = [name for name, enabled in termination.items() if bool(enabled)]
    if forbidden:
        raise RuntimeError(f"automatic termination unexpectedly enabled: {forbidden}")
    if bool(base.config.simulator.get("enable_cameras", False)):
        raise RuntimeError("cameras unexpectedly enabled")
    return wrapped


def _snapshot(wrapped: Any) -> CanonicalSnapshot:
    qpos, qvel = wrapped._get_qpos_qvel(to_numpy=True)
    simulator = wrapped.base_env.simulator
    body_positions = simulator._rigid_body_pos.detach().cpu().numpy().copy()
    return CanonicalSnapshot(
        np.asarray(qpos, dtype=np.float32),
        np.asarray(qvel, dtype=np.float32),
        np.asarray(body_positions, dtype=np.float32),
        tuple(simulator.body_names),
    )


def _validate_amp_runtime(wrapped: Any) -> dict[str, Any]:
    base = wrapped.base_env
    actual_names = tuple(base.simulator.dof_names)
    if actual_names != SDK_JOINT_NAMES:
        raise RuntimeError(f"AMP joint order mismatch: {actual_names}")
    native_names = tuple(base.simulator._robot.joint_names)
    if native_names != POLICY_JOINT_NAMES:
        raise RuntimeError(f"AMP policy joint order mismatch: {native_names}")
    default = base.default_dof_pos[0].detach().cpu().numpy()
    np.testing.assert_allclose(default, AMP_DEFAULT_JOINT_POSITION, rtol=0, atol=1e-7)
    probe = np.linspace(-1, 1, 29, dtype=np.float32)
    expected = action_to_target(probe)
    resolved = default + float(base.config.robot.control.action_scale) * probe
    np.testing.assert_allclose(resolved, expected, rtol=0, atol=1e-7)
    return {
        "canonical_joint_order_match": True,
        "policy_joint_order_match": True,
        "default_pose_max_abs_error": float(np.max(np.abs(default - AMP_DEFAULT_JOINT_POSITION))),
        "target_mapping_probe_max_abs_error": float(np.max(np.abs(resolved - expected))),
        "supported_matches": [
            "200 Hz physics",
            "50 Hz control",
            "action scale 0.25",
            "no normalization",
            "no action rescale",
            "no effective clipping",
            "joint defaults",
            "effort/velocity limits",
            "armature",
            "self collision",
        ],
        "irreducible_differences": [
            "HumanoidVerse Isaac articulation fixes solver iterations at 4 position / 0 velocity; "
            "training used 8 / 4"
        ],
    }


def _refresh_observation(wrapped: Any) -> dict[str, Any]:
    base = wrapped.base_env
    base._refresh_sim_tensors()
    base._pre_compute_observations_callback()
    base._compute_observations()
    return wrapped._get_g1env_observation(to_numpy=True)


def _apply_velocity_kicks(
    wrapped: Any, disturbances: list[Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    base = wrapped.base_env
    root_states = base.simulator.robot_root_states.clone()
    linear = torch.tensor([item.linear_velocity_delta for item in disturbances], device=base.device)
    angular = torch.tensor([item.angular_velocity_delta for item in disturbances], device=base.device)
    root_states[:, 7:10] += linear
    root_states[:, 10:13] += angular
    root_states[:, 3:7] = xyzw_to_wxyz(root_states[:, 3:7])
    env_ids = torch.arange(base.num_envs, device=base.device)
    base.simulator.set_actor_root_state_tensor(env_ids, root_states)
    observation = _refresh_observation(wrapped)
    qpos, qvel = wrapped._get_qpos_qvel(to_numpy=True)
    return observation, {"qpos": qpos, "qvel": qvel}


def _run_one_round(
    wrapped: Any,
    session: Any,
    provider: PolicyProvider,
    config: EvalConfig,
    round_index: int,
    actor_hash: str,
    checkpoint_hash: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]], list[Any], float]:
    num_envs = config.num_envs
    target_states = dict(wrapped._default_pose_target_reset)
    root_states = target_states["root_states"].clone()
    root_states[..., 3:7] = xyzw_to_wxyz(root_states[..., 3:7])
    target_states["root_states"] = root_states
    observation, reset_state = wrapped.reset(to_numpy=True, target_states=target_states)
    require_reset_identity(reset_state["qpos"])
    snapshot = _snapshot(wrapped)
    provider.reset(observation, snapshot)

    # Let the controller settle in the known standing pose before starting the trial clock.
    for _ in range(round(config.warmup_s * config.policy_hz)):
        _, action = provider.infer(observation, snapshot, session)
        observation, _, _, _, _ = wrapped.step(action, to_numpy=True)
        snapshot = _snapshot(wrapped)
    warmup_qpos, _ = wrapped._get_qpos_qvel(to_numpy=True)
    require_upright(warmup_qpos, "post-warmup")

    first_attempt = round_index * num_envs
    disturbances = [
        sample_velocity_disturbance(attempt_seed(config.seed, first_attempt + env_index), config)
        for env_index in range(num_envs)
    ]
    observation, replay_state = _apply_velocity_kicks(wrapped, disturbances)
    snapshot = _snapshot(wrapped)
    require_upright(replay_state["qpos"], "post-kick")

    trajectories = {
        name: np.empty((num_envs, config.observation_frames, dim), dtype=np.float32)
        for name, dim in provider.observation_fields.items()
    }
    trajectories.update(
        {
            name: np.empty((num_envs, config.observation_frames, dim), dtype=np.float32)
            for name, dim in REPLAY_STATE_FIELDS.items()
        }
    )
    trajectories.update(
        action=np.empty((num_envs, config.policy_steps, 29), dtype=np.float32),
        terminated=np.empty((num_envs, config.policy_steps), dtype=np.bool_),
        truncated=np.empty((num_envs, config.policy_steps), dtype=np.bool_),
    )
    for name, dim in REPLAY_STATE_FIELDS.items():
        value = np.asarray(replay_state[name], dtype=np.float32)
        if value.shape != (num_envs, dim) or not np.isfinite(value).all():
            raise ValueError(f"invalid initial {name}: {value.shape}")
        trajectories[name][:, 0] = value

    physics_heights: list[np.ndarray] = []
    simulator = wrapped.base_env.simulator
    original_simulate = simulator.simulate_at_each_physics_step

    def sampled_simulate() -> None:
        original_simulate()
        physics_heights.append(simulator.robot_root_states[:, 2].detach().cpu().numpy().copy())

    simulator.simulate_at_each_physics_step = sampled_simulate
    started = time.perf_counter()
    try:
        for step in range(config.policy_steps):
            model_input, action = provider.infer(observation, snapshot, session)
            if action.shape != (num_envs, 29) or not np.isfinite(action).all():
                raise ValueError(f"invalid actor action at step {step}: {action.shape}")
            for name, value in provider.replay_frame(observation, model_input).items():
                trajectories[name][:, step] = value
            trajectories["action"][:, step] = action
            observation, _, terminated, truncated, info = wrapped.step(action, to_numpy=True)
            snapshot = _snapshot(wrapped)
            trajectories["terminated"][:, step] = terminated
            trajectories["truncated"][:, step] = truncated
            for name, dim in REPLAY_STATE_FIELDS.items():
                value = np.asarray(info[name], dtype=np.float32)
                if value.shape != (num_envs, dim) or not np.isfinite(value).all():
                    raise ValueError(f"invalid {name} at step {step}: {value.shape}")
                trajectories[name][:, step + 1] = value
        final_input, _ = provider.infer(observation, snapshot, session)
        for name, value in provider.replay_frame(observation, final_input).items():
            trajectories[name][:, -1] = value
    finally:
        simulator.simulate_at_each_physics_step = original_simulate
    elapsed = time.perf_counter() - started

    heights = np.stack(physics_heights, axis=1)
    if heights.shape != (num_envs, config.physics_steps):
        expected = (num_envs, config.physics_steps)
        raise ValueError(f"pelvis height samples have shape {heights.shape}, expected {expected}")
    metrics = [
        trial_metrics(
            heights[index],
            physics_hz=config.physics_hz,
            fallen_height_m=config.fallen_height_m,
            recovery_height_m=config.recovery_height_m,
        )
        for index in range(num_envs)
    ]
    accepted = np.asarray([item.induced_fall for item in metrics], dtype=np.bool_)
    attempts = []
    for env_index, disturbance in enumerate(disturbances):
        attempts.append(
            {
                "attempt_index": first_attempt + env_index,
                "round_index": round_index,
                "env_index": env_index,
                "seed": disturbance.seed,
                "goal_key": config.goal_key,
                "disturbance": disturbance.as_dict(),
                "actor_sha256": actor_hash,
                "checkpoint_sha256": checkpoint_hash,
                "model_provider": provider.name,
            }
        )
    return trajectories, accepted, attempts, metrics, elapsed


def _aggregate(
    run_dir: Path,
    config: EvalConfig,
    session_wall_time_s: float,
    schema_version: int = SCHEMA_VERSION,
    observation_fields: dict[str, int] | None = None,
) -> dict[str, Any]:
    rows = []
    completed = []
    cumulative_rollout_wall_time_s = 0.0
    for round_index in range(config.rounds):
        path = run_dir / f"round_{round_index:03d}.h5"
        if valid_round(
            path, round_index, schema_version=schema_version, observation_fields=observation_fields
        ):
            rows.extend(read_attempt_summaries(path))
            completed.append(round_index)
            import h5py

            with h5py.File(path, "r") as handle:
                cumulative_rollout_wall_time_s += float(handle.attrs.get("rollout_wall_time_s", 0.0))
    attempted = len(rows)
    accepted = sum(bool(row["accepted"]) for row in rows)
    recovered = sum(bool(row["recovered"]) for row in rows)
    low, high = wilson_interval(recovered, accepted)
    return {
        "attempted_count": attempted,
        "accepted_induced_fall_count": accepted,
        "discarded_count": attempted - accepted,
        "recovery_count": recovered,
        "conditional_recovery_rate": recovered / accepted if accepted else None,
        "wilson_95_interval": [low, high] if accepted else [None, None],
        "episodes_per_second": (
            attempted / cumulative_rollout_wall_time_s if cumulative_rollout_wall_time_s > 0 else None
        ),
        "wall_time_s": cumulative_rollout_wall_time_s,
        "session_wall_time_s": session_wall_time_s,
        "completed_rounds": completed,
        "complete": len(completed) == config.rounds,
        "updated_at": now_iso(),
    }


def run_evaluation(
    run_id: str,
    config: EvalConfig,
    actor: Path = DEFAULT_ACTOR,
    checkpoint: Path | None = DEFAULT_CHECKPOINT,
    goals: Path | None = DEFAULT_GOALS,
    output_root: Path = OUTPUT_ROOT,
    *,
    allow_custom_model: bool = False,
    model_provider: str = "bfm-zero",
) -> dict[str, Any]:
    reexec_in_isaac()
    import h5py
    import onnxruntime as ort
    import torch

    ort.preload_dlls()
    config.validate()
    run_dir = output_root / run_id
    preflight_result = preflight(
        output_root,
        config,
        actor,
        checkpoint,
        goals,
        allow_custom_model=allow_custom_model,
        model_provider=model_provider,
    )
    if model_provider == "bfm-zero":
        assert goals is not None
        goal = _load_goal(goals, config.goal_key)
        provider: PolicyProvider = BfmZeroProvider(goal)
    else:
        goal = None
        provider = LeggedLabAmpProvider()
    patched = patch_dynamic_batch(actor, output_root / ".onnx-cache", provider.input_dim, provider.output_dim)
    providers = ["CUDAExecutionProvider"]
    session = ort.InferenceSession(str(patched), providers=providers)
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA ONNX Runtime is required, got {session.get_providers()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    actor_hash = sha256_file(actor)
    checkpoint_hash = str(
        preflight_result.get("checkpoint_sha256") or preflight_result.get("torchscript_sha256")
    )
    wrapped = _make_environment(config, model_provider)
    physics_report = _validate_amp_runtime(wrapped) if model_provider == "legged-lab-amp" else None
    manifest = {
        "schema_version": provider.schema_version,
        "replay_schema": {
            "observation_frames": config.observation_frames,
            "transitions": config.policy_steps,
            "replay_hz": config.policy_hz,
            "metric_height_hz": config.physics_hz,
            "state_fields": dict(REPLAY_STATE_FIELDS),
            "provider_observation_fields": provider.observation_fields,
            "quaternion_order": "wxyz",
        },
        "pose_contract": POSE_CONTRACT,
        "run_id": run_id,
        "created_at": now_iso(),
        "config": config.as_dict(),
        "model_provider": provider.metadata(),
        "goal_z": goal.tolist() if goal is not None else None,
        "paths": {
            "actor": str(actor.resolve()),
            "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
            "goals": str(goals.resolve()) if goals is not None else None,
            "torchscript": str(DEFAULT_AMP_TORCHSCRIPT) if model_provider == "legged-lab-amp" else None,
            "environment_config": str(DEFAULT_AMP_ENV_CONFIG) if model_provider == "legged-lab-amp" else None,
            "dynamic_actor_cache": str(patched.resolve()),
        },
        "hashes": {
            "actor_sha256": actor_hash,
            "checkpoint_sha256": checkpoint_hash,
            "torchscript_sha256": preflight_result.get("torchscript_sha256"),
            "environment_config_sha256": preflight_result.get("env_config_sha256"),
        },
        "physics_compatibility": physics_report,
        "runtime_versions": _versions(ort, torch, h5py),
        "preflight": preflight_result,
        "headless": True,
        "cameras_enabled": False,
        "rendering_enabled": False,
        "dds_enabled": False,
        "real_time_pacing": False,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in (
            "schema_version",
            "replay_schema",
            "pose_contract",
            "config",
            "hashes",
            "model_provider",
            "physics_compatibility",
        ):
            if previous.get(key) != manifest[key]:
                raise RuntimeError(f"cannot resume: manifest {key} differs")
        manifest["created_at"] = previous["created_at"]
    write_json_atomic(manifest_path, manifest)

    started = time.perf_counter()
    try:
        for round_index in range(config.rounds):
            round_path = run_dir / f"round_{round_index:03d}.h5"
            temporary = round_path.with_name(round_path.name + ".tmp")
            if valid_round(
                round_path,
                round_index,
                schema_version=provider.schema_version,
                observation_fields=provider.observation_fields,
            ):
                continue
            temporary.unlink(missing_ok=True)
            trajectories, accepted, attempts, metrics, elapsed = _run_one_round(
                wrapped, session, provider, config, round_index, actor_hash, checkpoint_hash
            )
            write_round_atomic(
                round_path,
                round_index,
                trajectories,
                accepted,
                attempts,
                metrics,
                rollout_wall_time_s=elapsed,
                schema_version=provider.schema_version,
                observation_fields=provider.observation_fields,
            )
            round_summary = {
                "round_index": round_index,
                "attempted": config.num_envs,
                "accepted": int(accepted.sum()),
                "recovered": sum(item.recovered for item in metrics),
                "rollout_wall_time_s": elapsed,
                "episodes_per_second": config.num_envs / elapsed,
            }
            print(json.dumps(round_summary, sort_keys=True), flush=True)
            write_json_atomic(
                run_dir / "summary.json",
                _aggregate(
                    run_dir,
                    config,
                    time.perf_counter() - started,
                    provider.schema_version,
                    provider.observation_fields,
                ),
            )
    finally:
        wrapped.close()
    summary = _aggregate(
        run_dir, config, time.perf_counter() - started, provider.schema_version, provider.observation_fields
    )
    write_json_atomic(run_dir / "summary.json", summary)
    return summary
