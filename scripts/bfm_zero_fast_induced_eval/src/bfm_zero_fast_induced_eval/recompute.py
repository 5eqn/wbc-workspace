from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .providers import POLICY_TO_SDK


def verify_bfm_recomputation(round_path: Path, checkpoint: Path, samples: int = 8) -> dict[str, object]:
    """Recompute actor, F, D, QD, auxiliary Q, and next-state variants from stored frames."""
    import torch
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

    model = load_model_from_checkpoint_dir(checkpoint, device="cuda")
    model.eval()
    with h5py.File(round_path, "r") as handle:
        accepted = handle["state"].shape[0]
        if accepted == 0:
            raise ValueError("round has no accepted trajectories")
        rng = np.random.default_rng(0)
        episodes = rng.integers(0, accepted, size=samples)
        steps = rng.integers(0, 400, size=samples)
        keys = ("state", "privileged_state", "last_action", "history_actor")
        obs = {
            name: torch.as_tensor(
                np.stack(
                    [handle[name][episode, step] for episode, step in zip(episodes, steps, strict=False)]
                ),
                device="cuda",
            )
            for name in keys
        }
        next_obs = {
            name: torch.as_tensor(
                np.stack(
                    [handle[name][episode, step + 1] for episode, step in zip(episodes, steps, strict=False)]
                ),
                device="cuda",
            )
            for name in keys
        }
        stored_action = np.stack(
            [handle["action"][episode, step] for episode, step in zip(episodes, steps, strict=False)]
        )
        action = torch.as_tensor(stored_action, device="cuda")

    goal_path = round_path.parent / "manifest.json"
    import json

    z_value = np.asarray(json.loads(goal_path.read_text())["goal_z"], dtype=np.float32)
    z = torch.as_tensor(z_value, device="cuda").reshape(1, -1).repeat(samples, 1)
    outputs = {
        "actor": model.act(obs, z, mean=True),
        "F": model.forward_map(obs, z, action),
        "D": model.discriminator(obs, z),
        "QD": model.critic(obs, z, action),
        "Q": model.aux_critic(obs, z, action),
        "next_F": model.forward_map(next_obs, z, action),
        "next_D": model.discriminator(next_obs, z),
        "next_QD": model.critic(next_obs, z, action),
        "next_Q": model.aux_critic(next_obs, z, action),
    }
    shapes = {}
    for name, value in outputs.items():
        array = value.detach().float().cpu().numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"{name} recomputation contains non-finite values")
        shapes[name] = list(array.shape)
    actor_error = float(np.max(np.abs(outputs["actor"].detach().cpu().numpy() - stored_action)))
    if actor_error > 1e-3:
        raise ValueError(f"stored action does not match checkpoint actor: max abs error {actor_error}")
    return {"samples": samples, "output_shapes": shapes, "actor_max_abs_error": actor_error}


def verify_amp_recomputation(round_path: Path, samples: int = 8) -> dict[str, object]:
    import json

    import onnxruntime as ort
    import torch

    manifest = json.loads((round_path.parent / "manifest.json").read_text())
    actor_path = Path(manifest["paths"]["dynamic_actor_cache"])
    torchscript_path = Path(manifest["paths"]["torchscript"])
    with h5py.File(round_path, "r") as handle:
        accepted = handle["provider_input"].shape[0]
        if accepted == 0:
            raise ValueError("round has no accepted trajectories")
        rng = np.random.default_rng(0)
        episodes = rng.integers(0, accepted, size=samples)
        steps = rng.integers(0, 400, size=samples)
        model_input = np.stack(
            [handle["provider_input"][episode, step] for episode, step in zip(episodes, steps, strict=True)]
        )
        stored = np.stack(
            [handle["action"][episode, step] for episode, step in zip(episodes, steps, strict=True)]
        )
    ort.preload_dlls()
    session = ort.InferenceSession(
        str(actor_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    onnx_action = session.run(None, {session.get_inputs()[0].name: model_input})[0]
    stored_error = float(np.max(np.abs(onnx_action[:, POLICY_TO_SDK] - stored)))
    # Collection uses batch 128 while sampled verification uses a smaller CUDA GEMM kernel.
    if stored_error > 2e-3:
        raise ValueError(f"stored AMP action does not match ONNX: max abs error {stored_error}")
    module = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    with torch.inference_mode():
        torch_action = module(torch.from_numpy(model_input)).detach().cpu().numpy()
    parity_error = float(np.max(np.abs(torch_action - onnx_action)))
    if parity_error > 1e-5:
        raise ValueError(f"AMP TorchScript/ONNX parity failed: max abs error {parity_error}")
    return {
        "model_provider": "legged-lab-amp",
        "samples": samples,
        "stored_onnx_max_abs_error": stored_error,
        "torchscript_onnx_max_abs_error": parity_error,
        "output_shape": list(onnx_action.shape),
    }


def verify_recomputation(
    round_path: Path, checkpoint: Path | None = None, samples: int = 8
) -> dict[str, object]:
    import json

    manifest = json.loads((round_path.parent / "manifest.json").read_text())
    identity = manifest.get("model_provider", {}).get("identity", "bfm-zero")
    if identity == "legged-lab-amp":
        return verify_amp_recomputation(round_path, samples)
    if checkpoint is None:
        checkpoint = Path(manifest["paths"]["checkpoint"])
    return verify_bfm_recomputation(round_path, checkpoint, samples)
