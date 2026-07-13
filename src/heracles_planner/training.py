from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from .config import HeraclesConfig
from .data import MotionWindowDataset
from .model import HeraclesPlanner, conditional_flow_sample
from .rotations import perturb_rot6d


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def _prepare_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: HeraclesConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state, residual, duration = (value.to(device=device, dtype=torch.float32) for value in batch)
    state[:, : config.joint_dim] += (
        torch.randn(
            state[:, : config.joint_dim].shape,
            device=device,
            dtype=state.dtype,
            generator=generator,
        )
        * config.joint_noise_sigma
    )
    state[:, config.joint_dim :] = perturb_rot6d(
        state[:, config.joint_dim :], config.root_noise_sigma, generator
    )
    return (state - mean) / std, residual / std, duration


def _loss(
    model: HeraclesPlanner,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    weights: torch.Tensor,
    config: HeraclesConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    state, clean, duration = _prepare_batch(batch, device, mean, std, config, generator)
    path, flow_time, target = conditional_flow_sample(clean, generator)
    predicted = model(path, state, flow_time, duration)
    error = (predicted - target).square() * weights[None, None]
    return error[:, 1:].mean()


def benchmark_training(
    data_root: Path,
    output: Path,
    steps: int,
    config: HeraclesConfig,
) -> dict[str, float | int | str | bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("the duration gate must be benchmarked on the RTX 3090 CUDA device")
    seed_everything(config.seed)
    device = torch.device("cuda")
    dataset = MotionWindowDataset(data_root, "train", config)
    validation_dataset = MotionWindowDataset(data_root, "validation", config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    stats = np.load(data_root / "normalization.npz")
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)
    weights = torch.from_numpy(stats["loss_weights"]).to(device)
    model = HeraclesPlanner(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device=device).manual_seed(config.seed)
    iterator = iter(loader)
    warmup = min(5, steps)
    timings = []
    for step in range(steps + warmup):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, batch, device, mean, std, weights, config, generator)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if step >= warmup:
            timings.append(elapsed)
    seconds_per_step = float(np.median(timings))
    steps_per_epoch = math.ceil(len(dataset) / config.batch_size)
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    validation_iterator = iter(validation_loader)
    validation_timings = []
    model.eval()
    for step in range(steps + warmup):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            batch = next(validation_iterator)
        except StopIteration:
            validation_iterator = iter(validation_loader)
            batch = next(validation_iterator)
        with torch.no_grad():
            _loss(model, batch, device, mean, std, weights, config, generator)
        torch.cuda.synchronize()
        if step >= warmup:
            validation_timings.append(time.perf_counter() - started)
    validation_seconds_per_step = float(np.median(validation_timings))
    validation_steps_per_epoch = math.ceil(len(validation_dataset) / config.batch_size)
    checkpoint_path = output.with_suffix(".benchmark-checkpoint.pt")
    checkpoint_timings = []
    for _ in range(3):
        started = time.perf_counter()
        torch.save({"model": model.state_dict()}, checkpoint_path)
        checkpoint_timings.append(time.perf_counter() - started)
    checkpoint_path.unlink()
    checkpoint_seconds = float(np.median(checkpoint_timings))
    estimated_training_hours = seconds_per_step * steps_per_epoch * config.epochs / 3600.0
    estimated_validation_hours = (
        validation_seconds_per_step * validation_steps_per_epoch * config.epochs / 3600.0
    )
    estimated_checkpoint_hours = checkpoint_seconds * 2 * config.epochs / 3600.0
    estimated_hours = (
        estimated_training_hours + estimated_validation_hours + estimated_checkpoint_hours
    )
    report: dict[str, float | int | str | bool] = {
        "device": torch.cuda.get_device_name(device),
        "parameter_count": model.parameter_count,
        "training_windows": len(dataset),
        "validation_windows": len(validation_dataset),
        "steps_per_epoch": steps_per_epoch,
        "validation_steps_per_epoch": validation_steps_per_epoch,
        "benchmark_steps": steps,
        "timing_scope": "batch construction + forward + backward + clipping + optimizer",
        "median_end_to_end_seconds_per_step": seconds_per_step,
        "median_validation_seconds_per_step": validation_seconds_per_step,
        "median_checkpoint_seconds": checkpoint_seconds,
        "checkpoint_assumption": "two checkpoint writes per epoch (conservative)",
        "estimated_training_hours": estimated_training_hours,
        "estimated_validation_hours": estimated_validation_hours,
        "estimated_checkpoint_hours": estimated_checkpoint_hours,
        "estimated_4000_epoch_hours": estimated_hours,
        "estimate_includes_validation_and_checkpoint_io": True,
        "limit_hours": config.max_full_training_hours,
        "full_training_allowed": estimated_hours <= config.max_full_training_hours,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def train(
    data_root: Path,
    output: Path,
    duration_report: Path,
    config: HeraclesConfig,
) -> None:
    gate = json.loads(duration_report.read_text())
    if not gate.get("full_training_allowed", False):
        raise RuntimeError(
            f"full training blocked: estimate {gate['estimated_4000_epoch_hours']:.1f} h exceeds "
            f"{gate['limit_hours']:.1f} h; human direction is required"
        )
    seed_everything(config.seed)
    device = torch.device("cuda")
    train_data = MotionWindowDataset(data_root, "train", config)
    validation_data = MotionWindowDataset(data_root, "validation", config)
    stats = np.load(data_root / "normalization.npz")
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)
    weights = torch.from_numpy(stats["loss_weights"]).to(device)
    model = HeraclesPlanner(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = config.epochs * math.ceil(len(train_data) / config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    best = float("inf")
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    last_path = output / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_validation_loss"])
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs):
        train_data.set_epoch(epoch)
        loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            generator=torch.Generator().manual_seed(config.seed + epoch),
        )
        train_generator = torch.Generator(device=device).manual_seed(config.seed + epoch * 2)
        model.train()
        training_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch, device, mean, std, weights, config, train_generator)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            training_losses.append(loss.item())
        model.eval()
        validation = DataLoader(
            validation_data, batch_size=config.batch_size, shuffle=False, num_workers=2
        )
        validation_generator = torch.Generator(device=device).manual_seed(
            config.seed + epoch * 2 + 1
        )
        with torch.no_grad():
            losses = [
                _loss(
                    model,
                    batch,
                    device,
                    mean,
                    std,
                    weights,
                    config,
                    validation_generator,
                ).item()
                for batch in validation
            ]
        score = float(np.mean(losses))
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "validation_loss": score,
            "best_validation_loss": min(best, score),
        }
        torch.save(checkpoint, last_path)
        if score < best:
            best = score
            torch.save(checkpoint, output / "best.pt")
        elapsed = time.monotonic() - started
        completed_here = epoch - start_epoch + 1
        remaining = config.epochs - epoch - 1
        status = {
            "epoch": epoch,
            "epochs": config.epochs,
            "resumed_from_epoch": start_epoch,
            "training_loss": float(np.mean(training_losses)),
            "validation_loss": score,
            "best_validation_loss": best,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_s_this_run": elapsed,
            "estimated_remaining_s": elapsed / completed_here * remaining,
        }
        (output / "training_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)
