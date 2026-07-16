from __future__ import annotations

import math

import torch
from torch import nn

from .config import HeraclesConfig


def sinusoidal_embedding(value: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=value.device, dtype=value.dtype)
        / max(half - 1, 1)
    )
    phase = value.reshape(-1, 1) * frequency.reshape(1, -1)
    embedding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
    if width % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class AdaLNBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(approximate="none"),
            nn.Linear(width * mlp_ratio, width),
        )
        self.modulation = nn.Linear(width, width * 6)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    @staticmethod
    def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return value * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(
            6, dim=-1
        )
        normalized = self._modulate(self.attention_norm(value), shift_a, scale_a)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        value = value + gate_a[:, None] * attended
        normalized = self._modulate(self.mlp_norm(value), shift_m, scale_m)
        return value + gate_m[:, None] * self.mlp(normalized)


class HeraclesPlanner(nn.Module):
    def __init__(self, config: HeraclesConfig | None = None):
        super().__init__()
        if config is None:
            config = HeraclesConfig()
        self.config = config
        self.input_projection = nn.Linear(config.state_dim, config.width)
        self.state_projection = nn.Linear(config.state_dim, config.width)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.width, config.width),
            nn.GELU(approximate="none"),
            nn.Linear(config.width, config.width),
        )
        self.blocks = nn.ModuleList(
            AdaLNBlock(config.width, config.heads, config.mlp_ratio) for _ in range(config.blocks)
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.output_projection = nn.Linear(config.width, config.state_dim)
        position = sinusoidal_embedding(
            torch.arange(config.keyframes, dtype=torch.float32), config.width
        )
        self.register_buffer("position_embedding", position, persistent=True)

    def forward(
        self,
        trajectory: torch.Tensor,
        state: torch.Tensor,
        flow_time: torch.Tensor,
        segment_duration: torch.Tensor,
    ) -> torch.Tensor:
        value = self.input_projection(trajectory) + self.position_embedding[None]
        condition = self.state_projection(state)
        condition = condition + self.time_mlp(sinusoidal_embedding(flow_time, self.config.width))
        condition = condition + sinusoidal_embedding(segment_duration, self.config.width)
        for block in self.blocks:
            value = block(value, condition)
        velocity = self.output_projection(self.output_norm(value))
        velocity = torch.cat((torch.zeros_like(velocity[:, :1]), velocity[:, 1:]), dim=1)
        return velocity

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def conditional_flow_sample(
    clean: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper linear path: x_t=(1-t)x_data+t*x_noise, target dx/dt=noise-data."""
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    noise[:, 0] = 0.0
    time = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype, generator=generator)
    path = (1.0 - time[:, None, None]) * clean + time[:, None, None] * noise
    target = noise - clean
    target[:, 0] = 0.0
    return path, time, target
