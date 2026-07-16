from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HeraclesConfig:
    state_dim: int = 35
    joint_dim: int = 29
    rotation_dim: int = 6
    keyframes: int = 8
    horizon_s: float = 0.2
    replan_hz: int = 25
    data_hz: int = 50
    source_hz: int = 30
    width: int = 512
    blocks: int = 6
    heads: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.0
    window_stride_s: float = 0.04
    segment_min_s: float = 0.2
    segment_max_s: float = 2.0
    joint_noise_sigma: float = 0.05
    root_noise_sigma: float = 0.02
    jacobian_fd_step: float = 1e-3
    jacobian_weight_floor: float = 0.1
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 4000
    gradient_clip: float = 1.0
    warmup_steps: int = 0
    seed: int = 42
    warm_start_t: float = 0.9
    euler_steps: int = 5
    max_full_training_hours: float = 72.0

    @property
    def stride_frames(self) -> int:
        return round(self.window_stride_s * self.data_hz)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
