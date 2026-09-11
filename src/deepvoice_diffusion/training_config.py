from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainingConfig:
    manifest_path: str = "data/processed/real_mels_2s/manifest.csv"
    output_dir: str = "artifacts/training"
    base_channels: int = 32
    time_dim: int = 128
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 5
    gradient_clip: float = 1.0
    seed: int = 42
    validation_seed: int = 10042
    device: str = "auto"
    cpu_threads: int = 4
    checkpoint_every: int = 50
    # Smoke 실행은 작은 subset으로만 학습한다. 기본 전체 실행에는 제한이 없다.
    train_limit: int | None = None
    validation_limit: int | None = None

    def __post_init__(self):
        for name in ("base_channels", "time_dim", "timesteps", "batch_size", "epochs",
                     "patience", "cpu_threads", "checkpoint_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "validation_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be an integer in [0, 2**63)")
        for name in ("train_limit", "validation_limit"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be None or a positive integer")
        for name in ("learning_rate", "gradient_clip", "beta_start", "beta_end"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not self.beta_start < self.beta_end < 1 or self.timesteps < 2:
            raise ValueError("Invalid diffusion schedule")
        if self.base_channels % 8 or self.time_dim < 4 or self.time_dim % 2:
            raise ValueError("Channels must be divisible by 8; time_dim must be even >= 4")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_training_config(path: str | Path) -> TrainingConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Training configuration must be a mapping")
    unknown = set(raw) - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    return TrainingConfig(**raw)
