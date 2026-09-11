"""DDPM forward 과정과 실제 음성 위치만 사용하는 손실."""
from __future__ import annotations

import hashlib

import torch
from torch import nn


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        if timesteps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("Require timesteps >= 2 and 0 < beta_start < beta_end < 1")
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    def add_noise(self, clean, timesteps, noise, mask):
        if noise.shape != clean.shape or timesteps.shape != (clean.shape[0],):
            raise ValueError("Noise or timestep shape does not match batch")
        alpha = self.alpha_bars[timesteps][:, None, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        # Mel의 -1은 정규화 범위의 바닥값이다. 빈 프레임은 항상 이 값으로 둔다.
        return torch.where(mask.bool(), noisy, -1.0)


def masked_error(prediction, target, mask):
    """평균 전의 오차 합/유효 원소 수를 반환해 epoch 전체를 정확히 집계한다."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have identical shapes")
    expected = (prediction.shape[0], 1, 1, prediction.shape[-1])
    if tuple(mask.shape) != expected:
        raise ValueError(f"Mask shape must be {expected}")
    expanded = mask.expand_as(prediction)
    count = expanded.sum()
    if count.item() <= 0:
        raise ValueError("Loss requires valid frames")
    error = torch.where(expanded.bool(), prediction - target, 0.0).square().sum()
    return error, count


def masked_mse(prediction, target, mask):
    error, count = masked_error(prediction, target, mask)
    return error / count


def seeded_noise(clean, indices, timesteps: int, seed: int, epoch: int):
    """샘플별 난수: 배치 크기/순서가 달라져도 검증 입력을 동일하게 만든다.

    학습에는 실제 epoch, 검증에는 고정 epoch=0을 전달한다.
    """
    steps, noises = [], []
    for index in indices:
        material = f"{seed}:{epoch}:{int(index)}".encode()
        value = int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63 - 1)
        generator = torch.Generator().manual_seed(value)
        steps.append(torch.randint(timesteps, (), generator=generator))
        noises.append(torch.randn(clean.shape[1:], generator=generator, dtype=clean.dtype))
    return torch.stack(steps).to(clean.device), torch.stack(noises).to(clean.device)
