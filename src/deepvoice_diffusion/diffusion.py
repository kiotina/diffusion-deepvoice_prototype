# diffusion noise와 loss 계산
"""DDPM forward 과정과 실제 음성 위치만 사용하는 손실."""
from __future__ import annotations

import hashlib

import torch
from torch import nn


class DiffusionSchedule(nn.Module):
    """timestep별 noise 양을 보관하고 원본 Mel에 noise를 섞는다."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        """beta에서 단계별 누적 원본 비율 alpha_bar를 계산해 저장한다."""
        super().__init__()
        if timesteps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("Require timesteps >= 2 and 0 < beta_start < beta_end < 1")
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        # alpha_bar[t]는 원본 Mel이 t단계까지 얼마나 남는지 나타낸다.
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    def add_noise(self, clean, timesteps, noise, mask):
        """각 Mel에 선택된 단계의 noise를 섞고 padding 위치는 -1로 유지한다."""
        if noise.shape != clean.shape or timesteps.shape != (clean.shape[0],):
            raise ValueError("Noise or timestep shape does not match batch")
        alpha = self.alpha_bars[timesteps][:, None, None, None]
        # 원본과 실제 noise를 정해진 비율로 섞는다. U-Net이 맞힐 정답은 noise다.
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
    """유효한 Mel 위치의 제곱오차 합을 유효 원소 수로 나눈다."""
    error, count = masked_error(prediction, target, mask)
    return error / count


def masked_mse_per_sample(prediction, target, mask):
    """배치의 각 구간에서 유효 Mel 원소만 평균낸 MSE를 반환한다."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Prediction and target must have identical 4D shapes")
    expected = (prediction.shape[0], 1, 1, prediction.shape[-1])
    if tuple(mask.shape) != expected:
        raise ValueError(f"Mask shape must be {expected}")
    expanded = mask.expand_as(prediction)
    counts = expanded.sum(dim=(1, 2, 3))
    if torch.any(counts <= 0):
        raise ValueError("Each sample requires valid frames")
    errors = torch.where(expanded.bool(), prediction - target, 0.0).square()
    return errors.sum(dim=(1, 2, 3)) / counts


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
