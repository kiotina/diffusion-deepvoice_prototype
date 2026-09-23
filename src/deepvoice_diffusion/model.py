# noise 예측 U-Net 모델
"""2초 Mel의 noise를 예측하는 작은 U-Net. 출력은 분류 확률이 아니다."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    """정수 timestep을 U-Net이 사용할 수 있는 특징 벡터로 바꾼다."""

    def __init__(self, dimension: int):
        """시간 특징의 차원과 학습 가능한 변환층을 준비한다."""
        super().__init__()
        half = dimension // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension))

    def forward(self, timesteps):
        """timestep마다 sin·cos 특징을 만든 뒤 MLP로 변환한다."""
        angles = timesteps.float()[:, None] * self.frequencies[None, :]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=1))


class ResidualBlock(nn.Module):
    """입력 특징과 시간 정보를 합쳐 처리하고 입력을 다시 더한다."""

    def __init__(self, inputs: int, outputs: int, time_dim: int):
        """두 convolution과 채널 수를 맞출 잔차 경로를 준비한다."""
        super().__init__()
        self.norm1 = nn.GroupNorm(8, inputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, outputs)
        self.norm2 = nn.GroupNorm(8, outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.skip = nn.Identity() if inputs == outputs else nn.Conv2d(inputs, outputs, 1)

    def forward(self, x, time):
        """시간 정보를 특징 맵에 더하고 입력의 잔차를 합친다."""
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(time)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class NoisePredictorUNet(nn.Module):
    """noisy Mel에서 추가된 noise와 같은 크기의 텐서를 예측한다."""

    def __init__(self, base_channels: int = 32, time_dim: int = 128):
        """encoder·decoder와 단계별 특징을 연결할 U-Net 층을 만든다."""
        super().__init__()
        if base_channels < 8 or base_channels % 8 or time_dim < 4 or time_dim % 2:
            raise ValueError("Channels must be a positive multiple of 8; time_dim must be even >= 4")
        c = base_channels
        self.time = TimeEmbedding(time_dim)
        self.input = nn.Conv2d(2, c, 3, padding=1)
        self.encoder1 = ResidualBlock(c, c, time_dim)
        self.down1 = nn.Conv2d(c, 2*c, 4, stride=2, padding=1)
        self.encoder2 = ResidualBlock(2*c, 2*c, time_dim)
        self.down2 = nn.Conv2d(2*c, 4*c, 4, stride=2, padding=1)
        self.middle = ResidualBlock(4*c, 4*c, time_dim)
        self.up2 = nn.ConvTranspose2d(4*c, 2*c, 4, stride=2, padding=1)
        self.decoder2 = ResidualBlock(4*c, 2*c, time_dim)
        self.up1 = nn.ConvTranspose2d(2*c, c, 4, stride=2, padding=1)
        self.decoder1 = ResidualBlock(2*c, c, time_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, noisy, timesteps, mask):
        """noisy Mel·단계·mask를 받아 입력 Mel과 같은 크기의 noise를 예측한다."""
        if noisy.ndim != 4 or noisy.shape[1] != 1:
            raise ValueError("Expected noisy Mel (B, 1, frequency, time)")
        if tuple(mask.shape) != (noisy.shape[0], 1, 1, noisy.shape[-1]):
            raise ValueError("Expected mask (B, 1, 1, time)")
        if timesteps.shape != (noisy.shape[0],):
            raise ValueError("Expected one timestep per sample")
        height, width = noisy.shape[-2:]
        # 두 번 축소하므로 4의 배수로 맞춘다. 기본 126 프레임은 128이 된다.
        padding = (0, (-width) % 4, 0, (-height) % 4)
        expanded_mask = mask.expand_as(noisy)
        noisy = torch.where(expanded_mask.bool(), noisy, -1.0)
        # Mel과 mask를 채널 방향으로 합친다. 기본 입력은 (B, 2, 80, 128)이 된다.
        x = torch.cat((F.pad(noisy, padding, value=-1), F.pad(expanded_mask, padding)), dim=1)
        time = self.time(timesteps)
        skip1 = self.encoder1(self.input(x), time)
        skip2 = self.encoder2(self.down1(skip1), time)
        x = self.middle(self.down2(skip2), time)
        # encoder의 세부 특징을 decoder에 다시 전달하는 U-Net skip connection이다.
        x = self.decoder2(torch.cat((self.up2(x), skip2), dim=1), time)
        x = self.decoder1(torch.cat((self.up1(x), skip1), dim=1), time)
        return self.output(x)[..., :height, :width]
