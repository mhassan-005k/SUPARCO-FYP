from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FaithSRConfig:
    in_channels: int = 3
    out_channels: int = 3
    channels: int = 48
    num_blocks: int = 8
    embed_dim: int = 64
    kernel_size: int = 7
    scale: int = 3
    use_degradation_adaptation: bool = True
    use_observation_consistency: bool = True
    use_uncertainty: bool = True

    def __post_init__(self) -> None:
        if self.scale != 3:
            raise ValueError("FaithSR3X supports exactly 3x scaling")
        if self.in_channels != self.out_channels:
            raise ValueError("Faithful TIFF inference requires matching input/output bands")
        if self.kernel_size % 2 == 0:
            raise ValueError("Observation kernel size must be odd")


class DegradationEncoder(nn.Module):
    def __init__(self, bands: int, embed_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(bands, 24, 3, 2, 1), nn.GELU(),
            nn.Conv2d(24, 40, 3, 2, 1), nn.GELU(),
            nn.Conv2d(40, 56, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.embedding = nn.Sequential(nn.Linear(56, embed_dim), nn.GELU())
        self.parameter_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, 4), nn.Sigmoid())

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embedding(self.features(image))
        return embedding, self.parameter_head(embedding)


class AdaptiveBlock(nn.Module):
    def __init__(self, channels: int, embed_dim: int, adaptive: bool) -> None:
        super().__init__()
        hidden = channels * 2
        self.adaptive = adaptive
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.modulation = nn.Linear(embed_dim, channels * 2) if adaptive else None
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 4), 1), nn.GELU(),
            nn.Conv2d(max(8, channels // 4), channels, 1), nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        if self.adaptive:
            gamma, beta = self.modulation(embedding).chunk(2, dim=1)
            residual = residual * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return x + self.scale * residual * self.gate(residual)


class LearnedObservationOperator(nn.Module):
    """Projects SR back to LR through a positive, normalized image-adaptive kernel."""

    def __init__(self, embed_dim: int, kernel_size: int, scale: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.scale = scale
        self.kernel_head = nn.Linear(embed_dim, kernel_size * kernel_size)

    def forward(self, sr: torch.Tensor, embedding: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        batch, bands, height, width = sr.shape
        kernels = self.kernel_head(embedding).softmax(dim=1)
        kernels = kernels.view(batch, 1, self.kernel_size, self.kernel_size)
        patches = F.unfold(sr, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.reshape(batch, bands, self.kernel_size * self.kernel_size, height * width)
        blurred = (patches * kernels.flatten(2).unsqueeze(-1)).sum(dim=2).reshape(batch, bands, height, width)
        observed = F.avg_pool2d(blurred, kernel_size=self.scale, stride=self.scale)
        return observed, kernels.squeeze(1)


class FaithSR3X(nn.Module):
    def __init__(self, config: FaithSRConfig | None = None) -> None:
        super().__init__()
        self.config = config or FaithSRConfig()
        cfg = self.config
        needs_embedding = cfg.use_degradation_adaptation or cfg.use_observation_consistency
        self.encoder = DegradationEncoder(cfg.in_channels, cfg.embed_dim) if needs_embedding else None
        self.shallow = nn.Conv2d(cfg.in_channels, cfg.channels, 3, 1, 1)
        self.blocks = nn.ModuleList([
            AdaptiveBlock(cfg.channels, cfg.embed_dim, cfg.use_degradation_adaptation)
            for _ in range(cfg.num_blocks)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(cfg.channels * cfg.num_blocks, cfg.channels, 1), nn.GELU(),
            nn.Conv2d(cfg.channels, cfg.channels, 3, 1, 1),
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.out_channels * 9, 3, 1, 1),
            nn.PixelShuffle(3),
        )
        self.observation = LearnedObservationOperator(cfg.embed_dim, cfg.kernel_size) if cfg.use_observation_consistency else None
        self.consistency_gate = nn.Sequential(nn.Linear(cfg.embed_dim, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid()) if cfg.use_observation_consistency else None
        self.uncertainty = nn.Sequential(
            nn.Conv2d(cfg.channels, max(8, cfg.channels // 2), 3, 1, 1), nn.GELU(),
            nn.Conv2d(max(8, cfg.channels // 2), 1, 3, 1, 1), nn.Sigmoid(),
        ) if cfg.use_uncertainty else None

    def forward(self, lr: torch.Tensor, return_aux: bool = False):
        if lr.ndim != 4 or lr.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected BCHW with {self.config.in_channels} bands")
        if self.encoder is None:
            embedding = lr.new_zeros((lr.shape[0], self.config.embed_dim))
            degradation = None
        else:
            embedding, degradation = self.encoder(lr)
        shallow = self.shallow(lr)
        feature = shallow
        history = []
        for block in self.blocks:
            feature = block(feature, embedding)
            history.append(feature)
        feature = self.fusion(torch.cat(history, dim=1)) + shallow
        base = F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False)
        sr = base + self.reconstruction(feature)
        observed = kernels = strength = None
        if self.observation is not None:
            observed, kernels = self.observation(sr, embedding, lr.shape[-2:])
            strength = 0.25 * self.consistency_gate(embedding).view(-1, 1, 1, 1)
            correction = F.interpolate(lr - observed, scale_factor=3, mode="bicubic", align_corners=False)
            sr = sr + strength * correction
            observed, kernels = self.observation(sr, embedding, lr.shape[-2:])
        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = F.interpolate(self.uncertainty(feature), scale_factor=3, mode="bilinear", align_corners=False)
        if not return_aux:
            return sr
        return sr, {
            "degradation_params": degradation,
            "reprojected_lr": observed,
            "observation_kernel": kernels,
            "consistency_strength": strength,
            "uncertainty": uncertainty,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
