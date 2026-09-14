from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


HYBRID_X4_VARIANTS = ("medium_base", "context_fusion", "residual_cascade")


@dataclass(frozen=True)
class HybridFaithSRX4Config:
    """Configuration shared by the matched native-x4 hybrid comparison."""

    in_channels: int = 3
    out_channels: int = 3
    channels: int = 48
    num_blocks: int = 8
    scale: int = 4
    variant: str = "medium_base"
    context_levels: int = 3
    context_kernel_size: int = 7
    cascade_blocks: int = 3
    max_cascade_strength: float = 0.15

    def __post_init__(self) -> None:
        if self.scale != 4:
            raise ValueError("HybridFaithSRX4 supports exactly native 4x scaling")
        if self.in_channels != self.out_channels:
            raise ValueError("Faithful TIFF inference requires matching input/output bands")
        if self.variant not in HYBRID_X4_VARIANTS:
            raise ValueError(f"Unknown hybrid x4 variant: {self.variant}")
        if self.channels < 8:
            raise ValueError("channels must be at least 8")
        if self.num_blocks < 1 or self.cascade_blocks < 1:
            raise ValueError("num_blocks and cascade_blocks must be positive")
        if self.context_levels < 1:
            raise ValueError("context_levels must be positive")
        if self.context_kernel_size % 2 == 0:
            raise ValueError("context_kernel_size must be odd")
        if not 0.0 <= self.max_cascade_strength <= 0.25:
            raise ValueError("max_cascade_strength must be within [0, 0.25]")


class EfficientResidualBlock(nn.Module):
    """Depthwise/pointwise residual block used by every matched variant."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.residual_scale * self.body(feature)


class MultiScaleContextBranch(nn.Module):
    """Cheap LR-space context aggregation with a learned local/context gate."""

    def __init__(self, channels: int, levels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.levels = levels
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, 1, padding, groups=channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
            )
            for _ in range(levels)
        )
        self.context_mix = nn.Sequential(
            nn.Conv2d(channels * levels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = local.shape[-2:]
        context_features = []
        for level, branch in enumerate(self.branches):
            factor = 2**level
            if factor == 1:
                pooled = local
            else:
                pooled = F.adaptive_avg_pool2d(
                    local,
                    (max(1, height // factor), max(1, width // factor)),
                )
            context = branch(pooled)
            if context.shape[-2:] != (height, width):
                context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=False)
            context_features.append(context)
        context = self.context_mix(torch.cat(context_features, dim=1))
        gate = self.gate(torch.cat((local, context), dim=1))
        return local + gate * context, gate


class ResidualConsistencyExpert(nn.Module):
    """Second-stage LR error expert; it never degrades the application's input."""

    def __init__(self, bands: int, channels: int, blocks: int, scale: int, max_strength: float) -> None:
        super().__init__()
        self.scale = scale
        self.max_strength = max_strength
        self.stem = nn.Conv2d(bands * 2, channels, 3, 1, 1)
        self.blocks = nn.Sequential(*(EfficientResidualBlock(channels) for _ in range(blocks)))
        self.delta = nn.Sequential(
            nn.Conv2d(channels, bands * scale**2, 3, 1, 1),
            nn.PixelShuffle(scale),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 4), 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, lr: torch.Tensor, initial_sr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed = F.interpolate(initial_sr, size=lr.shape[-2:], mode="area")
        error = lr - observed
        feature = self.blocks(self.stem(torch.cat((lr, error), dim=1)))
        strength = self.max_strength * self.gate(feature)
        final_sr = initial_sr + strength * self.delta(feature)
        final_observed = F.interpolate(final_sr, size=lr.shape[-2:], mode="area")
        return final_sr, final_observed, strength


class HybridFaithSRX4(nn.Module):
    """Matched control and two testable native-x4 hybrid alternatives."""

    scale = 4

    def __init__(self, config: HybridFaithSRX4Config | None = None) -> None:
        super().__init__()
        self.config = config or HybridFaithSRX4Config()
        cfg = self.config
        self.shallow = nn.Conv2d(cfg.in_channels, cfg.channels, 3, 1, 1)
        self.blocks = nn.ModuleList(EfficientResidualBlock(cfg.channels) for _ in range(cfg.num_blocks))
        self.fusion = nn.Sequential(
            nn.Conv2d(cfg.channels * cfg.num_blocks, cfg.channels, 1),
            nn.GELU(),
            nn.Conv2d(cfg.channels, cfg.channels, 3, 1, 1),
        )
        self.context = (
            MultiScaleContextBranch(cfg.channels, cfg.context_levels, cfg.context_kernel_size)
            if cfg.variant == "context_fusion"
            else None
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.out_channels * cfg.scale**2, 3, 1, 1),
            nn.PixelShuffle(cfg.scale),
        )
        self.cascade = (
            ResidualConsistencyExpert(
                cfg.in_channels,
                cfg.channels,
                cfg.cascade_blocks,
                cfg.scale,
                cfg.max_cascade_strength,
            )
            if cfg.variant == "residual_cascade"
            else None
        )

    def forward(self, lr: torch.Tensor, return_aux: bool = False):
        if lr.ndim != 4 or lr.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected BCHW with {self.config.in_channels} bands")
        shallow = self.shallow(lr)
        feature = shallow
        history = []
        for block in self.blocks:
            feature = block(feature)
            history.append(feature)
        feature = shallow + self.fusion(torch.cat(history, dim=1))

        context_gate = None
        if self.context is not None:
            feature, context_gate = self.context(feature)

        base = F.interpolate(lr, scale_factor=self.config.scale, mode="bicubic", align_corners=False)
        initial_sr = base + self.reconstruction(feature)
        sr = initial_sr
        cascade_strength = None
        reprojected = F.interpolate(sr, size=lr.shape[-2:], mode="area")
        if self.cascade is not None:
            sr, reprojected, cascade_strength = self.cascade(lr, initial_sr)

        if not return_aux:
            return sr
        return sr, {
            "degradation_params": None,
            "reprojected_lr": reprojected,
            "observation_kernel": None,
            "consistency_strength": cascade_strength,
            "cascade_strength": cascade_strength,
            "context_gate": context_gate,
            "compression_gate": None,
            "uncertainty": None,
        }
