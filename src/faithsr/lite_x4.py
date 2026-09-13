from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model import AdaptiveBlock, DegradationEncoder, LearnedObservationOperator


@dataclass(frozen=True)
class FaithSRLiteX4Config:
    in_channels: int = 3
    out_channels: int = 3
    channels: int = 32
    num_blocks: int = 6
    embed_dim: int = 48
    kernel_size: int = 7
    scale: int = 4
    use_degradation_adaptation: bool = True
    use_observation_consistency: bool = True
    use_compression_refinement: bool = True
    max_consistency_strength: float = 0.20

    def __post_init__(self) -> None:
        if self.scale != 4:
            raise ValueError("FaithSRLiteX4 supports exactly native 4x scaling")
        if self.in_channels != self.out_channels:
            raise ValueError("Faithful TIFF inference requires matching input/output bands")
        if self.kernel_size % 2 == 0:
            raise ValueError("Observation kernel size must be odd")
        if not 0 <= self.max_consistency_strength <= 0.25:
            raise ValueError("Consistency strength must be within [0,0.25]")


class CompressionRefinementBlock(nn.Module):
    """Small LR-space residual branch gated by estimated compression severity."""

    def __init__(self, channels: int, embed_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.channel_gate = nn.Sequential(nn.Linear(embed_dim, channels), nn.Sigmoid())
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        feature: torch.Tensor,
        embedding: torch.Tensor,
        compression: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.channel_gate(embedding)[:, :, None, None] * compression[:, None, None, None]
        refined = feature + self.residual_scale * gate * self.body(feature)
        return refined, gate


class FaithSRLiteX4(nn.Module):
    """Compact native-x4 student selected by the Swin2SR latency/compression diagnosis."""

    scale = 4

    def __init__(self, config: FaithSRLiteX4Config | None = None) -> None:
        super().__init__()
        self.config = config or FaithSRLiteX4Config()
        cfg = self.config
        needs_embedding = (
            cfg.use_degradation_adaptation
            or cfg.use_observation_consistency
            or cfg.use_compression_refinement
        )
        self.encoder = DegradationEncoder(cfg.in_channels, cfg.embed_dim) if needs_embedding else None
        self.shallow = nn.Conv2d(cfg.in_channels, cfg.channels, 3, 1, 1)
        self.blocks = nn.ModuleList(
            AdaptiveBlock(cfg.channels, cfg.embed_dim, cfg.use_degradation_adaptation)
            for _ in range(cfg.num_blocks)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(cfg.channels * cfg.num_blocks, cfg.channels, 1),
            nn.GELU(),
            nn.Conv2d(cfg.channels, cfg.channels, 3, 1, 1),
        )
        self.compression_refiner = (
            CompressionRefinementBlock(cfg.channels, cfg.embed_dim)
            if cfg.use_compression_refinement
            else None
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.out_channels * cfg.scale**2, 3, 1, 1),
            nn.PixelShuffle(cfg.scale),
        )
        self.observation = (
            LearnedObservationOperator(cfg.embed_dim, cfg.kernel_size, scale=cfg.scale)
            if cfg.use_observation_consistency
            else None
        )
        self.consistency_gate = (
            nn.Sequential(nn.Linear(cfg.embed_dim, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
            if cfg.use_observation_consistency
            else None
        )

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

        compression_gate = None
        if self.compression_refiner is not None:
            feature, compression_gate = self.compression_refiner(
                feature,
                embedding,
                degradation[:, 2],
            )

        scale = self.config.scale
        base = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
        sr = base + self.reconstruction(feature)
        observed = kernels = strength = None
        if self.observation is not None:
            observed, kernels = self.observation(sr, embedding, lr.shape[-2:])
            strength = self.config.max_consistency_strength * self.consistency_gate(embedding).view(-1, 1, 1, 1)
            correction = F.interpolate(lr - observed, scale_factor=scale, mode="bicubic", align_corners=False)
            sr = sr + strength * correction
            observed, kernels = self.observation(sr, embedding, lr.shape[-2:])

        if not return_aux:
            return sr
        return sr, {
            "degradation_params": degradation,
            "reprojected_lr": observed,
            "observation_kernel": kernels,
            "consistency_strength": strength,
            "compression_gate": compression_gate,
            "uncertainty": None,
        }
