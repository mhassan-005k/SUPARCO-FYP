from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class Swin2SRTeacher(nn.Module):
    """Frozen Hugging Face Swin2SR teacher adapted to the fixed 3x student target."""

    def __init__(self, model_id: str, checkpoint: str | Path | None = None, revision: str | None = None, batch_size: int | None = None) -> None:
        super().__init__()
        try:
            from transformers import Swin2SRForImageSuperResolution
        except ImportError as error:
            raise RuntimeError("Install faithsr[teacher] to use Swin2SR distillation") from error
        self.model = Swin2SRForImageSuperResolution.from_pretrained(model_id, revision=revision)
        self.batch_size = batch_size
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(f"Teacher checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, lr: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        chunk_size = self.batch_size or lr.shape[0]
        outputs = []
        for chunk in lr.split(chunk_size):
            output = self.model(pixel_values=chunk).reconstruction
            outputs.append(F.interpolate(output, size=target_size, mode="bicubic", align_corners=False, antialias=True))
        return torch.cat(outputs, dim=0)
