from __future__ import annotations

from pathlib import Path

import torch

from .model import FaithSR3X, FaithSRConfig


def save_checkpoint(path: str | Path, model: FaithSR3X, **state) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": model.config.__dict__, **state}, target)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[FaithSR3X, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = FaithSRConfig(**checkpoint["model_config"])
    model = FaithSR3X(config)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
