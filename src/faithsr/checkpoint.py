from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .lite_x4 import FaithSRLiteX4, FaithSRLiteX4Config
from .model import FaithSR3X, FaithSRConfig


def save_checkpoint(path: str | Path, model: nn.Module, **state) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model_type = "faithsr_lite_x4" if isinstance(model, FaithSRLiteX4) else "faithsr3x"
    torch.save({"model": model.state_dict(), "model_config": model.config.__dict__, "model_type": model_type, **state}, target)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") == "faithsr_lite_x4":
        model = FaithSRLiteX4(FaithSRLiteX4Config(**checkpoint["model_config"]))
    else:
        model = FaithSR3X(FaithSRConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
