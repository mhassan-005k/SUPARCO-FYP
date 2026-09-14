from argparse import Namespace

import numpy as np
from PIL import Image
import pytest
import yaml

from faithsr.checkpoint import load_checkpoint
from faithsr.hybrid_x4 import HybridFaithSRX4
from faithsr.hybrid_x4_train import run, variant_model_config


def test_variant_config_changes_only_registered_architecture_mode():
    common = {"channels": 8, "num_blocks": 1, "cascade_blocks": 1}
    base = variant_model_config(common, "medium_base")
    context = variant_model_config(common, "context_fusion")
    cascade = variant_model_config(common, "residual_cascade")

    assert base.variant == "medium_base"
    assert context.variant == "context_fusion"
    assert cascade.variant == "residual_cascade"
    assert base.channels == context.channels == cascade.channels
    assert base.num_blocks == context.num_blocks == cascade.num_blocks
    with pytest.raises(ValueError, match="Unknown"):
        variant_model_config(common, "fashionable_module")


def test_context_variant_completes_one_epoch_smoke_run(tmp_path):
    data = tmp_path / "data"
    for index, scene in enumerate(("scene-a", "scene-b")):
        directory = data / scene
        directory.mkdir(parents=True)
        values = np.arange(24 * 24 * 3, dtype=np.uint16).reshape(24, 24, 3)
        Image.fromarray(((values + index * 17) % 256).astype(np.uint8)).save(directory / "tile.png")
    config = {
        "seed": 0, "scale": 4,
        "model": {"channels": 8, "num_blocks": 1, "context_levels": 2, "context_kernel_size": 5, "cascade_blocks": 1, "max_cascade_strength": 0.15},
        "data": {"crop_size": 16, "train_ratio": 0.5, "split_grid_degrees": 0.1, "extensions": [".png"], "mixed_probability": 1.0},
        "degradation": {"curriculum_epochs": 1, "start_severity": 1.0, "mtf_probability": 0.0},
        "training": {"epochs": 1, "batch_size": 1, "validation_batch_size": 1, "workers": 0, "learning_rate": 0.0002, "weight_decay": 0.0001, "amp": False},
        "loss": {"charbonnier": 1.0, "ssim": 0.0, "gradient": 0.0, "frequency": 0.0, "observation": 0.01, "degradation": 0.0, "distillation": 0.0, "distillation_edge": 0.0, "compression_focus": 0.0},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "run"
    run(Namespace(data=str(data), config=str(config_path), output=str(output), variant="context_fusion", bands=3, device="cpu"))

    model, state = load_checkpoint(output / "best.pth")
    assert isinstance(model, HybridFaithSRX4)
    assert state["variant"] == "context_fusion"
    assert (output / "best-preview.png").exists()
    card = (output / "model-card.md").read_text(encoding="utf-8")
    assert "MultiScaleContextBranch" in card
    assert card.count("Measured training result") == 1
