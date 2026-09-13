from argparse import Namespace

import numpy as np
from PIL import Image
import pytest
import yaml

from faithsr.checkpoint import load_checkpoint
from faithsr.lite_x4 import FaithSRLiteX4
from faithsr.lite_x4_train import VARIANTS, run, variant_model_config


def test_lite_x4_ablation_switches_isolate_the_registered_components():
    assert set(VARIANTS) == {"base", "distilled", "compression", "full"}
    base = variant_model_config({"channels": 8, "num_blocks": 1, "embed_dim": 8}, "base")
    distilled = variant_model_config({"channels": 8, "num_blocks": 1, "embed_dim": 8}, "distilled")
    compression = variant_model_config({"channels": 8, "num_blocks": 1, "embed_dim": 8}, "compression")
    full = variant_model_config({"channels": 8, "num_blocks": 1, "embed_dim": 8}, "full")

    assert not base.use_degradation_adaptation and not base.use_compression_refinement
    assert distilled == base
    assert compression.use_degradation_adaptation
    assert compression.use_observation_consistency
    assert compression.use_compression_refinement
    assert full == compression
    assert VARIANTS["distilled"]["teacher"] and VARIANTS["full"]["teacher"]
    assert not VARIANTS["base"]["teacher"] and not VARIANTS["compression"]["teacher"]


def test_lite_x4_ablation_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown"):
        variant_model_config({}, "fashionable-module")


def test_base_variant_completes_a_one_epoch_native_x4_smoke_run(tmp_path):
    data = tmp_path / "data"
    for index, scene in enumerate(("scene-a", "scene-b")):
        directory = data / scene
        directory.mkdir(parents=True)
        values = np.arange(24 * 24 * 3, dtype=np.uint16).reshape(24, 24, 3)
        Image.fromarray(((values + index * 17) % 256).astype(np.uint8)).save(directory / "tile.png")
    config = {
        "seed": 0,
        "scale": 4,
        "model": {"channels": 8, "num_blocks": 1, "embed_dim": 8, "kernel_size": 5, "max_consistency_strength": 0.2},
        "data": {"crop_size": 16, "train_ratio": 0.5, "split_grid_degrees": 0.1, "extensions": [".png"], "mixed_probability": 1.0},
        "degradation": {"curriculum_epochs": 1, "start_severity": 1.0, "mtf_probability": 0.0},
        "training": {"epochs": 1, "batch_size": 1, "validation_batch_size": 1, "workers": 0, "learning_rate": 0.0002, "weight_decay": 0.0001, "amp": False},
        "loss": {"charbonnier": 1.0, "ssim": 0.0, "gradient": 0.0, "frequency": 0.0, "observation": 0.0, "degradation": 0.0, "distillation": 0.0, "distillation_edge": 0.0, "compression_focus": 0.0},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "run"

    run(Namespace(
        data=str(data), config=str(config_path), output=str(output), variant="base",
        bands=3, device="cpu", teacher_model="unused", teacher_revision="unused",
        teacher_checkpoint=None,
    ))

    model, state = load_checkpoint(output / "best.pth")
    assert isinstance(model, FaithSRLiteX4)
    assert state["variant"] == "base"
    assert (output / "best-preview.png").exists()
    assert (output / "best-preview.json").exists()
    assert (output / "model-card.md").read_text(encoding="utf-8").count("Measured training result") == 1
