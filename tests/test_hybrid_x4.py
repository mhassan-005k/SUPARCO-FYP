import pytest
import torch

from faithsr.checkpoint import load_checkpoint, save_checkpoint
from faithsr.hybrid_x4 import HYBRID_X4_VARIANTS, HybridFaithSRX4, HybridFaithSRX4Config
from faithsr.model import count_parameters


@pytest.mark.parametrize("variant", HYBRID_X4_VARIANTS)
def test_hybrid_variants_preserve_native_x4_contract(variant):
    model = HybridFaithSRX4(HybridFaithSRX4Config(channels=12, num_blocks=2, cascade_blocks=1, variant=variant))
    lr = torch.rand(2, 3, 11, 13)
    sr, auxiliary = model(lr, return_aux=True)

    assert sr.shape == (2, 3, 44, 52)
    assert auxiliary["reprojected_lr"].shape == lr.shape
    assert count_parameters(model) < 1_000_000
    if variant == "context_fusion":
        assert auxiliary["context_gate"].shape == (2, 12, 11, 13)
        assert auxiliary["context_gate"].min() >= 0
        assert auxiliary["context_gate"].max() <= 1
    elif variant == "residual_cascade":
        assert auxiliary["cascade_strength"].shape == (2, 1, 1, 1)
        assert auxiliary["cascade_strength"].max() <= 0.15
    else:
        assert auxiliary["context_gate"] is None
        assert auxiliary["cascade_strength"] is None


def test_hybrid_checkpoint_round_trip(tmp_path):
    model = HybridFaithSRX4(HybridFaithSRX4Config(channels=8, num_blocks=1, cascade_blocks=1, variant="residual_cascade"))
    path = tmp_path / "hybrid.pth"
    save_checkpoint(path, model, variant="residual_cascade")
    restored, state = load_checkpoint(path)

    assert isinstance(restored, HybridFaithSRX4)
    assert restored.config.variant == "residual_cascade"
    assert restored(torch.rand(1, 3, 5, 6)).shape == (1, 3, 20, 24)
    assert state["model_type"] == "faithsr_hybrid_x4"


def test_hybrid_config_rejects_non_native_scale_and_unknown_variant():
    with pytest.raises(ValueError, match="native 4x"):
        HybridFaithSRX4Config(scale=3)
    with pytest.raises(ValueError, match="Unknown"):
        HybridFaithSRX4Config(variant="unregistered")
