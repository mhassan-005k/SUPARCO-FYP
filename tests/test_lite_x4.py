import pytest
import torch

from faithsr.checkpoint import load_checkpoint, save_checkpoint
from faithsr.lite_x4 import FaithSRLiteX4, FaithSRLiteX4Config
from faithsr.model import count_parameters


def test_lite_x4_shape_auxiliary_contract_and_parameter_budget():
    model = FaithSRLiteX4(
        FaithSRLiteX4Config(channels=12, num_blocks=2, embed_dim=16, kernel_size=5)
    )
    lr = torch.rand(2, 3, 13, 17)
    sr, auxiliary = model(lr, return_aux=True)

    assert sr.shape == (2, 3, 52, 68)
    assert auxiliary["reprojected_lr"].shape == lr.shape
    assert auxiliary["compression_gate"].shape == (2, 12, 1, 1)
    assert auxiliary["consistency_strength"].max() <= 0.20
    assert torch.allclose(auxiliary["observation_kernel"].sum((1, 2)), torch.ones(2), atol=1e-6)
    assert count_parameters(model) < 500_000


def test_lite_x4_plain_ablation_has_no_hidden_conditioning_modules():
    model = FaithSRLiteX4(
        FaithSRLiteX4Config(
            channels=8,
            num_blocks=1,
            embed_dim=8,
            use_degradation_adaptation=False,
            use_observation_consistency=False,
            use_compression_refinement=False,
        )
    )
    sr, auxiliary = model(torch.rand(1, 3, 8, 9), return_aux=True)

    assert sr.shape == (1, 3, 32, 36)
    assert model.encoder is None
    assert auxiliary["degradation_params"] is None
    assert auxiliary["reprojected_lr"] is None
    assert auxiliary["compression_gate"] is None


def test_lite_x4_rejects_non_native_scale():
    with pytest.raises(ValueError, match="native 4x"):
        FaithSRLiteX4Config(scale=3)


def test_lite_x4_checkpoint_round_trip(tmp_path):
    model = FaithSRLiteX4(FaithSRLiteX4Config(channels=8, num_blocks=1, embed_dim=8))
    checkpoint = tmp_path / "lite-x4.pth"
    save_checkpoint(checkpoint, model, epoch=1)

    restored, state = load_checkpoint(checkpoint)

    assert isinstance(restored, FaithSRLiteX4)
    assert restored(torch.rand(1, 3, 5, 6)).shape == (1, 3, 20, 24)
    assert state["model_type"] == "faithsr_lite_x4"
