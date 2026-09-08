import torch

from faithsr.model import FaithSR3X, FaithSRConfig


def test_native_3x_shape_and_auxiliary_outputs():
    model = FaithSR3X(FaithSRConfig(channels=12, num_blocks=2, embed_dim=16, kernel_size=5))
    lr = torch.rand(2, 3, 17, 19)
    sr, aux = model(lr, return_aux=True)
    assert sr.shape == (2, 3, 51, 57)
    assert aux["reprojected_lr"].shape == lr.shape
    assert aux["uncertainty"].shape == (2, 1, 51, 57)
    assert torch.allclose(aux["observation_kernel"].sum((1, 2)), torch.ones(2), atol=1e-6)
    assert aux["consistency_strength"].max() <= 0.25


def test_arbitrary_band_count():
    model = FaithSR3X(FaithSRConfig(in_channels=4, out_channels=4, channels=8, num_blocks=1, embed_dim=8))
    assert model(torch.rand(1, 4, 8, 9)).shape == (1, 4, 24, 27)


def test_scale_other_than_three_rejected():
    try:
        FaithSRConfig(scale=4)
    except ValueError as error:
        assert "exactly 3x" in str(error)
    else:
        raise AssertionError("Scale validation did not run")
