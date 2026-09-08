import torch

from faithsr.metrics import false_edge_rate, psnr, spectral_angle_mapper, ssim


def test_identical_images_have_expected_fidelity_scores():
    image = torch.rand(1, 3, 24, 24)
    assert psnr(image, image) > 100
    assert torch.isclose(ssim(image, image), torch.tensor(1.0), atol=1e-5)
    assert false_edge_rate(image, image) == 0
    assert spectral_angle_mapper(image, image) < 1e-3
