import torch

from faithsr.metrics import (
    ergas,
    false_edge_rate,
    psnr,
    reconstruction_metrics_per_sample,
    segmentation_iou_f1,
    spectral_angle_mapper,
    ssim,
    uncertainty_calibration_error,
    uncertainty_metrics_per_sample,
)


def test_identical_images_have_expected_fidelity_scores():
    image = torch.rand(1, 3, 24, 24)
    assert psnr(image, image) > 100
    assert torch.isclose(ssim(image, image), torch.tensor(1.0), atol=1e-5)
    assert false_edge_rate(image, image) == 0
    assert spectral_angle_mapper(image, image) < 1e-3
    assert ergas(image, image) == 0
    assert uncertainty_calibration_error(torch.zeros_like(image[:, :1]), image, image) == 0


def test_downstream_binary_metrics():
    prediction = torch.tensor([[1, 1, 0, 0]])
    target = torch.tensor([[1, 0, 1, 0]])
    iou, f1 = segmentation_iou_f1(prediction, target)
    assert torch.isclose(iou, torch.tensor(1 / 3))
    assert torch.isclose(f1, torch.tensor(0.5))


def test_vectorized_metrics_return_one_value_per_image():
    target = torch.rand(3, 3, 24, 24)
    prediction = (target + 0.01 * torch.randn_like(target)).clamp(0, 1)
    uncertainty = torch.rand(3, 1, 24, 24)
    reconstruction = reconstruction_metrics_per_sample(prediction, target)
    calibration = uncertainty_metrics_per_sample(uncertainty, prediction, target)
    assert all(value.shape == (3,) for value in reconstruction.values())
    assert all(value.shape == (3,) for value in calibration.values())


def test_vectorized_ergas_uses_declared_x4_scale():
    target = torch.full((1, 3, 16, 16), 0.5)
    prediction = torch.full_like(target, 0.6)
    x3 = reconstruction_metrics_per_sample(prediction, target, scale=3)["ergas"]
    x4 = reconstruction_metrics_per_sample(prediction, target, scale=4)["ergas"]
    assert torch.allclose(x4, x3 * 3 / 4)
