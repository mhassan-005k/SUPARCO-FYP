import torch

from faithsr.metrics import ergas, false_edge_rate, psnr, segmentation_iou_f1, spectral_angle_mapper, ssim, uncertainty_calibration_error


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
