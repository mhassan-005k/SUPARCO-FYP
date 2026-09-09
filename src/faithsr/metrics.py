from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import sobel_magnitude, ssim_loss


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return 10 * torch.log10(1.0 / (mse + 1e-12))


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - ssim_loss(prediction, target)


def false_edge_rate(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.10) -> torch.Tensor:
    predicted_edges = sobel_magnitude(prediction).mean(1) > threshold
    target_edges = sobel_magnitude(target).mean(1) > threshold
    false_edges = predicted_edges & ~target_edges
    return false_edges.sum().float() / predicted_edges.sum().clamp_min(1)


def edge_precision_recall(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.10) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = sobel_magnitude(prediction).mean(1) > threshold
    actual = sobel_magnitude(target).mean(1) > threshold
    true_positive = (predicted & actual).sum().float()
    precision = true_positive / predicted.sum().clamp_min(1)
    recall = true_positive / actual.sum().clamp_min(1)
    return precision, recall


def spectral_angle_mapper(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dot = (prediction * target).sum(dim=1)
    norms = prediction.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    return torch.acos((dot / norms.clamp_min(1e-8)).clamp(-1 + 1e-7, 1 - 1e-7)).mean()


def uncertainty_error_correlation(uncertainty: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    u = uncertainty.flatten().float()
    error = (prediction - target).abs().mean(1, keepdim=True).flatten().float()
    u = u - u.mean()
    error = error - error.mean()
    return (u * error).mean() / (u.std(unbiased=False) * error.std(unbiased=False) + 1e-8)


def uncertainty_calibration_error(uncertainty: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, bins: int = 10) -> torch.Tensor:
    predicted_risk = uncertainty.flatten().float().clamp(0, 1)
    observed_error = (prediction - target).abs().mean(1, keepdim=True).flatten().float().clamp(0, 1)
    total = predicted_risk.new_tensor(0.0)
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = (predicted_risk >= lower) & (predicted_risk < upper if index < bins - 1 else predicted_risk <= upper)
        if members.any():
            total = total + members.float().mean() * (predicted_risk[members].mean() - observed_error[members].mean()).abs()
    return total


def ergas(prediction: torch.Tensor, target: torch.Tensor, scale: int = 3) -> torch.Tensor:
    spatial_dims = tuple(range(2, target.ndim))
    band_rmse = (prediction - target).square().mean(dim=spatial_dims).sqrt()
    band_mean = target.mean(dim=spatial_dims).abs().clamp_min(1e-8)
    return 100 / scale * torch.sqrt(((band_rmse / band_mean).square()).mean())


def segmentation_iou_f1(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, target = prediction.bool(), target.bool()
    intersection = (prediction & target).sum().float()
    union = (prediction | target).sum().float()
    iou = intersection / union.clamp_min(1)
    f1 = 2 * intersection / (prediction.sum() + target.sum()).clamp_min(1)
    return iou, f1
