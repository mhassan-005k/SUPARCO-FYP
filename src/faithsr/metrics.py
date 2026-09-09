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


def reconstruction_metrics_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Vectorized per-image metrics for exact CSV rows and paired intervals."""
    dimensions = tuple(range(1, prediction.ndim))
    mse = (prediction - target).square().mean(dim=dimensions)
    psnr_values = 10 * torch.log10(1.0 / (mse + 1e-12))
    mu_x = F.avg_pool2d(prediction, 11, 1, 5); mu_y = F.avg_pool2d(target, 11, 1, 5)
    sigma_x = F.avg_pool2d(prediction.square(), 11, 1, 5) - mu_x.square(); sigma_y = F.avg_pool2d(target.square(), 11, 1, 5) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, 11, 1, 5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + 0.01**2) * (2 * sigma_xy + 0.03**2)) / ((mu_x.square() + mu_y.square() + 0.01**2) * (sigma_x + sigma_y + 0.03**2) + 1e-12)
    ssim_values = ssim_map.flatten(1).mean(1)
    predicted_edges = sobel_magnitude(prediction).mean(1) > 0.10; target_edges = sobel_magnitude(target).mean(1) > 0.10
    edge_dims = tuple(range(1, predicted_edges.ndim)); true_positive = (predicted_edges & target_edges).sum(dim=edge_dims).float()
    predicted_count = predicted_edges.sum(dim=edge_dims).clamp_min(1); target_count = target_edges.sum(dim=edge_dims).clamp_min(1)
    false_edge_values = (predicted_edges & ~target_edges).sum(dim=edge_dims).float() / predicted_count
    dot = (prediction * target).sum(dim=1); norms = prediction.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    sam_values = torch.acos((dot / norms.clamp_min(1e-8)).clamp(-1 + 1e-7, 1 - 1e-7)).flatten(1).mean(1)
    spatial_dims = tuple(range(2, target.ndim)); band_rmse = (prediction - target).square().mean(dim=spatial_dims).sqrt(); band_mean = target.mean(dim=spatial_dims).abs().clamp_min(1e-8)
    ergas_values = 100 / 3 * torch.sqrt(((band_rmse / band_mean).square()).mean(dim=1))
    return {"psnr": psnr_values, "ssim": ssim_values, "false_edge_rate": false_edge_values, "edge_precision": true_positive / predicted_count, "edge_recall": true_positive / target_count, "sam_radians": sam_values, "ergas": ergas_values}


def uncertainty_metrics_per_sample(uncertainty: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, bins: int = 10) -> dict[str, torch.Tensor]:
    batch = prediction.shape[0]
    risk = uncertainty.flatten(1).float(); error = (prediction - target).abs().mean(1, keepdim=True).flatten(1).float()
    centered_risk = risk - risk.mean(1, keepdim=True); centered_error = error - error.mean(1, keepdim=True)
    correlation = (centered_risk * centered_error).mean(1) / (centered_risk.std(1, unbiased=False) * centered_error.std(1, unbiased=False) + 1e-8)
    calibration = risk.new_zeros(batch)
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = (risk >= lower) & (risk < upper if index < bins - 1 else risk <= upper)
        count = members.sum(1); safe_count = count.clamp_min(1)
        predicted_mean = (risk * members).sum(1) / safe_count; observed_mean = (error * members).sum(1) / safe_count
        calibration += (count.float() / risk.shape[1]) * (predicted_mean - observed_mean).abs()
    return {"uncertainty_correlation": correlation, "uncertainty_calibration_error": calibration}
