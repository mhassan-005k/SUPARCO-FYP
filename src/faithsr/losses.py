from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.epsilon**2).mean()


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sigma_x = F.avg_pool2d(x.square(), 11, 1, 5) - mu_x.square()
    sigma_y = F.avg_pool2d(y.square(), 11, 1, 5) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01**2) * (2 * sigma_xy + 0.03**2)) / ((mu_x.square() + mu_y.square() + 0.01**2) * (sigma_x + sigma_y + 0.03**2) + 1e-12)
    return 1 - score.mean()


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    bands = x.shape[1]
    kernel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).expand(bands, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(x, kernel_x, padding=1, groups=bands)
    gy = F.conv2d(x, kernel_y, padding=1, groups=bands)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def frequency_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(torch.abs(torch.fft.rfft2(x, norm="ortho")), torch.abs(torch.fft.rfft2(y, norm="ortho")))


def calibrated_uncertainty_loss(uncertainty: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction.detach() - target).abs().mean(dim=1, keepdim=True).clamp(0, 1)
    return F.binary_cross_entropy(uncertainty.clamp(1e-5, 1 - 1e-5), error)


class FaithfulObjective(nn.Module):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights
        self.charbonnier = CharbonnierLoss()

    def forward(self, sr, hr, lr, aux, degradation_target=None, teacher=None):
        terms = {
            "charbonnier": self.charbonnier(sr, hr),
            "ssim": ssim_loss(sr.float(), hr.float()),
            "gradient": F.l1_loss(sobel_magnitude(sr.float()), sobel_magnitude(hr.float())),
            "frequency": frequency_loss(sr.float(), hr.float()),
        }
        if aux.get("reprojected_lr") is not None:
            terms["observation"] = F.smooth_l1_loss(aux["reprojected_lr"].float(), lr.float())
        if degradation_target is not None and aux.get("degradation_params") is not None:
            terms["degradation"] = F.smooth_l1_loss(aux["degradation_params"].float(), degradation_target.float())
        if aux.get("uncertainty") is not None:
            terms["uncertainty"] = calibrated_uncertainty_loss(aux["uncertainty"].float(), sr.float(), hr.float())
        if teacher is not None:
            terms["distillation"] = F.smooth_l1_loss(sr, teacher.detach())
        total = sum(self.weights.get(name, 0.0) * value for name, value in terms.items())
        return total, terms
