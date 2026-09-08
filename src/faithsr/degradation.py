from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DegradationParameters:
    blur: float
    noise: float
    compression: float
    aliasing: float
    sigma_x: float
    sigma_y: float
    angle_degrees: float
    jpeg_quality: int

    def normalized_target(self) -> torch.Tensor:
        return torch.tensor([self.blur, self.noise, self.compression, self.aliasing], dtype=torch.float32)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _anisotropic_kernel(size: int, sigma_x: float, sigma_y: float, angle: float, device, dtype) -> torch.Tensor:
    axis = torch.arange(size, device=device, dtype=dtype) - size // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    cosine, sine = torch.cos(torch.tensor(angle, device=device, dtype=dtype)), torch.sin(torch.tensor(angle, device=device, dtype=dtype))
    xr = cosine * xx + sine * yy
    yr = -sine * xx + cosine * yy
    kernel = torch.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    return kernel / kernel.sum()


def _jpeg_roundtrip(image: torch.Tensor, quality: int) -> torch.Tensor:
    if image.shape[0] not in (1, 3):
        return image
    array = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    if array.shape[2] == 1:
        array = array[:, :, 0]
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    decoded = np.asarray(Image.open(buffer), dtype=np.float32)
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    return torch.from_numpy(decoded).permute(2, 0, 1).to(image.device, image.dtype) / 255.0


def sample_parameters(rng: random.Random) -> DegradationParameters:
    blur, noise, compression, aliasing = (rng.random() for _ in range(4))
    sigma_x = 0.25 + 2.25 * blur
    sigma_y = 0.25 + 2.25 * rng.random() * blur
    angle = rng.uniform(0, 180)
    quality = int(round(100 - 50 * compression))
    return DegradationParameters(blur, noise, compression, aliasing, sigma_x, sigma_y, angle, quality)


def degrade(hr: torch.Tensor, scale: int = 3, params: DegradationParameters | None = None, seed: int | None = None) -> tuple[torch.Tensor, DegradationParameters]:
    if scale != 3:
        raise ValueError("The approved degradation pipeline supports exactly 3x")
    if hr.ndim != 3:
        raise ValueError("Expected CHW image")
    rng = random.Random(seed)
    params = params or sample_parameters(rng)
    kernel_size = 3 if max(params.sigma_x, params.sigma_y) < 0.8 else (5 if max(params.sigma_x, params.sigma_y) < 1.5 else 7)
    kernel = _anisotropic_kernel(kernel_size, params.sigma_x, params.sigma_y, np.deg2rad(params.angle_degrees), hr.device, hr.dtype)
    bands = hr.shape[0]
    weight = kernel.view(1, 1, kernel_size, kernel_size).expand(bands, 1, -1, -1)
    blurred = F.conv2d(hr.unsqueeze(0), weight, padding=kernel_size // 2, groups=bands)
    mode = "bicubic" if params.aliasing < 0.5 else "bilinear"
    lr = F.interpolate(blurred, size=(hr.shape[-2] // 3, hr.shape[-1] // 3), mode=mode, align_corners=False, antialias=True)[0]
    generator = torch.Generator(device=hr.device).manual_seed(0 if seed is None else seed)
    gaussian_std = 0.035 * params.noise
    poisson_level = 80.0 - 60.0 * params.noise
    if gaussian_std:
        lr = lr + torch.randn(lr.shape, generator=generator, device=lr.device, dtype=lr.dtype) * gaussian_std
    if params.noise > 0.2:
        lr = torch.poisson(lr.clamp(0, 1) * poisson_level, generator=generator) / poisson_level
    lr = _jpeg_roundtrip(lr.clamp(0, 1), params.jpeg_quality)
    lr = torch.round(lr.clamp(0, 1) * 255) / 255
    return lr, params
