from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .metrics import reconstruction_metrics_per_sample, uncertainty_metrics_per_sample


def representative_sample_index(dataset: Any) -> tuple[int, float]:
    """Choose a deterministic median-texture sample without using model results."""

    if len(dataset) == 0:
        raise ValueError("Cannot select a representative sample from an empty dataset")
    ranked: list[tuple[float, int]] = []
    for index in range(len(dataset)):
        _, hr, *_ = dataset[index]
        ranked.append((float(hr.detach().float().std().cpu()), index))
    ranked.sort(key=lambda value: (value[0], value[1]))
    texture, index = ranked[(len(ranked) - 1) // 2]
    return index, texture


def _rgb_uint8(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("Qualitative preview accepts one image at a time")
        array = array[0]
    if array.ndim != 3:
        raise ValueError("Expected CHW or BCHW image")
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    elif array.shape[0] == 2:
        array = np.concatenate([array, array[-1:]], axis=0)
    array = array[:3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(np.rint(array * 255), 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array.transpose(1, 2, 0)


def _label(image: Image.Image, text: str) -> Image.Image:
    header = 24
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text((6, 6), text, fill="black")
    return canvas


def save_qualitative_panel(
    output: str | Path,
    lr: torch.Tensor,
    hr: torch.Tensor,
    sr: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    extra: dict[str, Any] | None = None,
    method_label: str = "FaithSR-3X",
) -> dict[str, Any]:
    """Save a reviewable held-out prediction panel and machine-readable metrics."""

    lr_b = lr.unsqueeze(0) if lr.ndim == 3 else lr
    hr_b = hr.unsqueeze(0) if hr.ndim == 3 else hr
    sr_b = sr.unsqueeze(0) if sr.ndim == 3 else sr
    if lr_b.shape[0] != 1 or hr_b.shape[0] != 1 or sr_b.shape[0] != 1:
        raise ValueError("Qualitative preview accepts one image at a time")
    if hr_b.shape != sr_b.shape:
        raise ValueError("SR and HR preview tensors must have identical shapes")
    # Training keeps the deterministic validation sample on CPU while its
    # prediction may be on CUDA. Align evidence tensors without changing values.
    lr_b = lr_b.to(sr_b.device)
    hr_b = hr_b.to(sr_b.device)

    lr_image = Image.fromarray(_rgb_uint8(lr_b))
    height, width = hr_b.shape[-2:]
    scale_y, scale_x = height / lr_b.shape[-2], width / lr_b.shape[-1]
    scale_label = f"{scale_y:g}x" if scale_y == scale_x else f"{scale_y:g}x{scale_x:g}"
    metric_scale = int(scale_y) if scale_y == scale_x and scale_y.is_integer() else 3
    bicubic_b = F.interpolate(lr_b.float(), size=(height, width), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    bicubic_image = Image.fromarray(_rgb_uint8(bicubic_b))
    sr_image = Image.fromarray(_rgb_uint8(sr_b.clamp(0, 1)))
    hr_image = Image.fromarray(_rgb_uint8(hr_b.clamp(0, 1)))

    error = (sr_b.float() - hr_b.float()).abs().mean(dim=1)[0].clamp(0, 1).detach().cpu().numpy()
    error_u8 = np.clip(np.rint(error * 255 * 4), 0, 255).astype(np.uint8)
    error_rgb = np.stack([error_u8, np.zeros_like(error_u8), 255 - error_u8], axis=-1)

    uncertainty_b = None
    if uncertainty is not None:
        uncertainty_b = uncertainty.unsqueeze(0) if uncertainty.ndim == 3 else uncertainty
    metric_tensors = reconstruction_metrics_per_sample(sr_b.clamp(0, 1), hr_b, metric_scale)
    bicubic_metric_tensors = reconstruction_metrics_per_sample(bicubic_b, hr_b, metric_scale)
    if uncertainty_b is not None:
        metric_tensors.update(uncertainty_metrics_per_sample(uncertainty_b, sr_b.clamp(0, 1), hr_b))
    metrics = {name: float(value[0].detach().cpu()) for name, value in metric_tensors.items()}
    bicubic_metrics = {name: float(value[0].detach().cpu()) for name, value in bicubic_metric_tensors.items()}

    panels = [
        _label(lr_image.resize((width, height), Image.Resampling.NEAREST), "Observed LR"),
        _label(bicubic_image, f'Bicubic {scale_label} | {bicubic_metrics["psnr"]:.2f} dB'),
        _label(sr_image, f'{method_label} | {metrics["psnr"]:.2f} dB'),
        _label(hr_image, "Held-out HR"),
        _label(Image.fromarray(error_rgb), "Error: blue low / red high"),
    ]

    if uncertainty is not None:
        risk = uncertainty_b.detach().float().cpu()[0, 0].clamp(0, 1).numpy()
        risk_u8 = np.rint(risk * 255).astype(np.uint8)
        risk_rgb = np.stack([risk_u8, risk_u8 // 2, np.zeros_like(risk_u8)], axis=-1)
        panels.append(_label(Image.fromarray(risk_rgb), "Risk: bright = high"))

    display_scale = max(1, (240 + width - 1) // width)
    display_width, display_height = width * display_scale, (height + 24) * display_scale
    panels = [panel.resize((display_width, display_height), Image.Resampling.NEAREST) for panel in panels]
    canvas = Image.new("RGB", (display_width * len(panels), display_height), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * display_width, 0))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, optimize=True)

    payload: dict[str, Any] = {
        "artifact": str(target.resolve()),
        "input_shape": list(lr_b.shape),
        "output_shape": list(sr_b.shape),
        "display_scale": display_scale,
        "metrics": metrics,
        "bicubic_metrics": bicubic_metrics,
        "method_minus_bicubic_psnr_db": metrics["psnr"] - bicubic_metrics["psnr"],
        "method_label": method_label,
        "panels": ["lr", "bicubic", "method", "hr", "absolute_error"] + (["uncertainty"] if uncertainty_b is not None else []),
    }
    if method_label == "FaithSR-3X":
        payload["faithsr_minus_bicubic_psnr_db"] = payload["method_minus_bicubic_psnr_db"]
    if extra:
        payload.update(extra)
    target.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
