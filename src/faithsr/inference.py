from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from .geotiff import capture_metadata, metadata_summary, write_raster


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= overlap:
        raise ValueError("Tile size must be larger than overlap")
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, tile - overlap))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _blend_window(height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.hann_window(height, periodic=False, device=device).clamp_min(1e-3)
    x = torch.hann_window(width, periodic=False, device=device).clamp_min(1e-3)
    return torch.outer(y, x)[None]


@torch.inference_mode()
def tiled_super_resolve(image: np.ndarray, model: torch.nn.Module, device: str | torch.device, tile_size: int = 512, overlap: int = 32) -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 CHW input")
    device = torch.device(device)
    model = model.to(device).eval()
    bands, height, width = image.shape
    output = torch.zeros((bands, height * 3, width * 3), device=device)
    weights = torch.zeros((1, height * 3, width * 3), device=device)
    uncertainty_output = torch.zeros((1, height * 3, width * 3), device=device)
    has_uncertainty = False
    for top in _positions(height, tile_size, overlap):
        for left in _positions(width, tile_size, overlap):
            tile = torch.from_numpy(image[:, top:min(top + tile_size, height), left:min(left + tile_size, width)].copy()).to(device).float() / 255
            original_height, original_width = tile.shape[-2:]
            pad_height, pad_width = tile_size - original_height, tile_size - original_width
            if pad_height or pad_width:
                mode = "reflect" if min(original_height, original_width) > 1 else "replicate"
                tile = F.pad(tile, (0, pad_width, 0, pad_height), mode=mode)
            sr, aux = model(tile.unsqueeze(0), return_aux=True)
            sr = sr[0, :, : original_height * 3, : original_width * 3].clamp(0, 1)
            window = _blend_window(original_height * 3, original_width * 3, device)
            out_top, out_left = top * 3, left * 3
            output[:, out_top:out_top + sr.shape[-2], out_left:out_left + sr.shape[-1]] += sr * window
            weights[:, out_top:out_top + sr.shape[-2], out_left:out_left + sr.shape[-1]] += window
            if aux.get("uncertainty") is not None:
                uncertainty = aux["uncertainty"][0, :, :sr.shape[-2], :sr.shape[-1]]
                uncertainty_output[:, out_top:out_top + sr.shape[-2], out_left:out_left + sr.shape[-1]] += uncertainty * window
                has_uncertainty = True
    output = (output / weights.clamp_min(1e-8) * 255).cpu().numpy()
    uncertainty_array = (uncertainty_output / weights.clamp_min(1e-8)).cpu().numpy() if has_uncertainty else None
    return output, uncertainty_array


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_tiff(input_path: str | Path, output_path: str | Path, model: torch.nn.Module, device: str | torch.device = "cpu", tile_size: int = 512, overlap: int = 32, model_version: str = "unversioned") -> dict:
    started = time.perf_counter()
    with rasterio.open(input_path) as source:
        metadata = capture_metadata(source)
        pixels = source.read()
        before = metadata_summary(metadata)
    read_finished = time.perf_counter()
    sr, uncertainty = tiled_super_resolve(pixels, model, device, tile_size, overlap)
    inference_finished = time.perf_counter()
    audit = write_raster(output_path, sr, metadata)
    write_finished = time.perf_counter()
    audit.update({
        "input": str(Path(input_path).resolve()),
        "input_sha256": file_sha256(input_path),
        "output_sha256": file_sha256(output_path),
        "model_version": model_version,
        "input_metadata": before,
        "timing_ms": {
            "read": (read_finished - started) * 1000,
            "inference_and_stitch": (inference_finished - read_finished) * 1000,
            "write": (write_finished - inference_finished) * 1000,
            "total": (write_finished - started) * 1000,
        },
        "uncertainty": None if uncertainty is None else {
            "mean": float(uncertainty.mean()),
            "p95": float(np.quantile(uncertainty, 0.95)),
        },
    })
    audit_path = Path(output_path).with_suffix(Path(output_path).suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
