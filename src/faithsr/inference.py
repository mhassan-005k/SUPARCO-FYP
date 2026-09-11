from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F

from .geotiff import apply_metadata, capture_metadata, metadata_summary, nearest_mask, output_profile


def _positions(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or overlap < 0 or tile <= overlap:
        raise ValueError("Tile size must be positive and larger than a non-negative overlap")
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


def _resolve_scale(model: torch.nn.Module, scale: int | None = None) -> int:
    config = getattr(model, "config", None)
    inferred = getattr(model, "scale", None)
    if inferred is None and config is not None:
        inferred = getattr(config, "scale", getattr(config, "upscale", None))
    resolved = int(scale if scale is not None else (inferred if inferred is not None else 3))
    if resolved < 3:
        raise ValueError("Output scale must be greater than or equal to 3")
    return resolved


@torch.inference_mode()
def tiled_super_resolve(image: np.ndarray, model: torch.nn.Module, device: str | torch.device, tile_size: int = 512, overlap: int = 32, precision: str = "fp32", scale: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 CHW input")
    device = torch.device(device)
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 inference requires a CUDA device")
    model = model.to(device).eval()
    scale = _resolve_scale(model, scale)
    bands, height, width = image.shape
    output = torch.zeros((bands, height * scale, width * scale), device=device)
    weights = torch.zeros((1, height * scale, width * scale), device=device)
    uncertainty_output = torch.zeros((1, height * scale, width * scale), device=device)
    has_uncertainty = False
    for top in _positions(height, tile_size, overlap):
        for left in _positions(width, tile_size, overlap):
            tile = torch.from_numpy(image[:, top:min(top + tile_size, height), left:min(left + tile_size, width)].copy()).to(device).float() / 255
            original_height, original_width = tile.shape[-2:]
            pad_height, pad_width = tile_size - original_height, tile_size - original_width
            if pad_height or pad_width:
                mode = "reflect" if min(original_height, original_width) > 1 else "replicate"
                tile = F.pad(tile, (0, pad_width, 0, pad_height), mode=mode)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=precision == "fp16"):
                sr, aux = model(tile.unsqueeze(0), return_aux=True)
            sr = sr[0, :, : original_height * scale, : original_width * scale].clamp(0, 1)
            expected_tile_shape = (bands, original_height * scale, original_width * scale)
            if tuple(sr.shape) != expected_tile_shape:
                raise ValueError(f"Model output does not match declared {scale}x scale: expected {expected_tile_shape}, received {tuple(sr.shape)}")
            window = _blend_window(original_height * scale, original_width * scale, device)
            out_top, out_left = top * scale, left * scale
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


def process_tiff(input_path: str | Path, output_path: str | Path, model: torch.nn.Module, device: str | torch.device = "cpu", tile_size: int = 512, overlap: int = 32, model_version: str = "unversioned", compression: str = "input", precision: str = "fp32", progress_callback: Callable[[int, int], None] | None = None, scale: int | None = None) -> dict:
    """Disk-backed overlap/blend inference; memory stays bounded by one tile and one output block."""
    started = time.perf_counter()
    target = Path(output_path)
    source_path = Path(input_path)
    if source_path.resolve() == target.resolve():
        raise ValueError("Input and output TIFF paths must be different")
    _positions(1, tile_size, overlap)
    target.parent.mkdir(parents=True, exist_ok=True)
    uncertainty_target = target.with_suffix(".uncertainty.tiff")
    device = torch.device(device)
    model = model.to(device).eval()
    scale = _resolve_scale(model, scale)
    read_seconds = inference_seconds = accumulator_seconds = 0.0
    uncertainty_seen = False
    uncertainty_sum = uncertainty_count = 0.0
    uncertainty_histogram = np.zeros(1000, dtype=np.int64)
    with rasterio.open(input_path) as source, tempfile.TemporaryDirectory(prefix="faithsr_stream_", dir=target.parent) as temporary:
        metadata = capture_metadata(source, include_mask=False)
        model_bands = getattr(getattr(model, "config", None), "in_channels", source.count)
        if source.count != model_bands:
            raise ValueError(f"Input TIFF has {source.count} bands but checkpoint expects {model_bands}")
        before = metadata_summary(metadata)
        profile = output_profile(metadata, scale)
        if compression == "none":
            profile.pop("compress", None)
        elif compression != "input":
            if compression not in {"deflate", "lzw", "zstd"}:
                raise ValueError("compression must be input, deflate, lzw, zstd, or none")
            profile["compress"] = compression
        if profile.get("compress") in {"deflate", "zstd"}:
            profile["num_threads"] = "ALL_CPUS"
        float_profile = profile.copy(); float_profile.update(driver="GTiff", dtype="float32", compress="none"); float_profile.pop("num_threads", None)
        weight_profile = float_profile.copy(); weight_profile.update(count=1)
        temporary = Path(temporary)
        top_positions = _positions(source.height, tile_size, overlap)
        left_positions = _positions(source.width, tile_size, overlap)
        total_tiles = len(top_positions) * len(left_positions)
        completed_tiles = 0
        with rasterio.open(temporary / "sum.tif", "w+", **float_profile) as summed, rasterio.open(temporary / "weights.tif", "w+", **weight_profile) as weights, rasterio.open(temporary / "uncertainty.tif", "w+", **weight_profile) as uncertainty_sum_raster:
            for top in top_positions:
                for left in left_positions:
                    height, width = min(tile_size, source.height - top), min(tile_size, source.width - left)
                    input_window = Window(left, top, width, height)
                    mark = time.perf_counter(); pixels = source.read(window=input_window); read_seconds += time.perf_counter() - mark
                    mark = time.perf_counter(); sr, uncertainty = tiled_super_resolve(pixels, model, device, tile_size, overlap, precision, scale); inference_seconds += time.perf_counter() - mark
                    output_window = Window(left * scale, top * scale, width * scale, height * scale)
                    blend = _blend_window(height * scale, width * scale, torch.device("cpu")).numpy().astype(np.float32)
                    mark = time.perf_counter()
                    summed.write(summed.read(window=output_window) + sr.astype(np.float32) * blend, window=output_window)
                    weights.write(weights.read(window=output_window) + blend, window=output_window)
                    if uncertainty is not None:
                        uncertainty_seen = True
                        uncertainty_sum_raster.write(uncertainty_sum_raster.read(window=output_window) + uncertainty.astype(np.float32) * blend, window=output_window)
                    accumulator_seconds += time.perf_counter() - mark
                    completed_tiles += 1
                    if progress_callback is not None:
                        progress_callback(completed_tiles, total_tiles)
            with rasterio.open(target, "w", **profile) as output:
                apply_metadata(output, metadata, scale)
                risk_output = None
                if uncertainty_seen:
                    risk_profile = profile.copy(); risk_profile.update(count=1, dtype="float32", nodata=None)
                    risk_output = rasterio.open(uncertainty_target, "w", **risk_profile)
                    risk_output.update_tags(FAITHSR_PRODUCT="uncertainty_risk", FAITHSR_SCALE=str(scale), FAITHSR_RANGE="0,1")
                    risk_output.set_band_description(1, "FaithSR uncertainty risk")
                try:
                    block = 1024
                    for row in range(0, profile["height"], block):
                        for column in range(0, profile["width"], block):
                            height, width = min(block, profile["height"] - row), min(block, profile["width"] - column)
                            window = Window(column, row, width, height)
                            divisor = weights.read(1, window=window).clip(1e-8)
                            output.write(np.clip(np.rint(summed.read(window=window) / divisor[None]), 0, 255).astype(np.uint8), window=window)
                            if risk_output is not None:
                                risk = np.clip(uncertainty_sum_raster.read(1, window=window) / divisor, 0, 1).astype(np.float32)
                                risk_output.write(risk, 1, window=window)
                                uncertainty_sum += float(risk.sum()); uncertainty_count += risk.size
                                uncertainty_histogram += np.histogram(risk, bins=1000, range=(0, 1))[0]
                    for row in range(0, source.height, tile_size):
                        for column in range(0, source.width, tile_size):
                            height, width = min(tile_size, source.height - row), min(tile_size, source.width - column)
                            mask = nearest_mask(source.dataset_mask(window=Window(column, row, width, height)), scale)
                            mask_window = Window(column * scale, row * scale, width * scale, height * scale)
                            output.write_mask(mask, window=mask_window)
                            if risk_output is not None: risk_output.write_mask(mask, window=mask_window)
                finally:
                    if risk_output is not None: risk_output.close()
    write_finished = time.perf_counter()
    expected = [metadata.profile["count"], metadata.profile["height"] * scale, metadata.profile["width"] * scale]
    audit = {
        "output": str(target.resolve()), "input_shape": [metadata.profile["count"], metadata.profile["height"], metadata.profile["width"]],
        "output_shape": expected, "dtype": "uint8", "scale": scale, "crs": None if metadata.profile.get("crs") is None else str(metadata.profile["crs"]),
        "metadata": {"dataset_tags": len(metadata.tags), "band_tags": [len(tags) for tags in metadata.band_tags], "tag_namespaces": sorted(metadata.namespaced_tags), "band_tag_namespaces": [sorted(namespaces) for namespaces in metadata.band_namespaced_tags], "colormap_bands": sorted(metadata.colormaps), "descriptions": list(metadata.descriptions), "nodata": metadata.profile.get("nodata"), "mask_preserved": True, "transform_recalculated": True},
        "warnings": [] if metadata.profile.get("crs") else ["Input has no CRS; output remains non-georeferenced."],
    }
    p95_index = int(np.searchsorted(np.cumsum(uncertainty_histogram), 0.95 * uncertainty_count)) if uncertainty_seen else None
    audit.update({
        "input": str(Path(input_path).resolve()),
        "input_sha256": file_sha256(input_path),
        "output_sha256": file_sha256(output_path),
        "model_version": model_version,
        "input_metadata": before,
        "timing_ms": {
            "read": read_seconds * 1000,
            "inference": inference_seconds * 1000,
            "disk_accumulation": accumulator_seconds * 1000,
            "finalize_and_write": max(0.0, (write_finished - started) - read_seconds - inference_seconds - accumulator_seconds) * 1000,
            "total": (write_finished - started) * 1000,
        },
        "streaming": {"disk_backed": True, "tile_size": tile_size, "overlap": overlap, "tiles_processed": total_tiles, "full_raster_allocated": False, "compression": profile.get("compress", "none"), "precision": precision},
        "uncertainty": None if not uncertainty_seen else {
            "mean": uncertainty_sum / max(1, uncertainty_count),
            "p95_approx": (p95_index + 0.5) / 1000,
            "artifact": str(uncertainty_target.resolve()),
        },
    })
    audit_path = Path(output_path).with_suffix(Path(output_path).suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
