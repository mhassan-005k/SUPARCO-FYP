from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F

from .geotiff import apply_metadata, capture_metadata, metadata_summary, nearest_mask_3x, output_profile


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
def tiled_super_resolve(image: np.ndarray, model: torch.nn.Module, device: str | torch.device, tile_size: int = 512, overlap: int = 32, precision: str = "fp32") -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 CHW input")
    device = torch.device(device)
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 inference requires a CUDA device")
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
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=precision == "fp16"):
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


def process_tiff(input_path: str | Path, output_path: str | Path, model: torch.nn.Module, device: str | torch.device = "cpu", tile_size: int = 512, overlap: int = 32, model_version: str = "unversioned", compression: str = "input", precision: str = "fp32") -> dict:
    """Disk-backed overlap/blend inference; memory stays bounded by one tile and one output block."""
    started = time.perf_counter()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    uncertainty_target = target.with_suffix(".uncertainty.tiff")
    device = torch.device(device)
    model = model.to(device).eval()
    read_seconds = inference_seconds = accumulator_seconds = 0.0
    uncertainty_seen = False
    uncertainty_sum = uncertainty_count = 0.0
    uncertainty_histogram = np.zeros(1000, dtype=np.int64)
    with rasterio.open(input_path) as source, tempfile.TemporaryDirectory(prefix="faithsr_stream_", dir=target.parent) as temporary:
        metadata = capture_metadata(source, include_mask=False)
        before = metadata_summary(metadata)
        profile = output_profile(metadata)
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
        with rasterio.open(temporary / "sum.tif", "w+", **float_profile) as summed, rasterio.open(temporary / "weights.tif", "w+", **weight_profile) as weights, rasterio.open(temporary / "uncertainty.tif", "w+", **weight_profile) as uncertainty_sum_raster:
            for top in _positions(source.height, tile_size, overlap):
                for left in _positions(source.width, tile_size, overlap):
                    height, width = min(tile_size, source.height - top), min(tile_size, source.width - left)
                    input_window = Window(left, top, width, height)
                    mark = time.perf_counter(); pixels = source.read(window=input_window); read_seconds += time.perf_counter() - mark
                    mark = time.perf_counter(); sr, uncertainty = tiled_super_resolve(pixels, model, device, tile_size, overlap, precision); inference_seconds += time.perf_counter() - mark
                    output_window = Window(left * 3, top * 3, width * 3, height * 3)
                    blend = _blend_window(height * 3, width * 3, torch.device("cpu")).numpy().astype(np.float32)
                    mark = time.perf_counter()
                    summed.write(summed.read(window=output_window) + sr.astype(np.float32) * blend, window=output_window)
                    weights.write(weights.read(window=output_window) + blend, window=output_window)
                    if uncertainty is not None:
                        uncertainty_seen = True
                        uncertainty_sum_raster.write(uncertainty_sum_raster.read(window=output_window) + uncertainty.astype(np.float32) * blend, window=output_window)
                    accumulator_seconds += time.perf_counter() - mark
            with rasterio.open(target, "w", **profile) as output:
                apply_metadata(output, metadata)
                risk_output = None
                if uncertainty_seen:
                    risk_profile = profile.copy(); risk_profile.update(count=1, dtype="float32", nodata=None)
                    risk_output = rasterio.open(uncertainty_target, "w", **risk_profile)
                    risk_output.update_tags(FAITHSR_PRODUCT="uncertainty_risk", FAITHSR_SCALE="3", FAITHSR_RANGE="0,1")
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
                            mask = nearest_mask_3x(source.dataset_mask(window=Window(column, row, width, height)))
                            mask_window = Window(column * 3, row * 3, width * 3, height * 3)
                            output.write_mask(mask, window=mask_window)
                            if risk_output is not None: risk_output.write_mask(mask, window=mask_window)
                finally:
                    if risk_output is not None: risk_output.close()
    write_finished = time.perf_counter()
    expected = [metadata.profile["count"], metadata.profile["height"] * 3, metadata.profile["width"] * 3]
    audit = {
        "output": str(target.resolve()), "input_shape": [metadata.profile["count"], metadata.profile["height"], metadata.profile["width"]],
        "output_shape": expected, "dtype": "uint8", "scale": 3, "crs": None if metadata.profile.get("crs") is None else str(metadata.profile["crs"]),
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
        "streaming": {"disk_backed": True, "tile_size": tile_size, "overlap": overlap, "full_raster_allocated": False, "compression": profile.get("compress", "none"), "precision": precision},
        "uncertainty": None if not uncertainty_seen else {
            "mean": uncertainty_sum / max(1, uncertainty_count),
            "p95_approx": (p95_index + 0.5) / 1000,
            "artifact": str(uncertainty_target.resolve()),
        },
    })
    audit_path = Path(output_path).with_suffix(Path(output_path).suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
