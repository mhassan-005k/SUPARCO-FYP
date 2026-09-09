from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine


@dataclass
class RasterMetadata:
    profile: dict[str, Any]
    tags: dict[str, str]
    band_tags: list[dict[str, str]]
    descriptions: tuple[str | None, ...]
    units: tuple[str | None, ...]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    colorinterp: tuple[Any, ...]
    mask: np.ndarray


def validate_input(dataset: rasterio.io.DatasetReader) -> None:
    if not dataset.name.lower().endswith((".tif", ".tiff")):
        raise ValueError("Input must be a .tif or .tiff file")
    if any(dtype != "uint8" for dtype in dataset.dtypes):
        raise ValueError(f"Input must be 8-bit uint8; received {dataset.dtypes}")
    if dataset.count < 1:
        raise ValueError("TIFF has no raster bands")


def capture_metadata(dataset: rasterio.io.DatasetReader) -> RasterMetadata:
    validate_input(dataset)
    return RasterMetadata(
        profile=dataset.profile.copy(),
        tags=dataset.tags().copy(),
        band_tags=[dataset.tags(index) for index in dataset.indexes],
        descriptions=dataset.descriptions,
        units=dataset.units,
        scales=dataset.scales,
        offsets=dataset.offsets,
        colorinterp=dataset.colorinterp,
        mask=dataset.dataset_mask(),
    )


def output_profile(metadata: RasterMetadata, scale: int = 3) -> dict[str, Any]:
    if scale != 3:
        raise ValueError("Approved output scale is exactly 3x")
    profile = metadata.profile.copy()
    profile.update(
        width=int(profile["width"]) * scale,
        height=int(profile["height"]) * scale,
        dtype="uint8",
        transform=profile["transform"] @ Affine.scale(1 / scale, 1 / scale),
        count=int(profile["count"]),
    )
    if profile.get("driver") not in {"GTiff", "COG"}:
        profile["driver"] = "GTiff"
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.setdefault("compress", "deflate")
    return profile


def nearest_mask_3x(mask: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(mask, 3, axis=0), 3, axis=1)


def write_raster(path: str | Path, pixels: np.ndarray, metadata: RasterMetadata) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = (metadata.profile["count"], metadata.profile["height"] * 3, metadata.profile["width"] * 3)
    if pixels.shape != expected:
        raise ValueError(f"Expected output shape {expected}, received {pixels.shape}")
    pixels = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    profile = output_profile(metadata)
    with rasterio.open(target, "w", **profile) as output:
        output.write(pixels)
        output.update_tags(**metadata.tags)
        output.update_tags(FAITHSR_SCALE="3", FAITHSR_GSD="nominal_model_derived")
        for index, tags in enumerate(metadata.band_tags, start=1):
            output.update_tags(index, **tags)
        output.descriptions = metadata.descriptions
        output.units = metadata.units
        output.scales = metadata.scales
        output.offsets = metadata.offsets
        try:
            output.colorinterp = metadata.colorinterp
        except ValueError:
            pass
        output.write_mask(nearest_mask_3x(metadata.mask))
    return {
        "output": str(target.resolve()),
        "input_shape": [metadata.profile["count"], metadata.profile["height"], metadata.profile["width"]],
        "output_shape": list(expected),
        "dtype": "uint8",
        "scale": 3,
        "crs": None if metadata.profile.get("crs") is None else str(metadata.profile["crs"]),
        "metadata": {
            "dataset_tags": len(metadata.tags),
            "band_tags": [len(tags) for tags in metadata.band_tags],
            "descriptions": list(metadata.descriptions),
            "nodata": metadata.profile.get("nodata"),
            "mask_preserved": True,
            "transform_recalculated": True,
        },
        "warnings": [] if metadata.profile.get("crs") else ["Input has no CRS; output remains non-georeferenced."],
    }


def write_uncertainty_raster(path: str | Path, uncertainty: np.ndarray, metadata: RasterMetadata) -> str:
    expected = (1, metadata.profile["height"] * 3, metadata.profile["width"] * 3)
    if uncertainty.shape != expected:
        raise ValueError(f"Expected uncertainty shape {expected}, received {uncertainty.shape}")
    profile = output_profile(metadata)
    profile.update(count=1, dtype="float32", nodata=None)
    target = Path(path)
    with rasterio.open(target, "w", **profile) as output:
        output.write(np.clip(uncertainty, 0, 1).astype(np.float32))
        output.update_tags(FAITHSR_PRODUCT="uncertainty_risk", FAITHSR_SCALE="3", FAITHSR_RANGE="0,1")
        output.set_band_description(1, "FaithSR uncertainty risk")
        output.write_mask(nearest_mask_3x(metadata.mask))
    return str(target.resolve())


def metadata_summary(metadata: RasterMetadata) -> dict[str, Any]:
    profile = metadata.profile
    return {
        "width": profile["width"],
        "height": profile["height"],
        "bands": profile["count"],
        "dtype": profile["dtype"],
        "crs": None if profile.get("crs") is None else str(profile["crs"]),
        "transform": tuple(profile["transform"]),
        "descriptions": list(metadata.descriptions),
        "nodata": profile.get("nodata"),
        "tags": metadata.tags,
    }
