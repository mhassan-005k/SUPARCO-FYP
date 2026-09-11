from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_remote(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https", "s3"}


def _window(value: tuple[int, int, int, int] | None, width: int, height: int) -> Window | None:
    if value is None:
        return None
    column, row, window_width, window_height = map(int, value)
    if column < 0 or row < 0 or window_width <= 0 or window_height <= 0:
        raise ValueError("Window must have non-negative origin and positive dimensions")
    if column + window_width > width or row + window_height > height:
        raise ValueError("Window exceeds Sentinel-2 source dimensions")
    return Window(column, row, window_width, window_height)


def prepare_rgb8(
    red: str,
    green: str,
    blue: str,
    output: str,
    quantification: float = 10000.0,
    additive_offset: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 0.3,
    window: tuple[int, int, int, int] | None = None,
    source_context: dict[str, Any] | None = None,
) -> dict:
    """Create a documented uint8 B4/B3/B2 product from aligned Sentinel-2 L2A bands."""
    if quantification <= 0 or clip_max <= clip_min:
        raise ValueError("Quantification must be positive and clip_max must exceed clip_min")
    sources = [red, green, blue]
    band_names = ["B04", "B03", "B02"]
    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    reference = None
    source_metadata = []
    selected_window = None
    for band_name, source in zip(band_names, sources):
        with rasterio.open(source) as dataset:
            signature = (dataset.width, dataset.height, dataset.crs, dataset.transform)
            if reference is None:
                reference = signature
                profile = dataset.profile.copy()
                selected_window = _window(window, dataset.width, dataset.height)
            elif signature != reference:
                raise ValueError("Sentinel-2 RGB bands must share dimensions, CRS, and affine transform")
            values = dataset.read(1, window=selected_window).astype(np.float32)
            masks.append(dataset.read_masks(1, window=selected_window) > 0)
            reflectance = (values + additive_offset) / quantification
            normalized = (np.clip(reflectance, clip_min, clip_max) - clip_min) / (clip_max - clip_min)
            arrays.append(np.rint(normalized * 255).astype(np.uint8))
            record = {
                "band": band_name,
                "href": source,
                "dtype": dataset.dtypes[0],
                "width": dataset.width,
                "height": dataset.height,
                "crs": str(dataset.crs),
            }
            if not _is_remote(source):
                path = Path(source)
                record.update({"path": str(path.resolve()), "sha256": _sha256(path)})
            source_metadata.append(record)
    if selected_window is not None:
        profile.update(
            width=int(selected_window.width),
            height=int(selected_window.height),
            transform=window_transform(selected_window, profile["transform"]),
        )
    profile.update(driver="GTiff", count=3, dtype="uint8", compress="deflate", nodata=None)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(target, "w", **profile) as destination:
        destination.write(np.stack(arrays))
        destination.write_mask(np.logical_and.reduce(masks).astype(np.uint8) * 255)
        destination.descriptions = ("B04_red", "B03_green", "B02_blue")
        destination.update_tags(FAITHSR_DERIVED_PRODUCT="Sentinel-2_L2A_RGB_uint8", FAITHSR_QUANTIFICATION=str(quantification), FAITHSR_ADDITIVE_OFFSET=str(additive_offset), FAITHSR_REFLECTANCE_CLIP=f"{clip_min},{clip_max}")
    manifest = {
        "output": str(target.resolve()), "output_sha256": _sha256(target), "dtype": "uint8",
        "band_order": ["B04", "B03", "B02"], "quantification": quantification,
        "additive_offset": additive_offset, "reflectance_clip": [clip_min, clip_max],
        "mapping": "round(clip(((DN + additive_offset) / quantification - clip_min) / (clip_max - clip_min), 0, 1) * 255)",
        "sources": source_metadata,
        "window": None if selected_window is None else {
            "column": int(selected_window.col_off), "row": int(selected_window.row_off),
            "width": int(selected_window.width), "height": int(selected_window.height),
        },
        "source_context": source_context or {},
    }
    target.with_suffix(target.suffix + ".derivation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_stac_rgb8(item: dict[str, Any], output: str, window: tuple[int, int, int, int] | None = None, clip_min: float = 0.0, clip_max: float = 0.3) -> dict:
    """Derive RGB8 from an Element84-style Sentinel-2 L2A STAC item with provenance."""
    assets = item.get("assets", {})
    missing = [name for name in ("red", "green", "blue") if name not in assets]
    if missing:
        raise ValueError(f"STAC item is missing RGB assets: {', '.join(missing)}")
    raster_bands = [assets[name].get("raster:bands", [{}])[0] for name in ("red", "green", "blue")]
    scales = [float(band.get("scale", 0.0001)) for band in raster_bands]
    offsets = [float(band.get("offset", 0.0)) for band in raster_bands]
    if max(scales) - min(scales) > 1e-12 or max(offsets) - min(offsets) > 1e-12 or scales[0] <= 0:
        raise ValueError("RGB STAC assets must use one positive radiometric scale and offset")
    canonical_item = json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8")
    context = {
        "stac_item_id": item.get("id"),
        "collection": item.get("collection"),
        "datetime": item.get("properties", {}).get("datetime"),
        "cloud_cover": item.get("properties", {}).get("eo:cloud_cover"),
        "proj_epsg": item.get("properties", {}).get("proj:epsg"),
        "bbox": item.get("bbox"),
        "stac_item_sha256": hashlib.sha256(canonical_item).hexdigest(),
        "radiometric_scale": scales[0],
        "radiometric_offset": offsets[0],
    }
    return prepare_rgb8(
        assets["red"]["href"], assets["green"]["href"], assets["blue"]["href"], output,
        quantification=1.0 / scales[0], additive_offset=offsets[0] / scales[0],
        clip_min=clip_min, clip_max=clip_max, window=window, source_context=context,
    )


def tile_rgb8_scene(
    scene: str | Path,
    output: str | Path,
    tile_size: int = 192,
    min_valid_fraction: float = 0.98,
    min_nonzero_fraction: float = 0.0,
    min_channel_std: float = 0.0,
) -> list[dict[str, Any]]:
    """Split one derived RGB8 GeoTIFF into georeferenced, scene-grouped training tiles."""
    if tile_size <= 0 or tile_size % 3:
        raise ValueError("Tile size must be positive and divisible by three")
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be within [0,1]")
    if not 0 <= min_nonzero_fraction <= 1:
        raise ValueError("min_nonzero_fraction must be within [0,1]")
    if min_channel_std < 0:
        raise ValueError("min_channel_std must be non-negative")
    source_path, target_dir = Path(scene), Path(output)
    target_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with rasterio.open(source_path) as source:
        if source.count < 3 or any(dtype != "uint8" for dtype in source.dtypes[:3]):
            raise ValueError("Tiling requires a three-band uint8 derived scene")
        base_profile = source.profile.copy()
        base_tags = source.tags()
        for row in range(0, source.height - tile_size + 1, tile_size):
            for column in range(0, source.width - tile_size + 1, tile_size):
                selected = Window(column, row, tile_size, tile_size)
                valid = source.dataset_mask(window=selected) > 0
                valid_fraction = float(valid.mean())
                if valid_fraction < min_valid_fraction:
                    continue
                pixels = source.read([1, 2, 3], window=selected)
                nonzero_fraction = float(np.any(pixels > 0, axis=0).mean())
                mean_channel_std = float(pixels.reshape(3, -1).std(axis=1).mean())
                if nonzero_fraction < min_nonzero_fraction or mean_channel_std < min_channel_std:
                    continue
                tile_path = target_dir / f"r{row:05d}_c{column:05d}.tif"
                profile = base_profile.copy()
                profile.update(width=tile_size, height=tile_size, transform=source.window_transform(selected))
                with rasterio.open(tile_path, "w", **profile) as destination:
                    destination.write(pixels)
                    destination.write_mask(valid.astype(np.uint8) * 255)
                    destination.descriptions = source.descriptions[:3]
                    destination.update_tags(**base_tags, FAITHSR_PARENT_SCENE=source_path.name, FAITHSR_PARENT_WINDOW=f"{column},{row},{tile_size},{tile_size}")
                records.append({
                    "path": str(tile_path.resolve()),
                    "sha256": _sha256(tile_path),
                    "window": {"column": column, "row": row, "width": tile_size, "height": tile_size},
                    "valid_fraction": valid_fraction,
                    "nonzero_fraction": nonzero_fraction,
                    "mean_channel_std": mean_channel_std,
                    "transform": list(profile["transform"])[:6],
                    "crs": str(source.crs),
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive a documented 8-bit RGB GeoTIFF from aligned Sentinel-2 L2A bands")
    parser.add_argument("--red", required=True); parser.add_argument("--green", required=True); parser.add_argument("--blue", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--quantification", type=float, default=10000.0); parser.add_argument("--additive-offset", type=float, default=0.0)
    parser.add_argument("--clip-min", type=float, default=0.0); parser.add_argument("--clip-max", type=float, default=0.3)
    print(json.dumps(prepare_rgb8(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
