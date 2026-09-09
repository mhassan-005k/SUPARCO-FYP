from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_rgb8(red: str, green: str, blue: str, output: str, quantification: float = 10000.0, additive_offset: float = 0.0, clip_min: float = 0.0, clip_max: float = 0.3) -> dict:
    """Create a documented uint8 B4/B3/B2 product from aligned Sentinel-2 L2A bands."""
    if quantification <= 0 or clip_max <= clip_min:
        raise ValueError("Quantification must be positive and clip_max must exceed clip_min")
    sources = [Path(red), Path(green), Path(blue)]
    arrays: list[np.ndarray] = []
    reference = None
    source_metadata = []
    for path in sources:
        with rasterio.open(path) as dataset:
            signature = (dataset.width, dataset.height, dataset.crs, dataset.transform)
            if reference is None:
                reference = signature
                profile = dataset.profile.copy()
            elif signature != reference:
                raise ValueError("Sentinel-2 RGB bands must share dimensions, CRS, and affine transform")
            values = dataset.read(1).astype(np.float32)
            reflectance = (values + additive_offset) / quantification
            normalized = (np.clip(reflectance, clip_min, clip_max) - clip_min) / (clip_max - clip_min)
            arrays.append(np.rint(normalized * 255).astype(np.uint8))
            source_metadata.append({"path": str(path.resolve()), "sha256": _sha256(path), "dtype": dataset.dtypes[0]})
    profile.update(driver="GTiff", count=3, dtype="uint8", compress="deflate", nodata=None)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(target, "w", **profile) as destination:
        destination.write(np.stack(arrays))
        destination.descriptions = ("B04_red", "B03_green", "B02_blue")
        destination.update_tags(FAITHSR_DERIVED_PRODUCT="Sentinel-2_L2A_RGB_uint8", FAITHSR_QUANTIFICATION=str(quantification), FAITHSR_ADDITIVE_OFFSET=str(additive_offset), FAITHSR_REFLECTANCE_CLIP=f"{clip_min},{clip_max}")
    manifest = {
        "output": str(target.resolve()), "output_sha256": _sha256(target), "dtype": "uint8",
        "band_order": ["B04", "B03", "B02"], "quantification": quantification,
        "additive_offset": additive_offset, "reflectance_clip": [clip_min, clip_max],
        "mapping": "round(clip(((DN + additive_offset) / quantification - clip_min) / (clip_max - clip_min), 0, 1) * 255)",
        "sources": source_metadata,
    }
    target.with_suffix(target.suffix + ".derivation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive a documented 8-bit RGB GeoTIFF from aligned Sentinel-2 L2A bands")
    parser.add_argument("--red", required=True); parser.add_argument("--green", required=True); parser.add_argument("--blue", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--quantification", type=float, default=10000.0); parser.add_argument("--additive-offset", type=float, default=0.0)
    parser.add_argument("--clip-min", type=float, default=0.0); parser.add_argument("--clip-max", type=float, default=0.3)
    print(json.dumps(prepare_rgb8(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
