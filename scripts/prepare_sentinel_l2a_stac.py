"""Build a reproducible RGB8 training corpus from public Sentinel-2 L2A COG windows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import urllib.request

import numpy as np
import rasterio

from faithsr.sentinel import prepare_stac_rgb8, tile_rgb8_scene


DEFAULT_REGIONS = {
    "karachi": [66.75, 24.70, 67.25, 25.10],
    "lahore": [74.05, 31.30, 74.60, 31.75],
    "islamabad": [72.80, 33.45, 73.35, 33.90],
}
GOOD_SCL_CLASSES = {2, 4, 5, 6, 7}
MIN_GOOD_SCL_FRACTION = 0.95
MIN_POSITIVE_REFLECTANCE_FRACTION = 0.55


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "FaithSR-3X/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def query_items(endpoint: str, bbox: list[float], date_range: str, cloud_cover: float, limit: int) -> list[dict]:
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": date_range,
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": cloud_cover}},
    }
    return _post_json(endpoint.rstrip("/") + "/search", payload).get("features", [])


def _best_window(item: dict, size: int, overview: int = 128) -> tuple[int, int, int, int, dict[str, float]]:
    red, scl = item["assets"]["red"]["href"], item["assets"]["scl"]["href"]
    environment = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    with rasterio.Env(**environment), rasterio.open(red) as red_source, rasterio.open(scl) as scl_source:
        if size > min(red_source.width, red_source.height):
            raise ValueError("Requested scene window exceeds source dimensions")
        red_small = red_source.read(1, out_shape=(overview, overview))
        scl_small = scl_source.read(1, out_shape=(overview, overview))
        good = (red_small > 0) & np.isin(scl_small, list(GOOD_SCL_CLASSES))
        raster_band = item["assets"]["red"].get("raster:bands", [{}])[0]
        reflectance = red_small.astype(np.float32) * float(raster_band.get("scale", 0.0001)) + float(raster_band.get("offset", 0.0))
        block_x = max(1, math.ceil(size / red_source.width * overview))
        block_y = max(1, math.ceil(size / red_source.height * overview))
        best: tuple[float, float, float, int, int] | None = None
        for row in range(0, overview - block_y + 1):
            for column in range(0, overview - block_x + 1):
                quality = good[row:row + block_y, column:column + block_x]
                values = reflectance[row:row + block_y, column:column + block_x]
                good_fraction = float(quality.mean())
                positive_fraction = float(((values > 0.01) & quality).mean())
                valid_values = values[quality & (values > 0)]
                dynamic_range = float(np.percentile(valid_values, 90) - np.percentile(valid_values, 10)) if valid_values.size else 0.0
                eligible = good_fraction >= MIN_GOOD_SCL_FRACTION and positive_fraction >= MIN_POSITIVE_REFLECTANCE_FRACTION
                score = dynamic_range if eligible else -1.0 + good_fraction * max(dynamic_range, 0.0)
                candidate = (score, good_fraction, positive_fraction, -row, -column)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            raise RuntimeError("Could not select a valid Sentinel-2 window")
        score, good_fraction, positive_fraction, negative_row, negative_column = best
        if score < 0:
            raise RuntimeError("No radiometrically informative cloud-free Sentinel-2 window was found")
        coarse_row, coarse_column = -negative_row, -negative_column
        column = min(red_source.width - size, round(coarse_column / overview * red_source.width))
        row = min(red_source.height - size, round(coarse_row / overview * red_source.height))
        return int(column), int(row), size, size, {
            "good_scl_fraction": good_fraction,
            "positive_reflectance_fraction": positive_fraction,
            "red_reflectance_p90_minus_p10": score,
            "minimum_good_scl_fraction": MIN_GOOD_SCL_FRACTION,
            "minimum_positive_reflectance_fraction": MIN_POSITIVE_REFLECTANCE_FRACTION,
        }


def _regions(values: list[str] | None) -> dict[str, list[float]]:
    if not values:
        return DEFAULT_REGIONS
    parsed = {}
    for value in values:
        parts = value.split(",")
        if len(parts) != 5:
            raise ValueError("Each --region must be name,min_lon,min_lat,max_lon,max_lat")
        parsed[parts[0]] = [float(number) for number in parts[1:]]
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="https://earth-search.aws.element84.com/v1")
    parser.add_argument("--date-range", default="2025-10-01T00:00:00Z/2026-03-31T23:59:59Z")
    parser.add_argument("--cloud-cover", type=float, default=5.0)
    parser.add_argument("--per-region", type=int, default=2)
    parser.add_argument("--query-limit", type=int, default=30)
    parser.add_argument("--scene-size", type=int, default=1152)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=0.3)
    parser.add_argument("--region", action="append", help="name,min_lon,min_lat,max_lon,max_lat; repeatable")
    args = parser.parse_args()
    if args.per_region <= 0:
        raise ValueError("per-region must be positive")
    output = Path(args.output)
    scenes_dir, patches_dir, items_dir = output / "scenes", output / "patches", output / "stac-items"
    for directory in (scenes_dir, patches_dir, items_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Element84 Earth Search public STAC sentinel-2-l2a",
        "endpoint": args.endpoint,
        "date_range": args.date_range,
        "cloud_cover_lt": args.cloud_cover,
        "scene_size": args.scene_size,
        "tile_size": args.tile_size,
        "reflectance_clip": [args.clip_min, args.clip_max],
        "window_selection": {
            "minimum_good_scl_fraction": MIN_GOOD_SCL_FRACTION,
            "minimum_positive_reflectance_fraction": MIN_POSITIVE_REFLECTANCE_FRACTION,
            "ranking": "maximum red-reflectance p90-p10 among eligible overview blocks",
        },
        "tile_selection": {"minimum_nonzero_fraction": 0.75, "minimum_mean_channel_std_uint8": 2.0},
        "split_unit": "STAC scene parent directory; never random patch",
        "regions": [],
        "scenes": [],
    }
    seen: set[str] = set()
    for region, bbox in _regions(args.region).items():
        candidates = query_items(args.endpoint, bbox, args.date_range, args.cloud_cover, args.query_limit)
        selected: list[tuple[dict, tuple[int, int, int, int, dict[str, float]]]] = []
        rejected = []
        for item in candidates:
            if item.get("id") in seen or not all(key in item.get("assets", {}) for key in ("red", "green", "blue", "scl")):
                continue
            try:
                selected.append((item, _best_window(item, args.scene_size)))
            except RuntimeError as error:
                rejected.append({"item_id": item.get("id"), "reason": str(error)})
            if len(selected) == args.per_region:
                break
        if len(selected) < args.per_region:
            raise RuntimeError(f"Region {region} returned only {len(selected)} usable scenes")
        manifest["regions"].append({"name": region, "bbox": bbox, "returned": len(candidates), "selected": [item["id"] for item, _ in selected], "rejected": rejected})
        for item, selected_window in selected:
            seen.add(item["id"])
            item_path = items_dir / f'{item["id"]}.json'
            item_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
            column, row, width, height, selection = selected_window
            scene_path = scenes_dir / f'{item["id"]}.tif'
            derivation = prepare_stac_rgb8(item, str(scene_path), (column, row, width, height), args.clip_min, args.clip_max)
            derivation["region"] = region
            derivation["window_selection"] = selection
            derivation_path = scene_path.with_suffix(scene_path.suffix + ".derivation.json")
            derivation_path.write_text(json.dumps(derivation, indent=2), encoding="utf-8")
            tiles = tile_rgb8_scene(
                scene_path,
                patches_dir / item["id"],
                args.tile_size,
                min_nonzero_fraction=0.75,
                min_channel_std=2.0,
            )
            if not tiles:
                raise RuntimeError(f"Selected scene {item['id']} produced no radiometrically informative tiles")
            manifest["scenes"].append({
                "region": region, "item_id": item["id"], "datetime": item.get("properties", {}).get("datetime"),
                "cloud_cover": item.get("properties", {}).get("eo:cloud_cover"), "window": derivation["window"],
                "window_selection": selection, "derived_scene": str(scene_path.resolve()),
                "derivation_manifest": str(derivation_path.resolve()), "tiles": tiles,
            })
            print(json.dumps({"region": region, "item": item["id"], "window_selection": selection, "tiles": len(tiles)}))
    manifest["scene_count"] = len(manifest["scenes"])
    manifest["tile_count"] = sum(len(scene["tiles"]) for scene in manifest["scenes"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str((output / 'manifest.json').resolve()), "scenes": manifest["scene_count"], "tiles": manifest["tile_count"]}, indent=2))


if __name__ == "__main__":
    main()
