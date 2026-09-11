"""Native Sentinel-2 L2A scene-separated FaithSR ablations with saved previews."""
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import numpy as np


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/faithsr3x-native-l2a")
STAC = "https://earth-search.aws.element84.com/v1"
DATE_RANGE = "2025-10-01T00:00:00Z/2026-03-31T23:59:59Z"
REGIONS = {
    "karachi": [66.75, 24.70, 67.25, 25.10],
    "lahore": [74.05, 31.30, 74.60, 31.75],
    "islamabad": [72.80, 33.45, 73.35, 33.90],
}
GOOD_SCL_CLASSES = {2, 4, 5, 6, 7}
MIN_GOOD_SCL_FRACTION = 0.95
MIN_POSITIVE_REFLECTANCE_FRACTION = 0.55


def run(command: list[str], environment: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def source_root() -> Path:
    mount = next((path for path in INPUT_ROOT.rglob("faithsr-3x-private-source") if path.is_dir()), None)
    if mount is None:
        raise RuntimeError("Private source dataset was not mounted")
    if (mount / "src" / "faithsr").exists():
        return mount
    source_zip = next(mount.rglob("src.zip"), None)
    configs_zip = next(mount.rglob("configs.zip"), None)
    if source_zip is None or configs_zip is None:
        raise RuntimeError("Source/config archives were not found")
    root = WORK / "source"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(source_zip, root / "src")
    shutil.unpack_archive(configs_zip, root / "configs")
    return root


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "User-Agent": "FaithSR-3X/0.2"}, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def query_items(bbox: list[float]) -> list[dict]:
    return post_json(STAC + "/search", {
        "collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": DATE_RANGE, "limit": 30,
        "query": {"eo:cloud_cover": {"lt": 5.0}},
    }).get("features", [])


def best_window(item: dict, size: int = 1152, overview: int = 128) -> tuple[int, int, int, int, dict[str, float]]:
    import rasterio

    with rasterio.open(item["assets"]["red"]["href"]) as red, rasterio.open(item["assets"]["scl"]["href"]) as scl:
        red_small = red.read(1, out_shape=(overview, overview))
        scl_small = scl.read(1, out_shape=(overview, overview))
        good = (red_small > 0) & np.isin(scl_small, list(GOOD_SCL_CLASSES))
        band = item["assets"]["red"].get("raster:bands", [{}])[0]
        reflectance = red_small.astype(np.float32) * float(band.get("scale", 0.0001)) + float(band.get("offset", 0.0))
        block_x, block_y = max(1, math.ceil(size / red.width * overview)), max(1, math.ceil(size / red.height * overview))
        candidates = []
        for row in range(overview - block_y + 1):
            for column in range(overview - block_x + 1):
                quality = good[row:row + block_y, column:column + block_x]
                values = reflectance[row:row + block_y, column:column + block_x]
                good_fraction = float(quality.mean())
                positive_fraction = float(((values > 0.01) & quality).mean())
                valid_values = values[quality & (values > 0)]
                dynamic_range = float(np.percentile(valid_values, 90) - np.percentile(valid_values, 10)) if valid_values.size else 0.0
                eligible = good_fraction >= MIN_GOOD_SCL_FRACTION and positive_fraction >= MIN_POSITIVE_REFLECTANCE_FRACTION
                score = dynamic_range if eligible else -1.0 + good_fraction * max(dynamic_range, 0.0)
                candidates.append((score, good_fraction, positive_fraction, -row, -column))
        score, good_fraction, positive_fraction, row, column = max(candidates)
        if score < 0:
            raise RuntimeError("No radiometrically informative cloud-free Sentinel-2 window was found")
        row, column = -row, -column
        source_column = min(red.width - size, round(column / overview * red.width))
        source_row = min(red.height - size, round(row / overview * red.height))
        return int(source_column), int(source_row), size, size, {
            "good_scl_fraction": good_fraction,
            "positive_reflectance_fraction": positive_fraction,
            "red_reflectance_p90_minus_p10": score,
            "minimum_good_scl_fraction": MIN_GOOD_SCL_FRACTION,
            "minimum_positive_reflectance_fraction": MIN_POSITIVE_REFLECTANCE_FRACTION,
        }


def build_dataset() -> Path:
    from faithsr.sentinel import prepare_stac_rgb8, tile_rgb8_scene

    root = WORK / "data"
    scenes, patches, items = root / "scenes", root / "patches", root / "stac-items"
    for directory in (scenes, patches, items):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": "Element84 Earth Search public STAC sentinel-2-l2a",
        "endpoint": STAC, "date_range": DATE_RANGE, "cloud_cover_lt": 5.0, "scene_size": 1152, "tile_size": 192,
        "reflectance_clip": [0.0, 0.3], "split_unit": "STAC scene parent; seed 0 holds out one scene per region", "scenes": [],
        "window_selection": {
            "minimum_good_scl_fraction": MIN_GOOD_SCL_FRACTION,
            "minimum_positive_reflectance_fraction": MIN_POSITIVE_REFLECTANCE_FRACTION,
            "ranking": "maximum red-reflectance p90-p10 among eligible overview blocks",
        },
        "tile_selection": {"minimum_nonzero_fraction": 0.75, "minimum_mean_channel_std_uint8": 2.0},
    }
    seen = set()
    for region, bbox in REGIONS.items():
        candidates = query_items(bbox)
        selected = []
        rejected = []
        for item in candidates:
            if item["id"] in seen or not all(key in item.get("assets", {}) for key in ("red", "green", "blue", "scl")):
                continue
            try:
                selected.append((item, best_window(item)))
            except RuntimeError as error:
                rejected.append({"item_id": item["id"], "reason": str(error)})
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise RuntimeError(f"Expected two usable scenes for {region}; found {len(selected)}")
        for item, window in selected:
            seen.add(item["id"])
            (items / f'{item["id"]}.json').write_text(json.dumps(item, indent=2), encoding="utf-8")
            scene = scenes / f'{item["id"]}.tif'
            derivation = prepare_stac_rgb8(item, str(scene), window[:4], 0.0, 0.3)
            derivation.update({"region": region, "window_selection": window[4]})
            scene.with_suffix(".tif.derivation.json").write_text(json.dumps(derivation, indent=2), encoding="utf-8")
            tiles = tile_rgb8_scene(
                scene,
                patches / item["id"],
                192,
                min_nonzero_fraction=0.75,
                min_channel_std=2.0,
            )
            if not tiles:
                raise RuntimeError(f"Selected scene {item['id']} produced no radiometrically informative tiles")
            manifest["scenes"].append({
                "region": region, "item_id": item["id"], "datetime": item["properties"].get("datetime"),
                "cloud_cover": item["properties"].get("eo:cloud_cover"), "window": derivation["window"],
                "window_selection": window[4], "tiles": tiles,
            })
            print(json.dumps({"region": region, "item": item["id"], "tiles": len(tiles)}), flush=True)
        manifest.setdefault("regions", []).append({"name": region, "bbox": bbox, "selected": [item["id"] for item, _ in selected], "rejected": rejected})
    manifest["scene_count"] = len(manifest["scenes"])
    manifest["tile_count"] = sum(len(scene["tiles"]) for scene in manifest["scenes"])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return patches


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"})
    os.environ.update(environment)
    run([sys.executable, "-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet"], environment)
    run([sys.executable, "-m", "pip", "install", "lpips==0.1.4", "scikit-image==0.25.2", "PyYAML==6.0.3", "--quiet"], environment)
    source = source_root()
    environment["PYTHONPATH"] = str(source / "src")
    os.environ["PYTHONPATH"] = environment["PYTHONPATH"]
    sys.path.insert(0, environment["PYTHONPATH"])
    (WORK / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], env=environment, text=True), encoding="utf-8")
    import yaml

    data = build_dataset()
    base = yaml.safe_load((source / "configs" / "sentinel_l2a_native_kaggle.yaml").read_text(encoding="utf-8"))
    variants = {
        "plain": {"model": {"use_degradation_adaptation": False, "use_observation_consistency": False, "use_uncertainty": False}},
        "uncertainty_only": {"model": {"use_degradation_adaptation": False, "use_observation_consistency": False, "use_uncertainty": True}},
        "full": {"model": {"use_degradation_adaptation": True, "use_observation_consistency": True, "use_uncertainty": True}},
    }
    lineage = {
        "plain": ("Establish a native-L2A custom-backbone control under scene-separated evaluation.", "Delhi RGB ablations used display tiles without native L2A radiometric provenance."),
        "uncertainty_only": ("Isolate the uncertainty head without degradation adaptation or observation consistency.", "Plain native-L2A control; only spatial uncertainty estimation and calibration are added."),
        "full": ("Evaluate the complete personalized model on native L2A-derived 8-bit imagery.", "Uncertainty-only control plus degradation adaptation and learned observation consistency."),
    }
    report = {"dataset_manifest": str((WORK / "data" / "manifest.json").resolve()), "variants": {}}
    for name, overrides in variants.items():
        config = deepcopy(base)
        for section, values in overrides.items():
            config[section].update(values)
        config_path, output = WORK / f"{name}.yaml", WORK / name
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run([sys.executable, "-m", "faithsr.train", "--data", str(data), "--config", str(config_path), "--output", str(output), "--experiment-rationale", lineage[name][0], "--previous-experiment", lineage[name][1]], environment)
        comparison = output / "comparison.csv"
        run([sys.executable, "-m", "faithsr.compare", "--data", str(data), "--checkpoint", str(output / "best.pth"), "--split-manifest", str(output / "split.json"), "--config", str(config_path), "--batch-size", "8", "--workers", "2", "--lpips", "--output", str(comparison)], environment)
        rows = list(csv.DictReader((output / "metrics.csv").open(encoding="utf-8")))
        report["variants"][name] = {
            "best_training_row": max(rows, key=lambda row: float(row["psnr"])),
            "evaluation": json.loads(comparison.with_suffix(".summary.json").read_text(encoding="utf-8")),
            "qualitative_preview": str((output / "best-preview.png").resolve()),
            "qualitative_metrics": json.loads((output / "best-preview.json").read_text(encoding="utf-8")),
        }
    (WORK / "experiment-summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
