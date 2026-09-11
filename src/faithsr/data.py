from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import rasterio
import torch
from torch.utils.data import Dataset

from .degradation import DegradationParameters, degrade


def discover_images(root: str | Path, extensions: set[str] | None = None) -> list[Path]:
    extensions = extensions or {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(path for path in Path(root).rglob("*") if path.suffix.lower() in extensions)


COORDINATE_STEM = re.compile(r"^(?P<lat>-?\d+(?:\.\d+)?)_(?P<lon>-?\d+(?:\.\d+)?)$")


def geographic_group(path: Path, grid_degrees: float = 0.1) -> str:
    """Return a leakage-resistant group for coordinate-named tiles or a parent scene/class."""
    match = COORDINATE_STEM.match(path.stem)
    if match:
        if grid_degrees <= 0:
            raise ValueError("grid_degrees must be positive")
        latitude, longitude = float(match.group("lat")), float(match.group("lon"))
        return f"grid:{math.floor(latitude / grid_degrees)}:{math.floor(longitude / grid_degrees)}"
    return f"parent:{path.parent.name or path.stem.split('_')[0]}"


def geographic_split(files: list[Path], train_ratio: float = 0.8, seed: int = 42, grid_degrees: float = 0.1) -> tuple[list[Path], list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        key = geographic_group(path, grid_degrees)
        groups.setdefault(key, []).append(path)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    split = max(1, min(len(keys) - 1, round(len(keys) * train_ratio))) if len(keys) > 1 else 1
    train_keys = set(keys[:split])
    return [p for key in keys if key in train_keys for p in groups[key]], [p for key in keys if key not in train_keys for p in groups[key]]


def read_image(path: Path, bands: int = 3) -> torch.Tensor:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            if any(dtype != "uint8" for dtype in source.dtypes):
                raise ValueError(f"Training TIFF must be derived/documented uint8: {path}")
            if source.count < bands:
                raise ValueError(f"Expected at least {bands} bands: {path}")
            array = source.read(list(range(1, bands + 1)))
    else:
        array = np.asarray(Image.open(path).convert("RGB")).transpose(2, 0, 1)
    return torch.from_numpy(array.copy()).float() / 255


class SyntheticSRDataset(Dataset):
    def __init__(self, files: list[Path], crop_size: int = 192, bands: int = 3, train: bool = True, seed: int = 42, max_severity: float = 1.0, mtf_probability: float = 0.0, complex_probability: float = 1.0, scale: int = 3) -> None:
        if scale not in {3, 4}:
            raise ValueError("Dataset scale must be 3 or 4")
        if crop_size % scale:
            raise ValueError(f"HR crop size must be divisible by scale {scale}")
        self.files, self.crop_size, self.bands, self.train, self.seed, self.scale = files, crop_size, bands, train, seed, scale
        if not 0 <= complex_probability <= 1:
            raise ValueError("Complex degradation probability must be within [0,1]")
        self.max_severity, self.mtf_probability, self.complex_probability = max_severity, mtf_probability, complex_probability

    def set_degradation_severity(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("Degradation severity must be within [0,1]")
        self.max_severity = value

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image = read_image(self.files[index], self.bands)
        _, height, width = image.shape
        if min(height, width) < self.crop_size:
            raise ValueError(f"Image smaller than {self.crop_size}: {self.files[index]}")
        rng = random if self.train else random.Random(self.seed + index)
        top = rng.randint(0, height - self.crop_size) if self.train else (height - self.crop_size) // 2
        left = rng.randint(0, width - self.crop_size) if self.train else (width - self.crop_size) // 2
        hr = image[:, top:top + self.crop_size, left:left + self.crop_size]
        if self.train:
            hr = torch.rot90(hr, rng.randint(0, 3), (1, 2))
            if rng.random() < 0.5:
                hr = hr.flip(2)
        degradation_seed = rng.randrange(0, 2**31) if self.train else self.seed + index
        use_complex = self.complex_probability >= 1 or rng.random() < self.complex_probability
        parameters = None if use_complex else DegradationParameters(0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 100, "bicubic_reference", None, 0.0)
        lr, parameters = degrade(hr, scale=self.scale, seed=degradation_seed, params=parameters, max_severity=self.max_severity, mtf_probability=self.mtf_probability)
        details = json.dumps({"seed": degradation_seed, "scale": self.scale, **parameters.to_dict()}, sort_keys=True)
        return lr, hr, parameters.normalized_target(), str(self.files[index]), details


def write_split_manifest(path: str | Path, train: list[Path], validation: list[Path], seed: int, grid_degrees: float = 0.1) -> None:
    payload = {"seed": seed, "strategy": "coordinate-grid-when-available-else-parent-group", "grid_degrees": grid_degrees, "train": [str(p) for p in train], "validation": [str(p) for p in validation]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_degradation_records(path: str | Path, epoch: int, sources: list[str] | tuple[str, ...], details: list[str] | tuple[str, ...]) -> None:
    """Persist the exact sampled LR formation parameters for every training observation."""
    if len(sources) != len(details):
        raise ValueError("Source and degradation record counts differ")
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        for source, detail in zip(sources, details):
            handle.write(json.dumps({"epoch": epoch, "source": source, "degradation": json.loads(detail)}, sort_keys=True) + "\n")
