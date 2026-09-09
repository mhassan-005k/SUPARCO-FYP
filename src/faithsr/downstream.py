from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
import re

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .checkpoint import load_checkpoint
from .data import geographic_group, read_image
from .degradation import degrade


COORDINATES = re.compile(r"^(?P<lat>-?\d+(?:\.\d+)?)_(?P<lon>-?\d+(?:\.\d+)?)$")
WORLDCOVER_NAMES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse", 70: "snow_ice", 80: "water",
    90: "herbaceous_wetland", 95: "mangroves", 100: "moss_lichen",
}


def resolve_split(data: str | Path, manifest: str | Path) -> tuple[list[Path], list[Path]]:
    root = Path(data)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    indexed = {path.name: path for path in root.glob("*.png")}
    def resolve(items: list[str]) -> list[Path]:
        missing = [Path(item).name for item in items if Path(item).name not in indexed]
        if missing:
            raise FileNotFoundError(f"Could not resolve {len(missing)} split paths; first={missing[0]}")
        return [indexed[Path(item).name] for item in items]
    return resolve(payload["train"]), resolve(payload["validation"])


def majority_worldcover_labels(paths: list[Path], worldcover: str | Path, window_size: int = 126) -> dict[Path, int]:
    """Assign the majority non-nodata class in a coordinate-centred map window."""

    labels: dict[Path, int] = {}
    half = window_size // 2
    with rasterio.open(worldcover) as source:
        for path in paths:
            match = COORDINATES.match(path.stem)
            if not match:
                raise ValueError(f"Coordinate filename required: {path.name}")
            latitude, longitude = float(match.group("lat")), float(match.group("lon"))
            row, column = source.index(longitude, latitude)
            values = source.read(1, window=Window(column - half, row - half, window_size, window_size), boundless=True, fill_value=0)
            valid = values[values != 0]
            if valid.size == 0:
                raise ValueError(f"No WorldCover label support for {path.name}")
            labels[path] = int(np.bincount(valid).argmax())
    return labels


class LandCoverDataset(Dataset):
    def __init__(self, paths: list[Path], labels: dict[Path, int], class_to_index: dict[int, int], crop_size: int = 126, augment: bool = False) -> None:
        self.samples = [(path, class_to_index[labels[path]]) for path in paths if labels[path] in class_to_index]
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = read_image(path, 3)
        top = (image.shape[-2] - self.crop_size) // 2
        left = (image.shape[-1] - self.crop_size) // 2
        image = image[:, top:top + self.crop_size, left:left + self.crop_size]
        if self.augment:
            image = torch.rot90(image, index % 4, (1, 2))
            if index % 2:
                image = image.flip(2)
        return image, label, str(path)


class LandCoverClassifier(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image).flatten(1))


def classification_metrics(targets: list[int], predictions: list[int], classes: int) -> dict[str, float]:
    target = torch.tensor(targets)
    prediction = torch.tensor(predictions)
    accuracy = (target == prediction).float().mean().item()
    f1_values = []
    for label in range(classes):
        tp = ((target == label) & (prediction == label)).sum().float()
        fp = ((target != label) & (prediction == label)).sum().float()
        fn = ((target == label) & (prediction != label)).sum().float()
        f1_values.append((2 * tp / (2 * tp + fp + fn + 1e-12)).item())
    return {"accuracy": accuracy, "macro_f1": statistics.fmean(f1_values)}


def paired_accuracy_interval(records: list[dict], candidate: str, reference: str = "bicubic") -> dict:
    indexed = {(row["method"], row["path"]): row for row in records}
    clusters: dict[str, list[float]] = {}
    for row in records:
        key = (reference, row["path"])
        if row["method"] == candidate and key in indexed:
            delta = float(row["correct"]) - float(indexed[key]["correct"])
            clusters.setdefault(row["geographic_group"], []).append(delta)
    means = [statistics.fmean(values) for values in clusters.values()]
    mean = statistics.fmean(means)
    margin = 1.96 * statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else None
    return {"mean_delta": mean, "ci95_low": mean - margin if margin is not None else None, "ci95_high": mean + margin if margin is not None else None, "clusters": len(means)}


def _predict(classifier: nn.Module, images: torch.Tensor) -> list[int]:
    return classifier(images).argmax(1).detach().cpu().tolist()


def write_classifier_card(path: Path, classes: list[int], parameters: int, split: dict) -> None:
    names = ", ".join(f"{code}={WORLDCOVER_NAMES.get(code, 'unknown')}" for code in classes)
    path.write_text(f"""# Auxiliary Land-Cover Classifier Model Card

## Why this experiment exists

This model is a frozen downstream probe, not the proposed super-resolution model. It tests whether detail produced by FaithSR preserves information useful for land-cover classification relative to bicubic interpolation. It follows the reconstruction/false-edge ablations and adds task-level evidence.

Labels are the majority non-zero ESA WorldCover class inside the 126x126 map window centred at the latitude/longitude encoded in each 128x128 RGB filename. Training/calibration/final-test geographic groups are disjoint. Classes: {names}.

## Mathematical function

For blocks i=1..3, `h_i = GELU(Conv3x3,stride2(h_(i-1)))`. Global average pooling gives `v_c = mean_(p,q) h_3[c,p,q]`, and logits are `z = Wv+b`; the predicted class is `argmax(z)`. Cross-entropy trains the probe. GELU is used for smooth gradients and global average pooling avoids a location-specific dense head.

## Layers

| Stage | Shape | Why |
|---|---|---|
| Conv + GELU 1 | 3 to 24, 3x3, stride 2 | Low-level colour/edge evidence with controlled downsampling. |
| Conv + GELU 2 | 24 to 48, 3x3, stride 2 | Mid-level texture evidence. |
| Conv + GELU 3 | 48 to 96, 3x3, stride 2 | Broader semantic context. |
| Global average pool | 96 channels to 96-vector | Spatially robust patch descriptor. |
| Linear | 96 to {len(classes)} logits | Auditable class decision. |

Trainable parameters: {parameters:,}.

```mermaid
flowchart LR
    I["HR training tile"] --> C1["Conv 3→24 + GELU"]
    C1 --> C2["Conv 24→48 + GELU"]
    C2 --> C3["Conv 48→96 + GELU"]
    C3 --> G["Global average pool"]
    G --> L["Linear logits"]
    T["Held-out LR"] --> B["Bicubic x3"]
    T --> S["Frozen FaithSR x3"]
    B --> F["Frozen classifier"]
    S --> F
```

Split: `{json.dumps(split, sort_keys=True)}`.
""", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_paths, test_paths = resolve_split(args.data, args.split_manifest)
    train_groups = sorted({geographic_group(path) for path in train_paths})
    calibration_groups = set(train_groups[-max(1, round(len(train_groups) * 0.15)):])
    fit_paths = [path for path in train_paths if geographic_group(path) not in calibration_groups]
    calibration_paths = [path for path in train_paths if geographic_group(path) in calibration_groups]
    all_paths = fit_paths + calibration_paths + test_paths
    labels = majority_worldcover_labels(all_paths, args.worldcover, args.crop_size)
    classes = sorted(set(labels[path] for path in fit_paths))
    class_to_index = {code: index for index, code in enumerate(classes)}
    fit_dataset = LandCoverDataset(fit_paths, labels, class_to_index, args.crop_size, True)
    calibration_dataset = LandCoverDataset(calibration_paths, labels, class_to_index, args.crop_size, False)
    test_dataset = LandCoverDataset(test_paths, labels, class_to_index, args.crop_size, False)
    counts = Counter(label for _, label in fit_dataset.samples)
    weights = torch.tensor([len(fit_dataset) / (len(classes) * counts[index]) for index in range(len(classes))], device=device)
    classifier = LandCoverClassifier(len(classes)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    fit_loader = DataLoader(fit_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    best_f1 = -1.0
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        classifier.train(); total_loss = 0.0
        for images, targets, _ in fit_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(classifier(images), targets)
            loss.backward(); optimizer.step(); total_loss += loss.item()
        classifier.eval(); targets_all: list[int] = []; predictions_all: list[int] = []
        with torch.inference_mode():
            for images, targets, _ in calibration_loader:
                targets_all.extend(targets.tolist()); predictions_all.extend(_predict(classifier, images.to(device)))
        metrics = classification_metrics(targets_all, predictions_all, len(classes))
        history.append({"epoch": epoch, "loss": total_loss / max(1, len(fit_loader)), **metrics})
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save({"model": classifier.state_dict(), "classes": classes, "epoch": epoch, "calibration": metrics}, output / "classifier_best.pth")
        print(json.dumps(history[-1]))
    checkpoint = torch.load(output / "classifier_best.pth", map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint["model"]); classifier.eval()
    sr_model, _ = load_checkpoint(args.checkpoint, device); sr_model.to(device).eval()
    records: list[dict] = []
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.inference_mode():
        sample_index = 0
        for hr, targets, paths in test_loader:
            hr = hr.to(device)
            lr = torch.stack([degrade(image, seed=42 + sample_index + offset)[0] for offset, image in enumerate(hr)]).to(device)
            sample_index += hr.shape[0]
            candidates = {
                "hr_reference": hr,
                "bicubic": F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1),
                "faithsr3x": sr_model(lr).clamp(0, 1),
            }
            for method, images in candidates.items():
                predictions = _predict(classifier, images)
                for path, target, prediction in zip(paths, targets.tolist(), predictions):
                    records.append({"method": method, "path": path, "geographic_group": geographic_group(Path(path)), "target": target, "prediction": prediction, "correct": target == prediction})
    methods = {}
    for method in candidates:
        selected = [row for row in records if row["method"] == method]
        methods[method] = classification_metrics([row["target"] for row in selected], [row["prediction"] for row in selected], len(classes))
        methods[method]["samples"] = len(selected)
    report = {
        "protocol": "HR-trained frozen classifier; untouched main validation groups are final test",
        "classes": [{"code": code, "name": WORLDCOVER_NAMES.get(code, "unknown")} for code in classes],
        "excluded_unknown_test_samples": len(test_paths) - len(test_dataset),
        "split": {"fit_groups": len(set(geographic_group(path) for path in fit_paths)), "calibration_groups": len(calibration_groups), "test_groups": len(set(geographic_group(path) for path in test_paths))},
        "classifier_best_epoch": checkpoint["epoch"], "classifier_calibration": checkpoint["calibration"],
        "methods": methods,
        "paired_accuracy_vs_bicubic": {name: paired_accuracy_interval(records, name) for name in ("hr_reference", "faithsr3x")},
    }
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "downstream_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_classifier_card(output / "model-card.md", classes, sum(p.numel() for p in classifier.parameters()), report["split"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True); parser.add_argument("--worldcover", required=True)
    parser.add_argument("--split-manifest", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/results/downstream_landcover")
    parser.add_argument("--crop-size", type=int, default=126); parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--device")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
