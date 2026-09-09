from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import SyntheticSRDataset, discover_images
from .metrics import edge_precision_recall, ergas, false_edge_rate, psnr, spectral_angle_mapper, ssim, uncertainty_calibration_error, uncertainty_error_correlation


def interpolate(lr: torch.Tensor, mode: str) -> torch.Tensor:
    arguments = {"scale_factor": 3, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        arguments["align_corners"] = False
        arguments["antialias"] = True
    return F.interpolate(lr, **arguments)


def _scores(method: str, prediction: torch.Tensor, target: torch.Tensor, path: str, degradation: str, uncertainty=None) -> dict:
    precision, recall = edge_precision_recall(prediction, target)
    row = {
        "method": method, "path": path, "degradation": degradation, "sample_count": prediction.shape[0], "psnr": psnr(prediction, target).item(), "ssim": ssim(prediction, target).item(),
        "false_edge_rate": false_edge_rate(prediction, target).item(), "edge_precision": precision.item(), "edge_recall": recall.item(),
        "sam_radians": spectral_angle_mapper(prediction, target).item(), "ergas": ergas(prediction, target).item(),
        "uncertainty_correlation": None, "uncertainty_calibration_error": None,
    }
    if uncertainty is not None:
        row["uncertainty_correlation"] = uncertainty_error_correlation(uncertainty, prediction, target).item()
        row["uncertainty_calibration_error"] = uncertainty_calibration_error(uncertainty, prediction, target).item()
    return row


def validation_files(data: str, split_manifest: str | None = None) -> list[Path]:
    if not split_manifest:
        return discover_images(data)
    payload = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    root = Path(data)
    files: list[Path] = []
    for recorded in payload["validation"]:
        recorded_path = Path(recorded)
        candidate = recorded_path if recorded_path.exists() else root / recorded_path.name
        if not candidate.exists():
            matches = list(root.rglob(recorded_path.name))
            if len(matches) != 1:
                raise FileNotFoundError(f"Could not uniquely resolve split item: {recorded}")
            candidate = matches[0]
        files.append(candidate)
    return files


@torch.inference_mode()
def compare(data: str, output: str, checkpoint: str | None = None, split_manifest: str | None = None, crop_size: int = 192, bands: int = 3, batch_size: int = 64, workers: int = 2, device: str | None = None) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = None
    if checkpoint:
        model, _ = load_checkpoint(checkpoint, device)
        model.to(device).eval(); bands = model.config.in_channels
    files = validation_files(data, split_manifest)
    if not files:
        raise ValueError("No validation images found")
    loader = DataLoader(SyntheticSRDataset(files, crop_size, bands, False), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    rows = []
    for lr, hr, _, paths, degradation_details in loader:
        lr, hr = lr.to(device), hr.to(device)
        for mode in ("nearest", "bilinear", "bicubic"):
            prediction = interpolate(lr, mode).clamp(0, 1)
            rows.append(_scores(mode, prediction, hr, f"{paths[0]}..{paths[-1]}", "deterministic-per-sample", None))
        if model is not None:
            prediction, aux = model(lr, return_aux=True)
            prediction = prediction.clamp(0, 1)
            uncertainty = aux.get("uncertainty")
            rows.append(_scores("faithsr3x", prediction, hr, f"{paths[0]}..{paths[-1]}", "deterministic-per-sample", uncertainty))
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        total_samples = sum(row["sample_count"] for row in selected)
        summary[method] = {key: sum(row[key] * row["sample_count"] for row in selected if row[key] is not None) / sum(row["sample_count"] for row in selected if row[key] is not None) for key in rows[0] if key not in {"method", "path", "degradation", "sample_count"} and any(row[key] is not None for row in selected)}
        summary[method]["samples"] = total_samples
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True); parser.add_argument("--output", default="artifacts/results/metrics/faithsr3x_comparison.csv")
    parser.add_argument("--checkpoint"); parser.add_argument("--split-manifest"); parser.add_argument("--crop-size", type=int, default=192); parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--workers", type=int, default=2); parser.add_argument("--device")
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
