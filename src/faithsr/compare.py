from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import SyntheticSRDataset, discover_images, geographic_group
from .metrics import reconstruction_metrics_per_sample, uncertainty_metrics_per_sample
from .teacher import Swin2SRTeacher


def interpolate(lr: torch.Tensor, mode: str) -> torch.Tensor:
    arguments = {"scale_factor": 3, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        arguments["align_corners"] = False
        arguments["antialias"] = True
    return F.interpolate(lr, **arguments)


def _score_rows(
    method: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    paths: list[str] | tuple[str, ...],
    degradations: list[str] | tuple[str, ...],
    uncertainty: torch.Tensor | None = None,
) -> list[dict]:
    """Return one independent metric record per source image."""

    values = reconstruction_metrics_per_sample(prediction, target)
    if uncertainty is not None:
        values.update(uncertainty_metrics_per_sample(uncertainty, prediction, target))
    cpu_values = {name: tensor.detach().cpu().tolist() for name, tensor in values.items()}
    rows: list[dict] = []
    for index, path in enumerate(paths):
        row = {
            "method": method,
            "path": path,
            "geographic_group": geographic_group(Path(path)),
            "degradation": degradations[index],
            "sample_count": 1,
            "psnr": cpu_values["psnr"][index],
            "ssim": cpu_values["ssim"][index],
            "false_edge_rate": cpu_values["false_edge_rate"][index],
            "edge_precision": cpu_values["edge_precision"][index],
            "edge_recall": cpu_values["edge_recall"][index],
            "sam_radians": cpu_values["sam_radians"][index],
            "ergas": cpu_values["ergas"][index],
            "uncertainty_correlation": None,
            "uncertainty_calibration_error": None,
        }
        if uncertainty is not None:
            row["uncertainty_correlation"] = cpu_values["uncertainty_correlation"][index]
            row["uncertainty_calibration_error"] = cpu_values["uncertainty_calibration_error"][index]
        rows.append(row)
    return rows


def paired_cluster_intervals(rows: list[dict], reference: str = "bicubic") -> dict:
    """Compute paired 95% normal CIs over geographic-cluster mean deltas."""

    metrics = ("psnr", "ssim", "false_edge_rate", "sam_radians", "ergas")
    indexed = {(row["method"], row["path"]): row for row in rows}
    output: dict[str, dict] = {}
    methods = sorted({row["method"] for row in rows if row["method"] != reference})
    for method in methods:
        paired = [
            (row, indexed[(reference, row["path"])])
            for row in rows
            if row["method"] == method and (reference, row["path"]) in indexed
        ]
        result: dict[str, dict] = {}
        for metric in metrics:
            clusters: dict[str, list[float]] = {}
            for candidate, baseline in paired:
                clusters.setdefault(candidate["geographic_group"], []).append(candidate[metric] - baseline[metric])
            deltas = [statistics.fmean(values) for values in clusters.values()]
            mean = statistics.fmean(deltas) if deltas else None
            margin = 1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else None
            result[metric] = {
                "mean_delta": mean,
                "ci95_low": mean - margin if margin is not None else None,
                "ci95_high": mean + margin if margin is not None else None,
                "clusters": len(deltas),
                "paired_images": len(paired),
            }
        output[f"{method}_vs_{reference}"] = result
    return output


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
def compare(data: str, output: str, checkpoint: str | None = None, split_manifest: str | None = None, teacher_model: str | None = None, teacher_revision: str | None = None, crop_size: int = 192, bands: int = 3, batch_size: int = 64, workers: int = 2, device: str | None = None) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = None
    if checkpoint:
        model, _ = load_checkpoint(checkpoint, device)
        model.to(device).eval()
        bands = model.config.in_channels
    teacher = Swin2SRTeacher(teacher_model, revision=teacher_revision).to(device) if teacher_model else None
    files = validation_files(data, split_manifest)
    if not files:
        raise ValueError("No validation images found")
    loader = DataLoader(SyntheticSRDataset(files, crop_size, bands, False), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    rows: list[dict] = []
    for lr, hr, _, paths, degradation_details in loader:
        lr, hr = lr.to(device), hr.to(device)
        for mode in ("nearest", "bilinear", "bicubic"):
            prediction = interpolate(lr, mode).clamp(0, 1)
            rows.extend(_score_rows(mode, prediction, hr, paths, degradation_details))
        if model is not None:
            prediction, aux = model(lr, return_aux=True)
            prediction = prediction.clamp(0, 1)
            rows.extend(_score_rows("faithsr3x", prediction, hr, paths, degradation_details, aux.get("uncertainty")))
        if teacher is not None:
            prediction = teacher(lr, hr.shape[-2:]).clamp(0, 1)
            rows.extend(_score_rows("swin2sr_x4_resized_x3", prediction, hr, paths, degradation_details))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, dict] = {}
    excluded = {"method", "path", "geographic_group", "degradation", "sample_count"}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: statistics.fmean(row[key] for row in selected if row[key] is not None)
            for key in rows[0]
            if key not in excluded and any(row[key] is not None for row in selected)
        }
        summary[method]["samples"] = len(selected)
    payload = {
        "methods": summary,
        "paired_cluster_robust_vs_bicubic": paired_cluster_intervals(rows),
        "aggregation": "Per-image means; paired normal-approximation 95% CI over geographic-group mean deltas.",
    }
    target.with_suffix(".summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts/results/metrics/faithsr3x_comparison.csv")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split-manifest")
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-revision")
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device")
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
