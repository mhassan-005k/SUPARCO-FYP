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
from .config import load_config
from .data import SyntheticSRDataset, discover_images, geographic_group
from .metrics import LPIPSMetric, reconstruction_metrics_per_sample, uncertainty_metrics_per_sample
from .teacher import Swin2SRTeacher


def interpolate(lr: torch.Tensor, mode: str, scale: int = 3) -> torch.Tensor:
    arguments = {"scale_factor": scale, "mode": mode}
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
    lpips_values: torch.Tensor | None = None,
    scale: int = 3,
) -> list[dict]:
    """Return one independent metric record per source image."""

    values = reconstruction_metrics_per_sample(prediction, target, scale)
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
            "lpips": None if lpips_values is None else float(lpips_values[index].detach().cpu()),
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

    metrics = ["psnr", "ssim", "false_edge_rate", "sam_radians", "ergas"]
    for optional_metric in ("edge_precision", "edge_recall", "observation_l1"):
        if rows and all(optional_metric in row and row[optional_metric] is not None for row in rows):
            metrics.append(optional_metric)
    if rows and all("lpips" in row and row["lpips"] is not None for row in rows):
        metrics.append("lpips")
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


def degradation_protocol(config_path: str | None, crop_size: int) -> tuple[int, dict[str, float | int], dict]:
    """Resolve and record the exact validation degradation protocol."""

    dataset_options = {"seed": 42, "mtf_probability": 0.0, "complex_probability": 1.0}
    protocol = {"source": "legacy compare defaults", "crop_size": crop_size, **dataset_options}
    if config_path:
        evaluation_config = load_config(config_path)
        crop_size = int(evaluation_config["data"]["crop_size"])
        dataset_options = {
            "seed": int(evaluation_config["seed"]),
            "mtf_probability": float(evaluation_config.get("degradation", {}).get("mtf_probability", 0.0)),
            "complex_probability": float(evaluation_config["data"].get("mixed_probability", 1.0)),
        }
        protocol = {"source": str(Path(config_path).resolve()), "crop_size": crop_size, **dataset_options}
    return crop_size, dataset_options, protocol


@torch.inference_mode()
def compare(data: str, output: str, checkpoint: str | None = None, split_manifest: str | None = None, teacher_model: str | None = None, teacher_revision: str | None = None, crop_size: int = 192, bands: int = 3, batch_size: int = 64, workers: int = 2, device: str | None = None, include_lpips: bool = False, lpips_network: str = "alex", config_path: str | None = None) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = None
    if checkpoint:
        model, _ = load_checkpoint(checkpoint, device)
        model.to(device).eval()
        bands = model.config.in_channels
    teacher = Swin2SRTeacher(teacher_model, revision=teacher_revision).to(device) if teacher_model else None
    perceptual = LPIPSMetric(device, lpips_network) if include_lpips else None
    if perceptual is not None and bands != 3:
        raise ValueError("LPIPS comparison is defined only for three-band RGB experiments")
    files = validation_files(data, split_manifest)
    if not files:
        raise ValueError("No validation images found")
    crop_size, dataset_options, protocol = degradation_protocol(config_path, crop_size)
    loader = DataLoader(SyntheticSRDataset(files, crop_size, bands, False, **dataset_options), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    rows: list[dict] = []
    for lr, hr, _, paths, degradation_details in loader:
        lr, hr = lr.to(device), hr.to(device)
        for mode in ("nearest", "bilinear", "bicubic"):
            prediction = interpolate(lr, mode).clamp(0, 1)
            rows.extend(_score_rows(mode, prediction, hr, paths, degradation_details, lpips_values=None if perceptual is None else perceptual(prediction, hr)))
        if model is not None:
            prediction, aux = model(lr, return_aux=True)
            prediction = prediction.clamp(0, 1)
            rows.extend(_score_rows("faithsr3x", prediction, hr, paths, degradation_details, aux.get("uncertainty"), None if perceptual is None else perceptual(prediction, hr)))
        if teacher is not None:
            prediction = teacher(lr, hr.shape[-2:]).clamp(0, 1)
            rows.extend(_score_rows("swin2sr_x4_resized_x3", prediction, hr, paths, degradation_details, lpips_values=None if perceptual is None else perceptual(prediction, hr)))
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
        "degradation_protocol": protocol,
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
    parser.add_argument("--lpips", action="store_true", dest="include_lpips")
    parser.add_argument("--lpips-network", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--config", dest="config_path", help="Exact training YAML whose validation degradation protocol must be replayed")
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
