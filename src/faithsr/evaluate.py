from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import SyntheticSRDataset, discover_images
from .metrics import edge_precision_recall, false_edge_rate, psnr, spectral_angle_mapper, ssim, uncertainty_error_correlation


@torch.inference_mode()
def run(checkpoint: str, data: str, output: str, device: str | None = None, crop_size: int = 192) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_checkpoint(checkpoint, device)
    model.to(device).eval()
    dataset = SyntheticSRDataset(discover_images(data), crop_size, model.config.in_channels, False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rows = []
    for lr, hr, _, paths in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr, aux = model(lr, return_aux=True)
        sr = sr.clamp(0, 1)
        precision, recall = edge_precision_recall(sr, hr)
        row = {
            "path": paths[0], "psnr": psnr(sr, hr).item(), "ssim": ssim(sr, hr).item(),
            "false_edge_rate": false_edge_rate(sr, hr).item(), "edge_precision": precision.item(),
            "edge_recall": recall.item(), "sam_radians": spectral_angle_mapper(sr, hr).item(),
            "observation_l1": torch.nn.functional.l1_loss(aux["reprojected_lr"], lr).item() if aux["reprojected_lr"] is not None else None,
            "uncertainty_correlation": uncertainty_error_correlation(aux["uncertainty"], sr, hr).item() if aux["uncertainty"] is not None else None,
        }
        rows.append(row)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    numeric = [key for key in rows[0] if key != "path"]
    summary = {key: sum(row[key] for row in rows if row[key] is not None) / sum(row[key] is not None for row in rows) for key in numeric}
    summary.update({"samples": len(rows), "checkpoint": checkpoint, "epoch": state.get("epoch")})
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts/results/metrics/faithsr3x_evaluation.csv")
    parser.add_argument("--device"); parser.add_argument("--crop-size", type=int, default=192)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__": main()
