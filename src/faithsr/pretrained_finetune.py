"""Full fine-tuning for a pinned Swin2SR x4 model on matched x3 or native x4 tasks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .compare import _score_rows, interpolate, paired_cluster_intervals
from .config import load_config, seed_everything
from .data import SyntheticSRDataset, append_degradation_records, discover_images, geographic_split, write_split_manifest
from .metrics import LPIPSMetric
from .qualitative import representative_sample_index, save_qualitative_panel


def exact_x3(output_x4: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Make the scale mismatch explicit; this baseline remains an x4 model followed by resize."""
    return F.interpolate(output_x4, size=target.shape[-2:], mode="bicubic", align_corners=False, antialias=True)


def output_for_target(output_x4: torch.Tensor, target: torch.Tensor, target_scale: int) -> torch.Tensor:
    if target_scale == 3:
        return exact_x3(output_x4, target)
    if target_scale != 4:
        raise ValueError("Swin2SR target scale must be 3 or 4")
    if output_x4.shape != target.shape:
        raise ValueError(f"Native x4 output shape {tuple(output_x4.shape)} does not match target {tuple(target.shape)}")
    return output_x4


def _load(model_id: str, revision: str | None, checkpoint: str | Path | None = None):
    try:
        from transformers import Swin2SRForImageSuperResolution
    except ImportError as error:
        raise RuntimeError("Install faithsr[teacher] to fine-tune Swin2SR") from error
    model = Swin2SRForImageSuperResolution.from_pretrained(model_id, revision=revision)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    return model


@torch.inference_mode()
def _evaluate(model, loader, device, method: str, perceptual: LPIPSMetric | None = None, target_scale: int = 3) -> list[dict]:
    model.eval()
    rows: list[dict] = []
    for lr, hr, _, paths, degradation_details in loader:
        lr, hr = lr.to(device), hr.to(device)
        prediction = output_for_target(model(pixel_values=lr).reconstruction, hr, target_scale).clamp(0, 1)
        rows.extend(_score_rows(method, prediction, hr, paths, degradation_details, lpips_values=None if perceptual is None else perceptual(prediction, hr), scale=target_scale))
    return rows


def _write_model_card(path: Path, model_id: str, revision: str | None, parameters: int, epochs: int, learning_rate: float, target_scale: int = 3) -> None:
    title = "Fully Fine-Tuned Swin2SR Native x4" if target_scale == 4 else "Fully Fine-Tuned Swin2SR x4-to-x3 Baseline"
    mapping = "native x4 output is supervised directly without a resize" if target_scale == 4 else "a declared antialiased bicubic operator maps x4 output to the matched x3 target"
    equation = "y_hat = Swin2SR_x4(x)" if target_scale == 4 else "y_hat = Resize_(3H,3W)(Swin2SR_x4(x))"
    resize_node = "" if target_scale == 4 else '    X4 --> R3["Explicit antialiased resize to x3"]\n    R3 --> LOSS["Smooth-L1 against held-out-protocol HR"]'
    if target_scale == 4:
        resize_node = '    X4 --> LOSS["Smooth-L1 against native-x4 held-out HR"]'
    path.write_text(f"""# Model Card: {title}

## Role and necessity

This is a strong personalized pretrained candidate, not a renamed custom FaithSR architecture. Every parameter of pinned pretrained `{model_id}` at revision `{revision}` is trainable. Target scale is x{target_scale}; {mapping}.

## Mathematical function

For LR input `x`, the evaluated mapping is:

`{equation}`.

All {parameters:,} parameters minimize `SmoothL1(y_hat, y)` for {epochs} epochs with AdamW learning rate {learning_rate}. The resize is not hidden and this baseline is never called native x3.

## Layers and activations

Swin2SR uses shallow convolutional embedding, residual Swin Transformer groups with shifted-window self-attention, GELU MLPs, layer normalization, convolutional residual connections, and an x4 sub-pixel reconstruction head. GELU is inherited from the published pretrained architecture because its smooth gating is standard inside transformer MLPs. PixelShuffle is inherited for x4 reconstruction. Any x4-to-x3 resize belongs only to the earlier matched-x3 comparison protocol, not the minimum-3x product requirement.

```mermaid
flowchart LR
    LR["Synthetic LR"] --> EMB["Shallow convolution"]
    EMB --> RSTB["Residual Swin Transformer groups\nwindow attention + GELU MLP"]
    RSTB --> X4["Pretrained x4 reconstruction"]
{resize_node}
```

## Interpretation rule

Frozen and fully fine-tuned results use the same scene-separated validation files and degradation records. A visually pleasing example cannot replace the saved per-image table, paired scene-cluster intervals, or negative failure cases.

## References

- Conde et al., “Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration,” ECCV Workshops 2022: https://arxiv.org/abs/2209.11345
- Immutable model configuration used here: https://huggingface.co/caidas/swin2SR-classical-sr-x4-64/blob/{revision}/config.json
- Project architecture and experiment rationale: `docs/research/model-and-experiment-methodology.md`
""", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    files = discover_images(args.data, set(config["data"]["extensions"]))
    train_files, val_files = geographic_split(files, config["data"]["train_ratio"], config["seed"], config["data"].get("split_grid_degrees", 0.1))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output / "split.json", train_files, val_files, config["seed"], config["data"].get("split_grid_degrees", 0.1))
    degradation = config.get("degradation", {})
    dataset_args = dict(
        crop_size=config["data"]["crop_size"], bands=3, seed=config["seed"], scale=args.target_scale,
        mtf_probability=float(degradation.get("mtf_probability", 0.0)),
        complex_probability=float(config["data"].get("mixed_probability", 1.0)),
    )
    train_ds = SyntheticSRDataset(train_files, train=True, **dataset_args)
    val_ds = SyntheticSRDataset(val_files, train=False, **dataset_args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)
    perceptual = LPIPSMetric(device, args.lpips_network) if args.lpips else None

    model = _load(args.model_id, args.revision).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    _write_model_card(output / "model-card.md", args.model_id, args.revision, parameters, args.epochs, args.learning_rate, args.target_scale)
    frozen_method = "swin2sr_frozen_native_x4" if args.target_scale == 4 else "swin2sr_frozen_x4_resized_x3"
    tuned_method = "swin2sr_full_finetune_native_x4" if args.target_scale == 4 else "swin2sr_full_finetune_x4_resized_x3"
    frozen_rows = _evaluate(model, val_loader, device, frozen_method, perceptual, args.target_scale)
    preview_index, preview_texture = representative_sample_index(val_ds)
    preview_lr, preview_hr, _, preview_path, preview_degradation = val_ds[preview_index]
    with torch.inference_mode():
        frozen_preview = output_for_target(model(pixel_values=preview_lr.unsqueeze(0).to(device)).reconstruction, preview_hr.unsqueeze(0).to(device), args.target_scale)[0].clamp(0, 1)
    save_qualitative_panel(output / "frozen-preview.png", preview_lr, preview_hr, frozen_preview, extra={
        "source": preview_path, "degradation": json.loads(preview_degradation),
        "selection_rule": "validation sample at median HR spatial standard deviation; fixed before training and independent of model output",
        "selection_index": preview_index, "selection_hr_spatial_std": preview_texture,
    }, method_label=f"Swin2SR frozen x{args.target_scale}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    metrics_path = output / "metrics.csv"
    metrics_path.write_text("epoch,train_smooth_l1,validation_psnr\n", encoding="utf-8")
    degradation_path = output / "degradations.jsonl"
    degradation_path.write_text("", encoding="utf-8")
    best_psnr, started = float("-inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        curriculum_epochs = max(1, int(degradation.get("curriculum_epochs", 1)))
        start_severity = float(degradation.get("start_severity", 1.0))
        progress = min(1.0, (epoch - 1) / max(1, curriculum_epochs - 1))
        train_ds.set_degradation_severity(start_severity + (1 - start_severity) * progress)
        model.train()
        total = 0.0
        for lr, hr, _, sources, details in train_loader:
            append_degradation_records(degradation_path, epoch, sources, details)
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = output_for_target(model(pixel_values=lr).reconstruction, hr, args.target_scale)
                loss = F.smooth_l1_loss(prediction, hr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        mean_loss = total / max(1, len(train_loader))
        validation_rows = _evaluate(model, val_loader, device, "validation", target_scale=args.target_scale)
        validation_psnr = sum(row["psnr"] for row in validation_rows) / len(validation_rows)
        with metrics_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([epoch, mean_loss, validation_psnr])
        if validation_psnr > best_psnr:
            best_psnr = validation_psnr
            torch.save({"model": model.state_dict(), "model_id": args.model_id, "revision": args.revision, "epoch": epoch, "train_loss": mean_loss, "validation_psnr": validation_psnr}, output / "best.pth")
            model.eval()
            with torch.inference_mode():
                prediction = output_for_target(model(pixel_values=preview_lr.unsqueeze(0).to(device)).reconstruction, preview_hr.unsqueeze(0).to(device), args.target_scale)[0].clamp(0, 1)
            save_qualitative_panel(output / "best-preview.png", preview_lr, preview_hr, prediction, extra={
                "epoch": epoch, "source": preview_path, "degradation": json.loads(preview_degradation),
                "selection_rule": "highest mean validation PSNR epoch; visual sample is fixed at median held-out HR spatial standard deviation",
                "selection_index": preview_index, "selection_hr_spatial_std": preview_texture,
            }, method_label=f"Swin2SR FT native x{args.target_scale}" if args.target_scale == 4 else "Swin2SR FT x4-to-x3")
        print(json.dumps({"epoch": epoch, "train_smooth_l1": mean_loss, "validation_psnr": validation_psnr}), flush=True)

    best_state = torch.load(output / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    fine_tuned_rows = _evaluate(model, val_loader, device, tuned_method, perceptual, args.target_scale)
    rows = []
    for lr, hr, _, paths, details in val_loader:
        lr, hr = lr.to(device), hr.to(device)
        prediction = interpolate(lr, "bicubic", args.target_scale).clamp(0, 1)
        rows.extend(_score_rows("bicubic", prediction, hr, paths, details, lpips_values=None if perceptual is None else perceptual(prediction, hr), scale=args.target_scale))
    rows.extend(frozen_rows)
    rows.extend(fine_tuned_rows)
    comparison = output / "comparison.csv"
    with comparison.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    excluded = {"method", "path", "geographic_group", "degradation", "sample_count"}
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {key: sum(row[key] for row in selected if row[key] is not None) / sum(row[key] is not None for row in selected) for key in rows[0] if key not in excluded and any(row[key] is not None for row in selected)}
        summary[method]["samples"] = len(selected)
    payload = {
        "methods": summary, "paired_scene_intervals_vs_bicubic": paired_cluster_intervals(rows),
        "model_id": args.model_id, "revision": args.revision, "all_parameters_trainable": True,
        "parameters": parameters, "duration_seconds": time.time() - started, "best_epoch": best_state["epoch"],
        "target_scale": args.target_scale, "native_x4_output": args.target_scale == 4,
    }
    comparison.with_suffix(".summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True); parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="caidas/swin2SR-classical-sr-x4-64")
    parser.add_argument("--revision", default="c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e")
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2); parser.add_argument("--learning-rate", type=float, default=1e-5); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-scale", type=int, choices=[3, 4], default=3)
    parser.add_argument("--device"); parser.add_argument("--lpips", action="store_true"); parser.add_argument("--lpips-network", choices=["alex", "vgg", "squeeze"], default="alex")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
