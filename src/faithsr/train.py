from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import load_config, seed_everything, write_run_manifest
from .data import SyntheticSRDataset, append_degradation_records, discover_images, geographic_split, write_split_manifest
from .documentation import append_run_results, write_model_card
from .losses import FaithfulObjective
from .metrics import false_edge_rate, psnr, ssim, uncertainty_error_correlation
from .model import FaithSR3X, FaithSRConfig, count_parameters
from .qualitative import representative_sample_index, save_qualitative_panel
from .teacher import Swin2SRTeacher


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    rows = []
    for lr, hr, _, _, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr, aux = model(lr, return_aux=True)
        sr = sr.clamp(0, 1)
        row = {"psnr": psnr(sr, hr).item(), "ssim": ssim(sr, hr).item(), "false_edge_rate": false_edge_rate(sr, hr).item()}
        if aux["reprojected_lr"] is not None:
            row["observation_l1"] = torch.nn.functional.l1_loss(aux["reprojected_lr"], lr).item()
        if aux["uncertainty"] is not None:
            row["uncertainty_correlation"] = uncertainty_error_correlation(aux["uncertainty"], sr, hr).item()
        rows.append(row)
    return {key: sum(row[key] for row in rows if key in row) / sum(key in row for row in rows) for key in rows[0]} if rows else {}


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    files = discover_images(args.data, set(config["data"]["extensions"]))
    if len(files) < 2:
        raise RuntimeError("At least two images are required")
    grid_degrees = config["data"].get("split_grid_degrees", 0.1)
    train_files, val_files = geographic_split(files, config["data"]["train_ratio"], config["seed"], grid_degrees)
    if not train_files or not val_files:
        raise ValueError("Geographic split produced an empty train or validation set; supply multiple parent/geographic groups")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output / "split.json", train_files, val_files, config["seed"], grid_degrees)
    model_cfg = FaithSRConfig(in_channels=args.bands, out_channels=args.bands, **config["model"])
    model = FaithSR3X(model_cfg).to(device)
    write_model_card(
        output / "model-card.md", model, config, output.name,
        args.experiment_rationale or "Evaluate the configured FaithSR-3X hypothesis under a frozen reproducible protocol.",
        args.previous_experiment or "See docs/research/model-and-experiment-methodology.md for the experiment lineage.",
    )
    teacher = None
    if args.teacher_model:
        teacher = Swin2SRTeacher(
            args.teacher_model,
            args.teacher_checkpoint,
            args.teacher_revision,
            config["training"].get("teacher_batch_size"),
        ).to(device)
    teacher_max_epoch = int(config["training"].get("teacher_max_epoch", config["training"]["epochs"]))
    degradation_config = config.get("degradation", {})
    mtf_probability = float(degradation_config.get("mtf_probability", 0.35))
    complex_probability = float(config["data"].get("mixed_probability", 1.0))
    train_ds = SyntheticSRDataset(train_files, config["data"]["crop_size"], args.bands, True, config["seed"], mtf_probability=mtf_probability, complex_probability=complex_probability)
    val_ds = SyntheticSRDataset(val_files, config["data"]["crop_size"], args.bands, False, config["seed"], mtf_probability=mtf_probability, complex_probability=complex_probability)
    preview_index, preview_texture = representative_sample_index(val_ds)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["workers"], drop_last=False)
    validation_batch_size = config["training"].get("validation_batch_size", config["training"]["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=validation_batch_size, shuffle=False, num_workers=config["training"]["workers"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    objective = FaithfulObjective(config["loss"])
    amp_enabled = device.type == "cuda" and config["training"]["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best = 1, float("-inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state.get("best_psnr", best)
    manifest = {"config": config, "git_commit": _git_commit(), "device": str(device), "parameters": count_parameters(model), "data_root": str(Path(args.data).resolve()), "started_unix": time.time()}
    write_run_manifest(output / "run.json", manifest)
    metrics_path = output / "metrics.csv"
    degradation_manifest = output / "degradations.jsonl"
    if start_epoch == 1:
        degradation_manifest.write_text("", encoding="utf-8")
    if not metrics_path.exists():
        metrics_path.write_text("epoch,train_loss,psnr,ssim,false_edge_rate,observation_l1,uncertainty_correlation,learning_rate\n", encoding="utf-8")
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        curriculum_epochs = max(1, int(degradation_config.get("curriculum_epochs", 1)))
        start_severity = float(degradation_config.get("start_severity", 1.0))
        progress = min(1.0, (epoch - 1) / max(1, curriculum_epochs - 1))
        severity = start_severity + (1.0 - start_severity) * progress
        train_ds.set_degradation_severity(severity)
        model.train()
        total = 0.0
        for lr, hr, degradation_target, source_paths, degradation_details in train_loader:
            append_degradation_records(degradation_manifest, epoch, source_paths, degradation_details)
            lr, hr, degradation_target = lr.to(device), hr.to(device), degradation_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                sr, aux = model(lr, return_aux=True)
                teacher_sr = None if teacher is None or epoch > teacher_max_epoch else teacher(lr, hr.shape[-2:])
                loss, _ = objective(sr, hr, lr, aux, degradation_target, teacher_sr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()
        validation = evaluate(model, val_loader, device)
        state = {"epoch": epoch, "model": model.state_dict(), "model_config": model.config.__dict__, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_psnr": max(best, validation["psnr"]), "config": config}
        save_checkpoint(output / "last.pth", model, **{k: v for k, v in state.items() if k not in {"model", "model_config"}})
        if validation["psnr"] > best:
            best = validation["psnr"]
            save_checkpoint(output / "best.pth", model, **{k: v for k, v in state.items() if k not in {"model", "model_config"}})
            preview_lr, preview_hr, _, preview_path, preview_degradation = val_ds[preview_index]
            with torch.inference_mode():
                preview_sr, preview_aux = model(preview_lr.unsqueeze(0).to(device), return_aux=True)
            save_qualitative_panel(
                output / "best-preview.png",
                preview_lr,
                preview_hr,
                preview_sr[0].clamp(0, 1),
                None if preview_aux["uncertainty"] is None else preview_aux["uncertainty"][0],
                {
                    "epoch": epoch,
                    "source": preview_path,
                    "degradation": json.loads(preview_degradation),
                    "selection_rule": "validation sample at median HR spatial standard deviation; fixed before training and independent of model output",
                    "selection_index": preview_index,
                    "selection_hr_spatial_std": preview_texture,
                },
            )
        with metrics_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([epoch, total / max(1, len(train_loader)), validation.get("psnr"), validation.get("ssim"), validation.get("false_edge_rate"), validation.get("observation_l1"), validation.get("uncertainty_correlation"), optimizer.param_groups[0]["lr"]])
        print(json.dumps({"epoch": epoch, "degradation_severity": severity, "train_loss": total / max(1, len(train_loader)), **validation}))
    finished = time.time()
    manifest.update({"finished_unix": finished, "duration_seconds": finished - manifest["started_unix"], "best_validation_psnr": best, "degradation_manifest": str(degradation_manifest.resolve()), "degradation_manifest_sha256": _sha256(degradation_manifest)})
    write_run_manifest(output / "run.json", manifest)
    append_run_results(output / "model-card.md", metrics_path, manifest["duration_seconds"])
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FaithSR-3X")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/faithsr3x_base.yaml")
    parser.add_argument("--output", default="runs/faithsr3x")
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--teacher-model", help="Hugging Face Swin2SR model ID used as a frozen teacher")
    parser.add_argument("--teacher-checkpoint", help="Optional compatible local Swin2SR state dict")
    parser.add_argument("--teacher-revision", help="Immutable Hugging Face model revision")
    parser.add_argument("--experiment-rationale", help="Why this run is scientifically necessary; written into model-card.md")
    parser.add_argument("--previous-experiment", help="Prior run/hypothesis this experiment extends; written into model-card.md")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
