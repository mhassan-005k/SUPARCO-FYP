from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import load_config, seed_everything, write_run_manifest
from .data import SyntheticSRDataset, discover_images, geographic_split, write_split_manifest
from .losses import FaithfulObjective
from .metrics import false_edge_rate, psnr, ssim, uncertainty_error_correlation
from .model import FaithSR3X, FaithSRConfig, count_parameters


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@torch.inference_mode()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    rows = []
    for lr, hr, _, _ in loader:
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
    train_files, val_files = geographic_split(files, config["data"]["train_ratio"], config["seed"])
    if not val_files:
        val_files = train_files[-1:]
        train_files = train_files[:-1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_split_manifest(output / "split.json", train_files, val_files, config["seed"])
    model_cfg = FaithSRConfig(in_channels=args.bands, out_channels=args.bands, **config["model"])
    model = FaithSR3X(model_cfg).to(device)
    train_ds = SyntheticSRDataset(train_files, config["data"]["crop_size"], args.bands, True, config["seed"])
    val_ds = SyntheticSRDataset(val_files, config["data"]["crop_size"], args.bands, False, config["seed"])
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["workers"], drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config["training"]["workers"])
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
    if not metrics_path.exists():
        metrics_path.write_text("epoch,train_loss,psnr,ssim,false_edge_rate,observation_l1,uncertainty_correlation,learning_rate\n", encoding="utf-8")
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        model.train()
        total = 0.0
        for lr, hr, degradation_target, _ in train_loader:
            lr, hr, degradation_target = lr.to(device), hr.to(device), degradation_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                sr, aux = model(lr, return_aux=True)
                loss, _ = objective(sr, hr, lr, aux, degradation_target)
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
        with metrics_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([epoch, total / max(1, len(train_loader)), validation.get("psnr"), validation.get("ssim"), validation.get("false_edge_rate"), validation.get("observation_l1"), validation.get("uncertainty_correlation"), optimizer.param_groups[0]["lr"]])
        print(json.dumps({"epoch": epoch, "train_loss": total / max(1, len(train_loader)), **validation}))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FaithSR-3X")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/faithsr3x_base.yaml")
    parser.add_argument("--output", default="runs/faithsr3x")
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
