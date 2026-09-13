"""Train one pre-registered FaithSR-Lite-X4 ablation variant."""
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
from .lite_x4 import FaithSRLiteX4, FaithSRLiteX4Config
from .losses import CompressionAwareObjective
from .model import count_parameters
from .qualitative import representative_sample_index, save_qualitative_panel
from .teacher import Swin2SRTeacher
from .train import evaluate


VARIANTS = {
    "base": {
        "use_degradation_adaptation": False,
        "use_observation_consistency": False,
        "use_compression_refinement": False,
        "teacher": False,
        "compression": False,
    },
    "distilled": {
        "use_degradation_adaptation": False,
        "use_observation_consistency": False,
        "use_compression_refinement": False,
        "teacher": True,
        "compression": False,
    },
    "compression": {
        "use_degradation_adaptation": True,
        "use_observation_consistency": True,
        "use_compression_refinement": True,
        "teacher": False,
        "compression": True,
    },
    "full": {
        "use_degradation_adaptation": True,
        "use_observation_consistency": True,
        "use_compression_refinement": True,
        "teacher": True,
        "compression": True,
    },
}


def variant_model_config(model_config: dict, variant: str) -> FaithSRLiteX4Config:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    switches = VARIANTS[variant]
    return FaithSRLiteX4Config(
        **model_config,
        use_degradation_adaptation=switches["use_degradation_adaptation"],
        use_observation_consistency=switches["use_observation_consistency"],
        use_compression_refinement=switches["use_compression_refinement"],
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config, allowed_scales={4})
    if int(config.get("scale", 4)) != 4:
        raise ValueError("FaithSR-Lite-X4 training requires scale: 4")
    switches = VARIANTS[args.variant]
    if switches["teacher"] and not args.teacher_checkpoint:
        raise ValueError(f"Variant {args.variant} requires --teacher-checkpoint")

    seed_everything(int(config["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    files = discover_images(args.data, set(config["data"]["extensions"]))
    train_files, validation_files = geographic_split(
        files,
        config["data"]["train_ratio"],
        config["seed"],
        config["data"].get("split_grid_degrees", 0.1),
    )
    if not train_files or not validation_files:
        raise ValueError("Scene-separated split produced an empty partition")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_split_manifest(
        output / "split.json",
        train_files,
        validation_files,
        config["seed"],
        config["data"].get("split_grid_degrees", 0.1),
    )

    degradation = config.get("degradation", {})
    data_options = {
        "crop_size": config["data"]["crop_size"],
        "bands": args.bands,
        "seed": config["seed"],
        "scale": 4,
        "mtf_probability": float(degradation.get("mtf_probability", 0.0)),
        "complex_probability": float(config["data"].get("mixed_probability", 1.0)),
    }
    train_data = SyntheticSRDataset(
        train_files,
        train=True,
        compression_focus_probability=(
            float(config["data"].get("compression_focus_probability", 0.0))
            if switches["compression"]
            else 0.0
        ),
        compression_focus_floor=float(config["data"].get("compression_focus_floor", 0.8)),
        **data_options,
    )
    validation_data = SyntheticSRDataset(validation_files, train=False, **data_options)
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["workers"],
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config["training"].get("validation_batch_size", config["training"]["batch_size"]),
        shuffle=False,
        num_workers=config["training"]["workers"],
    )

    model_config = variant_model_config({"in_channels": args.bands, "out_channels": args.bands, **config["model"]}, args.variant)
    model = FaithSRLiteX4(model_config).to(device)
    teacher = None
    if switches["teacher"]:
        teacher = Swin2SRTeacher(
            args.teacher_model,
            args.teacher_checkpoint,
            args.teacher_revision,
            config["training"].get("teacher_batch_size"),
        ).to(device)

    rationale = {
        "base": "Measure the compact native-x4 architecture without personalization or compression modules.",
        "distilled": "Isolate full-schedule output and edge distillation from the accepted personalized native-x4 teacher.",
        "compression": "Isolate compression-stratified training, degradation conditioning, compression gating, and bounded observation consistency.",
        "full": "Test whether personalized-teacher distillation and the diagnosed compression mechanism are complementary.",
    }[args.variant]
    write_model_card(
        output / "model-card.md",
        model,
        {**config, "variant": args.variant, "resolved_model": model.config.__dict__},
        f"faithsr-lite-x4-{args.variant}",
        rationale,
        "swin2sr-native-x4-diagnostic-v2",
    )

    loss_weights = dict(config["loss"])
    if not switches["teacher"]:
        loss_weights["distillation"] = 0.0
        loss_weights["distillation_edge"] = 0.0
    if not switches["compression"]:
        loss_weights["compression_focus"] = 0.0
        loss_weights["observation"] = 0.0
        loss_weights["degradation"] = 0.0
    objective = CompressionAwareObjective(loss_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = device.type == "cuda" and bool(config["training"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    preview_index, preview_texture = representative_sample_index(validation_data)
    degradation_path = output / "degradations.jsonl"
    degradation_path.write_text("", encoding="utf-8")
    metrics_path = output / "metrics.csv"
    metrics_path.write_text(
        "epoch,train_loss,psnr,ssim,false_edge_rate,observation_l1,learning_rate\n",
        encoding="utf-8",
    )
    started = time.time()
    best_psnr = float("-inf")

    manifest = {
        "variant": args.variant,
        "config": config,
        "git_commit": _git_commit(),
        "device": str(device),
        "parameters": count_parameters(model),
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()) if args.teacher_checkpoint else None,
        "teacher_checkpoint_sha256": _sha256(args.teacher_checkpoint) if args.teacher_checkpoint else None,
        "started_unix": started,
    }
    write_run_manifest(output / "run.json", manifest)

    for epoch in range(1, epochs + 1):
        curriculum_epochs = max(1, int(degradation.get("curriculum_epochs", 1)))
        progress = min(1.0, (epoch - 1) / max(1, curriculum_epochs - 1))
        start_severity = float(degradation.get("start_severity", 1.0))
        train_data.set_degradation_severity(start_severity + (1 - start_severity) * progress)
        model.train()
        total = 0.0
        for lr, hr, degradation_target, sources, details in train_loader:
            append_degradation_records(degradation_path, epoch, sources, details)
            lr, hr, degradation_target = lr.to(device), hr.to(device), degradation_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                sr, auxiliary = model(lr, return_aux=True)
                teacher_sr = teacher(lr) if teacher is not None else None
                if teacher_sr is not None and teacher_sr.shape != hr.shape:
                    raise ValueError(f"Native-x4 teacher shape {tuple(teacher_sr.shape)} != target {tuple(hr.shape)}")
                loss, _ = objective(sr, hr, lr, auxiliary, degradation_target, teacher_sr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()

        validation = evaluate(model, validation_loader, device)
        mean_loss = total / max(1, len(train_loader))
        with metrics_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([
                epoch,
                mean_loss,
                validation.get("psnr"),
                validation.get("ssim"),
                validation.get("false_edge_rate"),
                validation.get("observation_l1"),
                optimizer.param_groups[0]["lr"],
            ])
        checkpoint_state = {
            "epoch": epoch,
            "variant": args.variant,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": max(best_psnr, validation["psnr"]),
            "config": config,
        }
        save_checkpoint(output / "last.pth", model, **checkpoint_state)
        if validation["psnr"] > best_psnr:
            best_psnr = validation["psnr"]
            save_checkpoint(output / "best.pth", model, **checkpoint_state)
            preview_lr, preview_hr, _, preview_path, preview_degradation = validation_data[preview_index]
            with torch.inference_mode():
                preview_sr = model(preview_lr.unsqueeze(0).to(device))[0].clamp(0, 1)
            save_qualitative_panel(
                output / "best-preview.png",
                preview_lr,
                preview_hr,
                preview_sr,
                extra={
                    "epoch": epoch,
                    "variant": args.variant,
                    "source": preview_path,
                    "degradation": json.loads(preview_degradation),
                    "selection_rule": "maximum validation PSNR epoch; fixed median-texture held-out image",
                    "selection_index": preview_index,
                    "selection_hr_spatial_std": preview_texture,
                },
                method_label=f"FaithSR-Lite-X4 {args.variant}",
            )
        print(json.dumps({"variant": args.variant, "epoch": epoch, "train_loss": mean_loss, **validation}), flush=True)

    finished = time.time()
    manifest.update({
        "finished_unix": finished,
        "duration_seconds": finished - started,
        "best_validation_psnr": best_psnr,
        "degradation_manifest_sha256": _sha256(degradation_path),
    })
    write_run_manifest(output / "run.json", manifest)
    append_run_results(output / "model-card.md", metrics_path, manifest["duration_seconds"])
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--device")
    parser.add_argument("--teacher-model", default="caidas/swin2SR-classical-sr-x4-64")
    parser.add_argument("--teacher-revision", default="c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e")
    parser.add_argument("--teacher-checkpoint")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
