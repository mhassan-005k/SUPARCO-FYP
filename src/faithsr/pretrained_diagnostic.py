"""Native-x4 Swin2SR stress evaluation and deployment diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from .benchmark import benchmark_model, benchmark_tiff_pipeline
from .compare import _score_rows, paired_cluster_intervals
from .data import discover_images, geographic_split, read_image
from .degradation import DegradationParameters, degrade
from .metrics import LPIPSMetric
from .inference import file_sha256
from .pretrained_finetune import _load
from .qualitative import save_qualitative_panel


STRESS_PROFILES: dict[str, DegradationParameters] = {
    "clean_bicubic": DegradationParameters(0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 100, "bicubic_reference", None, 0.0),
    "mild_anisotropic": DegradationParameters(0.25, 0.10, 0.10, 0.20, 0.75, 0.55, 25.0, 95, "anisotropic_gaussian", None, 0.25),
    "severe_blur": DegradationParameters(0.90, 0.10, 0.10, 0.30, 2.25, 1.70, 70.0, 95, "anisotropic_gaussian", None, 0.90),
    "noise_heavy": DegradationParameters(0.20, 0.90, 0.10, 0.30, 0.70, 0.50, 10.0, 95, "anisotropic_gaussian", None, 0.90),
    "compression_heavy": DegradationParameters(0.20, 0.10, 0.90, 0.30, 0.70, 0.50, 40.0, 55, "anisotropic_gaussian", None, 0.90),
    "aliasing_heavy": DegradationParameters(0.25, 0.10, 0.10, 0.90, 0.80, 0.45, 110.0, 95, "anisotropic_gaussian", None, 0.90),
    "mtf_low_nyquist": DegradationParameters(0.80, 0.10, 0.10, 0.40, 0.82, 0.65, 15.0, 95, "mtf_gaussian", 0.15, 0.80),
}


class Swin2SRNativeX4Adapter(nn.Module):
    """Expose Hugging Face Swin2SR through the common TIFF inference contract."""

    scale = 4

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(in_channels=3, out_channels=3, scale=4)

    def forward(self, image: torch.Tensor, return_aux: bool = False):
        reconstruction = self.model(pixel_values=image).reconstruction
        if return_aux:
            return reconstruction, {"uncertainty": None, "reprojected_lr": None}
        return reconstruction


def _center_crop(path: Path, crop_size: int = 192) -> torch.Tensor:
    image = read_image(path, 3)
    _, height, width = image.shape
    if min(height, width) < crop_size:
        raise ValueError(f"Image smaller than {crop_size}: {path}")
    top, left = (height - crop_size) // 2, (width - crop_size) // 2
    return image[:, top:top + crop_size, left:left + crop_size]


def _mean_by_method(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    excluded = {"method", "path", "geographic_group", "degradation", "degradation_profile", "sample_count"}
    summary: dict[str, dict[str, float | int]] = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: statistics.fmean(row[key] for row in selected if row.get(key) is not None)
            for key in rows[0]
            if key not in excluded and any(row.get(key) is not None for row in selected)
        }
        summary[method]["samples"] = len(selected)
    return summary


def _predict_native_x4(model: nn.Module, image: torch.Tensor, input_style: str) -> torch.Tensor:
    if input_style == "pixel_values":
        return model(pixel_values=image).reconstruction
    if input_style == "tensor":
        output = model(image)
        return output.reconstruction if hasattr(output, "reconstruction") else output
    raise ValueError(f"Unknown native-x4 model input style: {input_style}")


@torch.inference_mode()
def stress_evaluate(
    model: nn.Module,
    files: list[Path],
    output_dir: str | Path,
    device: torch.device,
    crop_size: int = 192,
    perceptual: LPIPSMetric | None = None,
    frozen_model: nn.Module | None = None,
    candidate_method: str = "swin2sr_full_finetune_native_x4",
    candidate_label: str = "Swin2SR FT native x4",
    input_style: str = "pixel_values",
) -> dict:
    """Evaluate matched x4 methods under deterministic, named degradation families."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()
    if frozen_model is not None:
        frozen_model.to(device).eval()
    rows: list[dict] = []
    cases: list[dict] = []
    for file_index, path in enumerate(files):
        hr_cpu = _center_crop(path, crop_size)
        for profile_index, (profile_name, parameters) in enumerate(STRESS_PROFILES.items()):
            seed = 10_000 + file_index * 100 + profile_index
            lr_cpu, _ = degrade(hr_cpu, scale=4, params=parameters, seed=seed)
            lr, hr = lr_cpu.unsqueeze(0).to(device), hr_cpu.unsqueeze(0).to(device)
            prediction = _predict_native_x4(model, lr, input_style).clamp(0, 1)
            if prediction.shape != hr.shape:
                raise ValueError(f"Native x4 prediction shape {tuple(prediction.shape)} != HR {tuple(hr.shape)}")
            bicubic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            frozen = None if frozen_model is None else frozen_model(pixel_values=lr).reconstruction.clamp(0, 1)
            degradation_record = json.dumps({"profile": profile_name, "seed": seed, "scale": 4, **parameters.to_dict()}, sort_keys=True)
            lpips_prediction = None if perceptual is None else perceptual(prediction, hr)
            lpips_bicubic = None if perceptual is None else perceptual(bicubic, hr)
            lpips_frozen = None if perceptual is None or frozen is None else perceptual(frozen, hr)
            model_rows = _score_rows(candidate_method, prediction, hr, [str(path)], [degradation_record], lpips_values=lpips_prediction, scale=4)
            baseline_rows = _score_rows("bicubic_x4", bicubic, hr, [str(path)], [degradation_record], lpips_values=lpips_bicubic, scale=4)
            frozen_rows = [] if frozen is None else _score_rows("swin2sr_frozen_native_x4", frozen, hr, [str(path)], [degradation_record], lpips_values=lpips_frozen, scale=4)
            reobserved_model, _ = degrade(prediction[0].detach().cpu(), scale=4, params=parameters, seed=seed)
            reobserved_bicubic, _ = degrade(bicubic[0].detach().cpu(), scale=4, params=parameters, seed=seed)
            scored = [(model_rows[0], reobserved_model), (baseline_rows[0], reobserved_bicubic)]
            if frozen is not None:
                reobserved_frozen, _ = degrade(frozen[0].detach().cpu(), scale=4, params=parameters, seed=seed)
                scored.append((frozen_rows[0], reobserved_frozen))
            for row, reobserved in scored:
                row["degradation_profile"] = profile_name
                row["path"] = f"{path}#profile={profile_name}"
                row["observation_l1"] = float(F.l1_loss(reobserved, lr_cpu))
                rows.append(row)
            cases.append({
                "path": str(path), "profile": profile_name, "seed": seed,
                "psnr_delta": model_rows[0]["psnr"] - baseline_rows[0]["psnr"],
            })

    comparison = target / "stress-comparison.csv"
    with comparison.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    by_profile = {}
    for profile_name in STRESS_PROFILES:
        selected = [row for row in rows if row["degradation_profile"] == profile_name]
        by_profile[profile_name] = {
            "methods": _mean_by_method(selected),
            "paired_scene_intervals_vs_bicubic": paired_cluster_intervals(selected, reference="bicubic_x4"),
        }
    deficiency_ranking = []
    for profile_name, profile_result in by_profile.items():
        methods = profile_result["methods"]
        candidate = methods[candidate_method]
        baseline = methods["bicubic_x4"]
        deficiency_ranking.append({
            "profile": profile_name,
            "psnr_delta_db": candidate["psnr"] - baseline["psnr"],
            "false_edge_delta": candidate["false_edge_rate"] - baseline["false_edge_rate"],
            "observation_l1_delta": candidate["observation_l1"] - baseline["observation_l1"],
        })
    deficiency_ranking.sort(key=lambda item: (item["psnr_delta_db"], -item["false_edge_delta"], item["profile"]))
    summary = {
        "scale": 4,
        "protocol": "deterministic named degradation families; same LR/HR observation per paired method",
        "profiles": by_profile,
        "overall_methods": _mean_by_method(rows),
        "overall_paired_scene_intervals_vs_bicubic": paired_cluster_intervals(rows, reference="bicubic_x4"),
        "files": len(files),
        "paired_cases": len(cases),
        "deficiency_ranking_worst_first": deficiency_ranking,
    }
    (target / "stress-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ordered = sorted(cases, key=lambda item: (item["psnr_delta"], item["path"], item["profile"]))
    selections = {"worst": ordered[0], "median": ordered[(len(ordered) - 1) // 2], "best": ordered[-1]}
    for label, case in selections.items():
        path = Path(case["path"])
        parameters = STRESS_PROFILES[case["profile"]]
        hr = _center_crop(path, crop_size)
        lr, _ = degrade(hr, scale=4, params=parameters, seed=case["seed"])
        prediction = _predict_native_x4(model, lr.unsqueeze(0).to(device), input_style)[0].clamp(0, 1)
        save_qualitative_panel(
            target / f"{label}-native-x4.png", lr, hr, prediction,
            extra={**case, "selection_rule": f"{label} {candidate_method}-minus-bicubic PSNR over every held-out stress case", "degradation": parameters.to_dict()},
            method_label=candidate_label,
        )
    return summary


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_id = state.get("model_id", args.model_id)
    revision = state.get("revision", args.revision)
    model = _load(model_id, revision, args.checkpoint).to(device)
    frozen_model = _load(model_id, revision).to(device)
    files = discover_images(args.data)
    _, validation = geographic_split(files, args.train_ratio, args.seed, args.grid_degrees)
    if not validation:
        raise ValueError("No held-out validation files found")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    perceptual = LPIPSMetric(device, args.lpips_network) if args.lpips else None
    summary = stress_evaluate(model, validation, output, device, args.crop_size, perceptual, frozen_model)
    frozen_model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    adapter = Swin2SRNativeX4Adapter(model)
    benchmarks = {}
    for precision in args.precisions:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            benchmark = benchmark_model(adapter, device, 3, 512, args.warmup, args.runs, precision)
            benchmark.update({"status": "pass", "device": str(device), "input": [1, 3, 512, 512], "output": [1, 3, 2048, 2048], "scale": 4})
            if args.tiff_input:
                benchmark["end_to_end"] = benchmark_tiff_pipeline(adapter, device, args.tiff_input, str(output / f"pipeline-{precision}"), args.pipeline_runs, 512, 32, args.compression, precision)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            benchmark = {
                "status": "failed_out_of_memory", "error": str(error), "precision": precision,
                "device": str(device), "input": [1, 3, 512, 512], "intended_output": [1, 3, 2048, 2048], "scale": 4,
            }
        benchmarks[precision] = benchmark
        (output / f"latency-512-{precision}.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": file_sha256(args.checkpoint), "model_id": model_id, "revision": revision,
        "target_scale": 4, "validation_files": [str(path) for path in validation],
        "stress_summary": "stress-summary.json", "latency": {precision: f"latency-512-{precision}.json" for precision in benchmarks},
        "hybrid_decision_rule": "select a component only after ranking native-x4 degradation/fidelity weaknesses",
    }
    (output / "diagnostic-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"stress": summary, "latency": benchmarks}, indent=2))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--data", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="caidas/swin2SR-classical-sr-x4-64")
    parser.add_argument("--revision", default="c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e")
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--train-ratio", type=float, default=0.5); parser.add_argument("--grid-degrees", type=float, default=0.1)
    parser.add_argument("--crop-size", type=int, default=192); parser.add_argument("--device")
    parser.add_argument("--lpips", action="store_true"); parser.add_argument("--lpips-network", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "fp16"], default=["fp32", "fp16"]); parser.add_argument("--warmup", type=int, default=10); parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--tiff-input"); parser.add_argument("--pipeline-runs", type=int, default=3); parser.add_argument("--compression", choices=["input", "deflate", "lzw", "zstd", "none"], default="none")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
