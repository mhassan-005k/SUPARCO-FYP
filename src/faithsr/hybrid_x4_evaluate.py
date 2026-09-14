"""Evaluate the matched native-x4 control and both hybrid approaches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .ablation import paired_variant_intervals
from .benchmark import benchmark_model, benchmark_tiff_pipeline
from .checkpoint import load_checkpoint
from .data import discover_images, geographic_split
from .hybrid_x4 import HYBRID_X4_VARIANTS
from .inference import file_sha256
from .metrics import LPIPSMetric
from .pretrained_diagnostic import stress_evaluate


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Checkpoint must use variant=path")
        variant, path = value.split("=", 1)
        if variant not in HYBRID_X4_VARIANTS:
            raise ValueError(f"Unknown checkpoint variant: {variant}")
        if variant in parsed:
            raise ValueError(f"Duplicate checkpoint variant: {variant}")
        parsed[variant] = Path(path)
    missing = set(HYBRID_X4_VARIANTS) - parsed.keys()
    if missing:
        raise ValueError(f"Missing checkpoint variants: {sorted(missing)}")
    return parsed


def _latency_pass(benchmark: dict, limit_ms: float) -> bool:
    return benchmark.get("status") == "pass" and benchmark.get("p95_ms", float("inf")) <= limit_ms


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = parse_checkpoints(args.checkpoint)
    files = discover_images(args.data)
    _, validation = geographic_split(files, args.train_ratio, args.seed, args.grid_degrees)
    if not validation:
        raise ValueError("No held-out validation files found")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    perceptual = LPIPSMetric(device, args.lpips_network) if args.lpips else None
    results: dict[str, dict] = {}
    comparison_csv: dict[str, Path] = {}

    for variant in HYBRID_X4_VARIANTS:
        checkpoint = checkpoints[variant]
        model, state = load_checkpoint(checkpoint, device)
        if state.get("variant") != variant or model.config.variant != variant:
            raise ValueError(f"Checkpoint {checkpoint} does not match {variant}")
        model.to(device).eval()
        method = f"faithsr_hybrid_x4_{variant}"
        stress_dir = output / variant / "stress"
        stress = stress_evaluate(
            model, validation, stress_dir, device, args.crop_size, perceptual,
            candidate_method=method, candidate_label=f"FaithSR Hybrid X4 {variant}", input_style="tensor",
        )
        comparison_csv[variant] = stress_dir / "stress-comparison.csv"
        latency: dict[str, dict] = {}
        for precision in args.precisions:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            try:
                benchmark = benchmark_model(model, device, model.config.in_channels, 512, args.warmup, args.runs, precision)
                benchmark.update({"status": "pass", "device": str(device), "input": [1, model.config.in_channels, 512, 512], "output": [1, model.config.out_channels, 2048, 2048], "scale": 4})
                if variant != "medium_base" and args.tiff_input:
                    benchmark["end_to_end"] = benchmark_tiff_pipeline(model, device, args.tiff_input, str(output / variant / f"pipeline-{precision}"), args.pipeline_runs, 512, 32, args.compression, precision)
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                benchmark = {"status": "failed_out_of_memory", "error": str(error), "precision": precision, "device": str(device)}
            latency[precision] = benchmark
            latency_path = output / variant / f"latency-512-{precision}.json"
            latency_path.parent.mkdir(parents=True, exist_ok=True)
            latency_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

        vs_bicubic = stress["overall_paired_scene_intervals_vs_bicubic"][f"{method}_vs_bicubic_x4"]
        results[variant] = {
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "overall_metrics": stress["overall_methods"][method],
            "paired_scene_intervals_vs_bicubic": vs_bicubic,
            "deficiency_ranking_worst_first": stress["deficiency_ranking_worst_first"], "latency": latency,
        }
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    paired = paired_variant_intervals(
        comparison_csv,
        sequence=list(HYBRID_X4_VARIANTS),
        pairs=[("medium_base", "context_fusion"), ("medium_base", "residual_cascade"), ("context_fusion", "residual_cascade")],
    )
    primary_precision = "fp16" if "fp16" in args.precisions and device.type == "cuda" else args.precisions[0]
    decisions = {}
    for variant in ("context_fusion", "residual_cascade"):
        quality = paired["comparisons"][f"{variant}_vs_medium_base"]
        benchmark = results[variant]["latency"][primary_precision]
        pipeline = benchmark.get("end_to_end", {})
        decisions[variant] = {
            "quality_vs_medium_base": quality,
            "model_latency_pass_300ms": _latency_pass(benchmark, 300.0),
            "hard_model_latency_pass_1000ms": _latency_pass(benchmark, 1000.0),
            "tiff_pipeline_pass_800ms": bool(pipeline) and pipeline.get("p95_ms", float("inf")) <= 800.0,
            "hard_tiff_pipeline_pass_1000ms": bool(pipeline) and pipeline.get("p95_ms", float("inf")) <= 1000.0,
        }

    ranked = sorted(
        ("context_fusion", "residual_cascade"),
        key=lambda name: (
            decisions[name]["hard_model_latency_pass_1000ms"],
            decisions[name]["quality_vs_medium_base"]["psnr"]["mean_delta"],
            -results[name]["overall_metrics"]["false_edge_rate"],
        ),
        reverse=True,
    )
    payload = {
        "scale": 4, "input_policy": "real TIFF is normalized and passed directly; synthetic degradation is training/evaluation only",
        "validation_files": [str(path) for path in validation], "variants": results,
        "paired_variant_intervals": paired, "acceptance": decisions, "hybrid_ranking_best_first": ranked,
        "selection_rule": "Prefer a hybrid only if it improves the matched control without a material fidelity regression and remains below one second P95 for a 512x512 block on the declared device.",
        "primary_precision": primary_precision, "device": str(device),
    }
    (output / "hybrid-comparison-summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="Repeat variant=path for all three variants")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--grid-degrees", type=float, default=0.1)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--device")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--lpips-network", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "fp16"], default=["fp32", "fp16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--tiff-input")
    parser.add_argument("--pipeline-runs", type=int, default=3)
    parser.add_argument("--compression", choices=["input", "deflate", "lzw", "zstd", "none"], default="none")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
