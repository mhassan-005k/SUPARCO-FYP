"""Evaluate the four FaithSR-Lite-X4 ablations under the frozen native-x4 protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark import benchmark_model, benchmark_tiff_pipeline
from .checkpoint import load_checkpoint
from .data import discover_images, geographic_split
from .inference import file_sha256
from .metrics import LPIPSMetric
from .pretrained_diagnostic import stress_evaluate


EXPECTED_VARIANTS = ("base", "distilled", "compression", "full")


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Checkpoint must use variant=path")
        variant, path = value.split("=", 1)
        if variant not in EXPECTED_VARIANTS:
            raise ValueError(f"Unknown checkpoint variant: {variant}")
        if variant in parsed:
            raise ValueError(f"Duplicate checkpoint variant: {variant}")
        parsed[variant] = Path(path)
    missing = set(EXPECTED_VARIANTS) - parsed.keys()
    if missing:
        raise ValueError(f"Missing checkpoint variants: {sorted(missing)}")
    return parsed


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
    results = {}

    for variant in EXPECTED_VARIANTS:
        checkpoint = checkpoints[variant]
        model, state = load_checkpoint(checkpoint, device)
        if state.get("variant") != variant:
            raise ValueError(f"Checkpoint {checkpoint} records variant {state.get('variant')}, expected {variant}")
        if getattr(model.config, "scale", None) != 4:
            raise ValueError(f"Checkpoint {checkpoint} is not native x4")
        model.to(device).eval()
        method = f"faithsr_lite_x4_{variant}"
        stress = stress_evaluate(
            model,
            validation,
            output / variant / "stress",
            device,
            args.crop_size,
            perceptual,
            candidate_method=method,
            candidate_label=f"FaithSR-Lite-X4 {variant}",
            input_style="tensor",
        )
        latency = {}
        for precision in args.precisions:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            try:
                benchmark = benchmark_model(model, device, model.config.in_channels, 512, args.warmup, args.runs, precision)
                benchmark.update({
                    "status": "pass",
                    "device": str(device),
                    "input": [1, model.config.in_channels, 512, 512],
                    "output": [1, model.config.out_channels, 2048, 2048],
                    "scale": 4,
                })
                if variant == "full" and args.tiff_input:
                    benchmark["end_to_end"] = benchmark_tiff_pipeline(
                        model,
                        device,
                        args.tiff_input,
                        str(output / variant / f"pipeline-{precision}"),
                        args.pipeline_runs,
                        512,
                        32,
                        args.compression,
                        precision,
                    )
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                benchmark = {
                    "status": "failed_out_of_memory",
                    "error": str(error),
                    "precision": precision,
                    "device": str(device),
                }
            latency[precision] = benchmark
            latency_path = output / variant / f"latency-512-{precision}.json"
            latency_path.parent.mkdir(parents=True, exist_ok=True)
            latency_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

        comparison = stress["overall_paired_scene_intervals_vs_bicubic"][f"{method}_vs_bicubic_x4"]
        results[variant] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "overall_metrics": stress["overall_methods"][method],
            "paired_scene_intervals_vs_bicubic": comparison,
            "deficiency_ranking_worst_first": stress["deficiency_ranking_worst_first"],
            "latency": latency,
        }
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranked = sorted(
        EXPECTED_VARIANTS,
        key=lambda name: results[name]["paired_scene_intervals_vs_bicubic"]["psnr"]["mean_delta"],
        reverse=True,
    )
    payload = {
        "scale": 4,
        "validation_files": [str(path) for path in validation],
        "variants": results,
        "quality_ranking_best_first": ranked,
        "acceptance_rule": {
            "latency": "model-only P95 below 1000 ms on declared hardware",
            "compression": "positive compression-heavy PSNR CI without worse overall false-edge or observation L1",
            "selection": "quality-latency Pareto improvement; no visual-only claim",
        },
    }
    (output / "ablation-summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="Repeat variant=path for base, distilled, compression, and full")
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
