from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .inference import process_tiff


def synchronize(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_model(model, device: torch.device, bands: int, size: int = 512, warmup: int = 10, runs: int = 100, precision: str = "fp32") -> dict:
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 benchmarking requires CUDA")
    sample = torch.rand(1, bands, size, size, device=device)
    model.to(device).eval()
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=precision == "fp16"): model(sample)
    synchronize(device)
    values = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=precision == "fp16"): model(sample)
        synchronize(device)
        values.append((time.perf_counter() - start) * 1000)
    ordered = sorted(values)
    return {"runs": runs, "warmup": warmup, "mean_ms": statistics.mean(values), "median_ms": statistics.median(values), "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None, "scope": "model_only", "precision": precision}


def benchmark_tiff_pipeline(model, device: torch.device, input_path: str, output_dir: str, runs: int = 3, tile_size: int = 512, overlap: int = 32, compression: str = "input", precision: str = "fp32") -> dict:
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    values = []
    component_rows = []
    for index in range(runs):
        output = directory / f"pipeline_{index}.tiff"
        audit = process_tiff(input_path, output, model, device, tile_size, overlap, model_version="benchmark", compression=compression, precision=precision)
        values.append(audit["timing_ms"]["total"])
        component_rows.append(audit["timing_ms"])
    ordered = sorted(values)
    numeric_components = component_rows[0].keys()
    return {
        "runs": runs, "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], "scope": "end_to_end_tiff_disk_backed",
        "component_mean_ms": {key: statistics.mean(row[key] for row in component_rows) for key in numeric_components},
        "tile_size": tile_size, "overlap": overlap, "compression": compression, "precision": precision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--device")
    parser.add_argument("--size", type=int, default=512); parser.add_argument("--warmup", type=int, default=10); parser.add_argument("--runs", type=int, default=100); parser.add_argument("--output", default="artifacts/results/metrics/latency.json")
    parser.add_argument("--tiff-input"); parser.add_argument("--pipeline-runs", type=int, default=3); parser.add_argument("--pipeline-output-dir", default="runs/benchmark_pipeline")
    parser.add_argument("--pipeline-compression", choices=["input", "deflate", "lzw", "zstd", "none"], default="input")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args(); device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_checkpoint(args.checkpoint, device)
    result = benchmark_model(model, device, model.config.in_channels, args.size, args.warmup, args.runs, args.precision)
    result.update({"device": str(device), "input": [1, model.config.in_channels, args.size, args.size], "output": [1, model.config.out_channels, args.size * 3, args.size * 3]})
    if args.tiff_input:
        result["end_to_end"] = benchmark_tiff_pipeline(model, device, args.tiff_input, args.pipeline_output_dir, args.pipeline_runs, args.size, 32, args.pipeline_compression, args.precision)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
