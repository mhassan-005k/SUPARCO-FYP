from __future__ import annotations

import argparse
import json
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
def benchmark_model(model, device: torch.device, bands: int, size: int = 512, warmup: int = 10, runs: int = 100) -> dict:
    sample = torch.rand(1, bands, size, size, device=device)
    model.to(device).eval()
    for _ in range(warmup): model(sample)
    synchronize(device)
    values = []
    for _ in range(runs):
        start = time.perf_counter(); model(sample); synchronize(device)
        values.append((time.perf_counter() - start) * 1000)
    ordered = sorted(values)
    return {"runs": runs, "warmup": warmup, "mean_ms": statistics.mean(values), "median_ms": statistics.median(values), "p95_ms": ordered[max(0, int(0.95 * len(ordered)) - 1)], "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None, "scope": "model_only"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--device")
    parser.add_argument("--size", type=int, default=512); parser.add_argument("--warmup", type=int, default=10); parser.add_argument("--runs", type=int, default=100); parser.add_argument("--output", default="artifacts/results/metrics/latency.json")
    args = parser.parse_args(); device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_checkpoint(args.checkpoint, device)
    result = benchmark_model(model, device, model.config.in_channels, args.size, args.warmup, args.runs)
    result.update({"device": str(device), "input": [1, model.config.in_channels, args.size, args.size], "output": [1, model.config.out_channels, args.size * 3, args.size * 3]})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
