from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint


def export_onnx(checkpoint: str, output: str, size: int = 128, device: str = "cpu") -> dict:
    model, _ = load_checkpoint(checkpoint, device); model.to(device).eval()
    sample = torch.rand(1, model.config.in_channels, size, size, device=device)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, target, input_names=["lr"], output_names=["sr"], opset_version=18, dynamo=False, dynamic_axes={"lr": {2: "height", 3: "width"}, "sr": {2: "height_x3", 3: "width_x3"}})
    result = {"checkpoint": checkpoint, "onnx": str(target), "opset": 18, "batch_size": 1, "dynamic_spatial_axes": True}
    target.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", default="exports/faithsr3x.onnx"); parser.add_argument("--size", type=int, default=128); parser.add_argument("--device", default="cpu")
    print(json.dumps(export_onnx(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__": main()
