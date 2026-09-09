from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx(checkpoint: str, output: str, size: int = 128, device: str = "cpu", verify: bool = True) -> dict:
    model, _ = load_checkpoint(checkpoint, device); model.to(device).eval()
    sample = torch.rand(1, model.config.in_channels, size, size, device=device)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, target, input_names=["lr"], output_names=["sr"], opset_version=18, dynamo=False, dynamic_axes={"lr": {2: "height", 3: "width"}, "sr": {2: "height_x3", 3: "width_x3"}})
    result = {"checkpoint": checkpoint, "onnx": str(target), "onnx_sha256": _sha256(target), "opset": 18, "batch_size": 1, "dynamic_spatial_axes": True}
    if verify:
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install faithsr[export] to verify ONNX artifacts") from error
        onnx.checker.check_model(onnx.load(target))
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        checks = []
        generator = np.random.default_rng(42)
        for height, width in ((size, size), (size + 3, size + 5)):
            values = generator.random((1, model.config.in_channels, height, width), dtype=np.float32)
            with torch.inference_mode():
                expected = model(torch.from_numpy(values).to(device)).cpu().numpy()
            actual = session.run(None, {"lr": values})[0]
            difference = np.abs(expected - actual)
            checks.append({"input": list(values.shape), "output": list(actual.shape), "max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean())})
        result["verification"] = {"onnx_checker": "pass", "runtime": "onnxruntime-cpu", "dynamic_shape_checks": checks}
    target.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", default="exports/faithsr3x.onnx"); parser.add_argument("--size", type=int, default=128); parser.add_argument("--device", default="cpu"); parser.add_argument("--no-verify", action="store_false", dest="verify")
    print(json.dumps(export_onnx(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__": main()
