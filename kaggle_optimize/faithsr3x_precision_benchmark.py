"""Matched 100-run FP32/FP16 model and disk-backed TIFF benchmark."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/faithsr3x-precision-benchmark")


def run(command: list[str], environment: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def source_root() -> Path:
    mount = next((path for path in INPUT_ROOT.rglob("faithsr-3x-private-source") if path.is_dir()), None)
    if mount is None:
        raise RuntimeError("Private source dataset was not mounted")
    if (mount / "src" / "faithsr").exists():
        return mount
    root = WORK / "source"; (root / "src").mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(next(mount.rglob("src.zip")), root / "src")
    return root


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy(); environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    run([sys.executable, "-m", "pip", "install", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet"], environment)
    run([sys.executable, "-m", "pip", "install", "rasterio", "PyYAML", "--quiet"], environment)
    source = source_root(); environment["PYTHONPATH"] = str(source / "src")
    checkpoint = next(INPUT_ROOT.rglob("best.pth"), None)
    if checkpoint is None:
        raise RuntimeError("EuroSAT checkpoint source was not mounted")
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    fixture = Path("/kaggle/temp/faithsr_precision_512.tiff"); fixture.parent.mkdir(parents=True, exist_ok=True)
    axis = np.linspace(0, 255, 512, dtype=np.uint8)
    pixels = np.stack([np.tile(axis, (512, 1)), np.tile(axis[:, None], (1, 512)), np.full((512, 512), 127, np.uint8)])
    with rasterio.open(fixture, "w", driver="GTiff", width=512, height=512, count=3, dtype="uint8", crs="EPSG:32643", transform=from_origin(500000, 3000000, 10, 10), compress="deflate") as target:
        target.write(pixels); target.update_tags(PRODUCT="deterministic-precision-benchmark")
    for precision in ("fp32", "fp16"):
        run([
            sys.executable, "-m", "faithsr.benchmark", "--checkpoint", str(checkpoint), "--size", "512",
            "--warmup", "10", "--runs", "100", "--precision", precision, "--tiff-input", str(fixture),
            "--pipeline-runs", "10", "--pipeline-output-dir", f"/kaggle/temp/pipeline_{precision}",
            "--pipeline-compression", "none", "--output", str(WORK / f"benchmark_{precision}.json"),
        ], environment)
    with rasterio.open("/kaggle/temp/pipeline_fp32/pipeline_0.tiff") as source32, rasterio.open("/kaggle/temp/pipeline_fp16/pipeline_0.tiff") as source16:
        fp32 = source32.read().astype(np.int16); fp16 = source16.read().astype(np.int16)
        metadata_equal = source32.crs == source16.crs and source32.transform == source16.transform and source32.shape == source16.shape
    difference = np.abs(fp32 - fp16)
    report = {
        "protocol": "same checkpoint/input/P100; 10 warmups + 100 model runs; 10 full uncompressed TIFF runs",
        "fp32": json.loads((WORK / "benchmark_fp32.json").read_text()),
        "fp16": json.loads((WORK / "benchmark_fp16.json").read_text()),
        "uint8_output_parity": {"max_abs_dn": int(difference.max()), "mean_abs_dn": float(difference.mean()), "changed_fraction": float((difference != 0).mean()), "metadata_equal": metadata_equal},
    }
    (WORK / "precision_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
