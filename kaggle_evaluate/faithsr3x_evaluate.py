"""Matched EuroSAT evaluation for the completed preliminary FaithSR-3X run."""
from pathlib import Path
import os
import shutil
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/faithsr3x-evaluation")


def run(command: list[str], env: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    source_mount = next((path for path in INPUT_ROOT.rglob("faithsr-3x-private-source") if path.is_dir()), None)
    if source_mount is None:
        raise RuntimeError("Private source dataset mount could not be discovered")
    source_root = source_mount
    if not (source_root / "src" / "faithsr").exists():
        source_zip = next(source_mount.rglob("src.zip"), None)
        if source_zip is not None:
            source_root = Path("/kaggle/working/source")
            (source_root / "src").mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(source_zip, source_root / "src")
    annual_crop = next((path for path in INPUT_ROOT.rglob("AnnualCrop") if any(path.glob("*.jpg"))), None)
    checkpoint = next(INPUT_ROOT.rglob("best.pth"), None)
    split = next(INPUT_ROOT.rglob("split.json"), None)
    print({"input_roots": [str(path) for path in INPUT_ROOT.iterdir()], "source": str(source_root), "annual_crop": str(annual_crop), "checkpoint": str(checkpoint), "split": str(split)}, flush=True)
    if not (source_root / "src" / "faithsr").exists() or annual_crop is None or checkpoint is None or split is None:
        raise RuntimeError("Required source, EuroSAT, checkpoint, or frozen split input is missing; see discovery diagnostics above")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root / "src")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    run([sys.executable, "-m", "pip", "install", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet"], environment)
    run([sys.executable, "-m", "pip", "install", "rasterio", "PyYAML", "--quiet"], environment)
    run([
        sys.executable, "-m", "faithsr.compare", "--data", str(annual_crop.parent),
        "--checkpoint", str(checkpoint), "--split-manifest", str(split),
        "--crop-size", "63", "--batch-size", "64", "--workers", "2",
        "--output", str(OUTPUT_ROOT / "comparison.csv"),
    ], environment)
    run([
        sys.executable, "-m", "faithsr.benchmark", "--checkpoint", str(checkpoint),
        "--size", "512", "--warmup", "5", "--runs", "20",
        "--output", str(OUTPUT_ROOT / "latency_512.json"),
    ], environment)
