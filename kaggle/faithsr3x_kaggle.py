"""Thin Kaggle entry point for the preliminary EuroSAT RGB baseline."""
from pathlib import Path
import os
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
SOURCE_ROOT = Path("/kaggle/input/faithsr-3x-private-source")
OUTPUT_ROOT = Path("/kaggle/working/faithsr3x-run")


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    if not (SOURCE_ROOT / "src" / "faithsr").exists():
        raise RuntimeError(f"Private source dataset is not mounted correctly: {SOURCE_ROOT}")
    annual_crop = next((path for path in INPUT_ROOT.rglob("AnnualCrop") if any(path.glob("*.jpg"))), None)
    if annual_crop is None:
        raise RuntimeError(f"Could not discover EuroSAT RGB root under {INPUT_ROOT}: {list(INPUT_ROOT.iterdir())}")
    data_root = annual_crop.parent
    print(f"DATA_ROOT={data_root}", flush=True)
    run([
        sys.executable, "-m", "pip", "install", "torch==2.5.1",
        "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet",
    ])
    run([sys.executable, "-m", "pip", "install", "rasterio", "PyYAML", "--quiet"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT / "src")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    run([
        sys.executable, "-m", "faithsr.train", "--data", str(data_root),
        "--config", str(SOURCE_ROOT / "configs/eurosat_kaggle.yaml"),
        "--output", str(OUTPUT_ROOT),
    ], env=environment)
