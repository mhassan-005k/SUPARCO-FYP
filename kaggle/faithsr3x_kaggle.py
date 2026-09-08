"""Thin Kaggle entry point for the preliminary EuroSAT RGB baseline."""
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path("/kaggle/working/SUPARCO-FYP")
DATA_ROOT = Path("/kaggle/input/eurosat-dataset/EuroSAT")
OUTPUT_ROOT = Path("/kaggle/working/faithsr3x-run")


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    if not REPO_ROOT.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/mhassan-005k/SUPARCO-FYP.git", str(REPO_ROOT)])
    if not DATA_ROOT.exists():
        raise RuntimeError("Attach a versioned dataset and update DATA_ROOT")
    run([sys.executable, "-m", "pip", "install", "rasterio", "PyYAML", "--quiet"])
    run([sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT), "--no-deps", "--quiet"])
    run([
        sys.executable, "-m", "faithsr.train", "--data", str(DATA_ROOT),
        "--config", str(REPO_ROOT / "configs/eurosat_kaggle.yaml"),
        "--output", str(OUTPUT_ROOT),
    ])
