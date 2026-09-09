"""Frozen-probe land-cover utility evaluation for FaithSR-3X."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/faithsr3x-downstream-landcover")


def run(command: list[str], environment: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def source_root() -> Path:
    mount = next((path for path in INPUT_ROOT.rglob("faithsr-3x-private-source") if path.is_dir()), None)
    if mount is None:
        raise RuntimeError("Private source dataset was not mounted")
    if (mount / "src" / "faithsr").exists():
        return mount
    root = WORK / "source"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(next(mount.rglob("src.zip")), root / "src")
    shutil.unpack_archive(next(mount.rglob("configs.zip")), root / "configs")
    return root


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    run([sys.executable, "-m", "pip", "install", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet"], environment)
    run([sys.executable, "-m", "pip", "install", "rasterio", "PyYAML", "--quiet"], environment)
    source = source_root()
    environment["PYTHONPATH"] = str(source / "src")
    mounted_checkpoint = next((path for path in INPUT_ROOT.rglob("best.pth") if path.parent.name == "full"), None)
    if mounted_checkpoint is not None:
        checkpoint = mounted_checkpoint
        split = next(mounted_checkpoint.parent.rglob("split.json"))
    else:
        full_archive = next(INPUT_ROOT.rglob("full.zip"), None)
        if full_archive is None:
            raise RuntimeError("Frozen Full Sentinel artifact was not mounted")
        full = WORK / "frozen-full-v1"; full.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(full_archive, full)
        checkpoint = next(full.rglob("best.pth")); split = next(full.rglob("split.json"))
    rgb = next((path for path in INPUT_ROOT.rglob("rgb") if any(path.glob("*.png"))), None)
    worldcover = next(INPUT_ROOT.rglob("worldcover_bbox_delhi_ncr_2021.tif"), None)
    if rgb is None or worldcover is None:
        raise RuntimeError(f"Delhi inputs missing: rgb={rgb}, worldcover={worldcover}")
    run([
        sys.executable, "-m", "faithsr.downstream", "--data", str(rgb), "--worldcover", str(worldcover),
        "--split-manifest", str(split), "--checkpoint", str(checkpoint), "--output", str(WORK / "results"),
        "--crop-size", "126", "--epochs", "12", "--batch-size", "64", "--workers", "2",
    ], environment)
