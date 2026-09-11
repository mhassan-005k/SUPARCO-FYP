"""Run the required frozen-versus-full-fine-tuned Swin2SR control on native L2A tiles."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/faithsr3x-swin2sr-full-finetune")


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


def native_patches() -> Path:
    candidates = [path for path in INPUT_ROOT.rglob("patches") if path.is_dir() and len(list(path.glob("S2*_L2A"))) >= 6]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one native L2A patch corpus from the parent kernel; found {candidates}")
    return candidates[0]


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    run([sys.executable, "-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet"], environment)
    run([sys.executable, "-m", "pip", "install", "transformers==4.47.1", "lpips==0.1.4", "scikit-image==0.25.2", "PyYAML==6.0.3", "--quiet"], environment)
    source = source_root()
    environment["PYTHONPATH"] = str(source / "src")
    (WORK / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], env=environment, text=True), encoding="utf-8")
    run([
        sys.executable, "-m", "faithsr.pretrained_finetune",
        "--data", str(native_patches()),
        "--config", str(source / "configs" / "sentinel_l2a_native_kaggle.yaml"),
        "--output", str(WORK / "swin2sr"),
        "--epochs", "3", "--batch-size", "1", "--eval-batch-size", "1", "--workers", "2",
        "--learning-rate", "0.00001", "--lpips",
    ], environment)
