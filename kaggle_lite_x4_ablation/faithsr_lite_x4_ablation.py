"""Train and evaluate the pre-registered FaithSR-Lite-X4 four-way ablation."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/faithsr-lite-x4-ablation")


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
        raise RuntimeError(f"Expected one native L2A patch corpus; found {candidates}")
    return candidates[0]


def personalized_teacher_checkpoint() -> Path:
    candidates = [
        path for path in INPUT_ROOT.rglob("best.pth")
        if "swin2sr-native-x4" in path.as_posix().lower()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one accepted personalized native-x4 teacher checkpoint; found {candidates}")
    return candidates[0]


def make_fixture(patches: Path, target: Path) -> None:
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling

    source_path = next(patches.rglob("*.tif"))
    with rasterio.open(source_path) as source:
        profile = source.profile.copy()
        pixels = source.read([1, 2, 3], out_shape=(3, 512, 512), resampling=Resampling.cubic)
        transform = source.transform * Affine.scale(source.width / 512, source.height / 512)
        profile.update(driver="GTiff", width=512, height=512, count=3, dtype="uint8", transform=transform, compress="none")
        profile.pop("blockxsize", None); profile.pop("blockysize", None); profile.pop("tiled", None)
        tags = source.tags(); descriptions = source.descriptions[:3]
    with rasterio.open(target, "w", **profile) as output:
        output.write(pixels)
        output.update_tags(**tags)
        output.descriptions = descriptions


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    run([sys.executable, "-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet"], environment)
    run([sys.executable, "-m", "pip", "install", "transformers==4.47.1", "lpips==0.1.4", "scikit-image==0.25.2", "rasterio==1.4.3", "PyYAML==6.0.3", "--quiet"], environment)
    source = source_root()
    patches = native_patches()
    teacher = personalized_teacher_checkpoint()
    environment["PYTHONPATH"] = str(source / "src")
    (WORK / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], env=environment, text=True), encoding="utf-8")
    config = source / "configs" / "faithsr_lite_x4_ablation.yaml"
    checkpoints = []
    for variant in ("base", "distilled", "compression", "full"):
        output = WORK / variant
        command = [
            sys.executable, "-m", "faithsr.lite_x4_train",
            "--data", str(patches), "--config", str(config),
            "--output", str(output), "--variant", variant,
        ]
        if variant in {"distilled", "full"}:
            command.extend(["--teacher-checkpoint", str(teacher)])
        run(command, environment)
        (output / "last.pth").unlink(missing_ok=True)
        checkpoints.extend(["--checkpoint", f"{variant}={output / 'best.pth'}"])

    fixture = WORK / "benchmark-input-512.tiff"
    make_fixture(patches, fixture)
    run([
        sys.executable, "-m", "faithsr.lite_x4_evaluate",
        *checkpoints,
        "--data", str(patches), "--output", str(WORK / "evaluation"),
        "--seed", "0", "--train-ratio", "0.5", "--crop-size", "192", "--lpips",
        "--precisions", "fp32", "fp16", "--warmup", "10", "--runs", "100",
        "--tiff-input", str(fixture), "--pipeline-runs", "3", "--compression", "none",
    ], environment)
