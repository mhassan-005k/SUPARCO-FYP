"""Complete the pinned Swin2SR personalization experiment after the v1 ABI failure."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/faithsr3x-sentinel-distillation")
TEACHER_ID = "caidas/swin2SR-classical-sr-x4-64"
TEACHER_REVISION = "c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e"


def run(command: list[str], environment: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def source_root() -> Path:
    mount = next((path for path in INPUT_ROOT.rglob("faithsr-3x-private-source") if path.is_dir()), None)
    if mount is None:
        raise RuntimeError("Private source dataset was not mounted")
    if (mount / "src" / "faithsr").exists():
        return mount
    source_zip = next(mount.rglob("src.zip"), None)
    configs_zip = next(mount.rglob("configs.zip"), None)
    if source_zip is None or configs_zip is None:
        raise RuntimeError("Source/config archives were not found")
    root = WORK / "source"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(source_zip, root / "src")
    shutil.unpack_archive(configs_zip, root / "configs")
    return root


def materialize_previous() -> Path:
    """Unpack Kaggle's dir-mode zip artifacts into a stable local tree."""
    root = WORK / "previous-v1"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("plain", "degradation", "observation", "full"):
        destination = root / name
        mounted_checkpoint = next((path for path in INPUT_ROOT.rglob("best.pth") if path.parent.name == name), None)
        if mounted_checkpoint is not None:
            shutil.copytree(mounted_checkpoint.parent, destination, dirs_exist_ok=True)
            continue
        archive = next(INPUT_ROOT.rglob(f"{name}.zip"), None)
        if archive is None:
            raise RuntimeError(f"Previous {name}.zip artifact was not mounted")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(archive, destination)
    return root


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Kaggle changed its preinstalled Torch stack; pinning both packages prevents
    # the v1 torchvision::nms ABI mismatch and is part of the run provenance.
    run([
        sys.executable, "-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet",
    ], environment)
    run([sys.executable, "-m", "pip", "install", "rasterio", "PyYAML", "transformers==4.47.1", "--quiet"], environment)
    import yaml

    source = source_root()
    previous_root = materialize_previous()
    environment["PYTHONPATH"] = str(source / "src")
    rgb_root = next((path for path in INPUT_ROOT.rglob("rgb") if any(path.glob("*.png"))), None)
    config_path = source / "configs" / "sentinel_delhi_kaggle.yaml"
    previous_full_checkpoint = next((previous_root / "full").rglob("best.pth"), None)
    previous_full_split = next((previous_root / "full").rglob("split.json"), None)
    if rgb_root is None or not config_path.exists() or previous_full_checkpoint is None or previous_full_split is None:
        raise RuntimeError(
            f"Missing input: rgb={rgb_root}, config={config_path}, full={previous_full_checkpoint}, split={previous_full_split}"
        )

    config = deepcopy(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config["loss"]["distillation"] = 0.05
    config["training"]["teacher_batch_size"] = 2
    config["training"]["teacher_max_epoch"] = 3
    distilled_config = WORK / "distilled.yaml"
    distilled_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    distilled_root = WORK / "distilled"
    run([
        sys.executable, "-m", "faithsr.train", "--data", str(rgb_root), "--config", str(distilled_config),
        "--output", str(distilled_root), "--teacher-model", TEACHER_ID, "--teacher-revision", TEACHER_REVISION,
        "--experiment-rationale", "Transfer a pinned pretrained Swin2SR prior for three warm-start epochs, then personalize the custom student to Sentinel RGB evidence.",
        "--previous-experiment", "Full Sentinel v1 student; same deployed architecture, with a temporary frozen teacher loss added.",
    ], environment)
    distilled_comparison = distilled_root / "comparison.csv"
    run([
        sys.executable, "-m", "faithsr.compare", "--data", str(rgb_root), "--checkpoint", str(distilled_root / "best.pth"),
        "--split-manifest", str(distilled_root / "split.json"), "--crop-size", "120", "--batch-size", "32",
        "--workers", "2", "--output", str(distilled_comparison),
    ], environment)
    pretrained_comparison = WORK / "pretrained_comparison.csv"
    run([
        sys.executable, "-m", "faithsr.compare", "--data", str(rgb_root), "--checkpoint", str(previous_full_checkpoint),
        "--split-manifest", str(previous_full_split), "--teacher-model", TEACHER_ID, "--teacher-revision", TEACHER_REVISION,
        "--crop-size", "120", "--batch-size", "2", "--workers", "2", "--output", str(pretrained_comparison),
    ], environment)
    report = {
        "status": "complete",
        "dataset": str(rgb_root),
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION, "active_epochs": 3, "microbatch": 2},
        "previous_v1_variants": {
            name: json.loads(next((previous_root / name).rglob("comparison.summary.json")).read_text(encoding="utf-8"))
            for name in ("plain", "degradation", "observation", "full")
        },
        "distilled": json.loads(distilled_comparison.with_suffix(".summary.json").read_text(encoding="utf-8")),
        "pretrained_comparison": json.loads(pretrained_comparison.with_suffix(".summary.json").read_text(encoding="utf-8")),
    }
    (WORK / "personalization_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
