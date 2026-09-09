"""Per-image, geographic-cluster-robust evaluation of frozen Sentinel v1 checkpoints."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/faithsr3x-sentinel-exact-evaluation")


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
    rgb_root = next((path for path in INPUT_ROOT.rglob("rgb") if any(path.glob("*.png"))), None)
    if rgb_root is None:
        raise RuntimeError("Sentinel RGB input was not mounted")
    report = {"dataset": str(rgb_root), "aggregation": "per-image with paired 95% CIs over 0.1-degree geographic groups", "variants": {}}
    comparison_paths = {}
    for name in ("plain", "degradation", "observation", "full"):
        variant = WORK / "frozen-v1" / name
        variant.mkdir(parents=True, exist_ok=True)
        mounted_checkpoint = next((path for path in INPUT_ROOT.rglob("best.pth") if path.parent.name == name), None)
        if mounted_checkpoint is not None:
            checkpoint = mounted_checkpoint
            split = next(mounted_checkpoint.parent.rglob("split.json"))
        else:
            archive = next(INPUT_ROOT.rglob(f"{name}.zip"), None)
            if archive is None:
                raise RuntimeError(f"Missing mounted or archived {name} artifacts")
            shutil.unpack_archive(archive, variant)
            checkpoint = next(variant.rglob("best.pth"))
            split = next(variant.rglob("split.json"))
        output = WORK / name / "comparison.csv"
        run([
            sys.executable, "-m", "faithsr.compare", "--data", str(rgb_root), "--checkpoint", str(checkpoint),
            "--split-manifest", str(split), "--crop-size", "120", "--batch-size", "32", "--workers", "2",
            "--output", str(output),
        ], environment)
        report["variants"][name] = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
        comparison_paths[name] = output
    from faithsr.ablation import paired_variant_intervals
    report["paired_component_ablation"] = paired_variant_intervals(comparison_paths)
    (WORK / "exact_ablation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
