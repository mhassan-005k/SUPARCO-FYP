"""Geographically split Sentinel-2 RGB ablations and pretrained comparison."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/faithsr3x-sentinel-ablation")
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


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    run([sys.executable, "-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu121", "--quiet"], environment)
    run([sys.executable, "-m", "pip", "install", "rasterio", "PyYAML", "transformers==4.47.1", "--quiet"], environment)
    import yaml

    source = source_root()
    environment["PYTHONPATH"] = str(source / "src")
    rgb_root = next((path for path in INPUT_ROOT.rglob("rgb") if any(path.glob("*.png"))), None)
    config_path = source / "configs" / "sentinel_delhi_kaggle.yaml"
    if rgb_root is None or not config_path.exists():
        raise RuntimeError(f"RGB root/config missing: rgb={rgb_root}, config={config_path}")
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variants = {
        "plain": {"model": {"use_degradation_adaptation": False, "use_observation_consistency": False, "use_uncertainty": False}},
        "degradation": {"model": {"use_degradation_adaptation": True, "use_observation_consistency": False, "use_uncertainty": False}},
        "observation": {"model": {"use_degradation_adaptation": True, "use_observation_consistency": True, "use_uncertainty": False}},
        "full": {"model": {"use_degradation_adaptation": True, "use_observation_consistency": True, "use_uncertainty": True}},
        "distilled": {"model": {"use_degradation_adaptation": True, "use_observation_consistency": True, "use_uncertainty": True}, "loss": {"distillation": 0.05}, "training": {"teacher_batch_size": 2, "teacher_max_epoch": 3}},
    }
    lineage = {
        "plain": ("Establish the minimum custom native-3x backbone control.", "EuroSAT v8 established convergence but did not isolate components."),
        "degradation": ("Isolate the value of input-estimated degradation conditioning.", "Plain Sentinel control; only encoder and FiLM are added."),
        "observation": ("Test whether the learned observation operator fixes the v0.1 fixed-bicubic consistency regression.", "Degradation variant; learned re-observation and bounded correction are added."),
        "full": ("Measure the fidelity cost/benefit of adding spatial uncertainty.", "Observation variant; only uncertainty head and calibration loss are added."),
        "distilled": ("Test personalized transfer from a pinned pretrained Swin2SR teacher into the custom deployable student.", "Full variant; architecture is unchanged and only distillation supervision is added."),
    }
    report = {"dataset": str(rgb_root), "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION}, "variants": {}}
    for name, overrides in variants.items():
        config = deepcopy(base)
        for section, values in overrides.items():
            config[section].update(values)
        variant_root = WORK / name
        variant_config = WORK / f"{name}.yaml"
        variant_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [sys.executable, "-m", "faithsr.train", "--data", str(rgb_root), "--config", str(variant_config), "--output", str(variant_root)]
        command.extend(["--experiment-rationale", lineage[name][0], "--previous-experiment", lineage[name][1]])
        if name == "distilled":
            command.extend(["--teacher-model", TEACHER_ID, "--teacher-revision", TEACHER_REVISION])
        run(command, environment)
        comparison = variant_root / "comparison.csv"
        run([
            sys.executable, "-m", "faithsr.compare", "--data", str(rgb_root), "--checkpoint", str(variant_root / "best.pth"),
            "--split-manifest", str(variant_root / "split.json"), "--crop-size", "120", "--batch-size", "32", "--workers", "2", "--output", str(comparison),
        ], environment)
        rows = list(csv.DictReader((variant_root / "metrics.csv").open(encoding="utf-8")))
        report["variants"][name] = {
            "best_training_row": max(rows, key=lambda row: float(row["psnr"])),
            "evaluation": json.loads(comparison.with_suffix(".summary.json").read_text(encoding="utf-8")),
        }
    full_root = WORK / "full"
    teacher_output = WORK / "pretrained_comparison.csv"
    run([
        sys.executable, "-m", "faithsr.compare", "--data", str(rgb_root), "--checkpoint", str(full_root / "best.pth"),
        "--split-manifest", str(full_root / "split.json"), "--teacher-model", TEACHER_ID, "--teacher-revision", TEACHER_REVISION,
        "--crop-size", "120", "--batch-size", "2", "--workers", "2", "--output", str(teacher_output),
    ], environment)
    report["pretrained_comparison"] = json.loads(teacher_output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    (WORK / "ablation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
