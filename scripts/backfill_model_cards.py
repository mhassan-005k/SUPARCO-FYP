"""Regenerate model cards for historical FaithSR-3X runs from saved configs."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from faithsr.documentation import append_run_results, write_model_card
from faithsr.model import FaithSR3X, FaithSRConfig


ROOT = Path(__file__).resolve().parents[1]
SENTINEL = ROOT / "artifacts" / "training_runs" / "faithsr3x" / "sentinel_delhi_ablation_v1_partial"

LINEAGE = {
    "plain": (
        "Establish the minimum custom native-3x backbone control.",
        "EuroSAT v8 established convergence but did not isolate components.",
    ),
    "degradation": (
        "Isolate the value of input-estimated degradation conditioning.",
        "Plain Sentinel control; only encoder and FiLM are added.",
    ),
    "observation": (
        "Test whether a learned observation operator addresses the v0.1 fixed-bicubic consistency regression.",
        "Degradation variant; learned re-observation and bounded correction are added.",
    ),
    "full": (
        "Measure the fidelity cost/benefit of adding spatial uncertainty.",
        "Observation variant; only uncertainty head and calibration loss are added.",
    ),
}


def instantiate(config: dict) -> FaithSR3X:
    return FaithSR3X(FaithSRConfig(in_channels=3, out_channels=3, **config["model"]))


def main() -> None:
    eurosat = ROOT / "artifacts" / "training_runs" / "faithsr3x" / "eurosat_preliminary_v8"
    run = json.loads((eurosat / "run.json").read_text(encoding="utf-8"))
    write_model_card(
        eurosat / "model-card.md",
        instantiate(run["config"]),
        run["config"],
        "eurosat-preliminary-v8",
        "Verify native-3x convergence, learned observation consistency, and uncertainty on affordable public 8-bit RGB data.",
        "FaithSR v0.1 x4 experiments; native scale, richer degradation, learned observation, and uncertainty are new.",
    )
    append_run_results(eurosat / "model-card.md", eurosat / "metrics.csv", run.get("duration_seconds"))
    for name, (rationale, previous) in LINEAGE.items():
        config = yaml.safe_load((SENTINEL / f"{name}.yaml").read_text(encoding="utf-8"))
        write_model_card(SENTINEL / name / "model-card.md", instantiate(config), config, f"sentinel-delhi-{name}-v1", rationale, previous)
        append_run_results(SENTINEL / name / "model-card.md", SENTINEL / name / "metrics.csv")

    distilled = ROOT / "artifacts" / "training_runs" / "faithsr3x" / "sentinel_distillation_v3" / "faithsr3x-sentinel-distillation" / "distilled"
    if (distilled / "run.json").exists():
        run = json.loads((distilled / "run.json").read_text(encoding="utf-8"))
        write_model_card(
            distilled / "model-card.md",
            instantiate(run["config"]),
            run["config"],
            "sentinel-distillation-v3",
            "Transfer a pinned pretrained Swin2SR prior for three epochs, then personalize the compact native-3x student to Sentinel RGB evidence.",
            "Full Sentinel v1 student; deployed architecture is unchanged and only a temporary frozen-teacher loss is added.",
        )
        append_run_results(distilled / "model-card.md", distilled / "metrics.csv")


if __name__ == "__main__":
    main()
