"""Create deterministic held-out visual panels for preserved training runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from faithsr.checkpoint import load_checkpoint
from faithsr.compare import validation_files
from faithsr.data import SyntheticSRDataset
from faithsr.qualitative import save_qualitative_panel


ROOT = Path(__file__).resolve().parents[1]


def render(run_dir: Path, data: Path, crop_size: int = 120, seed: int = 42) -> dict:
    files = validation_files(str(data), str(run_dir / "split.json"))
    dataset = SyntheticSRDataset(files, crop_size=crop_size, bands=3, train=False, seed=seed)
    lr, hr, _, source, degradation = dataset[0]
    model, state = load_checkpoint(run_dir / "best.pth", "cpu")
    model.eval()
    with torch.inference_mode():
        sr, auxiliary = model(lr.unsqueeze(0), return_aux=True)
    return save_qualitative_panel(
        run_dir / "best-preview.png",
        lr,
        hr,
        sr[0].clamp(0, 1),
        None if auxiliary["uncertainty"] is None else auxiliary["uncertainty"][0],
        {
            "epoch": state.get("epoch"),
            "source": source,
            "degradation": json.loads(degradation),
            "selection_rule": "first deterministic held-out geographic validation sample",
            "checkpoint": str((run_dir / "best.pth").resolve()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Local directory containing the Sentinel RGB tiles")
    args = parser.parse_args()
    base = ROOT / "artifacts" / "training_runs" / "faithsr3x" / "sentinel_delhi_ablation_v1_partial"
    runs = [base / name for name in ("plain", "degradation", "observation", "full")]
    runs.append(
        ROOT
        / "artifacts"
        / "training_runs"
        / "faithsr3x"
        / "sentinel_distillation_v3"
        / "faithsr3x-sentinel-distillation"
        / "distilled"
    )
    results = {run.name: render(run, Path(args.data)) for run in runs}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
