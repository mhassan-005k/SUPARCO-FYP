"""Render best/median/worst held-out Sentinel examples from exact per-image results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from faithsr.checkpoint import load_checkpoint
from faithsr.data import read_image
from faithsr.degradation import DegradationParameters, degrade
from faithsr.metrics import reconstruction_metrics_per_sample
from faithsr.qualitative import save_qualitative_panel


def resolve_source(recorded: str, data: Path) -> Path:
    name = Path(recorded).name
    matches = list(data.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely resolve {name} below {data}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-size", type=int, default=120)
    args = parser.parse_args()

    with Path(args.comparison).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {(row["method"], row["path"]): row for row in rows}
    candidates = []
    for row in rows:
        if row["method"] != "faithsr3x":
            continue
        bicubic = by_method[("bicubic", row["path"])]
        candidates.append((float(row["psnr"]) - float(bicubic["psnr"]), row, bicubic))
    candidates.sort(key=lambda item: item[0])
    selected = {
        "worst_gain": candidates[0],
        "median_gain": candidates[len(candidates) // 2],
        "best_gain": candidates[-1],
    }

    model, state = load_checkpoint(args.checkpoint, "cpu")
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for label, (delta, row, bicubic_row) in selected.items():
        source = resolve_source(row["path"], Path(args.data))
        image = read_image(source, 3)
        crop = args.crop_size
        top, left = (image.shape[-2] - crop) // 2, (image.shape[-1] - crop) // 2
        hr = image[:, top:top + crop, left:left + crop]
        details = json.loads(row["degradation"])
        seed = int(details.pop("seed"))
        accepted = set(DegradationParameters.__dataclass_fields__)
        parameters = DegradationParameters(**{key: value for key, value in details.items() if key in accepted})
        lr, _ = degrade(hr, params=parameters, seed=seed)
        with torch.inference_mode():
            sr, auxiliary = model(lr.unsqueeze(0), return_aux=True)
        bicubic = F.interpolate(lr.unsqueeze(0), scale_factor=3, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        bicubic_metrics = {name: float(value[0]) for name, value in reconstruction_metrics_per_sample(bicubic, hr.unsqueeze(0)).items()}
        artifact = output / f"{label}.png"
        payload = save_qualitative_panel(
            artifact,
            lr,
            hr,
            sr[0].clamp(0, 1),
            None if auxiliary["uncertainty"] is None else auxiliary["uncertainty"][0],
            {
                "case": label,
                "source": str(source.resolve()),
                "checkpoint_epoch": state.get("epoch"),
                "degradation": {"seed": seed, **details},
                "faithsr_minus_bicubic_psnr_db": delta,
                "bicubic_metrics": bicubic_metrics,
            },
        )
        summaries.append(payload)

    lines = [
        "# Sentinel held-out qualitative cases",
        "",
        "Cases were selected mechanically from all 1,728 held-out per-image rows by FaithSR-minus-bicubic PSNR: minimum, median, and maximum. No image was hand-picked for appearance.",
        "",
        "| Case | FaithSR PSNR | Bicubic PSNR | Delta | Figure |",
        "|---|---:|---:|---:|---|",
    ]
    for payload in summaries:
        name = Path(payload["artifact"]).name
        lines.append(
            f'| {payload["case"]} | {payload["metrics"]["psnr"]:.4f} | {payload["bicubic_metrics"]["psnr"]:.4f} | {payload["faithsr_minus_bicubic_psnr_db"]:+.4f} | [{name}]({name}) |'
        )
    lines.extend(["", "Each PNG shows observed LR, bicubic, FaithSR, held-out HR, absolute error, and uncertainty. Its same-name JSON sidecar contains source path, exact degradation parameters/seed, checkpoint epoch, and per-image metrics."])
    (output / "failure-cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
