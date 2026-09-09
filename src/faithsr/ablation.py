from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .compare import paired_cluster_intervals


def paired_variant_intervals(variants: dict[str, str | Path], sequence: list[str] | None = None) -> dict:
    """Compare frozen variants on identical per-image rows and geographic groups."""

    rows_by_variant: dict[str, list[dict]] = {}
    for name, path in variants.items():
        with Path(path).open(encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["method"] == "faithsr3x"]
        for row in rows:
            row["method"] = name
            row["sample_count"] = int(row["sample_count"])
            for metric in ("psnr", "ssim", "false_edge_rate", "edge_precision", "edge_recall", "sam_radians", "ergas"):
                row[metric] = float(row[metric])
            for metric in ("uncertainty_correlation", "uncertainty_calibration_error"):
                row[metric] = None if row[metric] in {"", None} else float(row[metric])
        rows_by_variant[name] = rows
    sequence = sequence or list(variants)
    output: dict[str, dict] = {}
    for reference, candidate in zip(sequence, sequence[1:]):
        combined = rows_by_variant[reference] + rows_by_variant[candidate]
        key = f"{candidate}_vs_{reference}"
        output[key] = paired_cluster_intervals(combined, reference)[key]
    return {
        "sequence": sequence,
        "interpretation": "candidate minus previous variant; 95% normal CI over geographic-cluster mean paired image deltas",
        "comparisons": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, help="name=per-image-comparison.csv")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    variants = dict(item.split("=", 1) for item in arguments.variant)
    result = paired_variant_intervals(variants)
    target = Path(arguments.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
