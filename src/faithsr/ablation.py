from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .compare import paired_cluster_intervals


NUMERIC_METRICS = (
    "psnr",
    "ssim",
    "false_edge_rate",
    "edge_precision",
    "edge_recall",
    "sam_radians",
    "ergas",
    "lpips",
    "observation_l1",
    "uncertainty_correlation",
    "uncertainty_calibration_error",
)


def _read_variant_rows(name: str, path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    model_methods = sorted({row["method"] for row in raw_rows if not row["method"].startswith("bicubic")})
    if len(model_methods) != 1:
        raise ValueError(f"Expected one non-bicubic method in {path}; found {model_methods}")
    rows = [row for row in raw_rows if row["method"] == model_methods[0]]
    for row in rows:
        row["method"] = name
        row["sample_count"] = int(row["sample_count"])
        for metric in NUMERIC_METRICS:
            if metric in row:
                row[metric] = None if row[metric] in {"", None} else float(row[metric])
    return rows


def paired_variant_intervals(
    variants: dict[str, str | Path],
    sequence: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
) -> dict:
    """Compare frozen variants on identical per-image rows and geographic groups."""

    rows_by_variant = {name: _read_variant_rows(name, path) for name, path in variants.items()}
    sequence = sequence or list(variants)
    pairs = pairs or list(zip(sequence, sequence[1:]))
    output: dict[str, dict] = {}
    profile_output: dict[str, dict] = {}
    for reference, candidate in pairs:
        if reference not in rows_by_variant or candidate not in rows_by_variant:
            raise ValueError(f"Unknown comparison pair {candidate} vs {reference}")
        combined = rows_by_variant[reference] + rows_by_variant[candidate]
        key = f"{candidate}_vs_{reference}"
        output[key] = paired_cluster_intervals(combined, reference)[key]
        profiles = sorted({row.get("degradation_profile") for row in combined if row.get("degradation_profile")})
        profile_output[key] = {
            profile: paired_cluster_intervals(
                [row for row in combined if row.get("degradation_profile") == profile], reference
            )[key]
            for profile in profiles
        }
    return {
        "sequence": sequence,
        "pairs": [{"reference": reference, "candidate": candidate} for reference, candidate in pairs],
        "interpretation": "candidate minus named reference; 95% normal CI over geographic-cluster mean paired image deltas",
        "comparisons": output,
        "profile_comparisons": profile_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, help="name=per-image-comparison.csv")
    parser.add_argument("--pair", action="append", help="candidate=reference; repeat for pre-registered contrasts")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    variants = dict(item.split("=", 1) for item in arguments.variant)
    pairs = None if not arguments.pair else [tuple(reversed(item.split("=", 1))) for item in arguments.pair]
    result = paired_variant_intervals(variants, pairs=pairs)
    target = Path(arguments.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
