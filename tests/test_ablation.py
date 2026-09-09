import csv

from faithsr.ablation import paired_variant_intervals


FIELDS = ["method", "path", "geographic_group", "degradation", "sample_count", "psnr", "ssim", "false_edge_rate", "edge_precision", "edge_recall", "sam_radians", "ergas", "uncertainty_correlation", "uncertainty_calibration_error"]


def _write(path, offset):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        for name, group in (("a", "g1"), ("b", "g2")):
            writer.writerow({"method": "faithsr3x", "path": name, "geographic_group": group, "degradation": "fixed", "sample_count": 1, "psnr": 20 + offset, "ssim": .5, "false_edge_rate": .2, "edge_precision": .8, "edge_recall": .7, "sam_radians": .1, "ergas": 5})


def test_adjacent_variant_intervals_are_paired(tmp_path):
    plain, adaptive = tmp_path / "plain.csv", tmp_path / "adaptive.csv"
    _write(plain, 0); _write(adaptive, 1)
    result = paired_variant_intervals({"plain": plain, "adaptive": adaptive})
    psnr = result["comparisons"]["adaptive_vs_plain"]["psnr"]
    assert psnr["mean_delta"] == 1
    assert psnr["paired_images"] == 2
