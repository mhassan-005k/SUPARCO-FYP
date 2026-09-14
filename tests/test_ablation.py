import csv

import pytest

from faithsr.ablation import paired_variant_intervals


FIELDS = ["method", "path", "geographic_group", "degradation", "sample_count", "psnr", "ssim", "false_edge_rate", "edge_precision", "edge_recall", "sam_radians", "ergas", "lpips", "uncertainty_correlation", "uncertainty_calibration_error", "degradation_profile", "observation_l1"]


def _write(path, offset, method="faithsr3x"):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        for name, group in (("a", "g1"), ("b", "g2")):
            writer.writerow({"method": method, "path": name, "geographic_group": group, "degradation": "fixed", "sample_count": 1, "psnr": 20 + offset, "ssim": .5, "false_edge_rate": .2, "edge_precision": .8, "edge_recall": .7, "sam_radians": .1, "ergas": 5, "lpips": .3 - offset / 10, "degradation_profile": "compression_heavy", "observation_l1": .05 - offset / 100})
            writer.writerow({"method": "bicubic_x4", "path": name, "geographic_group": group, "degradation": "fixed", "sample_count": 1, "psnr": 19, "ssim": .4, "false_edge_rate": .3, "edge_precision": .7, "edge_recall": .6, "sam_radians": .2, "ergas": 6, "lpips": .4, "degradation_profile": "compression_heavy", "observation_l1": .06})


def test_adjacent_variant_intervals_are_paired(tmp_path):
    plain, adaptive = tmp_path / "plain.csv", tmp_path / "adaptive.csv"
    _write(plain, 0); _write(adaptive, 1)
    result = paired_variant_intervals({"plain": plain, "adaptive": adaptive})
    psnr = result["comparisons"]["adaptive_vs_plain"]["psnr"]
    assert psnr["mean_delta"] == 1
    assert psnr["paired_images"] == 2


def test_registered_pairs_accept_native_x4_labels_and_optional_metrics(tmp_path):
    base, distilled, compression, full = [tmp_path / f"{name}.csv" for name in ("base", "distilled", "compression", "full")]
    _write(base, 0, "faithsr_lite_x4_base")
    _write(distilled, 1, "faithsr_lite_x4_distilled")
    _write(compression, 2, "faithsr_lite_x4_compression")
    _write(full, 3, "faithsr_lite_x4_full")
    result = paired_variant_intervals(
        {"base": base, "distilled": distilled, "compression": compression, "full": full},
        pairs=[("base", "distilled"), ("compression", "full")],
    )
    assert result["comparisons"]["distilled_vs_base"]["lpips"]["mean_delta"] == pytest.approx(-0.1)
    assert result["comparisons"]["full_vs_compression"]["observation_l1"]["mean_delta"] == pytest.approx(-0.01)
    assert result["profile_comparisons"]["full_vs_compression"]["compression_heavy"]["psnr"]["paired_images"] == 2
