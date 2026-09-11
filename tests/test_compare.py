import json

import pytest
import torch

from faithsr.compare import _score_rows, degradation_protocol, interpolate, paired_cluster_intervals, validation_files


def test_all_interpolation_baselines_are_native_3x():
    image = torch.rand(1, 3, 7, 9)
    for mode in ("nearest", "bilinear", "bicubic"):
        assert interpolate(image, mode).shape == (1, 3, 21, 27)


def test_validation_manifest_paths_can_be_relocated(tmp_path):
    data = tmp_path / "EuroSAT" / "Forest"
    data.mkdir(parents=True)
    image = data / "Forest_1.jpg"
    image.write_bytes(b"fixture")
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"validation": ["/kaggle/input/old/EuroSAT/Forest/Forest_1.jpg"]}), encoding="utf-8")
    assert validation_files(str(tmp_path / "EuroSAT"), str(manifest)) == [image]


def test_paired_intervals_use_geographic_cluster_means():
    rows = []
    for path, group, delta in (("a", "g1", 1.0), ("b", "g1", 3.0), ("c", "g2", 5.0)):
        base = {"path": path, "geographic_group": group, "psnr": 10.0, "ssim": 0.5, "false_edge_rate": 0.2, "sam_radians": 0.1, "ergas": 4.0}
        rows.append({"method": "bicubic", **base})
        rows.append({"method": "candidate", **base, "psnr": 10.0 + delta})
    result = paired_cluster_intervals(rows)["candidate_vs_bicubic"]["psnr"]
    assert result["clusters"] == 2
    assert result["paired_images"] == 3
    assert result["mean_delta"] == 3.5
    assert result["ci95_low"] < result["mean_delta"] < result["ci95_high"]


def test_score_rows_records_optional_per_image_lpips():
    prediction = torch.zeros(2, 3, 12, 12)
    target = torch.ones_like(prediction) * 0.1
    rows = _score_rows(
        "candidate",
        prediction,
        target,
        ["rgb/28.1_77.1.png", "rgb/28.2_77.2.png"],
        ["{}", "{}"],
        lpips_values=torch.tensor([0.12, 0.34]),
    )
    assert [row["lpips"] for row in rows] == pytest.approx([0.12, 0.34])


def test_compare_replays_degradation_protocol_from_exact_config():
    crop_size, options, record = degradation_protocol("configs/sentinel_l2a_native_kaggle.yaml", 48)
    assert crop_size == 192
    assert options == {"seed": 0, "mtf_probability": 0.5, "complex_probability": 0.75}
    assert record["source"].endswith("configs\\sentinel_l2a_native_kaggle.yaml")
