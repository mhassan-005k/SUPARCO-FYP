import json

import torch

from faithsr.compare import interpolate, validation_files


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
