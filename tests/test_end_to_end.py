import json

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from torch import nn
import torch.nn.functional as F
import pytest

from faithsr.inference import process_tiff, tiled_super_resolve


class BicubicModel(nn.Module):
    def forward(self, image, return_aux=False):
        sr = F.interpolate(image, scale_factor=3, mode="bicubic", align_corners=False)
        return (sr, {"uncertainty": torch.full_like(sr[:, :1], 0.1)}) if return_aux else sr


class BicubicX4Model(nn.Module):
    scale = 4

    def forward(self, image, return_aux=False):
        sr = F.interpolate(image, scale_factor=4, mode="bicubic", align_corners=False)
        return (sr, {"uncertainty": None}) if return_aux else sr


def test_complete_tiff_read_infer_write_audit(tmp_path):
    source, target = tmp_path / "source.tiff", tmp_path / "result.tiff"
    with rasterio.open(source, "w", driver="GTiff", width=13, height=9, count=3, dtype="uint8", crs="EPSG:4326", transform=from_origin(70, 35, 0.01, 0.01)) as output:
        output.write(np.arange(3 * 9 * 13, dtype=np.uint8).reshape(3, 9, 13))
        output.update_tags(PRODUCT="fixture")
    audit = process_tiff(source, target, BicubicModel(), tile_size=8, overlap=2, model_version="test")
    assert target.exists()
    assert target.with_suffix(".tiff.audit.json").exists()
    assert audit["output_shape"] == [3, 27, 39]
    assert audit["model_version"] == "test"
    assert len(audit["input_sha256"]) == 64
    assert json.loads(target.with_suffix(".tiff.audit.json").read_text())["metadata"]["mask_preserved"]
    assert audit["streaming"]["disk_backed"] is True
    with rasterio.open(source) as dataset:
        expected, _ = tiled_super_resolve(dataset.read(), BicubicModel(), "cpu", tile_size=8, overlap=2)
    with rasterio.open(target) as dataset:
        actual = dataset.read()
    assert np.array_equal(actual, np.clip(np.rint(expected), 0, 255).astype(np.uint8))


def test_tiff_pipeline_reports_monotonic_tile_progress(tmp_path):
    source, target = tmp_path / "source.tiff", tmp_path / "result.tiff"
    with rasterio.open(source, "w", driver="GTiff", width=13, height=9, count=3, dtype="uint8") as output:
        output.write(np.zeros((3, 9, 13), dtype=np.uint8))
    events = []
    audit = process_tiff(
        source,
        target,
        BicubicModel(),
        tile_size=8,
        overlap=2,
        progress_callback=lambda completed, total: events.append((completed, total)),
    )
    assert events == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert audit["streaming"]["tiles_processed"] == 4


def test_complete_native_x4_tiff_pipeline_preserves_bounds(tmp_path):
    source, target = tmp_path / "source.tiff", tmp_path / "result_x4.tiff"
    with rasterio.open(source, "w", driver="GTiff", width=7, height=5, count=3, dtype="uint8", crs="EPSG:4326", transform=from_origin(70, 35, 0.01, 0.01)) as output:
        output.write(np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7))
    with rasterio.open(source) as dataset:
        original_bounds = dataset.bounds
    audit = process_tiff(source, target, BicubicX4Model(), tile_size=4, overlap=1, model_version="test-x4")
    assert audit["scale"] == 4
    assert audit["output_shape"] == [3, 20, 28]
    with rasterio.open(target) as output:
        assert (output.width, output.height) == (28, 20)
        assert output.bounds == original_bounds
        assert output.tags()["FAITHSR_SCALE"] == "4"


def test_corrupt_tiff_fails_without_creating_output(tmp_path):
    source, target = tmp_path / "corrupt.tiff", tmp_path / "result.tiff"
    source.write_bytes(b"not-a-tiff")
    with pytest.raises(rasterio.errors.RasterioIOError):
        process_tiff(source, target, BicubicModel())
    assert not target.exists()


def test_input_cannot_be_overwritten_and_invalid_tiling_is_rejected(tmp_path):
    source = tmp_path / "source.tiff"
    with rasterio.open(source, "w", driver="GTiff", width=3, height=3, count=3, dtype="uint8") as output:
        output.write(np.zeros((3, 3, 3), dtype=np.uint8))
    original = source.read_bytes()
    with pytest.raises(ValueError, match="must be different"):
        process_tiff(source, source, BicubicModel())
    with pytest.raises(ValueError, match="non-negative overlap"):
        process_tiff(source, tmp_path / "bad.tiff", BicubicModel(), tile_size=8, overlap=-1)
    assert source.read_bytes() == original


class FourBandModel(BicubicModel):
    class Config:
        in_channels = 4

    config = Config()


def test_checkpoint_band_mismatch_fails_before_output(tmp_path):
    source, target = tmp_path / "source.tiff", tmp_path / "result.tiff"
    with rasterio.open(source, "w", driver="GTiff", width=3, height=3, count=3, dtype="uint8") as output:
        output.write(np.zeros((3, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="checkpoint expects 4"):
        process_tiff(source, target, FourBandModel())
    assert not target.exists()
