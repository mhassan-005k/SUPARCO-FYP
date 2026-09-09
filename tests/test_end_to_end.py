import json

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from torch import nn
import torch.nn.functional as F

from faithsr.inference import process_tiff, tiled_super_resolve


class BicubicModel(nn.Module):
    def forward(self, image, return_aux=False):
        sr = F.interpolate(image, scale_factor=3, mode="bicubic", align_corners=False)
        return (sr, {"uncertainty": torch.full_like(sr[:, :1], 0.1)}) if return_aux else sr


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
