import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from faithsr.sentinel import prepare_rgb8


def test_sentinel_rgb8_derivation_is_documented_and_georeferenced(tmp_path):
    inputs = []
    for index, value in enumerate((3000, 1500, 0)):
        path = tmp_path / f"band_{index}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=4, height=3, count=1, dtype="uint16", crs="EPSG:32643", transform=from_origin(10, 20, 10, 10)) as dataset:
            dataset.write(np.full((1, 3, 4), value, dtype=np.uint16))
        inputs.append(path)
    output = tmp_path / "rgb8.tiff"
    prepare_rgb8(str(inputs[0]), str(inputs[1]), str(inputs[2]), str(output))
    with rasterio.open(output) as dataset:
        assert dataset.dtypes == ("uint8",) * 3
        assert dataset.descriptions == ("B04_red", "B03_green", "B02_blue")
        assert dataset.crs.to_epsg() == 32643
        assert dataset.read()[:, 0, 0].tolist() == [255, 128, 0]
        assert dataset.tags()["FAITHSR_DERIVED_PRODUCT"] == "Sentinel-2_L2A_RGB_uint8"
    manifest = json.loads(output.with_suffix(".tiff.derivation.json").read_text())
    assert manifest["band_order"] == ["B04", "B03", "B02"]
    assert len(manifest["sources"]) == 3


def test_sentinel_alignment_mismatch_is_rejected(tmp_path):
    paths = []
    for index, width in enumerate((4, 4, 5)):
        path = tmp_path / f"band_{index}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=width, height=3, count=1, dtype="uint16", transform=from_origin(0, 0, 10, 10)) as dataset:
            dataset.write(np.zeros((1, 3, width), dtype=np.uint16))
        paths.append(path)
    try:
        prepare_rgb8(*(str(path) for path in paths), str(tmp_path / "out.tif"))
    except ValueError as error:
        assert "share dimensions" in str(error)
    else:
        raise AssertionError("Misaligned bands were accepted")
