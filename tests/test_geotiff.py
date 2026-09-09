from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from faithsr.geotiff import capture_metadata, write_raster, write_uncertainty_raster


def _fixture(path: Path) -> None:
    profile = {"driver": "GTiff", "width": 11, "height": 7, "count": 3, "dtype": "uint8", "crs": "EPSG:32643", "transform": from_origin(500000, 3000000, 10, 10), "nodata": 0}
    with rasterio.open(path, "w", **profile) as output:
        output.write(np.arange(3 * 7 * 11, dtype=np.uint8).reshape(3, 7, 11))
        output.update_tags(SENSOR="TEST", ACQUISITION="2026-09-08")
        output.update_tags(1, BAND_NAME="red")
        output.descriptions = ("red", "green", "blue")
        mask = np.full((7, 11), 255, np.uint8); mask[0, 0] = 0; output.write_mask(mask)


def test_metadata_and_mask_survive_3x_round_trip(tmp_path):
    source_path, output_path = tmp_path / "input.tiff", tmp_path / "output.tiff"
    _fixture(source_path)
    with rasterio.open(source_path) as source:
        metadata = capture_metadata(source); original_bounds = source.bounds
    audit = write_raster(output_path, np.full((3, 21, 33), 128, np.float32), metadata)
    with rasterio.open(output_path) as output:
        assert (output.width, output.height) == (33, 21)
        assert output.dtypes == ("uint8",) * 3
        assert output.crs.to_epsg() == 32643
        assert output.bounds == original_bounds
        assert abs(output.transform.a - 10 / 3) < 1e-9
        assert abs(output.transform.e + 10 / 3) < 1e-9
        assert output.tags()["SENSOR"] == "TEST"
        assert output.tags(1)["BAND_NAME"] == "red"
        assert output.descriptions == ("red", "green", "blue")
        assert output.dataset_mask()[0, 0] == 0
    assert audit["metadata"]["transform_recalculated"] is True


def test_non_uint8_input_rejected(tmp_path):
    path = tmp_path / "uint16.tiff"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1, dtype="uint16") as output:
        output.write(np.ones((1, 2, 2), np.uint16))
    with rasterio.open(path) as source:
        try:
            capture_metadata(source)
        except ValueError as error:
            assert "8-bit" in str(error)
        else:
            raise AssertionError("uint16 input was accepted")


def test_uncertainty_geotiff_has_matching_grid(tmp_path):
    source_path, uncertainty_path = tmp_path / "input.tiff", tmp_path / "risk.tiff"
    _fixture(source_path)
    with rasterio.open(source_path) as source:
        metadata = capture_metadata(source)
    write_uncertainty_raster(uncertainty_path, np.full((1, 21, 33), 0.25, np.float32), metadata)
    with rasterio.open(uncertainty_path) as output:
        assert (output.width, output.height, output.count) == (33, 21, 1)
        assert output.dtypes == ("float32",)
        assert output.tags()["FAITHSR_PRODUCT"] == "uncertainty_risk"
        assert output.bounds == rasterio.open(source_path).bounds
