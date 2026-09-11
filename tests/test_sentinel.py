import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from faithsr.sentinel import prepare_rgb8, prepare_stac_rgb8, tile_rgb8_scene


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


def test_sentinel_window_preserves_window_georeferencing_and_mask(tmp_path):
    inputs = []
    for index, value in enumerate((2000, 1500, 1000)):
        path = tmp_path / f"window_band_{index}.tif"
        data = np.full((1, 8, 9), value, dtype=np.uint16)
        data[:, 3, 4] = 0
        with rasterio.open(path, "w", driver="GTiff", width=9, height=8, count=1, dtype="uint16", nodata=0, crs="EPSG:32643", transform=from_origin(100, 200, 10, 10)) as dataset:
            dataset.write(data)
        inputs.append(path)
    output = tmp_path / "window_rgb8.tif"
    manifest = prepare_rgb8(*(str(path) for path in inputs), str(output), window=(2, 1, 4, 5))
    with rasterio.open(output) as dataset:
        assert (dataset.width, dataset.height) == (4, 5)
        assert dataset.transform == from_origin(120, 190, 10, 10)
        assert dataset.dataset_mask()[2, 2] == 0
    assert manifest["window"] == {"column": 2, "row": 1, "width": 4, "height": 5}


def test_stac_radiometry_and_item_identity_are_recorded(tmp_path):
    inputs = []
    for index, value in enumerate((1000, 2000, 3000)):
        path = tmp_path / f"stac_band_{index}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1, dtype="uint16", crs="EPSG:32643", transform=from_origin(0, 40, 10, 10)) as dataset:
            dataset.write(np.full((1, 4, 4), value, dtype=np.uint16))
        inputs.append(path)
    item = {
        "id": "S2_TEST_L2A", "collection": "sentinel-2-l2a", "bbox": [0, 0, 1, 1],
        "properties": {"datetime": "2026-01-01T00:00:00Z", "eo:cloud_cover": 1.2, "proj:epsg": 32643},
        "assets": {
            name: {"href": str(path), "raster:bands": [{"scale": 0.0001, "offset": 0.0}]}
            for name, path in zip(("red", "green", "blue"), inputs)
        },
    }
    manifest = prepare_stac_rgb8(item, str(tmp_path / "stac_rgb8.tif"))
    assert manifest["source_context"]["stac_item_id"] == "S2_TEST_L2A"
    assert manifest["source_context"]["radiometric_scale"] == 0.0001
    assert len(manifest["source_context"]["stac_item_sha256"]) == 64


def test_derived_scene_tiles_keep_metadata_and_scene_group(tmp_path):
    scene = tmp_path / "scene.tif"
    data = np.full((3, 12, 12), 90, dtype=np.uint8)
    with rasterio.open(scene, "w", driver="GTiff", width=12, height=12, count=3, dtype="uint8", crs="EPSG:32643", transform=from_origin(100, 200, 10, 10)) as dataset:
        dataset.write(data)
        dataset.update_tags(FAITHSR_DERIVED_PRODUCT="Sentinel-2_L2A_RGB_uint8")
    records = tile_rgb8_scene(scene, tmp_path / "S2_TEST", tile_size=6)
    assert len(records) == 4
    with rasterio.open(records[0]["path"]) as tile:
        assert tile.tags()["FAITHSR_PARENT_SCENE"] == "scene.tif"
        assert tile.tags()["FAITHSR_DERIVED_PRODUCT"] == "Sentinel-2_L2A_RGB_uint8"
        assert tile.transform == from_origin(100, 200, 10, 10)


def test_tile_content_thresholds_reject_constant_and_dark_tiles(tmp_path):
    scene = tmp_path / "mixed.tif"
    data = np.zeros((3, 6, 12), dtype=np.uint8)
    ramp = np.tile(np.arange(6, dtype=np.uint8) * 30, (6, 1))
    data[:, :, 6:] = ramp
    with rasterio.open(scene, "w", driver="GTiff", width=12, height=6, count=3, dtype="uint8") as dataset:
        dataset.write(data)
    records = tile_rgb8_scene(
        scene,
        tmp_path / "filtered",
        tile_size=6,
        min_nonzero_fraction=0.75,
        min_channel_std=2.0,
    )
    assert len(records) == 1
    assert records[0]["window"]["column"] == 6
    assert records[0]["nonzero_fraction"] >= 0.75
    assert records[0]["mean_channel_std"] >= 2.0
