from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from faithsr.downstream import classification_metrics, majority_worldcover_labels, paired_accuracy_interval


def test_majority_label_uses_coordinate_centred_window(tmp_path):
    raster = tmp_path / "worldcover.tif"
    data = np.full((20, 20), 40, dtype=np.uint8)
    data[8:13, 8:13] = 50
    with rasterio.open(raster, "w", driver="GTiff", width=20, height=20, count=1, dtype="uint8", crs="EPSG:4326", transform=from_origin(0, 2, 0.1, 0.1), nodata=0) as target:
        target.write(data, 1)
    image = tmp_path / "1.0_1.0.png"
    assert majority_worldcover_labels([image], raster, window_size=5)[image] == 50


def test_classification_and_paired_cluster_metrics():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert result["accuracy"] == 0.75
    rows = []
    for path, group, base, candidate in (("a", "g1", False, True), ("b", "g1", True, True), ("c", "g2", False, False)):
        rows.extend([
            {"method": "bicubic", "path": path, "geographic_group": group, "correct": base},
            {"method": "faithsr3x", "path": path, "geographic_group": group, "correct": candidate},
        ])
    interval = paired_accuracy_interval(rows, "faithsr3x")
    assert interval["clusters"] == 2
    assert interval["mean_delta"] == 0.25
