from pathlib import Path

from faithsr.data import geographic_group, geographic_split


def test_coordinate_tiles_are_grouped_into_geographic_cells():
    near_a = Path("rgb/28.2056_76.8558.png")
    near_b = Path("rgb/28.2090_76.8646.png")
    far = Path("rgb/28.4056_77.1558.png")
    assert geographic_group(near_a, 0.1) == geographic_group(near_b, 0.1)
    assert geographic_group(near_a, 0.1) != geographic_group(far, 0.1)


def test_geographic_split_never_leaks_a_coordinate_grid():
    files = [Path(f"rgb/{lat}_{lon}.png") for lat in (28.20, 28.21, 28.40, 28.41) for lon in (76.85, 76.86)]
    train, validation = geographic_split(files, train_ratio=0.5, seed=42, grid_degrees=0.1)
    train_groups = {geographic_group(path, 0.1) for path in train}
    validation_groups = {geographic_group(path, 0.1) for path in validation}
    assert train and validation
    assert train_groups.isdisjoint(validation_groups)


def test_non_coordinate_dataset_uses_parent_groups():
    files = [Path("Forest/a.jpg"), Path("Forest/b.jpg"), Path("River/c.jpg")]
    train, validation = geographic_split(files, train_ratio=0.5, seed=1)
    assert {path.parent.name for path in train}.isdisjoint({path.parent.name for path in validation})
