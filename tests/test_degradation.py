import torch

from faithsr.degradation import degrade
from faithsr.data import SyntheticSRDataset


def test_degradation_is_exact_3x_and_repeatable():
    hr = torch.linspace(0, 1, 3 * 30 * 36).reshape(3, 30, 36)
    first, first_params = degrade(hr, seed=17)
    second, second_params = degrade(hr, seed=17)
    assert first.shape == (3, 10, 12)
    assert torch.equal(first, second)
    assert first_params == second_params
    assert torch.all((first * 255).round() == first * 255)


def test_multiband_degradation_does_not_require_jpeg():
    lr, _ = degrade(torch.rand(5, 30, 30), seed=5)
    assert lr.shape == (5, 10, 10)


def test_validation_dataset_records_replayable_degradation(tmp_path):
    from PIL import Image
    import json
    import numpy as np
    path = tmp_path / "image.png"
    Image.fromarray(np.full((48, 48, 3), 128, np.uint8)).save(path)
    dataset = SyntheticSRDataset([path], crop_size=48, train=False, seed=9)
    first = dataset[0]
    second = dataset[0]
    assert torch.equal(first[0], second[0])
    assert first[4] == second[4]
    assert json.loads(first[4])["seed"] == 9
