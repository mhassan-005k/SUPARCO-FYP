import json

import torch

import random

from faithsr.degradation import DegradationParameters, degrade, sample_parameters
from faithsr.data import SyntheticSRDataset, append_degradation_records


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
    import numpy as np
    path = tmp_path / "image.png"
    Image.fromarray(np.full((48, 48, 3), 128, np.uint8)).save(path)
    dataset = SyntheticSRDataset([path], crop_size=48, train=False, seed=9)
    first = dataset[0]
    second = dataset[0]
    assert torch.equal(first[0], second[0])
    assert first[4] == second[4]
    assert json.loads(first[4])["seed"] == 9


def test_curriculum_caps_all_sampled_degradation_severities():
    parameters = sample_parameters(random.Random(4), max_severity=0.25, mtf_probability=1.0)
    assert max(parameters.blur, parameters.noise, parameters.compression, parameters.aliasing) <= 0.25
    assert parameters.kernel_family == "mtf_gaussian"
    assert parameters.mtf_nyquist is not None
    assert parameters.severity_cap == 0.25


def test_exact_degradation_records_are_jsonl_replayable(tmp_path):
    details = json.dumps({"seed": 7, "blur": 0.2})
    target = tmp_path / "degradations.jsonl"
    append_degradation_records(target, 3, ["scene.tiff"], [details])
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record == {"epoch": 3, "source": "scene.tiff", "degradation": {"seed": 7, "blur": 0.2}}


def test_explicit_mtf_degradation_is_repeatable():
    parameters = DegradationParameters(0.4, 0.1, 0.2, 0.3, 0.5, 0.4, 15.0, 90, "mtf_gaussian", 0.3, 0.5)
    image = torch.rand(3, 30, 30)
    first, _ = degrade(image, params=parameters, seed=22)
    second, _ = degrade(image, params=parameters, seed=22)
    assert torch.equal(first, second)


def test_bicubic_reference_branch_is_recorded(tmp_path):
    from PIL import Image
    import numpy as np

    path = tmp_path / "image.png"
    Image.fromarray(np.full((48, 48, 3), 128, np.uint8)).save(path)
    dataset = SyntheticSRDataset([path], crop_size=48, train=False, seed=5, complex_probability=0.0)
    lr, _, target, _, details = dataset[0]
    assert lr.shape == (3, 16, 16)
    assert torch.equal(target, torch.zeros(4))
    assert json.loads(details)["kernel_family"] == "bicubic_reference"
