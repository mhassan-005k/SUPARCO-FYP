import torch

from faithsr.degradation import degrade


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
