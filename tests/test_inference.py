import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from faithsr.inference import tiled_super_resolve


class DeterministicBicubic(nn.Module):
    def forward(self, x, return_aux=False):
        sr = F.interpolate(x, scale_factor=3, mode="bicubic", align_corners=False)
        if return_aux:
            return sr, {"uncertainty": torch.zeros_like(sr[:, :1])}
        return sr


def test_tiled_output_shape_range_and_repeatability():
    image = np.arange(3 * 23 * 29, dtype=np.uint8).reshape(3, 23, 29)
    first, uncertainty = tiled_super_resolve(image, DeterministicBicubic(), "cpu", tile_size=16, overlap=4)
    second, _ = tiled_super_resolve(image, DeterministicBicubic(), "cpu", tile_size=16, overlap=4)
    assert first.shape == (3, 69, 87)
    assert uncertainty.shape == (1, 69, 87)
    assert np.array_equal(first, second)
    assert 0 <= first.min() <= first.max() <= 255
