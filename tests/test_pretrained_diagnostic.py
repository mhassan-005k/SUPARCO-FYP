from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from faithsr.pretrained_diagnostic import STRESS_PROFILES, Swin2SRNativeX4Adapter, _predict_native_x4


class FakeHuggingFaceX4(nn.Module):
    def forward(self, pixel_values):
        return SimpleNamespace(reconstruction=F.interpolate(pixel_values, scale_factor=4, mode="nearest"))


def test_native_x4_adapter_exposes_common_inference_contract():
    adapter = Swin2SRNativeX4Adapter(FakeHuggingFaceX4())
    image = torch.rand(1, 3, 5, 7)
    output, auxiliary = adapter(image, return_aux=True)
    assert output.shape == (1, 3, 20, 28)
    assert auxiliary["uncertainty"] is None
    assert adapter.scale == 4
    assert adapter.config.in_channels == 3


def test_stress_profiles_cover_distinct_failure_families():
    assert set(STRESS_PROFILES) == {
        "clean_bicubic", "mild_anisotropic", "severe_blur", "noise_heavy",
        "compression_heavy", "aliasing_heavy", "mtf_low_nyquist",
    }
    assert STRESS_PROFILES["noise_heavy"].noise == 0.9
    assert STRESS_PROFILES["compression_heavy"].jpeg_quality == 55
    assert STRESS_PROFILES["mtf_low_nyquist"].mtf_nyquist == 0.15


def test_native_x4_prediction_supports_huggingface_and_tensor_models():
    image = torch.rand(1, 3, 5, 7)
    huggingface = _predict_native_x4(FakeHuggingFaceX4(), image, "pixel_values")
    tensor_model = _predict_native_x4(nn.Upsample(scale_factor=4, mode="nearest"), image, "tensor")

    assert huggingface.shape == tensor_model.shape == (1, 3, 20, 28)
