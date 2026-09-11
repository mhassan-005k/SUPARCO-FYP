import torch

import pytest

from faithsr.pretrained_finetune import _write_model_card, exact_x3, output_for_target


def test_pretrained_x4_resize_is_explicitly_exact_x3():
    lr = torch.rand(2, 3, 5, 7)
    x4 = torch.rand(2, 3, 20, 28)
    target = torch.rand(2, 3, 15, 21)
    output = exact_x3(x4, target)
    assert output.shape == (2, 3, 15, 21)


def test_pretrained_native_x4_output_is_not_resized():
    target = torch.rand(2, 3, 20, 28)
    output = torch.rand_like(target)
    assert output_for_target(output, target, 4) is output


def test_pretrained_native_x4_shape_mismatch_fails():
    with pytest.raises(ValueError, match="does not match"):
        output_for_target(torch.rand(1, 3, 20, 28), torch.rand(1, 3, 16, 24), 4)


def test_native_x4_model_card_records_math_layers_activations_and_no_resize(tmp_path):
    target = tmp_path / "model-card.md"
    _write_model_card(target, "pinned/model", "revision", 100, 3, 1e-5, target_scale=4)
    text = target.read_text(encoding="utf-8")
    for required in ("Native x4", "Mathematical function", "Swin2SR_x4(x)", "Layers and activations", "GELU", "PixelShuffle"):
        assert required in text
    assert "native x4 output is supervised directly without a resize" in text
