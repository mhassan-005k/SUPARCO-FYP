import torch

from faithsr.pretrained_finetune import exact_x3


def test_pretrained_x4_resize_is_explicitly_exact_x3():
    lr = torch.rand(2, 3, 5, 7)
    x4 = torch.rand(2, 3, 20, 28)
    target = torch.rand(2, 3, 15, 21)
    output = exact_x3(x4, target)
    assert output.shape == (2, 3, 15, 21)
