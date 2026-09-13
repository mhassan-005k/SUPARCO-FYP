import sys
import types

import torch
from torch import nn

from faithsr.teacher import Swin2SRTeacher


class _FakeModel(nn.Module):
    calls = 0

    def forward(self, pixel_values):
        self.calls += 1
        return types.SimpleNamespace(reconstruction=torch.nn.functional.interpolate(pixel_values, scale_factor=4, mode="nearest"))


class _FakeFactory:
    call = None

    @classmethod
    def from_pretrained(cls, model_id, revision=None):
        cls.call = (model_id, revision)
        return _FakeModel()


def test_teacher_revision_is_pinned_and_x4_output_is_adapted_to_x3(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(Swin2SRForImageSuperResolution=_FakeFactory))
    teacher = Swin2SRTeacher("org/model", revision="immutable-sha", batch_size=1)
    output = teacher(torch.rand(2, 3, 8, 10), (24, 30))
    assert _FakeFactory.call == ("org/model", "immutable-sha")
    assert output.shape == (2, 3, 24, 30)
    assert teacher.model.calls == 2
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    student = torch.rand_like(output, requires_grad=True)
    torch.nn.functional.smooth_l1_loss(student, output).backward()
    assert student.grad is not None


def test_teacher_accepts_wrapped_checkpoint_and_keeps_native_x4(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(Swin2SRForImageSuperResolution=_FakeFactory))
    checkpoint = tmp_path / "teacher.pth"
    torch.save({"model": {}}, checkpoint)
    teacher = Swin2SRTeacher("org/model", checkpoint=checkpoint, revision="immutable-sha")

    image = torch.rand(1, 3, 6, 7)
    output = teacher(image)

    assert output.shape == (1, 3, 24, 28)
