import torch

from faithsr.losses import CompressionAwareObjective


def _auxiliary(batch: int):
    return {
        "reprojected_lr": None,
        "degradation_params": None,
        "uncertainty": None,
    }


def test_compression_focus_upweights_errors_from_heavily_compressed_samples():
    hr = torch.zeros(2, 3, 8, 8)
    sr = hr.clone()
    sr[0] += 0.1
    sr[1] += 0.2
    lr = torch.zeros(2, 3, 2, 2)
    degradation = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    plain = CompressionAwareObjective({"charbonnier": 1.0, "compression_focus": 0.0})
    focused = CompressionAwareObjective({"charbonnier": 1.0, "compression_focus": 3.0})

    _, plain_terms = plain(sr, hr, lr, _auxiliary(2), degradation)
    _, focused_terms = focused(sr, hr, lr, _auxiliary(2), degradation)

    assert focused_terms["charbonnier"] > plain_terms["charbonnier"]


def test_personalized_teacher_adds_pixel_and_edge_distillation_terms():
    sr = torch.rand(2, 3, 16, 16, requires_grad=True)
    hr = torch.rand_like(sr)
    lr = torch.rand(2, 3, 4, 4)
    teacher = torch.rand_like(sr)
    objective = CompressionAwareObjective(
        {"charbonnier": 1.0, "distillation": 0.05, "distillation_edge": 0.01}
    )

    loss, terms = objective(sr, hr, lr, _auxiliary(2), teacher=teacher)
    loss.backward()

    assert {"distillation", "distillation_edge"} <= terms.keys()
    assert sr.grad is not None
