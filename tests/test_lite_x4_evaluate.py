import pytest

from faithsr.lite_x4_evaluate import parse_checkpoints


def test_checkpoint_parser_requires_exact_four_way_ablation():
    parsed = parse_checkpoints([
        "base=a.pth",
        "distilled=b.pth",
        "compression=c.pth",
        "full=d.pth",
    ])
    assert tuple(parsed) == ("base", "distilled", "compression", "full")


def test_checkpoint_parser_rejects_missing_duplicate_and_unknown_variants():
    with pytest.raises(ValueError, match="Missing"):
        parse_checkpoints(["base=a.pth"])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_checkpoints([
            "base=a.pth", "base=b.pth", "distilled=c.pth", "compression=d.pth", "full=e.pth"
        ])
    with pytest.raises(ValueError, match="Unknown"):
        parse_checkpoints(["mamba=x.pth"])
