from pathlib import Path


def test_native_x4_fixture_uses_rasterio_cubic_resampling():
    runner = (
        Path(__file__).resolve().parents[1]
        / "kaggle_swin2sr_x4_diagnostic"
        / "swin2sr_native_x4_diagnostic.py"
    )
    source = runner.read_text(encoding="utf-8")

    assert "resampling=Resampling.cubic" in source
    assert "Resampling.bicubic" not in source


def test_lite_x4_kernel_runs_registered_ablation_and_uses_accepted_teacher():
    runner = (
        Path(__file__).resolve().parents[1]
        / "kaggle_lite_x4_ablation"
        / "faithsr_lite_x4_ablation.py"
    )
    source = runner.read_text(encoding="utf-8")

    assert '("base", "distilled", "compression", "full")' in source
    assert "personalized_teacher_checkpoint" in source
    assert '"--warmup", "10", "--runs", "100"' in source
    assert "resampling=Resampling.cubic" in source
    assert "Resampling.bicubic" not in source
