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
