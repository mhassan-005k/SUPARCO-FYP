import json

from faithsr.checkpoint import save_checkpoint
from faithsr.export import export_onnx
from faithsr.model import FaithSR3X, FaithSRConfig


def test_onnx_export_is_checked_for_dynamic_3x_shapes(tmp_path):
    model = FaithSR3X(FaithSRConfig(channels=8, num_blocks=1, embed_dim=12, kernel_size=3))
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(checkpoint, model)
    output = tmp_path / "model.onnx"
    result = export_onnx(str(checkpoint), str(output), size=8)
    assert result["verification"]["onnx_checker"] == "pass"
    assert [check["output"][-2:] for check in result["verification"]["dynamic_shape_checks"]] == [[24, 24], [33, 39]]
    assert max(check["max_abs_error"] for check in result["verification"]["dynamic_shape_checks"]) < 1e-4
    assert json.loads(output.with_suffix(".json").read_text())["onnx_sha256"] == result["onnx_sha256"]
