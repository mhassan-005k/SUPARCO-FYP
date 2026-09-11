import json

from PIL import Image
import torch

from faithsr.qualitative import representative_sample_index, save_qualitative_panel


class _PreviewDataset:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        hr = torch.tensor(self.values[index], dtype=torch.float32).reshape(1, 1, -1)
        return hr, hr, None, str(index), "{}"


def test_representative_sample_is_deterministic_median_texture():
    dataset = _PreviewDataset([[0, 4], [0, 1], [0, 3], [0, 2]])
    index, texture = representative_sample_index(dataset)
    assert index == 3
    assert texture == torch.tensor([0.0, 2.0]).std().item()


def test_qualitative_panel_and_metrics_are_saved(tmp_path):
    lr = torch.full((3, 4, 5), 0.4)
    hr = torch.full((3, 12, 15), 0.5)
    sr = torch.full((3, 12, 15), 0.48)
    uncertainty = torch.full((1, 12, 15), 0.2)
    output = tmp_path / "preview.png"

    payload = save_qualitative_panel(output, lr, hr, sr, uncertainty, {"epoch": 3}, method_label="Candidate x3")

    assert output.exists()
    display_scale = payload["display_scale"]
    assert Image.open(output).size == (15 * display_scale * 6, (12 + 24) * display_scale)
    saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["epoch"] == 3
    assert saved["method_label"] == "Candidate x3"
    assert saved["output_shape"] == [1, 3, 12, 15]
    assert "psnr" in payload["metrics"]
    assert "psnr" in payload["bicubic_metrics"]
    assert payload["method_minus_bicubic_psnr_db"] == payload["metrics"]["psnr"] - payload["bicubic_metrics"]["psnr"]
    assert "faithsr_minus_bicubic_psnr_db" not in payload
    assert "uncertainty_correlation" in payload["metrics"]
