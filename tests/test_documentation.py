from pathlib import Path

from faithsr.documentation import append_run_results, write_model_card
from faithsr.model import FaithSR3X, FaithSRConfig


def test_each_run_model_card_contains_math_layers_activations_and_lineage(tmp_path):
    model = FaithSR3X(FaithSRConfig(channels=8, num_blocks=2, embed_dim=12))
    path = write_model_card(tmp_path / "model-card.md", model, {"scale": 3, "model": {"channels": 8}}, "test-run", "Test one component.", "plain-run")
    text = path.read_text(encoding="utf-8")
    for required in ("Mathematical forward function", "Architecture figure", "Top-level layers", "Activations and rationale", "Test one component", "plain-run", "PixelShuffle"):
        assert required in text


def test_results_are_added_idempotently_to_model_card(tmp_path):
    model = FaithSR3X(FaithSRConfig(channels=8, num_blocks=2, embed_dim=12))
    card = write_model_card(tmp_path / "model-card.md", model, {"scale": 3, "model": {}}, "run", "why", "previous")
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("epoch,train_loss,psnr,ssim,learning_rate\n1,1.0,20.0,.5,.001\n2,.8,21.0,.6,.0001\n", encoding="utf-8")
    append_run_results(card, metrics, 12.5)
    append_run_results(card, metrics, 12.5)
    text = card.read_text(encoding="utf-8")
    assert text.count("GENERATED_RESULTS_START") == 1
    assert "| Best | 2 |" in text
    assert "12.50 seconds" in text


def test_native_x4_result_records_decision_evidence():
    result = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "research"
        / "swin2sr-native-x4-diagnostic-results.md"
    ).read_text(encoding="utf-8")

    for required in (
        "Accepted v2 Results",
        "Model function, layers",
        "GELU",
        "PixelShuffle",
        "Compression-heavy",
        "3,418.60 ms",
        "Fails <1 s gate",
        "compact native-x4 student",
    ):
        assert required in result
