from pathlib import Path

from faithsr.documentation import append_run_results, write_hybrid_x4_model_card, write_model_card
from faithsr.hybrid_x4 import HybridFaithSRX4, HybridFaithSRX4Config
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


def test_lite_x4_ablation_is_preregistered_with_math_and_rejection_rules():
    plan = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "research"
        / "faithsr-lite-x4-ablation-plan.md"
    ).read_text(encoding="utf-8")

    for required in (
        "Pre-registration",
        "Four matched variants",
        "Student mathematical function",
        "Compression gate",
        "GELU",
        "PixelShuffle x4",
        "below 1,000 ms",
        "Mamba/global-context is excluded",
    ):
        assert required in plan


def test_every_registered_experiment_has_a_self_contained_readme():
    root = Path(__file__).resolve().parents[1] / "docs" / "research" / "experiments"
    index = (root / "README.md").read_text(encoding="utf-8")
    folders = [path for path in root.iterdir() if path.is_dir()]
    assert len(folders) >= 14
    for folder in folders:
        readme = folder / "README.md"
        assert readme.is_file(), folder.name
        text = readme.read_text(encoding="utf-8")
        assert folder.name in index
        assert len(text) > 500


def test_docs_separate_training_degradation_from_application_inference():
    root = Path(__file__).resolve().parents[1]
    contract = (root / "docs" / "requirements" / "data-contract.md").read_text(encoding="utf-8")
    deployment = (
        root / "docs" / "research" / "experiments" / "deployment-tiff-latency" / "README.md"
    ).read_text(encoding="utf-8")
    assert "does **not** blur, downsample, compress" in contract
    assert "does **not** synthetically degrade" in deployment
    assert "13-band EuroSAT TIFF" in contract


def test_paper_math_uses_rendered_equations_and_code_mapping():
    root = Path(__file__).resolve().parents[1]
    standard = (root / "docs" / "research" / "experiment-template.md").read_text(encoding="utf-8")
    lite = (
        root
        / "docs"
        / "research"
        / "experiments"
        / "faithsr-lite-x4-personalized-ablation"
        / "README.md"
    ).read_text(encoding="utf-8")
    hybrid = (
        root / "docs" / "research" / "experiments" / "hybrid-x4-capacity-context" / "README.md"
    ).read_text(encoding="utf-8")
    for text in (standard, lite, hybrid):
        assert text.count("$$") >= 4
    for required in (
        "Code-to-equation mapping",
        "Activation functions",
        "FaithSRLiteX4.forward",
        r"\operatorname{GELU}",
        r"\operatorname{Softmax}",
    ):
        assert required in lite
    assert "Implementation status" in hybrid
    assert "Residual Consistency Cascade" in hybrid


def test_hybrid_model_card_has_word_friendly_math_and_code_mapping(tmp_path):
    model = HybridFaithSRX4(HybridFaithSRX4Config(channels=8, num_blocks=1, context_levels=2, variant="context_fusion"))
    path = write_hybrid_x4_model_card(tmp_path / "model-card.md", model, {"scale": 4}, "context", "why", "lite")
    text = path.read_text(encoding="utf-8")
    for required in ("batch x bands x height x width", "$$", "MultiScaleContextBranch", "Sigmoid", "PixelShuffle", "Application-input rule"):
        assert required in text
    assert r"\[" not in text
