import hashlib
import json
from pathlib import Path


def test_archived_hybrid_result_selects_context_with_passing_latency_and_matching_hashes():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "training_runs" / "faithsr3x" / "hybrid_x4_controlled_v1"
    summary = json.loads((root / "evaluation" / "hybrid-comparison-summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "artifact-provenance.json").read_text(encoding="utf-8"))

    assert summary["hybrid_ranking_best_first"][0] == "context_fusion"
    assert provenance["selected_variant"] == "context_fusion"
    context_delta = summary["paired_variant_intervals"]["comparisons"]["context_fusion_vs_medium_base"]["psnr"]
    assert context_delta["mean_delta"] > 0
    assert context_delta["ci95_low"] > 0
    for variant in ("context_fusion", "residual_cascade"):
        assert summary["acceptance"][variant]["hard_model_latency_pass_1000ms"]
        assert summary["acceptance"][variant]["hard_tiff_pipeline_pass_1000ms"]
        checkpoint = root / variant / "best.pth"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest == provenance["variants"][variant]["checkpoint_sha256"]
        assert digest == summary["variants"][variant]["checkpoint_sha256"]


def test_hybrid_dossier_records_functions_activations_results_and_tradeoffs():
    path = Path(__file__).resolve().parents[1] / "docs" / "research" / "experiments" / "hybrid-x4-capacity-context" / "README.md"
    text = path.read_text(encoding="utf-8")
    for required in (
        "Parallel Context Fusion",
        "Residual Consistency Cascade",
        "Common mathematical model",
        "Activation functions and reasons",
        "Layers and code-to-equation mapping",
        "Accepted Kaggle v1 result",
        "+0.03654 dB",
        "48.44 ms",
        "selected for the next experiment",
    ):
        assert required in text
