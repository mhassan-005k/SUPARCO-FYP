from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "research"
PAPER = RESEARCH / "papers" / "faithsr-controlled-ablation-paper.md"
DOCX = PAPER.with_suffix(".docx")


def test_closure_paper_contains_complete_evidence_chain_and_literature_review():
    text = PAPER.read_text(encoding="utf-8")

    for required in (
        "## II. Literature Review",
        "## V. Experiment Logic and Results",
        "Why it was run.",
        "What it proved and how it connected.",
        "Why the pretrained model became better.",
        "Frozen Swin2SR reached 21.92152 dB",
        "0.50377 dB",
        "95% interval 0.40189–0.60565",
        "## VII. Successor Architecture: Multispectral Radiometric Swin2SR",
        "## IX. Future Work",
        "SEN2VENµS provides same-day Sentinel-2 and VENµS",
        "three matched variants",
        "## Appendix A. Layer Inventory",
        "## References",
        "**Muhammad Hassan, Muhammad Mahad Khan**",
    ):
        assert required in text

    assert text.count("$$") == 48
    assert "AI-generated" not in text
    for excluded in ("FYP", "SUPARCO", "SRS", "final-year project"):
        assert excluded not in text


def test_closure_and_successor_documents_freeze_the_decision_boundary():
    closure = (RESEARCH / "approach-2-hybrid-closure.md").read_text(encoding="utf-8")
    successor = (RESEARCH / "swin2sr-multispectral-architecture-plan.md").read_text(
        encoding="utf-8"
    )

    for required in (
        "Status: Closed",
        "0.50377 dB",
        "no additional Mamba",
        "must not appear as positive findings",
    ):
        assert required in closure

    for required in (
        "MSR Swin2SR",
        "10-bit",
        "12-bit",
        "RMSE",
        "registration",
        "B2, B3, B4, and B8",
        "No context cascade, Mamba block",
    ):
        assert required in successor


def test_formatted_word_paper_is_valid_and_contains_native_math():
    assert DOCX.is_file()
    assert DOCX.stat().st_size > 500_000
    with ZipFile(DOCX) as archive:
        names = set(archive.namelist())
        assert "word/document.xml" in names
        xml = archive.read("word/document.xml").decode("utf-8")
    assert xml.count("<m:oMath>") == 29
    assert "Muhammad Hassan, Muhammad Mahad Khan" in xml
    for excluded in ("FYP", "SUPARCO", "SRS", "final-year project"):
        assert excluded not in xml
    assert "Median personalized Swin2SR native x4 case" in xml
    assert "Worst Context Fusion native x4 case" in xml
