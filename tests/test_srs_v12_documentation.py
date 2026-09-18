import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "docs" / "requirements"


def test_authoritative_srs_v12_pdf_and_markdown_are_complete():
    pdf = REQUIREMENTS / "requirements-for-super-resolution-project-v1.2.pdf"
    markdown = REQUIREMENTS / "requirements-for-super-resolution-project-v1.2.md"

    assert pdf.is_file()
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == (
        "97cb6cbfd2fc51e6045449f561e925b575d860312b542910361522022274d10e"
    )

    text = markdown.read_text(encoding="utf-8")
    assert "Version 1.2" in text
    for number in range(1, 13):
        assert re.search(rf"\bFR-{number}\b", text), f"FR-{number} missing"
    for number in range(1, 7):
        assert re.search(rf"\bRA-{number}\b", text), f"RA-{number} missing"
    for required in ("10-bit", "12-bit", "RMSE", "0.5 pixels", "80%"):
        assert required in text


def test_srs_v12_audit_has_complete_status_and_acceptance_evidence():
    text = (REQUIREMENTS / "srs-v1.2-compliance-audit.md").read_text(encoding="utf-8")

    for number in range(1, 13):
        assert re.search(rf"\| FR{number} \|", text), f"FR{number} audit row missing"
    for number in range(1, 7):
        assert re.search(rf"\| RA{number} \|", text), f"RA{number} audit row missing"

    for required in (
        "| Pass | 3 |",
        "| Partial | 8 |",
        "| Fail | 1 |",
        "10-bit imagery",
        "12-bit imagery",
        "Panchromatic",
        "Multispectral",
        "RMSE reduction",
        "Mean registration error",
        "at least 6 of 7 passes",
    ):
        assert required in text


def test_srs_v12_roadmap_registers_missing_capabilities_and_gates():
    text = (REQUIREMENTS / "srs-v1.2-implementation-roadmap.md").read_text(
        encoding="utf-8"
    )

    for experiment in range(20, 26):
        assert f"| E{experiment} |" in text

    for required in (
        "DigitalNumberCodec",
        "effective_bit_depth",
        "RMSE_reduction_percent >= 10",
        "mean_registration_error <= 0.5",
        "Sentinel-2 RGB+NIR",
        "independent set once",
        "at least 6 of 7 criteria pass",
        "has no panchromatic band",
    ):
        assert required in text
