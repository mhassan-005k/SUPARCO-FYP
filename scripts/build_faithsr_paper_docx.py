"""Build the FaithSR closure paper as a polished Word manuscript.

The Markdown file remains the reviewable source of truth. This builder keeps
display equations as native, searchable Word math objects using human-readable
Unicode notation rather than embedding LaTeX source or equation screenshots.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "research" / "papers" / "faithsr-controlled-ablation-paper.md"
OUTPUT = ROOT / "docs" / "research" / "papers" / "faithsr-controlled-ablation-paper.docx"


EQUATIONS = [
    "x = Q₈(Jq(Ng,p(↓s(k * y))))",
    "RMSE = √[(1/N) ∑ᵢ₌₁ⁿ (ŷᵢ − yᵢ)²]\nPSNR = 20 log₁₀(L / RMSE)",
    "SAM(ŝ, s) = arccos[(ŝᵀs) / (‖ŝ‖₂ ‖s‖₂ + ε)]",
    "rᵢ = C¹×¹ₙₒₒₙ[GELU(C³×³ᵈʷ[GELU(C¹×¹ᵤᵖ(fᵢ₋₁))])]",
    "fᵢ = fᵢ₋₁ + γᵢ(z) ⊙ rᵢ + βᵢ(z)",
    "ŷ₀ = Bicubic₃(x) + PixelShuffle₃(Cᵣ(f))",
    "kᵣ = softmax(Wₖz),    ∑p,q kᵣ[p,q] = 1,    kᵣ[p,q] ≥ 0",
    "α(z) = 0.25 σ[Wₐ₂ GELU(Wₐ₁z)]",
    "Attention(Q,K,V) = softmax[cos(Q,K)/τ + S] V",
    "Lₛᴡᴵⁿ = (1/N) ∑ᵢ SmoothL1[Tθ,₄(x)ᵢ − y₄,ᵢ]",
    "L = Lᴄʰᵃʳ + 0.10Lₛₛᴵᴹ + 0.03L∇ + 0.02Lᶠʳᵉᵠ\n    + 0.05Lᵒᵇˢ + 0.02Lᵈᵉᵍ + 0.02Lᵘᵠ + λᴷᴰLᴷᴰ",
    "Lᴄʰᵃʳ = (1/N) ∑ᵢ √[(ŷᵢ − yᵢ)² + ε²]",
    "Fᵢ = Fᵢ₋₁ + sᵢ C¹×¹{DW³×³[GELU(C¹×¹[GELU(DW³×³(Fᵢ₋₁))])]}",
    "G = σ{C¹×¹[GELU(C¹×¹[F,C])]}\nF′ = F + G ⊙ C",
    "GELU(a) = aΦ(a) ≈ (a/2){1 + tanh[√(2/π)(a + 0.044715a³)]}",
    "σ(a) = 1 / (1 + e⁻ᵃ)",
    "softmax(aᵢ) = eᵃⁱ / ∑ⱼ eᵃʲ",
    "ŷ(p) = [∑ᵢ wᵢ(p)ŷᵢ(p)] / max[∑ᵢ wᵢ(p), ε]",
    "x ∈ [0,1]ᴮ×ᶜ×ᴴ×ᵂ",
    "b = DWConv³×³(x),    f₀ = Conv¹×¹ ᶜ→¹⁸⁰(b)",
    "a = σ{W₂ GELU[W₁ GAP(f₀)]}\nfᵢₙ = f₀ + a ⊙ f₀",
    "ŷ = clip{Bicubic₄(x) + PixelShuffle₄[Conv³×³ ¹⁸⁰→¹⁶ᶜ(f)], 0, 1}",
    "xₙₒᵣₘ = xᴰᴺ / (2ᵇ − 1)\nẋᴰᴺ = round{clip(ẋₙₒᵣₘ, 0, 1)(2ᵇ − 1)}",
    "Lᴹᵂᴿ = Lᴄʰᵃʳ + λₛLₛₛᴵᴹ + λᵣLᴿᴹᵂᴱ + λₛₐₘLₛᴀᴹ + λᵍL∇ + λₒLᵒᵇˢ",
]


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for key, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{key}"))
        if node is None:
            node = OxmlElement(f"w:{key}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_text(cell, text: str, bold=False, white=False, size=8.2) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text.strip())
    r.bold = bold
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    if white:
        r.font.color.rgb = RGBColor(255, 255, 255)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)


def add_math_run(paragraph, expression: str) -> None:
    math = OxmlElement("m:oMath")
    run = OxmlElement("m:r")
    run_pr = OxmlElement("m:rPr")
    sty = OxmlElement("m:sty")
    sty.set(qn("m:val"), "p")
    run_pr.append(sty)
    run.append(run_pr)
    text = OxmlElement("m:t")
    text.set(qn("xml:space"), "preserve")
    text.text = expression
    run.append(text)
    math.append(run)
    paragraph._p.append(math)


def add_equation(document: Document, expression: str, number: int) -> None:
    table = document.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.15)
    table.columns[1].width = Inches(0.45)
    table_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    table_pr.append(borders)
    left, right = table.rows[0].cells
    left.text = ""
    for idx, line in enumerate(expression.splitlines()):
        p = left.paragraphs[0] if idx == 0 else left.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        add_math_run(p, line)
    right.text = ""
    p = right.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(f"({number})")
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    for cell in (left, right):
        set_cell_margins(cell, 35, 15, 35, 15)


TOKEN_RE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^\)]+\))")


def add_inline(paragraph, text: str) -> None:
    cursor = 0
    for match in TOKEN_RE.finditer(text):
        if match.start() > cursor:
            paragraph.add_run(text[cursor : match.start()])
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(8.4)
            run.font.color.rgb = RGBColor(50, 70, 90)
        else:
            label, target = re.match(r"\[([^\]]+)\]\(([^\)]+)\)", token).groups()
            run = paragraph.add_run(label)
            run.font.color.rgb = RGBColor(31, 78, 121)
            run.underline = True
            run._r.set(qn("w:rsidRPr"), target[:8])
        cursor = match.end()
    if cursor < len(text):
        paragraph.add_run(text[cursor:])


def configure_styles(document: Document) -> None:
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.3)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    normal.paragraph_format.space_after = Pt(3.2)
    normal.paragraph_format.widow_control = True

    for name, size, before, after, color in (
        ("Title", 17, 0, 8, RGBColor(0, 0, 0)),
        ("Heading 1", 12.5, 10, 4, RGBColor(0, 0, 0)),
        ("Heading 2", 11.2, 7, 3, RGBColor(0, 0, 0)),
        ("Heading 3", 10.4, 5, 2, RGBColor(0, 0, 0)),
    ):
        style = styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        p_pr = style.element.get_or_add_pPr()
        border = p_pr.find(qn("w:pBdr"))
        if border is not None:
            p_pr.remove(border)

    if "Paper Caption" not in styles:
        cap = styles.add_style("Paper Caption", WD_STYLE_TYPE.PARAGRAPH)
    else:
        cap = styles["Paper Caption"]
    cap.font.name = "Times New Roman"
    cap.font.size = Pt(8.5)
    cap.font.italic = True
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after = Pt(6)
    cap.paragraph_format.keep_with_next = False


def configure_page(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.62)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    section.header_distance = Inches(0.28)
    section.footer_distance = Inches(0.3)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = header.add_run("Fidelity Constrained Satellite Image Super Resolution")
    run.font.name = "Times New Roman"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0, 0, 0)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("Page ")
    run.font.name = "Times New Roman"
    run.font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def parse_table(lines: list[str], start: int, document: Document) -> int:
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        parts = [part.strip() for part in lines[i].strip().strip("|").split("|")]
        rows.append(parts)
        i += 1
    if len(rows) < 2:
        return i
    data = [rows[0]] + rows[2:]
    cols = max(len(row) for row in data)
    table = document.add_table(rows=len(data), cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    table.style = "Table Grid"
    for r_idx, row in enumerate(data):
        for c_idx in range(cols):
            text = row[c_idx] if c_idx < len(row) else ""
            set_cell_text(table.cell(r_idx, c_idx), text, bold=r_idx == 0, white=r_idx == 0)
            if r_idx == 0:
                set_cell_shading(table.cell(r_idx, c_idx), "1F4E79")
            elif r_idx % 2 == 0:
                set_cell_shading(table.cell(r_idx, c_idx), "EAF2F8")
    set_repeat_table_header(table.rows[0])
    for row in table.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    document.add_paragraph().paragraph_format.space_after = Pt(0)
    return i


def add_image(document: Document, target: str, alt: str) -> None:
    path = (SOURCE.parent / target).resolve()
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    if path.exists():
        run = p.add_run()
        picture = run.add_picture(str(path), width=Inches(6.75))
        picture._inline.docPr.set("descr", alt)
        picture._inline.docPr.set("title", alt)
    else:
        r = p.add_run(f"[Figure unavailable: {alt}]")
        r.italic = True
        r.font.color.rgb = RGBColor(160, 0, 0)


def add_title_block(document: Document, lines: list[str]) -> int:
    title = lines[0].removeprefix("# ").strip()
    p = document.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run(title)

    i = 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    for idx in range(3):
        if i >= len(lines):
            break
        raw = lines[i].strip().replace("  ", "").replace("<br>", "")
        raw = raw.strip("*")
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1 if idx < 2 else 7)
        run = p.add_run(raw)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10.5 if idx == 0 else 9.5)
        run.bold = idx == 0
        run.italic = idx > 0
        i += 1
    return i


def build() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    document = Document()
    configure_styles(document)
    configure_page(document)
    document.core_properties.title = "Fidelity Constrained Satellite Image Super Resolution"
    document.core_properties.subject = "Controlled FaithSR and Swin2SR experiments"
    document.core_properties.author = "Muhammad Hassan; Muhammad Mahad Khan"
    document.core_properties.keywords = "Sentinel-2, super-resolution, Swin2SR, GeoTIFF"

    i = add_title_block(document, lines)
    equation_index = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped == "$$":
            i += 1
            latex = []
            while i < len(lines) and lines[i].strip() != "$$":
                latex.append(lines[i])
                i += 1
            equation_index += 1
            if equation_index > len(EQUATIONS):
                raise RuntimeError(f"Missing readable equation mapping for equation {equation_index}")
            add_equation(document, EQUATIONS[equation_index - 1], equation_index)
            i += 1
            continue

        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^\)]+)\)", stripped)
        if image_match:
            add_image(document, image_match.group(2), image_match.group(1))
            i += 1
            continue

        if stripped.startswith("|"):
            i = parse_table(lines, i, document)
            continue

        if stripped.startswith("### "):
            document.add_heading(stripped[4:], level=2)
            i += 1
            continue
        if stripped.startswith("## "):
            document.add_heading(stripped[3:], level=1)
            i += 1
            continue
        if stripped.startswith("# "):
            document.add_heading(stripped[2:], level=1)
            i += 1
            continue

        if re.match(r"^\d+\. ", stripped):
            p = document.add_paragraph(style="List Number")
            p.paragraph_format.left_indent = Inches(0.18)
            p.paragraph_format.first_line_indent = Inches(-0.18)
            add_inline(p, re.sub(r"^\d+\. ", "", stripped))
            i += 1
            continue
        if stripped.startswith("- "):
            p = document.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Inches(0.18)
            p.paragraph_format.first_line_indent = Inches(-0.18)
            add_inline(p, stripped[2:])
            i += 1
            continue

        paragraph_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt == "$$" or nxt.startswith(("#", "|", "- ", "![")) or re.match(r"^\d+\. ", nxt):
                break
            paragraph_lines.append(nxt)
            i += 1
        text = " ".join(paragraph_lines)
        style = "Paper Caption" if text.startswith("**Fig.") else None
        p = document.add_paragraph(style=style)
        if text.startswith("**Abstract") or text.startswith("**Index Terms"):
            p.paragraph_format.left_indent = Inches(0.24)
            p.paragraph_format.right_indent = Inches(0.24)
        add_inline(p, text)

    if equation_index != len(EQUATIONS):
        raise RuntimeError(f"Expected {len(EQUATIONS)} equations but parsed {equation_index}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
