"""
src/exporter.py — Fully fixed version
"""

import os, sys, re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR


def _safe_filename(video_id):
    return f"notes_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _clean(text):
    """Replace all special unicode chars with safe ASCII equivalents."""
    table = {
        0x2014:"--",  0x2013:"-",   0x2012:"-",   0x2015:"--",
        0x2018:"'",   0x2019:"'",   0x201A:"'",   0x201B:"'",
        0x201C:"\"",  0x201D:"\"",  0x201E:"\"",  0x201F:"\"",
        0x2026:"...", 0x2022:"-",   0x00B7:"-",   0x2010:"-",
        0x2011:"-",   0x2039:"<",   0x203A:">",   0x2032:"'",
        0x00A0:" ",   0x00AD:"-",   0x2044:"/",   0x2060:"",
        0x200B:"",    0x200C:"",    0x200D:"",    0xFEFF:"",
    }
    text = text.translate(table)
    # Remove any remaining non-latin1 characters
    result = ""
    for ch in text:
        try:
            ch.encode("latin-1")
            result += ch
        except UnicodeEncodeError:
            result += "?"
    return result


def _strip_md(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`",        r"\1", text)
    return text


def _clean_md(text):
    return _clean(_strip_md(text))


# ── Markdown ──────────────────────────────────────────────────────────────────

def export_markdown(notes_md, video_id):
    path = os.path.join(OUTPUT_DIR, _safe_filename(video_id) + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(notes_md)
    print(f"[Export] Markdown saved: {path}")
    return path


# ── PDF ───────────────────────────────────────────────────────────────────────

def export_pdf(notes_md, video_id):
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("fpdf2 not installed.")

    class PDF(FPDF):
        def header(self):
            pass
        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=18, top=18, right=18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Page background — white
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, 210, 297, "F")

    W = 174  # usable width (210 - 18*2)

    # ── Style helpers ─────────────────────────────────────────────────────────

    def h1(text):
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(47, 84, 204)
        pdf.multi_cell(W, 11, _clean_md(text), align="L")
        # Underline
        y = pdf.get_y()
        pdf.set_draw_color(47, 84, 204)
        pdf.set_line_width(0.5)
        pdf.line(18, y, 192, y)
        pdf.ln(4)
        pdf.set_text_color(30, 30, 30)

    def h2(text):
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(47, 84, 204)
        # Coloured rect behind heading
        x, y = pdf.get_x(), pdf.get_y()
        pdf.set_fill_color(235, 240, 255)
        pdf.rect(16, y - 1, W + 2, 9, "F")
        pdf.multi_cell(W, 8, "  " + _clean_md(text), align="L")
        pdf.set_text_color(30, 30, 30)
        pdf.ln(1)

    def h3(text):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(70, 110, 200)
        pdf.multi_cell(W, 7, _clean_md(text), align="L")
        pdf.set_text_color(40, 40, 40)

    def body(text):
        cleaned = _clean_md(text)
        if not cleaned.strip():
            return
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(W, 6, cleaned, align="L")

    def bullet(text):
        cleaned = _clean_md(text)
        if not cleaned.strip():
            return
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        # bullet symbol + indent
        pdf.set_x(22)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(47, 84, 204)
        pdf.cell(5, 6, "-")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(W - 7, 6, cleaned, align="L")

    def keyword_line(text):
        # Render keywords as styled tags
        cleaned = _clean_md(text)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(47, 84, 150)
        pdf.set_fill_color(225, 232, 255)
        # Split by two or more spaces (how they're joined in note_generator)
        tags = [t.strip().strip("`") for t in re.split(r"\s{2,}", cleaned) if t.strip()]
        x_start = 18
        y = pdf.get_y()
        x = x_start
        for tag in tags:
            tag_w = pdf.get_string_width(tag) + 6
            if x + tag_w > 192:
                x = x_start
                y += 7
                pdf.set_y(y)
            pdf.set_xy(x, y)
            pdf.set_fill_color(225, 232, 255)
            pdf.set_draw_color(170, 190, 240)
            pdf.rect(x, y, tag_w, 6, "FD")
            pdf.set_xy(x + 3, y)
            pdf.cell(tag_w - 6, 6, tag)
            x += tag_w + 3
        pdf.set_y(y + 8)
        pdf.set_text_color(40, 40, 40)

    def hr():
        pdf.ln(3)
        pdf.set_draw_color(200, 210, 240)
        pdf.set_line_width(0.3)
        y = pdf.get_y()
        pdf.line(18, y, 192, y)
        pdf.ln(4)

    # ── Render lines ──────────────────────────────────────────────────────────

    in_keywords = False
    for raw in notes_md.splitlines():
        line = raw.rstrip()
        try:
            if line.startswith("# "):
                in_keywords = False
                h1(line[2:])
            elif line.startswith("## "):
                heading = line[3:].strip()
                in_keywords = "Key Terms" in heading or "Named Entities" in heading
                h2(heading)
            elif line.startswith("### "):
                in_keywords = False
                h3(line[4:])
            elif line.startswith("- "):
                content = line[2:]
                # Detect keyword/tag lines (contains backticks)
                if in_keywords and "`" not in content:
                    # Named entity bullet
                    bullet(_strip_md(content).replace("**", ""))
                else:
                    bullet(content)
            elif line == "---":
                hr()
            elif line == "":
                pdf.ln(2)
            else:
                # Keyword tag lines (contain backticks)
                if "`" in line:
                    keyword_line(line)
                else:
                    body(line)
        except Exception as e:
            # Never crash on a single line
            try:
                body(str(e)[:60])
            except Exception:
                pass

    path = os.path.join(OUTPUT_DIR, _safe_filename(video_id) + ".pdf")
    pdf.output(path)
    print(f"[Export] PDF saved:      {path}")
    return path


# ── DOCX ──────────────────────────────────────────────────────────────────────

def export_docx(notes_md, video_id):
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches, RGBColor
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        raise RuntimeError("python-docx not installed.")

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Inches(1)
        sec.left_margin = sec.right_margin = Inches(1.2)

    # Change default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    AC  = RGBColor(0x2F, 0x54, 0xCC)
    AC2 = RGBColor(0x46, 0x6E, 0xC8)
    BK  = RGBColor(0x20, 0x20, 0x20)

    def add_horizontal_line(paragraph):
        p = paragraph._p
        pPr = p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"),   "single")
        bottom.set(qn("w:sz"),    "6")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), "2F54CC")
        pBdr.append(bottom)
        pPr.append(pBdr)

    for raw in notes_md.splitlines():
        line = raw.rstrip()
        try:
            if line.startswith("# "):
                p = doc.add_heading(_strip_md(line[2:]), level=1)
                for run in p.runs:
                    run.font.color.rgb = AC
                    run.font.size = Pt(20)
                add_horizontal_line(p)

            elif line.startswith("## "):
                p = doc.add_heading(_strip_md(line[3:]), level=2)
                for run in p.runs:
                    run.font.color.rgb = AC
                    run.font.size = Pt(14)

            elif line.startswith("### "):
                p = doc.add_heading(_strip_md(line[4:]), level=3)
                for run in p.runs:
                    run.font.color.rgb = AC2
                    run.font.size = Pt(12)

            elif line.startswith("- "):
                content = _strip_md(line[2:]).replace("**", "")
                p = doc.add_paragraph(style="List Bullet")
                r = p.add_run(content)
                r.font.size      = Pt(11)
                r.font.color.rgb = BK

            elif line == "---":
                p = doc.add_paragraph()
                add_horizontal_line(p)

            elif line == "":
                doc.add_paragraph("")

            else:
                # Keyword lines — strip backticks
                cleaned = re.sub(r"`(.+?)`", r"\1", line)
                cleaned = _strip_md(cleaned)
                if cleaned.strip():
                    p = doc.add_paragraph()
                    r = p.add_run(cleaned)
                    r.font.size = Pt(10)
                    r.font.color.rgb = AC2

        except Exception:
            pass

    path = os.path.join(OUTPUT_DIR, _safe_filename(video_id) + ".docx")
    doc.save(path)
    print(f"[Export] DOCX saved:     {path}")
    return path


# ── Export all ────────────────────────────────────────────────────────────────

def export_all(notes_md, video_id, formats=None):
    if formats is None:
        formats = ["md", "pdf", "docx"]
    paths = {}
    if "md"   in formats: paths["md"]   = export_markdown(notes_md, video_id)
    if "pdf"  in formats: paths["pdf"]  = export_pdf(notes_md, video_id)
    if "docx" in formats: paths["docx"] = export_docx(notes_md, video_id)
    return paths
