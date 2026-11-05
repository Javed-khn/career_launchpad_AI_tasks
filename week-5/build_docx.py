"""Converts the report content (same source as make_report.py) into an editable Word file."""
import re, html
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import make_report as R
from reportlab.platypus import Paragraph, Table, Image, PageBreak

doc = Document()
sec = doc.sections[0]; sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(1.9); sec.top_margin = sec.bottom_margin = Cm(1.8)
doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(10.5)
BLUE = RGBColor(0x1F, 0x3B, 0x6F)

def add_runs(p, markup, size=None, color=None):
    markup = markup.replace("<br/>", "\n")
    for tok in re.split(r"(</?[bi]>)", markup):
        pass
    bold = ital = False
    for tok in re.split(r"(</?[bi]>)", markup):
        if tok == "<b>": bold = True
        elif tok == "</b>": bold = False
        elif tok == "<i>": ital = True
        elif tok == "</i>": ital = False
        elif tok:
            r = p.add_run(html.unescape(tok)); r.bold = bold; r.italic = ital
            if size: r.font.size = Pt(size)
            if color: r.font.color.rgb = color

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:fill"), hexcolor); tcPr.append(s)

for f in R.S:
    if isinstance(f, Paragraph):
        name = f.style.name; text = f.text
        if name == "H1":
            p = doc.add_heading(level=1); add_runs(p, text, 16, BLUE)
        elif name == "H2":
            p = doc.add_heading(level=2); add_runs(p, text, 12.5, BLUE)
        elif name == "T":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; add_runs(p, text.replace("<br/>", " "), 20, BLUE)
            for r in p.runs: r.bold = True
        elif name == "CAP":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; add_runs(p, text, 9, RGBColor(0x66, 0x66, 0x66))
        else:
            p = doc.add_paragraph(); add_runs(p, text); p.paragraph_format.space_after = Pt(6)
    elif isinstance(f, Table):
        data = f._cellvalues; t = doc.add_table(rows=len(data), cols=len(data[0])); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(data):
            for j, v in enumerate(row):
                c = t.cell(i, j); c.text = ""; para = c.paragraphs[0]; r = para.add_run(str(v)); r.font.size = Pt(9)
                if j > 0 and i >= 0 and len(data[0]) > 3: para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if i == 0: r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255); shade(c, "1F3B6F")
                elif i % 2 == 0: shade(c, "EEF2F9")
        doc.add_paragraph()
    elif isinstance(f, Image):
        doc.add_picture(f.filename, width=Cm(min(f.drawWidth / 28.35, 16.4)))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif isinstance(f, PageBreak):
        doc.add_page_break()
doc.save(R.B / "Week5_Task5_Report.docx"); print("docx written")
