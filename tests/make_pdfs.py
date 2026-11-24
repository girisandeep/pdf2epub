"""Generate synthetic PDFs that exercise the converter."""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus import (BaseDocTemplate, Frame, ListFlowable, ListItem, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle, FrameBreak, NextPageTemplate)

LOREM = ("The quick brown fox jumps over the lazy dog while the committee deliberates about internationalization "
         "and the surprisingly complicated matter of typographic conventions in modern publishing workflows. "
         "Readers expect consistent paragraphs, clean headings, and sensible reading order across columns and pages. ")


def book(path: str) -> None:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=11, leading=14,
                          alignment=TA_JUSTIFY, spaceAfter=8)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=22, spaceAfter=14)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=15, spaceAfter=8)
    mono = ParagraphStyle("mono", fontName="Courier", fontSize=9, leading=11)

    def deco(c, d):
        c.saveState()
        c.setFont("Helvetica", 9)
        c.drawString(72, letter[1] - 40, "A Synthetic Test Book")
        c.drawCentredString(letter[0] / 2, 30, f"Page {d.page}")
        c.restoreState()

    class Doc(BaseDocTemplate):
        def afterFlowable(self, f):
            if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
                lvl = 0 if f.style.name == "h1" else 1
                key = f"k{id(f)}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(f.getPlainText(), key, level=lvl, closed=False)

    doc = Doc(path, pagesize=letter, title="The Synthetic Book", author="Test Author")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(72, 60, letter[0] - 144, letter[1] - 130)], onPage=deco)])
    s = [Paragraph("Chapter One: Beginnings", h1)]
    s += [Paragraph(LOREM * 3, body)] * 2
    s += [Paragraph("A first section", h2), Paragraph(LOREM + "Visit https://example.com/docs for details. " + LOREM, body)]
    s += [ListFlowable([ListItem(Paragraph("First <b>bold</b> bullet " + LOREM[:90], body)),
                        ListItem(Paragraph("Second <i>italic</i> bullet", body)),
                        ListItem(Paragraph("Third bullet", body))], bulletType="bullet", start="•")]
    s += [ListFlowable([ListItem(Paragraph("Step one " + LOREM[:80], body)), ListItem(Paragraph("Step two", body)),
                        ListItem(Paragraph("Step three", body))], bulletType="1", bulletFormat="%s.")]
    s += [Preformatted("def hello():\n    print('hi')\n\nhello()", mono), Spacer(1, 10)]
    data = [["Name", "Qty", "Price"], ["Widget", "4", "$3.50"], ["Gadget", "10", "$12.00"], ["Doohickey", "1", "$99.99"]]
    t = Table(data)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    s += [t, Spacer(1, 12)]
    s += [PageBreak(), Paragraph("Chapter Two: Middles", h1)] + [Paragraph(LOREM * 4, body)] * 3
    s += [Paragraph("Another section", h2)] + [Paragraph(LOREM * 3, body)] * 4
    s += [PageBreak(), Paragraph("Chapter Three: Endings", h1)] + [Paragraph(LOREM * 4, body)] * 6
    doc.build(s)


def two_column(path: str) -> None:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=9.5, leading=11.5,
                          alignment=TA_JUSTIFY, spaceAfter=0)
    title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20, leading=24, alignment=1, spaceAfter=10)
    abstract = ParagraphStyle("abs", parent=body, fontSize=9.5, spaceAfter=10)
    h = ParagraphStyle("hh", fontName="Helvetica-Bold", fontSize=12, leading=14, spaceBefore=8, spaceAfter=4)
    W, H = letter
    top = Frame(54, H - 200, W - 108, 146, id="top")
    cols = [Frame(54, 54, (W - 108 - 20) / 2, H - 270, id="c1"), Frame(54 + (W - 108 + 20) / 2, 54, (W - 108 - 20) / 2, H - 270, id="c2")]
    full = [Frame(54, 54, (W - 108 - 20) / 2, H - 108, id="d1"), Frame(54 + (W - 108 + 20) / 2, 54, (W - 108 - 20) / 2, H - 108, id="d2")]
    doc = BaseDocTemplate(path, pagesize=letter)
    doc.addPageTemplates([PageTemplate(id="first", frames=[top] + cols, autoNextPageTemplate="later"),
                          PageTemplate(id="later", frames=full)])
    para = LOREM * 2
    s = [Paragraph("Column-Aware Reading Order in Practice", title), Paragraph("Jane Doe, John Roe", ParagraphStyle("a", parent=body, alignment=1)),
         Spacer(1, 6), Paragraph("<b>Abstract.</b> " + LOREM * 2, abstract), FrameBreak()]
    for i in range(1, 8):
        s += [Paragraph(f"{i}. Section Number {i}", h)] + [Paragraph(para, body), Spacer(1, 6)] * 3
    doc.build(s)


def hyphenated(path: str) -> None:
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Times-Roman", 12)
    y = 700
    lines = ["This paragraph demonstrates inter-", "national hyphenation handling, plus a well-",
             "known compound word that must survive.", "",
             ]
    for ln in lines:
        c.drawString(72, y, ln)
        y -= 15
    c.setFont("Times-Roman", 12)
    c.drawString(72, y - 15, "A new paragraph starts here and is separate.")
    c.save()


def footnotes(path: str) -> None:
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Times-Roman", 12)
    c.drawString(72, 700, "Footnotes should be linked to their references")
    w = c.stringWidth("Footnotes should be linked to their references", "Times-Roman", 12)
    c.setFont("Times-Roman", 7)
    c.drawString(72 + w + 1, 705, "1")
    c.setFont("Times-Roman", 12)
    c.drawString(72 + w + 6, 700, ", which is useful.")
    c.drawString(72, 685, "Second sentence on the same page keeps going on and on and on.")
    c.line(72, 120, 200, 120)
    c.setFont("Times-Roman", 8)
    c.drawString(72, 108, "1  This is the text of the first footnote.")
    c.showPage()
    c.save()


if __name__ == "__main__":
    import sys, os
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    book(f"{out}/book.pdf"); two_column(f"{out}/twocol.pdf"); hyphenated(f"{out}/hyphen.pdf"); footnotes(f"{out}/footnotes.pdf")


def equations(path: str) -> None:
    """Prose with a centred, numbered display equation (superscripts + relation) between paragraphs."""
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Times-Roman", 12)
    y = 700
    for ln in ["The Pythagorean relation is stated for any right triangle and is used throughout the",
               "remainder of this document without further comment, as follows:"]:
        c.drawString(72, y, ln); y -= 15
    y -= 20
    x = 250
    for txt, size, dy in [("x", 12, 0), ("2", 8, 5), (" + y", 12, 0), ("2", 8, 5), (" = z", 12, 0), ("2", 8, 5)]:
        c.setFont("Times-Italic" if txt.strip() in ("x", "+ y", "= z") else "Times-Roman", size)
        c.drawString(x, y + dy, txt)
        x += c.stringWidth(txt, "Times-Roman", size)
    c.setFont("Times-Roman", 12)
    c.drawString(500, y, "(1)")
    y -= 40
    for ln in ["where x and y are the lengths of the two shorter sides of the triangle in question and",
               "z is the length of the hypotenuse, which is always the longest side."]:
        c.drawString(72, y, ln); y -= 15
    c.save()
