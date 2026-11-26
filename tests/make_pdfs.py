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
    equations(f"{out}/equations.pdf"); crossrefs(f"{out}/crossrefs.pdf")
    bibliography(f"{out}/bibliography.pdf"); algorithm(f"{out}/algorithm.pdf")


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


def crossrefs(path: str) -> None:
    """Two-page PDF with sections, a figure caption, a table, and cross-references in prose."""
    from reportlab.lib.units import inch
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=11, leading=14, spaceAfter=8)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=18, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=14, spaceAfter=8)
    cap = ParagraphStyle("cap", parent=ss["Normal"], fontName="Times-Italic", fontSize=10, leading=12, spaceAfter=6)

    doc = BaseDocTemplate(path, pagesize=letter)
    f = Frame(72, 72, letter[0] - 144, letter[1] - 144, id="main")
    doc.addPageTemplates([PageTemplate("main", frames=[f])])

    story = [
        Paragraph("1 Introduction", h1),
        Paragraph("This paper introduces a new model. The main result is in Section 2 and "
                   "the architecture is shown in Figure 1. Performance numbers are in Table 1.", body),
        PageBreak(),
        Paragraph("2 Method", h1),
        Paragraph("2.1 Architecture", h2),
        Paragraph("The architecture is illustrated below.", body),
    ]
    # Simple figure placeholder
    t_fig = Table([["[Diagram]"]], colWidths=[3 * inch], rowHeights=[1 * inch])
    t_fig.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.black),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t_fig)
    story.append(Paragraph("Figure 1: Overview of the proposed architecture.", cap))
    story.append(Paragraph("As shown in Figure 1, the model uses an encoder-decoder structure. "
                           "See Section 1 for motivation and Table 1 for results.", body))

    story.append(Paragraph("2.2 Results", h2))
    data = [["Model", "Accuracy"], ["Ours", "95.2"], ["Baseline", "91.0"]]
    t = Table(data, colWidths=[2 * inch, 1.5 * inch])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story.append(Paragraph("Table 1: Comparison of model accuracy.", cap))
    story.append(t)
    story.append(Paragraph("Table 1 shows that our model outperforms the baseline.", body))

    doc.build(story)


def bibliography(path: str) -> None:
    """PDF with a References section containing [N]-style bibliography entries."""
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=11, leading=14, spaceAfter=8)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=18, spaceAfter=10)

    doc = BaseDocTemplate(path, pagesize=letter)
    f = Frame(72, 72, letter[0] - 144, letter[1] - 144, id="main")
    doc.addPageTemplates([PageTemplate("main", frames=[f])])

    story = [
        Paragraph("1 Introduction", h1),
        Paragraph("Deep learning has achieved remarkable results [1]. "
                   "Recent work [2] builds on these foundations. "
                   "Earlier methods [3] used different approaches.", body),
        Paragraph("References", h1),
        Paragraph("[1] A. Smith and B. Jones. Deep learning for natural language processing. "
                   "In Proceedings of ICML, pages 100-110, 2020.", body),
        Paragraph("[2] C. Lee. Attention mechanisms in neural networks. "
                   "Journal of Machine Learning Research, 21:1-45, 2021.", body),
        Paragraph("[3] D. Wang, E. Chen, and F. Garcia. Classical approaches to text classification. "
                   "In NeurIPS, 2019.", body),
    ]
    doc.build(story)


def algorithm(path: str) -> None:
    """PDF with algorithm pseudocode blocks using arrow notation."""
    c = canvas.Canvas(path, pagesize=letter)
    y = 700
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, y, "1 Method")
    y -= 30
    c.setFont("Times-Roman", 11)
    c.drawString(72, y, "We propose Algorithm 1 for optimizing the objective function.")
    y -= 25
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, y, "Algorithm 1: Gradient Descent")
    y -= 18
    algo_lines = [
        "Require: learning rate lr",
        "Require: initial parameters w0",
        "w ← w0",
        "while not converged do",
        "    grad ← ComputeGradient(w)",
        "    w ← w - lr * grad",
        "end while",
        "return w",
    ]
    c.setFont("Times-Roman", 10)
    for ln in algo_lines:
        c.drawString(90, y, ln)
        y -= 20
    y -= 12
    c.setFont("Times-Roman", 11)
    c.drawString(72, y, "The algorithm converges under standard assumptions.")
    c.save()


def multi_author(path: str) -> None:
    c = canvas.Canvas(path, pagesize=letter)
    W, _ = letter
    cx = W / 2
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(cx, 700, "Multi-Author Extraction in Practice")
    c.setFont("Times-Roman", 11)
    c.drawCentredString(cx, 670, "Alice Smith")
    c.drawCentredString(cx, 656, "Bob Jones")
    c.drawCentredString(cx, 642, "Carol Williams")
    c.setFont("Times-Roman", 9)
    c.drawCentredString(cx, 622, "University of Testing")
    c.drawCentredString(cx, 610, "alice@test.edu")
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, 580, "Abstract")
    c.setFont("Times-Roman", 11)
    c.drawString(72, 560, "This paper describes the extraction of author names from academic PDF documents.")
    c.drawString(72, 545, LOREM)
    c.save()
