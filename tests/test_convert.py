import re
import zipfile

import pytest
from lxml import etree

from pdf2epub import ConversionOptions, convert
from pdf2epub.converter import parse_pages

import make_pdfs
from epub_check import check

NS = {"x": "http://www.w3.org/1999/xhtml"}


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory):
    d = tmp_path_factory.mktemp("pdfs")
    make_pdfs.book(str(d / "book.pdf"))
    make_pdfs.two_column(str(d / "twocol.pdf"))
    make_pdfs.hyphenated(str(d / "hyphen.pdf"))
    make_pdfs.footnotes(str(d / "footnotes.pdf"))
    make_pdfs.equations(str(d / "equations.pdf"))
    make_pdfs.crossrefs(str(d / "crossrefs.pdf"))
    return d


def run(pdfs, name, **kw):
    out = pdfs / f"{name}.epub"
    rep = convert(str(pdfs / f"{name}.pdf"), str(out), ConversionOptions(**kw))
    assert check(str(out)) == []
    z = zipfile.ZipFile(out)
    chapters = [etree.fromstring(z.read(n)) for n in sorted(z.namelist()) if n.startswith("OEBPS/chap")]
    return rep, chapters, z


def text(ch):
    return " ".join(ch.xpath("//x:body//text()", namespaces=NS))


def test_book_structure(pdfs):
    rep, chapters, z = run(pdfs, "book")
    assert rep.title == "The Synthetic Book" and rep.heading_source == "outline"
    assert len(chapters) == 3
    h = [(e.tag.split("}")[1], "".join(e.itertext())) for ch in chapters for e in ch.xpath("//x:h1|//x:h2", namespaces=NS)]
    assert h[0] == ("h1", "Chapter One: Beginnings") and ("h2", "A first section") in h
    first = chapters[0]
    assert first.xpath("//x:ul/x:li", namespaces=NS) and len(first.xpath("//x:ol/x:li", namespaces=NS)) == 3
    assert first.xpath("//x:strong[.='bold']", namespaces=NS) and first.xpath("//x:em[.='italic']", namespaces=NS)
    assert first.xpath("//x:a[@href='https://example.com/docs']", namespaces=NS)
    assert "def hello():\n    print('hi')" in "".join(first.xpath("//x:pre//text()", namespaces=NS))
    cells = [c.strip() for c in first.xpath("//x:table//x:td//text()|//x:table//x:th//text()", namespaces=NS)]
    assert cells[:3] == ["Name", "Qty", "Price"] and "$3.50" in cells
    assert not re.search(r"A Synthetic Test Book|Page \d", " ".join(text(c) for c in chapters))  # running heads gone


def test_paragraphs_not_split_across_pages(pdfs):
    _, chapters, _ = run(pdfs, "book")
    ps = [" ".join(p.itertext()) for p in chapters[2].xpath("//x:p", namespaces=NS)]
    assert len(ps) == 6   # six body paragraphs, no stray fragments
    assert all(p.rstrip().endswith(".") for p in ps if p.strip())


def test_two_column_reading_order(pdfs):
    rep, chapters, _ = run(pdfs, "twocol")
    heads = ["".join(h.itertext()) for ch in chapters for h in ch.xpath("//x:h1|//x:h2", namespaces=NS)]
    assert heads[0].startswith("Column-Aware") and [h for h in heads[1:]] == [f"{i}. Section Number {i}" for i in range(1, 8)]
    full = " ".join(text(c) for c in chapters)
    # no interleaving: each paragraph is a clean repetition of the sample sentence
    for p in re.findall(r"The quick brown fox.*?(?=The quick brown fox|$)", full)[:5]:
        assert "lazy dog" in p


def test_dehyphenation_keeps_compounds(pdfs):
    _, chapters, _ = run(pdfs, "hyphen")
    t = text(chapters[0])
    assert "international hyphenation" in t and "well-known" in t


def test_footnotes_linked(pdfs):
    rep, chapters, _ = run(pdfs, "footnotes")
    ch = chapters[0]
    ref = ch.xpath("//x:a[@*[local-name()='type']='noteref']", namespaces=NS)
    note = ch.xpath("//x:aside[@*[local-name()='type']='footnote']", namespaces=NS)
    assert len(ref) == 1 and len(note) == 1 and rep.footnotes == 1
    assert ref[0].get("href") == "#" + note[0].get("id")
    assert "text of the first footnote" in "".join(note[0].itertext())


def test_page_selection_and_options(pdfs):
    assert parse_pages("2-3,5", 10) == [1, 2, 4] and parse_pages("8-", 10) == [7, 8, 9]
    rep, chapters, z = run(pdfs, "book", pages="1-2", cover="on")
    assert rep.pages == 2 and "OEBPS/images/cover.jpg" in z.namelist()


def test_ocr_fallback_on_scanned_pdf(pdfs):
    import pymupdf
    pytest.importorskip("rapidocr_onnxruntime")
    src = pymupdf.open(str(pdfs / "book.pdf"))
    out = pymupdf.open()
    pg = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
    pg.insert_image(pg.rect, stream=src[0].get_pixmap(dpi=200).tobytes("png"))
    out.save(str(pdfs / "scan.pdf"))
    rep, chapters, _ = run(pdfs, "scan", cover="off")
    assert rep.ocr_pages == 1 and "Beginnings" in text(chapters[0])


def test_display_equation_becomes_image_with_alt(pdfs):
    rep, chapters, z = run(pdfs, "equations", cover="off")
    eq = chapters[0].xpath("//x:img[@class='equation']", namespaces=NS)
    assert rep.equations == 1 and len(eq) == 1
    assert "=" in eq[0].get("alt") and "(1)" in eq[0].get("alt")
    paras = [" ".join(p.itertext()) for p in chapters[0].xpath("//x:p", namespaces=NS)]
    assert len(paras) == 2 and paras[0].endswith("as follows:") and paras[1].startswith("where x and y")
    assert not chapters[0].xpath("//x:sup|//x:sub", namespaces=NS)      # no script soup left in the text


def test_ml_layout_regions_drive_structure(pdfs, monkeypatch):
    """The model only supplies regions/labels; a stub detector exercises the whole ML code path offline."""
    import pymupdf
    from pdf2epub import ml

    page = pymupdf.open(str(pdfs / "equations.pdf"))[0]
    num = page.search_for("(1)")[0]
    words = page.get_text("words")
    top = min(w[1] for w in words)
    regions = {0: [
        ml.Region("text", (70, top - 2, 540, 735 - 0), 0.99, 1),                 # paragraph 1
        ml.Region("formula", (240, num.y0 - 4, 520, num.y1 + 4), 0.95, 2),
        ml.Region("text", (70, num.y1 + 20, 540, 780), 0.99, 3),                  # paragraph 2
    ]}
    regions[0][0] = ml.Region("text", (70, top - 2, 540, num.y0 - 20), 0.99, 1)

    class Stub:
        def __init__(self, **kw):
            pass

        def detect(self, path, first, last):
            return regions

    monkeypatch.setattr(ml, "LayoutDetector", Stub)
    rep, chapters, _ = run(pdfs, "equations", cover="off", ml="on")
    eq = chapters[0].xpath("//x:img[@class='equation']", namespaces=NS)
    assert len(eq) == 1 and "(1)" in eq[0].get("alt")
    paras = [" ".join(p.itertext()) for p in chapters[0].xpath("//x:p", namespaces=NS)]
    assert len(paras) == 2 and paras[1].startswith("where x and y")


def test_ml_section_header_label_makes_heading(pdfs, monkeypatch):
    import pymupdf
    from pdf2epub import ml

    page = pymupdf.open(str(pdfs / "hyphen.pdf"))[0]
    last = page.search_for("A new paragraph starts here")[0]
    regions = {0: [ml.Region("section_header", (60, last.y0 - 2, 540, last.y1 + 2), 0.95, 1),
                   ml.Region("text", (60, 60, 540, last.y0 - 8), 0.95, 2)]}

    class Stub:
        def __init__(self, **kw):
            pass

        def detect(self, path, first, last_):
            return regions

    monkeypatch.setattr(ml, "LayoutDetector", Stub)
    rep, chapters, _ = run(pdfs, "hyphen", cover="off", ml="on")
    heads = ["".join(h.itertext()) for h in chapters[0].xpath("//x:h1|//x:h2|//x:h3", namespaces=NS)]
    assert rep.heading_source == "layout-model" and heads == ["A new paragraph starts here and is separate."]


def test_latex_to_mathml_guards():
    pytest.importorskip("latex2mathml")
    from pdf2epub.mathml import equation_number, to_mathml

    good = to_mathml(r"\mathcal { L } ( \theta ) = \frac { 1 } { L } \sum _ { i = 1 } ^ { N } x _ { i }",
                     "L(θ) = 1 L Σ i=1 xi (4)", "images/e.png")
    assert good and "<mfrac>" in good and 'altimg="images/e.png"' in good
    etree.fromstring(good.encode())
    aligned = to_mathml(r"a & = b + c \\ d & = e", "a = b + c d = e")
    assert aligned and aligned.count("<mtr>") == 2          # latex2mathml alone emits an invalid '&' here
    etree.fromstring(aligned.encode())
    assert to_mathml(r"\varlaiceneq a + b", "a + b") is None                          # unknown command
    assert to_mathml(r"\text { that the generic accumulator } = x", "that the generic accumulator = x") is None  # prose
    assert to_mathml(r"x = y", "completely unrelated words") is None                  # does not match the PDF text
    assert equation_number("L(x) = y (12)") == "(12)" and equation_number("no number") is None


def test_ml_formula_latex_becomes_mathml(pdfs, monkeypatch):
    pytest.importorskip("latex2mathml")
    import pymupdf
    from pdf2epub import ml

    page = pymupdf.open(str(pdfs / "equations.pdf"))[0]
    num = page.search_for("(1)")[0]
    top = min(w[1] for w in page.get_text("words"))
    regions = {0: [ml.Region("text", (70, top - 2, 540, num.y0 - 20), 0.99, 1),
                   ml.Region("formula", (240, num.y0 - 4, 520, num.y1 + 4), 0.95, 2,
                             latex="x ^ { 2 } + y ^ { 2 } = z ^ { 2 }"),
                   ml.Region("text", (70, num.y1 + 20, 540, 780), 0.99, 3)]}

    class Stub:
        def __init__(self, **kw):
            assert kw.get("formulas") is True            # --mathml must request formula recognition

        def detect(self, path, first, last):
            return regions

    monkeypatch.setattr(ml, "LayoutDetector", Stub)
    rep, chapters, z = run(pdfs, "equations", cover="off", ml="on", mathml=True)
    M = {"m": "http://www.w3.org/1998/Math/MathML"}
    assert len(chapters[0].xpath("//m:math", namespaces=M)) == 1
    assert chapters[0].xpath("//x:span[@class='eqno']", namespaces=NS)[0].text == "(1)"
    assert 'properties="mathml"' in z.read("OEBPS/content.opf").decode()


def test_structured_table_rendering_spans_headers_and_gaps():
    from pdf2epub.analyze import DocStats
    from pdf2epub.model import Block, TableItem
    from pdf2epub.render import RenderCtx, render_blocks

    cells = [  # the model lists only non-empty cells; (r0,r1,c0,c1) with exclusive ends
        {"r0": 0, "r1": 2, "c0": 0, "c1": 1, "text": "Model", "head": True},
        {"r0": 0, "r1": 1, "c0": 1, "c1": 3, "text": "BLEU", "head": True},
        {"r0": 1, "r1": 2, "c0": 1, "c1": 2, "text": "EN-DE", "head": True},
        {"r0": 1, "r1": 2, "c0": 2, "c1": 3, "text": "EN-FR", "head": True},
        {"r0": 2, "r1": 3, "c0": 0, "c1": 1, "text": "ByteNet"},
        {"r0": 2, "r1": 3, "c0": 1, "c1": 2, "text": "23.75"},            # EN-FR cell is empty / unlisted
        {"r0": 3, "r1": 4, "c0": 0, "c1": 1, "text": "GNMT"},
        {"r0": 3, "r1": 4, "c0": 2, "c1": 3, "text": "39.92"},            # EN-DE cell is empty / unlisted
    ]
    t = TableItem(bbox=(0, 0, 1, 1), page=0, cells=cells, nrows=4, ncols=3)
    html = render_blocks([Block("table", 0, table=t)], RenderCtx(DocStats()), heading_id=lambda b: "")[0]
    doc = etree.fromstring(f'<div xmlns="http://www.w3.org/1999/xhtml">{html}</div>')
    assert doc.xpath("//x:th[@colspan='2' and .='BLEU']", namespaces=NS) and doc.xpath("//x:th[@rowspan='2' and .='Model']", namespaces=NS)
    assert len(doc.xpath("//x:thead/x:tr", namespaces=NS)) == 2 and len(doc.xpath("//x:tbody/x:tr", namespaces=NS)) == 2
    # every body row spans the full 3 columns, with empty cells filling the gaps
    for tr in doc.xpath("//x:tbody/x:tr", namespaces=NS):
        assert len(tr.xpath("x:td|x:th", namespaces=NS)) == 3
    row = [("".join(c.itertext())) for c in doc.xpath("//x:tbody/x:tr", namespaces=NS)[1].xpath("x:td", namespaces=NS)]
    assert row == ["GNMT", "", "39.92"]


def test_crossrefs(pdfs):
    """Cross-reference mentions in prose become internal links to their targets."""
    _, _, z = run(pdfs, "crossrefs")
    links: dict[str, str] = {}  # link text -> href
    targets: set[str] = set()
    for n in sorted(z.namelist()):
        if not n.startswith("OEBPS/chap"):
            continue
        doc = etree.fromstring(z.read(n))
        for a in doc.xpath("//x:a", namespaces=NS):
            href = a.get("href", "")
            text = "".join(a.itertext()).strip()
            if href and ("#xref" in href or "#h" in href):
                links[text] = href
        for el in doc.xpath("//*[@id]"):
            targets.add(el.get("id", ""))

    # Table 1 from caption should be linked in prose
    assert "Table 1" in links, f"Table 1 not linked; links: {links}"
    # the table anchor must actually exist
    anchor = links["Table 1"].split("#")[-1]
    assert anchor in targets, f"Table 1 anchor {anchor} not found; targets: {targets}"
