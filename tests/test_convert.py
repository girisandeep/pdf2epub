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
