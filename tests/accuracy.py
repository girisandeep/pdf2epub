"""Word-level recall/precision of an EPUB against the PDF's own text layer (order-insensitive)."""
import re, sys, zipfile
from collections import Counter

import pymupdf
from lxml import etree

NS = {"x": "http://www.w3.org/1999/xhtml"}


def toks(s):
    return Counter(re.findall(r"[^\W_]+", s.lower().replace("-", "")))


def main(pdf, epub):
    ref = toks(" ".join(p.get_text() for p in pymupdf.open(pdf)))
    z = zipfile.ZipFile(epub)
    txt = " ".join(" ".join(etree.fromstring(z.read(n)).xpath("//x:body//text()", namespaces=NS))
                   for n in sorted(z.namelist()) if n.startswith("OEBPS/chap"))
    got = toks(txt)
    common = sum((ref & got).values())
    print(f"{pdf.split('/')[-1]}: recall={common / sum(ref.values()):.4f} precision={common / sum(got.values()):.4f}"
          f"  (pdf words {sum(ref.values())}, epub words {sum(got.values())})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
