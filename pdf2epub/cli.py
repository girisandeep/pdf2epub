from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from . import __version__
from .converter import ConversionOptions, convert


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pdf2epub", description="Convert PDFs to structured EPUB 3 files.")
    ap.add_argument("inputs", nargs="+", help="PDF file(s) or directories containing PDFs")
    ap.add_argument("-o", "--output", help="output .epub (single input) or directory (multiple inputs)")
    ap.add_argument("--title"); ap.add_argument("--author"); ap.add_argument("--language", help="e.g. en, de, fr")
    ap.add_argument("--pages", help='page selection, e.g. "1-10,15,20-"')
    ap.add_argument("--no-images", action="store_true", help="omit images and figures")
    ap.add_argument("--cover", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--ocr", choices=["auto", "off", "force"], default="auto",
                    help="OCR scanned pages (Tesseract if installed, otherwise RapidOCR)")
    ap.add_argument("--ocr-lang", default="eng", help="Tesseract language code (default: eng)")
    ap.add_argument("--split-level", type=int, help="heading level that starts a new chapter (default: auto)")
    ap.add_argument("--keep-running-heads", action="store_true", help="do not strip repeating headers/footers")
    ap.add_argument("--no-footnotes", action="store_true")
    ap.add_argument("--ml", action="store_true", help="use the deep-learning layout model (needs: pip install docling)")
    ap.add_argument("--no-equations", action="store_true", help="leave equations as plain text instead of images")
    ap.add_argument("--ignore-outline", action="store_true", help="do not use the PDF bookmarks to find headings")
    ap.add_argument("--report", help="write a JSON conversion report to this path")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"pdf2epub {__version__}")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    files: list[str] = []
    for p in a.inputs:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".pdf"))
        else:
            files.append(p)
    if not files:
        print("no PDF files found", file=sys.stderr)
        return 2
    opts = ConversionOptions(title=a.title, author=a.author, language=a.language, pages=a.pages,
                             images=not a.no_images, cover=a.cover, ocr=a.ocr, ocr_lang=a.ocr_lang,
                             split_level=a.split_level, keep_running_heads=a.keep_running_heads,
                             footnotes=not a.no_footnotes, equations=not a.no_equations, ml="on" if a.ml else "off", use_outline=not a.ignore_outline)
    rc, reports = 0, []
    for f in files:
        if len(files) == 1 and a.output:
            out = a.output
        else:
            out = os.path.join(a.output or os.path.dirname(os.path.abspath(f)),
                               os.path.splitext(os.path.basename(f))[0] + ".epub")
        try:
            r = convert(f, out, opts)
        except Exception as e:  # keep batch going
            print(f"ERROR {f}: {e}", file=sys.stderr)
            rc = 1
            continue
        reports.append(r.to_dict())
        print(f"{f} -> {out}\n  {r.pages} pages, {r.chapters} chapters, {r.headings} headings ({r.heading_source}), "
              f"{r.lists} lists, {r.tables} tables, {r.images} images ({r.equations} equations), {r.footnotes} footnotes, "
              f"{r.running_lines_removed} header/footer lines removed, lang={r.language}"
              + (f", OCR on {r.ocr_pages} pages ({r.ocr_backend})" if r.ocr_pages else ""))
        for w in r.warnings:
            print(f"  warning: {w}")
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(reports, fh, indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
