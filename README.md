# pdf2epub

Layout-aware PDF → EPUB 3 converter. It reconstructs *structure* (chapters, headings, paragraphs,
lists, tables, footnotes, figures) instead of dumping text page by page.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[ocr,dev]'
.venv/bin/pdf2epub paper.pdf -o paper.epub            # single file
.venv/bin/pdf2epub books/ -o out/ --report report.json # batch a folder
.venv/bin/pdf2epub scan.pdf --ocr force --ocr-lang eng # force OCR
.venv/bin/pdf2epub big.pdf --pages "1-40,55-" --split-level 2
```
`pdf2epub -h` lists all options (`--title/--author/--language`, `--cover`, `--no-images`,
`--keep-running-heads`, `--ignore-outline`, `--no-footnotes`, …).

## What it does

| Stage | Technique |
|---|---|
| Extraction | PyMuPDF spans with font/size/bold/italic/mono; geometric super/subscript; link annotations |
| Reading order | Column-aware XY-cut: finds gutters, treats titles/abstracts/figures that cross them as spanning separators, so columns are never interleaved |
| Cleanup | Repeating headers/footers/page numbers removed by position + normalised text (digits masked); repeating logos dropped |
| Paragraphs | Adaptive leading, first-line indents, hanging indents, and a "would the next word have fit?" test for deliberate line ends; paragraphs are re-joined across columns and pages |
| Hyphenation | Drops typographic hyphens, keeps real compounds (`well-known`) using the document's own vocabulary, a system dictionary, then a prefix list |
| Headings | If the PDF has bookmarks they are authoritative; their visual style (size/weight) is learned and applied to headings missing from the outline. Otherwise font-size/weight/caps clustering ranks h1–h4 |
| Lists | Bullets (incl. Symbol/Wingdings glyphs), `1.`/`a)`/`iv.` with sequence validation, nesting by indent |
| Tables | PyMuPDF table finder; rule-less ("booktabs") tables are rebuilt from word geometry; header row detection; captions attached |
| Figures | Embedded images (masks/CMYK handled), vector charts/diagrams rasterised, captions → `<figcaption>`; full-page backgrounds ignored |
| Footnotes | Footnote areas located by size/position and linked to body markers as EPUB 3 `noteref`/`footnote` asides |
| Printed TOC pages | Dot-leader contents pages become link lists pointing at the real headings |
| Equations | Display equations (math fonts / relations + scripts, fraction pieces and equation numbers grouped) are rendered as images sized in `em`, with the text kept as `alt`; inline math stays as styled text. `--no-equations` disables |
| Code | Monospace runs → `<pre><code>` with indentation preserved; monospace URLs/emails stay inline |
| Scans | Pages with no text layer are OCR'd: Tesseract if installed, otherwise RapidOCR (pure pip). Without either, the page image is embedded and a warning is printed |
| Packaging | EPUB 3 + NCX, nested nav built from headings, cover from page 1, metadata/title/language detection, sensible chapter sizes |

## Verification

`pytest tests` builds synthetic PDFs (ReportLab) and checks structure, reading order, dehyphenation,
footnotes, OCR and EPUB well-formedness (`tests/epub_check.py` verifies container, OPF, nav, anchors and
links — epubcheck itself needs Java, which isn't assumed). `tests/accuracy.py pdf epub` reports word
recall/precision against the PDF text layer. On the arXiv papers used during development
(*Attention Is All You Need*, GPT-3) recall is 99%+ after accounting for text inside rasterised figures.

## Regression corpus

`python tests/fetch_corpus.py` downloads 10 varied open documents (2-column papers, math-heavy papers, a
244-page book, government PDF, Wikipedia in two languages). `python tests/run_corpus.py` converts them all and
prints recall, precision, EPUB validity, stray-fragment %, leftover math junk and duplicate paragraphs, and
fails on regressions against `tests/corpus_baseline.json` (`--update-baseline` to accept new numbers;
`--no-equations` for before/after comparisons).

## Known limits

* Display equations are images (not MathML, so they don't reflow or recolour); inline math is styled text.
* Algorithm/pseudocode boxes in papers are still flattened into ordinary paragraphs.
* Tables without any ruling lines and with no multi-line cells are not detected and stay as text.
* Reading order for heavily irregular layouts (magazines, sidebars) is a best effort.
* OCR quality depends on the engine; install Tesseract for the best results.
* PyMuPDF is AGPL-licensed; check that is compatible with how you distribute this tool.
