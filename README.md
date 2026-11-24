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

## Deep-learning layout mode (`--ml`)

```bash
pip install -e '.[ml]'        # pulls docling + torch (~1.4 GB); weights come from Hugging Face on first use
pdf2epub paper.pdf --ml
```
Docling's RT-DETR layout model (`docling-layout-heron`, MIT) labels page regions: title, section header, text,
list item, caption, footnote, formula, picture, table, page header/footer. **Only regions and labels come from the
model; all text, fonts and geometry still come from the PDF**, so the model can never mis-read a character.
Predictions are cached in `~/.cache/pdf2epub/layout/`, so re-runs are fast.

How the labels are used, and how they are kept from doing harm:

* **Headings**: a model-labelled section header is a heading; level from numbering (`3.2.1` → h3) and style.
  A block the model calls plain text can still be promoted if it shares the exact style of ≥2 confirmed headings
  and comes after the body starts, which recovers misses without promoting title-page author lines.
  List items, captions and footnotes can never become headings (the heuristics did this: "• Holiday gifts…").
* **Paragraphs**: a model region edge only splits a paragraph when the geometry agrees (different label, extra
  space, or an early-ending line), so a boundary cutting through the middle of a paragraph leaves no orphans.
* **Headers/footers/footnotes**: regions confidently labelled as page furniture are removed; footnote regions
  feed the footnote linker.
* **Equations/figures**: formula and picture regions are rasterised, then the heuristics run on what is left
  (union of model and heuristics). Captions and boxed prose inside a picture region are never swallowed.
* **Tables**: the model confirms PyMuPDF's tables and rebuilds ones PyMuPDF missed; a ruled "table" is dropped
  only if the model is confident the area is prose.

Measured on the 10-document corpus (`tests/run_corpus.py`, heuristics → `--ml`): stray fragments `vae` 29%→14%,
`resnet` 14%→8%, `adam` 28%→20%, `bert` 10%→8%, `thinkpython` 16%→14%; math junk `vae` 19→8, `adam` 33→23;
`irs_pub` headings 100→85 by removing ~14 list items wrongly promoted to headings; recall equal or better
(−0.5 pt on `thinkpython`/−0.1 pt on `wiki_de`, which is page headers/footers now correctly removed).
Speed: roughly 1 s/page on Apple-silicon CPU/MPS for the model.

### Table structure (`--ml`, on by default; `--no-ml-tables` to skip)

Docling's TableFormer (accurate mode) recovers each table's grid, including merged cells (`rowspan`/`colspan`) and
header rows, which rule-less ("booktabs") tables never gave the heuristics. Structure comes from the model; **cell
text comes from the PDF's own words**. The model's table is used only if its cells account for ≥95% of the words
inside the region, otherwise the earlier PyMuPDF/word-geometry table is kept. Empty cells (which the model omits)
are re-inserted so columns stay aligned. Example: the Transformer paper's two-level-header results table now has
`BLEU` spanning EN-DE/EN-FR and each value under its own column, where before `24.6 39.92` were merged in one cell.

### Equations as MathML (`--ml --mathml`)

```bash
pdf2epub paper.pdf --ml --mathml
```
Docling's formula model (`CodeFormulaV2`) reads each formula region back into LaTeX, which is converted to MathML
(`<math altimg=… alttext=…>`, so readers without MathML support show the image). MathML equations reflow, scale and
are readable by screen readers. A recognised formula is only used when it passes guards, otherwise the equation stays
an image: the LaTeX must account for ≥60% of the symbols in the PDF's own text for that region, must contain no
commands the converter cannot translate, and no prose (a sign the model clipped surrounding text). Multi-line equations
become an `<mtable>`; equation numbers are kept. On the VAE paper's two equation pages 7 of 12 formulas passed, the rest fell back to images.

**Cost:** the formula model runs on CPU/CUDA only (not Apple MPS), about 30 s per formula on a laptop CPU, so a math-heavy
paper takes a long time the first run (results are cached per file). That is why it is opt-in. Set `PDF2EPUB_DEVICE=cuda` on a GPU machine.

## Regression corpus

`python tests/fetch_corpus.py` downloads 10 varied open documents (2-column papers, math-heavy papers, a
244-page book, government PDF, Wikipedia in two languages). `python tests/run_corpus.py` converts them all and
prints recall, precision, EPUB validity, stray-fragment %, leftover math junk and duplicate paragraphs, and
fails on regressions against `tests/corpus_baseline.json` (`--update-baseline` to accept new numbers;
`--no-equations` for before/after comparisons). With `--ml` it uses `tests/corpus_baseline_ml.json`.

## Known limits

* Display equations are images (not MathML, so they don't reflow or recolour); inline math is styled text.
* Algorithm/pseudocode boxes in papers are still flattened into ordinary paragraphs.
* Tables without any ruling lines and with no multi-line cells are not detected and stay as text.
* Reading order for heavily irregular layouts (magazines, sidebars) is a best effort.
* OCR quality depends on the engine; install Tesseract for the best results.

## License

AGPL-3.0-or-later (see `LICENSE`), matching the license of PyMuPDF, which this tool depends on.
