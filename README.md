# pdf2epub

Layout-aware PDF → EPUB 3 / HTML converter. It reconstructs *structure* (chapters, headings, paragraphs,
lists, tables, footnotes, figures) instead of dumping text page by page.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[ocr,dev]'
.venv/bin/pdf2epub paper.pdf -o paper.epub            # single file → EPUB
.venv/bin/pdf2epub paper.pdf -o paper.html            # single file → HTML
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
| Cross-references | Mentions of "Figure 1", "Table 2", "Section 3.1", "Eq. (4)", "Theorem 1", "Lemma 2" etc. in prose become clickable links to the actual figure, table, heading, equation or theorem — across chapters. Plural forms ("Tables 1, 2") are handled |
| Bibliography | `[N]` numeric and `[KEY]` alphabetic citation entries after a "References" heading are reformatted as a list; continuation lines are merged. Inline citations like `[33]` or `[1, 2, 3]` become clickable links to the bibliography entry |
| Code | Monospace runs → `<pre><code>` with indentation preserved; monospace URLs/emails stay inline |
| Scans | Pages with no text layer are OCR'd: Tesseract if installed, otherwise RapidOCR (pure pip). Without either, the page image is embedded and a warning is printed |
| Packaging | EPUB 3 + NCX, nested nav built from headings, cover from page 1, metadata/title/language detection, sensible chapter sizes |
| HTML output | Same pipeline, single self-contained HTML file with inline CSS (dark-mode), clickable TOC, and images in a sibling folder — use `-o file.html` |

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
* **Algorithms**: regions the model labels as `code` but that use a proportional font are rendered as styled
  algorithm/pseudocode blocks instead of individual paragraphs, cutting fragment and mathjunk counts.

Measured on the 10-document corpus (`tests/run_corpus.py`, heuristics → `--ml`): stray fragments `vae` 12%→4%,
`adam` 16%→12%, `attention` 2%→1%, `resnet` 9%→3%, `bert` 10%→6%, `thinkpython` 3%→2%, `wiki_de` 5%→6%;
math junk `vae` 11→6, `adam` 8→12; `irs_pub` headings 85→85 (heuristic now matches ML by excluding bullet list
items from heading candidates); recall equal or better (−0.5 pt on `thinkpython`/−0.1 pt on `wiki_de`, which is
page headers/footers now correctly removed).
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

## Benchmark: pdf2epub vs Calibre 9.19

Measured on 10 open documents (arXiv papers, a 244-page programming book, a government tax guide,
Wikipedia articles in English and German). Calibre is the most widely-used open-source converter;
Pandoc cannot read PDF files. All metrics use the same evaluator (`tests/benchmark.py`).

| Metric | pdf2epub | Calibre 9.19 |
|---|---|---|
| **Avg Recall** (text recovery) | 0.947 | 0.974 |
| **Avg Precision** (no junk text) | **0.989** | 0.958 |
| **Valid EPUBs** | **10/10** | 0/10 |
| **Headings detected** | **556** | 39 |
| **Tables detected** | **94** | 0 |
| **Images extracted** | **326** | 166 |
| **Avg Fragment %** (stray short lines) | **8.9%** | 17.3% |
| **TOC Navigation** | **10/10** | 0/10 |
| **Page-list Navigation** | **10/10** | 0/10 |
| **Avg Metadata Fields** (of 7) | **5.7** | 2.0 |

Calibre's higher recall comes from retaining running headers, footers, and page numbers that
pdf2epub strips — this boosts word count but lowers precision and adds noise. Calibre's output
treats every PDF line as a separate `<p>` tag with no paragraph merging, produces no headings from
most PDFs, cannot detect tables, and fails EPUB structural validation on all 10 documents.

Per-document results:

| Document | Pages | Tool | Recall | Precision | Valid | Headings | Tables | Images | Frag% |
|---|---|---|---|---|---|---|---|---|---|
| adam | 15 | pdf2epub | 0.776 | **0.966** | **Yes** | **21** | 0 | **65** | **3.8** |
| | | Calibre | **0.890** | 0.809 | No | 0 | 0 | 0 | 28.9 |
| attention | 15 | pdf2epub | 0.933 | **0.986** | **Yes** | **26** | **4** | **17** | **0.0** |
| | | Calibre | **0.982** | 0.964 | No | 2 | 0 | 3 | 13.6 |
| bert | 16 | pdf2epub | 0.962 | **0.988** | **Yes** | **33** | **6** | 21 | **5.5** |
| | | Calibre | **0.975** | 0.951 | No | 0 | 0 | **38** | 16.1 |
| gpt3 | 75 | pdf2epub | **0.996** | **0.997** | **Yes** | **55** | **54** | **86** | **5.1** |
| | | Calibre | 0.994 | 0.995 | No | 2 | 0 | 83 | 13.2 |
| irs_pub | 36 | pdf2epub | **0.991** | **1.000** | **Yes** | **51** | **4** | 2 | 15.4 |
| | | Calibre | 0.985 | 0.985 | No | 3 | 0 | 3 | **14.0** |
| resnet | 12 | pdf2epub | 0.980 | **0.995** | **Yes** | **22** | **16** | **9** | **0.0** |
| | | Calibre | **0.993** | 0.987 | No | 2 | 0 | 0 | 15.3 |
| thinkpython | 244 | pdf2epub | 0.979 | **0.999** | **Yes** | **249** | **8** | **28** | 40.2 |
| | | Calibre | **0.986** | 0.998 | No | 28 | 0 | 0 | **32.5** |
| vae | 14 | pdf2epub | 0.855 | **0.964** | **Yes** | **23** | 0 | **68** | **3.2** |
| | | Calibre | **0.938** | 0.901 | No | 2 | 0 | 6 | 24.9 |
| wiki_de | 19 | pdf2epub | **0.998** | **0.998** | **Yes** | **33** | 0 | 13 | 9.8 |
| | | Calibre | 0.996 | 0.993 | No | 0 | 0 | 13 | **9.6** |
| wiki_en | 42 | pdf2epub | 0.999 | **0.998** | **Yes** | **43** | **2** | 17 | 5.9 |
| | | Calibre | **0.999** | 0.998 | No | 0 | 0 | **20** | **4.9** |

Reproduce with `python tests/benchmark.py` (requires Calibre installed).

## Known limits

* Display equations are images by default (not MathML); use `--ml --mathml` for reflowable MathML where the formula model is confident. Inline math is styled text.
* Tables without any ruling lines and with no multi-line cells are not detected and stay as text.
* Reading order for heavily irregular layouts (magazines, sidebars) is a best effort.
* OCR quality depends on the engine; install Tesseract for the best results.

## License

AGPL-3.0-or-later (see `LICENSE`), matching the license of PyMuPDF, which this tool depends on.
