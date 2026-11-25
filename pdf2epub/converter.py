"""Orchestrates the PDF -> EPUB pipeline."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import pymupdf

from . import analyze as A
from .epub import Chapter, EpubBook, NavEntry, write_epub
from .extract import OcrEngine, extract_page
from .layout import reading_order
from .model import Block, FootnoteDef
from .mathml import equation_number
from .render import RenderCtx, inline_html, render_blocks
from .text import guess_language, norm_title, strip_numbering

log = logging.getLogger("pdf2epub")


@dataclass
class ConversionOptions:
    title: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    pages: Optional[str] = None              # e.g. "1-10,15"
    images: bool = True
    cover: str = "auto"                      # auto | on | off
    ocr: str = "auto"                        # auto | off | force
    ocr_lang: str = "eng"
    split_level: Optional[int] = None        # heading level that starts a new chapter file
    keep_running_heads: bool = False
    footnotes: bool = True
    mathml: bool = False                     # with ml: recognise formulas as LaTeX -> MathML (slow on CPU; image fallback)
    ml_tables: bool = True                   # with ml: TableFormer structure (spans, headers, rule-less tables)
    ml: str = "off"                          # off | on : deep-learning layout analysis (needs docling)
    equations: bool = True                   # render display equations as images (alt text kept)
    use_outline: bool = True
    max_chapter_chars: int = 350_000
    nav_depth: int = 3
    min_chapter_chars: int = 2500   # fold tiny sections into the following file


@dataclass
class ConversionReport:
    source: str = ""
    output: str = ""
    pages: int = 0
    chapters: int = 0
    headings: int = 0
    heading_source: str = ""
    lists: int = 0
    tables: int = 0
    images: int = 0
    footnotes: int = 0
    equations: int = 0
    running_lines_removed: int = 0
    merged_paragraphs: int = 0
    ocr_pages: int = 0
    ocr_backend: str = ""
    language: str = ""
    title: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_pages(spec: Optional[str], n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d*)-(\d*)", part)
        if m:
            a = int(m.group(1) or 1)
            b = int(m.group(2) or n)
            out.extend(range(a - 1, min(b, n)))
        elif part.isdigit():
            out.append(int(part) - 1)
        else:
            raise ValueError(f"bad page spec: {part!r}")
    return sorted({p for p in out if 0 <= p < n})


_BAD_TITLE = re.compile(r"^(untitled|microsoft word|\.?pdf|slide \d+|document\d*|\s*)$|\.(docx?|pdf|indd|tex|pptx?)$", re.I)


_INSTITUTION_RE = re.compile(
    r"university|universiteit|college|institute|institute|department|laboratory|"
    r"school|research|group|center|centre|press|foundation|"
    r"microsoft|google|facebook|meta ai|openai|deepmind|amazon|ibm|"
    r"@|\.edu|\.com|\.org", re.I)

_AUTHOR_JUNK = re.compile(r"[∗†‡§¶\d,]+$")

_SECTION_START = re.compile(
    r"^(abstract|introduction|contents|1\s|1\.\s|§1)", re.I)

_META_AUTHOR_BAD = re.compile(r"[:\\{}]|^\d|^[A-Z]{2,}(:[A-Z]{2,})+$")


def _valid_meta_author(s: str) -> bool:
    if not s or len(s) < 3:
        return False
    if _META_AUTHOR_BAD.search(s):
        return False
    if not any(c.isalpha() for c in s):
        return False
    return True


def _extract_authors(blocks: list[Block], body_size: float) -> Optional[str]:
    """Extract author names from the first page by looking between the title and Abstract."""
    if not blocks:
        return None
    first_page = blocks[0].page
    title_idx = None
    for i, b in enumerate(blocks):
        if b.page != first_page:
            break
        if b.kind == "heading" and b.level == 1:
            title_idx = i
            break
    if title_idx is None:
        best_i, best_sz = None, body_size * 1.3
        for i, b in enumerate(blocks):
            if b.page != first_page:
                break
            if b.lines and len(b.lines) <= 3:
                sz = max(l.size for l in b.lines)
                if sz > best_sz:
                    best_i, best_sz = i, sz
        title_idx = best_i
    if title_idx is None:
        return None
    end_idx = None
    for i in range(title_idx + 1, len(blocks)):
        b = blocks[i]
        if b.page != first_page:
            end_idx = i
            break
        t = b.text.strip()
        if _SECTION_START.match(t):
            end_idx = i
            break
        if b.kind == "para" and len(t) >= 120:
            end_idx = i
            break
    if end_idx is None:
        end_idx = min(title_idx + 25, len(blocks))
    authors = []
    for i in range(title_idx + 1, end_idx):
        b = blocks[i]
        if b.kind not in ("para", "heading") or not b.lines:
            continue
        t = b.text.strip()
        if len(t) > 60 or len(t) < 3:
            continue
        if len(b.lines) > 2:
            continue
        sz = max(l.size for l in b.lines)
        if sz < body_size * 0.85:
            continue
        if _INSTITUTION_RE.search(t):
            continue
        if "@" in t or t.startswith("http"):
            continue
        name = _AUTHOR_JUNK.sub("", t).strip().rstrip(",").strip()
        words = name.split()
        if len(words) < 2 or len(words) > 5:
            continue
        if not all(w[0].isupper() or w in ("de", "di", "van", "von", "der", "del", "la", "le", "al", "el",
                                            "and", "et", "und") for w in words):
            continue
        authors.append(name)
    if not authors:
        authors = _authors_from_table(blocks, title_idx, end_idx)
    if not authors:
        return None
    return ", ".join(authors[:8])


_NAME_PART = re.compile(r"^[A-Z][a-z]+\.?$|^[A-Z]\.$|^(?:de|di|van|von|der|del|la|le|al|el)$")
_MARKER_SPLIT = re.compile(r"[∗†‡§¶]+")


def _authors_from_table(blocks: list[Block], title_idx: int, end_idx: int) -> list[str]:
    """Extract author names from a table block between title and Abstract (e.g. GPT-3)."""
    for i in range(title_idx + 1, end_idx):
        b = blocks[i]
        if b.kind != "table" or not b.table or not b.table.rows:
            continue
        names = []
        for row in b.table.rows:
            for cell in row:
                cell = cell.strip()
                if not cell:
                    continue
                for fragment in _MARKER_SPLIT.split(cell):
                    fragment = fragment.strip()
                    if not fragment:
                        continue
                    if _INSTITUTION_RE.search(fragment):
                        continue
                    words = fragment.split()
                    if 2 <= len(words) <= 5 and all(_NAME_PART.match(w) for w in words):
                        names.append(fragment)
        if names:
            return names
    return []


def _clean_title(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t = t.strip()
    t = re.sub(r"^Microsoft Word - ", "", t)
    return None if _BAD_TITLE.search(t) else t


def _heading_plain(b: Block, ctx: RenderCtx) -> str:
    if b.synthetic_text is not None:
        return b.synthetic_text.strip()
    html = inline_html(b.lines, ctx, strip_bold=True)
    html = re.sub(r"<[^>]+>", "", html)
    return html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()


_strip_numbering = strip_numbering


# ---- cross-reference targets -------------------------------------------------

_CAPTION_NUM = re.compile(
    r"^\s*(Fig(?:ure)?s?\.?|Table|Tab\.|Algorithm|Listing|Scheme|Exhibit)"
    r"\s*(\d+(?:\.\d+)*[a-z]?)\s*[.:)\s-]", re.I)

_THEOREM_NUM = re.compile(
    r"^\s*(Theorem|Lemma|Corollary|Proposition|Definition|Remark|Conjecture|Claim|Example|Fact)"
    r"\s+(\d+(?:\.\d+)*[a-z]?)\s*[.:(]", re.I)

_SECTION_NUM = re.compile(r"^(\d+(?:\.\d+)*[a-z]?)(?:\s|[.)])")

_SECTION_WORD = re.compile(
    r"^(?:Section|Appendix|Chapter)\s+(\d+(?:\.\d+)*[a-z]?)\s", re.I)


def _canon(kind: str) -> str:
    k = kind.rstrip(".").lower()
    if k.startswith("fig"):
        return "figure"
    if k in ("tab",):
        return "table"
    return k


def _build_xref_targets(blocks: list[Block], chapters: list[dict], ctx: RenderCtx) -> None:
    """Scan blocks for figures, tables, equations, algorithms and register xref targets."""
    # Map each block id to its chapter file
    block_file: dict[int, str] = {}
    for ch in chapters:
        for b in ch["blocks"]:
            block_file[id(b)] = ch["file"]

    xid = 0  # counter for unique xref anchors
    for b in blocks:
        file = block_file.get(id(b), "")
        if not file:
            continue

        # --- figures and tables (from captions) ---
        if b.kind in ("image", "table") and b.caption:
            caption_text = " ".join(s.text for l in b.caption for s in l.spans)
            m = _CAPTION_NUM.match(caption_text)
            if m:
                kind = _canon(m.group(1))
                num = m.group(2)
                xid += 1
                b.anchor = b.anchor or f"xref{xid}"
                ctx.xref_targets.setdefault(f"{kind} {num}", (file, b.anchor))

        # --- equations (from equation number in alt text) ---
        if b.kind == "image" and b.image and b.image.kind == "equation":
            eqno = equation_number(b.image.alt or "")
            if eqno:
                num = eqno.strip("()")
                xid += 1
                b.anchor = b.anchor or f"xref{xid}"
                ctx.xref_targets.setdefault(f"equation {num}", (file, b.anchor))
                ctx.xref_targets.setdefault(f"eq {num}", (file, b.anchor))

        # --- algorithms (from first line or caption) ---
        if b.kind == "algorithm" and b.lines:
            # Look for "Algorithm N" in the first line or in an adjacent heading
            first_text = b.lines[0].text.strip()
            m = re.match(r"Algorithm\s+(\d+)", first_text, re.I)
            if m:
                xid += 1
                b.anchor = b.anchor or f"xref{xid}"
                ctx.xref_targets.setdefault(f"algorithm {m.group(1)}", (file, b.anchor))

        # --- standalone caption paragraphs (not attached to image/table) ---
        if b.kind == "para" and b.lines:
            first_text = " ".join(s.text for s in b.lines[0].spans).strip()
            m = _CAPTION_NUM.match(first_text)
            if m:
                kind = _canon(m.group(1))
                num = m.group(2)
                key = f"{kind} {num}"
                if key not in ctx.xref_targets:
                    xid += 1
                    b.anchor = b.anchor or f"xref{xid}"
                    ctx.xref_targets.setdefault(key, (file, b.anchor))

        # --- theorems, lemmas, definitions (from paragraph start) ---
        if b.kind == "para" and b.lines:
            first_text = " ".join(s.text for s in b.lines[0].spans).strip()
            m = _THEOREM_NUM.match(first_text)
            if m:
                kind = m.group(1).lower()
                num = m.group(2)
                xid += 1
                b.anchor = b.anchor or f"xref{xid}"
                ctx.xref_targets.setdefault(f"{kind} {num}", (file, b.anchor))

        # --- bibliography entries ---
        if b.kind == "li" and b.anchor and b.anchor.startswith("bib"):
            cite_key = b.anchor.rsplit("-", 1)[-1]
            ctx.xref_targets.setdefault(f"cite {cite_key}", (file, b.anchor))

        # --- headings with section numbers ---
        if b.kind == "heading" and b.anchor:
            text = b.synthetic_text if b.synthetic_text else " ".join(
                s.text for l in b.lines for s in l.spans)
            text = text.strip()
            m = _SECTION_NUM.match(text)
            if m:
                num = m.group(1)
                ctx.xref_targets.setdefault(f"section {num}", (file, b.anchor))
                ctx.xref_targets.setdefault(f"subsection {num}", (file, b.anchor))
            m = _SECTION_WORD.match(text)
            if m:
                num = m.group(1)
                ctx.xref_targets.setdefault(f"section {num}", (file, b.anchor))
                ctx.xref_targets.setdefault(f"appendix {num}", (file, b.anchor))


def _split_chapters(blocks: list[Block], opts: ConversionOptions, ctx: RenderCtx):
    levels = [b.level for b in blocks if b.kind == "heading"]
    split = opts.split_level
    if not split:
        for lv in range(1, 7):
            if levels.count(lv) >= 2:
                split = lv
                break
    chapters: list[dict] = []
    cur: dict | None = None

    def new(title: str, head: Optional[Block]):
        nonlocal cur
        cur = {"title": title, "blocks": [], "head": head, "chars": 0}
        chapters.append(cur)

    for b in blocks:
        if b.kind == "heading" and split and b.level <= split:
            new(_heading_plain(b, ctx) or "Untitled", b)
        elif cur is None:
            new("", None)
        cur["blocks"].append(b)
        if b.kind == "toc":
            cur["chars"] += sum(len(t) for t, _ in b.entries)
        elif b.kind != "heading":
            cur["chars"] += len(b.text)
        # keep chapter files a sane size: break on subheadings, or on any paragraph when huge
        if cur["chars"] > opts.max_chapter_chars and b.kind in ("para", "li"):
            cur = None
            new("", None)
            chapters[-1]["cont"] = True
    chapters = [c for c in chapters if c["blocks"]]
    merged: list[dict] = []
    for c in chapters:
        if merged and merged[-1]["chars"] < opts.min_chapter_chars and not c.get("cont"):
            prev = merged[-1]
            if prev["chars"] == 0 and c["chars"] > 0:
                c["blocks"] = prev["blocks"] + c["blocks"]
                c["chars"] += prev["chars"]
                merged[-1] = c
            else:
                prev["blocks"] += c["blocks"]
                prev["chars"] += c["chars"]
            continue
        merged.append(c)
    if len(merged) > 1 and merged[-1]["chars"] < opts.min_chapter_chars // 2 and not merged[-1].get("cont"):
        last = merged.pop()
        merged[-1]["blocks"] += last["blocks"]
        merged[-1]["chars"] += last["chars"]
    return merged, split


def _sample_body_size(doc, selected: list[int]) -> float:
    from collections import Counter
    c: Counter = Counter()
    step = max(1, len(selected) // 12)
    for pno in selected[::step]:
        for b in doc[pno].get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    c[round(s["size"] * 2) / 2] += len(s["text"].strip())
    return max(c, key=c.get) if c else 10.0


def convert(src: str, dst: Optional[str] = None, options: Optional[ConversionOptions] = None) -> ConversionReport:
    opts = options or ConversionOptions()
    dst = dst or os.path.splitext(src)[0] + ".epub"
    rep = ConversionReport(source=src, output=dst)
    doc = pymupdf.open(src)
    if doc.needs_pass:
        raise PermissionError("PDF is password protected")
    selected = parse_pages(opts.pages, doc.page_count)
    if not selected:
        raise ValueError("no pages selected")
    rep.pages = len(selected)
    counters = A.Counters()

    # ---- 1. extraction -------------------------------------------------------
    ocr = OcrEngine(opts.ocr_lang) if opts.ocr != "off" else None
    pages = []
    regions = None
    if opts.ml != "off":
        from .ml import LayoutDetector
        regions = LayoutDetector(formulas=opts.mathml and opts.equations, tables=opts.ml_tables).detect(src, min(selected), max(selected))
    body_size = _sample_body_size(doc, selected)
    for pno in selected:
        pd = extract_page(doc, pno, with_images=opts.images, ocr=ocr, ocr_mode=opts.ocr,
                          body_size=body_size, equations=opts.equations,
                          regions=(regions.get(pno, []) if regions is not None else None))
        pages.append(pd)
        if pd.ocr:
            rep.ocr_pages += 1
        elif any(i.kind == "page" for i in pd.images):
            rep.warnings.append(f"page {pno + 1} appears scanned and no OCR engine is available; page image embedded")
    if ocr and rep.ocr_pages:
        rep.ocr_backend = ocr.backend or ""

    # ---- 2. cover + metadata -------------------------------------------------
    cover_bytes = None
    first = pages[0]
    first_words = sum(len(l.text.split()) for l in first.lines)
    want_cover = opts.cover == "on" or (opts.cover == "auto" and (first_words <= 40 or
                                        any(i.width * i.height > 0.5 * first.width * first.height for i in first.images)))
    if want_cover and opts.cover != "off":
        pix = doc[first.number].get_pixmap(dpi=130, colorspace=pymupdf.csRGB)
        cover_bytes = pix.tobytes("jpeg", jpg_quality=88)
        if first_words <= 40 and not first.ocr and len(pages) > 1:
            pages = pages[1:]          # the cover already shows this page
    meta = doc.metadata or {}
    title = opts.title or _clean_title(meta.get("title"))
    raw_author = (meta.get("author") or "").strip()
    if raw_author and not _valid_meta_author(raw_author):
        raw_author = ""
    author = opts.author or raw_author or None

    # ---- 3. statistics & cleanup --------------------------------------------
    stats = A.compute_stats(pages)
    A.strip_running(pages, counters, keep=opts.keep_running_heads)
    fn_defs: dict[tuple[int, str], FootnoteDef] = {}
    if opts.footnotes:
        prev_last: FootnoteDef | None = None
        for p in pages:
            page_defs = A.extract_footnotes(p, stats, prev_last=prev_last)
            for d in page_defs:
                fn_defs[(p.number, d.marker)] = d
            if page_defs:
                prev_last = page_defs[-1]
        counters.footnotes = len(fn_defs)

    # ---- 4. reading order & segmentation -------------------------------------
    blocks: list[Block] = []
    region = 0
    for p in pages:
        items = [*p.lines, *p.images, *p.tables]
        ordered, region = reading_order(items, region)
        blocks += A.segment_page(ordered, p.number, stats)
    counters.tables = sum(1 for b in blocks if b.kind == "table")
    counters.images = sum(1 for b in blocks if b.kind == "image")

    # ---- 5. structure ---------------------------------------------------------
    toc = []
    if opts.use_outline:
        try:
            sel = set(selected)
            toc = [t for t in doc.get_toc(simple=False) if t[2] - 1 in sel and t[1].strip()]
        except Exception:
            toc = []
    blocks = A.classify_headings(blocks, toc, stats, counters)
    detected_title = A.detect_title(blocks, stats)
    blocks = A.detect_index(blocks)
    blocks = A.detect_separators(blocks)
    blocks = A.demote_pseudo_code(blocks)
    blocks = A.detect_algorithms(blocks)
    blocks = A.merge_continuations(blocks, counters)
    if not author:
        author = _extract_authors(blocks, stats.body_size)
    blocks = A.merge_title_page(blocks)
    blocks = A.format_bibliography(blocks)
    blocks = A.build_lists(blocks, counters)
    blocks = A.detect_naked_lists(blocks, counters)
    blocks = A.detect_definition_lists(blocks, counters)
    blocks = A._detect_heading_toc(blocks)
    counters.headings = sum(1 for b in blocks if b.kind == "heading")
    blocks = A.detect_toc(blocks, counters)
    blocks = A.attach_captions(blocks, body_size=stats.body_size)
    blocks = A.merge_continuations(blocks, counters)
    blocks = [b for b in blocks if not (b.kind == "para" and not b.text.strip())]
    blocks = [b for b in blocks if not (
        b.kind == "para" and len(b.text.strip()) <= 3
        and b.lines and max(l.size for l in b.lines) <= stats.body_size * 0.9
    )]

    # ---- 6. metadata fallbacks -------------------------------------------------
    if not title and detected_title:
        title = detected_title
    if not title:
        h = next((b for b in blocks if b.kind == "heading" and b.page == blocks[0].page), None)
        if h is None:
            h = next((b for b in blocks if b.kind == "heading"), None)
        ctx0 = RenderCtx(stats)
        title = _heading_plain(h, ctx0) if h is not None else None
    title = title or os.path.splitext(os.path.basename(src))[0]
    sample = " ".join(b.text for b in blocks[:200] if b.kind == "para")
    lang = opts.language
    if not lang:
        try:
            lang = doc.xref_get_key(doc.pdf_catalog(), "Lang")[1].strip("() ").split("-")[0].lower() or None
        except Exception:
            lang = None
    lang = lang if lang and re.fullmatch(r"[a-z]{2,3}", lang) else guess_language(sample)
    rep.title, rep.language = title, lang

    # ---- 7. chapters, ids, images ----------------------------------------------
    ctx = RenderCtx(stats, footnotes=fn_defs)
    chapters, split = _split_chapters(blocks, opts, ctx)
    book = EpubBook(title=title, language=lang, author=author, cover=cover_bytes)

    n_img = 0
    for b in blocks:
        if b.kind == "image" and b.image.key not in ctx.image_href:
            n_img += 1
            ext = b.image.ext
            href = f"images/img{n_img:04d}.{ext}"
            ctx.image_href[b.image.key] = href
            book.images[href] = (b.image.data, "image/jpeg" if ext == "jpg" else "image/png")
    rep.images = len(book.images)
    rep.equations = sum(1 for b in blocks if b.kind == "image" and b.image.kind == "equation")

    hid = 0
    nav_raw: list[tuple[int, str, str, str]] = []     # level, title, file, id
    for ci, ch in enumerate(chapters):
        ch["file"] = f"chap{ci + 1:03d}.xhtml"
        for b in ch["blocks"]:
            if b.kind == "heading":
                hid += 1
                b.anchor = f"h{hid}"
                text = _heading_plain(b, ctx)
                if text:
                    nav_raw.append((b.level, text, ch["file"], b.anchor))
                    ctx.heading_targets.setdefault(norm_title(text), (ch["file"], b.anchor))
                    ctx.heading_targets.setdefault(_strip_numbering(text), (ch["file"], b.anchor))

    _build_xref_targets(blocks, chapters, ctx)

    # ---- 8. render ----------------------------------------------------------------
    for ch in chapters:
        ctx.current_file = ch["file"]
        body = render_blocks(ch["blocks"], ctx, heading_id=lambda b: b.anchor)
        ch["html"] = "\n".join(body)
    for (pg, mk), d in fn_defs.items():
        if not d.used:
            rep.warnings.append(f"footnote {mk} on page {pg + 1}: reference not located in the body text (note dropped)")

    base_title = title
    for ch in chapters:
        if not ch["title"]:
            ch["title"] = base_title if ch is chapters[0] else f"{base_title} (continued)"
        book.chapters.append(Chapter(ch["file"], ch["title"], ch["html"] or "<p></p>"))

    # navigation: headings down to nav_depth, relative to the shallowest level used
    if nav_raw:
        top = min(l for l, *_ in nav_raw)
        for lv, text, f, hidn in nav_raw:
            if lv - top + 1 <= opts.nav_depth:
                book.nav.append(NavEntry(lv - top + 1, text, f"{f}#{hidn}"))
        if chapters and not chapters[0]["head"] and chapters[0]["blocks"] and (
                not book.nav or not book.nav[0].href.startswith(chapters[0]["file"])):
            book.nav.insert(0, NavEntry(1, chapters[0]["title"], chapters[0]["file"]))
    rep.chapters = len(book.chapters)
    rep.headings = counters.headings
    rep.heading_source = counters.heading_source
    rep.lists = counters.lists
    rep.tables = counters.tables
    rep.footnotes = sum(1 for d in fn_defs.values() if d.used)
    rep.running_lines_removed = counters.running_removed
    rep.merged_paragraphs = counters.merged_paragraphs
    if not any(b.kind == "heading" for b in blocks):
        rep.warnings.append("no headings detected; chapters were split by size")
    write_epub(book, dst)
    return rep
