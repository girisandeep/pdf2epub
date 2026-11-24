"""Page-level extraction: text lines with style, raster images, vector figures, tables, OCR."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Optional

import pymupdf

from .model import ImageItem, Line, PageData, Span, TableItem
from .text import clean

log = logging.getLogger("pdf2epub")

_BOLD = re.compile(r"bold|black|heavy|semibold|demi|extrabold", re.I)
_ITALIC = re.compile(r"italic|oblique|ital\b|slanted", re.I)
_MONO = re.compile(r"mono|courier|consolas|menlo|typewriter|lucida console|inconsolata|fixed|cmtt|source code", re.I)


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def _area(r) -> float:
    return max(r[2] - r[0], 0) * max(r[3] - r[1], 0)


def _inside(inner, outer, frac=0.6) -> bool:
    a = _area(inner)
    return a > 0 and _overlap(inner, outer) / a >= frac


# --------------------------------------------------------------------------- OCR
class OcrEngine:
    """Tesseract (via PyMuPDF) if present, else RapidOCR (pure pip), else none."""

    def __init__(self, lang: str = "eng", dpi: int = 300):
        self.lang, self.dpi = lang, dpi
        self._probed = False
        self._backend: Optional[str] = None
        self._rapid = None

    def _probe(self) -> None:
        if self._probed:
            return
        self._probed = True
        try:
            if pymupdf.get_tessdata():
                self._backend = "tesseract"
                return
        except Exception:
            pass
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._rapid = RapidOCR()
            self._backend = "rapidocr"
        except Exception:
            self._backend = None

    @property
    def backend(self) -> Optional[str]:
        self._probe()
        return self._backend

    @property
    def available(self) -> bool:
        return self.backend is not None

    def page_lines(self, page: pymupdf.Page) -> list[Line]:
        if self.backend == "tesseract":
            tp = page.get_textpage_ocr(language=self.lang, dpi=self.dpi, full=True)
            return _lines_from_dict(page.get_text("dict", textpage=tp,
                                    flags=pymupdf.TEXT_PRESERVE_WHITESPACE), page.number)
        zoom = self.dpi / 72
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB)
        res, _ = self._rapid(pix.tobytes("png"))
        out: list[Line] = []
        for box, text, conf in res or []:
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            bb = (min(xs) / zoom, min(ys) / zoom, max(xs) / zoom, max(ys) / zoom)
            h = bb[3] - bb[1]
            sp = Span(clean(text), max(h * 0.78, 4.0), bb, origin_y=bb[3] - h * 0.2)
            out.append(Line(bbox=bb, page=page.number, spans=[sp]))
        # box heights jitter with ascenders/descenders: snap near-median sizes to the median
        if out:
            sizes = sorted(l.spans[0].size for l in out)
            med = sizes[len(sizes) // 2]
            for l in out:
                sp = l.spans[0]
                if 0.72 * med <= sp.size <= 1.3 * med:
                    sp.size = med
        return out


# ------------------------------------------------------------------ line building
def _span_from_dict(s: dict) -> Optional[Span]:
    text = clean(s["text"])
    if not text:
        return None
    font = s.get("font", "")
    flags = s.get("flags", 0)
    return Span(
        text=text, size=float(s["size"]), bbox=tuple(s["bbox"]), origin_y=float(s["origin"][1]),
        bold=bool(flags & 16) or bool(_BOLD.search(font)),
        italic=bool(flags & 2) or bool(_ITALIC.search(font)),
        mono=bool(flags & 8) or bool(_MONO.search(font)),
        font=font,      # MuPDF's superscript flag fires after any lowered glyph, so scripts are found geometrically
    )


def _lines_from_dict(d: dict, pno: int) -> list[Line]:
    lines: list[Line] = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            dx, dy = l.get("dir", (1, 0))
            if l.get("wmode", 0) or abs(dy) > 0.2 or dx < 0.8:
                continue  # vertical / rotated text (watermarks, margin stamps)
            spans = [sp for sp in (_span_from_dict(s) for s in l["spans"]) if sp]
            if not spans or not "".join(s.text for s in spans).strip():
                continue
            lines.append(Line(bbox=tuple(l["bbox"]), page=pno, spans=spans))
    return lines


def _mark_scripts(line: Line) -> None:
    """Geometric super/subscript detection (many PDFs don't set the flag)."""
    dom = line.size
    base = {}
    for s in line.spans:
        if abs(s.size - dom) < 0.6 and s.text.strip():
            base[round(s.origin_y, 1)] = base.get(round(s.origin_y, 1), 0) + len(s.text)
    if not base:
        return
    base_y = max(base, key=base.get)
    for s in line.spans:
        if s.size < dom * 0.82 and len(line.spans) > 1:
            if s.origin_y < base_y - 0.15 * dom:
                s.sup = True
            elif s.origin_y > base_y + 0.12 * dom:
                s.sub = True


def _gap_needs_space(prev: Span, nxt: Span) -> bool:
    if prev.text.endswith(" ") or nxt.text.startswith(" "):
        return False
    gap = nxt.bbox[0] - prev.bbox[2]
    return gap > 0.18 * min(prev.size, nxt.size)


def _merge_row(a: Line, b: Line) -> Line:
    spans = sorted(a.spans + b.spans, key=lambda s: s.bbox[0])
    out: list[Span] = []
    for s in spans:
        if out and _gap_needs_space(out[-1], s):
            s = replace(s, text=" " + s.text)
        out.append(s)
    bb = (min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))
    return Line(bbox=bb, page=a.page, spans=out)


_MARKER_ONLY = re.compile(r"^\(?(\d{1,3}|[a-z]|[ivxlc]{1,5})[.)]$", re.I)
_NUMERIC_TAIL = re.compile(r"^\s*(\d{1,4}|[ivxlc]{1,6})\s*$", re.I)


def _merge_rows(lines: list[Line], gap_factor: float = 1.2, min_gap: float = -2.0) -> list[Line]:
    """Join fragments that sit on one visual row (bullets, split spans, TOC page numbers)."""
    lines = sorted(lines, key=lambda l: (round(l.y0), l.x0))
    out: list[Line] = []
    for ln in lines:
        merged = False
        for i in range(len(out) - 1, max(len(out) - 8, -1), -1):
            p = out[i]
            ov = min(p.y1, ln.y1) - max(p.y0, ln.y0)
            h = min(p.height, ln.height)
            if h <= 0 or ov < 0.6 * h:
                continue
            left, right = (p, ln) if ln.x0 >= p.x0 else (ln, p)       # merge left-to-right
            gap = right.x0 - left.x1
            if gap < min_gap:
                continue
            size = max(p.size, ln.size)
            ltxt = left.text.strip()
            is_marker = (len(ltxt) <= 2 and not ltxt.isalnum()) or bool(_MARKER_ONLY.match(ltxt))
            numeric_tail = bool(_NUMERIC_TAIL.match(right.text)) and gap > 0 and left.width > 3 * size
            if gap < gap_factor * size or (is_marker and gap < 4 * size) or numeric_tail:
                if numeric_tail and gap >= gap_factor * size:
                    first = replace(right.spans[0], text="  " + right.spans[0].text)
                    right = Line(bbox=right.bbox, page=right.page, spans=[first] + right.spans[1:])
                out[i] = _merge_row(left, right)
                merged = True
                break
        if not merged:
            out.append(ln)
    for l in out:
        _mark_scripts(l)
    return out


def _attach_links(page: pymupdf.Page, lines: list[Line]) -> None:
    try:
        links = [l for l in page.get_links() if l.get("kind") == pymupdf.LINK_URI and l.get("uri")]
    except Exception:
        return
    for lk in links:
        r = tuple(lk["from"])
        for ln in lines:
            if _overlap(ln.bbox, r) <= 0:
                continue
            for s in ln.spans:
                if _inside(s.bbox, r, 0.5):
                    s.href = lk["uri"]


# ------------------------------------------------------------------- tables
def _cell_words(words, bbox):
    x0, y0, x1, y1 = bbox
    inside = [w for w in words if x0 - 1 <= (w[0] + w[2]) / 2 <= x1 + 1 and y0 - 1 <= (w[1] + w[3]) / 2 <= y1 + 1]
    inside.sort(key=lambda w: (w[5], w[6], w[7]))
    return inside


def _cell_text(words, bbox) -> str:
    return clean(" ".join(w[4] for w in _cell_words(words, bbox))).strip()


def _grid_from_words(words, bbox) -> Optional[list[list[str]]]:
    """Rebuild a rule-less (booktabs-style) table from word positions."""
    ws = _cell_words(words, bbox)
    if len(ws) < 4:
        return None
    hs = sorted(w[3] - w[1] for w in ws)
    med_h = hs[len(hs) // 2]
    lines: list[dict] = []
    for w in sorted(ws, key=lambda w: (w[1] + w[3]) / 2):
        cy = (w[1] + w[3]) / 2
        if lines and abs(cy - lines[-1]["cy"]) < 0.45 * max(w[3] - w[1], 1):
            lines[-1]["w"].append(w)
        else:
            lines.append({"cy": cy, "w": [w]})
    bands: list[list[float]] = []
    for a, b in sorted((w[0], w[2]) for w in ws):
        if bands and a <= bands[-1][1] + 0.35 * med_h:
            bands[-1][1] = max(bands[-1][1], b)
        else:
            bands.append([a, b])
    if len(bands) < 2:
        return None
    rows, cys = [], []
    for ln in lines:
        cells = [""] * len(bands)
        for w in sorted(ln["w"], key=lambda w: w[0]):
            ci = next((i for i, (a, b) in enumerate(bands) if w[0] <= b + 0.5 and w[2] >= a - 0.5), None)
            if ci is not None:
                cells[ci] = (cells[ci] + " " + w[4]).strip()
        if rows and not cells[0] and any(cells[1:]) and ln["cy"] - cys[-1] < 1.7 * med_h:
            for i, c in enumerate(cells):                      # wrapped cell text -> previous row
                if c:
                    rows[-1][i] = (rows[-1][i] + " " + c).strip()
        else:
            rows.append(cells)
        cys.append(ln["cy"])
    return [[clean(c) for c in r] for r in rows]


def _extract_tables(page: pymupdf.Page, area: float) -> list[TableItem]:
    out: list[TableItem] = []
    try:
        found = page.find_tables()
    except Exception as e:  # pragma: no cover
        log.debug("find_tables failed on page %d: %s", page.number, e)
        return out
    if not found.tables:
        return out
    words = page.get_text("words")
    for t in found.tables:
        try:
            rows, multiline = [], False
            for r in t.rows:
                row = []
                for c in r.cells:
                    cw = _cell_words(words, c) if c else []
                    multiline |= len({(w[5], w[6]) for w in cw}) > 1
                    row.append(clean(" ".join(w[4] for w in cw)).strip())
                rows.append(row)
            if multiline:                      # rules only mark table edges: rebuild from word geometry
                grid = _grid_from_words(words, tuple(t.bbox))
                if grid and len(grid) >= 2:
                    rows = grid
        except Exception:
            continue
        rows = [r for r in rows if any(r)]
        if len(rows) < 2 or max(len(r) for r in rows) < 2:
            continue
        cells = [c for r in rows for c in r]
        if sum(1 for c in cells if c) < 0.4 * len(cells):
            continue
        ncols = max(len(r) for r in rows)
        filled_cols = sum(1 for ci in range(ncols) if sum(1 for r in rows if ci < len(r) and r[ci]) >= 2)
        if filled_cols < 2:
            continue                                  # one real column: a text box, not a table
        bb = tuple(t.bbox)
        if _area(bb) > 0.92 * area:
            continue
        out.append(TableItem(bbox=bb, page=page.number, rows=rows))
    return out


# ------------------------------------------------------------- images / figures
def _png(pix: pymupdf.Pixmap) -> bytes:
    if pix.colorspace is not None and pix.colorspace.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return pix.tobytes("png")


def _render_clip(page: pymupdf.Page, bbox, dpi=200) -> tuple[bytes, int, int]:
    clip = pymupdf.Rect(bbox) & page.rect
    pix = page.get_pixmap(clip=clip, dpi=dpi, colorspace=pymupdf.csRGB)
    return pix.tobytes("png"), pix.width, pix.height


def _extract_images(doc: pymupdf.Document, page: pymupdf.Page) -> list[ImageItem]:
    out: list[ImageItem] = []
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return out
    for info in infos:
        bb = tuple(info["bbox"])
        if bb[2] - bb[0] < 18 or bb[3] - bb[1] < 18:
            continue
        w, h = info.get("width", 0), info.get("height", 0)
        if w and h and (w < 24 or h < 24):
            continue
        xref = info.get("xref", 0)
        tr = info.get("transform", (1, 0, 0, 1, 0, 0))
        rotated = abs(tr[1]) > 0.01 or abs(tr[2]) > 0.01
        data, ext = b"", "png"
        try:
            if xref and not rotated:
                ex = doc.extract_image(xref)
                if ex and ex.get("smask", 0) == 0 and ex["ext"] in ("jpeg", "jpg", "png") \
                        and ex.get("colorspace", 3) in (1, 3):
                    data, ext = ex["image"], "jpg" if ex["ext"] in ("jpeg", "jpg") else "png"
                elif ex:
                    pix = pymupdf.Pixmap(doc, xref)
                    if ex.get("smask", 0):
                        pix = pymupdf.Pixmap(pix, pymupdf.Pixmap(doc, ex["smask"]))
                    data = _png(pix)
            if not data:
                data, w, h = _render_clip(page, bb)
        except Exception as e:
            log.debug("image extract fallback p%d: %s", page.number, e)
            try:
                data, w, h = _render_clip(page, bb)
            except Exception:
                continue
        key = hashlib.sha1(data).hexdigest()[:16]
        out.append(ImageItem(bbox=bb, page=page.number, data=data, ext=ext, px_w=w, px_h=h, key=key))
    return out


@dataclass
class _Cluster:
    rect: pymupdf.Rect
    paths: int = 1
    curved: bool = False


def _vector_figures(page: pymupdf.Page, lines: list[Line], tables: list[TableItem],
                    images: list[ImageItem]) -> list[ImageItem]:
    """Cluster vector drawings into figures and rasterise them (charts, diagrams)."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    if not drawings or len(drawings) > 6000:
        return []
    prect = page.rect
    clusters: list[_Cluster] = []
    for d in drawings:
        r = pymupdf.Rect(d["rect"])
        if r.is_empty and (r.width < 0.01 and r.height < 0.01):
            continue
        if r.width * r.height > 0.85 * prect.width * prect.height:
            continue                                  # page background
        thin = (r.width < 1.6 or r.height < 1.6)
        if thin and (r.width > 40 or r.height > 40) and len(d["items"]) <= 2:
            continue                                  # rules / underlines
        if any(_inside(tuple(r), t.bbox, 0.7) for t in tables):
            continue
        curved = any(it[0] in ("c", "qu") or (it[0] == "l" and abs(it[1].x - it[2].x) > 1
                                              and abs(it[1].y - it[2].y) > 1) for it in d["items"])
        clusters.append(_Cluster(r, 1, curved))
    # greedy merge of nearby rectangles
    pad = 8
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        out: list[_Cluster] = []
        for c in clusters:
            for o in out:
                big = pymupdf.Rect(o.rect.x0 - pad, o.rect.y0 - pad, o.rect.x1 + pad, o.rect.y1 + pad)
                if big.intersects(c.rect):
                    o.rect |= c.rect
                    o.paths += c.paths
                    o.curved |= c.curved
                    merged = True
                    break
            else:
                out.append(c)
        clusters = out
    figs: list[ImageItem] = []
    for c in clusters:
        r = c.rect
        if r.width < 40 or r.height < 40:
            continue
        inside = [l for l in lines if _inside(l.bbox, tuple(r), 0.8)]
        words = [len(l.text.split()) for l in inside]
        avg_words = sum(words) / len(words) if words else 0
        is_fig = (c.curved and c.paths >= 3) or (c.paths >= 12 and avg_words < 5)
        if not is_fig:
            continue
        if any(_inside(i.bbox, tuple(r), 0.8) and _area(i.bbox) > 0.6 * _area(tuple(r)) for i in images):
            continue
        data, w, h = _render_clip(page, tuple(r + (-2, -2, 2, 2)))
        figs.append(ImageItem(bbox=tuple(r), page=page.number, data=data, ext="png", px_w=w, px_h=h,
                              key=hashlib.sha1(data).hexdigest()[:16], kind="figure"))
        lines[:] = [l for l in lines if l not in inside]
    return figs



# ------------------------------------------------------------------ display equations
_MATH_FONT = re.compile(r"CMMI|CMSY|CMEX|CMBSY|CMMIB|MSAM|MSBM|Symbol|STIX|Math|Cambria|Euler|MTMI|MTSY|"
                        r"MT-?Extra|MTEX|RMTMI|Mathematica|CMR\d+-?Math", re.I)
_MATH_CHARS = set("∑∏∫∮√∂∇≤≥≠≈≡∈∉⊂⊆⊃⊇∪∩∀∃±×÷∞→←↔⇒⇐⇔∝∼≪≫⊗⊕∘′∗⋅·|‖⟨⟩ˆ˜¯")
_RELATION = set("=≤≥≈≡<>∈⊂⊆→⇒∝≠")
_EQ_NUMBER = re.compile(r"^\(?\s*[A-Z]?\d+(\.\d+)?[a-z]?\s*\)?$")


def _is_math_char(ch: str) -> bool:
    o = ord(ch)
    return (ch in _MATH_CHARS or 0x0370 <= o <= 0x03FF or 0x2200 <= o <= 0x22FF
            or 0x1D400 <= o <= 0x1D7FF or 0x2190 <= o <= 0x21FF)


def _math_profile(l: Line) -> tuple[float, int, bool, bool]:
    """(math-char share, long plain words, has relation, has sup/sub)"""
    total = math = 0
    plain = []
    relation = scripts = False
    for s in l.spans:
        t = s.text.strip()
        if not t:
            continue
        mf = bool(_MATH_FONT.search(s.font))
        n = len(t.replace(" ", ""))
        total += n
        if mf:
            math += n
        else:
            math += sum(1 for c in t if _is_math_char(c))
            plain.append(t)
        if s.sup or s.sub:
            scripts = True
        relation |= any(c in _RELATION for c in t)
    words = sum(1 for w in re.findall(r"[A-Za-z]{3,}", " ".join(plain)))
    return (math / total if total else 0.0, words, relation, scripts)


def _display_math(page: pymupdf.Page, lines: list[Line], body: float) -> list[ImageItem]:
    """Find display equations, rasterise them (keeping the text as alt) and remove their lines."""
    widths = sorted(l.width for l in lines)
    text_w = widths[int(len(widths) * 0.85)] if widths else 0

    def mathy(l: Line) -> bool:
        share, words, rel, scripts = _math_profile(l)
        if len(l.text.strip()) > 160 or l.width > 0.88 * text_w:
            return False                       # full-measure lines are prose with inline math
        ok_words = words <= 2 or (words <= 4 and share >= 0.35)
        return ok_words and ((share >= 0.4) or (rel and (scripts or share >= 0.15)))

    seeds = [l for l in lines if mathy(l)]
    if not seeds:
        return []
    seeds.sort(key=lambda l: (l.y0, l.x0))
    clusters: list[list[Line]] = []
    boxes: list[list[float]] = []
    for l in seeds:
        for c, b in zip(clusters, boxes):
            if l.y0 - b[3] < 1.5 * l.size and l.y1 > b[1] - 1.5 * l.size and l.x0 < b[2] and l.x1 > b[0]:
                c.append(l)
                b[:] = [min(b[0], l.x0), min(b[1], l.y0), max(b[2], l.x1), max(b[3], l.y1)]
                break
        else:
            clusters.append([l])
            boxes.append([l.x0, l.y0, l.x1, l.y1])
    items: list[ImageItem] = []
    taken: set[int] = set()
    for c, b in zip(clusters, boxes):
        members = list(c)
        for _ in range(2):                          # absorb fraction pieces, limits, equation numbers
            for l in lines:
                if l in members or id(l) in taken:
                    continue
                cy = (l.y0 + l.y1) / 2
                near = b[1] - 1.1 * l.size <= cy <= b[3] + 1.1 * l.size
                inside_x = l.x0 >= b[0] - 2 and l.x1 <= b[2] + 2
                short = len(l.text.strip()) <= 10
                number = bool(_EQ_NUMBER.match(l.text.strip())) and l.x0 >= b[2] - 5 \
                    and min(l.y1, b[3]) - max(l.y0, b[1]) > 0.4 * l.height
                share = _math_profile(l)[0]
                if (near and inside_x and (short or share >= 0.3)) or number:
                    members.append(l)
                    b[:] = [min(b[0], l.x0), min(b[1], l.y0), max(b[2], l.x1), max(b[3], l.y1)]
        if any(id(m) in taken for m in members):
            continue
        # a "display" must stand apart: skip when it is a full-width line of a paragraph
        if len(members) == 1 and members[0].width > 0.92 * page.rect.width * 0.75 and \
                _math_profile(members[0])[0] < 0.6:
            continue
        if sum(len(m.text.strip()) for m in members) <= 2:
            continue                                # lone glyphs are figure labels, not equations
        bbox = (b[0], b[1], b[2], b[3])
        clip = pymupdf.Rect(b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2) & page.rect
        pix = page.get_pixmap(clip=clip, dpi=220, colorspace=pymupdf.csRGB)
        data = pix.tobytes("png")
        alt = re.sub(r"\s+", " ", " ".join(m.text.strip() for m in sorted(members, key=lambda m: (round(m.y0 / 3), m.x0)))).replace("\ufffd", "").strip()
        items.append(ImageItem(bbox=bbox, page=page.number, data=data, ext="png", px_w=pix.width, px_h=pix.height,
                               key=hashlib.sha1(data).hexdigest()[:16], kind="equation", alt=alt))
        taken.update(id(m) for m in members)
    lines[:] = [l for l in lines if id(l) not in taken]
    return items

# ------------------------------------------------------------------------- main
def extract_page(doc: pymupdf.Document, pno: int, *, with_images: bool = True,
                 ocr: Optional[OcrEngine] = None, ocr_mode: str = "auto", body_size: float = 10.0,
                 equations: bool = True) -> PageData:
    page = doc[pno]
    rect = page.rect
    pd = PageData(pno, rect.width, rect.height)
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    raw = _lines_from_dict(page.get_text("dict", flags=flags), pno)
    nchars = sum(len(l.text.strip()) for l in raw)

    images = _extract_images(doc, page) if with_images else []
    page_area = rect.width * rect.height
    covers = [i for i in images if _area(i.bbox) > 0.6 * page_area]

    need_ocr = ocr_mode == "force" or (ocr_mode == "auto" and nchars < 15 and bool(covers))
    if need_ocr:
        if ocr and ocr.available:
            try:
                raw = ocr.page_lines(page)
                pd.ocr = True
                images = [i for i in images if i not in covers]   # the scan itself is replaced by text
            except Exception as e:
                log.warning("OCR failed on page %d: %s", pno + 1, e)
        else:
            log.warning("page %d looks scanned but no OCR engine is available; embedding page image", pno + 1)
            if not covers:
                data, w, h = _render_clip(page, tuple(rect), 150)
                images = [ImageItem(bbox=tuple(rect), page=pno, data=data, px_w=w, px_h=h,
                                    key=hashlib.sha1(data).hexdigest()[:16], kind="page")]
            else:
                for c in covers:
                    c.kind = "page"
    elif covers and nchars >= 15:
        images = [i for i in images if i not in covers]           # decorative full-page background

    tables = _extract_tables(page, page_area)
    if tables:
        raw = [l for l in raw if not any(_inside(l.bbox, t.bbox, 0.5) for t in tables)]
    lines = _merge_rows(raw, 2.5, -10.0) if pd.ocr else _merge_rows(raw)
    if not pd.ocr:
        _attach_links(page, lines)
    if with_images:
        images += _vector_figures(page, lines, tables, images)
    if equations and with_images and not pd.ocr:
        images += _display_math(page, lines, body_size)
    pd.lines, pd.images, pd.tables = lines, images, tables
    return pd
