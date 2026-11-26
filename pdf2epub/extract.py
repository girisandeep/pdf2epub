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
        # OCR has no font info so sizes beyond 2× median are noise, not real headings
        if out:
            sizes = sorted(l.spans[0].size for l in out)
            med = sizes[len(sizes) // 2]
            for l in out:
                sp = l.spans[0]
                if 0.72 * med <= sp.size <= 1.3 * med:
                    sp.size = med
                elif sp.size > 2.0 * med:
                    sp.size = med * 1.3
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


def _merge_rows(lines: list[Line], gap_factor: float = 1.2, min_gap: float = -2.0,
                page_width: float = 0.0, col_cut: float = 0.0) -> list[Line]:
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
            smin = min(p.size, ln.size)
            ltxt = left.text.strip()
            is_marker = (len(ltxt) <= 2 and not ltxt.isalnum()) or bool(_MARKER_ONLY.match(ltxt))
            numeric_tail = bool(_NUMERIC_TAIL.match(right.text)) and gap > 0 and left.width > 3 * size
            if col_cut and left.x1 <= col_cut and right.x0 >= col_cut:
                continue
            if page_width and not is_marker and smin > 0:
                ratio = size / smin
                if (ratio > 1.2 and gap > smin
                        and left.width > 0.3 * page_width
                        and right.width > 2 * size):
                    continue
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


def _ruled_tables(page: pymupdf.Page, words, area: float) -> list[TableItem]:
    """Detect booktabs-style tables from horizontal rules + word geometry."""
    pw = page.rect.width
    rules: list[tuple[float, float, float]] = []  # (y, x0, x1)
    for path in page.get_drawings():
        for item in path.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 2 and abs(p2.x - p1.x) > 0.3 * pw:
                    rules.append((round((p1.y + p2.y) / 2, 1), min(p1.x, p2.x), max(p1.x, p2.x)))
            elif item[0] == "re":
                r = item[1]
                if abs(r.y1 - r.y0) < 2 and abs(r.x1 - r.x0) > 0.3 * pw:
                    rules.append((round((r.y0 + r.y1) / 2, 1), r.x0, r.x1))
    if len(rules) < 2:
        return []
    rules.sort()
    groups: list[list[tuple[float, float, float]]] = [[rules[0]]]
    for r in rules[1:]:
        prev = groups[-1][-1]
        x_overlap = min(r[2], prev[2]) - max(r[1], prev[1])
        same_extent = x_overlap > 0.5 * min(r[2] - r[1], prev[2] - prev[1])
        if same_extent:
            groups[-1].append(r)
        else:
            groups.append([r])

    def _try_grid(g: list[tuple[float, float, float]]) -> Optional[TableItem]:
        if len(g) < 2:
            return None
        y0, y1 = g[0][0] - 2, g[-1][0] + 2
        x_all = [w[0] for w in words if y0 <= (w[1] + w[3]) / 2 <= y1] + \
                [w[2] for w in words if y0 <= (w[1] + w[3]) / 2 <= y1]
        if not x_all:
            return None
        bbox = (min(x_all) - 1, y0, max(x_all) + 1, y1)
        if _area(bbox) > 0.92 * area:
            return None
        grid = _grid_from_words(words, bbox)
        if not grid or len(grid) < 2 or max(len(r) for r in grid) < 2:
            return None
        ncols = max(len(r) for r in grid)
        cells = [c for r in grid for c in r if c]
        if len(cells) < 0.4 * sum(len(r) for r in grid):
            return None
        avg_len = sum(len(c) for c in cells) / max(len(cells), 1)
        if ncols <= 2 and avg_len > 40:
            return None
        filled_cols = sum(1 for ci in range(ncols) if sum(1 for r in grid if ci < len(r) and r[ci]) >= 2)
        if filled_cols < 2:
            return None
        return TableItem(bbox=bbox, page=page.number, rows=grid)

    out: list[TableItem] = []
    for g in groups:
        t = _try_grid(g)
        if t:
            out.append(t)
            continue
        if len(g) >= 4:
            gaps = [(g[i + 1][0] - g[i][0], i + 1) for i in range(len(g) - 1)]
            gaps.sort(reverse=True)
            for _, split_at in gaps[:3]:
                left = _try_grid(g[:split_at])
                right = _try_grid(g[split_at:])
                if left or right:
                    if left:
                        out.append(left)
                    if right:
                        out.append(right)
                    break
    return out


def _extract_tables(page: pymupdf.Page, area: float) -> list[TableItem]:
    out: list[TableItem] = []
    try:
        found = page.find_tables()
    except Exception as e:  # pragma: no cover
        log.debug("find_tables failed on page %d: %s", page.number, e)
        return out
    words = page.get_text("words")
    if not found.tables:
        return _ruled_tables(page, words, area)
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
        _toc_re = re.compile(r".*\S\s*(?:\.[\s.]{2,}|·{2,}|…+)\s*\d{1,4}\s*$")
        toc_rows = sum(1 for r in rows if _toc_re.match(" ".join(c for c in r if c)))
        if toc_rows > 0.5 * len(rows):
            continue
        out.append(TableItem(bbox=bb, page=page.number, rows=rows))
    for rt in _ruled_tables(page, words, area):
        if not any(_overlap(rt.bbox, t.bbox) > 0.3 * _area(rt.bbox) for t in out):
            out.append(rt)
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
        if any((i.kind != "figure" and _overlap(i.bbox, tuple(r)) > 0.6 * _area(tuple(r)))
               or (i.kind == "figure" and _overlap(i.bbox, tuple(r)) > 0.5 * min(_area(i.bbox), _area(tuple(r))))
               for i in images):
            continue
        data, w, h = _render_clip(page, tuple(r + (-2, -2, 2, 2)))
        alt = " ".join(l.text.strip() for l in sorted(inside, key=lambda l: (round(l.y0 / 3), l.x0)) if l.text.strip())
        figs.append(ImageItem(bbox=tuple(r), page=page.number, data=data, ext="png", px_w=w, px_h=h,
                              key=hashlib.sha1(data).hexdigest()[:16], kind="figure", alt=alt[:1000]))
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


# ----------------------------------------------------------------- layout-model regions
_CAPTION_START = re.compile(r"^\s*(fig(?:ure)?s?\.?|table|tab\.|algorithm|listing)\s*[\dIVXA-Z]", re.I)


def _center_in(bbox, region, pad: float = 1.0) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return region[0] - pad <= cx <= region[2] + pad and region[1] - pad <= cy <= region[3] + pad


def _render(page: pymupdf.Page, bbox, kind: str, alt: str = "", dpi: int = 220) -> ImageItem:
    clip = pymupdf.Rect(bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2) & page.rect
    pix = page.get_pixmap(clip=clip, dpi=dpi, colorspace=pymupdf.csRGB)
    data = pix.tobytes("png")
    return ImageItem(bbox=tuple(bbox), page=page.number, data=data, ext="png", px_w=pix.width, px_h=pix.height,
                     key=hashlib.sha1(data).hexdigest()[:16], kind=kind, alt=alt)


def _drop_furniture(lines: list[Line], regions) -> list[Line]:
    furn = [r.bbox for r in regions if r.label in ("page_header", "page_footer") and r.score >= 0.5]
    return [l for l in lines if not any(_center_in(l.bbox, f) for f in furn)]


def _structured_table(page: pymupdf.Page, r, words) -> Optional[TableItem]:
    """Table from the structure model's grid: structure from the model, text from the PDF's own words."""
    spec = r.table
    if not spec or not spec.get("cells") or spec["nrows"] < 2 or spec["ncols"] < 2:
        return None
    # each PDF word goes to exactly one cell (the smallest cell box containing its centre), so overlapping
    # boxes of merged cells can never duplicate text
    specs = spec["cells"]
    owner: dict[int, list] = {i: [] for i in range(len(specs))}
    for w in words:
        wb = (w[0], w[1], w[2], w[3])
        best, best_area = None, None
        for i, c in enumerate(specs):
            if c.get("bbox") and _center_in(wb, c["bbox"], 1.5):
                a = _area(c["bbox"])
                if best is None or a < best_area:
                    best, best_area = i, a
        if best is not None:
            owner[best].append(w)
    cells = []
    for i, c in enumerate(specs):
        ws = sorted(owner[i], key=lambda w: (w[5], w[6], w[7]))
        text = clean(" ".join(w[4] for w in ws)).strip() or clean(c.get("text", "")).strip()
        cells.append({"r0": c["r0"], "r1": max(c["r1"], c["r0"] + 1), "c0": c["c0"], "c1": max(c["c1"], c["c0"] + 1),
                      "text": text, "head": bool(c.get("head")), "rhead": bool(c.get("rhead"))})
    # coverage: the model's cells must account for (almost) every word the PDF has inside the region
    inside = [w for w in words if _center_in(((w[0]), w[1], w[2], w[3]), r.bbox)]
    boxes = [c["bbox"] for c in spec["cells"] if c.get("bbox")]
    covered = sum(1 for w in inside if any(_center_in((w[0], w[1], w[2], w[3]), b, 1.5) for b in boxes))
    if inside and covered < 0.95 * len(inside):
        return None
    filled = sum(1 for c in cells if c["text"])
    if filled < 0.3 * len(cells):
        return None
    grid = [[""] * spec["ncols"] for _ in range(spec["nrows"])]
    for c in cells:
        if c["r0"] < spec["nrows"] and c["c0"] < spec["ncols"]:
            grid[c["r0"]][c["c0"]] = c["text"]
    return TableItem(bbox=tuple(r.bbox), page=page.number, rows=grid, cells=cells,
                     nrows=spec["nrows"], ncols=spec["ncols"])


def _ml_tables(page: pymupdf.Page, regions, tables: list[TableItem], area: float) -> list[TableItem]:
    """Keep PyMuPDF tables the model agrees with; rebuild tables the model sees but PyMuPDF missed."""
    treg = [r for r in regions if r.label == "table" and r.score >= 0.5]
    words = None
    structured: list[TableItem] = []
    for r in treg:
        if r.table:
            words = words or page.get_text("words")
            t = _structured_table(page, r, words)
            if t is not None:
                structured.append(t)
    tables = [t for t in tables if not any(_overlap(t.bbox, s.bbox) > 0.3 * _area(t.bbox) for s in structured)]
    textual = [r.bbox for r in regions if r.label in ("text", "section_header", "list_item", "title") and r.score >= 0.8]
    kept = []
    for t in tables:
        agree = any(_overlap(t.bbox, r.bbox) > 0.3 * _area(t.bbox) for r in treg)
        # drop a ruled "table" only when the model is confident the area is plain prose and sees no table
        prose = sum(_overlap(t.bbox, b) for b in textual) > 0.7 * _area(t.bbox)
        if agree or not prose:
            kept.append(t)
    kept += structured
    for r in treg:
        if any(_overlap(t.bbox, r.bbox) > 0.3 * _area(r.bbox) for t in kept):
            continue
        words = words or page.get_text("words")
        grid = _grid_from_words(words, r.bbox)
        if grid and len(grid) >= 2 and max(len(row) for row in grid) >= 2:
            kept.append(TableItem(bbox=r.bbox, page=page.number, rows=grid))
    return kept


def _ml_formulas_figures(page: pymupdf.Page, regions, lines: list[Line], images: list[ImageItem],
                         tables: list[TableItem], body_size: float = 10.0) -> list[ImageItem]:
    out: list[ImageItem] = []
    taken: set[int] = set()
    for r in regions:
        if r.score < 0.4 or r.label not in ("formula", "picture"):
            continue
        members = [l for l in lines if id(l) not in taken and _center_in(l.bbox, r.bbox)]
        if any(_inside(t.bbox, r.bbox, 0.6) for t in tables):
            continue
        if r.label == "formula":
            plain = sum(1 for l in members for w in re.findall(r"[A-Za-z]{4,}", l.text))
            if plain >= 9:                       # a paragraph the model mistook for a formula
                continue
            bbox = r.bbox
            if members:
                bbox = (min(r.bbox[0], *(m.x0 for m in members)), min(r.bbox[1], *(m.y0 for m in members)),
                        max(r.bbox[2], *(m.x1 for m in members)), max(r.bbox[3], *(m.y1 for m in members)))
            alt = re.sub(r"\s+", " ", " ".join(m.text.strip() for m in sorted(members, key=lambda m: (round(m.y0 / 3), m.x0)))).replace("\ufffd", "").strip()
            img = _render(page, bbox, "equation", alt)
            img.latex = r.latex
            out.append(img)
            taken.update(id(m) for m in members)
        else:
            if any(_overlap(i.bbox, r.bbox) > 0.5 * _area(r.bbox) for i in images):
                continue                          # an embedded raster already covers most of it
            caps = [c.bbox for c in regions if c.label == "caption" and c.score >= 0.5]
            # captions and prose inside the box are real text: never swallow them into the picture
            members = [m for m in members if not any(_center_in(m.bbox, c) for c in caps)
                       and not _CAPTION_START.match(m.text) and m.size < 1.15 * body_size]   # headings stay text
            words = [len(m.text.split()) for m in members]
            if members and (sum(words) >= 80 or sum(words) / len(words) >= 6):
                continue                          # a text box, not a figure: keep it as text
            out.append(_render(page, r.bbox, "figure", dpi=200))
            taken.update(id(m) for m in members)
    lines[:] = [l for l in lines if id(l) not in taken]
    return out


def _assign_labels(lines: list[Line], regions) -> None:
    for l in lines:
        best, best_key = None, (0.0, 0.0)
        for r in regions:
            ov = _overlap(l.bbox, r.bbox) / max(_area(l.bbox), 1e-6)
            if ov >= 0.5 and (ov, r.score) > best_key:
                best, best_key = r, (ov, r.score)
        if best is not None:
            l.ml_label, l.ml_id = best.label, best.id

# ------------------------------------------------------------------------- main
def extract_page(doc: pymupdf.Document, pno: int, *, with_images: bool = True,
                 ocr: Optional[OcrEngine] = None, ocr_mode: str = "auto", body_size: float = 10.0,
                 equations: bool = True, regions=None) -> PageData:
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
    if not need_ocr and ocr_mode == "auto" and bool(covers) and 15 <= nchars <= 300:
        all_text = " ".join(l.text for l in raw)
        words = re.findall(r"[a-zA-Z]{3,}", all_text)
        if words:
            from .text import english_words
            ew = english_words()
            good = sum(1 for w in words if w.lower() in ew)
            if good / len(words) < 0.3:
                need_ocr = True
    if need_ocr:
        if ocr and ocr.available:
            try:
                raw = ocr.page_lines(page)
                pd.ocr = True
                for l in raw:
                    l.ocr = True
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
    if regions is not None:
        tables = _ml_tables(page, regions, tables, page_area)
        raw = _drop_furniture(raw, regions)
    if tables:
        raw = [l for l in raw if not any(_inside(l.bbox, t.bbox, 0.5) for t in tables)]
    if pd.ocr:
        lines = _merge_rows(raw, 2.5, -10.0)
    else:
        from .layout import _find_cut
        cut = _find_cut(raw) or 0.0
        lines = _merge_rows(raw, page_width=rect.width, col_cut=cut)
    if not pd.ocr:
        _attach_links(page, lines)
    if regions is not None and not pd.ocr and with_images:
        images += _ml_formulas_figures(page, regions, lines, images, tables, body_size)
    if with_images:
        images += _vector_figures(page, lines, tables, images)        # heuristics catch what the model missed
        if equations and not pd.ocr:
            images += _display_math(page, lines, body_size)
    if regions is not None and not pd.ocr:
        _assign_labels(lines, regions)
    pd.lines, pd.images, pd.tables = lines, images, tables
    return pd
