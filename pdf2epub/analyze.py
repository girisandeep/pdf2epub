"""Document analysis: statistics, running-head removal, footnotes, paragraphs, headings, lists."""
from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from statistics import median

from .model import Block, FootnoteDef, ImageItem, Line, PageData, TableItem
from .text import BULLETS, TERMINAL, list_marker, norm_key, norm_title

CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?s?\.?|table|tab\.|chart|diagram|photo|plate|scheme|listing|exhibit)"
                        r"\s*[\dIVXA-Z]+[.:)\s-]", re.I)
PAGE_NUM_RE = re.compile(r"^\s*(page\s+)?[-–—\s]*(\d{1,5}|[ivxlc]{1,7})[-–—\s]*(of\s+\d+|/\s*\d+)?\s*$", re.I)
TOC_LINE_RE = re.compile(r"^(?P<t>.*?\S)\s*(?:\.[\s.]{2,}|·{2,}|…+|\s{2,})\s*(?P<n>\d{1,4}|[ivxlc]{1,6})\s*$", re.I)
INDEX_ENTRY_RE = re.compile(r"^(.{2,50}?)\s+(\d{1,4}(?:[,–-]\s*\d{1,4})*)$")
_NAKED_LIST_HEADING = re.compile(
    r"^(see also|related|external links|further reading|related articles"
    r"|related pages|related topics|voir aussi|siehe auch|véase también"
    r"|voci correlate|enlaces externos|liens externes|weblinks)$", re.I)
FN_SYMBOLS = "*†‡§¶∗⋆★"
_LATEX_TOC_JUNK = re.compile(r"bold0mu\s+mumu\s+\d{4}/\d{2}/\d{2}\s+ver:\s+[\d.]+\s+\w+\s+package")
FN_MARK_RE = re.compile(rf"^\s*(\d{{1,3}}|[{FN_SYMBOLS}])(?=[\s)\]]|\.\D|\.$|$)")
FN_SUP_RE = re.compile(rf"^(\d{{1,3}}|[{FN_SYMBOLS}])$")


def footnote_marker(l: Line) -> str | None:
    """Marker of a footnote definition line: a leading superscript, or a plain leading number/symbol."""
    first = l.spans[0]
    t = first.text.strip()
    if (first.sup or first.size < l.size * 0.85) and FN_SUP_RE.match(t):
        return t
    m = FN_MARK_RE.match(l.text)
    return m.group(1) if m else None
SECTION_WORDS = re.compile(r"^(abstract|summary|preface|foreword|introduction|contents|table of contents|"
                           r"inhaltsverzeichnis|table des matières|índice|sommario|"
                           r"acknowledge?ments?|references|bibliography|appendix|index|glossary|dedication|"
                           r"prologue|epilogue|conclusions?)\b", re.I)
_CHAPTER_LABEL = re.compile(
    r"^(chapter|part|appendix|teil|chapitre|capítulo|capitolo)\s+(\d+|[IVXLC]+|[A-Z])$", re.I)
CODE_ISH_RE = re.compile(r"^(?:[\w.+-]+@[\w.-]+\.\w+|https?://\S+|[\w.-]+(?:/[\w.-]+)+\.?)$")
SOLID_BULLETS = set("•●▪■◦○‣∙►▸◆◇□☐✓✔➢➤")


@dataclass
class DocStats:
    body_size: float = 10.0
    leading: float = 1.2
    vocab: Counter = field(default_factory=Counter)
    bold_ratio: float = 0.0
    italic_ratio: float = 0.0
    n_pages: int = 0


@dataclass
class Counters:
    running_removed: int = 0
    footnotes: int = 0
    headings: int = 0
    lists: int = 0
    tables: int = 0
    images: int = 0
    merged_paragraphs: int = 0
    toc_pages_entries: int = 0
    heading_source: str = "heuristic"


def base_y(l: Line) -> float:
    ys = [s.origin_y for s in l.spans if not s.sup and not s.sub and s.text.strip()]
    return median(ys) if ys else l.y1


def _starts_lower(text: str) -> bool:
    for ch in text.lstrip():
        if ch.isalpha():
            return ch.islower()
        if ch.isdigit():
            return False
    return False


# ------------------------------------------------------------------------ stats
def compute_stats(pages: list[PageData]) -> DocStats:
    sizes: Counter = Counter()
    ratios: list[float] = []
    vocab: Counter = Counter()
    chars = bold = ital = 0
    for p in pages:
        ls = sorted(p.lines, key=lambda l: (l.y0, l.x0))
        for l in ls:
            n = len(l.text.strip())
            sizes[l.size] += n
            for s in l.spans:
                k = len(s.text.strip())
                chars += k
                bold += k if s.bold else 0
                ital += k if s.italic else 0
            toks = re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", l.text)
            for t in toks[:-1] if toks else []:
                vocab[t.lower()] += 1
        for a, b in zip(ls, ls[1:]):
            if abs(a.size - b.size) < 0.4 and abs(a.x0 - b.x0) < 3 * a.size:
                r = (base_y(b) - base_y(a)) / a.size
                if 0.9 <= r <= 2.2:
                    ratios.append(r)
    st = DocStats(n_pages=len(pages), vocab=vocab)
    if sizes:
        st.body_size = max(sizes, key=sizes.get)
    if ratios:
        st.leading = median(ratios)
    if chars:
        st.bold_ratio, st.italic_ratio = bold / chars, ital / chars
    if st.bold_ratio > 0.45:     # fonts that merely *report* bold: ignore emphasis
        for p in pages:
            for l in p.lines:
                for s in l.spans:
                    s.bold = False
    if st.italic_ratio > 0.45:
        for p in pages:
            for l in p.lines:
                for s in l.spans:
                    s.italic = False
    return st


# ------------------------------------------------------------- running head/foot
def strip_running(pages: list[PageData], c: Counters, keep: bool = False) -> None:
    if keep:
        return
    cnt: Counter = Counter()
    zones: dict[int, list[tuple[Line, tuple]]] = defaultdict(list)
    for p in pages:
        top, bot = 0.12 * p.height, 0.88 * p.height
        for l in p.lines:
            if l.y1 <= top or l.y0 >= bot:
                nk = norm_key(l.text)
                if PAGE_NUM_RE.match(l.text):
                    nk = "#"
                key = (nk, round(l.y0 / 6))
                if key[0]:
                    cnt[key] += 1
                    zones[p.number].append((l, key))
    n = len(pages)
    ybin_caught: Counter = Counter()
    for p in pages:
        drop = set()
        for l, key in zones.get(p.number, []):
            pn = bool(PAGE_NUM_RE.match(l.text))
            if cnt[key] >= 3 or (pn and (n <= 3 or cnt[key] >= 2)):
                drop.add(id(l))
                ybin_caught[key[1]] += 1
        if drop:
            c.running_removed += len(drop)
            p.lines = [l for l in p.lines if id(l) not in drop]
    # position-based: if a y-bin has confirmed headers on many pages,
    # also strip remaining short text at the same position
    hot_ybins = {yb for yb, c2 in ybin_caught.items() if c2 >= max(n * 0.3, 4)}
    if hot_ybins:
        for p in pages:
            drop = set()
            top, bot = 0.12 * p.height, 0.88 * p.height
            for l in p.lines:
                if (l.y1 <= top or l.y0 >= bot) and round(l.y0 / 6) in hot_ybins \
                        and len(l.text.strip()) < 80:
                    drop.add(id(l))
            if drop:
                c.running_removed += len(drop)
                p.lines = [l for l in p.lines if id(l) not in drop]
    # repeating small images (logos, rules) in margins or on many pages
    img_pages: dict[str, set[int]] = defaultdict(set)
    for p in pages:
        for im in p.images:
            img_pages[im.key].add(p.number)
    for p in pages:
        keep_imgs = []
        for im in p.images:
            if im.kind == "equation":
                keep_imgs.append(im)
                continue
            area = im.width * im.height / (p.width * p.height)
            margin = im.y1 <= 0.12 * p.height or im.y0 >= 0.88 * p.height
            reps = len(img_pages[im.key])
            if (reps >= 3 and (area < 0.12 or margin)) or (margin and reps >= 2 and area < 0.05):
                continue
            keep_imgs.append(im)
        p.images = keep_imgs


# -------------------------------------------------------------------- footnotes
def extract_footnotes(page: PageData, stats: DocStats,
                      prev_last: FootnoteDef | None = None) -> list[FootnoteDef]:
    """Split a page's footnote area from its body, but only when body markers confirm it."""
    body = stats.body_size
    ml_zone = [l for l in sorted(page.lines, key=lambda l: (l.y0, l.x0)) if l.ml_label == "footnote"]
    if ml_zone:
        start, zone = ml_zone[0].y0, ml_zone
    else:
        lower = [l for l in sorted(page.lines, key=lambda l: l.y0) if l.y0 > 0.5 * page.height]
        start = None
        for l in lower:
            if l.size <= body * 0.93 and footnote_marker(l):
                start = l.y0
                break
        if start is None:
            return []
        zone = [l for l in sorted(page.lines, key=lambda l: (l.y0, l.x0))
                if l.y0 >= start - 1 and l.size <= body * 0.95]
    # In two-column layouts, a footnote may start on one page and continue on
    # the next page's other column.  Process columns separately to avoid
    # appending cross-column continuations to the wrong footnote.
    mid = page.width / 2
    left = [l for l in zone if l.x0 < mid]
    right = [l for l in zone if l.x0 >= mid]
    if left and right:
        left_markers = any(footnote_marker(l) for l in left)
        right_markers = any(footnote_marker(l) for l in right)
        if right_markers and not left_markers:
            orphans = sorted(left, key=lambda l: l.y0)
            zone = sorted(right, key=lambda l: (l.y0, l.x0))
        elif left_markers and not right_markers:
            orphans = sorted(right, key=lambda l: l.y0)
            zone = sorted(left, key=lambda l: (l.y0, l.x0))
        else:
            orphans = []
            zone = (sorted(left, key=lambda l: (l.y0, l.x0))
                    + sorted(right, key=lambda l: (l.y0, l.x0)))
        if orphans:
            if prev_last:
                prev_last.lines.extend(orphans)
            orphan_ids = {id(l) for l in orphans}
            page.lines = [l for l in page.lines if id(l) not in orphan_ids]
    defs: list[FootnoteDef] = []
    for l in zone:
        mk = footnote_marker(l)
        if mk:
            defs.append(FootnoteDef(page.number, mk, [l]))
        elif defs:
            defs[-1].lines.append(l)
    if not defs:
        return []
    sup_marks = set()
    for l in page.lines:
        if l.y0 < start - 1:
            for s in l.spans:
                if s.sup:
                    t = s.text.strip()
                    sup_marks.add(t)
                    if all(ch in FN_SYMBOLS for ch in t):
                        sup_marks.update(t)
    defs = [d for d in defs if d.marker in sup_marks]      # unmatched notes stay in the normal text flow
    if not defs:
        return []
    zone_ids = {id(l) for d in defs for l in d.lines}
    page.lines = [l for l in page.lines if id(l) not in zone_ids]
    return defs


# ------------------------------------------------------------------ segmentation
def _region_stats(items) -> dict[int, tuple[float, float]]:
    by: dict[int, list[Line]] = defaultdict(list)
    for it in items:
        if isinstance(it, Line):
            by[it.region].append(it)
    out = {}
    for rid, ls in by.items():
        xs0 = sorted(l.x0 for l in ls)
        xs1 = sorted(l.x1 for l in ls)
        out[rid] = (xs0[len(xs0) // 4], xs1[min(len(xs1) - 1, int(len(xs1) * 0.9))])
    return out


def _right_edges(items, rst) -> dict[int, float]:
    """Right edge of the text column each line belongs to, estimated from lines aligned with it."""
    by: dict[int, list[Line]] = defaultdict(list)
    for it in items:
        if isinstance(it, Line):
            by[it.region].append(it)
    out: dict[int, float] = {}
    for rid, ls in by.items():
        for ln in ls:
            peers = sorted(p.x1 for p in ls if abs(p.size - ln.size) < 0.5 and abs(p.x0 - ln.x0) < 4)
            out[id(ln)] = peers[min(len(peers) - 1, int(len(peers) * 0.9))] if len(peers) >= 3 else rst[rid][1]
    return out


def _ends_complete(l: Line) -> bool:
    return l.text.rstrip().endswith(TERMINAL)


def _is_break(prev: Line, ln: Line, cur: Block, left: float, right: float, st: DocStats) -> bool:
    if ln.region != prev.region:
        return True
    size = max(ln.size, prev.size)
    ml_known = prev.ml_id >= 0 and ln.ml_id >= 0
    if ml_known and prev.ml_id != ln.ml_id:
        # The model's region edge is only trusted when the geometry agrees: a different kind of region,
        # extra vertical space, or a line that ended early.  Otherwise a boundary that slices through
        # the middle of a paragraph would leave one-line orphans.
        gap = base_y(ln) - base_y(prev)
        if prev.ml_label != ln.ml_label or gap > size * (st.leading + 0.12) or not prev.text.rstrip().endswith(
                ("-", "\u00ad", ",")) and _ends_complete(prev) and prev.x1 < right - 2.5 * size:
            return True
    dy = base_y(ln) - base_y(prev)
    if dy < -0.3 * size:
        return True
    if ln.mono != prev.mono:
        return True
    if cur.kind in ("code", "algorithm"):
        return False
    if abs(ln.size - prev.size) > 0.8:
        return True
    if (ln.bold != prev.bold) and (ln.bold or prev.bold):
        return True
    complete = _ends_complete(prev)
    lm = list_marker(ln.text)
    if lm and (lm[0] == "bullet" or complete or dy > size * (st.leading + 0.1)):
        return True
    if complete and _BIB_MARKER.match(ln.text):
        return True
    if dy > size * (st.leading + 0.28):
        return True
    if abs(dy) < 0.3 * size and ln.x0 > prev.x1:
        return False
    ptxt = prev.text.rstrip()
    words = ln.text.split()
    if words and not ptxt.endswith(("-", "­")) and len(ptxt) > 3:
        cw = prev.width / max(len(prev.text.strip()), 1)
        if prev.x1 + (len(words[0]) + 1) * cw * 1.15 + 0.3 * size < right:
            return True
    indent = ln.x0 - left
    if cur.marker is None and indent > 0.9 * size and (ln.x0 - prev.x0) > 0.7 * size and complete:
        return True
    return False


def segment_page(items: list, page: int, st: DocStats) -> list[Block]:
    blocks: list[Block] = []
    rst = _region_stats(items)
    redge = _right_edges(items, rst)

    for it in items:
        if not isinstance(it, ImageItem):
            continue
        ix0, iy0, _, iy1 = it.bbox
        for jt in items:
            if not isinstance(jt, Line):
                continue
            cy = (jt.y0 + jt.y1) / 2
            if iy0 - jt.height <= cy <= iy1 + jt.height \
                    and ix0 > jt.x1 and ix0 < redge.get(id(jt), ix0 + 1):
                redge[id(jt)] = ix0

    cur: Block | None = None
    prev: Line | None = None

    def flush():
        nonlocal cur, prev
        if cur is not None:
            blocks.append(cur)
        cur, prev = None, None

    for it in items:
        if isinstance(it, ImageItem):
            flush(); blocks.append(Block("image", page, image=it, y0=it.y0, region=it.region)); continue
        if isinstance(it, TableItem):
            flush(); blocks.append(Block("table", page, table=it, y0=it.y0, region=it.region)); continue
        ln: Line = it
        left, right = rst[ln.region][0], redge[id(ln)]
        if cur is None or prev is None or _is_break(prev, ln, cur, left, right, st):
            if cur is not None:
                blocks.append(cur)
            complete = prev is None or _ends_complete(prev) or ln.region != prev.region \
                or base_y(ln) - base_y(prev) > ln.size * (st.leading + 0.1)
            lm = list_marker(ln.text) if not ln.mono else None
            if lm and lm[0] != "bullet" and not (complete or cur is None):
                lm = None
            kind = "code" if ln.mono else "algorithm" if ln.ml_label == "code" else "para"
            cur = Block(kind, page, [ln], region=ln.region, y0=ln.y0,
                        marker=lm, indent=ln.x0)
            cur.indented_first = (ln.x0 - left) > 0.8 * ln.size
        else:
            cur.lines.append(ln)
        prev = ln
        cur.full_last = ln.x1 >= right - 1.3 * ln.size
    flush()
    return blocks


# ---------------------------------------------------------------------- headings
def _bkey(b: Block):
    sizes = sorted(l.size for l in b.lines)
    size = sizes[len(sizes) // 2]
    if _is_caps(b):
        max_span = max((s.size for l in b.lines for s in l.spans), default=size)
        size = max(size, max_span)
    return (size, all(l.bold for l in b.lines))


def _is_caps(b: Block) -> bool:
    t = re.sub(r"[^A-Za-z]", "", b.text)
    return len(t) >= 3 and t.isupper()


def _candidate(b: Block) -> bool:
    if b.kind != "para" or not b.lines or len(b.lines) > 4:
        return False
    t = b.text
    if len(t) > 200 or len(t.split()) > 28 or t.rstrip().endswith(","):
        return False
    if any(l.mono for l in b.lines):
        return False
    ts = t.strip()
    if _ALGO_KEYWORD.match(ts) or re.match(r"^end\s+(if|for|while|loop|function|procedure)$", ts, re.I):
        return False
    if list_marker_bullet(ts):
        return False
    return True


def _strip_num(t: str) -> str:
    return re.sub(r"^(chapter|part|section)?\s*([\dIVXLC]+[.)]?\s+|[\d.]+\s+)", "", norm_title(t)).strip()


def _title_match(a: str, b: str) -> float:
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return 0.0
    r = difflib.SequenceMatcher(None, na, nb).ratio()
    sa, sb = _strip_num(a), _strip_num(b)
    if sa and sb:
        r = max(r, difflib.SequenceMatcher(None, sa, sb).ratio())
        shorter = min(len(sa), len(sb))
        if shorter >= 5 and (sa in sb or sb in sa) and shorter * 3 >= max(len(sa), len(sb)):
            r = max(r, 0.9)
    return r


NUMBERED_RE = re.compile(r"^\s*(?:(?:chapter|section|part)\s+)?(\d+(?:\.\d+){0,4}|[A-Z](?:\.\d+){1,4})[.)]?\s+\S", re.I)


def _ml_heading_level(b: Block, key_level: dict) -> int | None:
    m = NUMBERED_RE.match(b.text)
    if m and "." in m.group(1).strip("."):
        return min(m.group(1).strip(".").count(".") + 1, 4)
    return key_level.get(_bkey(b))


# confident furniture is already removed upstream, so only labels that truly exclude a heading veto it
NON_HEADING_LABELS = {"list_item", "caption", "footnote", "formula", "code", "table"}


def _numbered_section_fallback(blocks: list[Block]) -> None:
    """Promote numbered section paragraphs when the primary classifier missed them.

    Fires when ≥3 paragraphs match "N Title" and use a different font family from
    the body text (e.g. Wikipedia PDFs that set headings in sans-serif while body is serif).
    """
    existing = sum(1 for b in blocks if b.kind == "heading")
    numbered_missed: list[Block] = []
    for b in blocks:
        if b.kind == "para" and _candidate(b) and NUMBERED_RE.match(b.text):
            t = b.text.strip()
            if len(t) <= 100 and not t.rstrip().endswith((".", "?", "!", ",", ";")) \
                    and len(t.split()) <= 14 and not CAPTION_RE.match(t):
                numbered_missed.append(b)
    if len(numbered_missed) < 3:
        return
    body_fonts = Counter()
    for b in blocks:
        if b.kind == "para" and not _candidate(b):
            for l in b.lines:
                for s in l.spans:
                    if s.text.strip():
                        body_fonts[s.font.split("-")[0]] += len(s.text)
    body_font = body_fonts.most_common(1)[0][0] if body_fonts else ""
    nm_fonts = Counter()
    for b in numbered_missed:
        for l in b.lines:
            for s in l.spans:
                if s.text.strip():
                    nm_fonts[s.font.split("-")[0]] += len(s.text)
    nm_font = nm_fonts.most_common(1)[0][0] if nm_fonts else ""
    diff_font = nm_font and body_font and nm_font != body_font
    nm_keys = Counter(_bkey(b) for b in numbered_missed)
    dominant_key = nm_keys.most_common(1)[0] if nm_keys else (None, 0)
    body_key_count = sum(1 for b in blocks if b.kind == "para" and not _candidate(b) and _bkey(b) == dominant_key[0])
    distinct_style = dominant_key[1] >= 3 and body_key_count <= 2
    if not (diff_font or distinct_style):
        return
    top = min((b.level for b in blocks if b.kind == "heading"), default=1) + 1 if existing else 2
    for b in numbered_missed:
        m = NUMBERED_RE.match(b.text)
        depth = m.group(1).strip(".").count(".") if m else 0
        b.kind, b.level, b.marker = "heading", min(top + depth, 4), None


def _classify_with_layout(blocks: list[Block], c: Counters) -> list[Block]:
    """Model-labelled headers are authoritative; levels come from numbering ("3.2.1" -> 3) and style.

    Blocks the model calls plain text may still be headings the model missed, but only when they share
    the exact style of >=2 confirmed headings and come after the first confirmed heading (so the
    author block on a title page is never promoted).
    """
    heads = [b for b in blocks if b.kind == "para" and b.ml_label in ("section_header", "title")
             and 0 < len(b.lines) <= 5 and len(b.text) <= 220 and not list_marker_bullet(b.text)]
    votes: dict[tuple, Counter] = defaultdict(Counter)
    for b in heads:
        m = NUMBERED_RE.match(b.text)
        if m:
            votes[_bkey(b)][min(m.group(1).strip(".").count(".") + 1, 4)] += 1
    key_level = {k: v.most_common(1)[0][0] for k, v in votes.items()}
    sizes = sorted({_bkey(b)[0] for b in heads}, reverse=True)
    levels: dict[int, int] = {}
    for b in heads:
        if b.ml_label == "title":
            lvl = 1
        else:
            lvl = _ml_heading_level(b, key_level)
            if lvl is None:
                if SECTION_WORDS.match(b.text.strip()):
                    lvl = 1 if not key_level else min(key_level.values())
                else:
                    lvl = min(sizes.index(_bkey(b)[0]) + 1, 4)
        levels[id(b)] = lvl
        b.kind, b.level, b.marker = "heading", lvl, None
    # headings the model missed: same visual style as confirmed ones
    style_votes: dict[tuple, Counter] = defaultdict(Counter)
    for b in heads:
        style_votes[_bkey(b)][levels[id(b)]] += 1
    confirmed = [i for i, b in enumerate(blocks) if b.kind == "heading"]
    # the body starts at the first numbered / standard section heading ("1 Introduction", "Abstract");
    # anything before it is title-page matter (authors, affiliations) and is never promoted
    first = next((i for i in confirmed if NUMBERED_RE.match(blocks[i].text) or SECTION_WORDS.match(blocks[i].text.strip())),
                 confirmed[0] if confirmed else len(blocks))
    extra = 0
    for i, b in enumerate(blocks):
        if i <= first or b.kind != "para" or b.ml_label in NON_HEADING_LABELS or not _candidate(b):
            continue
        k = _bkey(b)
        v = style_votes.get(k)
        t = b.text.strip()
        if v and sum(v.values()) >= 2 and len(t) <= 120 and not list_marker_bullet(t) \
                and not CAPTION_RE.match(t) and not t.endswith((".", ",", ";")) and k[1]:
            b.kind, b.level, b.marker = "heading", v.most_common(1)[0][0], None
            extra += 1
    _numbered_section_fallback(blocks)
    c.heading_source = "layout-model"
    c.headings = sum(1 for b in blocks if b.kind == "heading")
    return blocks


def list_marker_bullet(text: str) -> bool:
    m = list_marker(text)
    return bool(m and m[0] == "bullet")


def _fix_numbered_heading_levels(blocks: list[Block]) -> None:
    """Adjust heading levels so section numbering depth is reflected in the hierarchy.

    E.g. if "3 Results" is h1 and "3.1.1 Language Modeling" was assigned h2 by style
    propagation, bump it to h3 because numbering depth 2 is two levels below depth 0.
    """
    numbered: list[tuple[Block, int]] = []
    for b in blocks:
        if b.kind != "heading":
            continue
        m = NUMBERED_RE.match(b.text)
        if m:
            depth = m.group(1).strip(".").count(".")
            numbered.append((b, depth))
    if len(numbered) < 3:
        return
    depths = {d for _, d in numbered}
    if len(depths) < 2:
        return
    min_depth = min(depths)
    levels_at_min = [b.level for b, d in numbered if d == min_depth]
    if not levels_at_min:
        return
    base_level = min(levels_at_min)
    changed = 0
    for b, depth in numbered:
        ideal = base_level + (depth - min_depth)
        ideal = max(1, min(ideal, 6))
        if ideal != b.level:
            b.level = ideal
            changed += 1
    if not changed:
        return


_CHAPTER_LABEL = re.compile(
    r"^\s*(?:chapter|part|appendix|annex|teil|kapitel|chapitre|capitulo)\s+"
    r"(?:\d+|[IVXLC]+|[A-Z])\s*$", re.I)


def _merge_chapter_headings(blocks: list[Block]) -> None:
    """Merge 'Chapter N' + 'Title' consecutive same-level headings into one."""
    to_remove: list[int] = []
    for i in range(len(blocks) - 1):
        a, b = blocks[i], blocks[i + 1]
        if (a.kind == "heading" and b.kind == "heading"
                and a.level == b.level and a.page == b.page
                and _CHAPTER_LABEL.match(a.text)):
            b.synthetic_text = a.text.strip() + ": " + (b.synthetic_text or b.text).strip()
            b.anchor = b.anchor or a.anchor
            to_remove.append(i)
    for i in reversed(to_remove):
        del blocks[i]


def _collapse_heading_gaps(blocks: list[Block]) -> None:
    """Ensure heading levels don't skip (h2 -> h4 becomes h2 -> h3)."""
    headings = [(i, b) for i, b in enumerate(blocks) if b.kind == "heading"]
    if not headings:
        return
    stack: list[tuple[int, int]] = []  # (original_level, collapsed_level)
    for _, b in headings:
        ol = b.level
        while stack and stack[-1][0] >= ol:
            stack.pop()
        parent_cl = stack[-1][1] if stack else 0
        cl = min(ol, parent_cl + 1)
        b.level = cl
        stack.append((ol, cl))


def classify_headings(blocks: list[Block], toc: list, st: DocStats, c: Counters) -> list[Block]:
    body = st.body_size
    has_ml = any(l.ml_label for b in blocks for l in b.lines)
    if has_ml and len(toc) < 2:
        return _classify_with_layout(blocks, c)
    by_page: dict[int, list[Block]] = defaultdict(list)
    for b in blocks:
        by_page[b.page].append(b)

    def mark(b: Block, level: int):
        b.kind, b.level, b.marker = "heading", max(1, min(level, 6)), None

    key_levels: dict[tuple, Counter] = defaultdict(Counter)
    used_toc = len(toc) >= 2
    synth: list[tuple[int, int, str, float | None]] = []   # (page, level, title, dest_y)
    if used_toc:
        c.heading_source = "outline"
        base_level = min(e[0] for e in toc)
        for lvl, title, pg, dest in toc:
            title = _LATEX_TOC_JUNK.sub("", title).strip()
            title = re.sub(r"\s{2,}", " ", title)
            pno = pg - 1
            lvl = lvl - base_level + 1
            if pno < 0:
                continue
            best, best_r = None, 0.0
            for q in (pno, pno + 1, pno - 1):
                for b in by_page.get(q, []):
                    if b.kind != "para" or not _candidate(b):
                        continue
                    r = _title_match(title, b.text)
                    if q != pno:
                        r -= 0.1
                    if r > best_r:
                        best, best_r = b, r
            if best is not None and best_r >= 0.82 and not best.text.strip()[:1].islower():
                mark(best, lvl)
                key_levels[_bkey(best)][min(lvl, 4)] += 1
            else:
                dy = None
                try:
                    dy = float(dest["to"].y) if dest and dest.get("to") is not None else None
                except Exception:
                    dy = None
                synth.append((pno, lvl, title, dy))
        # propagate learned heading styles to un-listed headings (only after the first outline hit,
        # so title/author blocks on the cover page are not mistaken for headings)
        first_hit = next((i for i, b in enumerate(blocks) if b.kind == "heading"), len(blocks))
        for bi, b in enumerate(blocks):
            if has_ml and b.ml_label in NON_HEADING_LABELS:
                continue
            if (bi > first_hit or SECTION_WORDS.match(b.text.strip())) and b.kind == "para" and _candidate(b):
                k = _bkey(b)
                if k in key_levels:
                    n = sum(key_levels[k].values())
                    lvl = key_levels[k].most_common(1)[0][0]
                    big = k[0] >= body * 1.15
                    if (big and n >= 1) or n >= 3:
                        ts = b.text.strip()
                        if ts.count(",") >= 3:
                            continue
                        if ts.endswith(("-", "­")):
                            continue
                        if not any(c.isalpha() for c in ts):
                            continue
                        if TOC_LINE_RE.match(ts):
                            continue
                        if ts[:1].islower():
                            continue
                        if CAPTION_RE.match(ts):
                            continue
                        if k[0] <= body * 1.05:
                            if ts.endswith("."):
                                continue
                        mark(b, lvl)
    else:
        cands = [b for b in blocks if _candidate(b)]
        keys: dict[tuple, list[Block]] = defaultdict(list)
        total = max(len([b for b in blocks if b.kind == "para"]), 1)
        for b in cands:
            size, bold = _bkey(b)
            t = b.text
            if CAPTION_RE.match(t) and size <= body * 1.25:
                continue
            if t.rstrip().endswith(("-", "­")):
                continue
            if not any(c.isalpha() for c in t):
                continue
            if TOC_LINE_RE.match(t.strip()):
                continue
            if size >= body * 1.12:
                keys[(size, bold, False)].append(b)
            elif size >= body * 0.95 and (bold or _is_caps(b)):
                if len(t) <= 110 and not t.rstrip().endswith(".") and len(t.split()) <= 14:
                    keys[(size, bold, not bold)].append(b)
        accepted = []
        for k, bs in keys.items():
            if len(bs) <= max(0.2 * total, 3) or k[0] >= body * 1.4:
                if k[0] <= body * 1.05 and len(bs) > 0.12 * total:
                    continue
                accepted.append(k)
        accepted.sort(key=lambda k: (-round(k[0]), not k[1], k[2]))
        levels = {k: min(i + 1, 4) for i, k in enumerate(accepted)}
        for k in accepted:
            for b in keys[k]:
                mark(b, levels[k])
    if not used_toc:
        _numbered_section_fallback(blocks)

    # promote large-font "Chapter N" / "Appendix A" labels to headings
    for b in blocks:
        if b.kind == "para" and b.lines and _CHAPTER_LABEL.match(b.text.strip()):
            avg = sum(l.size for l in b.lines) / len(b.lines)
            if avg >= body * 1.5:
                mark(b, 1)

    # synthetic headings from outline entries whose text could not be located
    for pno, lvl, title, dy in sorted(synth, key=lambda s: (s[0], s[3] or 0), reverse=True):
        pb = by_page.get(pno, [])
        idx = None
        if pb:
            if dy is not None:
                idx = next((blocks.index(b) for b in pb if b.y0 >= dy - 3), None)
            else:
                idx = blocks.index(pb[0])
            if idx is None:
                idx = blocks.index(pb[-1]) + 1
        else:
            nxt = next((i for i, b in enumerate(blocks) if b.page > pno), len(blocks))
            idx = nxt
        blocks.insert(idx, Block("heading", pno, level=max(1, min(lvl, 6)), synthetic_text=title, y0=dy or 0))
    _fix_numbered_heading_levels(blocks)
    _collapse_heading_gaps(blocks)
    _merge_chapter_headings(blocks)
    c.headings = sum(1 for b in blocks if b.kind == "heading")
    return blocks


# ------------------------------------------------------------ continuations/lists
def _find_last_para(out: list[Block]) -> Block | None:
    for i in range(len(out) - 1, -1, -1):
        if out[i].kind == "para":
            return out[i]
        if out[i].kind not in ("image", "table"):
            return None
    return None


def _max_span_size(line: Line) -> float:
    return max((s.size for s in line.spans if s.text.strip()), default=line.size)


def _try_merge(p: Block, b: Block, c: Counters) -> bool:
    pl, nl = p.lines[-1], b.lines[0]
    if abs(_max_span_size(pl) - _max_span_size(nl)) > 0.8:
        return False
    pl_bold = pl.spans[-1].bold if pl.spans else pl.bold
    nl_bold = nl.spans[0].bold if nl.spans else nl.bold
    pend = p.text.rstrip()
    hy = pend.endswith(("-", "­"))
    term = pend.endswith(TERMINAL)
    lower = _starts_lower(b.text)
    if CAPTION_RE.match(b.text):
        return False
    if CAPTION_RE.match(p.text) and not (hy and lower):
        return False
    cont_term = pend.endswith((")", "]"))
    if pl_bold != nl_bold and not (lower and (not term or cont_term)):
        return False
    if (hy and lower) or (not term and lower) or (cont_term and lower) \
            or (not term and p.full_last and not b.indented_first):
        p.lines += b.lines
        p.full_last = b.full_last
        p.region, p.page = b.region, b.page
        c.merged_paragraphs += 1
        return True
    return False


def _short_fragment(b: Block) -> bool:
    t = b.text.rstrip()
    return len(t) < 40 and not t.endswith(TERMINAL)


def _para_duplicates_table(b: Block, table: "TableItem") -> bool:
    """True when a paragraph is entirely composed of text already in a nearby table."""
    text = b.text.strip()
    if len(text) > 200:
        return False
    para_words = set(re.findall(r"\w+", text.lower()))
    if len(para_words) < 3:
        return False
    cell_words: set[str] = set()
    for row in table.rows:
        for cell in row:
            cell_words.update(re.findall(r"\w+", cell.lower()))
    return para_words <= cell_words


def merge_continuations(blocks: list[Block], c: Counters) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        if b.kind == "code" and out and out[-1].kind == "para" \
                and out[-1].text.rstrip().endswith(("https:", "http:")) \
                and b.lines and b.lines[0].text.lstrip().startswith("//"):
            out[-1].lines += b.lines
            out[-1].full_last = b.full_last
            c.merged_paragraphs += 1
            continue
        if b.kind != "para" or b.marker is not None or not out:
            out.append(b)
            continue
        prev = out[-1]
        if prev.kind == "para" and (b.region != prev.region or b.page != prev.page):
            if _try_merge(prev, b, c):
                continue
        elif prev.kind == "image":
            p = _find_last_para(out)
            if p is not None:
                if _try_merge(p, b, c):
                    continue
                pt = p.text.rstrip()
                if len(pt) < 50 and not pt.endswith(TERMINAL) and not CAPTION_RE.match(b.text):
                    p.lines += b.lines
                    p.full_last = b.full_last
                    p.region, p.page = b.region, b.page
                    c.merged_paragraphs += 1
                    continue
        elif prev.kind == "table":
            if prev.table and _para_duplicates_table(b, prev.table):
                continue
            p = _find_last_para(out)
            if p is not None and _try_merge(p, b, c):
                continue
        elif prev.kind == "para" and (
                _short_fragment(prev) or _short_fragment(b)
                or prev.text.rstrip().endswith(("-", "­"))
                or (not prev.text.rstrip().endswith(TERMINAL) and _starts_lower(b.text))
            ) and _try_merge(prev, b, c):
            continue
        out.append(b)
    return out


_SECTION_NUM = re.compile(r"^\s*\d+(\.\d+)*[\s.]")


def _is_real_heading(b: Block) -> bool:
    t = b.text.strip()
    return bool(SECTION_WORDS.match(t) or _SECTION_NUM.match(t))


_SEPARATOR_RE = re.compile(r"^[\s—–\-_=~·•.─━]{5,}$")


def detect_separators(blocks: list[Block]) -> list[Block]:
    """Convert paragraphs of repeated dashes/rules to separator blocks."""
    for b in blocks:
        if b.kind == "para" and b.lines and _SEPARATOR_RE.match(b.text.strip()):
            b.kind = "separator"
    return blocks


def merge_title_page(blocks: list[Block]) -> list[Block]:
    """Merge short author/affiliation/email fragments on the first page into one block."""
    if not blocks:
        return blocks
    first_page = blocks[0].page
    title_idx = None
    end = None
    for i, b in enumerate(blocks):
        if b.page != first_page:
            end = i
            break
        if b.kind == "heading" and title_idx is None:
            if _is_real_heading(b):
                return blocks
            title_idx = i
            continue
        if title_idx is not None:
            if b.kind == "heading" and _is_real_heading(b):
                end = i
                break
            if b.kind == "para" and len(b.text) >= 120:
                end = i
                break
    if title_idx is None:
        return blocks
    if end is None:
        end = i + 1
    start = title_idx + 1
    run = [(j, blocks[j]) for j in range(start, end)
           if blocks[j].page == first_page and blocks[j].kind in ("para", "heading")
           and len(blocks[j].text.rstrip()) < 80]
    if len(run) < 3:
        return blocks
    merged = run[0][1]
    merged.kind = "para"
    for _, b in run[1:]:
        merged.lines += b.lines
    merged.full_last = run[-1][1].full_last
    remove = {j for j, _ in run[1:]}
    return [b for i, b in enumerate(blocks) if i not in remove]


def build_lists(blocks: list[Block], c: Counters) -> list[Block]:
    i, n = 0, len(blocks)
    while i < n:
        b = blocks[i]
        if b.kind == "para" and b.marker:
            j = i
            kind = b.marker[0]
            run = [b]
            while j + 1 < n and blocks[j + 1].kind == "para" and blocks[j + 1].marker \
                    and blocks[j + 1].marker[0] == kind:
                j += 1
                run.append(blocks[j])
            groups: list[list[Block]] = []
            if kind == "bullet":
                groups = [run]
            else:   # split ordered runs at numbering discontinuities
                g = [run[0]]
                for x in run[1:]:
                    if x.marker[2] == g[-1].marker[2] + 1:
                        g.append(x)
                    else:
                        groups.append(g); g = [x]
                groups.append(g)
            for g in groups:
                raw = g[0].marker[3]
                valid = len(g) >= 2 or (kind == "bullet" and raw in SOLID_BULLETS) \
                    or (kind == "bullet" and raw == "•")
                if kind in ("alpha", "roman") and len(g) < 2:
                    valid = False
                if kind == "ordered" and len(g) < 2:
                    valid = False
                if not valid:
                    for x in g:
                        x.marker = None
                    continue
                c.lists += 1
                stack: list[float] = []
                for x in g:
                    tol = 0.8 * x.lines[0].size
                    while stack and x.indent < stack[-1] - tol:
                        stack.pop()
                    if not stack or x.indent > stack[-1] + tol:
                        stack.append(x.indent)
                    x.kind, x.level = "li", len(stack) - 1
                    x.ordered = kind != "bullet"
                    x.number = x.marker[2]
                    _strip_marker(x)
            i = j + 1
        else:
            i += 1
    return blocks


def detect_naked_lists(blocks: list[Block], c: Counters) -> list[Block]:
    """Convert runs of short single-line paragraphs after 'See also'-style headings to list items."""
    i, n = 0, len(blocks)
    while i < n:
        b = blocks[i]
        if b.kind == "heading" and _NAKED_LIST_HEADING.match(b.text.strip()):
            hlevel = b.level
            j = i + 1
            # skip sub-headings (may be false positives from bold style propagation)
            while j < n and blocks[j].kind == "heading" and blocks[j].level > hlevel:
                j += 1
            start = j
            while j < n and blocks[j].kind == "para" and len(blocks[j].lines) <= 2:
                t = blocks[j].text.strip()
                if not t or len(t) > 120:
                    break
                j += 1
            run_len = j - start
            if run_len >= 3:
                c.lists += 1
                for k in range(start, j):
                    blocks[k].kind = "li"
                    blocks[k].level = 0
                    blocks[k].ordered = False
            i = j
        else:
            i += 1
    return blocks


def detect_definition_lists(blocks: list[Block], c: Counters) -> list[Block]:
    """Convert runs of 'bold term: description' paragraphs to list items."""
    i, n = 0, len(blocks)
    while i < n:
        b = blocks[i]
        if b.kind == "para" and _is_defn_entry(b):
            start = i
            while i < n and blocks[i].kind == "para" and _is_defn_entry(blocks[i]):
                i += 1
            if i - start >= 3:
                c.lists += 1
                for k in range(start, i):
                    blocks[k].kind = "dl"
                    blocks[k].level = 0
                    blocks[k].ordered = False
            continue
        i += 1
    return blocks


def _is_defn_entry(b: Block) -> bool:
    if not b.lines or not b.lines[0].spans:
        return False
    s0 = b.lines[0].spans[0]
    if not s0.bold or not s0.text.rstrip().endswith(":"):
        return False
    if len(b.lines[0].spans) < 2:
        return False
    return True


def _strip_marker(b: Block) -> None:
    n = b.marker[1]
    first = b.lines[0]
    spans, left = [], n
    # marker length was measured on the line text, which may start with whitespace
    for s in first.spans:
        if left <= 0:
            spans.append(s)
            continue
        if len(s.text) <= left:
            left -= len(s.text)
            continue
        spans.append(replace(s, text=s.text[left:]))
        left = 0
    first.spans = spans


_CONTENTS_RE = re.compile(
    r"^(contents|table of contents|inhaltsverzeichnis|table des matières|índice|sommario)$", re.I)


def _detect_heading_toc(blocks: list[Block]) -> list[Block]:
    """Convert heading-only inline TOCs (e.g. Wikipedia) into toc blocks.

    When a "Contents" heading is followed by numbered headings whose titles
    also appear later in the document, collapse the run into a toc block.
    """
    contents_idx = None
    for i, b in enumerate(blocks):
        if b.kind == "heading" and _CONTENTS_RE.match(b.text.strip()):
            contents_idx = i
            break
    if contents_idx is None:
        return blocks
    run_start = contents_idx + 1
    run_end = run_start
    for j in range(run_start, len(blocks)):
        if blocks[j].kind != "heading":
            break
        run_end = j + 1
    run_len = run_end - run_start
    if run_len < 3:
        return blocks
    later_titles = set()
    for b in blocks[run_end:]:
        if b.kind == "heading":
            later_titles.add(re.sub(r"^\d+(\.\d+)*\s+", "", b.text.strip()).lower())
    matched = 0
    entries = []
    for j in range(run_start, run_end):
        t = blocks[j].text.strip()
        stripped = re.sub(r"^\d+(\.\d+)*\s+", "", t).lower()
        if stripped in later_titles:
            matched += 1
        entries.append((t, ""))
    if matched < run_len * 0.5:
        return blocks
    toc_block = Block("toc", blocks[run_start].page, entries=entries, y0=blocks[run_start].y0)
    return blocks[:contents_idx + 1] + [toc_block] + blocks[run_end:]


_SECTION_NUM_ONLY = re.compile(r"^\s*\d+(\.\d+)*\s*$")
_DOTLEADER_RE = re.compile(r"^(?P<t>.*?\S)\s*(?:\.[\s.]{5,}|·{3,}|…{2,})\s*\.?\s*$")


def detect_toc(blocks: list[Block], c: Counters) -> list[Block]:
    out: list[Block] = []
    i = 0
    while i < len(blocks):
        run, j, skipped = [], i, 0
        while j < len(blocks):
            bk = blocks[j]
            if bk.kind not in ("para", "heading"):
                break
            t = bk.text.strip()
            if len(bk.lines) > 3 and ". . ." in t:
                for ln in bk.lines:
                    lt = ln.text.strip()
                    lm = TOC_LINE_RE.match(lt)
                    if lm:
                        run.append((lm.group("t").strip(" .·…"), lm.group("n")))
                    elif _DOTLEADER_RE.match(lt):
                        run.append((_DOTLEADER_RE.match(lt).group("t").strip(" .·…"), ""))
                    elif _SECTION_NUM_ONLY.match(lt):
                        pass
                j += 1
                continue
            if len(bk.lines) > 3:
                break
            m = TOC_LINE_RE.match(t)
            if m:
                title = m.group("t").strip(" .·…")
                if ". . ." in title or "·· " in title:
                    for sm in re.finditer(
                        r"(.*?\S)\s*(?:\.[\s.]{2,}|·{2,}|…+)\s*(\d{1,4})", t
                    ):
                        run.append((sm.group(1).strip(" .·…"), sm.group(2)))
                elif len(bk.lines) > 1 and "  " in title:
                    per_line: list[tuple[str, str]] = []
                    for ln in bk.lines:
                        lt = ln.text.strip()
                        lm = TOC_LINE_RE.match(lt)
                        if lm:
                            per_line.append((lm.group("t").strip(" .·…"), lm.group("n")))
                    if len(per_line) >= 2:
                        run.extend(per_line)
                    else:
                        run.append((title, m.group("n")))
                else:
                    run.append((title, m.group("n")))
                j += 1
            elif _DOTLEADER_RE.match(t):
                run.append((_DOTLEADER_RE.match(t).group("t").strip(" .·…"), ""))
                j += 1
            elif _SECTION_NUM_ONLY.match(t) and skipped < len(run) + 3:
                skipped += 1
                j += 1
            elif run and ". . ." in t and len(t) < 120:
                skipped += 1
                j += 1
            elif bk.kind == "heading" and bk.level > 1 and len(run) >= 3 and len(t) < 80:
                skipped += 1
                j += 1
            else:
                break
        avg_len = sum(len(t) for t, _ in run) / max(len(run), 1) if run else 0
        min_entries = 5 if avg_len < 25 else 3
        if len(run) >= min_entries:
            c.toc_pages_entries += len(run)
            out.append(Block("toc", blocks[i].page, entries=run, y0=blocks[i].y0))
            i = j
        else:
            out.append(blocks[i])
            i += 1
    return out


_INDEX_HEADING = re.compile(r"^index$", re.I)
_INDEX_LETTER_RE = re.compile(r"^[A-Z]$")
_INDEX_PAGE_CONT = re.compile(r"^\s*\d{1,4}(?:\s*[,–-]\s*\d{1,4})*\s*,?\s*$")
_INDEX_ENTRY_TRAIL = re.compile(r"^(.{2,50}?)\s+(\d{1,4}(?:[,–-]\s*\d{1,4})*)\s*,\s*$")


def _index_line_texts(b: Block) -> list[str]:
    """Extract individual line texts from a block for index processing."""
    if b.lines and len(b.lines) > 1:
        return [" ".join(s.text for s in ln.spans).strip() for ln in b.lines]
    return [b.text.strip()]


def _clean_index_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().rstrip(",").strip())


def detect_index(blocks: list[Block]) -> list[Block]:
    """Merge back-of-book index entries into compact list blocks."""
    if len(blocks) < 20:
        return blocks
    last_page = max(b.page for b in blocks)
    cutoff_page = last_page - max(last_page // 5, 3)
    idx_start = None
    for i in range(len(blocks) - 1, -1, -1):
        b = blocks[i]
        if b.kind == "heading" and _INDEX_HEADING.match(b.text.strip()) and b.page >= cutoff_page:
            match_count = 0
            for j in range(i + 1, min(i + 15, len(blocks))):
                if blocks[j].kind != "para":
                    continue
                for lt in _index_line_texts(blocks[j]):
                    if INDEX_ENTRY_RE.match(lt) or _INDEX_LETTER_RE.match(lt):
                        match_count += 1
            if match_count >= 4:
                idx_start = i + 1
                break
    if idx_start is None:
        return blocks
    idx_page = blocks[idx_start].page if idx_start < len(blocks) else last_page
    out = blocks[:idx_start]
    run: list[tuple[str, str]] = []
    run_page = idx_page
    j = idx_start
    while j < len(blocks):
        b = blocks[j]
        if b.kind == "para" and b.page >= idx_page:
            for lt in _index_line_texts(b):
                if not lt:
                    continue
                if _INDEX_PAGE_CONT.match(lt) and run:
                    prev_name, prev_pages = run[-1]
                    nums = re.sub(r"\s+", " ", lt.strip().rstrip(",").strip())
                    base = prev_pages.rstrip(",").rstrip()
                    run[-1] = (prev_name, (base + ", " + nums) if base else nums)
                    continue
                m = INDEX_ENTRY_RE.match(lt)
                if m:
                    run.append((_clean_index_name(m.group(1)), m.group(2)))
                    continue
                m = _INDEX_ENTRY_TRAIL.match(lt)
                if m:
                    run.append((_clean_index_name(m.group(1)), m.group(2)))
                    continue
                if _INDEX_LETTER_RE.match(lt):
                    pass
                elif re.match(r"^\(?[Ss]ee\b", lt):
                    run.append((lt, ""))
                else:
                    run.append((lt, ""))
            j += 1
            continue
        if b.kind == "heading" and _INDEX_HEADING.match(b.text.strip()):
            j += 1
            continue
        if run:
            out.append(Block("index", run_page, entries=run, y0=blocks[idx_start].y0))
            run = []
        out.append(b)
        run_page = b.page
        j += 1
    if run:
        out.append(Block("index", run_page, entries=run, y0=blocks[idx_start].y0))
    return out


_TABLE_CAPTION_RE = re.compile(r"^\s*(?:table|tab\.)\s", re.I)
_FIG_CAPTION_RE = re.compile(r"^\s*(?:fig(?:ure)?s?\.?|chart|diagram|photo|plate|scheme)\s", re.I)


def attach_captions(blocks: list[Block], body_size: float = 0) -> list[Block]:
    drop: set[int] = set()
    for i, b in enumerate(blocks):
        if b.kind not in ("image", "table") or (b.image is not None and b.image.kind == "equation"):
            continue
        if b.kind == "image":
            candidates = [i + d for d in (1, 2, 3, 4, 5, -1, -2, -3, -4, -5)]
        else:
            candidates = [i + d for d in (-1, -2, -3, -4, -5, 1, 2, 3, 4, 5)]
        for j in candidates:
            if 0 <= j < len(blocks) and j not in drop and blocks[j].kind == "para" \
                    and (CAPTION_RE.match(blocks[j].text) or blocks[j].ml_label == "caption") \
                    and len(blocks[j].lines) <= 14 \
                    and len(blocks[j].text) <= 1400 and blocks[j].page == b.page:
                ct = blocks[j].text
                if b.kind == "table" and _FIG_CAPTION_RE.match(ct):
                    continue
                if b.kind == "image" and _TABLE_CAPTION_RE.match(ct):
                    continue
                b.caption = blocks[j].lines
                drop.add(j)
                break

    if body_size > 0:
        _attach_captions_by_size(blocks, drop, body_size)
    _drop_figure_labels(blocks, drop)
    _drop_table_annotations(blocks, drop, body_size)
    return [b for k, b in enumerate(blocks) if k not in drop]


def _avg_span_size(b: Block) -> float:
    sizes = [s.size for l in b.lines for s in l.spans if s.text.strip()]
    return sum(sizes) / len(sizes) if sizes else 999


def _attach_captions_by_size(blocks: list[Block], drop: set[int], body_size: float) -> None:
    """Attach captions to images based on font size (smaller than body = caption)."""
    img_positions = {k for k, bl in enumerate(blocks)
                     if bl.kind == "image" and not bl.caption
                     and (bl.image is None or bl.image.kind != "equation")
                     and (bl.image is None or min(bl.image.width, bl.image.height) >= 50)}
    for i in sorted(img_positions):
        b = blocks[i]
        if b.caption:
            continue
        cap_lines = []
        cap_idxs = []
        for d in range(-10, 11):
            j = i + d
            if j == i or j < 0 or j >= len(blocks) or j in drop:
                continue
            c = blocks[j]
            if c.kind != "para" or c.page != b.page or len(c.lines) > 3:
                continue
            if _avg_span_size(c) > body_size - 2.0:
                continue
            closer_img = any(k != i and k in img_positions and abs(j - k) < abs(j - i)
                             for k in img_positions)
            if closer_img:
                continue
            cap_lines.extend(c.lines)
            cap_idxs.append(j)
        if cap_idxs and sum(1 for j in cap_idxs if len(blocks[j].text.strip()) <= 5) > len(cap_idxs) / 2:
            continue
        merged = " ".join(s.text for l in cap_lines for s in l.spans).strip()
        if len(merged) >= 15 and len(merged) <= 500 and any(ch.isalpha() for ch in merged):
            b.caption = cap_lines
            drop.update(cap_idxs)


_FIGURE_LABEL_RE = re.compile(r"^[\d.,\s%]+$|^[A-Za-z]$")


def _drop_figure_labels(blocks: list[Block], drop: set[int]) -> None:
    for i, b in enumerate(blocks):
        if b.kind != "image" or i in drop:
            continue
        if b.image is not None and b.image.kind == "equation":
            continue
        if b.image is not None and min(b.image.width, b.image.height) < 50:
            continue
        for d in (1, -1):
            run: list[int] = []
            j = i + d
            while 0 <= j < len(blocks) and j not in drop:
                c = blocks[j]
                if c.page != b.page or c.kind != "para":
                    break
                t = c.text.strip()
                if len(t) > 25:
                    break
                if not _FIGURE_LABEL_RE.match(t) and len(t.split()) >= 3 and t.rstrip().endswith(TERMINAL):
                    break
                run.append(j)
                j += d
            if len(run) >= 2:
                drop.update(run)


def _drop_table_annotations(blocks: list[Block], drop: set[int], body_size: float) -> None:
    """Drop short, small-font paragraphs adjacent to tables or images (escaped content)."""
    if body_size <= 0:
        return
    anchor_pages: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        if i in drop:
            continue
        if b.kind == "table" or (b.kind == "image" and b.image and b.image.kind != "equation"):
            anchor_pages.setdefault(b.page, []).append(i)
    threshold = body_size * 0.85
    for i, b in enumerate(blocks):
        if i in drop or b.kind != "para" or b.page not in anchor_pages:
            continue
        t = b.text.strip()
        if len(t) >= 40 or t.rstrip().endswith(TERMINAL):
            continue
        avg = _avg_span_size(b)
        if avg >= threshold:
            continue
        near = any(abs(i - ai) <= 5 for ai in anchor_pages[b.page])
        if near:
            drop.add(i)


def detect_title(blocks: list[Block], st: DocStats) -> str | None:
    """Largest-type block on the first page; promoted to <h1> when it is clearly a title."""
    if not blocks:
        return None
    first_page = blocks[0].page
    best, best_size = None, st.body_size * 1.5
    for b in blocks:
        if b.page != first_page:
            break
        if b.kind in ("para", "heading") and b.lines and len(b.lines) <= 3 and len(b.text) <= 200:
            sz = max(l.size for l in b.lines)
            if sz > best_size + 0.3:
                best, best_size = b, sz
    if best is None:
        return None
    if best.kind == "para":
        best.kind, best.level, best.marker = "heading", 1, None
    return re.sub(r"\s+", " ", best.text).strip()


def demote_pseudo_code(blocks: list[Block]) -> list[Block]:
    """Monospace e-mail addresses / URLs / paths are inline code, not code blocks."""
    for b in blocks:
        if b.kind == "code" and all(CODE_ISH_RE.match(l.text.strip()) for l in b.lines):
            b.kind = "para"
    return blocks


# ------------------------------------------------------------------ algorithms

_ALGO_HEADER = re.compile(r"^\s*Algorithm\s+\d+", re.I)
_ALGO_KEYWORD = re.compile(
    r"^\s*(Require|Ensure|Input|Output|Procedure|Function|Initialize|Initialise)\s*:", re.I)
_ALGO_LINE = re.compile(
    r"←|\\leftarrow|:=|\bdo\b|\bthen\b|\belse\b|\bend\s+(if|for|while|loop|function|procedure)\b"
    r"|\breturn\b|\bwhile\b.*\bdo\b|\bfor\b.*\bdo\b|\bif\b.*\bthen\b", re.I)


def _is_pseudocode(b: Block) -> bool:
    if b.kind not in ("para", "algorithm"):
        return False
    t = b.text.strip()
    if _ALGO_KEYWORD.match(t):
        return True
    if _ALGO_LINE.search(t) and len(t) < 200:
        return True
    if "←" in t and len(t) < 200:
        return True
    return False


def _algo_scannable(b: Block) -> bool:
    if b.kind in ("para", "algorithm"):
        return True
    return b.kind == "image" and b.image is not None and b.image.kind == "equation"


_ALGO_TERMINATOR = re.compile(
    r"^\s*(return\b|end\s+(if|for|while|loop|function|procedure))", re.I)


def _algo_should_stop(blocks: list[Block], j: int, saw_terminator: bool) -> bool:
    bj = blocks[j]
    if not bj.lines:
        return True
    if saw_terminator:
        return True
    prev_y1 = None
    for k in range(j - 1, -1, -1):
        if blocks[k].lines:
            prev_y1 = blocks[k].lines[-1].y1
            break
        if blocks[k].kind == "image" and blocks[k].image:
            prev_y1 = blocks[k].image.bbox[3]
            break
    if prev_y1 is None:
        return False
    return bj.y0 - prev_y1 > bj.lines[0].size * 3


def detect_algorithms(blocks: list[Block]) -> list[Block]:
    """Detect algorithm/pseudocode blocks by structural patterns (Require/←/while...do)."""
    out: list[Block] = []
    i, n = 0, len(blocks)
    while i < n:
        b = blocks[i]
        if b.kind in ("para", "heading") and _ALGO_HEADER.match(b.text.strip()):
            is_heading = b.kind == "heading"
            run_start = i + 1 if is_heading else i
            j = run_start
            pseudocode_count = 0
            saw_term = False
            while j < n and _algo_scannable(blocks[j]):
                if _is_pseudocode(blocks[j]):
                    pseudocode_count += 1
                    if _ALGO_TERMINATOR.match(blocks[j].text.strip()):
                        saw_term = True
                elif blocks[j].kind == "image":
                    pass
                elif pseudocode_count == 0:
                    break
                elif _algo_should_stop(blocks, j, saw_term):
                    break
                else:
                    saw_term = False
                j += 1
            if pseudocode_count >= 2 and j > run_start:
                if is_heading:
                    out.append(b)
                merged = Block("algorithm", blocks[run_start].page,
                               region=blocks[run_start].region,
                               y0=blocks[run_start].y0)
                for k in range(run_start, j):
                    if blocks[k].kind != "image":
                        merged.lines.extend(blocks[k].lines)
                    else:
                        out.append(blocks[k])
                out.append(merged)
                i = j
                continue
        elif b.kind == "para" and _ALGO_KEYWORD.match(b.text.strip()):
            j = i
            pseudocode_count = 0
            saw_term = False
            while j < n and _algo_scannable(blocks[j]):
                if _is_pseudocode(blocks[j]):
                    pseudocode_count += 1
                    if _ALGO_TERMINATOR.match(blocks[j].text.strip()):
                        saw_term = True
                elif blocks[j].kind == "image":
                    pass
                elif pseudocode_count > 0:
                    if _algo_should_stop(blocks, j, saw_term):
                        break
                    saw_term = False
                else:
                    break
                j += 1
            if pseudocode_count >= 3 and j > i:
                merged = Block("algorithm", blocks[i].page,
                               region=blocks[i].region,
                               y0=blocks[i].y0)
                for k in range(i, j):
                    if blocks[k].kind != "image":
                        merged.lines.extend(blocks[k].lines)
                    else:
                        out.append(blocks[k])
                out.append(merged)
                i = j
                continue
        out.append(b)
        i += 1
    return out


# ------------------------------------------------------------------ bibliography

_BIB_HEADING = re.compile(
    r"^(references|bibliography|works cited|cited literature"
    r"|literaturverzeichnis|literatur|einzelnachweise|quellenangaben"
    r"|endnotes|notes|bibliograf[ií]a|r[ée]f[ée]rences|riferimenti bibliografici)(\s*\(.*\))?$", re.I)
_BIB_MARKER = re.compile(r"^\s*\[(\d+)\]\s*")
_BIB_ALPHA_MARKER = re.compile(r"^\s*\[([A-Z][A-Za-z+\d]{1,20})\]\s*")
_BIB_DOT_MARKER = re.compile(r"^\s*(\d{1,4})\.\s+")
_BIB_YEAR = re.compile(r"\b((?:19|20)\d{2}[a-z]?)\b")


_BIB_AUTHOR_START = re.compile(
    r"^[A-ZÀ-ɏ][\wÀ-ɏ .'-]+,"
)


def _bib_author_key(text: str) -> str:
    """Extract 'lastname+year' key from an author-year bibliography entry."""
    t = text.strip()
    ym = _BIB_YEAR.search(t[:250])
    year = ym.group(1) if ym else ""
    comma = t.find(",")
    if 0 < comma < (ym.start() if ym else len(t)):
        before = t[:comma].split()
    else:
        end = ym.start() if ym else len(t)
        before = t[:end].split()
    name = before[-1].rstrip(".,;:").lower() if before else "ref"
    if len(before) == 1:
        name = before[0].rstrip(".,;:").lower()
    return f"{name}{year}"


def _format_bib_authoryear(blocks: list[Block], ref_idx: int, section_id: int, n: int) -> bool:
    """Detect author-year bibliography entries (no markers) and format as list items."""
    entries: list[int] = []
    for i in range(ref_idx + 1, n):
        b = blocks[i]
        if b.kind == "heading":
            break
        if b.kind != "para":
            continue
        t = b.text.strip()
        if _BIB_AUTHOR_START.match(t) and _BIB_YEAR.search(t[:250]):
            entries.append(i)
    if len(entries) < 5:
        return False
    entry_set = set(entries)
    seen_keys: dict[str, int] = {}
    for i in entries:
        b = blocks[i]
        b.kind = "li"
        b.ordered = False
        b.number = None
        b.level = 1
        key = _bib_author_key(b.text)
        cnt = seen_keys.get(key, 0)
        seen_keys[key] = cnt + 1
        b.anchor = f"bib{section_id}-{key}" if cnt == 0 else f"bib{section_id}-{key}-{cnt}"
    bib_start, bib_end = entries[0], entries[-1] + 1
    for i in range(bib_start, bib_end):
        b = blocks[i]
        if i in entry_set or b.kind != "para":
            continue
        prev_li = None
        for j in range(i - 1, bib_start - 1, -1):
            if blocks[j].kind == "li":
                prev_li = blocks[j]
                break
        if prev_li is not None:
            prev_li.lines.extend(b.lines)
            b.kind = "_drop"
    return True


def _format_bib_section(blocks: list[Block], ref_idx: int, section_id: int) -> bool:
    """Try to format a single bibliography section starting at ref_idx. Returns True if modified."""
    n = len(blocks)
    numeric_count, alpha_count, dot_count = 0, 0, 0
    for i in range(ref_idx + 1, min(ref_idx + 20, n)):
        if blocks[i].kind == "heading":
            break
        if blocks[i].kind != "para":
            continue
        if _BIB_MARKER.match(blocks[i].text):
            numeric_count += 1
        elif _BIB_ALPHA_MARKER.match(blocks[i].text):
            alpha_count += 1
        elif _BIB_DOT_MARKER.match(blocks[i].text):
            dot_count += 1
    best = max(numeric_count, alpha_count, dot_count)
    if best == 0:
        return _format_bib_authoryear(blocks, ref_idx, section_id, n)
    if numeric_count >= best:
        marker_re = _BIB_MARKER
    elif alpha_count >= best:
        marker_re = _BIB_ALPHA_MARKER
    else:
        marker_re = _BIB_DOT_MARKER
    is_numeric = marker_re is not _BIB_ALPHA_MARKER

    bib_entries: list[int] = []
    last_num = 0
    for i in range(ref_idx + 1, n):
        b = blocks[i]
        if b.kind == "heading":
            break
        if b.kind != "para":
            continue
        m = marker_re.match(b.text)
        if m:
            if is_numeric:
                num = int(m.group(1))
                if num > last_num:
                    bib_entries.append(i)
                    last_num = num
            else:
                bib_entries.append(i)

    if len(bib_entries) < 3:
        return False

    bib_set = set(bib_entries)
    for i in bib_entries:
        b = blocks[i]
        m = marker_re.match(b.text)
        if not m:
            continue
        b.kind = "li"
        b.ordered = False
        b.number = None
        b.level = 1
        b.anchor = f"bib{section_id}-{m.group(1)}"

    bib_start, bib_end = bib_entries[0], bib_entries[-1] + 1
    for i in range(bib_start, bib_end):
        b = blocks[i]
        if i in bib_set or b.kind != "para":
            continue
        prev_li = None
        for j in range(i - 1, bib_start - 1, -1):
            if blocks[j].kind == "li":
                prev_li = blocks[j]
                break
        if prev_li is not None:
            prev_li.lines.extend(b.lines)
            b.kind = "_drop"
    return True


def format_bibliography(blocks: list[Block]) -> list[Block]:
    """Detect [N]-style bibliography entries after 'References' headings and format as lists."""
    modified = False
    section_id = 0
    for i, b in enumerate(blocks):
        if b.kind == "heading" and _BIB_HEADING.match(b.text.strip()):
            if _format_bib_section(blocks, i, section_id):
                modified = True
                section_id += 1
    if not modified:
        return blocks
    return [b for b in blocks if b.kind != "_drop"]
