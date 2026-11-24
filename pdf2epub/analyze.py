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
FN_SYMBOLS = "*†‡§¶∗⋆★"
FN_MARK_RE = re.compile(rf"^\s*(\d{{1,3}}|[{FN_SYMBOLS}])(?=[\s.)\]]|$)")
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
                           r"acknowledge?ments?|references|bibliography|appendix|index|glossary|dedication|"
                           r"prologue|epilogue|conclusions?)\b", re.I)
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
                key = (norm_key(l.text), round(l.y0 / 6))
                if key[0]:
                    cnt[key] += 1
                    zones[p.number].append((l, key))
    n = len(pages)
    for p in pages:
        drop = set()
        for l, key in zones.get(p.number, []):
            pn = bool(PAGE_NUM_RE.match(l.text))
            if cnt[key] >= 3 or (pn and (n <= 3 or cnt[key] >= 2)):
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
def extract_footnotes(page: PageData, stats: DocStats) -> list[FootnoteDef]:
    """Split a page's footnote area from its body, but only when body markers confirm it."""
    body = stats.body_size
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
    dy = base_y(ln) - base_y(prev)
    if dy < -0.3 * size:
        return True
    if ln.mono != prev.mono:
        return True
    if cur.kind == "code":
        return False
    if abs(ln.size - prev.size) > 0.8:
        return True
    if (ln.bold != prev.bold) and (ln.bold or prev.bold):
        return True
    complete = _ends_complete(prev)
    lm = list_marker(ln.text)
    if lm and (lm[0] == "bullet" or complete or dy > size * (st.leading + 0.1)):
        return True
    if dy > size * (st.leading + 0.28):
        return True
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
            cur = Block("code" if ln.mono else "para", page, [ln], region=ln.region, y0=ln.y0,
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
    return (sizes[len(sizes) // 2], all(l.bold for l in b.lines))


def _is_caps(b: Block) -> bool:
    t = re.sub(r"[^A-Za-z]", "", b.text)
    return len(t) >= 3 and t.isupper()


def _candidate(b: Block) -> bool:
    if b.kind != "para" or not b.lines or len(b.lines) > 4:
        return False
    t = b.text
    if len(t) > 200 or len(t.split()) > 28 or t.rstrip().endswith(","):
        return False
    return not any(l.mono for l in b.lines)


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
        if min(len(sa), len(sb)) >= 5 and (sa in sb or sb in sa):
            r = max(r, 0.9)
    return r


def classify_headings(blocks: list[Block], toc: list, st: DocStats, c: Counters) -> list[Block]:
    body = st.body_size
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
            if best is not None and best_r >= 0.82:
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
            if (bi > first_hit or SECTION_WORDS.match(b.text.strip())) and b.kind == "para" and _candidate(b):
                k = _bkey(b)
                if k in key_levels:
                    n = sum(key_levels[k].values())
                    lvl = key_levels[k].most_common(1)[0][0]
                    big = k[0] >= body * 1.15
                    if (big and n >= 1) or n >= 3:
                        if k[0] <= body * 1.05 and (b.text.rstrip().endswith(".") or CAPTION_RE.match(b.text)):
                            continue
                        mark(b, lvl)
    else:
        cands = [b for b in blocks if _candidate(b)]
        keys: dict[tuple, list[Block]] = defaultdict(list)
        total = max(len([b for b in blocks if b.kind == "para"]), 1)
        for b in cands:
            size, bold = _bkey(b)
            t = b.text
            if CAPTION_RE.match(t) and size <= body * 1.1:
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
    c.headings = sum(1 for b in blocks if b.kind == "heading")
    return blocks


# ------------------------------------------------------------ continuations/lists
def merge_continuations(blocks: list[Block], c: Counters) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        if out and b.kind == "para" and out[-1].kind == "para" and b.marker is None \
                and (b.region != out[-1].region or b.page != out[-1].page):
            p = out[-1]
            pl, nl = p.lines[-1], b.lines[0]
            if abs(pl.size - nl.size) < 0.8 and pl.bold == nl.bold:
                pend = p.text.rstrip()
                hy = pend.endswith(("-", "­"))
                term = pend.endswith(TERMINAL)
                lower = _starts_lower(b.text)
                if (hy and lower) or (not term and lower) or (not term and p.full_last and not b.indented_first):
                    p.lines += b.lines
                    p.full_last = b.full_last
                    p.region, p.page = b.region, b.page
                    c.merged_paragraphs += 1
                    continue
        out.append(b)
    return out


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


def detect_toc(blocks: list[Block], c: Counters) -> list[Block]:
    out: list[Block] = []
    i = 0
    while i < len(blocks):
        run, j = [], i
        while j < len(blocks) and blocks[j].kind == "para" and len(blocks[j].lines) <= 2:
            m = TOC_LINE_RE.match(blocks[j].text.strip())
            if not m:
                break
            run.append((m.group("t").strip(" .·…"), m.group("n")))
            j += 1
        if len(run) >= 3:
            c.toc_pages_entries += len(run)
            out.append(Block("toc", blocks[i].page, entries=run, y0=blocks[i].y0))
            i = j
        else:
            out.append(blocks[i])
            i += 1
    return out


def attach_captions(blocks: list[Block]) -> list[Block]:
    drop: set[int] = set()
    for i, b in enumerate(blocks):
        if b.kind not in ("image", "table") or (b.image is not None and b.image.kind == "equation"):
            continue
        order = (i + 1, i - 1) if b.kind == "image" else (i - 1, i + 1)
        for j in order:
            if 0 <= j < len(blocks) and j not in drop and blocks[j].kind == "para" \
                    and CAPTION_RE.match(blocks[j].text) and len(blocks[j].lines) <= 14 \
                    and len(blocks[j].text) <= 1400 and blocks[j].page == b.page:
                b.caption = blocks[j].lines
                drop.add(j)
                break
    return [b for k, b in enumerate(blocks) if k not in drop]


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
