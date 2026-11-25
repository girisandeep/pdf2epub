"""Reading-order recovery: a column-aware XY-cut that tolerates spanning elements.

Classic XY-cut interleaves columns whenever paragraph gaps happen to line up, or when a
full-width figure sits between column text.  Here we instead find a vertical cut using only
"narrow" items, treat everything that crosses the cut as a spanning separator, and read each
band between separators left column first, then right column, recursively.
"""
from __future__ import annotations

from statistics import median

from .model import Item, Line


def _words(it: Item) -> float:
    return len(it.text.split()) if isinstance(it, Line) else 6.0


def _find_cut(items: list[Item]) -> float | None:
    """Locate a vertical column gutter.

    For every x interval between item edges we count how many items cross it.  A gutter is a run
    of intervals (wide enough, in the central part of the region) whose crossing count is minimal;
    the few items that still cross it (titles, abstracts, figures) are the "spanning" items.
    """
    if len(items) < 4:
        return None
    x0 = min(i.x0 for i in items)
    x1 = max(i.x1 for i in items)
    extent = x1 - x0
    if extent < 150:
        return None
    sizes = [i.size for i in items if isinstance(i, Line)]
    g_min = max(10.0, 0.9 * (median(sizes) if sizes else 10.0))
    edges = sorted({round(i.x0, 1) for i in items} | {round(i.x1, 1) for i in items})
    ivals = list(zip(edges, edges[1:]))
    lo, hi = x0 + 0.2 * extent, x1 - 0.2 * extent
    cross = [sum(1 for i in items if i.x0 < (a + b) / 2 < i.x1) for a, b in ivals]
    central = [(iv, c) for iv, c in zip(ivals, cross) if lo <= (iv[0] + iv[1]) / 2 <= hi]
    for thr in sorted({c for _, c in central}):
        runs: list[list[float]] = []
        cur = None
        for (a, b), c in central:
            if c <= thr:
                if cur is not None and abs(cur[1] - a) < 0.2:
                    cur[1] = b
                else:
                    cur = [a, b]
                    runs.append(cur)
            else:
                cur = None
        for a, b in sorted((r for r in runs if r[1] - r[0] >= g_min), key=lambda r: r[0] - r[1]):
            cut = (a + b) / 2
            left = [i for i in items if i.x1 <= a + 0.2]
            right = [i for i in items if i.x0 >= b - 0.2]
            span = [i for i in items if i.x0 < cut < i.x1]
            if len(left) < 2 or len(right) < 2:
                continue
            if sum(_words(i) for i in left) / len(left) < 3 or sum(_words(i) for i in right) / len(right) < 3:
                continue
            ly0, ly1 = min(i.y0 for i in left), max(i.y1 for i in left)
            ry0, ry1 = min(i.y0 for i in right), max(i.y1 for i in right)
            if min(ly1, ry1) - max(ly0, ry0) < 0.3 * min(ly1 - ly0, ry1 - ry0):
                continue
            if len(span) > 0.5 * (len(left) + len(right)):
                continue
            return cut
    return None


class _Counter:
    def __init__(self, start: int = 0):
        self.n = start

    def next(self) -> int:
        self.n += 1
        return self.n


def _leaf(items: list[Item], ctr: _Counter) -> list[Item]:
    rid = ctr.next()
    items = sorted(items, key=lambda i: (round(i.y0), i.x0))
    for it in items:
        it.region = rid
    return items


def _columns(items: list[Item], cut: float, ctr: _Counter) -> list[Item]:
    left = [i for i in items if (i.x0 + i.x1) / 2 < cut]
    right = [i for i in items if (i.x0 + i.x1) / 2 >= cut]
    out: list[Item] = []
    for side in (left, right):
        if side:
            out += _order(side, ctr)
    return out


def _order(items: list[Item], ctr: _Counter) -> list[Item]:
    cut = _find_cut(items)
    if cut is None:
        return _leaf(items, ctr)
    spanning = sorted((i for i in items if i.x0 < cut < i.x1), key=lambda i: (i.y0, i.x0))
    rest = [i for i in items if not (i.x0 < cut < i.x1)]
    out: list[Item] = []
    span_rid = None
    for s in spanning:
        cy = (s.y0 + s.y1) / 2
        before = [o for o in rest if (o.y0 + o.y1) / 2 < cy]
        if before:
            rest = [o for o in rest if o not in before]
            out += _columns(before, cut, ctr)
            span_rid = None
        if span_rid is None:
            span_rid = ctr.next()
        s.region = span_rid
        out.append(s)
    if rest:
        out += _columns(rest, cut, ctr)
    return out


def reading_order(items: list[Item], first_region: int = 0) -> tuple[list[Item], int]:
    ctr = _Counter(first_region)
    ordered = _order(list(items), ctr) if items else []
    return ordered, ctr.n
