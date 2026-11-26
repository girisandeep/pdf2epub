"""Blocks -> XHTML fragments (inline styling, dehyphenation, links, footnotes, lists, tables)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from xml.sax.saxutils import escape, quoteattr

from .analyze import FN_SYMBOLS, DocStats, base_y
from .model import Block, FootnoteDef, Line, Span
from .mathml import equation_number, to_mathml
from .text import _COMPOUND_PREFIX, english_words, known_word, norm_title, strip_numbering

URL_RE = re.compile(r"(?<![\w@/])((?:https?://|www\.)[^\s<>\"']+[^\s<>\"'.,;:!?)\]}])")
MAIL_RE = re.compile(r"(?<![\w.@])([\w.+-]+@[\w-]+(?:\.[\w-]+)+)")

# Cross-reference patterns: "Figure 1", "Fig. 2", "Table 3", "Eq. (4)", "Section 5.1", etc.
_XREF_RE = re.compile(
    r"(?<![.\w])"                               # not mid-word / decimal continuation
    r"(Fig(?:ure)?s?\.?|Tables?|Tab\.|"
    r"Eq(?:uation)?s?\.?|"
    r"(?:Sub)?[Ss]ections?|Sect?\.|§|"
    r"Chapters?|Parts?|"
    r"Appendi(?:x|ces)|Algorithms?|Theorems?|Lemmas?|"
    r"Corollar(?:y|ies)|Propositions?|Definitions?)"
    r"\s*"
    r"(?:\((\d+(?:\.\d+)*[a-z]?)\)|(\d+(?:\.\d+)*[a-z]?))"   # (4) or 1.2
    , re.I)

_NUM = r"\d+(?:\.\d+)*[a-z]?"
_XREF_MULTI_RE = re.compile(
    r"(?<![.\w])"
    r"(Fig(?:ure)?s?\.?|Tables?|Tab\.|"
    r"Eq(?:uation)?s?\.?|"
    r"(?:Sub)?[Ss]ections?|Sect?\.|§|"
    r"Chapters?|Parts?|"
    r"Appendi(?:x|ces)|Algorithms?|Theorems?|Lemmas?|"
    r"Corollar(?:y|ies)|Propositions?|Definitions?)"
    r"\s*"
    r"(" + _NUM + r"(?:\s*[,;]\s*" + _NUM + r")*"
    r"(?:\s+and\s+" + _NUM + r")?)"
    , re.I)

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_CITE_ALPHA_RE = re.compile(r"\[([A-Z][A-Za-z+\d]{1,20}(?:\s*[,;]\s*[A-Z][A-Za-z+\d]{1,20})*)\]")
_CITE_PAREN_RE = re.compile(r"\(([^)]{8,200})\)")
_CITE_SINGLE_RE = re.compile(
    r"([A-Z][\wÀ-ɏ'-]+)"
    r"(?:\s+(?:et\s+al\.|and|&)\s*,?\s*(?:[A-Z][\wÀ-ɏ'-]+)?)?"
    r",?\s*((?:19|20)\d{2}[a-z]?)\s*$"
)


@dataclass
class RenderCtx:
    stats: DocStats
    footnotes: dict[tuple[int, str], FootnoteDef] = field(default_factory=dict)
    image_href: dict[str, str] = field(default_factory=dict)       # image key -> href
    heading_targets: dict[str, tuple[str, str]] = field(default_factory=dict)  # norm title -> (file,id)
    xref_targets: dict[str, tuple[str, str]] = field(default_factory=dict)     # "figure 1" -> (file,id)
    fn_counter: int = 0
    current_file: str = ""


# ------------------------------------------------------------------ inline text
Seg = tuple[str, tuple]     # (text, style)  style=(bold,italic,mono,sup,sub,href)


def _style(s: Span) -> tuple:
    return (s.bold, s.italic, s.mono, s.sup, s.sub, s.href)


def _join_vocab(prefix_word: str, suffix_word: str, vocab) -> bool:
    """Decide whether a hyphen at a line break is typographic (drop it) or part of a compound (keep).

    Evidence order: the document's own vocabulary, then a dictionary, then a small prefix list.
    """
    pre, suf = prefix_word.lower(), suffix_word.lower()
    joined, hyph = pre + suf, f"{pre}-{suf}"
    if vocab.get(joined, 0):
        return True
    if vocab.get(hyph, 0):
        return False
    words = english_words()
    if words:
        if joined in words:
            return True
        return not (known_word(pre, words) and known_word(suf, words))
    return pre not in _COMPOUND_PREFIX


def line_segments(lines: list[Line], ctx: RenderCtx, strip_bold: bool = False) -> list[Seg]:
    segs: list[Seg] = []
    vocab = ctx.stats.vocab
    for li, ln in enumerate(lines):
        spans = [s for s in ln.spans]
        for s in spans:
            st = _style(s)
            if strip_bold:
                st = (False,) + st[1:]
            segs.append((s.text, st))
        if li + 1 < len(lines):
            nxt = lines[li + 1].text.lstrip()
            last = "".join(t for t, _ in segs).rstrip()
            if last.endswith(("-", "­")) and nxt[:1].islower() and len(last) > 1 and last[-2].isdigit():
                segs[-1] = (segs[-1][0].rstrip(), segs[-1][1])
                segs.append(("\x00", ("nospace",)))
                continue
            if last.endswith(("-", "­")) and nxt[:1].isalpha() and len(last) > 1 and last[-2].isalpha():
                pm = re.search(r"([^\W\d_]+)[-\u00ad]$", last)
                sm = re.match(r"[^\W\d_]+", nxt)
                t, st = segs[-1]
                t = t.rstrip()
                if last.endswith("\u00ad") or (nxt[:1].islower() and pm and sm
                                                and _join_vocab(pm.group(1), sm.group(), vocab)):
                    t = t[:-1]                      # typographic hyphen: drop it
                segs[-1] = (t, st)
                while segs and not segs[-1][0]:
                    segs.pop()
                segs.append(("\x00", ("nospace",)))   # and glue the next line on without a space
                continue
            if last.endswith(("https:", "http:")) and nxt.startswith("//"):
                segs.append(("\x00", ("nospace",)))
                continue
            segs.append((" ", (False, False, False, False, False, None)))
    # apply the "no space" markers
    out: list[Seg] = []
    skip_ws = False
    for t, st in segs:
        if t == "\x00":
            skip_ws = True
            continue
        if skip_ws:
            t = t.lstrip()
            if not t:
                continue
            skip_ws = False
        out.append((t, st))
    return out


def _wrap(text: str, st: tuple) -> str:
    bold, ital, mono, sup, sub, href = st
    t = escape(text)
    if mono: t = f"<code>{t}</code>"
    if ital: t = f"<em>{t}</em>"
    if bold: t = f"<strong>{t}</strong>"
    if sup: t = f"<sup>{t}</sup>"
    if sub: t = f"<sub>{t}</sub>"
    if href: t = f"<a href={quoteattr(href)}>{t}</a>"
    return t


def _linkify(escaped: str) -> str:
    escaped = URL_RE.sub(lambda m: f'<a href="{"http://" if m.group(1).startswith("www.") else ""}{m.group(1)}">{m.group(1)}</a>', escaped)
    return MAIL_RE.sub(lambda m: f'<a href="mailto:{m.group(1)}">{m.group(1)}</a>', escaped)


def _xref_key(kind: str, number: str) -> str:
    """Normalise a cross-reference mention to a lookup key."""
    k = kind.rstrip(".").lower()
    # Collapse synonyms to canonical forms
    if k.startswith("fig"):
        k = "figure"
    elif k.startswith("tab"):
        k = "table"
    elif k.startswith("eq"):
        k = "equation"
    elif k in ("§",) or k.startswith("sec") or k.startswith("subsec"):
        k = "section"
    elif k.startswith("chapter"):
        k = "section"
    elif k.startswith("part"):
        k = "section"
    elif k.startswith("appendi"):
        k = "appendix"
    elif k.startswith("algorithm"):
        k = "algorithm"
    elif k.startswith("theorem"):
        k = "theorem"
    elif k.startswith("lemma"):
        k = "lemma"
    elif k.startswith("corollar"):
        k = "corollary"
    elif k.startswith("proposition"):
        k = "proposition"
    elif k.startswith("definition"):
        k = "definition"
    return f"{k} {number}"


def _insert_xrefs(html: str, ctx: RenderCtx) -> str:
    """Replace cross-reference mentions in rendered HTML with internal links."""
    # Don't match inside HTML tags or existing <a> elements
    parts: list[str] = []
    pos = 0
    in_tag = False
    in_a = 0  # nesting depth of <a> elements

    # Split into tag / text segments to avoid matching inside tags
    TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*>")
    segments: list[tuple[str, bool]] = []  # (text, is_linkable)
    last = 0
    for m in TAG_RE.finditer(html):
        if m.start() > last:
            segments.append((html[last:m.start()], in_a == 0))
        tag_name = m.group(2).lower()
        if m.group(1):  # closing tag
            if tag_name == "a":
                in_a = max(0, in_a - 1)
        else:  # opening tag
            if tag_name == "a":
                in_a += 1
        segments.append((m.group(), False))  # tag itself is never linkable
        last = m.end()
    if last < len(html):
        segments.append((html[last:], in_a == 0))

    out: list[str] = []
    for text, linkable in segments:
        if not linkable:
            out.append(text)
            continue
        # Collect all linkable matches, then process in position order so that
        # an xref late in the paragraph doesn't block earlier citations.
        candidates: list[tuple[int, int, str]] = []  # (start, end, replacement_html)
        for m in _XREF_RE.finditer(text):
            number = m.group(2) or m.group(3)
            key = _xref_key(m.group(1), number)
            tgt = ctx.xref_targets.get(key)
            if not tgt:
                continue
            file, anchor = tgt
            href = f"#{anchor}" if file == ctx.current_file else f"{file}#{anchor}"
            candidates.append((m.start(), m.end(), f'<a href="{href}">{m.group()}</a>'))
        for m in _XREF_MULTI_RE.finditer(text):
            kind = m.group(1)
            nums_str = m.group(2)
            nums = re.findall(r"\d+(?:\.\d+)*[a-z]?", nums_str)
            if len(nums) < 2:
                continue
            parts_x: list[str] = []
            any_linked = False
            pos = 0
            for nm in re.finditer(r"\d+(?:\.\d+)*[a-z]?", nums_str):
                parts_x.append(nums_str[pos:nm.start()])
                key = _xref_key(kind, nm.group())
                tgt = ctx.xref_targets.get(key)
                if tgt:
                    file, anchor = tgt
                    href = f"#{anchor}" if file == ctx.current_file else f"{file}#{anchor}"
                    parts_x.append(f'<a href="{href}">{nm.group()}</a>')
                    any_linked = True
                else:
                    parts_x.append(nm.group())
                pos = nm.end()
            parts_x.append(nums_str[pos:])
            if any_linked:
                ws = m.group()[len(kind):len(m.group()) - len(nums_str)]
                repl = f"{kind}{ws}{''.join(parts_x)}"
                candidates.append((m.start(), m.end(), repl))
        for cite_re in (_CITE_RE, _CITE_ALPHA_RE):
            for m in cite_re.finditer(text):
                keys = re.split(r"[,;]\s*", m.group(1).strip())
                parts_c = []
                any_linked = False
                for key in keys:
                    key = key.strip()
                    tgt = ctx.xref_targets.get(f"cite {key}")
                    if tgt:
                        file, anchor = tgt
                        href = f"#{anchor}" if file == ctx.current_file else f"{file}#{anchor}"
                        parts_c.append(f'<a href="{href}">{key}</a>')
                        any_linked = True
                    else:
                        parts_c.append(key)
                if any_linked:
                    candidates.append((m.start(), m.end(), "[" + ", ".join(parts_c) + "]"))
        for m in _CITE_PAREN_RE.finditer(text):
            parts = [p.strip() for p in m.group(1).split(";")]
            parts_out = []
            any_linked = False
            for part in parts:
                sm = _CITE_SINGLE_RE.match(part)
                if not sm:
                    parts_out.append(part)
                    continue
                lastname = sm.group(1).lower()
                year = sm.group(2)
                tgt = ctx.xref_targets.get(f"cite {lastname}{year}")
                if tgt:
                    file, anchor = tgt
                    href = f"#{anchor}" if file == ctx.current_file else f"{file}#{anchor}"
                    parts_out.append(f'<a href="{href}">{part}</a>')
                    any_linked = True
                else:
                    parts_out.append(part)
            if any_linked:
                candidates.append((m.start(), m.end(), "(" + "; ".join(parts_out) + ")"))
        candidates.sort(key=lambda c: (c[0], -c[1]))
        result = []
        last_end = 0
        for start, end, repl in candidates:
            if start < last_end:
                continue
            result.append(text[last_end:start])
            result.append(repl)
            last_end = end
        if result:
            result.append(text[last_end:])
            out.append("".join(result))
        else:
            out.append(text)
    return "".join(out)


def inline_html(lines: list[Line], ctx: RenderCtx, *, strip_bold=False, footnote_sink: list | None = None) -> str:
    segs = line_segments(lines, ctx, strip_bold)
    # merge neighbours with identical style
    merged: list[list] = []
    for t, st in segs:
        if merged and merged[-1][1] == st:
            merged[-1][0] += t
        else:
            merged.append([t, st])
    # whitespace between styled runs belongs outside the tags
    out: list[str] = []
    for i, (t, st) in enumerate(merged):
        lead = t[: len(t) - len(t.lstrip())]
        trail = t[len(t.rstrip()):]
        core = t.strip()
        if not core:
            out.append(escape(t) if t else "")
            continue
        sup = st[3]
        # footnote reference?
        if sup and footnote_sink is not None and ctx.footnotes and not st[5]:
            pg = _page_of(lines, core)
            keys = [core] if (pg, core) in ctx.footnotes else (
                list(core) if all(ch in FN_SYMBOLS for ch in core) else [])
            parts, any_ref = [], False
            for k in keys:
                fn = ctx.footnotes.get((pg, k))
                if fn and not fn.used:
                    fn.used = True
                    ctx.fn_counter += 1
                    fn.id = f"fn{ctx.fn_counter}"
                    footnote_sink.append(fn)
                    any_ref = True
                    parts.append(f'<a epub:type="noteref" role="doc-noteref" id="r{fn.id}" '
                                 f'href="#{fn.id}"><sup>{escape(k)}</sup></a>')
                else:
                    parts.append(f"<sup>{escape(k)}</sup>")      # already-cited symbol: plain marker
            if any_ref:
                out.append(escape(lead) + "".join(parts) + escape(trail))
                continue
        html = _wrap(core, st)
        if not st[5] and not st[2]:
            # linkify only plain text (escape happened in _wrap; operate on the wrapped string
            # cautiously: only when no tags were added)
            if html == escape(core):
                html = _linkify(html)
        out.append(escape(lead) + html + escape(trail))
    s = "".join(out)
    s = s.replace("�", "")
    s = re.sub(r"[ \t]+", " ", s).strip()
    for tag in ("em", "strong", "code", "sub", "sup"):
        s = re.sub(rf"</{tag}>(\s*)<{tag}>", r"\1", s)
    if ctx.xref_targets:
        s = _insert_xrefs(s, ctx)
    return s


def _page_of(lines: list[Line], core: str) -> int:
    for l in lines:
        for s in l.spans:
            if s.sup and s.text.strip() == core:
                return l.page
    return lines[0].page


# --------------------------------------------------------------------- blocks
def _plain(lines: list[Line], ctx: RenderCtx) -> str:
    return "".join(t for t, _ in line_segments(lines, ctx)).strip()


def render_code(b: Block) -> str:
    rows: list[tuple[int, str]] = []        # (blank lines before, raw text)
    prev = None
    for ln in b.lines:
        blanks = 0
        if prev is not None:
            dy = base_y(ln) - base_y(prev)
            blanks = min(3, max(0, round(dy / (ln.size * 1.2)) - 1))
        raw = ln.text.expandtabs(4).rstrip()
        rows.append((blanks, raw))
        prev = ln
    common = min((len(r) - len(r.lstrip()) for _, r in rows if r.strip()), default=0)
    out: list[str] = []
    for blanks, raw in rows:
        out.extend([""] * blanks)
        out.append(raw[common:])
    return "<pre><code>" + _linkify(escape("\n".join(out))) + "</code></pre>"


def render_algorithm(b: Block) -> str:
    """Render ML-detected algorithm / pseudocode as a styled block.

    Uses plain text (no sub/sup processing) to avoid mathjunk from
    subscripted variable names like θ_t, β_1, g_t.
    """
    rows: list[tuple[int, str]] = []
    prev = None
    for ln in b.lines:
        blanks = 0
        if prev is not None:
            dy = base_y(ln) - base_y(prev)
            blanks = min(3, max(0, round(dy / (ln.size * 1.2)) - 1))
        raw = ln.text.expandtabs(4).rstrip()
        rows.append((blanks, raw))
        prev = ln
    common = min((len(r) - len(r.lstrip()) for _, r in rows if r.strip()), default=0)
    out: list[str] = []
    for blanks, raw in rows:
        out.extend([""] * blanks)
        out.append(raw[common:])
    return '<div class="algorithm"><pre>' + escape("\n".join(out)) + "</pre></div>"


def _fn_aside(fn: FootnoteDef, ctx: RenderCtx) -> str:
    lines = list(fn.lines)
    first = lines[0]
    # drop the marker from the footnote text
    spans, dropped = [], False
    for s in first.spans:
        if not dropped:
            t = s.text.lstrip()
            if t.startswith(fn.marker):
                t = t[len(fn.marker):].lstrip(" .)]")
                dropped = True
                if t:
                    spans.append(replace(s, text=t, sup=False))
                continue
        spans.append(s)
    lines[0] = Line(bbox=first.bbox, page=first.page, spans=spans)
    body = inline_html(lines, ctx, strip_bold=False)
    return (f'<aside epub:type="footnote" role="doc-footnote" id="{fn.id}" class="footnote">'
            f'<p><a href="#r{fn.id}" class="fn-back">{escape(fn.marker)}</a> {body}</p></aside>')


def render_blocks(blocks: list[Block], ctx: RenderCtx, *, heading_id,
                   page_map: list[tuple[int, str]] | None = None) -> list[str]:
    html: list[str] = []
    seen_pages = {pg for pg, _ in page_map} if page_map is not None else set()
    last_page = -1
    prev_eq = False
    i, n = 0, len(blocks)
    while i < n:
        b = blocks[i]
        if page_map is not None and b.page != last_page:
            pg = b.page + 1
            if pg not in seen_pages:
                pid = f"pg-{pg}"
                html.append(f'<span id="{pid}" epub:type="pagebreak" role="doc-pagebreak" '
                            f'aria-label="Page {pg}"/>')
                page_map.append((pg, f"{ctx.current_file}#{pid}"))
                seen_pages.add(pg)
            last_page = b.page
        k = b.kind
        if k == "heading":
            text = escape(b.synthetic_text) if b.synthetic_text is not None else \
                inline_html(b.lines, ctx, strip_bold=True)
            if not text:
                i += 1; continue
            hid = heading_id(b)
            html.append(f'<h{b.level} id="{hid}">{text}</h{b.level}>')
        elif k == "para":
            _collapsed_eq_frag = False
            sink: list[FootnoteDef] = []
            body = inline_html(b.lines, ctx, footnote_sink=sink).replace("�", "").strip()
            if body:
                plain = b.text.replace("�", "").strip() if b.text else ""
                if prev_eq and len(plain) < 20 and not re.match(r"[a-z]{3,}", plain.lower()):
                    _collapsed_eq_frag = True
                else:
                    cls = ""
                    if all(l.italic for l in b.lines) and len(b.lines) >= 2:
                        cls = "quote"
                    elif prev_eq and b.text and b.text[0].islower():
                        cls = "eq-cont"
                    attrs = f' class="{cls}"' if cls else ""
                    if b.anchor:
                        attrs += f' id="{b.anchor}"'
                    html.append(f"<p{attrs}>{body}</p>")
            for fn in sink:
                html.append(_fn_aside(fn, ctx))
        elif k == "dl":
            j = i
            while j < n and blocks[j].kind == "dl":
                j += 1
            html.append(_render_defn_list(blocks[i:j], ctx))
            i = j
            continue
        elif k == "li":
            j = i
            while j < n and blocks[j].kind == "li":
                j += 1
            html.append(_render_list(blocks[i:j], ctx))
            i = j
            continue
        elif k == "separator":
            html.append("<hr/>")
        elif k == "code":
            html.append(render_code(b))
        elif k == "algorithm":
            aid = f' id="{b.anchor}"' if b.anchor else ""
            html.append(render_algorithm(b).replace('<div class="algorithm">', f'<div class="algorithm"{aid}>', 1))
        elif k == "image":
            im = b.image
            href = ctx.image_href.get(im.key)
            aid = f' id="{b.anchor}"' if b.anchor else ""
            if href and im.kind == "equation":
                w = (im.width + 4) / max(ctx.stats.body_size, 6.0)
                mathml = to_mathml(im.latex, im.alt, altimg=href) if im.latex else None
                if mathml:
                    no = equation_number(im.alt)
                    tag = f'<span class="eqno">{escape(no)}</span>' if no else ""
                    html.append(f'<div class="equation mathml"{aid}>{mathml}{tag}</div>')
                else:
                    html.append(f'<div class="equation"{aid}><img class="equation" src="{href}" alt={quoteattr(im.alt[:300])} '
                                f'role="img" style="width:{w:.2f}em"/></div>')
            elif href:
                cap = inline_html(b.caption, ctx) if b.caption else ""
                alt = quoteattr(_plain(b.caption, ctx)[:200] if b.caption else (im.alt[:200] if im.alt else "Figure"))
                fig = f'<img src="{href}" alt={alt} role="img"/>'
                if cap:
                    fig += f"<figcaption>{cap}</figcaption>"
                html.append(f"<figure{aid}>{fig}</figure>")
        elif k == "table":
            tbl = _render_table(b, ctx)
            if b.anchor:
                tbl = tbl.replace("<table>", f'<table id="{b.anchor}">', 1)
            html.append(f'<div class="table-wrap">{tbl}</div>')
        elif k == "toc":
            html.append(_render_toc(b, ctx))
        elif k == "index":
            html.append(_render_index(b))
        if k == "image" and b.image is not None and b.image.kind == "equation":
            prev_eq = True
        elif k != "para" or not _collapsed_eq_frag:
            prev_eq = False
        i += 1
    return html


def _render_list(items: list[Block], ctx: RenderCtx) -> str:
    n = len(items)

    def build(i: int, level: int) -> tuple[str, int]:
        first = items[i]
        tag = "ol" if first.ordered else "ul"
        attr = f' start="{first.number}"' if tag == "ol" and first.number and first.number != 1 else ""
        if first.anchor and first.anchor.startswith("bib"):
            attr += ' role="doc-bibliography"'
        out = [f"<{tag}{attr}>"]
        while i < n and items[i].level == level and items[i].ordered == first.ordered:
            sink: list[FootnoteDef] = []
            aid = f' id="{items[i].anchor}"' if items[i].anchor else ""
            out.append(f"<li{aid}>{inline_html(items[i].lines, ctx, footnote_sink=sink)}")
            out.extend(_fn_aside(fn, ctx) for fn in sink)
            i += 1
            while i < n and items[i].level > level:
                sub, i = build(i, items[i].level)
                out.append(sub)
            out.append("</li>")
        out.append(f"</{tag}>")
        return "\n".join(out), i

    parts, i = [], 0
    while i < n:
        html, i = build(i, items[i].level)
        parts.append(html)
    return "".join(parts)


def _render_defn_list(items: list[Block], ctx: RenderCtx) -> str:
    out = ["<dl>"]
    for b in items:
        term_spans: list[Span] = []
        desc_lines: list[Line] = []
        if b.lines and b.lines[0].spans:
            first_line = b.lines[0]
            split_idx = 0
            for si, s in enumerate(first_line.spans):
                if s.bold and s.text.rstrip().endswith(":"):
                    split_idx = si + 1
                    break
                elif s.bold:
                    split_idx = si + 1
                else:
                    break
            term_spans = first_line.spans[:split_idx]
            rest_spans = first_line.spans[split_idx:]
            if rest_spans:
                rest_line = Line(bbox=first_line.bbox, page=first_line.page, spans=rest_spans)
                desc_lines = [rest_line] + b.lines[1:]
            else:
                desc_lines = b.lines[1:]
        term_text = "".join(s.text for s in term_spans).strip().rstrip(":")
        out.append(f"<dt>{escape(term_text)}</dt>")
        sink: list[FootnoteDef] = []
        dd_html = inline_html(desc_lines, ctx, footnote_sink=sink) if desc_lines else ""
        out.append(f"<dd>{dd_html}")
        out.extend(_fn_aside(fn, ctx) for fn in sink)
        out.append("</dd>")
    out.append("</dl>")
    return "".join(out)


def _render_structured_table(b: Block, ctx: RenderCtx) -> str:
    t = b.table
    by_row: dict[int, list[dict]] = {}
    for c in sorted(t.cells, key=lambda c: (c["r0"], c["c0"])):
        by_row.setdefault(c["r0"], []).append(c)
    head_rows = 0
    while head_rows < t.nrows and by_row.get(head_rows) and all(c.get("head") for c in by_row[head_rows]):
        head_rows += 1
    numeric = lambda x: bool(re.fullmatch(r"[\d.,%$€£()\-+\s]+", x or "")) and any(ch.isdigit() for ch in x)
    occupied = {(r, c) for cell in t.cells for r in range(cell["r0"], cell["r1"]) for c in range(cell["c0"], cell["c1"])
                if (r, c) != (cell["r0"], cell["c0"])}
    out = ["<table>"]
    if b.caption:
        out.append(f"<caption>{inline_html(b.caption, ctx)}</caption>")
    for r in range(t.nrows):
        if r == 0 and head_rows:
            out.append("<thead>")
        if r == head_rows:
            out.append("<tbody>")
        cells = []
        starting = {c["c0"]: c for c in by_row.get(r, [])}
        col = 0
        while col < t.ncols:
            if (r, col) in occupied:               # covered by a row/col span from a cell above or to the left
                col += 1
                continue
            c = starting.get(col)
            if c is None:                           # the model lists only non-empty cells: keep the grid aligned
                cells.append("<td></td>")
                col += 1
                continue
            head = r < head_rows or c.get("head") or c.get("rhead")
            tag = "th" if head else "td"
            attrs = ""
            if c["r1"] - c["r0"] > 1: attrs += f' rowspan="{c["r1"] - c["r0"]}"'
            if c["c1"] - c["c0"] > 1: attrs += f' colspan="{c["c1"] - c["c0"]}"'
            if head: attrs += ' scope="col"' if (r < head_rows or c.get("head")) else ' scope="row"'
            elif numeric(c["text"]): attrs += ' class="num"'
            cells.append(f"<{tag}{attrs}>{escape(c['text'])}</{tag}>")
            col += max(c["c1"] - c["c0"], 1)
        out.append("<tr>" + "".join(cells) + "</tr>")
        if r == head_rows - 1 and head_rows:
            out.append("</thead>")
    out.append("</tbody></table>" if t.nrows > head_rows else "</table>")
    return "\n".join(out)


def _render_table(b: Block, ctx: RenderCtx) -> str:
    if b.table.cells:
        return _render_structured_table(b, ctx)
    rows = b.table.rows
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    first = rows[0]
    numeric = lambda s: bool(re.fullmatch(r"[\d.,%$€£()\-+\s]+", s or "")) and any(ch.isdigit() for ch in s)
    header = all(c and not numeric(c) for c in first) or (not any(numeric(c) for c in first)
                                                          and any(numeric(c) for r in rows[1:] for c in r))
    out = ["<table>"]
    if b.caption:
        out.append(f"<caption>{inline_html(b.caption, ctx)}</caption>")
    for ri, r in enumerate(rows):
        tag = "th" if (header and ri == 0) else "td"
        cells = []
        for c in r:
            attrs = ""
            if tag == "th":
                attrs = ' scope="col"'
            elif numeric(c):
                attrs = ' class="num"'
            cell_html = escape(c)
            if ctx.xref_targets and not numeric(c):
                cell_html = _insert_xrefs(cell_html, ctx)
            cells.append(f"<{tag}{attrs}>{cell_html}</{tag}>")
        row = "<tr>" + "".join(cells) + "</tr>"
        if header and ri == 0:
            row = "<thead>" + row + "</thead><tbody>"
        out.append(row)
    out.append("</tbody></table>" if header else "</table>")
    s = "\n".join(out)
    if not header:
        s = s.replace("<table>", "<table><tbody>", 1).replace("</table>", "</tbody></table>")
    return s


def _render_toc(b: Block, ctx: RenderCtx) -> str:
    items = []
    for title, label in b.entries:
        tgt = ctx.heading_targets.get(norm_title(title))
        if not tgt:
            sn = strip_numbering(title)
            if sn != norm_title(title):
                tgt = ctx.heading_targets.get(sn)
        label_html = escape(title)
        if tgt:
            f, hid = tgt
            href = (hid and f"{f}#{hid}") if f != ctx.current_file else f"#{hid}"
            label_html = f'<a href="{href}">{label_html}</a>'
        elif label:
            label_html = f"{label_html} {escape(label)}"
        items.append(f"<li>{label_html}</li>")
    return '<ul class="toc">' + "".join(items) + "</ul>"


def _render_index(b: Block) -> str:
    items = []
    for term, pages in b.entries:
        t = escape(term.rstrip(" ,;"))
        if pages:
            p = re.sub(r"\s+", " ", pages.strip())
            items.append(f"<li>{t}, {escape(p)}</li>")
        else:
            items.append(f"<li>{t}</li>")
    return '<ul class="index">' + "".join(items) + "</ul>"
