"""Single-file HTML writer — reuses the same EpubBook structure as the EPUB path."""
from __future__ import annotations

import os
import re
from xml.sax.saxutils import escape

from .epub import CSS, EpubBook, NavEntry


_HTML_CSS = re.sub(r"\[epub\\\\:type~=\"pagebreak\"\]", "[data-pagebreak]", CSS)


def _toc_html(nav: list[NavEntry]) -> str:
    if not nav:
        return ""
    items = "".join(f'<li><a href="#{_anchor(e.href)}">{escape(e.title)}</a></li>' for e in nav)
    return f'<nav id="toc"><h2>Contents</h2><ol>{items}</ol></nav>\n<hr/>\n'


def _anchor(href: str) -> str:
    if "#" in href:
        return href.split("#", 1)[1]
    return href.replace(".xhtml", "").replace("/", "-")


def _rewrite_body(body: str, all_files: set[str]) -> str:
    def _fix(m):
        href = m.group(1)
        if href.startswith("#") or href.startswith("http://") or href.startswith("https://"):
            return m.group(0)
        if "#" in href:
            _, frag = href.split("#", 1)
            return f'href="#{frag}"'
        return m.group(0)
    body = re.sub(r'href="([^"]*)"', _fix, body)
    body = re.sub(r'epub:type="[^"]*"', lambda m: 'data-pagebreak="true"'
                  if "pagebreak" in m.group(0) else "", body)
    return body


def write_html(book: EpubBook, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    img_dir = os.path.join(out_dir, "images")

    if book.images:
        os.makedirs(img_dir, exist_ok=True)
    for href, (data, _) in book.images.items():
        img_path = os.path.join(out_dir, href)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(data)

    if book.cover:
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(img_dir, "cover.jpg"), "wb") as f:
            f.write(book.cover)

    all_files = {c.filename for c in book.chapters}

    parts: list[str] = []
    if book.cover:
        parts.append('<div class="cover"><img src="images/cover.jpg" alt="Cover"/></div>\n')
    parts.append(_toc_html(book.nav))
    for ch in book.chapters:
        body = _rewrite_body(ch.body, all_files)
        parts.append(f'<section id="{ch.filename.replace(".xhtml", "")}">\n{body}\n</section>\n')

    author_meta = f'<meta name="author" content="{escape(book.author)}"/>' if book.author else ""
    desc_meta = f'<meta name="description" content="{escape(book.description)}"/>' if book.description else ""

    html = (f'<!DOCTYPE html>\n<html lang="{book.language}">\n<head>\n'
            f'<meta charset="utf-8"/>\n<meta name="viewport" content="width=device-width, initial-scale=1"/>\n'
            f'<title>{escape(book.title)}</title>\n{author_meta}{desc_meta}'
            f'<style>\n{_HTML_CSS}\n</style>\n</head>\n<body>\n'
            + "\n".join(parts)
            + '\n</body>\n</html>\n')

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
