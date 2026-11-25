"""Minimal, strictly-structured EPUB 3 writer (with EPUB 2 NCX for older readers)."""
from __future__ import annotations

import datetime as dt
import uuid
import zipfile
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

CSS = """\
html { font-size: 100%; }
body { font-family: Georgia, "Times New Roman", serif; line-height: 1.45; margin: 5%; text-align: left; hyphens: auto; -webkit-hyphens: auto; overflow-wrap: break-word; }
h1, h2, h3, h4, h5, h6 { font-family: Helvetica, Arial, sans-serif; line-height: 1.2; margin: 1.6em 0 .6em; page-break-after: avoid; break-after: avoid; }
h1 { font-size: 1.8em; margin-top: 2em; } h2 { font-size: 1.45em; } h3 { font-size: 1.2em; } h4, h5, h6 { font-size: 1.05em; }
p { margin: 0 0 .85em; orphans: 2; widows: 2; }
p.quote { margin-left: 1.5em; margin-right: 1.5em; }
ul, ol { margin: 0 0 .9em; padding-left: 1.6em; } li { margin: 0 0 .3em; }
dl { margin: 0 0 .9em; } dt { font-weight: bold; margin: .5em 0 .15em; } dd { margin: 0 0 .3em 1.2em; }
pre { font-family: Menlo, Consolas, "Courier New", monospace; font-size: .85em; line-height: 1.3; background: #f5f5f5; padding: .6em .8em; margin: 0 0 1em; white-space: pre-wrap; overflow-wrap: anywhere; border-radius: 3px; }
code { font-family: Menlo, Consolas, "Courier New", monospace; font-size: .9em; }
pre code { font-size: 1em; }
figure { margin: 1.2em 0; text-align: center; page-break-inside: avoid; break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
div.equation.mathml { position: relative; } span.eqno { position: absolute; right: 0; top: 50%; transform: translateY(-50%); }
div.equation { text-align: center; margin: .9em 0; } img.equation { max-width: 100%; height: auto; vertical-align: middle; }
figcaption { font-size: .88em; font-style: italic; margin-top: .4em; text-align: center; }
table { border-collapse: collapse; margin: 1em auto; max-width: 100%; font-size: .9em; word-break: break-word; }
caption { font-weight: bold; margin-bottom: .4em; text-align: left; }
th, td { border: 1px solid #999; padding: .25em .5em; vertical-align: top; }
th { background: #eee; } td.num { text-align: right; }
aside.footnote { font-size: .85em; margin: 0 0 .8em 1.2em; padding-left: .6em; border-left: 2px solid #ccc; }
aside.footnote p { margin: 0; } a.fn-back { text-decoration: none; font-weight: bold; }
ul.toc { list-style: none; padding-left: 0; } ul.toc li { margin-bottom: .4em; }
ul.index { list-style: none; padding-left: 0; font-size: .85em; column-count: 2; column-gap: 1.5em; } ul.index li { margin-bottom: .15em; break-inside: avoid; }
.algorithm { background: #f8f8f0; border: 1px solid #ddd; padding: .6em .8em; margin: 0 0 1em; border-radius: 3px; }
.algorithm pre { font-family: Georgia, "Times New Roman", serif; font-size: .95em; line-height: 1.5; background: none; padding: 0; margin: 0; border-radius: 0; }
hr { border: none; border-top: 1px solid #999; margin: 1.5em 15%; }
sup, sub { line-height: 0; }
.cover { text-align: center; margin: 0; padding: 0; } .cover img { max-width: 100%; max-height: 100%; }
@media (max-width: 600px) {
  body { margin: 3%; }
  table { font-size: .8em; }
  pre { font-size: .8em; padding: .4em .5em; }
}
@media (prefers-color-scheme: dark) {
  body { color: #ddd; background: #1a1a1a; }
  pre { background: #2a2a2a; }
  th { background: #333; }
  th, td { border-color: #555; }
  .algorithm { background: #252520; border-color: #444; }
  aside.footnote { border-left-color: #555; }
  hr { border-top-color: #555; }
  a { color: #7cafc2; }
}
"""


@dataclass
class Chapter:
    filename: str
    title: str
    body: str


@dataclass
class NavEntry:
    level: int
    title: str
    href: str


@dataclass
class EpubBook:
    title: str
    language: str = "en"
    author: str | None = None
    identifier: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    chapters: list[Chapter] = field(default_factory=list)
    nav: list[NavEntry] = field(default_factory=list)
    images: dict[str, tuple[bytes, str]] = field(default_factory=dict)   # href -> (data, mime)
    cover: bytes | None = None


def xhtml(title: str, body: str, lang: str, css: str = "style.css", body_class: str = "") -> str:
    cls = f' class="{body_class}"' if body_class else ""
    return ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
            f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" '
            f'lang="{lang}" xml:lang="{lang}">\n<head><meta charset="utf-8"/><title>{escape(title)}</title>'
            f'<link rel="stylesheet" type="text/css" href="{css}"/></head>\n<body{cls}>\n{body}\n</body>\n</html>\n')


def _tree(entries: list[NavEntry]):
    root: list[dict] = []
    stack: list[tuple[int, list]] = [(0, root)]
    for e in entries:
        node = {"e": e, "kids": []}
        while len(stack) > 1 and stack[-1][0] >= e.level:
            stack.pop()
        stack[-1][1].append(node)
        stack.append((e.level, node["kids"]))
    return root


def _nav_ol(nodes) -> str:
    if not nodes:
        return ""
    items = "".join(f'<li><a href="{escape(n["e"].href)}">{escape(n["e"].title)}</a>{_nav_ol(n["kids"])}</li>'
                    for n in nodes)
    return f"<ol>{items}</ol>"


def _ncx_points(nodes, counter: list[int]) -> str:
    out = []
    for n in nodes:
        counter[0] += 1
        pid = counter[0]
        out.append(f'<navPoint id="np{pid}" playOrder="{pid}"><navLabel><text>{escape(n["e"].title)}</text></navLabel>'
                   f'<content src="{escape(n["e"].href)}"/>{_ncx_points(n["kids"], counter)}</navPoint>')
    return "".join(out)


def _depth(nodes) -> int:
    return 0 if not nodes else 1 + max(_depth(n["kids"]) for n in nodes)


def write_epub(book: EpubBook, path: str) -> None:
    nav_entries = book.nav or [NavEntry(1, c.title, c.filename) for c in book.chapters]
    tree = _tree(nav_entries)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lang = book.language

    manifest = ['<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
                '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
                '<item id="css" href="style.css" media-type="text/css"/>']
    spine = []
    if book.cover:
        manifest.append('<item id="cover-image" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>')
        manifest.append('<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>')
        spine.append('<itemref idref="cover" linear="yes"/>')
    for i, c in enumerate(book.chapters):
        props = ' properties="mathml"' if "<math" in c.body else ""
        manifest.append(f'<item id="c{i}" href="{c.filename}" media-type="application/xhtml+xml"{props}/>')
        spine.append(f'<itemref idref="c{i}"/>')
    for j, (href, (_, mime)) in enumerate(book.images.items()):
        manifest.append(f'<item id="i{j}" href="{href}" media-type="{mime}"/>')

    creator = f"<dc:creator>{escape(book.author)}</dc:creator>" if book.author else ""
    opf = ('<?xml version="1.0" encoding="utf-8"?>\n'
           '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" '
           f'xml:lang="{lang}">\n<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
           f'<dc:identifier id="bookid">{escape(book.identifier)}</dc:identifier>'
           f'<dc:title>{escape(book.title)}</dc:title>{creator}<dc:language>{lang}</dc:language>'
           f'<meta property="dcterms:modified">{now}</meta>'
           '<meta property="schema:accessMode">textual</meta>'
           '<meta property="schema:accessMode">visual</meta>'
           '<meta property="schema:accessibilityFeature">structuralNavigation</meta>'
           '<meta property="schema:accessibilityFeature">tableOfContents</meta>'
           '<meta property="schema:accessibilityFeature">alternativeText</meta>'
           + ('<meta name="cover" content="cover-image"/>' if book.cover else "") +
           f'</metadata>\n<manifest>\n{chr(10).join(manifest)}\n</manifest>\n'
           f'<spine toc="ncx">\n{chr(10).join(spine)}\n</spine>\n</package>\n')

    landmarks = ['<nav epub:type="landmarks" hidden="hidden"><h2>Guide</h2><ol>']
    if book.cover:
        landmarks.append('<li><a epub:type="cover" href="cover.xhtml">Cover</a></li>')
    landmarks.append('<li><a epub:type="toc" href="nav.xhtml">Contents</a></li>')
    if book.chapters:
        landmarks.append(f'<li><a epub:type="bodymatter" href="{book.chapters[0].filename}">Start</a></li>')
    landmarks.append('</ol></nav>')
    nav_body = ('<nav epub:type="toc" id="toc"><h1>Contents</h1>' + _nav_ol(tree) + "</nav>\n"
                + "".join(landmarks))
    nav = xhtml("Contents", nav_body, lang)
    ncx = ('<?xml version="1.0" encoding="utf-8"?>\n<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
           f'<head><meta name="dtb:uid" content="{escape(book.identifier)}"/><meta name="dtb:depth" content="{max(_depth(tree), 1)}"/>'
           '<meta name="dtb:totalPageCount" content="0"/><meta name="dtb:maxPageNumber" content="0"/></head>'
           f'<docTitle><text>{escape(book.title)}</text></docTitle><navMap>{_ncx_points(tree, [0])}</navMap></ncx>')
    container = ('<?xml version="1.0"?>\n<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                 '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                 '</rootfiles></container>')

    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", container, zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/content.opf", opf, zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/nav.xhtml", nav, zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/toc.ncx", ncx, zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/style.css", CSS, zipfile.ZIP_DEFLATED)
        if book.cover:
            z.writestr("OEBPS/images/cover.jpg", book.cover, zipfile.ZIP_STORED)
            z.writestr("OEBPS/cover.xhtml", xhtml(book.title, '<div class="cover"><img src="images/cover.jpg" alt="Cover"/></div>',
                                                  lang, body_class="cover"), zipfile.ZIP_DEFLATED)
        for c in book.chapters:
            z.writestr(f"OEBPS/{c.filename}", xhtml(c.title, c.body, lang), zipfile.ZIP_DEFLATED)
        for href, (data, _) in book.images.items():
            z.writestr(f"OEBPS/{href}", data, zipfile.ZIP_STORED)
