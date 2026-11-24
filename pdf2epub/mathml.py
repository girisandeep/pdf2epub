"""LaTeX -> MathML with a guard against model hallucination."""
from __future__ import annotations

import re
from xml.sax.saxutils import quoteattr

_WRAP = [(re.compile(r"^\s*\$\$?|\$\$?\s*$"), ""), (re.compile(r"^\s*\\\[|\\\]\s*$"), ""),
         (re.compile(r"\\begin\{(equation|align|gather|displaymath)\*?\}|\\end\{(equation|align|gather|displaymath)\*?\}"), ""),
         (re.compile(r"\\tag\*?\{[^}]*\}"), ""), (re.compile(r"\\label\{[^}]*\}"), ""),
         (re.compile(r"\\(?:displaystyle|textstyle|limits|nonumber|notag)\b"), "")]
_EQNO = re.compile(r"\(\s*(\d+(?:\.\d+)?[a-z]?)\s*\)\s*$")


def clean_latex(latex: str) -> str:
    for rx, rep in _WRAP:
        latex = rx.sub(rep, latex)
    return latex.strip()


def equation_number(alt: str) -> str | None:
    m = _EQNO.search(alt.strip())
    return f"({m.group(1)})" if m else None


def _symbols(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]", s))


def consistent(latex: str, alt: str, threshold: float = 0.6) -> bool:
    """The PDF's own text for the region must be (mostly) accounted for in the recognised LaTeX."""
    want = _symbols(re.sub(r"\(\s*\d+(?:\.\d+)?[a-z]?\s*\)\s*$", "", alt))
    if len(want) < 2:
        return False
    have = _symbols(re.sub(r"\\[A-Za-z]+", "", latex))
    return len(want & have) / len(want) >= threshold


_TEXT = re.compile(r"\\(?:text|mathrm|textrm|mbox)\s*\{([^{}]*)\}")


def _trustworthy(tex: str) -> bool:
    """Prose inside the formula (words in \\text{}) means the model clipped surrounding text: don't trust it."""
    return sum(len(t.split()) for t in _TEXT.findall(tex)) < 3


def _fully_converted(mathml_root) -> bool:
    """The converter leaves unknown commands as literal text containing a backslash."""
    return not any("\\" in (t or "") for t in mathml_root.itertext())


_NS = "http://www.w3.org/1998/Math/MathML"


def _convert(tex: str, conv, etree):
    """Parsed <math> element for `tex`.  Multi-line/aligned input becomes an <mtable> built here, because
    latex2mathml emits an unescaped '&' for alignment points."""
    rows = [r.strip() for r in re.split(r"\\\\", tex) if r.strip()]
    if len(rows) <= 1 and "&" not in tex:
        return etree.fromstring(conv.convert(tex, display="block").encode())
    math = etree.Element(f"{{{_NS}}}math", nsmap={None: _NS}, display="block")
    table = etree.SubElement(math, f"{{{_NS}}}mtable", columnalign="right left", columnspacing="0")
    for row in rows or [tex]:
        tr = etree.SubElement(table, f"{{{_NS}}}mtr")
        for cell in row.split("&"):
            td = etree.SubElement(tr, f"{{{_NS}}}mtd")
            cell = cell.strip()
            if cell:
                for child in etree.fromstring(conv.convert(cell, display="block").encode()):
                    td.append(child)
    return math


def to_mathml(latex: str, alt: str, altimg: str | None = None) -> str | None:
    """MathML for a display equation, or None when it cannot be trusted (caller falls back to an image)."""
    try:
        import latex2mathml.converter as conv
        from lxml import etree
    except ImportError:
        return None
    tex = clean_latex(latex)
    if not tex or len(tex) > 2000 or not consistent(tex, alt):
        return None
    try:
        if not _trustworthy(tex):
            return None
        root = _convert(tex, conv, etree)
        if not _fully_converted(root):                          # unknown commands are left as literal text
            return None
        ml = etree.tostring(root, encoding="unicode")
    except Exception:
        return None
    attrs = f" alttext={quoteattr(re.sub(r'\\s+', ' ', alt)[:300])}"
    if altimg:
        attrs += f" altimg={quoteattr(altimg)}"
    return ml.replace("<math ", f"<math{attrs} ", 1)
