"""Core data model shared by the pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

BBox = tuple[float, float, float, float]


@dataclass
class Span:
    text: str
    size: float
    bbox: BBox
    origin_y: float = 0.0
    bold: bool = False
    italic: bool = False
    mono: bool = False
    sup: bool = False
    sub: bool = False
    href: Optional[str] = None
    font: str = ""


@dataclass
class Item:
    """Anything that occupies a rectangle on a page and takes part in reading order."""
    bbox: BBox
    page: int
    region: int = 0

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]
    @property
    def width(self): return self.bbox[2] - self.bbox[0]
    @property
    def height(self): return self.bbox[3] - self.bbox[1]


@dataclass
class Line(Item):
    spans: list[Span] = field(default_factory=list)
    ml_label: str = ""          # layout-model label of the region containing this line
    ml_id: int = -1             # id of that region (-1: none / ML off)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def size(self) -> float:
        w: dict[float, int] = {}
        for s in self.spans:
            k = round(s.size * 2) / 2
            w[k] = w.get(k, 0) + len(s.text.strip())
        return max(w, key=w.get) if w else 10.0

    def _frac(self, attr: str) -> float:
        tot = sum(len(s.text.strip()) for s in self.spans)
        if not tot:
            return 0.0
        return sum(len(s.text.strip()) for s in self.spans if getattr(s, attr)) / tot

    @property
    def bold(self) -> bool: return self._frac("bold") > 0.85
    @property
    def italic(self) -> bool: return self._frac("italic") > 0.85
    @property
    def mono(self) -> bool: return self._frac("mono") > 0.85


@dataclass
class ImageItem(Item):
    data: bytes = b""
    ext: str = "png"
    px_w: int = 0
    px_h: int = 0
    key: str = ""
    kind: str = "image"          # image | figure (rasterised vector art) | page (scan) | equation
    alt: str = ""
    latex: str = ""              # recognised LaTeX (equations, when the formula model ran)


@dataclass
class TableItem(Item):
    rows: list[list[str]] = field(default_factory=list)
    header: bool = True


@dataclass
class PageData:
    number: int                  # 0-based
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)
    images: list[ImageItem] = field(default_factory=list)
    tables: list[TableItem] = field(default_factory=list)
    ocr: bool = False


@dataclass
class FootnoteDef:
    page: int
    marker: str
    lines: list[Line]
    used: bool = False
    id: str = ""


@dataclass
class Block:
    """A semantic unit after segmentation."""
    kind: str                    # para | heading | li | code | image | table | toc | caption
    page: int
    lines: list[Line] = field(default_factory=list)
    region: int = 0
    level: int = 0               # heading level, or list depth
    ordered: bool = False
    number: Optional[int] = None
    image: Optional[ImageItem] = None
    table: Optional[TableItem] = None
    caption: Optional[list[Line]] = None
    marker: Optional[str] = None
    indent: float = 0.0
    full_last: bool = False      # last line reaches the right edge of its column
    indented_first: bool = False
    y0: float = 0.0
    anchor: str = ""
    entries: list = field(default_factory=list)   # toc entries
    synthetic_text: Optional[str] = None

    @property
    def ml_label(self) -> str:
        labels = [l.ml_label for l in self.lines if l.ml_label]
        return max(set(labels), key=labels.count) if labels else ""

    @property
    def text(self) -> str:
        if self.synthetic_text is not None:
            return self.synthetic_text
        return " ".join(l.text.strip() for l in self.lines)
