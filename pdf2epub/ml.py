"""Optional deep-learning layout analysis (Docling's layout model, RT-DETR based, MIT licensed).

The model only supplies *regions and labels* (title, section_header, text, list_item, caption,
footnote, formula, picture, table, page_header, page_footer, code ...).  All text, fonts and
geometry still come from the PDF itself, so nothing is ever mis-read by the model.

Install:  pip install 'docling'   (pulls torch).  Weights are fetched from Hugging Face on first use.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass

log = logging.getLogger("pdf2epub")

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pdf2epub", "layout")
MODEL_TAG = "docling-layout-heron/v1"


@dataclass
class Region:
    label: str
    bbox: tuple[float, float, float, float]      # PDF points, top-left origin (same as PyMuPDF)
    score: float
    id: int = 0
    latex: str = ""                             # recognised LaTeX for formula regions (formulas=True)


class LayoutDetector:
    def __init__(self, min_score: float = 0.35, cache: bool = True, formulas: bool = False):
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as e:
            raise ImportError("ML layout analysis needs Docling: pip install docling") from e
        self.min_score, self.cache, self.formulas = min_score, cache, formulas
        opts = PdfPipelineOptions(do_ocr=False, do_table_structure=False, do_formula_enrichment=formulas)
        dev = os.environ.get("PDF2EPUB_DEVICE", "auto").lower()          # auto | cpu | mps | cuda
        if dev != "auto":
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice(dev))
        self._conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    def _key(self, path: str, first: int, last: int) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(f"{MODEL_TAG}:{first}:{last}:{self.min_score}:f{int(self.formulas)}".encode())
        return os.path.join(CACHE_DIR, h.hexdigest() + ".json")

    @staticmethod
    def _attach_latex(res, out: dict[int, list[Region]]) -> None:
        """Give each formula region the LaTeX of the document item that overlaps it most."""
        for item, _ in res.document.iterate_items():
            if str(item.label.value) != "formula" or not getattr(item, "text", ""):
                continue
            for prov in item.prov:
                h = res.document.pages[prov.page_no].size.height
                b = prov.bbox.to_top_left_origin(h)
                best, best_ov = None, 0.0
                for r in out.get(prov.page_no - 1, []):
                    if r.label != "formula":
                        continue
                    w = min(b.r, r.bbox[2]) - max(b.l, r.bbox[0])
                    hh = min(b.b, r.bbox[3]) - max(b.t, r.bbox[1])
                    ov = max(w, 0) * max(hh, 0)
                    if ov > best_ov:
                        best, best_ov = r, ov
                if best is not None and not best.latex:
                    best.latex = item.text

    def detect(self, path: str, first: int, last: int) -> dict[int, list[Region]]:
        """Regions per 0-based page index for pages first..last (inclusive)."""
        cp = self._key(path, first, last)
        if self.cache and os.path.exists(cp):
            raw = json.load(open(cp))
            return {int(p): [Region(**r) for r in rs] for p, rs in raw.items()}
        res = self._conv.convert(path, page_range=(first + 1, last + 1))
        out: dict[int, list[Region]] = {}
        for pg in res.pages:
            if pg.predictions is None or pg.predictions.layout is None:
                continue
            regs = []
            for c in pg.predictions.layout.clusters:
                if c.confidence < self.min_score:
                    continue
                b = c.bbox.to_top_left_origin(pg.size.height)
                regs.append(Region(c.label.value, (b.l, b.t, b.r, b.b), float(c.confidence)))
            regs.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
            for i, r in enumerate(regs):
                r.id = (pg.page_no - 1) * 1000 + i
            out[pg.page_no - 1] = regs
        if self.formulas:
            self._attach_latex(res, out)
        if self.cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            json.dump({p: [r.__dict__ for r in rs] for p, rs in out.items()}, open(cp, "w"))
        return out
