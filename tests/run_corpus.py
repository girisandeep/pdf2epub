"""Run the converter over tests/corpus/*/*.pdf and report quality metrics.

    python tests/run_corpus.py                  # run + compare against tests/corpus_baseline.json
    python tests/run_corpus.py --update-baseline
    python tests/run_corpus.py thinkpython bert # subset

Metrics (all computed from the produced EPUB, no ground truth needed):
  recall / precision  word overlap with the PDF text layer (order-insensitive)
  valid               structural EPUB check passes
  frag%               paragraphs shorter than 40 chars that don't end a sentence (stray fragments)
  mathjunk            short paragraphs (<120 chars) with dense sub/sup tags (<15 chars/tag; equations left as text soup)
  dup                 identical paragraphs (>=60 chars) repeated back-to-back (duplication bugs)
  eq                  equations rendered as images
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
CORPUS = os.path.join(HERE, "corpus")
BASELINE = os.path.join(HERE, "corpus_baseline.json")
BASELINE_ML = os.path.join(HERE, "corpus_baseline_ml.json")
NS = {"x": "http://www.w3.org/1999/xhtml"}


_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
_HYPHEN_EOL = re.compile(r"(\w)-\s*\n\s*(\w)")


def toks(s: str) -> Counter:
    s = _HYPHEN_EOL.sub(r"\1\2", s)
    return Counter(re.findall(r"[^\W_]+", s.lower().translate(_LIGATURES).replace("-", "")))


def evaluate(args: tuple) -> dict:
    pdf, equations, ml, mathml = args
    import pymupdf
    from lxml import etree
    from epub_check import check
    from pdf2epub import ConversionOptions, convert

    out = os.path.join(tempfile.mkdtemp(), "o.epub")
    t0 = time.time()
    try:
        rep = convert(pdf, out, ConversionOptions(equations=equations, ml="on" if ml else "off", mathml=mathml))
    except Exception as e:                      # a crash is the worst regression
        return {"error": f"{type(e).__name__}: {e}"}
    secs = time.time() - t0
    z = zipfile.ZipFile(out)
    paras, body = [], []
    stats = Counter()
    for n in sorted(z.namelist()):
        if not n.startswith("OEBPS/chap"):
            continue
        doc = etree.fromstring(z.read(n))
        body.append(" ".join(doc.xpath("//x:body//text()", namespaces=NS)))
        body.append(" ".join(doc.xpath("//x:img[@class='equation']/@alt", namespaces=NS)))   # equation text kept as alt
        for p in doc.xpath("//x:p|//x:li|//x:dd|//x:dt", namespaces=NS):
            par = p.getparent()
            if par is not None and par.get("class") == "toc":
                continue
            tag = p.tag.rsplit("}", 1)[-1]
            is_li = tag in ("li", "dt", "dd")
            paras.append(("".join(p.itertext()).strip(), len(p.xpath(".//x:sub|.//x:sup", namespaces=NS)), is_li))
        stats["mathml"] += len(doc.xpath("//*[local-name()='math']"))
        stats["eq"] += len(doc.xpath("//x:img[@class='equation']", namespaces=NS))
    ref = toks(" ".join(p.get_text() for p in pymupdf.open(pdf)))
    got = toks(" ".join(body))
    common = sum((ref & got).values())
    texts = [t for t, _, _ in paras if t]
    frag = sum(1 for t, _, li in paras if t and not li and len(t) < 40
               and not t.endswith((".", "?", "!", ":", '"', "”", "]", ")")))
    dup = sum(1 for a, b in zip(texts, texts[1:]) if a == b and len(a) >= 60)
    return {
        "pages": rep.pages, "secs": round(secs, 1), "chapters": rep.chapters, "headings": rep.headings,
        "tables": rep.tables, "images": rep.images, "lists": rep.lists, "footnotes": rep.footnotes,
        "recall": round(common / max(sum(ref.values()), 1), 4),
        "precision": round(common / max(sum(got.values()), 1), 4),
        "valid": check(out) == [],
        "frag_pct": round(100 * frag / max(len(texts), 1), 1),
        "mathjunk": sum(1 for t, s, _ in paras if s >= 3 and len(t) < 120 and len(t) / s < 15),
        "dup": dup, "eq": stats["eq"], "mathml": stats["mathml"], "paras": len(texts),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: 4, or 1 with --ml)")
    ap.add_argument("--ml", action="store_true", help="use the deep-learning layout model")
    ap.add_argument("--no-equations", action="store_true", help="measure without equation images (for before/after)")
    ap.add_argument("--mathml", action="store_true", help="also convert formulas to MathML (needs --ml; slow; never touches baselines)")
    a = ap.parse_args()
    files = {}
    for d in sorted(os.listdir(CORPUS)):
        p = os.path.join(CORPUS, d, f"{d}.pdf")
        if os.path.isfile(p) and (not a.names or d in a.names):
            files[d] = p
    with ProcessPoolExecutor(max_workers=a.workers or (1 if a.ml else min(4, os.cpu_count() or 2))) as ex:
        res = dict(zip(files, ex.map(evaluate, [(f, not a.no_equations, a.ml, a.mathml) for f in files.values()])))
    bpath = BASELINE_ML if a.ml else BASELINE
    base = json.load(open(bpath)) if os.path.exists(bpath) else {}
    cols = ["pages", "secs", "recall", "precision", "valid", "headings", "tables", "images", "frag_pct", "mathjunk", "dup", "eq", "mathml"]
    print(f"{'doc':14}" + "".join(f"{c:>10}" for c in cols))
    problems = []
    for name, r in res.items():
        if "error" in r:
            print(f"{name:14} ERROR {r['error']}")
            problems.append(f"{name}: {r['error']}")
            continue
        print(f"{name:14}" + "".join(f"{str(r[c]):>10}" for c in cols))
        b = base.get(name)
        if not b or "error" in b:
            continue
        if r["recall"] < b["recall"] - 0.005: problems.append(f"{name}: recall {b['recall']} -> {r['recall']}")
        if r["precision"] < b["precision"] - 0.005: problems.append(f"{name}: precision {b['precision']} -> {r['precision']}")
        if r["frag_pct"] > b["frag_pct"] + 2: problems.append(f"{name}: fragments {b['frag_pct']}% -> {r['frag_pct']}%")
        if r["dup"] > b["dup"]: problems.append(f"{name}: duplicated paragraphs {b['dup']} -> {r['dup']}")
        if b["valid"] and not r["valid"]: problems.append(f"{name}: EPUB no longer valid")
        if r["mathjunk"] > b["mathjunk"] + 1: problems.append(f"{name}: mathjunk {b['mathjunk']} -> {r['mathjunk']}")
    if a.no_equations or a.mathml:
        return 0
    if a.update_baseline:
        base.update(res)
        json.dump(base, open(bpath, "w"), indent=1, sort_keys=True)
        print("baseline updated")
    elif problems:
        print("\nREGRESSIONS:\n  " + "\n  ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
