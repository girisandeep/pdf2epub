"""Benchmark pdf2epub against Calibre and Pandoc on the test corpus.

Usage:
    python tests/benchmark.py              # run all tools on all corpus docs
    python tests/benchmark.py thinkpython  # single document
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

CORPUS = os.path.join(HERE, "corpus")
NS = {"x": "http://www.w3.org/1999/xhtml"}

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
_HYPHEN_EOL = re.compile(r"(\w)-\s*\n\s*(\w)")


def toks(s: str) -> Counter:
    s = _HYPHEN_EOL.sub(r"\1\2", s)
    return Counter(re.findall(r"[^\W_]+", s.lower().translate(_LIGATURES).replace("-", "")))


def pdf_ref_tokens(pdf: str) -> Counter:
    import pymupdf
    return toks(" ".join(p.get_text() for p in pymupdf.open(pdf)))


def evaluate_epub(epub_path: str, ref: Counter) -> dict:
    """Evaluate an EPUB using the same metrics as run_corpus.py."""
    from epub_check import check

    try:
        z = zipfile.ZipFile(epub_path)
    except Exception as e:
        return {"error": str(e)}

    paras, body = [], []
    headings = 0
    chapters = 0
    tables = 0
    images = 0
    lists = 0
    footnotes = 0
    has_toc = False
    has_page_list = False
    has_cover = False
    has_metadata = {}

    # Check OPF for metadata
    opf_names = [n for n in z.namelist() if n.endswith(".opf")]
    if opf_names:
        try:
            opf = z.read(opf_names[0]).decode()
            has_metadata["title"] = bool(re.search(r"<dc:title>[^<]+</dc:title>", opf))
            has_metadata["author"] = bool(re.search(r"<dc:creator>[^<]+</dc:creator>", opf))
            has_metadata["language"] = bool(re.search(r"<dc:language>[^<]+</dc:language>", opf))
            has_metadata["date"] = bool(re.search(r"<dc:date>", opf))
            has_metadata["description"] = bool(re.search(r"<dc:description>", opf))
            has_metadata["accessibility"] = bool(re.search(r"schema:accessibilityFeature", opf))
            has_metadata["page_numbers"] = bool(re.search(r"printPageNumbers", opf))
        except Exception:
            pass

    # Check nav for TOC and page-list
    nav_names = [n for n in z.namelist() if "nav" in n.lower() and n.endswith(".xhtml")]
    for nav_name in nav_names:
        try:
            nav = z.read(nav_name).decode()
            if 'epub:type="toc"' in nav or "type='toc'" in nav:
                has_toc = True
            if "page-list" in nav:
                has_page_list = True
        except Exception:
            pass

    has_cover = any("cover" in n.lower() for n in z.namelist() if n.endswith((".xhtml", ".jpg", ".png")))

    # Count structural elements in chapter files
    chap_files = sorted(n for n in z.namelist()
                        if n.endswith((".xhtml", ".html"))
                        and not any(x in n.lower() for x in ("nav.", "toc.", "cover", "title")))

    chapters = len(chap_files)

    for n in chap_files:
        try:
            raw = z.read(n)
            # Try to parse as XML first, fall back to permissive HTML parsing
            try:
                doc = etree.fromstring(raw)
            except Exception:
                parser = etree.HTMLParser()
                doc = etree.fromstring(raw, parser)

            body.append(" ".join(doc.xpath("//x:body//text()", namespaces=NS)))
            # Also try without namespace for Calibre/Pandoc output
            body.append(" ".join(doc.xpath("//body//text()")))

            for tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                headings += len(doc.xpath(f"//x:{tag}", namespaces=NS))
                headings += len(doc.xpath(f"//{tag}"))

            tables += len(doc.xpath("//x:table", namespaces=NS))
            tables += len(doc.xpath("//table"))

            images += len(doc.xpath("//x:img", namespaces=NS))
            images += len(doc.xpath("//img"))

            lists += len(doc.xpath("//x:ul|//x:ol", namespaces=NS))
            lists += len(doc.xpath("//ul|//ol"))

            footnotes += len(doc.xpath("//*[@class='footnote']", namespaces=NS))
            footnotes += len(doc.xpath("//*[@class='footnote']"))

            for p in doc.xpath("//x:p|//x:li|//x:dd|//x:dt", namespaces=NS):
                tag = p.tag.rsplit("}", 1)[-1]
                is_li = tag in ("li", "dt", "dd")
                paras.append(("".join(p.itertext()).strip(),
                              len(p.xpath(".//x:sub|.//x:sup", namespaces=NS)), is_li))
            for p in doc.xpath("//p|//li|//dd|//dt"):
                txt = "".join(p.itertext()).strip()
                if txt:
                    tag = p.tag.rsplit("}", 1)[-1] if "}" in p.tag else p.tag
                    is_li = tag in ("li", "dt", "dd")
                    paras.append((txt, len(p.xpath(".//sub|.//sup")), is_li))
        except Exception:
            continue

    got = toks(" ".join(body))
    common = sum((ref & got).values())
    texts = [t for t, _, _ in paras if t]
    frag = sum(1 for t, _, li in paras if t and not li and len(t) < 40
               and not t.endswith((".", "?", "!", ":", '"', "”", "]", ")")))
    dup = sum(1 for a, b in zip(texts, texts[1:]) if a == b and len(a) >= 60)
    mathjunk = sum(1 for t, s, _ in paras if s >= 3 and len(t) < 120 and len(t) / s < 15)

    valid_errors = check(epub_path)

    file_size = os.path.getsize(epub_path)

    return {
        "recall": round(common / max(sum(ref.values()), 1), 4),
        "precision": round(common / max(sum(got.values()), 1), 4),
        "valid": valid_errors == [],
        "valid_errors": len(valid_errors),
        "chapters": chapters,
        "headings": headings,
        "tables": tables,
        "images": images,
        "lists": lists,
        "footnotes": footnotes,
        "paras": len(texts),
        "frag_pct": round(100 * frag / max(len(texts), 1), 1),
        "mathjunk": mathjunk,
        "dup": dup,
        "has_toc": has_toc,
        "has_page_list": has_page_list,
        "has_cover": has_cover,
        "metadata": has_metadata,
        "file_size_kb": round(file_size / 1024, 1),
    }


def run_pdf2epub(pdf: str, out: str) -> float:
    from pdf2epub import ConversionOptions, convert
    t0 = time.time()
    convert(pdf, out, ConversionOptions())
    return time.time() - t0


def run_calibre(pdf: str, out: str) -> float:
    t0 = time.time()
    subprocess.run(["ebook-convert", pdf, out], capture_output=True, timeout=300)
    return time.time() - t0


TOOLS = {
    "pdf2epub": run_pdf2epub,
    "calibre": run_calibre,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark pdf2epub against Calibre and Pandoc")
    ap.add_argument("names", nargs="*", help="corpus names to benchmark (default: all)")
    ap.add_argument("--tools", nargs="*", default=list(TOOLS.keys()),
                    help="tools to benchmark (default: all)")
    ap.add_argument("--output", default=os.path.join(HERE, "benchmark_results.json"),
                    help="output JSON path")
    a = ap.parse_args()

    # Check tool availability
    available_tools = {}
    for t in a.tools:
        if t == "pdf2epub":
            available_tools[t] = TOOLS[t]
        elif t == "calibre" and shutil.which("ebook-convert"):
            available_tools[t] = TOOLS[t]
        elif t == "pandoc" and shutil.which("pandoc"):
            available_tools[t] = TOOLS[t]
        else:
            print(f"  {t}: not found, skipping", file=sys.stderr)

    if not available_tools:
        print("No tools available", file=sys.stderr)
        return 1

    dirs = sorted(d for d in os.listdir(CORPUS) if os.path.isdir(os.path.join(CORPUS, d)))
    if a.names:
        dirs = [d for d in dirs if d in a.names]

    results = {}
    for name in dirs:
        pdfs = [f for f in os.listdir(os.path.join(CORPUS, name)) if f.endswith(".pdf")]
        if not pdfs:
            continue
        pdf = os.path.join(CORPUS, name, pdfs[0])
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        ref = pdf_ref_tokens(pdf)
        results[name] = {}

        for tool_name, tool_fn in available_tools.items():
            tmpdir = tempfile.mkdtemp()
            out = os.path.join(tmpdir, f"{name}.epub")
            print(f"  {tool_name:12s} ... ", end="", flush=True)

            try:
                secs = tool_fn(pdf, out)
                if os.path.exists(out):
                    metrics = evaluate_epub(out, ref)
                    metrics["secs"] = round(secs, 1)
                    results[name][tool_name] = metrics
                    v = "✓" if metrics["valid"] else "✗"
                    print(f"{secs:5.1f}s  recall={metrics['recall']:.4f}  "
                          f"precision={metrics['precision']:.4f}  "
                          f"valid={v}  headings={metrics['headings']}  "
                          f"tables={metrics['tables']}  images={metrics['images']}  "
                          f"frag={metrics['frag_pct']}%")
                else:
                    results[name][tool_name] = {"error": "no output file produced"}
                    print("FAILED (no output)")
            except Exception as e:
                results[name][tool_name] = {"error": str(e)}
                print(f"FAILED ({e})")

    # Save results
    with open(a.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {a.output}")

    # Print summary table
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")

    # Aggregate metrics
    tool_names = list(available_tools.keys())
    agg = {t: {"recall": [], "precision": [], "valid": 0, "total": 0, "headings": 0,
               "tables": 0, "images": 0, "frag": [], "secs": [], "toc": 0,
               "page_list": 0, "metadata_fields": []} for t in tool_names}

    header = f"{'doc':15s}"
    for t in tool_names:
        header += f" | {'recall':>8s} {'prec':>8s} {'valid':>5s} {'heads':>5s} {'tbls':>5s} {'imgs':>5s} {'frag%':>5s} {'secs':>5s}"
    print(header)
    print("-" * len(header))

    for name in dirs:
        if name not in results:
            continue
        row = f"{name:15s}"
        for t in tool_names:
            m = results[name].get(t, {})
            if "error" in m:
                row += f" | {'ERR':>8s} {'':>8s} {'':>5s} {'':>5s} {'':>5s} {'':>5s} {'':>5s} {'':>5s}"
                continue
            v = "✓" if m.get("valid") else "✗"
            row += (f" | {m.get('recall',0):8.4f} {m.get('precision',0):8.4f} "
                    f"{v:>5s} {m.get('headings',0):5d} {m.get('tables',0):5d} "
                    f"{m.get('images',0):5d} {m.get('frag_pct',0):5.1f} {m.get('secs',0):5.1f}")
            agg[t]["recall"].append(m.get("recall", 0))
            agg[t]["precision"].append(m.get("precision", 0))
            agg[t]["valid"] += 1 if m.get("valid") else 0
            agg[t]["total"] += 1
            agg[t]["headings"] += m.get("headings", 0)
            agg[t]["tables"] += m.get("tables", 0)
            agg[t]["images"] += m.get("images", 0)
            agg[t]["frag"].append(m.get("frag_pct", 0))
            agg[t]["secs"].append(m.get("secs", 0))
            agg[t]["toc"] += 1 if m.get("has_toc") else 0
            agg[t]["page_list"] += 1 if m.get("has_page_list") else 0
            meta = m.get("metadata", {})
            agg[t]["metadata_fields"].append(sum(1 for v in meta.values() if v))
        print(row)

    print("-" * len(header))
    # Averages
    avg_row = f"{'AVERAGE':15s}"
    for t in tool_names:
        a_ = agg[t]
        n = max(len(a_["recall"]), 1)
        avg_r = sum(a_["recall"]) / n
        avg_p = sum(a_["precision"]) / n
        avg_f = sum(a_["frag"]) / n
        avg_s = sum(a_["secs"]) / n
        vstr = f"{a_['valid']}/{a_['total']}"
        avg_row += (f" | {avg_r:8.4f} {avg_p:8.4f} "
                    f"{vstr:>5s} {a_['headings']:5d} {a_['tables']:5d} "
                    f"{a_['images']:5d} {avg_f:5.1f} {avg_s:5.1f}")
    print(avg_row)

    # Feature comparison
    print(f"\n{'='*80}")
    print("FEATURE COMPARISON")
    print(f"{'='*80}")
    feat_header = f"{'Feature':35s}"
    for t in tool_names:
        feat_header += f" | {t:>12s}"
    print(feat_header)
    print("-" * len(feat_header))

    features = [
        ("Avg Recall (text accuracy)", lambda t: f"{sum(agg[t]['recall'])/max(len(agg[t]['recall']),1):.4f}"),
        ("Avg Precision", lambda t: f"{sum(agg[t]['precision'])/max(len(agg[t]['precision']),1):.4f}"),
        ("Valid EPUBs", lambda t: f"{agg[t]['valid']}/{agg[t]['total']}"),
        ("Total Headings detected", lambda t: str(agg[t]["headings"])),
        ("Total Tables detected", lambda t: str(agg[t]["tables"])),
        ("Total Images extracted", lambda t: str(agg[t]["images"])),
        ("Avg Fragment %", lambda t: f"{sum(agg[t]['frag'])/max(len(agg[t]['frag']),1):.1f}%"),
        ("Avg Time (seconds)", lambda t: f"{sum(agg[t]['secs'])/max(len(agg[t]['secs']),1):.1f}"),
        ("TOC Navigation", lambda t: f"{agg[t]['toc']}/{agg[t]['total']}"),
        ("Page-list Navigation", lambda t: f"{agg[t]['page_list']}/{agg[t]['total']}"),
        ("Avg Metadata Fields", lambda t: f"{sum(agg[t]['metadata_fields'])/max(len(agg[t]['metadata_fields']),1):.1f}/7"),
    ]
    for feat_name, feat_fn in features:
        row = f"{feat_name:35s}"
        for t in tool_names:
            row += f" | {feat_fn(t):>12s}"
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
