"""Download the regression corpus (third-party open documents; not stored in the repo)."""
import os, subprocess, sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
DOCS = {
    "attention": "https://arxiv.org/pdf/1706.03762",      # 2-col paper, figures, tables, footnotes, outline
    "gpt3": "https://arxiv.org/pdf/2005.14165",           # 75 pp, printed TOC, many tables
    "bert": "https://arxiv.org/pdf/1810.04805",           # 2-col ACL
    "resnet": "https://arxiv.org/pdf/1512.03385",         # 2-col CVPR, tables
    "adam": "https://arxiv.org/pdf/1412.6980",            # math-heavy, algorithm box
    "vae": "https://arxiv.org/pdf/1312.6114",             # math-heavy, outline
    "thinkpython": "https://greenteapress.com/thinkpython2/thinkpython2.pdf",   # 244-page book, code
    "irs_pub": "https://www.irs.gov/pub/irs-pdf/p15b.pdf",                       # government, tables, lists
    "wiki_en": "https://en.wikipedia.org/api/rest_v1/page/pdf/Alan_Turing",      # web-print layout
    "wiki_de": "https://de.wikipedia.org/api/rest_v1/page/pdf/Alan_Turing",      # non-English
}
for name, url in DOCS.items():
    d = os.path.join(HERE, name)
    out = os.path.join(d, f"{name}.pdf")
    if os.path.exists(out):
        continue
    os.makedirs(d, exist_ok=True)
    subprocess.run(["curl", "-sL", "-m", "180", "-A", "Mozilla/5.0 pdf2epub-test", "-o", out, url], check=False)
    ok = open(out, "rb").read(4) == b"%PDF"
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        os.remove(out)
