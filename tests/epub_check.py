"""Structural EPUB validation (stand-in for epubcheck, which needs Java)."""
from __future__ import annotations

import posixpath
import zipfile
from urllib.parse import unquote, urldefrag

from lxml import etree

NS = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/",
      "x": "http://www.w3.org/1999/xhtml", "c": "urn:oasis:names:tc:opendocument:xmlns:container",
      "ncx": "http://www.daisy.org/z3986/2005/ncx/"}


def check(path: str) -> list[str]:
    errs: list[str] = []
    z = zipfile.ZipFile(path)
    names = z.namelist()
    if names[0] != "mimetype" or z.getinfo("mimetype").compress_type != 0 or z.read("mimetype") != b"application/epub+zip":
        errs.append("mimetype must be the first entry, stored, with the exact content")
    cont = etree.fromstring(z.read("META-INF/container.xml"))
    opf_path = cont.xpath("//c:rootfile/@full-path", namespaces=NS)[0]
    base = posixpath.dirname(opf_path)
    opf = etree.fromstring(z.read(opf_path))
    for req in ("dc:identifier", "dc:title", "dc:language"):
        if not opf.xpath(f"//{req}", namespaces=NS):
            errs.append(f"missing {req}")
    if not opf.xpath("//opf:meta[@property='dcterms:modified']", namespaces=NS):
        errs.append("missing dcterms:modified")
    manifest = {i.get("id"): i for i in opf.xpath("//opf:manifest/opf:item", namespaces=NS)}
    hrefs = {posixpath.normpath(posixpath.join(base, i.get("href"))): i for i in manifest.values()}
    for h in hrefs:
        if h not in names:
            errs.append(f"manifest item missing from zip: {h}")
    for ir in opf.xpath("//opf:spine/opf:itemref/@idref", namespaces=NS):
        if ir not in manifest:
            errs.append(f"spine idref not in manifest: {ir}")
    if len(opf.xpath("//opf:manifest/opf:item[@properties='nav' or contains(@properties,'nav')]", namespaces=NS)) != 1:
        errs.append("need exactly one nav item")
    ids_by_file: dict[str, set[str]] = {}
    docs = {}
    for h, item in hrefs.items():
        if item.get("media-type") == "application/xhtml+xml":
            try:
                doc = etree.fromstring(z.read(h))
            except etree.XMLSyntaxError as e:
                errs.append(f"{h}: not well-formed: {e}")
                continue
            docs[h] = doc
            ids = doc.xpath("//@id")
            if len(ids) != len(set(ids)):
                errs.append(f"{h}: duplicate ids {[i for i in set(ids) if ids.count(i) > 1][:5]}")
            ids_by_file[h] = set(ids)
            if not doc.xpath("//x:head/x:title", namespaces=NS):
                errs.append(f"{h}: missing <title>")
    # every local link / image resolves
    for h, doc in docs.items():
        d = posixpath.dirname(h)
        for ref in doc.xpath("//x:a/@href | //x:img/@src", namespaces=NS):
            if ref.startswith(("http:", "https:", "mailto:")):
                continue
            target, frag = urldefrag(ref)
            tpath = posixpath.normpath(posixpath.join(d, unquote(target))) if target else h
            if tpath not in names:
                errs.append(f"{h}: broken reference {ref}")
            elif frag and tpath in ids_by_file and frag not in ids_by_file[tpath]:
                errs.append(f"{h}: missing anchor {ref}")
    # nav links
    for h, item in hrefs.items():
        if "nav" in (item.get("properties") or ""):
            nav = docs[h]
            if not nav.xpath("//x:nav[@*[local-name()='type']='toc']//x:a", namespaces=NS):
                errs.append("nav has no toc links")
            d = posixpath.dirname(h)
            for ref in nav.xpath("//x:a/@href", namespaces=NS):
                target, frag = urldefrag(ref)
                tpath = posixpath.normpath(posixpath.join(d, target))
                if tpath not in names or (frag and frag not in ids_by_file.get(tpath, set())):
                    errs.append(f"nav: broken link {ref}")
    ncx = [h for h, i in hrefs.items() if i.get("media-type") == "application/x-dtbncx+xml"]
    for h in ncx:
        try:
            etree.fromstring(z.read(h))
        except etree.XMLSyntaxError as e:
            errs.append(f"{h}: {e}")
    return errs


if __name__ == "__main__":
    import sys
    bad = 0
    for p in sys.argv[1:]:
        e = check(p)
        print(p.split("/")[-1], "OK" if not e else f"{len(e)} problems")
        for x in e[:20]:
            print("   ", x)
        bad += bool(e)
    sys.exit(bad)
