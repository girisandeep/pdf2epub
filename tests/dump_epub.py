import re, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 160
for n in sorted(z.namelist()):
    if n.startswith("OEBPS/chap"):
        body = z.read(n).decode()
        body = body[body.index("<body"):]
        print("=====", n)
        for part in re.split(r"(?=<(?:h\d|p|ul|ol|pre|figure|table|aside)[ >])", body):
            part = part.strip()
            if part and not part.startswith("<body"):
                print(part[:lim].replace("\n", "\\n"))
