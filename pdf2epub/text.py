"""Text helpers: normalisation, regexes, language guess."""
from __future__ import annotations

import re
import unicodedata

_XML_BAD = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿\ud800-\udfff]")
_LIGS = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
         "ﬅ": "st", "ﬆ": "st", "­": ""}
_PUA_MAP = {
    "": "⎛", "": "⎜", "": "⎝",
    "": "⎞", "": "⎟", "": "⎠",
    "": "⎡", "": "⎢", "": "⎣",
    "": "⎤", "": "⎥", "": "⎦",
    "": "⎧", "": "⎨", "": "⎩",
    "": "⎫", "": "⎬", "": "⎭",
    "": "", "": "", "": "",
}
_LIG_RE = re.compile("[" + "".join(_LIGS) + "]")

BULLETS = "•●▪■◦○‣·∙►▸–—-*◆◇□☐✓✔➢➤"
# Private-use glyphs commonly produced by Symbol / Wingdings bullets.
_PUA_BULLET = re.compile(r"^[-]\s*")
BULLET_RE = re.compile(rf"^\s*([{re.escape(BULLETS)}])\s+")
ORDERED_RE = re.compile(r"^\s*\(?(\d{1,3}|[a-z]|[ivxlc]{1,5})([.)])\s+(?=\S)")
TERMINAL = tuple('.?!:;"\')”’»…]')
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
         "xi": 11, "xii": 12}

_STOP = {
    "en": "the of and to in is that for it with as was on be by this are from or an at which not have",
    "de": "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an",
    "fr": "le la les de des et en un une du que est pour qui dans par sur au avec ne pas plus ce",
    "es": "el la los de las y en un una que es por con para del se no al lo como más su",
    "it": "il di che e la in un per è con non una le del da si dei al come più ma sono",
    "pt": "o a os de e que do da em um para com não uma os no se na por mais as dos como",
    "nl": "de het een en van in is dat op te voor met zijn niet aan er maar om ook als bij",
}
_STOP = {k: set(v.split()) for k, v in _STOP.items()}


_PUA_RE = re.compile("[" + "".join(re.escape(k) for k in _PUA_MAP) + "]")
_SPACED_URL = re.compile(r"^(\s*)(https?)\s*:\s+//\s*(.*)|^(\s*)(https?)\s+:\s*//\s*(.*)", re.DOTALL)
_SPACED_URL_PATH = re.compile(r"^(\s*)//\s*(\S.*\.\s*\S.*)", re.DOTALL)


def clean(text: str) -> str:
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = _LIG_RE.sub(lambda m: _LIGS[m.group()], text)
    text = _PUA_RE.sub(lambda m: _PUA_MAP.get(m.group(), ""), text)
    text = _XML_BAD.sub("", text)
    m = _SPACED_URL.match(text)
    if m:
        if m.group(2) is not None:
            lead, proto, rest = m.group(1), m.group(2), m.group(3)
        else:
            lead, proto, rest = m.group(4), m.group(5), m.group(6)
        text = lead + proto + "://" + rest.replace(" ", "")
    else:
        m = _SPACED_URL_PATH.match(text)
        if m:
            text = m.group(1) + "//" + m.group(2).replace(" ", "")
    return text


def norm_key(text: str) -> str:
    """Aggressive normalisation for comparing titles / running heads."""
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"[^\w#]+", " ", text).strip()


def norm_title(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    return re.sub(r"[^\w]+", " ", text).strip()


def strip_numbering(text: str) -> str:
    text = re.sub(
        r"^(?:chapter|part|section|appendix|annex)?\s*"
        r"(?:[\dIVXLC]+[.:)]?\s+|[\d.]+\s+|[A-Z][.:)]\s+|(?-i:[A-Z])\s+(?=(?-i:[A-Z])))",
        "", text, flags=re.I,
    ).strip()
    return norm_title(text)


def guess_language(text: str) -> str:
    words = re.findall(r"[^\W\d_]+", text.lower())[:6000]
    if len(words) < 30:
        return "en"
    scores = {k: sum(1 for w in words if w in v) for k, v in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.04 * len(words) else "en"


def roman_to_int(s: str) -> int | None:
    return ROMAN.get(s.lower())


def list_marker(text: str):
    """Return (kind, marker_len, number, raw) for a line starting a list item, else None."""
    m = BULLET_RE.match(text)
    if m:
        return ("bullet", m.end(), None, m.group(1))
    m = _PUA_BULLET.match(text)
    if m:
        return ("bullet", m.end(), None, "•")
    m = ORDERED_RE.match(text)
    if m:
        tok = m.group(1)
        if tok.isdigit():
            return ("ordered", m.end(), int(tok), tok)
        if len(tok) == 1 and tok.isalpha():
            if tok in "ivxlc" and tok != "i" and tok not in "cdlmvx":
                pass
            return ("alpha", m.end(), ord(tok) - 96, tok)
        n = roman_to_int(tok)
        if n:
            return ("roman", m.end(), n, tok)
    return None


_WORDS: set[str] | None = None
_COMPOUND_PREFIX = {"self", "well", "ill", "half", "quasi", "ex", "all", "cross", "high", "low", "long", "short"}


def english_words() -> set[str]:
    """System word list when available (macOS/Linux); empty set otherwise."""
    global _WORDS
    if _WORDS is None:
        _WORDS = set()
        for path in ("/usr/share/dict/words", "/usr/share/dict/american-english", "/usr/dict/words"):
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    _WORDS = {w.strip().lower() for w in fh if w.strip().isalpha()}
                break
            except OSError:
                continue
    return _WORDS


def known_word(w: str, words: set[str]) -> bool:
    """Dictionary lookup that tolerates the inflections missing from small word lists."""
    w = w.lower()
    if w in words:
        return True
    for suf, repl in (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("es", ""), ("s", ""), ("ly", ""),
                      ("ers", ""), ("er", ""), ("ness", ""), ("ment", ""), ("ments", ""), ("ies", "y"), ("ied", "y")):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and (w[: -len(suf)] + repl) in words:
            return True
    return False
