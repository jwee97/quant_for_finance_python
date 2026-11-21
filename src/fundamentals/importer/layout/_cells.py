"""Cell text to numbers, shared by every physical reader."""

from __future__ import annotations

import re

DASHES = {"-", "–", "—", "―", "−", "‒", "nil", "n/a", "na", "n.a.", "—-", "--"}
_PLAIN = re.compile(r"^\d+(\.\d+)?$")
_US = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")
_EU = re.compile(r"^\d{1,3}(\.\d{3})+(,\d+)?$")
_SPACE = re.compile(r"^\d{1,3}([   ]\d{3})+(,\d+|\.\d+)?$")


def to_number(text: str) -> tuple[float | None, bool]:
    """``(number, is_dash)`` for a cell's text. ``(1,234)``, ``△1,234`` and ``-1,234`` are negative; ``1.234,56`` (European) and ``15 170`` (spaced thousands) are read; a percentage, a date, a
    year heading or anything with letters is not a number. A dash is ``(None, True)``: a figure that is not there."""
    t = (text or "").strip().replace("−", "-").replace(" ", " ")
    if not t:
        return None, False
    if t.lower() in DASHES:
        return None, True
    neg = False
    if t.startswith("(") and t.rstrip("*").endswith(")"):
        neg, t = True, t[1:t.rindex(")")]
    elif t[:1] in ("△", "▲", "-"):
        neg, t = True, t[1:]
    elif t.endswith("-") and t[:-1].strip()[:1].isdigit():
        neg, t = True, t[:-1]
    t = re.sub(r"^[$€£¥￥₩]|[$€£¥￥₩]$", "", t.strip()).strip()
    t = re.sub(r"[*†‡]+$", "", t).strip()
    t = re.sub(r"^(HK|US|S|NT|R|A|C)\$", "", t).strip()
    if t.endswith("%") or not t or not t[0].isdigit():
        return None, False
    if re.match(r"^\d{1,3}(\.\d{3})+$", t):                                                         # 1.300 / 12.345.678: dots are thousands separators (statements do not print three decimals)
        v = float(t.replace(".", ""))
    elif _PLAIN.match(t):
        v = float(t)
    elif _US.match(t):
        v = float(t.replace(",", ""))
    elif _EU.match(t):
        v = float(t.replace(".", "").replace(",", "."))
    elif _SPACE.match(t):
        v = float(re.sub(r"[   ]", "", t).replace(",", "."))
    else:
        return None, False
    return (-v if neg else v), False


__all__ = ["to_number", "DASHES"]
