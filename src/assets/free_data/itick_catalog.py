"""The products iTick lists, bundled with the app so the Futures and Crypto tabs can search them without spending a call.

``itick_futures.csv`` holds iTick's published list of futures products (about 1,400 in the United States, 180 in Hong Kong, 50 in mainland China, as published in October 2026) and
``itick_crypto.csv`` its list of coins quoted against USDT. They are copies of iTick's own product sheets (docs.itick.org, "Product List"), reshaped to ``region, code, name, exchange, group``; they
are not a promise that a product has a long history, a recent bar, or is part of your plan: iTick answers that when a download asks. Products it adds later are reachable by typing ``US:CODE``
(the finder offers it when a search finds nothing).

A futures **code** is a product, not a contract month: ``NQ`` is the E-mini Nasdaq-100 whichever quarter is the front one.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

REGIONS = ("US", "HK", "CN")
REGION_NAMES = {"US": "United States (CME, CBOT, NYMEX, COMEX, ICE US)", "HK": "Hong Kong (HKEX)", "CN": "Mainland China (CFFEX, SHFE, CZCE, DCE)"}
ITEM = re.compile(r"^(US|HK|CN):([A-Z0-9][A-Z0-9.]{0,11})$")                       # iTick's own codes include a dot for a few products (DX.Z, a trade-at-settlement contract)
# Shown first when nobody has typed anything, and ahead of the rest in a search: the products people look for.
FEATURED = {
    "US": ("ES", "NQ", "YM", "RTY", "MES", "MNQ", "CL", "NG", "BZ", "HO", "RB", "MCL", "GC", "SI", "HG", "PL", "PA", "MGC", "SIL", "ZC", "ZS", "ZW", "KE", "ZM", "ZL", "ZO", "ZR", "LE", "HE", "GF", "SB", "KC", "CC", "CT",
           "OJ", "ZT", "ZF", "ZN", "TN", "ZB", "UB", "6E", "6J", "6B", "6A", "6C", "6S", "6N", "DX", "NKD", "VX", "BTC", "ETH"),
    "HK": ("HSI", "MHI", "HHI", "HTI", "MCH", "TCH", "CUS"),
    "CN": ("IF", "IC", "IH", "IM", "T", "TF", "TS", "TL", "AU", "AG", "CU", "AL", "RB", "HC", "FU", "NI"),
}


@dataclass(frozen=True)
class Product:
    region: str
    code: str
    name: str
    exchange: str
    group: str

    @property
    def item(self) -> str:
        """What the download endpoint is given: ``US:NQ``."""
        return f"{self.region}:{self.code}"

    @property
    def featured(self) -> bool:
        return self.code in FEATURED.get(self.region, ())

    @property
    def featured_rank(self) -> int:
        """The place in the featured list (0 first); products not on it come after every featured one."""
        listed = FEATURED.get(self.region, ())
        return listed.index(self.code) if self.code in listed else len(listed)

    def view(self) -> dict:
        return {**asdict(self), "item": self.item, "featured": self.featured}


def _read(name: str) -> list[dict]:
    with open(Path(__file__).with_name(name), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def futures_products() -> tuple[Product, ...]:
    return tuple(Product(r["region"], r["code"], r["name"], r["exchange"], r["group"]) for r in _read("itick_futures.csv"))


@lru_cache(maxsize=1)
def _futures_index() -> dict[tuple[str, str], Product]:
    return {(p.region, p.code): p for p in futures_products()}


@lru_cache(maxsize=1)
def coins() -> dict[str, str]:
    """``{coin: name}`` of the coins iTick quotes against USDT."""
    return {r["coin"]: r["name"] for r in _read("itick_crypto.csv")}


def groups(region: str | None = None) -> list[str]:
    """The groups of the product list (``Energy``, ``Metals`` …), most products first."""
    counts: dict[str, int] = {}
    for p in futures_products():
        if region is None or p.region == region:
            counts[p.group] = counts.get(p.group, 0) + 1
    return [g for g, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def parse_item(text: str) -> tuple[str, str]:
    """``(region, code)`` of ``"us:nq"`` or ``"US:NQ"``. A bare code is a United States product. Raises ``ValueError`` for anything else."""
    raw = re.sub(r"\s*:\s*", ":", str(text or "").strip().upper())
    if ":" not in raw:
        raw = "US:" + raw
    match = ITEM.match(raw)
    if not match:
        raise ValueError(f"'{text}' is not an iTick futures product: write the region and the product code, such as US:NQ, HK:HSI or CN:IF")
    return match.group(1), match.group(2)


def find(region: str, code: str) -> Product | None:
    """The listed product, or ``None`` for a code the bundled list does not have (iTick may still serve it)."""
    return _futures_index().get((region, code))


def _score(p: Product, q: str) -> int | None:
    """How well a product matches a lower-case query: 0 the code itself, 1 starts with it, 2 a word of the name starts with it, 3 the name contains it, ``None`` no match."""
    code, name = p.code.lower(), p.name.lower()
    if code == q:
        return 0
    if code.startswith(q):
        return 1
    if any(word.startswith(q) for word in re.split(r"[^a-z0-9]+", name)):
        return 2
    if q in name:
        return 3
    return None


def search(query: str = "", region: str | None = None, group: str | None = None, limit: int = 40) -> dict:
    """Products matching ``query`` (words must all match the code or the name), best first; with no query, the featured products (or, when a region or group is chosen, every product in it). ``{results, total, shown}``."""
    words = [w for w in re.split(r"\s+", str(query or "").strip().lower()) if w]
    ranked = []
    for p in futures_products():
        if region and p.region != region:
            continue
        if group and p.group != group:
            continue
        if not words:
            if p.featured or region or group:                                      # browsing a market or a group lists all of it, featured first
                ranked.append((0, p))
            continue
        scores = [_score(p, w) for w in words]
        if any(s is None for s in scores):
            continue
        ranked.append((max(scores), p))
    order = {r: i for i, r in enumerate(REGIONS)}
    ranked.sort(key=lambda kv: (kv[0], 0 if kv[1].featured else 1, order[kv[1].region], kv[1].featured_rank, kv[1].code))
    limit = max(1, min(int(limit), 200))
    return {"results": [p.view() for _, p in ranked[:limit]], "total": len(ranked), "shown": min(limit, len(ranked))}


def search_coins(query: str = "", limit: int = 40) -> dict:
    """Coins iTick quotes against USDT whose symbol or name matches ``query`` (symbol first); with no query, the biggest coins. ``{results, total, shown}``."""
    q = str(query or "").strip().lower()
    listed = coins()
    if not q:
        order = [c for c in ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "LTC", "DOT", "TRX", "MATIC", "ATOM", "UNI", "XLM", "ETC", "BCH") if c in listed]
        hits = list(enumerate(order))
    else:
        hits = []
        for coin, name in listed.items():
            c, n = coin.lower(), name.lower()
            rank = 0 if c == q else 1 if c.startswith(q) else 2 if any(w.startswith(q) for w in re.split(r"[^a-z0-9]+", n)) else 3 if q in n or q in c else None
            if rank is not None:
                hits.append((rank, coin))
    hits.sort()
    limit = max(1, min(int(limit), 200))
    return {"results": [{"coin": c, "name": listed[c], "item": c} for _, c in hits[:limit]], "total": len(hits), "shown": min(limit, len(hits))}
