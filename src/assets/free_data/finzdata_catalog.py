"""The futures markets Finzdata lists, bundled with the app so the Futures tab can search them without spending a call.

``finzdata_futures.csv`` holds the 139 markets of Finzdata's public futures page (``finzdata.com/futures-data``, October 2026): the symbol, the name, the exchange, the asset class it files the market
under and the year its history starts. It is a copy of that page's table, not a promise that a market has a recent bar or is part of your plan: Finzdata answers that when a download asks. The list the
service itself sends (``/v1/futures/roots``, one call, kept on disk) replaces a market's row and adds any market the copy does not have.

A **symbol** is a root, not a contract month: ``ES`` is the E-mini S&P 500 whichever quarter is the front one. The contracts of a root come from ``/v1/futures/contracts``. Finzdata's symbols are its own
(the euro currency future is ``E6``, the Treasury bond ``US``); where one is also a root the platform has contract terms for, the download uses those terms.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .finzdata import ALIAS, platform_spec
from .futures import FREE_ROOTS

# What the platform's margin and dollar figures call each of Finzdata's asset classes.
ASSET_CLASS = {"Equity index": "equity_index", "Interest rate": "rates", "Energy": "commodity", "Metals": "metal", "Agricultural": "agricultural", "Currency": "fx", "Volatility": "volatility",
               "Crypto": "crypto"}
# Shown first when nobody has typed anything, and ahead of the rest in a search: the markets people look for.
FEATURED = ("ES", "NQ", "YM", "RTY", "MES", "MNQ", "CL", "NG", "BZ", "HO", "RB", "GC", "SI", "HG", "PL", "PA", "ZC", "ZS", "ZW", "ZM", "ZL", "KC", "SB", "CC", "CT", "LE", "HE", "ZT", "ZF", "ZN", "US", "UB",
            "E6", "J1", "B6", "A6", "AD", "DX", "VX", "BTC", "FDAX", "FESX", "FGBL")
ROOT = re.compile(r"^[A-Z0-9]{1,6}$")


@dataclass(frozen=True)
class Market:
    root: str
    name: str
    exchange: str
    group: str
    history_from: str

    @property
    def item(self) -> str:
        """What the download endpoint is given: the root."""
        return self.root

    @property
    def featured_rank(self) -> int:
        return FEATURED.index(self.root) if self.root in FEATURED else len(FEATURED)

    def view(self, live: dict | None = None) -> dict:
        row = {"root": self.root, "item": self.item, "name": self.name, "exchange": self.exchange, "group": self.group, "history_from": self.history_from, "featured": self.root in FEATURED,
               "terms_known": platform_spec(self.root) is not None}
        if live:
            row.update({k: live.get(k) for k in ("first_date", "last_date", "contracts")})
        return row


@lru_cache(maxsize=1)
def bundled() -> tuple[Market, ...]:
    with open(Path(__file__).with_name("finzdata_futures.csv"), newline="", encoding="utf-8") as handle:
        return tuple(Market(r["root"], r["name"], r["exchange"], r["group"], r["history_from"]) for r in csv.DictReader(handle))


def markets(live: list[dict] | None = None) -> tuple[list[Market], dict[str, dict]]:
    """``(markets, live_by_root)``: the bundled list with the rows the service sent laid over it (its name, exchange and class win; a market only it has is added)."""
    by_root = {m.root: m for m in bundled()}
    seen = {r["root"]: r for r in (live or []) if ROOT.match(str(r.get("root", "")))}
    for root, r in seen.items():
        old = by_root.get(root)
        by_root[root] = Market(root, str(r.get("name") or (old.name if old else root)), str(r.get("exchange") or (old.exchange if old else "?")), str(r.get("group") or (old.group if old else "Other")),
                               (str(r["first_date"])[:4] if r.get("first_date") else (old.history_from if old else "")))
    return list(by_root.values()), seen


def groups(live: list[dict] | None = None) -> list[str]:
    """The asset classes of the list, most markets first."""
    counts: dict[str, int] = {}
    for m in markets(live)[0]:
        counts[m.group] = counts.get(m.group, 0) + 1
    return [g for g, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def find(root: str, live: list[dict] | None = None) -> Market | None:
    root = str(root or "").strip().lstrip("/").upper()
    return next((m for m in markets(live)[0] if m.root == root), None)


def terms(root: str, market: Market | None = None) -> dict:
    """What a download of ``root`` records of its contract: the platform's own terms where it has them (``FREE_ROOTS``), else the name, exchange and class from Finzdata's list with a multiplier of 1 and a
    tick of 0.01, which are placeholders (the notes of the download say so)."""
    spec = platform_spec(str(root))
    if spec is not None:
        return {**spec.terms(), "root": str(root).upper().lstrip("/")}
    market = market or find(root)
    return {"root": str(root).upper(), "exchange": market.exchange if market else "?", "asset_class": ASSET_CLASS.get(market.group if market else "", "unknown"), "multiplier": 1.0, "tick_size": 0.01,
            "root_name": market.name if market else str(root).upper()}


def _score(m: Market, q: str) -> int | None:
    """How well a market matches a lower-case word: 0 the symbol itself, 1 starts with it, 2 a word of the name starts with it, 3 the name or the exchange contains it, ``None`` no match."""
    root, name = m.root.lower(), m.name.lower()
    if root == q:
        return 0
    if root.startswith(q):
        return 1
    if any(word.startswith(q) for word in re.split(r"[^a-z0-9]+", name)):
        return 2
    if q in name or q in m.exchange.lower():
        return 3
    return None


def search(query: str = "", group: str | None = None, limit: int = 40, live: list[dict] | None = None) -> dict:
    """Markets matching ``query`` (every word must match the symbol, the name or the exchange), best first; with no query, the featured markets (or, with a group chosen, all of it). ``{results, total,
    shown}``. ``live`` is the list the service sent, laid over the bundled one."""
    words = [w for w in re.split(r"\s+", str(query or "").strip().lstrip("/").lower()) if w]
    listed, seen = markets(live)
    ranked = []
    for m in listed:
        if group and m.group != group:
            continue
        if not words:
            if m.root in FEATURED or group:
                ranked.append((0, m))
            continue
        scores = [_score(m, w) for w in words]
        if any(s is None for s in scores):
            continue
        ranked.append((max(scores), m))
    ranked.sort(key=lambda kv: (kv[0], kv[1].featured_rank, kv[1].root))
    limit = max(1, min(int(limit), 200))
    return {"results": [m.view(seen.get(m.root)) for _, m in ranked[:limit]], "total": len(ranked), "shown": min(limit, len(ranked))}


def platform_overlap(live: list[dict] | None = None) -> dict:
    """Which of Finzdata's symbols are also roots the platform has terms and expiry rules for, and which of the platform's are not in Finzdata's list under the same symbol."""
    listed = {m.root for m in markets(live)[0]}
    known = {r for r in listed if platform_spec(r) is not None}
    covered = {r if r in FREE_ROOTS else ALIAS[r] for r in known}
    return {"both": sorted(known), "only_finzdata": len(listed - known), "only_platform": sorted(set(FREE_ROOTS) - covered)}
