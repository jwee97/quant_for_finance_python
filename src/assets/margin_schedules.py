"""Versioned exchange margin schedules.

A schedule is what an exchange PUBLISHED: risk tiers by position size with initial and maintenance rates and the leverage limit, together with the source address, the time it was retrieved and the date from which
it applies. A schedule is never treated as timeless: every retrieval is stored as a new version, and ``ScheduleStore.at`` returns the version in force on a date, or ``None`` when the date is earlier than the first
version held: the platform does not back-date today's tiers.

Only an exchange's own public, unauthenticated endpoint counts as verified. Today that is OKX (``public/position-tiers``, verified live). Deribit publishes the maximum leverage and its commissions through its API
but its maintenance-margin formula is documentation only, Bybit's endpoint is refused from the network this was built on, and Binance's brackets need a signed request: for those the model is HYPOTHETICAL and says so
(``verified`` false, ``exchange_accurate`` false in every result built on it).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .free_data.http import FetchError, Http

OKX = "https://www.okx.com/api/v5"
NAME = re.compile(r"^[A-Z0-9_-]{2,40}$")
SOURCE_DOC = {"OKX": "https://www.okx.com/docs-v5/en/#public-data-rest-api-get-position-tiers"}

HYPOTHETICAL_TIERS = {                                                                         # shapes only: NOT any exchange's schedule
    "BINANCE": [(50e3, 0.004, 125), (250e3, 0.005, 100), (1e6, 0.01, 50), (5e6, 0.025, 20), (20e6, 0.05, 10), (float("inf"), 0.10, 5)],
    "BYBIT": [(2e6, 0.005, 100), (10e6, 0.01, 50), (20e6, 0.025, 25), (float("inf"), 0.05, 10)],
    "OKX": [(1e6, 0.004, 100), (5e6, 0.01, 50), (20e6, 0.025, 20), (float("inf"), 0.05, 10)],
    "DERIBIT": [(1e6, 0.005, 50), (10e6, 0.01, 25), (float("inf"), 0.025, 10)],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hypothetical(venue: str) -> dict:
    """A clearly labelled stand-in schedule, in USD notional brackets."""
    return {"venue": venue, "instrument": None, "margin_mode": "cross", "verified": False, "method": "hypothetical", "size_unit": "usd_notional", "contract_value": None, "contract_type": None, "retrieved_at": None, "effective_from": None,
            "source_url": None, "tiers": [{"tier": i + 1, "min_size": (HYPOTHETICAL_TIERS[venue][i - 1][0] if i else 0.0), "max_size": up, "imr": 1.0 / lev, "mmr": mmr, "max_leverage": lev} for i, (up, mmr, lev) in enumerate(HYPOTHETICAL_TIERS[venue])],
            "notes": ["Hypothetical: the shape of a tiered schedule, not the exchange's. Exchange-accurate liquidation estimates are NOT claimed on it."]}


def fetch_okx(http: Http, family: str = "BTC-USDT", inst: str | None = None, mode: str = "cross") -> dict:
    """OKX's published risk tiers for a swap family (``public/position-tiers``) with the contract's own value, type and settlement currency (``public/instruments``). Public endpoints, no key."""
    if mode not in ("cross", "isolated"):
        raise ValueError("mode must be cross or isolated")
    inst = inst or f"{family}-SWAP"
    d = http.get_json(f"{OKX}/public/position-tiers", {"instType": "SWAP", "tdMode": mode, "instFamily": family})
    rows = d.get("data") if isinstance(d, dict) and str(d.get("code")) == "0" else None
    if not rows:
        raise FetchError(f"OKX publishes no risk tiers for {family} ({(d or {}).get('msg') or 'no data'})", status=404)
    spec = http.get_json(f"{OKX}/public/instruments", {"instType": "SWAP", "instId": inst})
    s = (spec.get("data") or [None])[0] if isinstance(spec, dict) else None
    if not s:
        raise FetchError(f"OKX does not describe {inst}", status=404)
    tiers = [{"tier": int(r["tier"]), "min_size": float(r["minSz"]), "max_size": float(r["maxSz"]), "imr": float(r["imr"]), "mmr": float(r["mmr"]), "max_leverage": float(r["maxLever"])} for r in sorted(rows, key=lambda x: int(x["tier"]))]
    when = _now()
    return {"venue": "OKX", "instrument": inst, "family": family, "margin_mode": mode, "verified": True, "method": "official public API", "size_unit": "contracts", "contract_value": float(s["ctVal"]), "contract_value_ccy": s.get("ctValCcy"),
            "contract_type": s.get("ctType"), "settle_ccy": s.get("settleCcy"), "retrieved_at": when, "effective_from": when, "source_url": f"{OKX}/public/position-tiers", "documentation": SOURCE_DOC["OKX"], "tiers": tiers,
            "notes": ["Tiers as the endpoint returned them at the retrieval time; it states no effective date, so the retrieval time is used.", "The size unit is the instrument's contract count (per OKX's documentation; not independently confirmed): tier boundaries are converted with the contract value and a price.",
                      "Not in the endpoint: the maintenance-margin deduction, partial-liquidation steps, liquidation fees and the insurance fund."]}


class ScheduleStore:
    def __init__(self, root):
        self.dir = Path(root) / "data" / "user" / "margin_schedules"

    def _folder(self, venue: str, instrument: str, mode: str) -> Path:
        key = f"{instrument}_{mode}".upper()
        if not NAME.match(venue.upper()) or not NAME.match(key):
            raise ValueError("invalid schedule name")
        return self.dir / venue.upper() / key

    def save(self, schedule: dict) -> dict:
        if not schedule.get("verified"):
            raise ValueError("only a verified schedule is stored; a hypothetical one is built on demand")
        d = self._folder(schedule["venue"], schedule["instrument"], schedule["margin_mode"])
        d.mkdir(parents=True, exist_ok=True)
        stamp = re.sub(r"[^0-9TZ]", "", schedule["retrieved_at"])
        (d / f"{stamp}.json").write_text(json.dumps(schedule, indent=1))
        return schedule

    def versions(self, venue: str, instrument: str, mode: str = "cross") -> list[dict]:
        d = self._folder(venue, instrument, mode)
        out = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []
        return sorted(out, key=lambda s: s["effective_from"])

    def at(self, venue: str, instrument: str, mode: str = "cross", when=None) -> dict | None:
        """The version in force at ``when`` (default now): the latest whose ``effective_from`` is not after it. ``None`` before the first version: nothing is back-dated."""
        when = pd.Timestamp(when if when is not None else datetime.now(timezone.utc))
        when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
        live = [s for s in self.versions(venue, instrument, mode) if pd.Timestamp(s["effective_from"]) <= when]
        return live[-1] if live else None

    def listing(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*/*")) if self.dir.is_dir() else []:
            vs = [json.loads(f.read_text()) for f in sorted(p.glob("*.json"))]
            if vs:
                out.append({"venue": vs[-1]["venue"], "instrument": vs[-1]["instrument"], "mode": vs[-1]["margin_mode"], "versions": len(vs), "latest": vs[-1]["retrieved_at"], "first": vs[0]["effective_from"], "source_url": vs[-1]["source_url"]})
        return out


def tier_for(schedule: dict, notional_usd: float, price: float | None) -> dict:
    """The tier a position of ``notional_usd`` falls in. A schedule in contracts needs the price and its contract value to convert; one in USD notional does not. The whole position takes its tier's rates (the
    maintenance deduction of a marginal scheme is not published, so it is not applied)."""
    if schedule["size_unit"] == "contracts":
        if not price or not schedule.get("contract_value"):
            raise ValueError("a price is needed to convert a notional into contracts")
        size = notional_usd / (price * schedule["contract_value"])
    else:
        size = notional_usd
    for t in schedule["tiers"]:
        if size <= t["max_size"]:
            return {**t, "size": float(size)}
    return {**schedule["tiers"][-1], "size": float(size), "above_top_tier": True}
