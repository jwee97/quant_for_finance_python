"""MarketDataService: incremental downloads, gap filling, corporate-action tracking, and the reconstruction of historical option chains and futures curves from what Massive does serve.

* **Incremental.** A dataset already stored is extended from its last bar (a few days back, to catch revisions), never downloaded whole again; a request that starts earlier than the stored range fetches
  only the earlier part; the gaps found inside the range are filled from the same provider. A new split since the last synchronisation re-fetches an adjusted series (its whole history is re-adjusted)
  and the version log says so.
* **No mixing.** Everything is stored under the provider that sent it. This module only ever asks Massive; legacy iTick series stay under ``itick``.
* **Never fabricated.** A past option chain is rebuilt from the contract list of that date and each contract's own daily bar: contracts with no bar that day are left out, there is no bid or ask, and
  implied volatility and Greeks are computed here and labelled ``calculated`` (Massive's snapshot values are labelled ``provider``).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..derivatives.greeks import bsm_greeks
from ..derivatives.iv import implied_vol
from .massive import MassiveProvider
from .models import Bars, BarsRequest, MarketDataError, Table
from .storage import MarketStore, is_daily

OVERLAP_DAYS = 5
FRESH_SECONDS = 6 * 3600
CHUNK_DAYS = {"second": 1, "minute": 30, "hour": 180}


def _today() -> str:
    return str(datetime.now(timezone.utc).date())


class MarketDataService:
    def __init__(self, root, provider: MassiveProvider | None = None, store: MarketStore | None = None):
        self.root = root
        self.provider = provider or MassiveProvider(root=root)
        self.store = store or MarketStore(root)

    # ------------------------------------------------------------------------------------------------------------ status
    def status(self) -> dict:
        """Connection, verified entitlements, coverage per asset class, last synchronisation, freshness and warnings: what the dashboards show."""
        d = self.provider.describe()
        caps = [c.as_dict() for c in self.provider.get_provider_capabilities()]
        series = self.store.list_series()
        by: dict = {}
        for s in series:
            a = by.setdefault(f"{s['provider']}:{s['asset_class']}", {"provider": s["provider"], "asset_class": s["asset_class"], "series": 0, "rows": 0, "first": None, "last": None, "last_sync": None, "stale": 0})
            a["series"] += 1
            a["rows"] += s["rows"] or 0
            a["first"] = min(filter(None, [a["first"], s["first_ts"]]), default=None)
            a["last"] = max(filter(None, [a["last"], s["last_ts"]]), default=None)
            a["last_sync"] = max(filter(None, [a["last_sync"], s["last_sync"]]), default=None)
            if s["last_ts"] and s["timespan"] == "day" and s["status"] != "expired" and (pd.Timestamp(_today()) - pd.Timestamp(s["last_ts"][:10])).days > 5:
                a["stale"] += 1
        warnings = []
        if not d["configured"]:
            warnings.append(d["problem"])
        for a in by.values():
            if a["stale"]:
                warnings.append(f"{a['stale']} {a['provider']} {a['asset_class']} series end more than five days ago: synchronise them")
        refused = [f"{c['asset_class']} {c['dataset']}" for c in caps if c["status"] == "not_entitled"]
        if refused:
            warnings.append("not included in your plan (as Massive answered): " + ", ".join(refused))
        verified = [c for c in caps if c["checked_at"]]
        return {"provider": "massive", "label": "Massive", **d, "capabilities": caps, "verified": len(verified), "coverage": list(by.values()), "jobs": self.store.jobs(5), "warnings": warnings,
                "integrity": self.store.integrity(), "last_probe": max((c["checked_at"] for c in caps if c["checked_at"]), default=None)}

    # ------------------------------------------------------------------------------------------------------------ bars
    def _plan(self, key: tuple, ac: str, symbol: str, start: str | None, end: str, adjusted: bool, force: bool) -> list[tuple[str, str, str]]:
        info = self.store.series_info(key)
        if info is None:
            return [(start or str(pd.Timestamp(end) - pd.Timedelta(days=365 * 5))[:10], end, "initial download")]
        need: list[tuple[str, str, str]] = []
        first, last = info["first_ts"][:10], info["last_ts"][:10]
        if start and start < first:
            need.append((start, first, "earlier history"))
        fresh = (time.time() - time.mktime(time.strptime(info["last_sync"], "%Y-%m-%dT%H:%M:%SZ")) + time.timezone) < FRESH_SECONDS if info.get("last_sync") else False
        if info.get("status") != "expired" and end > last and (force or not fresh):
            need.append((str((pd.Timestamp(last) - pd.Timedelta(days=OVERLAP_DAYS)).date()), end, "newer bars"))
        if key[3] == "day":
            for a, b, _n in self.store.missing_ranges(key, start):
                need.append((a, b, "gap fill"))
        return need

    def sync_bars(self, symbol: str, asset_class: str = "stocks", start: str | None = None, end: str | None = None, timespan: str = "day", multiplier: int = 1, adjusted: bool = True,
                  progress=None, cancel=None, force: bool = False) -> dict:
        """Bring a stored dataset up to date and return what changed. Safe to repeat and to interrupt: what was stored stays, and the next call continues from it."""
        say = progress or (lambda _t: None)
        end = end or _today()
        adj = adjusted if asset_class == "stocks" else False if asset_class == "futures" else adjusted
        key = self.store.key("massive", asset_class, symbol, timespan, multiplier, adj)
        job = self.store.job_start("bars", f"{asset_class}:{symbol}:{timespan}")
        added = revised = 0
        warnings: list[str] = []
        try:
            plan = self._plan(key, asset_class, symbol, start, end, adj, force)
            if asset_class == "stocks" and adj and self.store.series_info(key) and (plan or force):
                refetch = self._split_since_sync(symbol, key)
                if refetch:
                    first = self.store.series_info(key)["first_ts"][:10]
                    plan = [(min(start or first, first), end, "re-adjusted by a split on " + refetch)]
                    warnings.append(f"a split on {refetch} re-adjusted the history of {symbol}: the stored bars were revised (see the version log)")
            if not plan:
                self.store.job_end(job, "done", "up to date")
                return {"symbol": symbol, "skipped": True, "reason": "up to date", "added": 0, "revised": 0, "info": self.store.series_info(key), "warnings": warnings}
            for a, b, why in plan:
                for lo, hi in self._windows(a, b, timespan):
                    if cancel is not None and cancel():
                        raise MarketDataError("cancelled", "cancelled")
                    say(f"{symbol} {lo} to {hi} ({why})")
                    bars = self.provider.get_historical_bars(BarsRequest(symbol, asset_class, lo, hi, multiplier, timespan, adj), cancel=cancel)
                    if len(bars.frame):
                        r = self.store.upsert_bars(key, bars.frame, bars.provenance.endpoint, reason=why)
                        added += r["added"]
                        revised += r["revised"]
                    warnings += bars.warnings
            info = self.store.series_info(key)
            if info is None:
                warnings.append(f"Massive returned no bars for {symbol} between {plan[0][0]} and {end}: check the symbol and your plan's history depth")
            else:
                q = self.store.quality(key)
                warnings += [f"{x['check']}: {x['detail']}" for x in q if x["level"] in ("error", "warning")][:4]
            self.store.job_end(job, "done", f"+{added} rows, {revised} revised")
            return {"symbol": symbol, "skipped": False, "added": added, "revised": revised, "plan": plan, "info": info, "warnings": warnings}
        except MarketDataError as exc:
            self.store.job_end(job, "cancelled" if exc.kind == "cancelled" else "failed", str(exc))
            raise

    @staticmethod
    def _windows(a: str, b: str, timespan: str) -> list[tuple[str, str]]:
        if is_daily(timespan):
            return [(a, b)]
        step = pd.Timedelta(days=CHUNK_DAYS.get(timespan, 30))
        out, lo, hi = [], pd.Timestamp(a), pd.Timestamp(b)
        while lo <= hi:
            out.append((str(lo.date()), str(min(lo + step - pd.Timedelta(days=1), hi).date())))
            lo += step
        return out

    def _split_since_sync(self, symbol: str, key: tuple) -> str | None:
        """The date of a split executed after the last synchronisation of the series, if there is one (two calls)."""
        info = self.store.series_info(key)
        try:
            t = self.provider.get_corporate_actions(symbol, max_pages=2)
        except MarketDataError:
            return None
        if len(t.frame):
            self.store.save_actions("massive", symbol, t.frame)
            s = t.frame[t.frame["kind"] == "split"]
            last_sync = pd.Timestamp(info["last_sync"][:10])
            new = s[s["date"] > last_sync]
            if len(new):
                return str(new["date"].max().date())
        return None

    def bars(self, symbol: str, asset_class: str = "stocks", start=None, end=None, timespan="day", multiplier=1, adjusted=True, provider: str = "massive") -> pd.DataFrame:
        adj = False if asset_class == "futures" else adjusted
        return self.store.read_bars(self.store.key(provider, asset_class, symbol, timespan, multiplier, adj), start, end)

    def sync_corporate_actions(self, ticker: str) -> dict:
        t = self.provider.get_corporate_actions(ticker)
        n = self.store.save_actions("massive", ticker, t.frame) if len(t.frame) else 0
        return {"ticker": ticker, "new": n, "splits": int((t.frame["kind"] == "split").sum()) if len(t.frame) else 0, "dividends": int((t.frame["kind"] == "dividend").sum()) if len(t.frame) else 0}

    # ------------------------------------------------------------------------------------------------------------ options: a past chain
    def reconstruct_option_chain(self, underlying: str, as_of: str, min_dte: int = 7, max_dte: int = 120, moneyness: float = 0.2, max_contracts: int = 80, rate: float = 0.04, dividend: float = 0.0,
                                 progress=None, cancel=None) -> Table:
        """The chain of ``underlying`` on a past date, from Massive's contract list for that date and each contract's own bar of that day.

        What it is and is not: the price of each contract is its daily CLOSE (a last-trade price, not synchronous with the underlying close), there is no bid or ask (``bid``, ``ask`` are NaN), contracts that
        did not trade that day are missing (counted in the warnings), and implied volatility and Greeks are CALCULATED here (Black-Scholes at the stated rate and yield) and labelled so. At most
        ``max_contracts`` contracts nearest the money are read (one request each)."""
        say = progress or (lambda _t: None)
        day = pd.Timestamp(as_of)
        spot_bars = self.provider.get_historical_bars(BarsRequest(underlying, "stocks", str((day - pd.Timedelta(days=7)).date()), str(day.date()), limit=10), cancel=cancel).frame
        spot_bars = spot_bars[spot_bars.index <= day]
        if not len(spot_bars):
            raise MarketDataError(f"no {underlying} price on or before {day.date()}: the chain cannot be placed", "not_found")
        spot = float(spot_bars["close"].iloc[-1])
        spot_day = spot_bars.index[-1]
        c = self.provider.get_contracts("options", underlying, as_of=str(day.date()), expired=True, expiration_from=str((day + pd.Timedelta(days=min_dte)).date()), expiration_to=str((day + pd.Timedelta(days=max_dte)).date()),
                                        strike_from=spot * (1 - moneyness), strike_to=spot * (1 + moneyness), limit=1000, max_pages=3, cancel=cancel).frame
        warns: list[str] = []
        if not len(c):
            return Table(pd.DataFrame(), self.provider._prov("/v3/reference/options/contracts", "options"), [f"Massive lists no {underlying} option contracts for {day.date()} in this window"])
        c = c.assign(dist=(c["strike_price"] - spot).abs()).sort_values(["dist", "expiration_date"]).head(max_contracts)
        rows = []
        none_traded = 0
        for i, (_, k) in enumerate(c.iterrows()):
            if cancel is not None and cancel():
                raise MarketDataError("cancelled", "cancelled")
            say(f"{k['ticker']} ({i + 1}/{len(c)})")
            b = self.provider.get_historical_bars(BarsRequest(k["ticker"], "options", str(day.date()), str(day.date()), limit=2), cancel=cancel).frame
            b = b[b.index == day.normalize()]
            if not len(b):
                none_traded += 1
                continue
            x = b.iloc[-1]
            rows.append({"date": day.normalize(), "ticker": k["ticker"], "expiry": k["expiration_date"], "strike": float(k["strike_price"]), "right": "C" if k["contract_type"] == "call" else "P", "exercise_style": k.get("exercise_style"),
                         "underlying": spot, "bid": np.nan, "ask": np.nan, "mid": float(x["close"]), "price_source": "daily_close", "volume": float(x["volume"]), "open_interest": np.nan, "rate": rate, "dividend": dividend})
        frame = pd.DataFrame(rows)
        if none_traded:
            warns.append(f"{none_traded} of {len(c)} listed contracts had no trade on {day.date()} and are left out: nothing was filled in")
        if spot_day != day.normalize():
            warns.append(f"the underlying price is the close of {spot_day.date()}, the last session on or before {day.date()}")
        warns.append("prices are daily closes (last trades), not quotes: there is no bid or ask, so spread-based filters and the fills of a backtest cannot use observed spreads")
        if len(frame):
            frame["T"] = (frame["expiry"] - frame["date"]).dt.days / 365.0
            frame["iv"] = np.nan
            for right, call in (("C", True), ("P", False)):
                m = (frame["right"] == right).to_numpy()
                if m.any():
                    s = frame[m]
                    frame.loc[m, "iv"] = implied_vol(s["mid"].to_numpy(), s["underlying"].to_numpy(), s["strike"].to_numpy(), s["T"].to_numpy(), rate, dividend, call)
                    g = bsm_greeks(s["underlying"].to_numpy(), s["strike"].to_numpy(), s["T"].to_numpy(), rate, dividend, np.where(np.isfinite(frame.loc[m, "iv"]), frame.loc[m, "iv"], np.nan), call)
                    for name in ("delta", "gamma", "vega", "theta", "rho"):
                        frame.loc[m, name] = np.asarray(g[name], dtype=float)
            frame["iv_source"] = np.where(frame["iv"].notna(), "calculated", "missing")
            frame["greeks_source"] = np.where(frame["iv"].notna(), "calculated", "missing")
            bad = int(frame["iv"].isna().sum())
            if bad:
                warns.append(f"{bad} contracts have a close outside the no-arbitrage bounds (no implied volatility): kept, marked missing")
            warns.append(f"implied volatility and Greeks are calculated with Black-Scholes at rate {rate:.2%} and dividend yield {dividend:.2%}; exercise is treated as European")
        return Table(frame, self.provider._prov("/v3/reference/options/contracts + /v2/aggs", "options", None, 1, ["historical chain rebuilt from contract bars"]), warns)

    # ------------------------------------------------------------------------------------------------------------ futures: contract-level history
    def sync_futures_chain(self, product_code: str, start: str, end: str | None = None, max_contracts: int = 40, progress=None, cancel=None) -> dict:
        """Download the session bars of every contract of a product that was alive between ``start`` and ``end``, each incrementally (an expired contract is archived: marked ``expired`` and never asked again)."""
        say = progress or (lambda _t: None)
        end = end or _today()
        c = self.provider.get_contracts("futures", product_code, expiration_from=start, limit=1000, max_pages=3, cancel=cancel).frame
        if not len(c):
            return {"product": product_code, "contracts": 0, "warnings": [f"Massive lists no {product_code} contracts expiring on or after {start}"]}
        if "first_trade_date" in c:
            c = c[c["first_trade_date"].isna() | (c["first_trade_date"] <= pd.Timestamp(end))]
        c = c.sort_values("last_trade_date").head(max_contracts)
        done, added, warns = 0, 0, []
        for _, k in c.iterrows():
            t, expiry = k["ticker"], k["last_trade_date"]
            lo = max(pd.Timestamp(start), k["first_trade_date"]) if pd.notna(k.get("first_trade_date")) else pd.Timestamp(start)
            hi = min(pd.Timestamp(end), expiry) if pd.notna(expiry) else pd.Timestamp(end)
            try:
                r = self.sync_bars(t, "futures", str(lo.date()), str(hi.date()), progress=lambda x, t=t: say(f"{t}: {x}"), cancel=cancel)
            except MarketDataError as exc:
                if exc.kind in ("entitlement", "cancelled", "auth", "not_configured"):
                    raise
                warns.append(f"{t}: {exc}")
                continue
            added += r["added"]
            done += 1
            if pd.notna(expiry) and pd.Timestamp(expiry) < pd.Timestamp(_today()):
                self.store.set_status(self.store.key("massive", "futures", t, "day", 1, False), "expired", f"archived: last trade {expiry.date()}")
        return {"product": product_code, "contracts": done, "rows_added": added, "warnings": warns}

    def futures_curve_history(self, product_code: str, start: str, end: str | None = None) -> pd.DataFrame:
        """The stored contract-level history of a product as the platform's long curve table: ``date, contract, expiry, price, ttm, open_interest, volume`` (+ ``price_source``). The price is the exchange
        settlement price of the session where there is one, else the close. Open interest is NaN: Massive's futures endpoints carry none."""
        end = end or _today()
        c = self.provider.get_contracts("futures", product_code, expiration_from=start, limit=1000, max_pages=3).frame
        frames = []
        for _, k in c.iterrows():
            b = self.store.read_bars(self.store.key("massive", "futures", k["ticker"], "day", 1, False), start, end)
            if not len(b):
                continue
            px = b["settlement_price"].where(b["settlement_price"].notna(), b["close"])
            f = pd.DataFrame({"date": b.index, "contract": k["ticker"], "expiry": k["last_trade_date"], "price": px.to_numpy(), "volume": b["volume"].to_numpy(),
                              "price_source": np.where(b["settlement_price"].notna(), "settlement", "close")})
            frames.append(f)
        if not frames:
            return pd.DataFrame(columns=["date", "contract", "expiry", "price", "ttm", "open_interest", "volume", "price_source"])
        out = pd.concat(frames, ignore_index=True)
        out = out[out["date"] <= out["expiry"]]
        out["ttm"] = ((out["expiry"] - out["date"]).dt.days / 365.0).clip(lower=0.0)
        out["open_interest"] = np.nan
        return out.sort_values(["date", "expiry"]).reset_index(drop=True)
