"""Massive as a price source of the Backtest tab: the fetcher a :class:`~webapp.universe.TickerStore` calls, the same way it calls Yahoo Finance.

Tickers are written the way Yahoo writes them and mapped to Massive's: ``AAPL`` and ETFs as they are, ``BTC-USD`` to ``X:BTCUSD`` (spot), ``^SPX`` to ``I:SPX`` (an index, if the plan includes the Indices
family). What Massive does not serve is refused with the reason, never replaced: a Yahoo continuous future (``ES=F``), a forex pair, and the other exchanges' suffixes (``.HK``, ``.T``, ...): Massive covers US
markets. Bars come from the normalised store (``MarketDataService.sync_bars``: incremental, adjusted for splits as Massive defines it) and leave as ``close, high, low, volume``, the columns the tab uses.
"""

from __future__ import annotations

import re

import pandas as pd

from .models import BarsRequest, MarketDataError
from .service import MarketDataService

MAP_HELP = ("US stocks and ETFs as AAPL or SPY, crypto as BTC-USD (spot), indices as ^SPX (if your plan has Indices). Futures have contract-level data in the Futures tab, not as ES=F here; "
            "other countries' exchanges (.HK, .T, .L ...) and forex are not served")


def map_symbol(ticker: str) -> tuple[str, str]:
    """``(Massive ticker, asset class)`` for a Yahoo-style ticker, or a ``MarketDataError`` that says why not."""
    t = str(ticker).strip().upper()
    if t.endswith("=F"):
        raise MarketDataError(f"'{ticker}' is a continuous future: Massive has the contracts themselves (Futures tab, Massive source), not a spliced series. {MAP_HELP}", "unsupported")
    if t.endswith("=X"):
        raise MarketDataError(f"'{ticker}' is a forex pair, which this source does not serve. {MAP_HELP}", "unsupported")
    if re.fullmatch(r"[A-Z0-9]{2,10}-USD", t):
        return f"X:{t[:-4]}USD", "crypto"
    if t.startswith("^"):
        return f"I:{t[1:]}", "indices"
    if re.fullmatch(r"[A-Z]{1,6}(?:[.-][A-Z])?", t):
        return t.replace("-", "."), "stocks"
    raise MarketDataError(f"Massive cannot place '{ticker}'. {MAP_HELP}", "unsupported")


class MassivePrices:
    """``fetch(tickers, start, known=None, progress=None)`` returning ``{ticker: DataFrame | message}``."""

    supports_context = True
    name, label = "massive", "Massive"

    def __init__(self, service: MarketDataService):
        self.service = service

    # --- the interface Source and the Backtest tab's status use
    def available(self) -> tuple[bool, str]:
        return self.service.provider.client.available()

    def describe(self) -> dict:
        d = self.service.provider.describe()
        missing = not self.service.provider.client.key
        return {"configured": d["configured"], "problem": d["problem"], "setup": "key" if missing else "", "host": d["base_url"]}

    def preflight(self, ticker: str) -> str | None:
        try:
            map_symbol(ticker)
        except MarketDataError as exc:
            return str(exc)
        ok, why = self.available()
        return None if ok else why

    def calls_for(self, ticker: str, start) -> int:
        return 1

    def test(self, symbol: str = "AAPL", kind: str | None = None) -> dict:
        """Two requests: the key and the plan's stock bars (a refusal is shown in Massive's words)."""
        ok, why = self.available()
        if not ok:
            return {"ok": False, "message": why}
        try:
            s, ac = map_symbol({"futures": "AAPL", "crypto": "BTC-USD"}.get(kind or "", symbol))
            r = self.service.provider.get_historical_bars(BarsRequest(s, ac, str((pd.Timestamp.now() - pd.Timedelta(days=10)).date()), str(pd.Timestamp.now().date()), limit=5))
        except MarketDataError as exc:
            return {"ok": False, "message": str(exc), "kind": exc.kind}
        n = len(r.frame)
        return {"ok": n > 0, "message": f"Massive answered: {n} daily bars of {s} (last {r.frame.index[-1].date() if n else 'none'}). Run the entitlement check on the Futures, Options or Crypto tab to see every dataset." if n else
                f"Massive answered but sent no bars for {s}: check the symbol and your plan's history depth", "bars": n}

    def __call__(self, tickers, start, known=None, progress=None) -> dict:
        out: dict = {}
        ok, why = self.available()
        if not ok:
            return {t: why for t in tickers}
        for i, t in enumerate(tickers):
            try:
                sym, ac = map_symbol(t)
            except MarketDataError as exc:
                out[t] = str(exc)
                continue
            try:
                self.service.sync_bars(sym, ac, str(start)[:10], adjusted=(ac == "stocks"), progress=(lambda x, t=t, i=i: progress(f"Massive {t} ({i + 1} of {len(tickers)}): {x}")) if progress else None)
            except MarketDataError as exc:
                out[t] = str(exc)
                if exc.kind in ("auth", "not_configured", "cancelled"):
                    out.update({r: out[t] for r in tickers[i + 1:]})
                    break
                continue
            bars = self.service.bars(sym, ac, str(start)[:10], adjusted=(ac == "stocks"))
            bars = bars[bars["close"] > 0]
            out[t] = bars[["close", "high", "low", "volume"]] if len(bars) else f"Massive returned no daily bars for {sym} since {str(start)[:10]}: check the symbol and your plan's history depth"
        return out
