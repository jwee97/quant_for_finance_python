"""Historical exchange rates from the platform's market-data infrastructure (the ticker store and its downloader), as a point-in-time ``FxTable``.

A rate series is the Yahoo Finance pair ``<local><base>=X`` (for a USD base and a JPY stock, ``JPYUSD=X``): the price of one unit of the local currency in the base currency, which is the platform's quotation
(``alloc_fx``: base per local). It is fetched, cached and versioned exactly like a ticker (the store records the provider, the time of download and the first and last date), so a study can say which rates produced
which number. Each daily close is taken to be available by the end of its own date plus ``available_lag_days`` (0 by default: an assumption about the provider's publication time, stated in the report).
A missing or failed series is an error that names the symbol and the reason; nothing is substituted.
"""

from __future__ import annotations

import pandas as pd

from .alloc_fx import FxError, FxTable


def fx_symbol(ccy: str, base: str) -> str:
    return f"{ccy.upper()}{base.upper()}=X"


def load_table(store, base: str, currencies, start: str, *, available_lag_days: int = 0, max_stale_days: int = 5, refresh: bool = False) -> tuple[FxTable, dict]:
    """``(table, report)`` for the foreign currencies named, from the store. ``report`` has, per currency, the symbol, the provider and download time the store recorded, the span and size of the series and
    its largest single-day move (a sanity check for a data error)."""
    base = base.upper()
    foreign = sorted({c.upper() for c in currencies if c.upper() != base})
    table = FxTable(base, max_stale_days)
    if not foreign:
        return table, {"base": base, "series": {}, "assumptions": _assumptions(available_lag_days)}
    symbols = {c: fx_symbol(c, base) for c in foreign}
    frames, errors = store.get(list(symbols.values()), str(pd.Timestamp(start).date()), refresh=refresh)
    missing = {c: errors.get(sym, "no data found") for c, sym in symbols.items() if sym not in frames}
    if missing:
        raise FxError("no exchange-rate history could be loaded for " + "; ".join(f"{c} ({symbols[c]}): {why}" for c, why in missing.items())
                      + f". The portfolio cannot be valued in {base} without it, and a rate is never guessed or taken as 1")
    series = {}
    for c, sym in symbols.items():
        f = frames[sym]
        col = "close" if "close" in f.columns else f.columns[0]
        r = pd.Series(f[col].astype(float).to_numpy(), index=pd.DatetimeIndex(f.index)).dropna()
        r = r[r > 0]
        table.set_series(c, r, pd.Series(r.index + pd.Timedelta(days=int(available_lag_days)), index=r.index))
        meta = store._meta(sym) or {}
        series[c] = {"symbol": sym, "provider": meta.get("provider", getattr(store, "provider", "unknown")), "fetched_at": meta.get("fetched_at"), "first": str(r.index[0].date()), "last": str(r.index[-1].date()),
                     "observations": int(len(r)), "largest_daily_move": float(r.pct_change().abs().max()) if len(r) > 1 else None, "warnings": list(meta.get("warnings") or [])}
    return table, {"base": base, "series": series, "assumptions": _assumptions(available_lag_days)}


def _assumptions(lag: int) -> list[str]:
    return [f"each daily close is taken as available {'by the end of its own date' if lag == 0 else f'{lag} day(s) after its date'} (the provider's publication time is not recorded)",
            "forward points and financing rates are not available to the platform: a hedge has no carry in this model and is not exchange-accurate",
            "rates are the provider's daily close for the pair, not an executable quote: conversion costs are an explicit assumption (fx.cost_bps)"]
