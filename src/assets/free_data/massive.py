"""What the Futures, Crypto and Options tabs download from Massive, as :class:`~assets.free_data.store.Download` values the store can save. Everything goes through ``marketdata.MarketDataService``: one key,
one client, one rate limiter, the normalised store beside the files (so a repeat download asks only for the newest bars) and the provider's own provenance in every file's record.

* **Futures** (:func:`futures_curve`): the contract-level history of a product (``CL``): each contract's daily session bars with the exchange settlement price, joined into the curve table the Futures
  tab reads (``date, contract, expiry, price, open_interest, volume``). Open interest is empty (no Massive futures endpoint carries it). The contract list is Massive's; expiry is its ``last_trade_date``.
* **Crypto** (:func:`coin_history`): a coin's daily closes against the US dollar (``X:BTCUSD``), saved as the coin's SPOT history. It is not a futures or perpetual price; Deribit and the exchanges stay
  the sources for those.
* **Options** (:func:`chain`, :func:`historical_chain`, :func:`underlying_prices`): the provider's current chain snapshot (observed bid and ask; provider implied volatility where it sends one), the chain of a
  past date rebuilt from contract bars (daily closes used as the price, labelled so; no bid or ask exists), and the underlying's daily prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...marketdata.models import BarsRequest, MarketDataError
from ...marketdata.service import MarketDataService
from .options import DEFAULT_RATE, build_chain
from .store import Download

INDEX_UNDERLYINGS = {"SPX": "I:SPX", "VIX": "I:VIX", "NDX": "I:NDX", "RUT": "I:RUT", "DJI": "I:DJI"}      # Massive's index tickers (the Indices family, not verified for every plan)
MIN_COIN_DAYS = 250


def _today() -> pd.Timestamp:
    return pd.Timestamp.now().normalize()


def futures_curve(svc: MarketDataService, root: str, years: float = 1.0, progress=None, cancel=None) -> Download:
    start = str((_today() - pd.Timedelta(days=int(round(years * 365.25)))).date())
    r = svc.sync_futures_chain(root, start, progress=progress, cancel=cancel)
    long = svc.futures_curve_history(root, start)
    if long.empty:
        why = "; ".join(r.get("warnings") or []) or f"Massive has no {root} contract bars since {start}"
        raise MarketDataError(f"no {root} curve could be built: {why}", "not_found")
    frame = long[["date", "contract", "expiry", "price", "open_interest", "volume"]]
    info: dict = {}
    try:                                                                       # the product's name, venue and sector (one request); the contract multiplier is not in what Massive sends
        rows = svc.provider.client.get("/futures/v1/products", {"product_code": root, "limit": 1}, cancel=cancel).get("results") or []
        info = rows[0] if rows else {}
    except MarketDataError as exc:
        if exc.kind in ("auth", "cancelled", "not_configured"):
            raise
    last = long["date"].max()
    settled = float((long["price_source"] == "settlement").mean())
    notes = [f"Contract-level daily prices of {root} futures from Massive: {long['contract'].nunique()} contracts, {long['date'].nunique()} sessions, {long['date'].min().date()} to {last.date()}.",
             f"The price is the exchange settlement price of the session where Massive sends one ({settled:.0%} of rows), else the session close.",
             "Open interest is not available from Massive's futures endpoints: anything that needs it (open-interest roll, liquidity by open interest) is unavailable for this download.",
             "Expiry is Massive's last_trade_date. Each contract's bars are stored incrementally; an expired contract is archived and not asked for again.",
             "The contract multiplier and the margins are not in what Massive sends: the platform's own terms are used for roots it knows, otherwise dollar figures are in price units."] + list(r.get("warnings") or [])
    meta = {"source_id": "massive", "source": "Massive", "url": "https://api.massive.com/futures/v1/aggs", "as_of": str(last.date()), "first_date": str(long["date"].min().date()), "root": root, "root_name": str(info.get("name") or root), "exchange": str(info.get("trading_venue") or ""), "asset_class": str(info.get("sector") or info.get("asset_class") or "unknown"),
            "contracts": int(long["contract"].nunique()), "notes": notes}
    return Download("futures", root, "curve", frame=frame, meta=meta)


def coin_history(svc: MarketDataService, coin: str, days: int = 400, progress=None, cancel=None) -> Download:
    coin = str(coin).strip().upper()
    if not MIN_COIN_DAYS <= int(days) <= 1500:
        raise ValueError(f"days must be between {MIN_COIN_DAYS} and 1500 (the Crypto tab needs at least {MIN_COIN_DAYS} daily prices)")
    sym = f"X:{coin}USD"
    start = str((_today() - pd.Timedelta(days=int(days))).date())
    svc.sync_bars(sym, "crypto", start, adjusted=False, progress=progress, cancel=cancel)
    daily = svc.bars(sym, "crypto", start, adjusted=False)
    daily = daily[daily["close"] > 0].tail(int(days))
    if len(daily) < MIN_COIN_DAYS:
        raise MarketDataError(f"Massive has only {len(daily)} daily bars of {sym}; the Crypto tab needs at least {MIN_COIN_DAYS}", "not_found")
    first, last = daily.index[0], daily.index[-1]
    frame = pd.DataFrame({"time": daily.index, "spot": daily["close"].to_numpy(dtype="float64")})
    notes = [f"Daily closes of {sym} from Massive (UTC days): {len(frame):,} days, {first.date()} to {last.date()}. This is a SPOT price, not a futures or perpetual price.",
             "There is no perpetual price or funding rate in it: the basis and funding views have nothing from this download (Deribit and the exchanges supply derivatives data)."]
    meta = {"source_id": "massive", "source": "Massive (spot)", "url": "https://api.massive.com/v2/aggs", "as_of": str(last.date()), "coin": coin, "sources": {"spot": "Massive (spot)"}, "first_date": str(first.date()), "notes": notes}
    return Download("crypto", coin, "history", frame=frame, meta=meta)


def underlying_ticker(ticker: str) -> tuple[str, str]:
    t = str(ticker).strip().upper()
    return (INDEX_UNDERLYINGS[t], "indices") if t in INDEX_UNDERLYINGS else (t, "stocks")


def underlying_prices(svc: MarketDataService, ticker: str, years: float = 3.0, progress=None, cancel=None) -> Download:
    sym, ac = underlying_ticker(ticker)
    start = str((_today() - pd.Timedelta(days=int(round(years * 365.25)))).date())
    svc.sync_bars(sym, ac, start, adjusted=True, progress=progress, cancel=cancel)
    daily = svc.bars(sym, ac, start, adjusted=True)
    daily = daily[daily["close"] > 0]
    if daily.empty:
        raise MarketDataError(f"Massive returned no daily prices for {sym}", "not_found")
    first, last = daily.index[0], daily.index[-1]
    frame = daily.reset_index()[["date", "close", "volume"]]
    notes = [f"Massive's daily prices of {sym} ({len(frame):,} days, {first.date()} to {last.date()}), adjusted for splits as Massive defines it (dividends are not in the price): used for the realised volatility beside the implied one."]
    name = str(ticker).strip().upper()
    meta = {"source_id": "massive", "source": "Massive", "url": "https://api.massive.com/v2/aggs", "as_of": str(last.date()), "first_date": str(first.date()), "underlying": name, "bars": int(len(frame)), "notes": notes}
    return Download("options", f"{name}_prices", "prices", frame=frame, meta=meta)


def chain(svc: MarketDataService, symbol: str, rate: float = DEFAULT_RATE, max_dte: int = 365, progress=None, cancel=None) -> Download:
    """Massive's current chain snapshot as the platform's chain. Bid, ask, volume and open interest are the provider's; its implied volatility is kept where sent (``iv_vendor``)."""
    symbol = symbol.strip().upper()
    today = _today()
    t = svc.provider.get_option_chain(symbol, expiration_to=str((today + pd.Timedelta(days=max_dte)).date()), cancel=cancel)
    f = t.frame
    if f.empty:
        raise MarketDataError(f"Massive returned no option contracts for {symbol}", "not_found")
    spot = float(pd.to_numeric(f["underlying"], errors="coerce").dropna().median()) if f["underlying"].notna().any() else float("nan")
    if not np.isfinite(spot):
        bars = svc.provider.get_historical_bars(BarsRequest(symbol, "stocks", str((today - pd.Timedelta(days=7)).date()), str(today.date()), limit=10)).frame
        spot = float(bars["close"].iloc[-1])
    rows = f.rename(columns={"iv": "iv_vendor"})[["expiry", "strike", "right", "bid", "ask", "volume", "open_interest", "iv_vendor"]]
    df, counts = build_chain(rows, spot, today, rate, max_dte)
    from .options import _chain_notes
    notes = _chain_notes(df, counts, rate, True, "Massive's chain snapshot (15 minutes delayed on Starter and Developer, real time on Advanced)")
    n_iv = int(f["iv"].notna().sum())
    compare = _iv_agreement(f, df, spot, today, rate)
    notes.append(f"Massive sent its own implied volatility for {n_iv:,} of {len(f):,} contracts and Greeks where available; contracts without them get a volatility calculated here from the mid, labelled calculated. "
                 "Provider and calculated values are never mixed without the label." + (f" Where both exist they agree to a median {compare['median_pts']:.2f} volatility points (90th percentile {compare['p90_pts']:.2f}, {compare['n']} contracts)." if compare else ""))
    notes += list(t.warnings)
    meta = {"source_id": "massive", "source": "Massive", "url": f"https://api.massive.com/v3/snapshot/options/{symbol}", "as_of": str(today.date()), "underlying": symbol, "spot": spot, "contracts": int(len(df)),
            "expiries": int(df["expiry"].nunique()), "american_style": True, "rate": float(rate), "carry": counts["carry"], "notes": notes, "iv_provider_vs_calculated": compare}
    return Download("options", symbol, "chain", frame=df, meta=meta)


def _iv_agreement(snap: pd.DataFrame, chain: pd.DataFrame, spot: float, today: pd.Timestamp, rate: float) -> dict | None:
    """How far Massive's own implied volatility is from the one calculated here from the same mid (Black-Scholes, the chain's implied dividend yield), over the contracts that have both."""
    from ...derivatives.iv import implied_vol

    key = ["expiry", "strike", "right"]
    prov = snap.dropna(subset=["iv"]).drop_duplicates(key)[key + ["iv"]]
    m = chain.merge(prov, on=key, how="inner")
    if m.empty:
        return None
    T = ((pd.to_datetime(m["expiry"]) - today).dt.days / 365.0).to_numpy(float)
    mid = (0.5 * (m["bid"] + m["ask"])).to_numpy(float)
    calc = np.full(len(m), np.nan)
    for right, call in (("C", True), ("P", False)):
        sel = (m["right"] == right).to_numpy()
        if sel.any():
            calc[sel] = implied_vol(mid[sel], np.full(sel.sum(), spot), m["strike"].to_numpy(float)[sel], T[sel], rate, m["dividend"].to_numpy(float)[sel], call)
    diff = np.abs(calc - m["iv"].to_numpy(float)) * 100.0
    diff = diff[np.isfinite(diff)]
    if not len(diff):
        return None
    return {"n": int(len(diff)), "median_pts": float(np.median(diff)), "p90_pts": float(np.percentile(diff, 90))}


def historical_chain(svc: MarketDataService, symbol: str, as_of: str, rate: float = DEFAULT_RATE, max_dte: int = 120, max_contracts: int = 80, progress=None, cancel=None) -> Download:
    """A past date's chain rebuilt from contract bars (see ``MarketDataService.reconstruct_option_chain``). The daily close stands in for bid and ask (``bid = ask = close``): this is declared in the
    notes and in the file's record, and it is a last-trade price, not a quote. Contracts that did not trade are absent."""
    symbol = symbol.strip().upper()
    t = svc.reconstruct_option_chain(symbol, as_of, max_dte=max_dte, max_contracts=max_contracts, rate=rate, progress=progress, cancel=cancel)
    f = t.frame
    if f.empty:
        raise MarketDataError(f"no {symbol} option traded in the window on {as_of}: " + "; ".join(t.warnings), "not_found")
    spot = float(f["underlying"].iloc[0])
    rows = pd.DataFrame({"expiry": f["expiry"], "strike": f["strike"], "right": f["right"], "bid": f["mid"], "ask": f["mid"], "volume": f["volume"], "open_interest": 0.0})
    df, counts = build_chain(rows, spot, pd.Timestamp(as_of), rate, max_dte + 5)
    from .options import _chain_notes
    notes = [f"A REBUILT chain of {symbol} for {as_of}: Massive's contract list for that date and each contract's daily bar. The price of each contract is its daily CLOSE (a last trade), used as both bid and ask: "
             "there are no quotes and no spreads, so spread-based filters and costs are not observed here."] + _chain_notes(df, counts, rate, True, "Massive contract bars") + list(t.warnings)
    meta = {"source_id": "massive", "source": "Massive (rebuilt from daily closes)", "url": "https://api.massive.com/v3/reference/options/contracts", "as_of": str(pd.Timestamp(as_of).date()), "underlying": symbol, "spot": spot,
            "contracts": int(len(df)), "expiries": int(df["expiry"].nunique()), "american_style": True, "rate": float(rate), "carry": counts["carry"], "price_basis": "daily_close", "notes": notes}
    return Download("options", f"{symbol}_{pd.Timestamp(as_of).strftime('%Y%m%d')}", "chain", frame=df, meta=meta)
