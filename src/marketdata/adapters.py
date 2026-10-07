"""Adapters from the repository's existing data structures to normalised market events.

Each function takes a familiar object (a wide price frame, a long futures contract table, a synthetic FX market, a yield-curve panel, a synthetic option market, a limit-order-book
simulation) and returns canonical events with EXPLICIT ``available_at``: the time of day a value is stamped (``close_time``) and the delay before it can be used (``lag``) are
arguments, never silent defaults of zero. Daily data therefore reach the engine at ``date + close_time + lag``, and a strategy deciding at the close sees yesterday's settlement but not
today's unless you say it may.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import normalise_events


def _stamp(index, close_time: str) -> pd.DatetimeIndex:
    h, m = (int(x) for x in close_time.split(":"))
    return pd.DatetimeIndex(pd.to_datetime(index)).normalize() + pd.Timedelta(hours=h, minutes=m)


def concat_events(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and len(f)]
    return normalise_events(pd.concat(parts, ignore_index=True), lag="0s") if parts else normalise_events(pd.DataFrame(columns=["timestamp", "instrument_id", "event_type"]), lag="0s")


def events_from_prices(prices: pd.DataFrame, event_type: str = "bar", close_time: str = "16:00", lag: str = "0s", spread_bps: float | None = None, volume: pd.DataFrame | None = None,
                       mapping: dict | None = None, source: str = "prices") -> pd.DataFrame:
    """A wide price frame (dates x instruments) as one event per instrument per date. ``event_type`` is ``bar`` (a close), ``settlement`` or ``mark``. With ``spread_bps`` a synthetic
    quote is emitted at the same time with that full spread around the price (labelled ``source='synthetic_spread'``) so that the engine fills at the bid and the ask."""
    stamps = _stamp(prices.index, close_time)
    parts = []
    for col in prices.columns:
        iid = (mapping or {}).get(col, col)
        s = prices[col].to_numpy(dtype=float)
        ok = np.isfinite(s)
        base = pd.DataFrame({"timestamp": stamps[ok], "instrument_id": iid, "event_type": event_type, "source": source})
        field = {"bar": "close", "settlement": "settlement", "mark": "mark_price"}[event_type]
        base[field] = s[ok]
        if volume is not None and col in volume.columns:
            base["volume"] = volume[col].to_numpy(dtype=float)[ok]
        parts.append(base)
        if spread_bps is not None:
            half = s[ok] * spread_bps * 1e-4 / 2.0
            parts.append(pd.DataFrame({"timestamp": stamps[ok], "instrument_id": iid, "event_type": "quote", "bid": s[ok] - half, "ask": s[ok] + half, "source": "synthetic_spread"}))
    df = pd.concat(parts, ignore_index=True)
    return normalise_events(df, lag=lag)


def events_from_futures_table(long: pd.DataFrame, root: str, close_time: str = "16:00", lag: str = "0s", spread_bps: float | None = None) -> pd.DataFrame:
    """A long contract table (``date, contract, expiry, price``) as settlement events for contract ids ``{root}{contract}`` (``CLF18``)."""
    df = long.copy()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="contract", values="price")
    wide.columns = [f"{root}{c}" for c in wide.columns]
    return events_from_prices(wide, "settlement", close_time, lag, spread_bps, source="futures_table")


def events_from_curve_panel(panel: pd.DataFrame, curve_id: str, close_time: str = "16:00", lag: str = "0s", source: str = "curve_panel") -> pd.DataFrame:
    """A yield-curve panel (dates x maturities in years, rates as decimals) as one curve event per date."""
    stamps = _stamp(panel.index, close_time)
    maturities = [float(c) for c in panel.columns]
    values = [{m: float(v) for m, v in zip(maturities, row)} for row in panel.to_numpy(dtype=float)]
    df = pd.DataFrame({"timestamp": stamps, "instrument_id": curve_id, "event_type": "curve", "curve_values": values, "source": source})
    return normalise_events(df, lag=lag)


def events_from_fx_market(market: dict, quote: str = "USD", close_time: str = "16:00", lag: str = "0s", spread_bps: float = 1.0, mapping: dict | None = None) -> pd.DataFrame:
    """``assets.fx.synthetic_fx_market`` output as spot quotes for pairs ``{currency}{quote}`` and the interest rates as ``reference`` events ``RATE-{currency}`` (value = annual rate)."""
    spot = market["spot"].copy()
    spot.columns = [(mapping or {}).get(c, f"{c}{quote}") for c in spot.columns]
    parts = [events_from_prices(spot, "bar", close_time, lag, spread_bps, source="fx_market")]
    rates = market["rates"]
    for col in rates.columns:
        df = pd.DataFrame({"timestamp": _stamp(rates.index, close_time), "instrument_id": f"RATE-{col}", "event_type": "reference", "value": rates[col].to_numpy(dtype=float), "source": "fx_market"})
        parts.append(normalise_events(df, lag=lag))
    if "usd_rate" in market:
        u = market["usd_rate"]
        parts.append(normalise_events(pd.DataFrame({"timestamp": _stamp(u.index, close_time), "instrument_id": f"RATE-{quote}", "event_type": "reference", "value": u.to_numpy(dtype=float),
                                                    "source": "fx_market"}), lag=lag))
    return concat_events(*parts)


def events_from_funding(funding: pd.DataFrame, event_type: str = "funding", lag: str = "0s", source: str = "funding") -> pd.DataFrame:
    """A frame of funding rates (timestamps x perpetual ids, fractions per interval) as funding events stamped at the payment time."""
    parts = []
    for col in funding.columns:
        s = funding[col].dropna()
        parts.append(pd.DataFrame({"timestamp": s.index, "instrument_id": col, "event_type": event_type, "funding": s.to_numpy(dtype=float), "source": source}))
    return normalise_events(pd.concat(parts, ignore_index=True), lag=lag)


def events_from_option_chains(market, underlying_id: str, close_time: str = "16:00", lag: str = "0s", underlying_kind: str = "spot", multiplier: float = 100.0, currency: str = "USD",
                              every: int = 1):
    """A ``derivatives.synthetic`` market as option quote events plus the :class:`~src.instruments.options.Option` contracts they refer to.

    Returns ``(events, options, underlying_events)``: quote events for every listed contract each ``every``-th day, the contracts (European, physically settled for a spot underlying),
    and bar events for the underlying. Each chain's reference ``rate`` and ``dividend`` ride in ``reference_values`` of the underlying's events."""
    from ..instruments.options import make_option

    dates = list(market.dates())[::every]
    contracts: dict[str, object] = {}
    rows = []
    for d in dates:
        chain = market.chain(d)
        ts = pd.Timestamp(d).normalize() + pd.Timedelta(hours=int(close_time[:2]), minutes=int(close_time[3:]))
        for rec in chain.itertuples(index=False):
            right = "call" if rec.right == "C" else "put"
            opt = make_option(underlying_id, pd.Timestamp(rec.expiry).normalize() + pd.Timedelta(hours=int(close_time[:2]), minutes=int(close_time[3:])), right, float(rec.strike),
                              multiplier=multiplier, currency=currency, underlying_kind=underlying_kind)
            contracts.setdefault(opt.instrument_id, opt)
            rows.append((ts, opt.instrument_id, float(rec.bid), float(rec.ask), float(getattr(rec, "volume", np.nan)), float(getattr(rec, "open_interest", np.nan))))
    q = pd.DataFrame(rows, columns=["timestamp", "instrument_id", "bid", "ask", "volume", "open_interest"])
    q["event_type"], q["source"] = "quote", "option_chains"
    events = normalise_events(q, lag=lag)
    spot = market.underlying["spot"].loc[dates]
    u = events_from_prices(spot.to_frame(underlying_id), "bar", close_time, lag, 1.0, source="option_chains")
    return events, list(contracts.values()), u


def events_from_lob(book: pd.DataFrame, instrument_id: str, start="2024-01-02 14:30:00", step: str = "1s", lag: str = "0s") -> pd.DataFrame:
    """A ``microstructure.lob.simulate_lob`` frame (``bid, ask, bid_size, ask_size``) as quote events one ``step`` apart."""
    ts = pd.Timestamp(start) + pd.to_timedelta(np.arange(len(book)) * pd.Timedelta(step).value, unit="ns")
    df = pd.DataFrame({"timestamp": ts, "instrument_id": instrument_id, "event_type": "quote", "bid": book["bid"].to_numpy(dtype=float), "ask": book["ask"].to_numpy(dtype=float),
                       "bid_size": book["bid_size"].to_numpy(dtype=float), "ask_size": book["ask_size"].to_numpy(dtype=float), "source": "lob_simulation"})
    return normalise_events(df, lag=lag)


def events_from_bundle(bundle, close_time: str = "16:00", lag: str = "0s", spread_bps: float | None = None) -> pd.DataFrame:
    """A framework :class:`MarketBundle`'s prices as bar events (one instrument per asset)."""
    return events_from_prices(bundle.prices, "bar", close_time, lag, spread_bps, source="bundle")
