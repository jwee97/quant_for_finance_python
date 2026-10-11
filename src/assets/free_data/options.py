"""Real option chains from free sources, in the file format the Options workspace reads.

``fetch_chain`` takes a US stock, ETF or index symbol from Cboe's delayed quotes and, when Cboe cannot be reached, from Yahoo Finance (through the ``yfinance`` package, when it is installed).
``fetch_deribit_chain`` takes the Bitcoin or Ether chain from Deribit. Each returns a table with the columns of a vendor file (``date, expiry, strike, right, bid, ask, underlying, rate, dividend,
volume, open_interest``) plus the vendor's own implied volatility, which the workspace ignores and recomputes from the mid price so that every chain is treated the same.

**Forward and carry.** A vendor chain says what each option costs but not what rate and dividend yield to price it with, and a wrong pair moves the implied volatilities of in-the-money options and
makes put-call parity look violated. The risk-free rate is therefore an assumption (``DEFAULT_RATE``), and the dividend yield of each expiry is the one that makes the forward implied by put-call
parity at the money equal to ``spot x exp((rate - dividend) x T)``: the market's own forward, expressed in the two numbers the pricing code takes. Deribit publishes each expiry's forward, so
there the same is read off directly (its "dividend" is minus the annual basis, usually negative).
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..options_analytics import DEFAULT_RATE
from .http import FetchError, Http
from .store import Download, utc_now

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
CBOE_INDEXES = {"SPX", "NDX", "RUT", "VIX", "XSP", "DJX", "OEX", "XND", "MRUT"}               # Cboe serves an index chain under an underscore
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
DERIBIT_CURRENCIES = ("BTC", "ETH")
SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
MAX_DTE = 1100
CARRY_RANGE = (-0.10, 0.25)                                                                   # an implied dividend yield outside this is a bad parity estimate, not a market
COLUMNS = ["date", "expiry", "strike", "right", "bid", "ask", "underlying", "rate", "dividend", "volume", "open_interest", "iv_vendor"]


def check_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper().lstrip("^$")
    if not SYMBOL.match(s):
        raise ValueError(f"'{symbol}' is not a ticker symbol (letters and digits, at most 10)")
    return s


# ------------------------------------------------------------------------------------------------------------------ the forward
def implied_carry(rows: pd.DataFrame, spot: float, quote_date: pd.Timestamp, rate: float, min_days: int = 7, band: float = 0.06, k: int = 7) -> tuple[pd.Series, dict]:
    """The dividend yield of each expiry implied by put-call parity, as ``(series indexed by expiry, info)``.

    For each expiry with at least ``min_days`` to run, take the strikes within ``band`` of the spot that have a live bid on both the call and the put, estimate the forward from each as
    ``K + (C - P) x exp(rate x T)``, keep the ``k`` with the narrowest quoted spreads and use the median. An expiry with too few such strikes (or too near) takes the yield of the nearest expiry
    that has one; with none at all the yield is zero and ``info['estimated']`` is 0. A yield outside ``CARRY_RANGE`` is clipped and counted in ``info['clipped']``."""
    found: dict[pd.Timestamp, float] = {}
    clipped = 0
    for expiry, g in rows.groupby("expiry"):
        days = (pd.Timestamp(expiry) - quote_date).days
        if days < min_days:
            continue
        T = days / 365.0
        calls, puts = g[g["right"] == "C"].set_index("strike"), g[g["right"] == "P"].set_index("strike")
        both = calls.index.intersection(puts.index)
        both = both[(np.abs(both / spot - 1.0) <= band)]
        if len(both) < 3:
            continue
        c, p = calls.loc[both], puts.loc[both]
        live = (c["bid"] > 0) & (p["bid"] > 0)
        if live.sum() < 3:
            continue
        c, p = c[live], p[live]
        spread = (c["ask"] - c["bid"]) + (p["ask"] - p["bid"])
        keep = spread.nsmallest(k).index
        mid_c, mid_p = (c.loc[keep, "bid"] + c.loc[keep, "ask"]) / 2.0, (p.loc[keep, "bid"] + p.loc[keep, "ask"]) / 2.0
        forward = float(np.median(keep.to_numpy(dtype=float) + (mid_c - mid_p).to_numpy() * math.exp(rate * T)))
        if not np.isfinite(forward) or forward <= 0:
            continue
        q = rate - math.log(forward / spot) / T
        if not CARRY_RANGE[0] <= q <= CARRY_RANGE[1]:
            clipped += 1
            q = min(max(q, CARRY_RANGE[0]), CARRY_RANGE[1])
        found[pd.Timestamp(expiry)] = q
    expiries = sorted(pd.Timestamp(e) for e in rows["expiry"].unique())
    out = {}
    for e in expiries:
        if e in found:
            out[e] = found[e]
        elif found:
            nearest = min(found, key=lambda f: abs((f - e).days))
            out[e] = found[nearest]
        else:
            out[e] = 0.0
    info = {"estimated": len(found), "expiries": len(expiries), "clipped": clipped,
            "median": float(np.median(list(found.values()))) if found else 0.0, "low": float(min(found.values())) if found else 0.0, "high": float(max(found.values())) if found else 0.0}
    return pd.Series(out, name="dividend"), info


# ------------------------------------------------------------------------------------------------------------------ one chain table
def build_chain(rows: pd.DataFrame, spot: float, quote_date, rate: float = DEFAULT_RATE, max_dte: int = 365, min_dte: int = 1) -> tuple[pd.DataFrame, dict]:
    """The standard chain from parsed quotes. ``rows`` has ``expiry, strike, right, bid, ask`` and optionally ``volume, open_interest, iv_vendor``. Rows with no offer, a crossed market, an
    expiry outside ``[min_dte, max_dte]`` days or a duplicate contract are dropped (the duplicate with the larger open interest stays); the counts come back in the second value."""
    quote_date = pd.Timestamp(quote_date).normalize()
    if not np.isfinite(spot) or spot <= 0:
        raise FetchError("the source gave no usable price for the underlying")
    df = rows.copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    df["right"] = df["right"].astype(str).str.upper().str[0]
    for col, default in (("volume", 0.0), ("open_interest", 0.0), ("iv_vendor", np.nan)):
        if col not in df:
            df[col] = default
    for col in ("strike", "bid", "ask", "volume", "open_interest", "iv_vendor"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_raw = len(df)
    dte = (df["expiry"] - quote_date).dt.days
    crossed = df["bid"] > df["ask"]
    no_offer = ~(df["ask"] > 0) | df["bid"].isna()
    out_of_range = (dte < min_dte) | (dte > max_dte)
    bad = crossed | no_offer | out_of_range | ~(df["strike"] > 0) | ~df["right"].isin(["C", "P"])
    df = df[~bad]
    before = len(df)
    df = df.sort_values("open_interest", ascending=False).drop_duplicates(["expiry", "strike", "right"]).sort_values(["expiry", "strike", "right"])
    if df.empty:
        raise FetchError("the source returned no option with a live offer in the requested range")
    q, info = implied_carry(df, float(spot), quote_date, rate)
    df["date"] = quote_date
    df["underlying"] = float(spot)
    df["rate"] = float(rate)
    df["dividend"] = df["expiry"].map(q).astype(float)
    df["volume"] = df["volume"].fillna(0.0)
    df["open_interest"] = df["open_interest"].fillna(0.0)
    df = df[COLUMNS].reset_index(drop=True)
    counts = {"raw": int(n_raw), "kept": int(len(df)), "no_offer_or_crossed": int((crossed | no_offer).sum()), "outside_range": int(out_of_range.sum()), "duplicates": int(before - len(df)), "carry": info}
    return df, counts


def _chain_notes(df: pd.DataFrame, counts: dict, rate: float, american: bool, delay: str) -> list[str]:
    carry = counts["carry"]
    notes = [f"{delay}. {len(df):,} contracts on {df['expiry'].nunique()} expiries were kept of {counts['raw']:,} received; contracts with no offer, a crossed market or an expiry out of range were dropped."]
    if carry["estimated"]:
        notes.append(f"The risk-free rate is an assumption ({rate:.2%}). Each expiry's dividend yield is the one implied by put-call parity at the money: median {carry['median']:.2%} "
                     f"(from {carry['low']:.2%} to {carry['high']:.2%}) over {carry['estimated']} of {carry['expiries']} expiries; the others take their nearest neighbour's.")
    else:
        notes.append(f"The risk-free rate ({rate:.2%}) and a zero dividend yield are assumptions: too few strikes had a live bid on both the call and the put to read a forward from parity.")
    if american:
        notes.append("Stock and ETF options can be exercised early; the workspace prices them as European (Black-Scholes), which slightly understates the value of the right to exercise early.")
    return notes


# ------------------------------------------------------------------------------------------------------------------ Cboe
def cboe_rows(payload: dict) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    """``(quotes, spot, quote date)`` from Cboe's delayed-quote document: ``data.options`` is a list of contracts named in the OCC way (root, ``YYMMDD``, ``C`` or ``P``, the strike times 1000 in eight digits)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("options"), list) or not data["options"]:
        raise FetchError("Cboe's answer has no option chain (the symbol may not have listed options)")
    try:
        spot = float(data.get("current_price") or data.get("close") or float("nan"))
    except (TypeError, ValueError):
        spot = float("nan")
    stamp = data.get("last_trade_time") or payload.get("timestamp")
    try:
        quote_date = pd.Timestamp(str(stamp)[:10])
    except (ValueError, TypeError):
        raise FetchError("Cboe's answer does not say when the quotes are from") from None
    recs = []
    for o in data["options"]:
        name = str(o.get("option") or "")
        if len(name) < 16 or name[-9] not in "CP":
            continue
        try:
            recs.append((datetime.strptime(name[-15:-9], "%y%m%d"), int(name[-8:]) / 1000.0, name[-9], o.get("bid"), o.get("ask"), o.get("volume"), o.get("open_interest"), o.get("iv")))
        except ValueError:
            continue
    if not recs:
        raise FetchError("Cboe's answer has no option contract named in the usual way (root, date, C or P, strike)")
    rows = pd.DataFrame(recs, columns=["expiry", "strike", "right", "bid", "ask", "volume", "open_interest", "iv_vendor"])
    rows["iv_vendor"] = rows["iv_vendor"].where(rows["iv_vendor"] > 0)
    return rows, spot, quote_date


def fetch_cboe_chain(http: Http, symbol: str, rate: float = DEFAULT_RATE, max_dte: int = 365) -> Download:
    symbol = check_symbol(symbol)
    code = f"_{symbol}" if symbol in CBOE_INDEXES else symbol
    url = CBOE_URL.format(symbol=code)
    try:
        payload = http.get_json(url)
    except FetchError as exc:
        if exc.status == 404:                                                              # Cboe lists every US option: a 404 means the symbol has none
            raise FetchError(f"Cboe has no option chain for {symbol} ({exc})", status=404) from None
        raise
    rows, spot, quote_date = cboe_rows(payload)
    df, counts = build_chain(rows, spot, quote_date, rate, max_dte)
    meta = {"source_id": "cboe", "source": "Cboe delayed quotes", "url": url, "as_of": str(quote_date.date()), "underlying": symbol, "spot": float(spot), "contracts": int(len(df)),
            "expiries": int(df["expiry"].nunique()), "american_style": symbol not in CBOE_INDEXES, "rate": float(rate), "carry": counts["carry"],
            "notes": _chain_notes(df, counts, rate, symbol not in CBOE_INDEXES, "Cboe's delayed quotes: about 15 minutes behind during the session, the last session's close otherwise")}
    return Download("options", symbol, "chain", frame=df, meta=meta)


# ------------------------------------------------------------------------------------------------------------------ Yahoo
def fetch_yahoo_chain(symbol: str, rate: float = DEFAULT_RATE, max_dte: int = 365, yf=None) -> Download:
    """The same chain from Yahoo Finance through ``yfinance`` (one request per expiry). ``yf`` is the module, for tests; without it the package is imported."""
    symbol = check_symbol(symbol)
    if yf is None:
        try:
            import yfinance as yf                                                      # noqa: PLC0415 - optional, and only this fallback needs it
        except Exception as exc:                                                       # noqa: BLE001 - not installed, or installed and unable to load
            raise FetchError(f"Yahoo Finance is not available here ({type(exc).__name__}: {exc})") from None
    try:
        ticker = yf.Ticker(symbol)
        expiries = list(ticker.options or ())
        closes = ticker.history(period="5d")["Close"].dropna()
    except Exception as exc:                                                           # noqa: BLE001 - yfinance raises many types for a refusal
        raise FetchError(f"Yahoo Finance did not answer for {symbol} ({type(exc).__name__}: {exc})") from None
    if not expiries or len(closes) == 0:
        raise FetchError(f"Yahoo Finance has no option chain for {symbol}")
    spot, last = float(closes.iloc[-1]), pd.Timestamp(closes.index[-1])
    quote_date = (last.tz_localize(None) if last.tzinfo is not None else last).normalize()
    frames, failed = [], 0
    for e in expiries:
        day = pd.Timestamp(e)
        if (day - quote_date).days > max_dte:
            break
        try:
            chain = ticker.option_chain(e)
        except Exception:                                                              # noqa: BLE001 - one expiry failing must not lose the rest
            failed += 1
            continue
        for right, table in (("C", chain.calls), ("P", chain.puts)):
            frames.append(pd.DataFrame({"expiry": day, "strike": table["strike"].to_numpy(), "right": right, "bid": table["bid"].to_numpy(), "ask": table["ask"].to_numpy(),
                                        "volume": table["volume"].to_numpy(), "open_interest": table["openInterest"].to_numpy(), "iv_vendor": table["impliedVolatility"].to_numpy()}))
    if not frames:
        raise FetchError(f"Yahoo Finance returned no usable expiry for {symbol}")
    rows = pd.concat(frames, ignore_index=True)
    rows["iv_vendor"] = rows["iv_vendor"].where(rows["iv_vendor"] > 0.01)
    df, counts = build_chain(rows, spot, quote_date, rate, max_dte)
    notes = _chain_notes(df, counts, rate, True, "Yahoo Finance's delayed quotes through the yfinance package")
    if failed:
        notes.append(f"{failed} expiries could not be read and are missing.")
    meta = {"source_id": "yahoo", "source": "Yahoo Finance", "url": f"https://finance.yahoo.com/quote/{symbol}/options", "as_of": str(quote_date.date()), "underlying": symbol, "spot": spot,
            "contracts": int(len(df)), "expiries": int(df["expiry"].nunique()), "american_style": True, "rate": float(rate), "carry": counts["carry"], "notes": notes}
    return Download("options", symbol, "chain", frame=df, meta=meta)


def fetch_chain(http: Http, symbol: str, rate: float = DEFAULT_RATE, max_dte: int = 365, yf=None) -> Download:
    """A real chain for a stock, ETF or index: Cboe first, Yahoo Finance when Cboe cannot give it. The error, when both fail, says why each did."""
    symbol = check_symbol(symbol)
    try:
        return fetch_cboe_chain(http, symbol, rate, max_dte)
    except FetchError as first:
        if first.status == 404:                                                            # no listed options: Yahoo would not know better, so do not ask it
            raise
        try:
            out = fetch_yahoo_chain(symbol, rate, max_dte, yf)
        except FetchError as second:
            raise FetchError(f"{first}; and {second}", status=first.status, blocked=first.blocked) from None
        out.meta["notes"] = [f"Cboe did not give the chain ({first}), so Yahoo Finance was used."] + list(out.meta["notes"])
        return out


# ------------------------------------------------------------------------------------------------------------------ Deribit
def deribit_rows(result: list, currency: str, max_dte: int = 365) -> tuple[pd.DataFrame, float, pd.Timestamp, dict]:
    """``(quotes in dollars, index price, quote date, forwards)`` from Deribit's book summaries. Deribit quotes an option in units of the coin, so every price is multiplied by the index price
    (``estimated_delivery_price``) to get dollars. ``forwards`` maps each expiry to the forward Deribit prices it with (``underlying_price``)."""
    recs, forwards, index, stamp = [], {}, [], []
    for o in result:
        parts = str(o.get("instrument_name") or "").split("-")
        if len(parts) != 4 or parts[3] not in ("C", "P"):
            continue
        try:
            expiry = pd.Timestamp(datetime.strptime(parts[1], "%d%b%y"))
            strike = float(parts[2].replace("d", "."))
        except ValueError:
            continue
        idx = o.get("estimated_delivery_price")
        if not idx:
            continue
        ask = o.get("ask_price")
        if not ask:
            continue
        bid = o.get("bid_price") or 0.0
        recs.append((expiry, strike, parts[3], float(bid) * float(idx), float(ask) * float(idx), o.get("volume"), o.get("open_interest"), o.get("mark_iv")))
        index.append(float(idx))
        stamp.append(o.get("creation_timestamp"))
        if o.get("underlying_price"):
            forwards[expiry] = float(o["underlying_price"])
    if not recs:
        raise FetchError(f"Deribit returned no {currency} option with an offer")
    rows = pd.DataFrame(recs, columns=["expiry", "strike", "right", "bid", "ask", "volume", "open_interest", "iv_vendor"])
    rows["iv_vendor"] = pd.to_numeric(rows["iv_vendor"], errors="coerce") / 100.0
    spot = float(np.median(index))
    ts = [s for s in stamp if s]
    quote_date = pd.Timestamp(datetime.fromtimestamp(float(np.median(ts)) / 1000.0, tz=timezone.utc)).tz_localize(None).normalize() if ts else pd.Timestamp(utc_now()).tz_localize(None).normalize()
    return rows, spot, quote_date, forwards


def fetch_deribit_chain(http: Http, currency: str, max_dte: int = 365) -> Download:
    currency = str(currency).upper()
    if currency not in DERIBIT_CURRENCIES:
        raise ValueError(f"Deribit options are available for {', '.join(DERIBIT_CURRENCIES)}")
    answer = http.get_json(DERIBIT_URL, {"currency": currency, "kind": "option"})
    result = answer.get("result") if isinstance(answer, dict) else None
    if not isinstance(result, list):
        raise FetchError("Deribit's answer has no option list")
    rows, spot, quote_date, forwards = deribit_rows(result, currency, max_dte)
    df, counts = build_chain(rows, spot, quote_date, 0.0, max_dte)
    # Deribit publishes the forward of each expiry: read the carry off it instead of off parity.
    carry = {}
    for e, f in forwards.items():
        days = (e - quote_date).days
        if days >= 1 and f > 0:
            carry[e] = -math.log(f / spot) / (days / 365.0)
    if carry:
        df["dividend"] = df["expiry"].map(carry).fillna(df["dividend"]).astype(float)
        counts["carry"] = {"estimated": len(carry), "expiries": int(df["expiry"].nunique()), "clipped": 0, "median": float(np.median(list(carry.values()))), "low": float(min(carry.values())),
                           "high": float(max(carry.values()))}
    notes = [f"Deribit's live quotes for {currency} options, converted from coin units to dollars with the index price {spot:,.2f}. {len(df):,} contracts on {df['expiry'].nunique()} expiries were kept of {counts['raw']:,}.",
             "The rate is zero and each expiry's 'dividend yield' is minus the annualised basis of Deribit's own forward for it (usually negative: the forward is above the index).",
             "Volume and open interest are in units of the coin. Deribit options are European and settle at 08:00 UTC."]
    meta = {"source_id": "deribit", "source": "Deribit", "url": DERIBIT_URL, "as_of": str(quote_date.date()), "underlying": currency, "spot": spot, "contracts": int(len(df)),
            "expiries": int(df["expiry"].nunique()), "american_style": False, "rate": 0.0, "carry": counts["carry"], "notes": notes}
    return Download("options", f"{currency}-Deribit", "chain", frame=df, meta=meta)
