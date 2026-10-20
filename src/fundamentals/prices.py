"""Prices and exchange rates for the companies in the database.

A fundamentals study needs two prices for every company: the price it traded at (unadjusted, to turn a share count into a market value) and the price adjusted for splits and dividends (to turn it
into a return). Neither comes from the statements provider, so they are fetched from a market-data source and kept in the same database, where the research reads them: the research never
calls a source.

**Symbols.** A company is stored under its provider's identity (OpenFilings: a market id and a ticker). Yahoo Finance writes the same company with a suffix for its exchange (``7203.T``,
``HSBA.L``, ``0700.HK``), so :func:`yahoo_symbol` builds the symbol from the market and the ticker. That is a rule of thumb, not a lookup: the first time a symbol returns prices it is remembered
as the company's ``price_symbol`` identifier, and a person can set a different one (:func:`set_price_symbol`) when the rule is wrong. A symbol that returns nothing is reported, never guessed at.

**Quote units.** London quotes in pence and Tel Aviv in agorot, not in pounds and shekels; a market value built from them would be a hundred times too large, so those prices are divided by 100
(:data:`QUOTE_DIVISOR`). Whatever else a source quotes in is not known here: the unit check in the Data Manager compares each market value with the company's reported revenue, so a mistake of that size
shows.

**What is not here.** A company that has been delisted has no history at Yahoo, so a study over Yahoo prices leaves out the companies that failed: the survivorship bias is reported by the research
context and cannot be removed by this module. Prices are only as good as the source, which does not publish its adjustment method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .errors import FundamentalsError
from .store import FundamentalsStore

MAX_BATCH = 40
OVERLAP_DAYS = 10                    # an update starts this many days before the last stored price, so a revised bar replaces what was stored
DEFAULT_START = "2005-01-01"

# OpenFilings market id -> Yahoo Finance suffix. "eu" filers trade on exchanges named by their country (below).
YAHOO_SUFFIX = {"us": "", "jp": ".T", "hk": ".HK", "tw": ".TW", "kr": ".KS", "cn": ".SS", "fr": ".PA", "uk": ".L", "br": ".SA", "mx": ".MX", "au": ".AX", "in": ".NS", "sg": ".SI",
                "sa": ".SR", "il": ".TA", "ca": ".TO", "th": ".BK", "my": ".KL", "id": ".JK", "ph": ".PS", "tr": ".IS", "ch": ".SW"}
COUNTRY_SUFFIX = {"DE": ".DE", "NL": ".AS", "ES": ".MC", "IT": ".MI", "BE": ".BR", "PT": ".LS", "AT": ".VI", "IE": ".IR", "FI": ".HE", "SE": ".ST", "DK": ".CO", "NO": ".OL", "PL": ".WA", "FR": ".PA",
                  "GB": ".L", "CH": ".SW", "LU": ".PA"}
QUOTE_DIVISOR = {".L": 100.0, ".TA": 100.0}


def yahoo_symbol(ticker: str, market: str | None, country: str | None = None) -> tuple[str, float]:
    """``(Yahoo symbol, divisor)`` for a company: its ticker with the suffix of its market, and the number its quoted prices are divided by to get the main currency unit."""
    ticker = str(ticker).strip().upper()
    market = (market or "us").lower()
    suffix = YAHOO_SUFFIX.get(market)
    if market == "eu":
        suffix = COUNTRY_SUFFIX.get((country or "").upper())
    if suffix is None:
        raise FundamentalsError(f"no Yahoo suffix is known for the market {market!r}" + (f" and country {country!r}" if market == "eu" else "") + "; set the company's price symbol by hand")
    if suffix == ".HK" and ticker.isdigit():
        ticker = ticker.zfill(4)
    if suffix == ".SS" and ticker.isdigit() and ticker.startswith(("0", "2", "3")):
        suffix = ".SZ"                                                         # Shenzhen codes start with 0, 2 or 3; Shanghai's with 6 or 9
    if suffix == ".KS" and ticker.isdigit():
        ticker = ticker.zfill(6)
    return ticker + suffix, QUOTE_DIVISOR.get(suffix, 1.0)


def yahoo_download(symbols: list[str], start: str) -> dict[str, pd.DataFrame | str]:
    """Daily ``close`` (unadjusted), ``adj_close`` (adjusted for splits and dividends) and ``volume`` per symbol from Yahoo Finance, or a short reason for a symbol that returned nothing."""
    import yfinance as yf

    try:
        raw = yf.download(list(symbols), start=start, auto_adjust=False, progress=False, threads=False, actions=False)
    except Exception as error:                                                  # network, rate limit, a change at the source
        return {s: f"download failed: {type(error).__name__}: {str(error)[:120]}" for s in symbols}
    out: dict[str, pd.DataFrame | str] = {}
    for s in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                level = next(i for i in range(raw.columns.nlevels) if raw.columns.get_level_values(i).isin(list(symbols)).any())
                frame = raw.xs(s, axis=1, level=level)
            else:
                frame = raw
            frame = frame.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
            if "close" not in frame:
                out[s] = "no data found"
                continue
            keep = frame[[c for c in ("close", "adj_close", "volume") if c in frame]].dropna(subset=["close"])
            out[s] = keep if len(keep) else "no data found (the symbol may be wrong, or the company delisted)"
        except (KeyError, StopIteration):
            out[s] = "no data found (the symbol may be wrong, or the company delisted)"
    return out


def price_symbol(store: FundamentalsStore, security: dict) -> tuple[str, float, str]:
    """``(symbol, divisor, how)``: the symbol saved for the company if there is one, else the one the market's rule gives."""
    ids = store.identifiers(int(security["security_id"]))
    saved = ids[(ids["provider"] == "yahoo") & (ids["id_type"] == "price_symbol")] if len(ids) else ids
    if len(saved):
        symbol = str(saved.iloc[0]["provider_id"])
        return symbol, _divisor_of(symbol), "saved"
    symbol, divisor = yahoo_symbol(security["ticker"], security.get("exchange"), security.get("country"))
    return symbol, divisor, "rule"


def _divisor_of(symbol: str) -> float:
    return QUOTE_DIVISOR.get("." + symbol.rsplit(".", 1)[-1], 1.0) if "." in symbol else 1.0


def set_price_symbol(store: FundamentalsStore, security_id: int, symbol: str) -> None:
    """Remember the symbol a source knows a company by (it replaces the one the market's rule would build)."""
    symbol = str(symbol).strip().upper()
    if not symbol or len(symbol) > 24 or not all(c.isalnum() or c in ".-^=" for c in symbol):
        raise FundamentalsError("a price symbol is letters, digits and . - ^ = only, up to 24 characters")
    with store.db.transaction() as conn:
        conn.execute("DELETE FROM security_identifiers WHERE security_id = ? AND provider = 'yahoo' AND id_type = 'price_symbol'", (int(security_id),))
        conn.execute("INSERT INTO security_identifiers (security_id, provider, id_type, provider_id) VALUES (?,?,?,?)", (int(security_id), "yahoo", "price_symbol", symbol))


def _last_dates(store: FundamentalsStore, ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    with store.db.transaction() as conn:
        rows = conn.execute(f"SELECT security_id, MAX(date) FROM prices_daily WHERE security_id IN ({','.join('?' * len(ids))}) GROUP BY security_id", [int(i) for i in ids]).fetchall()
    return {int(r[0]): str(r[1]) for r in rows}


def load_prices(store: FundamentalsStore, security_ids: list[int] | None = None, start: str = DEFAULT_START, fetch=yahoo_download, progress=None, stop=None, refresh: bool = False) -> dict:
    """Fetch daily prices for ``security_ids`` (all companies by default) and store them. A company that already has prices is updated from just before its last one unless ``refresh``.

    ``fetch(symbols, start)`` returns ``{symbol: frame | reason}`` (see :func:`yahoo_download`). Returns a report: how many companies got prices, how many rows were stored, and for every company that
    got none the reason, so nothing disappears silently."""
    say = progress or (lambda text: None)
    securities = store.securities()
    if security_ids is not None:
        securities = securities[securities["security_id"].isin([int(i) for i in security_ids])]
    if securities.empty:
        raise FundamentalsError("there are no companies to load prices for: add some first")
    last = {} if refresh else _last_dates(store, [int(i) for i in securities["security_id"]])
    jobs = []
    report: dict = {"companies": int(len(securities)), "loaded": 0, "rows": 0, "failed": [], "symbols": {}, "notes": [], "stopped": ""}
    for row in securities.to_dict("records"):
        try:
            symbol, divisor, how = price_symbol(store, row)
        except FundamentalsError as error:
            report["failed"].append({"security_id": int(row["security_id"]), "ticker": row["ticker"], "reason": str(error)})
            continue
        sid = int(row["security_id"])
        since = start
        if sid in last:
            since = max(start, str((pd.Timestamp(last[sid]) - pd.Timedelta(days=OVERLAP_DAYS)).date()))
        jobs.append((sid, row["ticker"], symbol, divisor, how, since))
    for lo in range(0, len(jobs), MAX_BATCH):
        if stop is not None and stop():
            report["stopped"] = "stopped on request"
            break
        batch = jobs[lo:lo + MAX_BATCH]
        say(f"prices for {lo + 1} to {lo + len(batch)} of {len(jobs)} companies")
        got = fetch([b[2] for b in batch], min(b[5] for b in batch))
        for sid, ticker, symbol, divisor, how, since in batch:
            answer = got.get(symbol)
            if not isinstance(answer, pd.DataFrame) or answer.empty:
                report["failed"].append({"security_id": sid, "ticker": ticker, "symbol": symbol, "reason": answer if isinstance(answer, str) else "no answer"})
                continue
            frame = answer.copy()
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
            frame = frame.loc[frame.index >= pd.Timestamp(since)]
            close = frame["close"].astype(float) / divisor
            adj = (frame["adj_close"].astype(float) / divisor) if "adj_close" in frame else close
            rows = pd.DataFrame({"security_id": sid, "date": frame.index, "adj_close": adj.to_numpy(), "close": close.to_numpy(),
                                 "volume": frame["volume"].astype(float).to_numpy() if "volume" in frame else np.nan})
            n = store.store_prices(rows)
            if n == 0:
                report["failed"].append({"security_id": sid, "ticker": ticker, "symbol": symbol, "reason": "no usable prices in the answer"})
                continue
            if how == "rule":
                set_price_symbol(store, sid, symbol)
            report["loaded"] += 1
            report["rows"] += n
            report["symbols"][str(sid)] = symbol
    if any(s.endswith(tuple(QUOTE_DIVISOR)) for s in report["symbols"].values()):
        report["notes"].append("Prices quoted in pence (London) or agorot (Tel Aviv) were divided by 100.")
    report["notes"].append("A company that has been delisted usually has no history at Yahoo, so a study over these prices leaves out the companies that failed: read the survivorship note of the research context.")
    return report


def load_fx(store: FundamentalsStore, start: str = DEFAULT_START, fetch=yahoo_download, progress=None) -> dict:
    """Exchange rates (units of a currency per US dollar) for every currency the companies trade or report in, from Yahoo's ``<CCY>USD=X`` pairs."""
    say = progress or (lambda text: None)
    with store.db.transaction() as conn:
        rows = conn.execute("SELECT DISTINCT currency FROM securities WHERE currency IS NOT NULL UNION SELECT DISTINCT reporting_currency FROM financial_periods WHERE reporting_currency IS NOT NULL").fetchall()
    currencies = sorted({str(r[0]).upper() for r in rows if r[0]} - {"USD"})
    report: dict = {"currencies": currencies, "loaded": [], "failed": {}, "rows": 0}
    if not currencies:
        return report
    say(f"exchange rates for {len(currencies)} currencies")
    got = fetch([f"{c}USD=X" for c in currencies], start)
    for c in currencies:
        answer = got.get(f"{c}USD=X")
        if not isinstance(answer, pd.DataFrame) or answer.empty:
            report["failed"][c] = answer if isinstance(answer, str) else "no answer"
            continue
        usd_per = answer["close"].astype(float)
        idx = pd.DatetimeIndex(answer.index).tz_localize(None).normalize()
        frame = pd.DataFrame({"date": idx, "base": "USD", "quote": c, "rate": 1.0 / usd_per.to_numpy()})
        report["rows"] += store.store_fx(frame, "yahoo")
        report["loaded"].append(c)
    return report


def import_csv(store: FundamentalsStore, path: str | Path, ticker_column: str = "ticker") -> dict:
    """Prices from a file with the columns ``ticker, date, close`` and optionally ``adj_close`` and ``volume`` (a ticker as stored in the database, e.g. ``us:AAPL`` or just ``AAPL``)."""
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    need = {ticker_column, "date", "close"}
    if not need <= set(frame.columns):
        raise FundamentalsError(f"the file needs the columns {sorted(need)} (adj_close and volume are optional)")
    securities = store.securities()
    by_key = {}
    for r in securities.to_dict("records"):
        by_key[str(r["ticker"]).upper()] = int(r["security_id"])
        by_key[f"{r.get('exchange') or ''}:{r['ticker']}".upper()] = int(r["security_id"])
    frame["security_id"] = frame[ticker_column].astype(str).str.strip().str.upper().map(by_key)
    unknown = sorted(set(frame.loc[frame["security_id"].isna(), ticker_column].astype(str)))
    frame = frame.dropna(subset=["security_id"])
    frame["security_id"] = frame["security_id"].astype(int)
    frame["adj_close"] = frame["adj_close"] if "adj_close" in frame else frame["close"]
    rows = store.store_prices(frame[["security_id", "date", "adj_close", "close"] + (["volume"] if "volume" in frame else [])])
    return {"rows": rows, "companies": int(frame["security_id"].nunique()), "unknown_tickers": unknown[:50], "unknown_count": len(unknown)}


__all__ = ["yahoo_symbol", "yahoo_download", "price_symbol", "set_price_symbol", "load_prices", "load_fx", "import_csv", "YAHOO_SUFFIX", "QUOTE_DIVISOR", "MAX_BATCH"]
