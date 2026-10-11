"""Real futures curves from free sources, in the file format the Futures workspace reads (``date, contract, expiry, price`` and optionally ``open_interest``, ``volume``, ``spot``).

**Exchange-traded futures (Yahoo Finance).** Yahoo publishes a daily price history for each individual contract (``CLZ26.NYM`` is December 2026 WTI crude), as long as the contract has not expired.
For a root the fetcher builds the next ``n`` contract symbols from the root's delivery-month cycle, asks for each one's history and stacks them into one long table: one row per contract per day, so
the curve of any past date in the window can be drawn and the roll between contracts followed. Yahoo drops a contract when it expires, so a download is merged with the file from the last one
(``merge_history``): the rows of contracts Yahoo no longer serves stay, and the history grows past the window of any single request. What the feed does not have: open interest (the workspace's
roll-by-open-interest read-outs are then unavailable), and anything official: these are Yahoo's delayed prices, not exchange settlements, and the last bar of a session in progress is not a settlement.

**Expiry dates** are not in the feed. They are computed from each exchange's published rule (``last_trade_date``) without a holiday calendar, so a date can be a day early or late; the curve only
uses them to measure the time between contracts, where a day is a small error.

**Bitcoin and Ether (Deribit).** Deribit lists dated futures as well as the perpetual and publishes the whole strip at once, with open interest. One download is one day's curve; the file adds a day
with every download, so the history of the curve is the history of your downloads.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .http import FetchError, Http
from .store import Download

MONTH_CODES = "FGHJKMNQUVXZ"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
ROOT = re.compile(r"^[A-Z0-9]{1,4}$")


@dataclass(frozen=True)
class FreeRoot:
    """A futures root that can be downloaded: Yahoo's exchange suffix, the delivery months that are listed, the rule for the last trading day, and the terms the workspace needs for dollar figures."""

    root: str
    name: str
    exchange: str
    suffix: str
    asset_class: str
    multiplier: float
    tick_size: float
    months: tuple
    rule: str

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.multiplier

    def terms(self) -> dict:
        """What a record keeps of the root: its long name as ``root_name`` (``name`` is the file's), exchange, class, multiplier and tick size."""
        d = asdict(self)
        d.pop("months"), d.pop("rule"), d.pop("suffix")
        d["root_name"] = d.pop("name")
        return d


def _roots(*rows: FreeRoot) -> dict[str, FreeRoot]:
    return {r.root: r for r in rows}


ALL = tuple(range(1, 13))
QUARTERLY = (3, 6, 9, 12)
FREE_ROOTS = _roots(
    FreeRoot("CL", "WTI crude oil", "NYMEX", ".NYM", "commodity", 1000.0, 0.01, ALL, "cl"),
    FreeRoot("NG", "Henry Hub natural gas", "NYMEX", ".NYM", "commodity", 10000.0, 0.001, ALL, "ng"),
    FreeRoot("GC", "Gold", "COMEX", ".CMX", "metal", 100.0, 0.10, (2, 4, 6, 8, 10, 12), "third_last_bday"),
    FreeRoot("SI", "Silver", "COMEX", ".CMX", "metal", 5000.0, 0.005, (1, 3, 5, 7, 9, 12), "third_last_bday"),
    FreeRoot("HG", "Copper", "COMEX", ".CMX", "commodity", 25000.0, 0.0005, ALL, "third_last_bday"),
    FreeRoot("ES", "E-mini S&P 500", "CME", ".CME", "equity_index", 50.0, 0.25, QUARTERLY, "third_friday"),
    FreeRoot("NQ", "E-mini Nasdaq-100", "CME", ".CME", "equity_index", 20.0, 0.25, QUARTERLY, "third_friday"),
    FreeRoot("ZN", "10-year Treasury note", "CBOT", ".CBT", "rates", 1000.0, 0.015625, QUARTERLY, "treasury"),
    FreeRoot("ZB", "Treasury bond", "CBOT", ".CBT", "rates", 1000.0, 0.03125, QUARTERLY, "treasury"),
    FreeRoot("6E", "Euro FX", "CME", ".CME", "fx", 125000.0, 0.00005, QUARTERLY, "euro_fx"),
    FreeRoot("ZC", "Corn", "CBOT", ".CBT", "agricultural", 50.0, 0.25, (3, 5, 7, 9, 12), "grain"),
    FreeRoot("ZS", "Soybeans", "CBOT", ".CBT", "agricultural", 50.0, 0.25, (1, 3, 5, 7, 8, 9, 11), "grain"),
    FreeRoot("ZW", "Chicago wheat", "CBOT", ".CBT", "agricultural", 50.0, 0.25, (3, 5, 7, 9, 12), "grain"),
)
CRYPTO_ROOTS = {"BTC": "Bitcoin", "ETH": "Ether"}


# ------------------------------------------------------------------------------------------------------------------ expiry rules
def _last_bday(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year, month, 1) + pd.offsets.BMonthEnd(0)


def _third(weekday: int, year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(weekday - first.weekday()) % 7 + 14)


def last_trade_date(rule: str, year: int, month: int) -> pd.Timestamp:
    """The last trading day of the contract that delivers in ``year``-``month``, by the exchange's rule and without holidays (so it can be a day off):

    ``cl``               three business days before the 25th calendar day of the month before delivery (before the business day preceding the 25th when that is not one)
    ``ng``               three business days before the first calendar day of the delivery month
    ``third_last_bday``  the third last business day of the delivery month (metals)
    ``third_friday``     the third Friday of the delivery month (stock index futures)
    ``treasury``         the seventh business day before the last business day of the delivery month
    ``euro_fx``          two business days before the third Wednesday of the delivery month
    ``grain``            the business day before the 15th calendar day of the delivery month
    """
    if rule == "cl":
        prior = pd.Timestamp(year, month, 1) - pd.offsets.MonthBegin(1)
        day = pd.Timestamp(prior.year, prior.month, 25)
        if day.weekday() >= 5:
            day = day - pd.offsets.BDay(1)
        return day - pd.offsets.BDay(3)
    if rule == "ng":
        return pd.Timestamp(year, month, 1) - pd.offsets.BDay(3)
    if rule == "third_last_bday":
        return _last_bday(year, month) - pd.offsets.BDay(2)
    if rule == "third_friday":
        return _third(4, year, month)
    if rule == "treasury":
        return _last_bday(year, month) - pd.offsets.BDay(7)
    if rule == "euro_fx":
        return _third(2, year, month) - pd.offsets.BDay(2)
    if rule == "grain":
        return pd.Timestamp(year, month, 15) - pd.offsets.BDay(1)
    raise ValueError(f"unknown expiry rule '{rule}'")


def upcoming_contracts(root: FreeRoot, today, n: int = 12) -> list[dict]:
    """The next ``n`` contracts of a root that have not expired on ``today``: ``{contract, symbol, expiry}`` with ``contract`` like ``Z26`` and ``symbol`` the Yahoo symbol ``CLZ26.NYM``."""
    today = pd.Timestamp(today).normalize()
    out = []
    for year in (today.year, today.year + 1, today.year + 2, today.year + 3, today.year + 4):
        for month in root.months:
            expiry = last_trade_date(root.rule, year, month).normalize()
            if expiry < today:
                continue
            code = f"{MONTH_CODES[month - 1]}{year % 100:02d}"
            out.append({"contract": code, "symbol": f"{root.root}{code}{root.suffix}", "expiry": expiry})
    out.sort(key=lambda c: c["expiry"])
    return out[:n]


# ------------------------------------------------------------------------------------------------------------------ Yahoo
def yahoo_history(payload: dict) -> pd.DataFrame:
    """``date, price, volume`` from Yahoo's chart answer. The date is the exchange's calendar day of each bar."""
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict) or chart.get("error") or not chart.get("result"):
        description = (chart or {}).get("error") or {}
        raise FetchError(f"Yahoo Finance has no history for this symbol ({description.get('description', 'no result')})", status=404)
    result = chart["result"][0]
    stamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    close, volume = quote.get("close") or [], quote.get("volume") or []
    if not stamps or len(close) != len(stamps):
        raise FetchError("Yahoo Finance's answer has no prices")
    offset = int((result.get("meta") or {}).get("gmtoffset") or 0)
    dates = [datetime.fromtimestamp(t + offset, tz=timezone.utc).date() for t in stamps]
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "price": pd.to_numeric(pd.Series(close), errors="coerce"), "volume": pd.to_numeric(pd.Series(volume if len(volume) == len(stamps) else [np.nan] * len(stamps)), errors="coerce")})
    frame = frame.dropna(subset=["price"])
    frame = frame[frame["price"] > 0].drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def merge_history(old: pd.DataFrame | None, new: pd.DataFrame, keys=("date", "contract")) -> pd.DataFrame:
    """The union of an earlier download and a new one: where both have a row for the same key the new one wins; the rows of contracts the new download no longer has are kept."""
    if old is None or old.empty:
        return new
    old = old.copy()
    new = new.copy()
    for frame in (old, new):
        frame["date"] = pd.to_datetime(frame["date"])
        if "expiry" in frame:
            frame["expiry"] = pd.to_datetime(frame["expiry"])
    merged = pd.concat([old[~old.set_index(list(keys)).index.isin(new.set_index(list(keys)).index)], new], ignore_index=True)
    return merged.sort_values(["date", "expiry", "contract"]).reset_index(drop=True)


def fetch_yahoo_curve(http: Http, root: str, n_contracts: int = 12, history: str = "1y", today=None, progress=None) -> Download:
    """The curve of ``root`` from Yahoo's per-contract histories. ``history`` is a Yahoo range (``6mo``, ``1y``, ``2y``, ``5y``). Contracts Yahoo does not have are skipped and counted; with none,
    the download fails."""
    root = str(root).upper()
    if root not in FREE_ROOTS:
        raise ValueError(f"free curves are available for {', '.join(FREE_ROOTS)}")
    if history not in ("3mo", "6mo", "1y", "2y", "5y"):
        raise ValueError("history must be 3mo, 6mo, 1y, 2y or 5y")
    spec = FREE_ROOTS[root]
    today = pd.Timestamp(today if today is not None else datetime.now(timezone.utc).date())
    contracts = upcoming_contracts(spec, today, n_contracts)
    frames, missing, blocked = [], [], None
    for i, c in enumerate(contracts):
        if progress:
            progress(f"{root}: {c['contract']} ({i + 1} of {len(contracts)})")
        try:
            h = yahoo_history(http.get_json(YAHOO_CHART.format(symbol=c["symbol"]), {"range": history, "interval": "1d"}))
        except FetchError as exc:
            if exc.blocked or (exc.status is not None and exc.status >= 429):
                blocked = exc
                break
            missing.append(c["contract"])
            continue
        h = h[h["date"] <= c["expiry"] + pd.Timedelta(days=3)]
        h = h.assign(contract=c["contract"], expiry=c["expiry"])
        if len(h):
            frames.append(h)
        else:
            missing.append(c["contract"])
    if blocked is not None and not frames:
        raise blocked
    if not frames:
        raise FetchError(f"Yahoo Finance had no price history for any of the next {len(contracts)} {root} contracts")
    df = pd.concat(frames, ignore_index=True)[["date", "contract", "expiry", "price", "volume"]]
    df["volume"] = df["volume"].fillna(0.0)
    df = df.sort_values(["date", "expiry"]).reset_index(drop=True)
    last = df["date"].max()
    today_rows = df[df["date"] == last]
    notes = [f"Yahoo Finance's delayed daily prices of each {root} contract ({len(frames)} contracts, {df['date'].min().date()} to {last.date()}); the last bar can be a session in progress, not a settlement.",
             "Expiry dates are computed from the exchange's rule without a holiday calendar and can be a day off. The feed has no open interest, so the roll read-outs that need it are unavailable.",
             "Yahoo no longer serves a contract once it expires: later downloads are merged with this one, so the history keeps what has expired."]
    if missing:
        notes.append(f"Yahoo had no history for {', '.join(missing)}; the curve is shorter by those contracts.")
    if blocked is not None:
        notes.append(f"Yahoo Finance stopped answering after {len(frames)} of {len(contracts)} contracts ({blocked}); this is common on shared cloud addresses. Download again in a few minutes to add the rest.")
    meta = {"incomplete": blocked is not None, "source_id": "yahoo", "source": "Yahoo Finance", "url": YAHOO_CHART.format(symbol=f"{root}<month><yy>{spec.suffix}"), "as_of": str(last.date()), "root": root, "contracts": len(frames),
            "contracts_on_last_day": int(len(today_rows)), "first_date": str(df["date"].min().date()), **spec.terms(), "notes": notes}
    return Download("futures", root, "curve", frame=df, meta=meta)


# ------------------------------------------------------------------------------------------------------------------ Deribit
def deribit_curve(result: list, currency: str, min_days: int = 5) -> tuple[pd.DataFrame, dict]:
    """One day's dated-futures curve from Deribit's book summaries: ``date, contract, expiry, price, open_interest, volume, spot``. The price is the mark; open interest and volume are in US dollars.
    Deribit also lists a future for each of the next few days; those and any contract within ``min_days`` of expiry are left out (the roll between two neighbouring days is not a curve), so the strip
    is the Friday expiries: weeklies, monthlies and quarterlies. The perpetual has no expiry and goes into the second value with its funding rate."""
    rows, perp, stamp, index = [], None, [], []
    for f in result:
        name = str(f.get("instrument_name") or "")
        parts = name.split("-")
        if len(parts) != 2 or parts[0] != currency:
            continue
        mark, idx = f.get("mark_price"), f.get("estimated_delivery_price")
        if not mark or not idx:
            continue
        stamp.append(f.get("creation_timestamp"))
        index.append(float(idx))
        if parts[1] == "PERPETUAL":
            perp = {"mark": float(mark), "open_interest": float(f.get("open_interest") or 0.0), "funding_8h": f.get("funding_8h"), "current_funding": f.get("current_funding")}
            continue
        try:
            expiry = pd.Timestamp(datetime.strptime(parts[1], "%d%b%y"))
        except ValueError:
            continue
        rows.append({"contract": parts[1], "expiry": expiry, "price": float(mark), "open_interest": float(f.get("open_interest") or 0.0), "volume": float(f.get("volume_usd") or 0.0), "spot": float(idx)})
    if len(rows) < 2:
        raise FetchError(f"Deribit returned fewer than two dated {currency} futures")
    ts = [s for s in stamp if s]
    day = pd.Timestamp(datetime.fromtimestamp(float(np.median(ts)) / 1000.0, tz=timezone.utc).date()) if ts else pd.Timestamp(datetime.now(timezone.utc).date())
    df = pd.DataFrame(rows)
    df.insert(0, "date", day)
    df = df[(df["expiry"] >= day + pd.Timedelta(days=min_days)) & (df["expiry"].dt.weekday == 4)].sort_values("expiry").reset_index(drop=True)
    if len(df) < 2:
        raise FetchError(f"Deribit has fewer than two weekly, monthly or quarterly {currency} futures at least {min_days} days from expiry")
    return df, {"index": float(np.median(index)), "perpetual": perp}


def fetch_deribit_curve(http: Http, currency: str) -> Download:
    currency = str(currency).upper()
    if currency not in CRYPTO_ROOTS:
        raise ValueError(f"Deribit futures are available for {', '.join(CRYPTO_ROOTS)}")
    answer = http.get_json(DERIBIT_URL, {"currency": currency, "kind": "future"})
    result = answer.get("result") if isinstance(answer, dict) else None
    if not isinstance(result, list):
        raise FetchError("Deribit's answer has no futures list")
    df, extra = deribit_curve(result, currency)
    notes = [f"Deribit's live mark prices of the {currency} weekly, monthly and quarterly futures on {df['date'].iloc[0].date()} ({len(df)} contracts, the daily ones left out); one download is one day's curve, and the file adds a day with each download.",
             "Open interest and volume are in US dollars; the price is the mark. The perpetual is not in the curve (it has no expiry); the spot column is Deribit's index.",
             "Deribit's futures are inverse (margined and settled in the coin), which this workspace treats as linear."]
    meta = {"source_id": "deribit", "source": "Deribit", "url": DERIBIT_URL, "as_of": str(df["date"].iloc[0].date()), "root": currency, "contracts": int(len(df)), "root_name": f"{CRYPTO_ROOTS[currency]} futures",
            "exchange": "Deribit", "asset_class": "crypto", "multiplier": 1.0, "tick_size": 0.5, "perpetual": extra["perpetual"], "index": extra["index"], "notes": notes}
    return Download("futures", currency, "curve", frame=df, meta=meta)
