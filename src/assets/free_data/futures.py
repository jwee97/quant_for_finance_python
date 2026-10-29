"""Real futures curves from free sources, in the file format the Futures workspace reads (``date, contract, expiry, price`` and optionally ``open_interest``, ``volume``, ``spot``).

**Exchange-traded futures (Yahoo Finance).** Yahoo publishes a daily price history for each individual contract (``CLZ26.NYM`` is December 2026 WTI crude), as long as the contract has not expired.
For a root the fetcher builds the next ``n`` contract symbols from the root's delivery-month cycle, asks for each one's history and stacks them into one long table: one row per contract per day, so
the curve of any past date in the window can be drawn and the roll between contracts followed. Yahoo drops a contract when it expires, so a download is merged with the file from the last one
(``merge_history``): the rows of contracts Yahoo no longer serves stay, and the history grows past the window of any single request. What the feed does not have: open interest (the workspace's
roll-by-open-interest read-outs are then unavailable), and anything official: these are Yahoo's delayed prices, not exchange settlements, and the last bar of a session in progress is not a settlement.

**Expiry dates** are not in the feed. They are computed from each exchange's published rule (``last_trade_date``) without a holiday calendar, so a date can be a day early or late; the curve only
uses them to measure the time between contracts, where a day is a small error.

**How far out Yahoo goes.** The nearest contracts are not all it has. Measured on 29 October 2026, Yahoo served 53 WTI crude contracts (to December 2036), 72 Henry Hub natural gas (to 2034), 24 copper,
21 Brent, 14 gold, 13 corn, 6 euro FX, 5 E-mini S&P 500 and 3 ten-year note contracts: a download asks for 2 to 120 of them, or with ``n_contracts=None`` for every one that Yahoo lists (the search stops
when three in a row are missing, because its listing has no gaps), and for as much of each one's history as it has (``max``; the far contracts trade rarely, so their histories are sparse). What Yahoo
does not have is any contract that has expired: a curve you download today starts, for the dates before today, at the contracts that are still alive.

**Bitcoin and Ether (Deribit).** Deribit lists dated futures as well as the perpetual and publishes the whole strip at once, with open interest. ``fetch_deribit_curve`` takes one day's curve; the file adds a
day with every download. ``fetch_deribit_history`` takes the whole past instead: Deribit keeps the daily chart of a future after it has expired, so the monthly and quarterly futures since 2019 come
back in one download (prices are the last trade of each day, volume is in dollars, and there is no open interest in a chart).

**Any futures product (iTick).** iTick serves one daily price series per product (``NQ``, ``HSI``, ``RB``), not one per contract month, so it cannot draw a curve; ``data_itick`` downloads it
as a price history, which the Futures tab's History view reads. The product list is ``itick_catalog``.
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
DERIBIT_CHART = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
DERIBIT_INDEX = "https://www.deribit.com/api/v2/public/get_index_chart_data"
ROOT = re.compile(r"^[A-Z0-9]{1,4}$")
HISTORY_RANGES = ("3mo", "6mo", "1y", "2y", "5y", "10y", "max")
MAX_CONTRACTS = 160                       # the most contracts one download asks Yahoo for
DEEP_YEARS = 14                           # a deep download looks at most this many years ahead
GAP_DAYS = 730                            # ... and ends when two years pass after the last contract Yahoo has with none found: it lists far contracts only when they trade (WTI after five years is June
                                          # and December only, 8 months in a row can be missing), so a few misses in a row are not the end of its listing, two years of them are
FIRST_DAYS = 400                          # before the first contract Yahoo has, it may skip the nearest months (gone from Yahoo, still traded), but not more than this many days
SPARSE_BARS = 20                          # a contract with fewer daily bars than this is called sparse in the notes
DERIBIT_FIRST_YEAR = 2019                 # the first year Deribit's monthly and quarterly futures are looked for


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
    group: str = "Other"                  # how the page groups the roots: Energy, Metals, Grains, Livestock, Softs, Equity indexes, Rates, Currencies

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.multiplier

    def terms(self) -> dict:
        """What a record keeps of the root: its long name as ``root_name`` (``name`` is the file's), exchange, class, multiplier and tick size."""
        d = asdict(self)
        d.pop("months"), d.pop("rule"), d.pop("suffix"), d.pop("group")
        d["root_name"] = d.pop("name")
        return d


def _roots(*rows: FreeRoot) -> dict[str, FreeRoot]:
    return {r.root: r for r in rows}


ALL = tuple(range(1, 13))
QUARTERLY = (3, 6, 9, 12)
EVEN = (2, 4, 6, 8, 10, 12)
GRAIN_CYCLE = (3, 5, 7, 9, 12)
SOY_CYCLE = (1, 3, 5, 7, 8, 9, 10, 12)
# Contract terms are the exchanges' published specifications (price units: a bushel crop in cents, meat in cents a pound, fuels in dollars a gallon, metals in dollars an ounce, rates in points).
# The delivery months are the standard cycle; Yahoo lists only the nearest few of the later ones and the fetcher skips any contract Yahoo does not have. Margins are not kept: they change every few
# months, so the workspace shows an illustrative share of the contract's value for these (see ``futures_analytics.illustrative_margin``).
FREE_ROOTS = _roots(
    FreeRoot("CL", "WTI crude oil", "NYMEX", ".NYM", "commodity", 1000.0, 0.01, ALL, "cl", "Energy"),
    FreeRoot("NG", "Henry Hub natural gas", "NYMEX", ".NYM", "commodity", 10000.0, 0.001, ALL, "ng", "Energy"),
    FreeRoot("HO", "NY Harbor ULSD (heating oil)", "NYMEX", ".NYM", "commodity", 42000.0, 0.0001, ALL, "last_bday_prior", "Energy"),
    FreeRoot("RB", "RBOB gasoline", "NYMEX", ".NYM", "commodity", 42000.0, 0.0001, ALL, "last_bday_prior", "Energy"),
    FreeRoot("BZ", "Brent crude oil (last day)", "NYMEX", ".NYM", "commodity", 1000.0, 0.01, ALL, "brent", "Energy"),
    FreeRoot("GC", "Gold", "COMEX", ".CMX", "metal", 100.0, 0.10, EVEN, "third_last_bday", "Metals"),
    FreeRoot("SI", "Silver", "COMEX", ".CMX", "metal", 5000.0, 0.005, (1, 3, 5, 7, 9, 12), "third_last_bday", "Metals"),
    FreeRoot("HG", "Copper", "COMEX", ".CMX", "commodity", 25000.0, 0.0005, ALL, "third_last_bday", "Metals"),
    FreeRoot("PL", "Platinum", "NYMEX", ".NYM", "metal", 50.0, 0.10, (1, 4, 7, 10), "third_last_bday", "Metals"),
    FreeRoot("PA", "Palladium", "NYMEX", ".NYM", "metal", 100.0, 0.05, QUARTERLY, "third_last_bday", "Metals"),
    FreeRoot("ZC", "Corn", "CBOT", ".CBT", "agricultural", 50.0, 0.25, GRAIN_CYCLE, "grain", "Grains"),
    FreeRoot("ZS", "Soybeans", "CBOT", ".CBT", "agricultural", 50.0, 0.25, (1, 3, 5, 7, 8, 9, 11), "grain", "Grains"),
    FreeRoot("ZW", "Chicago wheat", "CBOT", ".CBT", "agricultural", 50.0, 0.25, GRAIN_CYCLE, "grain", "Grains"),
    FreeRoot("KE", "Kansas City hard red winter wheat", "CBOT", ".CBT", "agricultural", 50.0, 0.25, GRAIN_CYCLE, "grain", "Grains"),
    FreeRoot("ZM", "Soybean meal", "CBOT", ".CBT", "agricultural", 100.0, 0.10, SOY_CYCLE, "grain", "Grains"),
    FreeRoot("ZL", "Soybean oil", "CBOT", ".CBT", "agricultural", 600.0, 0.01, SOY_CYCLE, "grain", "Grains"),
    FreeRoot("ZO", "Oats", "CBOT", ".CBT", "agricultural", 50.0, 0.25, GRAIN_CYCLE, "grain", "Grains"),
    FreeRoot("ZR", "Rough rice", "CBOT", ".CBT", "agricultural", 2000.0, 0.005, (1, 3, 5, 7, 9, 11), "grain", "Grains"),
    FreeRoot("LE", "Live cattle", "CME", ".CME", "agricultural", 400.0, 0.025, EVEN, "last_bday", "Livestock"),
    FreeRoot("HE", "Lean hogs", "CME", ".CME", "agricultural", 400.0, 0.025, (2, 4, 5, 6, 7, 8, 10, 12), "tenth_bday", "Livestock"),
    FreeRoot("GF", "Feeder cattle", "CME", ".CME", "agricultural", 500.0, 0.025, (1, 3, 4, 5, 8, 9, 10, 11), "last_thursday", "Livestock"),
    FreeRoot("SB", "Sugar No. 11", "ICE US", ".NYB", "agricultural", 1120.0, 0.01, (3, 5, 7, 10), "last_bday_prior", "Softs"),
    FreeRoot("KC", "Coffee C", "ICE US", ".NYB", "agricultural", 375.0, 0.05, GRAIN_CYCLE, "coffee", "Softs"),
    FreeRoot("CC", "Cocoa", "ICE US", ".NYB", "agricultural", 10.0, 1.0, GRAIN_CYCLE, "cocoa", "Softs"),
    FreeRoot("CT", "Cotton No. 2", "ICE US", ".NYB", "agricultural", 500.0, 0.01, (3, 5, 7, 10, 12), "cotton", "Softs"),
    FreeRoot("OJ", "Orange juice", "ICE US", ".NYB", "agricultural", 150.0, 0.05, (1, 3, 5, 7, 9, 11), "oj", "Softs"),
    FreeRoot("ES", "E-mini S&P 500", "CME", ".CME", "equity_index", 50.0, 0.25, QUARTERLY, "third_friday", "Equity indexes"),
    FreeRoot("NQ", "E-mini Nasdaq-100", "CME", ".CME", "equity_index", 20.0, 0.25, QUARTERLY, "third_friday", "Equity indexes"),
    FreeRoot("YM", "E-mini Dow Jones", "CBOT", ".CBT", "equity_index", 5.0, 1.0, QUARTERLY, "third_friday", "Equity indexes"),
    FreeRoot("RTY", "E-mini Russell 2000", "CME", ".CME", "equity_index", 50.0, 0.10, QUARTERLY, "third_friday", "Equity indexes"),
    FreeRoot("ZT", "2-year Treasury note", "CBOT", ".CBT", "rates", 2000.0, 0.00390625, QUARTERLY, "last_bday", "Rates"),
    FreeRoot("ZF", "5-year Treasury note", "CBOT", ".CBT", "rates", 1000.0, 0.0078125, QUARTERLY, "last_bday", "Rates"),
    FreeRoot("ZN", "10-year Treasury note", "CBOT", ".CBT", "rates", 1000.0, 0.015625, QUARTERLY, "treasury", "Rates"),
    FreeRoot("TN", "Ultra 10-year Treasury note", "CBOT", ".CBT", "rates", 1000.0, 0.015625, QUARTERLY, "treasury", "Rates"),
    FreeRoot("ZB", "Treasury bond", "CBOT", ".CBT", "rates", 1000.0, 0.03125, QUARTERLY, "treasury", "Rates"),
    FreeRoot("UB", "Ultra Treasury bond", "CBOT", ".CBT", "rates", 1000.0, 0.03125, QUARTERLY, "treasury", "Rates"),
    FreeRoot("6E", "Euro FX", "CME", ".CME", "fx", 125000.0, 0.00005, QUARTERLY, "euro_fx", "Currencies"),
    FreeRoot("6J", "Japanese yen", "CME", ".CME", "fx", 12500000.0, 0.0000005, QUARTERLY, "euro_fx", "Currencies"),
    FreeRoot("6B", "British pound", "CME", ".CME", "fx", 62500.0, 0.0001, QUARTERLY, "euro_fx", "Currencies"),
    FreeRoot("6A", "Australian dollar", "CME", ".CME", "fx", 100000.0, 0.0001, QUARTERLY, "euro_fx", "Currencies"),
    FreeRoot("6C", "Canadian dollar", "CME", ".CME", "fx", 100000.0, 0.00005, QUARTERLY, "cad", "Currencies"),
    FreeRoot("6S", "Swiss franc", "CME", ".CME", "fx", 125000.0, 0.0001, QUARTERLY, "euro_fx", "Currencies"),
    FreeRoot("6N", "New Zealand dollar", "CME", ".CME", "fx", 100000.0, 0.0001, QUARTERLY, "euro_fx", "Currencies"),
    FreeRoot("DX", "US dollar index", "ICE US", ".NYB", "fx", 1000.0, 0.005, QUARTERLY, "euro_fx", "Currencies"),
)
CRYPTO_ROOTS = {"BTC": "Bitcoin", "ETH": "Ether"}


# ------------------------------------------------------------------------------------------------------------------ expiry rules
def _last_bday(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year, month, 1) + pd.offsets.BMonthEnd(0)


def _third(weekday: int, year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(weekday - first.weekday()) % 7 + 14)


def _prior_month(year: int, month: int, back: int = 1) -> tuple[int, int]:
    index = year * 12 + (month - 1) - back
    return index // 12, index % 12 + 1


def _first_bday(year: int, month: int) -> pd.Timestamp:
    return pd.bdate_range(pd.Timestamp(year, month, 1), periods=1)[0]


def last_trade_date(rule: str, year: int, month: int) -> pd.Timestamp:
    """The last trading day of the contract that delivers in ``year``-``month``, by the exchange's rule and without holidays (so it can be a day off):

    ``cl``               three business days before the 25th calendar day of the month before delivery (before the business day preceding the 25th when that is not one)
    ``ng``               three business days before the first calendar day of the delivery month
    ``third_last_bday``  the third last business day of the delivery month (metals)
    ``third_friday``     the third Friday of the delivery month (stock index futures)
    ``treasury``         the seventh business day before the last business day of the delivery month
    ``euro_fx``          two business days before the third Wednesday of the delivery month
    ``grain``            the business day before the 15th calendar day of the delivery month
    ``last_bday``        the last business day of the delivery month (the 2- and 5-year notes, live cattle)
    ``last_bday_prior``  the last business day of the month before delivery (heating oil, gasoline, sugar)
    ``brent``            the last business day of the second month before delivery
    ``tenth_bday``       the tenth business day of the delivery month (lean hogs)
    ``last_thursday``    the last Thursday of the delivery month (feeder cattle)
    ``cad``              the business day before the third Wednesday of the delivery month (Canadian dollar)
    ``coffee``           eight business days before the first business day of the delivery month (one before the first notice day)
    ``cocoa``            eleven business days before the last business day of the delivery month
    ``cotton``           seventeen business days before the last business day of the delivery month
    ``oj``               fourteen business days before the last business day of the delivery month

    These follow the exchanges' published specifications from memory of their wording and without a holiday calendar: a date can be a day or two off.
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
    if rule == "last_bday":
        return _last_bday(year, month)
    if rule == "last_bday_prior":
        return _last_bday(*_prior_month(year, month))
    if rule == "brent":
        return _last_bday(*_prior_month(year, month, 2))
    if rule == "tenth_bday":
        return pd.bdate_range(pd.Timestamp(year, month, 1), periods=10)[-1]
    if rule == "last_thursday":
        last = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
        return last - pd.Timedelta(days=(last.weekday() - 3) % 7)
    if rule == "cad":
        return _third(2, year, month) - pd.offsets.BDay(1)
    if rule == "coffee":
        return _first_bday(year, month) - pd.offsets.BDay(8)
    if rule == "cocoa":
        return _last_bday(year, month) - pd.offsets.BDay(11)
    if rule == "cotton":
        return _last_bday(year, month) - pd.offsets.BDay(17)
    if rule == "oj":
        return _last_bday(year, month) - pd.offsets.BDay(14)
    raise ValueError(f"unknown expiry rule '{rule}'")


def upcoming_contracts(root: FreeRoot, today, n: int = 12) -> list[dict]:
    """The next ``n`` contracts of a root that have not expired on ``today``: ``{contract, symbol, expiry}`` with ``contract`` like ``Z26`` and ``symbol`` the Yahoo symbol ``CLZ26.NYM``."""
    today = pd.Timestamp(today).normalize()
    out = []
    span = max(5, -(-int(n) // max(len(root.months), 1)) + 2)                                  # as many years ahead as n contracts of this root's cycle need, and two to spare
    for year in range(today.year, today.year + span):
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


def fetch_yahoo_curve(http: Http, root: str, n_contracts: int | None = 12, history: str = "1y", today=None, progress=None) -> Download:
    """The curve of ``root`` from Yahoo's per-contract histories. ``history`` is a Yahoo range (``6mo`` ... ``10y``, ``max``). Contracts Yahoo does not have are skipped and counted; with none, the
    download fails.

    ``n_contracts`` is how many of the next delivery months to ask for (1 to ``MAX_CONTRACTS``), or ``None`` (``0`` is the same) for every contract Yahoo lists: the search then goes month by month for
    up to ``DEEP_YEARS`` years and ends when ``GAP_DAYS`` pass after the last contract found with no other, so the request count follows what exists."""
    root = str(root).upper()
    if root not in FREE_ROOTS:
        raise ValueError(f"free curves are available for {', '.join(FREE_ROOTS)}")
    if history not in HISTORY_RANGES:
        raise ValueError(f"history must be one of {', '.join(HISTORY_RANGES)}")
    deep = n_contracts is None or int(n_contracts) <= 0
    if not deep and not 1 <= int(n_contracts) <= MAX_CONTRACTS:
        raise ValueError(f"n_contracts must be 1 to {MAX_CONTRACTS}, or None for every contract Yahoo lists")
    spec = FREE_ROOTS[root]
    today = pd.Timestamp(today if today is not None else datetime.now(timezone.utc).date())
    contracts = upcoming_contracts(spec, today, MAX_CONTRACTS if deep else int(n_contracts))
    if deep:
        contracts = [c for c in contracts if c["expiry"] <= today + pd.DateOffset(years=DEEP_YEARS)]
    frames, missing, blocked, last_found = [], [], None, None
    for i, c in enumerate(contracts):
        if deep and ((last_found is not None and c["expiry"] > last_found + pd.Timedelta(days=GAP_DAYS)) or (last_found is None and c["expiry"] > today + pd.Timedelta(days=FIRST_DAYS))):
            break
        if progress:
            progress(f"{root}: {c['contract']} ({i + 1} of {len(contracts)})" if not deep else f"{root}: {c['contract']} ({len(frames)} found)")
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
            last_found = c["expiry"]
        else:
            missing.append(c["contract"])
    if blocked is not None and not frames:
        raise blocked
    if not frames:
        raise FetchError(f"Yahoo Finance had no price history for any of the next {len(contracts) if not deep else len(missing)} {root} contracts")
    if deep:
        missing = [m for m in missing if next(c["expiry"] for c in contracts if c["contract"] == m) < last_found]       # months after the last contract Yahoo has are the end of its listing, not gaps
    df = pd.concat(frames, ignore_index=True)[["date", "contract", "expiry", "price", "volume"]]
    df["volume"] = df["volume"].fillna(0.0)
    df = df.sort_values(["date", "expiry"]).reset_index(drop=True)
    last = df["date"].max()
    today_rows = df[df["date"] == last]
    bars = df.groupby("contract")["date"].count()
    sparse = [c["contract"] for c in contracts if c["contract"] in bars.index and bars[c["contract"]] < SPARSE_BARS]
    furthest = max((c for c in contracts if c["contract"] in bars.index), key=lambda c: c["expiry"])
    notes = [f"Yahoo Finance's delayed daily prices of each {root} contract ({len(frames)} contracts, {df['date'].min().date()} to {last.date()}); the last bar can be a session in progress, not a settlement.",
             "Expiry dates are computed from the exchange's rule without a holiday calendar and can be a day off. The feed has no open interest, so the roll read-outs that need it are unavailable.",
             "Yahoo no longer serves a contract once it expires: later downloads are merged with this one, so the history keeps what has expired."]
    if deep:
        notes.append(f"Yahoo lists {len(frames)} {root} contracts, as far out as {furthest['contract']} ({furthest['expiry'].date()}); it lists a far contract only once it trades, so {len(missing)} of the delivery months before it are not in the file "
                     "and the far end of the curve has gaps." if missing else f"Yahoo lists {len(frames)} {root} contracts, as far out as {furthest['contract']} ({furthest['expiry'].date()}); it does not list the ones after.")
        if sparse:
            notes.append(f"The far contracts trade rarely: {', '.join(sparse[:8])}{' ...' if len(sparse) > 8 else ''} have fewer than {SPARSE_BARS} daily bars, so read the far end of the curve as indicative.")
    elif missing:
        notes.append(f"Yahoo had no history for {', '.join(missing)}; the curve is shorter by those contracts.")
    if blocked is not None:
        notes.append(f"Yahoo Finance stopped answering after {len(frames)} of {len(contracts)} contracts ({blocked}); this is common on shared cloud addresses. Download again in a few minutes to add the rest.")
    meta = {"incomplete": blocked is not None, "source_id": "yahoo", "source": "Yahoo Finance", "url": YAHOO_CHART.format(symbol=f"{root}<month><yy>{spec.suffix}"), "as_of": str(last.date()), "root": root, "contracts": len(frames),
            "contracts_on_last_day": int(len(today_rows)), "first_date": str(df["date"].min().date()), "deep": deep, "furthest": furthest["contract"], "furthest_expiry": str(furthest["expiry"].date()),
            "sparse_contracts": sparse, **spec.terms(), "notes": notes}
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


# ------------------------------------------------------------------------------------------------------------------ Deribit: the past of the dated futures
def _last_friday(year: int, month: int) -> pd.Timestamp:
    last = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.weekday() - 4) % 7)


def deribit_expiries(today, first_year: int = DERIBIT_FIRST_YEAR, months_ahead: int = 12) -> list[pd.Timestamp]:
    """The last Friday of every month from January of ``first_year`` to ``months_ahead`` months after ``today``: the days Deribit's monthly and quarterly futures expire (08:00 UTC). Not every one of them
    existed (the early years had only the quarters, ETH started later); the download asks for each and keeps the ones Deribit answers for."""
    today = pd.Timestamp(today).normalize()
    stop = today + pd.DateOffset(months=months_ahead)
    out = []
    for year in range(first_year, stop.year + 1):
        for month in range(1, 13):
            day = _last_friday(year, month)
            if day <= stop:
                out.append(day)
    return out


def deribit_chart(result: dict, expiry) -> pd.DataFrame:
    """``date, price, volume`` of one future from Deribit's daily chart. The price is the last trade of the day (a day without a trade repeats the day before), the volume is in US dollars. Deribit goes on
    drawing the chart of a future after it has expired, as a flat line: rows after the expiry day are dropped, and so are the rows before the first day with a trade."""
    ticks = (result or {}).get("ticks") or []
    if (result or {}).get("status") != "ok" or not ticks or len(result.get("close") or []) != len(ticks):
        raise FetchError("Deribit has no daily prices for this future")
    cost = result.get("cost") or [0.0] * len(ticks)
    df = pd.DataFrame({"date": pd.to_datetime(ticks, unit="ms").normalize(), "price": pd.to_numeric(pd.Series(result["close"]), errors="coerce"),
                       "volume": pd.to_numeric(pd.Series(cost if len(cost) == len(ticks) else [0.0] * len(ticks)), errors="coerce").fillna(0.0)})
    df = df[(df["price"] > 0) & (df["date"] <= pd.Timestamp(expiry).normalize())]
    traded = df.index[df["volume"] > 0]
    if len(traded) == 0:
        raise FetchError("Deribit shows no trade in this future")
    return df.loc[traded[0]:].drop_duplicates("date", keep="last").reset_index(drop=True)


def deribit_index(result: list) -> pd.Series:
    """The coin's index price by day (the last of the day's readings) from Deribit's index chart: the ``spot`` of the curve."""
    rows = [(int(t), float(v)) for t, v in (result or []) if v]
    if not rows:
        raise FetchError("Deribit's index chart is empty")
    frame = pd.DataFrame(rows, columns=["t", "v"])
    frame["date"] = pd.to_datetime(frame["t"], unit="ms").dt.normalize()
    return frame.groupby("date")["v"].last()


def fetch_deribit_history(http: Http, currency: str, today=None, first_year: int = DERIBIT_FIRST_YEAR, progress=None) -> Download:
    """The curve of Deribit's monthly and quarterly futures on every day since ``first_year``, expired contracts included, from the daily charts it keeps (one call per contract and one for the index).
    A contract Deribit has no chart for is skipped; with none the download fails."""
    currency = str(currency).upper()
    if currency not in CRYPTO_ROOTS:
        raise ValueError(f"Deribit futures are available for {', '.join(CRYPTO_ROOTS)}")
    now = pd.Timestamp(today if today is not None else datetime.now(timezone.utc).date()).normalize()
    expiries = deribit_expiries(now, first_year)
    frames, missing, blocked = [], [], None
    for i, expiry in enumerate(expiries):
        name = f"{currency}-{expiry:%d%b%y}".upper()
        listed = expiry - pd.Timedelta(days=400)
        if listed > now:
            continue
        if progress:
            progress(f"{currency}: {name[len(currency) + 1:]} ({i + 1} of {len(expiries)})")
        params = {"instrument_name": name, "resolution": "1D", "start_timestamp": int(listed.timestamp() * 1000), "end_timestamp": int((min(expiry, now) + pd.Timedelta(days=2)).timestamp() * 1000)}
        try:
            answer = http.get_json(DERIBIT_CHART, params)
            h = deribit_chart(answer.get("result") if isinstance(answer, dict) else None, expiry)
        except FetchError as exc:
            if exc.blocked or (exc.status is not None and exc.status >= 429):
                blocked = exc
                break
            missing.append(name)
            continue
        frames.append(h.assign(contract=name[len(currency) + 1:], expiry=expiry))
    if blocked is not None and not frames:
        raise blocked
    if not frames:
        raise FetchError(f"Deribit had no daily chart for any {currency} monthly or quarterly future")
    df = pd.concat(frames, ignore_index=True)
    spot_note = ""
    try:
        answer = http.get_json(DERIBIT_INDEX, {"index_name": f"{currency.lower()}_usd", "range": "all"})
        spot = deribit_index(answer.get("result") if isinstance(answer, dict) else None)
        df["spot"] = df["date"].map(spot)
    except FetchError as exc:
        spot_note = f" The index price could not be read ({exc}), so this file has no spot column."
    df = df[["date", "contract", "expiry", "price", "volume"] + (["spot"] if "spot" in df else [])].sort_values(["date", "expiry"]).reset_index(drop=True)
    last = df["date"].max()
    notes = [f"Deribit's daily charts of the {currency} monthly and quarterly futures, {df['date'].min().date()} to {last.date()}: {len(frames)} contracts, expired ones included. The price is the last trade of each day "
             "(a day without a trade repeats the day before); the volume is in US dollars; a chart has no open interest.",
             "Deribit's futures are inverse (margined and settled in the coin), which this workspace treats as linear. Weekly and daily futures are left out, so the curve is the monthly and quarterly strip.",
             "Downloading the current curve (mark prices and open interest) adds today's row with open interest to this file."]
    if missing:
        notes.append(f"Deribit had no chart for {len(missing)} of the {len(frames) + len(missing)} month-end Fridays asked about ({', '.join(m[len(currency) + 1:] for m in missing[:6])}{' ...' if len(missing) > 6 else ''}): no future of that name existed.")
    if blocked is not None:
        notes.append(f"Deribit stopped answering after {len(frames)} contracts ({blocked}); download again in a few minutes to add the rest.")
    if spot_note:
        notes.append(spot_note.strip())
    meta = {"incomplete": blocked is not None, "source_id": "deribit", "source": "Deribit", "url": DERIBIT_CHART, "as_of": str(last.date()), "root": currency, "contracts": len(frames), "root_name": f"{CRYPTO_ROOTS[currency]} futures",
            "exchange": "Deribit", "asset_class": "crypto", "multiplier": 1.0, "tick_size": 0.5, "first_date": str(df["date"].min().date()), "history": True, "notes": notes}
    return Download("futures", currency, "curve", frame=df, meta=meta)
