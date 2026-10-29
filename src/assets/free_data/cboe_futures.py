"""VIX futures from the exchange that lists them, as a curve the Futures workspace reads: ``date, contract, expiry, price, open_interest, volume, spot``.

Cboe Futures Exchange (CFE) publishes, on its historical-data page, one CSV file for every VX contract since 2013: the day's open, high, low, close, **settlement**, volume and **open interest** of that
contract on every day it traded. The page's own list of the contracts (``product/list/VX/``: every expiry with its date and file) is a small JSON document, so a download needs no guess at a symbol or an
expiry rule: both come from the exchange. The files are Cboe's own, free to read, need no account and no key, and are the only free source in this app that has the open interest of a futures contract and
that keeps the contracts after they expire (Yahoo Finance drops them), so the curve of VIX futures can be followed back to January 2013 and a carry or roll-down strategy can be tested on it.

* **Monthly VX contracts only.** The 456 weekly contracts (VX01 ... VX53) expire every Wednesday; mixed into the strip they would make the "next contract" a few days away. They are left out.
* **The price is the settlement.** Before 2014 the files give no settlement (the column is 0), and the close is used; a contract's first rows can be a settlement without a trade (volume 0).
* **The spot column is the VIX index** (Cboe's daily history file), the close of the index and not of the moment the futures settle, so the basis it gives is approximate by that difference.
* **Expired contracts do not change**, so a second download fetches only the contracts that were still alive at the last one and the ones listed since.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from .futures import MONTH_CODES, FreeRoot
from .http import FetchError, Http
from .store import Download

LIST_URL = "https://www-api.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/"
FILE_URL = "https://cdn.cboe.com/{path}"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
PATH = re.compile(r"^data/us/futures/market_statistics/historical_data/VX/VX_\d{4}-\d{2}-\d{2}\.csv$")
FIRST_YEAR = 2013

CBOE_ROOTS = {"VX": FreeRoot("VX", "Cboe Volatility Index (VX) futures", "CFE", "", "volatility", 1000.0, 0.05, tuple(range(1, 13)), "cboe", "Volatility")}


def contracts(listing, since: int = FIRST_YEAR) -> list[dict]:
    """The monthly VX contracts in Cboe's list, oldest first: ``{contract, expiry, path}`` with ``contract`` like ``Z26`` (the month and year it expires in). Anything that is not a monthly VX
    contract with a file path this module knows is skipped, so a change in the list's shape shows as an empty result, not as a download from another address."""
    if not isinstance(listing, dict):
        raise FetchError("Cboe's list of VX contracts is not in the shape this app reads")
    out = []
    for rows in listing.values():
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or row.get("futures_root") != "VX" or row.get("duration_type") != "M" or not PATH.match(str(row.get("path") or "")):
                continue
            try:
                expiry = pd.Timestamp(str(row["expire_date"]))
            except (KeyError, ValueError):
                continue
            if expiry.year >= since:
                out.append({"contract": f"{MONTH_CODES[expiry.month - 1]}{expiry.year % 100:02d}", "expiry": expiry, "path": str(row["path"])})
    out.sort(key=lambda c: c["expiry"])
    return out


def parse_contract(text: str, expiry) -> pd.DataFrame:
    """``date, price, open_interest, volume`` of one contract from its CSV: the settlement (the close where there is none), the open interest and the volume of each day up to the expiry."""
    try:
        raw = pd.read_csv(io.StringIO(text))
    except (ValueError, pd.errors.ParserError) as exc:
        raise FetchError(f"a Cboe VX file could not be read ({exc})") from None
    need = {"Trade Date", "Settle", "Close", "Total Volume", "Open Interest"}
    if not need.issubset(raw.columns):
        raise FetchError(f"a Cboe VX file lacks the columns {sorted(need - set(raw.columns))}")
    settle, close = pd.to_numeric(raw["Settle"], errors="coerce"), pd.to_numeric(raw["Close"], errors="coerce")
    frame = pd.DataFrame({"date": pd.to_datetime(raw["Trade Date"], errors="coerce"), "price": settle.where(settle > 0, close),
                          "open_interest": pd.to_numeric(raw["Open Interest"], errors="coerce"), "volume": pd.to_numeric(raw["Total Volume"], errors="coerce")})
    frame = frame.dropna(subset=["date", "price"])
    frame = frame[(frame["price"] > 0) & (frame["date"] <= pd.Timestamp(expiry))]
    return frame.drop_duplicates("date", keep="last").reset_index(drop=True)


def parse_vix(text: str) -> pd.Series:
    """The VIX index close by day from Cboe's ``VIX_History.csv`` (``DATE`` as month/day/year, ``CLOSE``)."""
    try:
        raw = pd.read_csv(io.StringIO(text))
        dates = pd.to_datetime(raw["DATE"], format="%m/%d/%Y", errors="coerce")
        close = pd.to_numeric(raw["CLOSE"], errors="coerce")
    except (KeyError, ValueError, pd.errors.ParserError):
        raise FetchError("Cboe's VIX history file is not in the shape this app reads") from None
    series = pd.Series(close.to_numpy(), index=dates).dropna()
    return series[~series.index.isna()]


def fetch_vx_curve(http: Http, since: int = FIRST_YEAR, earlier: pd.DataFrame | None = None, progress=None) -> Download:
    """The monthly VX futures of every day since ``since`` (2013 at the earliest), from Cboe's own files. ``earlier`` is the table of a previous download: contracts that had expired by its last day are
    final and are not fetched again, so an update costs the live contracts and the new ones (about eight calls) and not the 175 of the first download."""
    since = max(int(since), FIRST_YEAR)
    todo = contracts(http.get_json(LIST_URL), since)                                              # the list holds only contracts that exist, with a file each
    final: set[str] = set()
    if earlier is not None and len(earlier):
        done = pd.to_datetime(earlier["date"]).max()
        have = earlier.assign(expiry=pd.to_datetime(earlier["expiry"])).groupby("contract")["expiry"].first()
        final = {code for code, expiry in have.items() if expiry + pd.Timedelta(days=3) <= done}
    frames, missing, blocked = [], [], None
    fetch = [c for c in todo if c["contract"] not in final]
    for i, c in enumerate(fetch):
        if progress:
            progress(f"VX: {c['contract']} ({i + 1} of {len(fetch)})")
        try:
            h = parse_contract(http.get_text(FILE_URL.format(path=c["path"])), c["expiry"])
        except FetchError as exc:
            if exc.blocked or (exc.status is not None and exc.status >= 429):
                blocked = exc
                break
            missing.append(c["contract"])
            continue
        if len(h):
            frames.append(h.assign(contract=c["contract"], expiry=c["expiry"]))
        else:
            missing.append(c["contract"])
    if blocked is not None and not frames and not final:
        raise blocked
    if not frames and not final:
        raise FetchError("Cboe had no VIX futures file this app could read")
    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "contract", "expiry", "price", "open_interest", "volume"])
    spot_note = ""
    spot = None
    try:
        spot = parse_vix(http.get_text(VIX_URL))
    except FetchError as exc:
        spot_note = f"The VIX index could not be read ({exc}), so the new rows have no spot."
    if len(fresh) and spot is not None:
        fresh["spot"] = fresh["date"].map(spot)
    keep = earlier[earlier["contract"].isin(final)] if final else None
    df = pd.concat([x for x in (keep, fresh) if x is not None and len(x)], ignore_index=True)
    for col in ("date", "expiry"):
        df[col] = pd.to_datetime(df[col])
    df = df[["date", "contract", "expiry", "price", "open_interest", "volume"] + (["spot"] if "spot" in df else [])].sort_values(["date", "expiry"]).reset_index(drop=True)
    last = df["date"].max()
    first = df["date"].min()
    notes = [f"Cboe Futures Exchange's own daily files for the monthly VX contracts, {first.date()} to {last.date()}: {df['contract'].nunique()} contracts, expired ones included, with settlement, volume and open interest.",
             "The price is the settlement (the close in 2013, when the files give none). The spot column is the VIX index close, not the value at the moment the futures settle. Weekly VX contracts are left out.",
             "Cboe publishes the day's files in the evening of the trading day or the next business morning, so the last row can be a day old."]
    if missing:
        notes.append(f"Cboe's files for {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''} could not be read; the curve lacks those contracts.")
    if blocked is not None:
        notes.append(f"Cboe stopped answering after {len(frames)} files ({blocked}); download again in a few minutes to add the rest.")
    if spot_note:
        notes.append(spot_note)
    spec = CBOE_ROOTS["VX"]
    meta = {"incomplete": blocked is not None, "source_id": "cboe_cfe", "source": "Cboe Futures Exchange", "url": LIST_URL, "as_of": str(last.date()), "first_date": str(first.date()), "root": "VX", "contracts": int(df["contract"].nunique()),
            "contracts_on_last_day": int((df["date"] == last).sum()), **spec.terms(), "has_open_interest": True, "notes": notes}
    return Download("futures", "VX", "curve", frame=df, meta=meta)
