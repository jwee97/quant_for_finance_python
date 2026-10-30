"""Futures from Finzdata (``api.finzdata.com``): every listed contract of a root, with open interest, in the curve file the Futures tab reads (``date, contract, expiry, price`` and ``open_interest``,
``volume``).

**What is known, and from where.** The paths and parameters are from the service's own OpenAPI document (``/openapi.json``): ``/v1/futures/roots`` (all markets), ``/v1/futures/contracts?root=ES`` (every
contract of a root), ``/v1/futures/daily?root=ES[&contract=ESZ26][&continuous=true][&adjust=none][&start=][&end=][&limit=]`` (daily bars) and ``/v1/futures/bars`` (intraday, paid). The key goes in an
``X-API-Key`` header and starts with ``fz_``; an answer is ``{"data": [rows ...]}`` with, on a plan that cuts the history short, a ``history_note`` and an ``upgrade`` offer (the vendor's own Python client
reads exactly that). Its public pages say that a daily bar has open, high, low, close, volume and open interest, that every individual contract since 2007 is included (88 for ES, 17,500+ in all), that a
contract is written like ``CLZ26`` and that **the free plan covers the last 30 days only**, 1,000 calls a day. A 401 is a missing or bad key, a 403 a plan without this data and a 429 the day's limit.

**What is not known.** The OpenAPI document leaves the rows of these four answers undefined and the pages give no sample, so the names of the columns are not documented. The readers below therefore accept
the usual names for each field (``date`` or ``trade_date``; ``close``, ``settle`` or ``price``; ``open_interest`` or ``oi`` ...) and, when a field they need is missing, say which columns the service did send,
so that one failed download is enough to correct a name. Nothing here has been run against the live service with a key.

**Contracts.** A contract code is decoded to its delivery month and year (``ESZ26`` is December 2026) and written ``Z26``, the label the Yahoo curves use, so the two sources merge into one file per root.
The expiry is the service's own when it sends one; else, for a contract that no longer trades, the day of its last bar; else the exchange's rule where the platform has it (``FREE_ROOTS``); else the 15th of the
delivery month, which only orders the contracts and measures the time between them.

**Thirty days.** A plan that shows 30 days shows the last 30 days at the time of the call, so the history is only kept by calling again before it slides past: each download is merged into the file of the
last one by the caller (``free_data.futures.merge_history``), and :func:`gaps` says which days no download ever caught.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd

from .futures import FREE_ROOTS, MONTH_CODES, last_trade_date
from .http import FetchError
from .store import Download

BASE_URL = "https://api.finzdata.com"
KEY_VARIABLE = "FINZDATA_API_KEY"
FREE_WINDOW_DAYS = 30
SOURCE = "Finzdata"
SOURCE_ID = "finzdata"
ROWS_LIMIT = 1_000_000                                          # the largest ``limit`` the service accepts
EMPTY_RUN = 5                                                   # contract-by-contract: this many answers in a row with no bar end the search (the period has none)
MONTH_YEAR = re.compile(r"([FGHJKMNQUVXZ])(\d{4}|\d{2}|\d)")
# Finzdata's symbol -> the platform's root for the same market, where they differ (the names are from Finzdata's list: E6 is "Euro FX", US the "US Treasury Bond").
ALIAS = {"A6": "6A", "B6": "6B", "E6": "6E", "J1": "6J", "N6": "6N", "E1": "6S", "AD": "6C", "US": "ZB"}

ROOT_NAMES = ("root", "symbol", "code", "ticker", "market")
NAME_NAMES = ("name", "description", "contract_name", "title", "market_name")
EXCHANGE_NAMES = ("exchange", "venue", "market_exchange")
GROUP_NAMES = ("asset_class", "class", "sector", "category", "group", "type")
FIRST_NAMES = ("first_date", "first", "start", "start_date", "history_from", "first_bar", "from")
LAST_NAMES = ("last_date", "last", "end", "end_date", "last_bar", "to")
COUNT_NAMES = ("contracts", "n_contracts", "contract_count", "num_contracts")
CONTRACT_NAMES = ("contract", "contract_code", "symbol", "code", "ticker", "contract_symbol", "source_contract")
EXPIRY_NAMES = ("expiry", "expiration", "expiration_date", "expiry_date", "last_trade", "last_trade_date", "maturity", "maturity_date")
DATE_NAMES = ("date", "trade_date", "ts", "timestamp", "datetime", "day")
PRICE_NAMES = ("settle", "settlement", "settlement_price", "close", "price", "last")
OPEN_NAMES, HIGH_NAMES, LOW_NAMES = ("open",), ("high",), ("low",)
VOLUME_NAMES = ("volume", "vol")
OI_NAMES = ("open_interest", "openinterest", "oi", "open_int")
CURVE_COLUMNS = ["date", "contract", "expiry", "price", "open_interest", "volume"]


class ShapeError(FetchError):
    """An answer from Finzdata that does not have what a reader needs. The message lists the columns it did have, so that a name can be added."""


# ------------------------------------------------------------------------------------------------------------------ the envelope
def rows_of(payload, what: str) -> list[dict]:
    """The rows of an answer: ``payload["data"]`` (the service's envelope), else a bare list, else the only list of objects in the document."""
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    elif isinstance(payload, dict):
        lists = [v for v in payload.values() if isinstance(v, list) and (not v or isinstance(v[0], dict))]
        if len(lists) != 1:
            raise ShapeError(f"Finzdata's answer for {what} has no list of rows: its keys are {sorted(payload)[:12]}")
        rows = lists[0]
    else:
        raise ShapeError(f"Finzdata's answer for {what} is not a document of rows")
    if any(not isinstance(r, dict) for r in rows):
        raise ShapeError(f"Finzdata's rows for {what} are not objects")
    return rows


def notes_of(payload) -> list[str]:
    """What the service says about the history it left out: ``history_note`` and the message of ``upgrade`` (a plain sentence; links are not kept)."""
    upgrade = payload.get("upgrade") if isinstance(payload, dict) else None
    out: list[str] = []
    for text in ((payload.get("history_note") if isinstance(payload, dict) else None), upgrade.get("message") if isinstance(upgrade, dict) else None):
        if isinstance(text, str) and text.strip():
            line = " ".join(text.split())[:300]
            if line not in out:
                out.append(line)
    return out


def _lower(rows: list[dict]) -> list[dict]:
    return [{str(k).lower(): v for k, v in r.items()} for r in rows]


def _columns(rows: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows[:50]:
        seen.update(dict.fromkeys(r))
    return list(seen)


def _pick(row: dict, names) -> object:
    for name in names:
        value = row.get(name)
        if value is not None and value == value and value != "":
            return value
    return None


def _need(rows: list[dict], names, what: str, field: str) -> None:
    if rows and not any(_pick(r, names) is not None for r in rows[:50]):
        raise ShapeError(f"Finzdata's rows for {what} have no {field} (looked for {', '.join(names)}); the columns it sent are: {', '.join(_columns(rows))}")


def _date(value) -> pd.Timestamp:
    """A calendar day (no time, no zone) from whatever the service wrote, or ``NaT``."""
    if value is None:
        return pd.NaT
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if stamp is pd.NaT:
        return pd.NaT
    return (stamp.tz_localize(None) if stamp.tzinfo is None else stamp.tz_convert(None)).normalize()


def _iso(value) -> str | None:
    day = _date(value)
    return None if day is pd.NaT else str(day.date())


# ------------------------------------------------------------------------------------------------------------------ roots
def clean_root(value) -> str:
    """``/ES``, ``es`` and ``ES`` are the same root."""
    return str(value or "").strip().lstrip("/").upper()


def normalise_roots(payload) -> list[dict]:
    """``[{root, name, exchange, group, first_date, last_date, contracts}]`` from the answer of ``/v1/futures/roots``; a field the service does not send is ``None``."""
    rows = _lower(rows_of(payload, "the list of futures markets"))
    _need(rows, ROOT_NAMES, "the list of futures markets", "symbol")
    out, seen = [], set()
    for r in rows:
        root = clean_root(_pick(r, ROOT_NAMES))
        if not root or root in seen:
            continue
        seen.add(root)
        count = _pick(r, COUNT_NAMES)
        out.append({"root": root, "name": str(_pick(r, NAME_NAMES) or root), "exchange": _pick(r, EXCHANGE_NAMES), "group": _pick(r, GROUP_NAMES),
                    "first_date": _iso(_pick(r, FIRST_NAMES)), "last_date": _iso(_pick(r, LAST_NAMES)),
                    "contracts": int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else None})
    return out


# ------------------------------------------------------------------------------------------------------------------ contracts
def decode_contract(code, root: str | None = None, reference=None) -> tuple[int, int]:
    """``(year, month)`` of a contract code such as ``ESZ26`` (December 2026), ``CLZ2026`` or ``NGF7``. A one-digit year is the year ending in that digit nearest to ``reference`` (default today).
    Raises ``ValueError`` for a code that is not a root followed by a month letter and a year."""
    text = re.sub(r"[^A-Z0-9]", "", str(code or "").upper().split(".")[0])
    stem = clean_root(root)
    match = MONTH_YEAR.fullmatch(text[len(stem):]) if stem and text.startswith(stem) else None
    if match is None:                                                              # another spelling of the root (6E, E6): the tail is what counts
        tail = re.search(r"([FGHJKMNQUVXZ])(\d{4}|\d{2}|\d)$", text)
        match = MONTH_YEAR.fullmatch(tail.group(0)) if tail else None
    if match is None:
        raise ValueError(f"'{code}' is not a futures contract code (a root, a month letter and a year, such as ESZ26)")
    month, digits = MONTH_CODES.index(match.group(1)) + 1, match.group(2)
    if len(digits) == 4:
        return int(digits), month
    if len(digits) == 2:
        return 2000 + int(digits), month
    today = pd.Timestamp(reference if reference is not None else datetime.now(timezone.utc).date())
    decade = today.year - today.year % 10
    return min((decade + shift + int(digits) for shift in (-10, 0, 10)), key=lambda y: abs(y - today.year)), month


def label(year: int, month: int) -> str:
    """The platform's contract label: the month letter and two digits of the year (``Z26``)."""
    return f"{MONTH_CODES[month - 1]}{year % 100:02d}"


def platform_spec(root: str):
    """The platform's own description of ``root`` (its contract terms and expiry rule), or ``None``: the same symbol, or the one Finzdata writes differently (``ALIAS``)."""
    root = clean_root(root)
    return FREE_ROOTS.get(root) or FREE_ROOTS.get(ALIAS.get(root, ""))


def estimate_expiry(root: str, year: int, month: int) -> tuple[pd.Timestamp, str]:
    """``(expiry, basis)``: the exchange's rule for the roots the platform knows, else the 15th of the delivery month (rolled back to a weekday). Without a holiday calendar it can be a day off."""
    spec = platform_spec(root)
    if spec is not None:
        return last_trade_date(spec.rule, year, month).normalize(), "rule"
    day = pd.Timestamp(year, month, 15)
    return (day if day.weekday() < 5 else day - pd.offsets.BDay(1)).normalize(), "15th"


def normalise_contracts(payload, root: str, today=None) -> pd.DataFrame:
    """One row per contract the service lists for ``root``: ``code`` (as sent), ``contract`` (``Z26``), ``year``, ``month``, ``expiry``, ``expiry_basis`` (``service``, ``last bar``, ``rule`` or ``15th``),
    ``first_date`` and ``last_date`` (the first and last bar; ``NaT`` when not sent), soonest expiry first. A row whose code cannot be decoded is left out and listed in ``frame.attrs['undecodable']``."""
    root = clean_root(root)
    today = pd.Timestamp(today if today is not None else datetime.now(timezone.utc).date())
    rows = _lower(rows_of(payload, f"the contracts of {root}"))
    _need(rows, CONTRACT_NAMES, f"the contracts of {root}", "contract code")
    out, bad = [], []
    for r in rows:
        code = _pick(r, CONTRACT_NAMES)
        first, last, given = _date(_pick(r, FIRST_NAMES)), _date(_pick(r, LAST_NAMES)), _date(_pick(r, EXPIRY_NAMES))
        try:
            year, month = decode_contract(code, root, last if last is not pd.NaT else today)
        except ValueError:
            bad.append(str(code))
            continue
        if given is not pd.NaT:
            expiry, basis = given, "service"
        elif last is not pd.NaT and last < today - pd.Timedelta(days=7):
            expiry, basis = last, "last bar"                       # a contract that stopped trading expired about when its bars end
        else:
            expiry, basis = estimate_expiry(root, year, month)
        out.append({"code": str(code), "contract": label(year, month), "year": year, "month": month, "expiry": expiry, "expiry_basis": basis, "first_date": first, "last_date": last})
    frame = pd.DataFrame(out, columns=["code", "contract", "year", "month", "expiry", "expiry_basis", "first_date", "last_date"])
    frame = frame.drop_duplicates("contract").sort_values(["expiry", "contract"]).reset_index(drop=True)
    frame.attrs["undecodable"], frame.attrs["columns"] = bad, _columns(rows)
    return frame


# ------------------------------------------------------------------------------------------------------------------ bars
def normalise_bars(payload, root: str) -> pd.DataFrame:
    """``date, code, price, open, high, low, volume, open_interest`` from an answer of ``/v1/futures/daily`` (``code`` is empty when the rows do not say which contract they are of). The price is the
    settlement when the rows have one, else the close."""
    root = clean_root(root)
    rows = _lower(rows_of(payload, f"the daily bars of {root}"))
    if not rows:
        empty = pd.DataFrame(columns=["date", "code", "price", "open", "high", "low", "volume", "open_interest"])
        empty.attrs["columns"] = []
        return empty
    _need(rows, DATE_NAMES, f"the daily bars of {root}", "date")
    _need(rows, PRICE_NAMES, f"the daily bars of {root}", "price (settle, close or price)")

    def number(names) -> pd.Series:
        return pd.to_numeric(pd.Series([_pick(r, names) for r in rows], dtype=object), errors="coerce")

    frame = pd.DataFrame({"date": [_date(_pick(r, DATE_NAMES)) for r in rows], "code": [None if _pick(r, CONTRACT_NAMES) is None else str(_pick(r, CONTRACT_NAMES)) for r in rows],
                          "price": number(PRICE_NAMES), "open": number(OPEN_NAMES), "high": number(HIGH_NAMES), "low": number(LOW_NAMES), "volume": number(VOLUME_NAMES),
                          "open_interest": number(OI_NAMES)})
    frame = frame.dropna(subset=["date", "price"])
    frame = frame[frame["price"] > 0].reset_index(drop=True)
    frame.attrs["price_column"], frame.attrs["columns"] = next((n for n in PRICE_NAMES if any(n in r for r in rows[:50])), "close"), _columns(rows)
    return frame


# ------------------------------------------------------------------------------------------------------------------ the curve
def curve_frame(bars: pd.DataFrame, contracts: pd.DataFrame, root: str, code: str | None = None) -> pd.DataFrame:
    """The long table the Futures tab reads, ``date, contract, expiry, price, open_interest, volume``, from the bars and the contract list. Bars and list are joined on the contract's label (``Z26``),
    so a code written ``ESZ26`` in one and ``ESZ2026`` in the other is still the same contract. ``code`` names the contract of bars that do not say (a call made for one contract). A bar of a contract
    the list does not have is dropped (its expiry would be a guess)."""
    if bars.empty:
        return pd.DataFrame(columns=CURVE_COLUMNS)
    bars = bars.copy()
    if code is not None:
        bars["code"] = bars["code"].fillna(code)
    if bars["code"].isna().all():
        raise ShapeError("Finzdata's daily bars do not say which contract each row is of (no contract column)")
    labels = {}
    for text, reference in bars.dropna(subset=["code"]).groupby("code")["date"].max().items():
        try:
            labels[text] = label(*decode_contract(text, root, reference))
        except ValueError:
            continue
    bars["contract"] = bars["code"].map(labels)
    bars = bars.dropna(subset=["contract"])
    by_label = contracts.drop_duplicates("contract").set_index("contract")
    keep = bars[bars["contract"].isin(by_label.index)].copy()
    if keep.empty:
        raise ShapeError(f"none of the contracts in the bars ({', '.join(map(str, bars['code'].dropna().unique()[:4]))} ...) is in the contract list "
                         f"({', '.join(contracts['code'].astype(str).head(4))} ...)")
    expiry = keep["contract"].map(by_label["expiry"])
    estimated = keep["contract"].map(by_label["expiry_basis"]) != "service"
    last_bar = keep.groupby("contract")["date"].transform("max")
    dead = last_bar < keep["date"].max() - pd.Timedelta(days=5)
    # An expiry before the last bar is mistaken: the contract traded until then. The tab drops a row dated after its expiry, so a live contract with an estimated date (a rule or the 15th, a day or more
    # off) would vanish from the curve for those days; it is kept, expiring on its last bar. The service's own date for a live contract is left alone.
    keep["expiry"] = expiry.where(~((dead | estimated) & (expiry < last_bar)), last_bar)
    out = keep.assign(volume=keep["volume"].fillna(0.0))[CURVE_COLUMNS]
    return out.drop_duplicates(["date", "contract"], keep="last").sort_values(["date", "expiry", "contract"]).reset_index(drop=True)


def fetch_curve(client, root: str, *, start=None, terms: dict | None = None, today=None, max_calls: int = 60, progress=None) -> Download:
    """The curve of ``root`` from Finzdata, as a :class:`Download`.

    ``client`` has ``contracts(root)`` and ``daily(root, contract=None, start=None, limit=None)`` that return the parsed answers (``webapp.finzdata.FinzdataClient``). The contract list comes first (one
    call). Then one call asks for the bars of **every contract of the root at once** (no ``contract``); if the service answers with rows that say which contract each is of, that is all. If it refuses
    that, or sends rows with no contract, the contracts that traded since ``start`` are asked for one by one, at most ``max_calls`` calls in all (the rest are named in the notes). A refusal of the key,
    the plan or the day's allowance (401, 403, 429) is raised as it is: asking again contract by contract would meet it again."""
    root = clean_root(root)
    today = pd.Timestamp(today if today is not None else datetime.now(timezone.utc).date())
    say = progress or (lambda message: None)
    say(f"{root}: the list of contracts")
    contracts = normalise_contracts(client.contracts(root), root, today)
    if contracts.empty:
        undecodable = contracts.attrs["undecodable"]
        raise FetchError(f"Finzdata lists no contract for {root}" + (f" (it sent codes that could not be read: {', '.join(undecodable[:4])})" if undecodable else ""))
    notes: list[str] = []
    history: list[str] = []
    calls, mode, frames, daily_columns = 1, "every contract in one call", [], []
    say(f"{root}: the bars of {len(contracts)} contracts")
    try:
        calls += 1
        answer = client.daily(root, start=start, limit=ROWS_LIMIT)
        history += notes_of(answer)
        bars = normalise_bars(answer, root)
        daily_columns = daily_columns or bars.attrs["columns"]
        if len(bars) >= ROWS_LIMIT:
            notes.append(f"Finzdata sent {ROWS_LIMIT:,} rows, the most it will in one answer: the oldest bars of some contracts may be missing. Ask for a shorter period.")
        if not bars.empty and bars["code"].notna().any():
            frames.append(curve_frame(bars, contracts, root))
    except ShapeError:
        raise
    except FetchError as exc:
        if exc.status in (401, 403, 429) or getattr(exc, "fatal", False):
            raise
        notes.append(f"Finzdata would not give every contract of {root} at once ({exc}); they were asked for one by one.")
    if not frames or frames[0].empty:
        mode, frames = "one call per contract", []
        horizon = (pd.Timestamp(start) if start is not None else today - pd.Timedelta(days=FREE_WINDOW_DAYS)) - pd.Timedelta(days=5)
        started = contracts["first_date"].isna() | (contracts["first_date"] <= today)                          # a contract whose first bar is still to come has none to give
        unfinished = contracts["last_date"].isna() | (contracts["last_date"] >= horizon)
        wanted = contracts[(contracts["expiry"] >= horizon) & started & unfinished]
        asked, empty = 0, 0
        budget = max(int(max_calls) - calls, 0)
        for c in wanted.itertuples():
            if asked >= budget or empty >= EMPTY_RUN:
                break
            say(f"{root}: {c.contract} ({asked + 1} of {min(len(wanted), budget)})")
            answer = client.daily(root, contract=c.code, start=start, limit=ROWS_LIMIT)
            asked += 1
            calls += 1
            history += notes_of(answer)
            bars = normalise_bars(answer, root)
            daily_columns = daily_columns or bars.attrs["columns"]
            frame = curve_frame(bars, contracts, root, code=c.code) if not bars.empty else pd.DataFrame(columns=CURVE_COLUMNS)
            empty = 0 if len(frame) else empty + 1
            if len(frame):
                frames.append(frame)
        if empty >= EMPTY_RUN and not frames:
            notes.append(f"The first {EMPTY_RUN} contracts asked for had no bar in the period, so the rest were not asked for.")
        elif asked < len(wanted) and empty < EMPTY_RUN:
            skipped = wanted.iloc[asked:]["contract"].tolist()
            notes.append(f"Only {asked} of the {len(wanted)} contracts that traded in the period were asked for, to stay within {max_calls} calls: {', '.join(skipped[:8])}{' ...' if len(skipped) > 8 else ''} are missing. "
                         "Download again to add them.")
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FetchError(f"Finzdata had no daily bars for {root}" + (f": {history[0]}" if history else ""))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "contract"], keep="last").sort_values(["date", "expiry", "contract"]).reset_index(drop=True)
    last = df["date"].max()
    furthest = df.drop_duplicates("contract").sort_values("expiry").iloc[-1]                  # the furthest contract that has a bar, not the furthest one listed
    basis = contracts["expiry_basis"].value_counts().to_dict()
    explained = [f"Expiry dates: {', '.join(f'{n} from {k}' for k, n in basis.items())}. A date from the exchange's rule or the 15th can be a day or more off; the curve only uses it to order the contracts and "
                 "measure the time between them."] if set(basis) - {"service"} else []
    undecodable = contracts.attrs["undecodable"]
    listed_expiry = df["contract"].map(contracts.drop_duplicates("contract").set_index("contract")["expiry"])
    moved = df[df["expiry"] != listed_expiry].drop_duplicates("contract")["contract"].tolist()
    moved_note = [f"{len(moved)} contracts have bars after the expiry they were listed or estimated with ({', '.join(moved[:5])}{' ...' if len(moved) > 5 else ''}), so their expiry is the day of their last bar."] if moved else []
    notes = [f"Finzdata's daily bars of {df['contract'].nunique()} of the {len(contracts)} {root} contracts it lists ({df['date'].min().date()} to {last.date()}), with open interest where it has it. "
             "The price is the close; Finzdata does not document a separate settlement.", *explained, *moved_note, *history, *notes,
             *([f"{len(undecodable)} contract codes could not be read and were left out: {', '.join(undecodable[:5])}."] if undecodable else [])]
    meta = {"incomplete": False, "source_id": SOURCE_ID, "source": SOURCE, "url": f"{BASE_URL}/v1/futures/daily?root={root}", "as_of": str(last.date()), "root": root,
            "contracts": int(df["contract"].nunique()), "contracts_listed": int(len(contracts)), "contracts_on_last_day": int((df["date"] == last).sum()), "first_date": str(df["date"].min().date()),
            "furthest": furthest["contract"], "furthest_expiry": str(pd.Timestamp(furthest["expiry"]).date()), "calls": calls, "mode": mode, "has_open_interest": bool(df["open_interest"].notna().any()),
            "expiry_basis": {str(k): int(v) for k, v in basis.items()},
            "columns": {"contracts": contracts.attrs["columns"], "daily": daily_columns}, **(terms or {}), "notes": notes}
    return Download("futures", root, "curve", frame=df, meta=meta)


# ------------------------------------------------------------------------------------------------------------------ what a thirty-day window leaves out
def gaps(dates, today=None, min_gap_days: int = 4) -> list[dict]:
    """The stretches of the stored history with no bar: ``[{from, to, days}]`` for every two consecutive stored dates more than ``min_gap_days`` apart (a long weekend is four), and one for the time
    between the newest and ``today``. A thirty-day window cannot be asked for again once it has slid past, so these are the days no download caught."""
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates)).dropna().unique()))
    out = [{"from": str(a.date()), "to": str(b.date()), "days": int((b - a).days)} for a, b in zip(idx[:-1], idx[1:]) if (b - a).days > min_gap_days]
    if len(idx) and today is not None and (pd.Timestamp(today).normalize() - idx[-1]).days > min_gap_days:
        out.append({"from": str(idx[-1].date()), "to": str(pd.Timestamp(today).date()), "days": int((pd.Timestamp(today).normalize() - idx[-1]).days), "open": True})
    return out


def staleness(last_date, today=None, window_days: int = FREE_WINDOW_DAYS) -> dict:
    """How many days ago the newest stored bar is (``age``) and how many days are left before a window of ``window_days`` slides past it: ``{age, left, urgent}``."""
    today = pd.Timestamp(today if today is not None else datetime.now(timezone.utc).date()).normalize()
    age = max(int((today - pd.Timestamp(last_date).normalize()).days), 0)
    left = window_days - age
    return {"age": age, "left": left, "urgent": left <= 7}
