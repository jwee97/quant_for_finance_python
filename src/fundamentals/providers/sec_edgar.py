"""SEC EDGAR (https://www.sec.gov/edgar): the financial statements every US-listed company files with the Securities and Exchange Commission, read from the XBRL data the SEC publishes for free.

What it is. Since 2009-2011 (the largest companies first) every filer tags the numbers in its 10-K and 10-Q with XBRL elements, and the SEC serves them per company at
``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``: every fact the company has ever tagged, with the period it covers, the accession number of the filing it appeared in, that
filing's form and the date it was filed. One request returns the whole history, there is no key and no quota, and the figures are as filed. That is what makes it the answer to "how do I get
more years": a US company has about fifteen to seventeen annual statements here, where a provider that sells a short window cannot give them.

What it is not. It covers companies that file with the SEC: US companies, and the foreign companies that list in the US and file annual reports (20-F, 40-F). Nothing is older than the company's
first XBRL filing (2009 for the largest filers, 2011 for the rest); earlier statements exist only as documents. The statements are not normalised by anyone: a company may change the element it uses
for a line from one year to the next, so :data:`canonical.SEC_MAP` lists the elements for each metric in the order they are tried, within one filing. Nothing here is a cross-company standard
beyond that mapping, and a figure the filing does not tag is missing, not guessed.

Point in time. Each fiscal year is the filing in which it *first appeared*, with that filing's real date: the figures of the 10-K for the year, as filed, and nothing from a later filing (a later
10-K repeats the year as a comparative, sometimes restated, and that restatement was not public yet). The filing date is the date the SEC deems it filed, which can be a day after it was accepted;
the point-in-time engine adds its own day on top, so a figure is never used before it was public. A year that first appears as a comparative (the company's first XBRL filing was a later one) is
dated by that later filing: late, never early.

Fair access. The SEC asks every automated client to say who it is and to stay under ten requests a second. The first is the contact in ``SEC_EDGAR_USER_AGENT`` (a name and an email address, for
example ``Jane Doe jane@example.com``): the user supplies it, the application never makes one up, and nothing in the page or the API takes it. The second is a pause of at least
:data:`MIN_INTERVAL` seconds between requests, here. The SEC answers a client over its limit (or one that does not identify itself) with HTTP 403 or 429; a run stops at the first such answer instead of
asking again.

The ticker list. ``company_tickers_exchange.json`` (a file on www.sec.gov) maps tickers to the CIK numbers the data is filed under; it is fetched once, kept next to the database and renewed weekly.
A company can always be named by its CIK instead (``cik:320193``), which needs no list.

Written from the SEC's documentation (https://www.sec.gov/search-filings/edgar-application-programming-interfaces) and tested on the company facts of twelve US companies and on stand-in answers.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import zlib
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request

from .. import canonical
from ..errors import NotFound, PlanError, ProviderError, RateLimited
from ..store import CompanyProfile, PeriodStatement
from .base import FinancialsResult, FundamentalDataProvider

BASE = "https://data.sec.gov"
TICKER_URLS = ("https://www.sec.gov/files/company_tickers_exchange.json", "https://www.sec.gov/files/company_tickers.json")
ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "10-K405", "10-K405/A", "10-KT", "10-KT/A", "10-KSB", "10-KSB/A", "20-F", "20-F/A", "40-F", "40-F/A"})
MIN_INTERVAL = 0.15                          # seconds between two requests: well inside the SEC's ten a second
YEAR_DAYS = (350, 380)                       # a fiscal year, with the 52- and 53-week years
END_TOLERANCE_DAYS = 7                       # period ends of the same year written a few days apart (a balance sheet dated the month end, the flows the last Saturday)
MIN_FIELDS_PER_PERIOD = 3                    # a date is a fiscal year end only when at least this many different elements cover the year ending on it
MAX_BYTES = 96_000_000
TICKER_KEEP_DAYS = 7
_TICKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,11}$")
_CIK = re.compile(r"^(?:CIK)?0*(\d{1,10})$", re.IGNORECASE)
USER_AGENT_VARIABLE = "SEC_EDGAR_USER_AGENT"
COUNTRY = "US"


# ------------------------------------------------------------------------------------------------------------------ the sector of a company, from its SIC code
# The SEC gives each filer a Standard Industrial Classification code and its description (``Electronic Computers``); the description is kept as the industry. The sector is this application's
# own approximation of the eleven usual sectors from the code, first matching rule wins, and is only as exact as that: a code does not say what a company mostly does.
_SECTOR_RULES: tuple[tuple[int, int, str | None], ...] = (
    (9990, 9999, None), (6770, 6770, "Financials"), (6798, 6798, "Real Estate"), (5311, 5311, "Consumer Discretionary"), (5912, 5912, "Consumer Staples"), (5961, 5961, "Consumer Discretionary"),
    (5045, 5045, "Technology"), (5065, 5065, "Technology"), (5047, 5047, "Health Care"), (5122, 5122, "Health Care"), (5172, 5172, "Energy"), (3826, 3826, "Health Care"),
    (3711, 3711, "Consumer Discretionary"), (3714, 3714, "Consumer Discretionary"), (3751, 3751, "Consumer Discretionary"), (2750, 2799, "Industrials"), (8731, 8731, "Health Care"),
    (7310, 7319, "Communication Services"), (7800, 7829, "Communication Services"), (1220, 1221, "Energy"), (4610, 4619, "Energy"),
    (100, 999, "Consumer Staples"), (1000, 1199, "Materials"), (1200, 1399, "Energy"), (1400, 1499, "Materials"), (1500, 1799, "Industrials"), (2000, 2199, "Consumer Staples"),
    (2200, 2399, "Consumer Discretionary"), (2400, 2499, "Materials"), (2500, 2599, "Consumer Discretionary"), (2600, 2699, "Materials"), (2700, 2749, "Communication Services"),
    (2800, 2829, "Materials"), (2830, 2836, "Health Care"), (2840, 2844, "Consumer Staples"), (2850, 2899, "Materials"), (2900, 2999, "Energy"), (3000, 3049, "Consumer Discretionary"),
    (3050, 3099, "Materials"), (3100, 3199, "Consumer Discretionary"), (3200, 3399, "Materials"), (3400, 3569, "Industrials"), (3570, 3579, "Technology"), (3580, 3629, "Industrials"),
    (3630, 3639, "Consumer Discretionary"), (3640, 3649, "Industrials"), (3650, 3659, "Consumer Discretionary"), (3660, 3699, "Technology"), (3700, 3799, "Industrials"),
    (3800, 3819, "Industrials"), (3820, 3829, "Technology"), (3830, 3839, "Industrials"), (3840, 3859, "Health Care"), (3860, 3879, "Consumer Discretionary"), (3900, 3999, "Consumer Discretionary"),
    (4000, 4599, "Industrials"), (4700, 4799, "Industrials"), (4800, 4899, "Communication Services"), (4900, 4999, "Utilities"), (5000, 5199, "Industrials"),
    (5200, 5299, "Consumer Discretionary"), (5300, 5499, "Consumer Staples"), (5500, 5999, "Consumer Discretionary"), (6000, 6499, "Financials"), (6500, 6599, "Real Estate"),
    (6600, 6799, "Financials"), (7000, 7299, "Consumer Discretionary"), (7300, 7369, "Industrials"), (7370, 7379, "Technology"), (7380, 7399, "Industrials"),
    (7500, 7599, "Consumer Discretionary"), (7900, 7999, "Consumer Discretionary"), (8000, 8099, "Health Care"), (8200, 8299, "Consumer Discretionary"), (8700, 8799, "Industrials"),
)


def sector_of(sic) -> str | None:
    """The sector this application files a company under, from its SIC code (a number or the text the SEC sends); ``None`` when the code is missing, unknown or a non-operating one."""
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return None
    for low, high, sector in _SECTOR_RULES:
        if low <= code <= high:
            return sector
    return None


# ------------------------------------------------------------------------------------------------------------------ http
class _NoRedirect(request.HTTPRedirectHandler):
    """Nothing the application asks the SEC for is moved; a redirect is an answer to be told about, not followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = request.build_opener(_NoRedirect)


def _read_body(response) -> bytes:
    """The body of an answer, unzipped when it came gzipped, never more than :data:`MAX_BYTES` once unzipped (a gzip bomb is refused)."""
    raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ProviderError("SEC EDGAR sent an answer larger than this application reads")
    if str(response.headers.get("Content-Encoding", "")).lower() == "gzip":
        unzip = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            body = unzip.decompress(raw, MAX_BYTES + 1)
        except zlib.error:
            raise ProviderError("SEC EDGAR sent an answer that could not be unzipped") from None
        if len(body) > MAX_BYTES or unzip.unconsumed_tail:
            raise ProviderError("SEC EDGAR sent an answer larger than this application reads")
        return body
    return raw


def _urllib_transport(url: str, headers: dict, timeout: float) -> tuple[int, dict, bytes]:
    req = request.Request(url, headers=headers, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as response:                      # noqa: S310  (https only; the address is built here)
            return response.status, dict(response.headers.items()), _read_body(response)
    except urlerror.HTTPError as err:
        try:
            body = _read_body(err)[:200_000]                                          # the SEC's refusal pages are gzipped too
        except ProviderError:
            body = b""
        return err.code, dict(err.headers.items()) if err.headers else {}, body
    except (urlerror.URLError, TimeoutError, OSError) as err:
        raise ProviderError(f"could not reach the SEC ({type(err).__name__}); check the network") from None


def _day(value) -> date | None:
    text = str(value or "")[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _title(text) -> str:
    text = " ".join(str(text or "").split())
    return text.title() if text.isupper() else text


# ------------------------------------------------------------------------------------------------------------------ the ticker list
class TickerList:
    """The SEC's list of tickers with their CIK numbers: ``rows`` of ``(cik, ticker, name, exchange)`` in the order the SEC gives them (the larger companies first)."""

    def __init__(self, rows: list[tuple[int, str, str, str]], fetched: str = ""):
        self.rows, self.fetched = rows, fetched
        self._by_ticker: dict[str, int] = {}
        for n, (_, ticker, _, _) in enumerate(rows):
            self._by_ticker.setdefault(_key(ticker), n)

    @classmethod
    def parse(cls, data, fetched: str = "") -> "TickerList":
        """From either file the SEC publishes: ``company_tickers_exchange.json`` (``{"fields": [...], "data": [[cik, name, ticker, exchange], ...]}``) or ``company_tickers.json``
        (``{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}``). Anything else, and every row that is not a company, is left out."""
        raw: list[tuple] = []
        if isinstance(data, dict) and isinstance(data.get("data"), list) and isinstance(data.get("fields"), list):
            names = [str(f).lower() for f in data["fields"]]
            at = {k: names.index(k) for k in ("cik", "name", "ticker") if k in names}
            if len(at) == 3:
                for item in data["data"]:
                    if isinstance(item, list) and len(item) > max(at.values()):
                        exchange = item[names.index("exchange")] if "exchange" in names and len(item) > names.index("exchange") else ""
                        raw.append((item[at["cik"]], item[at["ticker"]], item[at["name"]], exchange))
        elif isinstance(data, dict):
            for item in data.values():
                if isinstance(item, dict):
                    raw.append((item.get("cik_str"), item.get("ticker"), item.get("title"), ""))
        rows = []
        for cik, ticker, name, exchange in raw:
            if isinstance(cik, bool) or not isinstance(cik, (int, str)) or not str(cik).isdecimal() or not isinstance(ticker, str) or not _TICKER.match(ticker.strip()):
                continue
            rows.append((int(cik), ticker.strip().upper(), " ".join(str(name or "").split())[:200], str(exchange or "").strip()[:20]))
        return cls(rows, fetched)

    def to_json(self) -> dict:
        return {"fetched": self.fetched, "rows": [list(r) for r in self.rows]}

    @classmethod
    def from_json(cls, data) -> "TickerList | None":
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            return None
        rows = []
        for r in data["rows"]:
            if isinstance(r, list) and len(r) == 4 and isinstance(r[0], int) and isinstance(r[1], str) and _TICKER.match(r[1]):
                rows.append((r[0], r[1], str(r[2])[:200], str(r[3])[:20]))
        return cls(rows, str(data.get("fetched") or "")[:10]) if rows else None

    def find(self, ticker: str) -> tuple[int, str, str, str] | None:
        n = self._by_ticker.get(_key(ticker))
        return self.rows[n] if n is not None else None

    def of_cik(self, cik: int) -> list[tuple[int, str, str, str]]:
        return [r for r in self.rows if r[0] == cik]

    def search(self, text: str, limit: int = 10) -> list[tuple[float, tuple[int, str, str, str]]]:
        """Companies matching ``text``: a ticker (exactly or by its start) or the words of a name, best first. One company is one hit however many tickers it has (the first one is shown)."""
        query = " ".join(str(text or "").split())
        needle = _key(query)
        words = [w for w in re.split(r"[^a-z0-9]+", query.lower()) if w]
        if not needle:
            return []
        by_cik: dict[int, tuple[float, int, tuple]] = {}
        for n, row in enumerate(self.rows):
            cik, ticker, name, _ = row
            tk, low = _key(ticker), name.lower()
            score = 0.0
            if query.isdecimal() and int(query) == cik:
                score = 120.0
            elif tk == needle:
                score = 100.0
            elif tk.startswith(needle) and len(needle) >= 2:
                score = 80.0 - min(len(tk) - len(needle), 10)
            elif words and low.startswith(" ".join(words)):
                score = 70.0
            elif words and all(w in low for w in words):
                score = 50.0
            if score and (cik not in by_cik or score > by_cik[cik][0]):
                by_cik[cik] = (score, n, row)
        ranked = sorted(by_cik.values(), key=lambda t: (-t[0], t[1]))
        return [(score, row) for score, _, row in ranked[:limit]]


def _key(ticker: str) -> str:
    """A ticker with the dots and dashes of share classes made alike (``BRK.B`` and ``BRK-B`` are one company)."""
    return re.sub(r"[.\-]", "", str(ticker or "").upper())


# ------------------------------------------------------------------------------------------------------------------ reading company facts
class _Fact:
    __slots__ = ("start", "end", "value", "accn", "filed", "form", "fy", "unit")

    def __init__(self, start, end, value, accn, filed, form, fy, unit):
        self.start, self.end, self.value, self.accn, self.filed, self.form, self.fy, self.unit = start, end, value, accn, filed, form, fy, unit


def _is_year(fact: _Fact) -> bool:
    return fact.start is not None and YEAR_DAYS[0] <= (fact.end - fact.start).days <= YEAR_DAYS[1]


def _read_facts(payload: dict) -> tuple[dict[str, list[_Fact]], str]:
    """``({source field: facts}, currency)``: the facts the mapping reads, from annual reports only, and the currency the company reports in (the money unit most facts are in)."""
    taxonomies = payload.get("facts") if isinstance(payload.get("facts"), dict) else {}
    found: dict[str, list[_Fact]] = {}
    currencies: Counter = Counter()
    for field in canonical.SEC_FIELDS:
        taxonomy, _, element = field.partition(":")
        entry = (taxonomies.get(taxonomy) or {}).get(element) if isinstance(taxonomies.get(taxonomy), dict) else None
        units = entry.get("units") if isinstance(entry, dict) and isinstance(entry.get("units"), dict) else {}
        facts = []
        for unit, items in units.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or item.get("form") not in ANNUAL_FORMS:
                    continue
                value, end, filed, accn = item.get("val"), _day(item.get("end")), str(item.get("filed") or "")[:10], item.get("accn")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or end is None or not accn or _day(filed) is None:
                    continue
                fy = item.get("fy")
                facts.append(_Fact(_day(item.get("start")) if item.get("start") else None, end, float(value), str(accn), filed, item["form"], fy if isinstance(fy, int) and not isinstance(fy, bool) else None, str(unit)))
                if re.fullmatch(r"[A-Z]{3}", str(unit)) and field.split(":")[1] in ("Assets", "Revenues", "NetIncomeLoss", "Revenue", "ProfitLoss"):
                    currencies[str(unit)] += 1
        if facts:
            found[field] = facts
    currency = "USD" if not currencies or currencies.get("USD", 0) * 2 >= sum(currencies.values()) else currencies.most_common(1)[0][0]
    return found, currency


def _unit_ok(unit: str, currency: str) -> bool:
    return unit in (currency, f"{currency}/shares", "shares")


def _closest(facts: list[_Fact], target: date, kind: str, currency: str) -> _Fact | None:
    """The fact of ``kind`` (a year-long ``duration``, an ``instant``) nearest ``target``, in this currency; of equal ones the last listed (the SEC lists a restated value after the first)."""
    best, best_key = None, None
    for f in facts:
        if not _unit_ok(f.unit, currency):
            continue
        if kind == "duration":
            if not _is_year(f):
                continue
            key = (abs((f.end - target).days), abs((f.end - f.start).days - 365))
        elif kind == "instant":
            if f.start is not None:
                continue
            key = (abs((f.end - target).days), 0)
        else:
            key = (-f.end.toordinal(), 0)
        if kind != "cover" and key[0] > END_TOLERANCE_DAYS:
            continue
        if best_key is None or key <= best_key:
            best, best_key = f, key
    return best


def _build_synthetic(values: dict) -> None:
    """The figures the filing has no single element for, from lines of the same filing: borrowings due within a year, cash with short-term investments, and total liabilities (assets less all equity)."""
    for name, (needed, groups) in canonical.SEC_SUMS.items():
        parts = []
        for group in groups:
            hit = next((values[f] for f in group if f in values), None)
            if hit is not None:
                parts.append(hit)
        if len(parts) >= needed and name not in values:
            values[name] = float(sum(parts))
    if "us-gaap:Assets" in values:
        total = values.get("us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
        if total is None and "us-gaap:StockholdersEquity" in values:
            total = values["us-gaap:StockholdersEquity"] + values.get("us-gaap:MinorityInterest", 0.0)
        if total is not None:
            values[canonical.SEC_LIABILITIES] = values["us-gaap:Assets"] - total
    elif "ifrs-full:Assets" in values and "ifrs-full:Equity" in values:
        values[canonical.SEC_LIABILITIES] = values["ifrs-full:Assets"] - values["ifrs-full:Equity"]


_MAPPED = frozenset(source for options in canonical.SEC_MAP.values() for source, _ in options)           # the elements (and made-up sums) that stand for a metric; the parts of a sum do not


def parse_companyfacts(payload: dict) -> tuple[list[PeriodStatement], dict]:
    """``(periods, info)`` from the SEC's company-facts answer (or the trimmed copy of it this module keeps, which has the same shape).

    One :class:`PeriodStatement` per fiscal year, built only from the filing in which that year first appeared (see the module notes): every fact in it belongs to one filing, so every figure was public
    on the filing date. ``info`` holds the company's name and CIK, the currency, and a line per period for the stored answer.
    """
    facts, currency = _read_facts(payload if isinstance(payload, dict) else {})
    # the fiscal years each annual report covers: a date is a year end of a report when several elements cover a whole year ending on it
    ends: dict[str, dict[date, set]] = defaultdict(lambda: defaultdict(set))
    meta: dict[str, tuple[str, str]] = {}
    for field, items in facts.items():
        if canonical.SEC_FIELDS[field] != "duration":
            continue
        for f in items:
            if _is_year(f) and _unit_ok(f.unit, currency):
                ends[f.accn][f.end].add(field)
                meta[f.accn] = (f.filed, f.form)
    reports: dict[str, list[date]] = {}
    for accn, by_end in ends.items():
        strong = [e for e in sorted(by_end, key=lambda e: -len(by_end[e])) if len(by_end[e]) >= MIN_FIELDS_PER_PERIOD]
        kept: list[date] = []
        for e in strong:
            if all(abs((e - k).days) > END_TOLERANCE_DAYS for k in kept):
                kept.append(e)
        if kept:
            reports[accn] = sorted(kept)
    summary = {"currency": currency, "name": str(payload.get("entityName") or "")[:200] if isinstance(payload, dict) else "", "cik": payload.get("cik") if isinstance(payload, dict) else None, "lines": []}
    if not reports:
        return [], summary
    by_accn: dict[str, dict[str, list[_Fact]]] = defaultdict(lambda: defaultdict(list))
    for field, items in facts.items():
        for f in items:
            if f.accn in reports:
                by_accn[f.accn][field].append(f)
    # each fiscal year is the earliest report that has it
    first: dict[date, str] = {}
    for accn in sorted(reports, key=lambda a: (meta[a][0], a)):
        for e in reports[accn]:
            if all(abs((e - k).days) > END_TOLERANCE_DAYS for k in first):
                first[e] = accn
    # the fiscal year is numbered the way the company numbers it now (the SEC's own label of its latest report), counted back from there by whole years: a company that renamed its years
    # (a retailer whose year ending in January was "2008" and is now "2009") would otherwise have two years with one number
    latest_accn = max(reports, key=lambda a: (max(reports[a]), meta[a][0]))
    latest_end = max(reports[latest_accn])
    labels = Counter(f.fy for items in by_accn[latest_accn].values() for f in items if f.end == latest_end and f.fy is not None)
    latest_year = labels.most_common(1)[0][0] if labels else latest_end.year
    if abs(latest_year - latest_end.year) > 1:
        latest_year = latest_end.year
    periods: list[PeriodStatement] = []
    lines: list[dict] = []
    for end in sorted(first):
        accn = first[end]
        report_end = min(reports[accn], key=lambda e: abs((e - end).days))
        values: dict[str, float] = {}
        starts: Counter = Counter()
        for field, kind in canonical.SEC_FIELDS.items():
            fact = _closest(by_accn[accn].get(field, []), report_end, kind, currency)
            if fact is not None:
                values[field] = fact.value
                if fact.start is not None:
                    starts[fact.start] += 1
        _build_synthetic(values)
        values = {field: value for field, value in values.items() if field in _MAPPED}
        if not values:
            continue
        fiscal_year = latest_year - round((latest_end - report_end).days / 365.25)
        taxonomy = "IFRS" if sum(k.startswith("ifrs-full:") for k in values) > sum(k.startswith("us-gaap:") for k in values) else "US GAAP"
        filed, form = meta[accn]
        periods.append(PeriodStatement("FY", report_end.isoformat(), values, fiscal_year=fiscal_year, period_start=starts.most_common(1)[0][0].isoformat() if starts else None, filing_date=filed,
                                       reporting_currency=currency, accounting_standard=taxonomy, provider_record_id=accn))
        lines.append({"fiscal_year": fiscal_year, "fiscal_period": "FY", "period_end": report_end.isoformat(), "filing_date": filed, "form": form, "accession": accn})
    summary["lines"] = lines
    return periods, summary


def trim_companyfacts(payload: dict) -> dict:
    """The part of a company-facts answer this application uses, in the same shape: the elements in the mapping, from annual reports, every column of each fact kept. What is stored as the raw answer
    (the full answer is megabytes of facts nothing here reads), and enough to read the periods again without asking the SEC."""
    taxonomies = payload.get("facts") if isinstance(payload.get("facts"), dict) else {}
    out: dict = {}
    for field in canonical.SEC_FIELDS:
        taxonomy, _, element = field.partition(":")
        entry = taxonomies.get(taxonomy, {}).get(element) if isinstance(taxonomies.get(taxonomy), dict) else None
        units = entry.get("units") if isinstance(entry, dict) and isinstance(entry.get("units"), dict) else {}
        kept = {unit: [item for item in items if isinstance(item, dict) and item.get("form") in ANNUAL_FORMS] for unit, items in units.items() if isinstance(items, list)}
        kept = {unit: items for unit, items in kept.items() if items}
        if kept:
            out.setdefault(taxonomy, {})[element] = {"units": kept}
    return {"cik": payload.get("cik"), "entityName": payload.get("entityName"), "facts": out}


# ------------------------------------------------------------------------------------------------------------------ the provider
def _profile(ticker: str, name: str = "", currency: str | None = None, sector: str | None = None, industry: str | None = None, lei: str | None = None, extra: dict | None = None) -> CompanyProfile:
    return CompanyProfile("sec_edgar", f"us:{ticker}", ticker, name, exchange="US", market="US", country=COUNTRY, currency=currency, sector=sector, industry=industry, lei=lei, extra=extra or {})


class SecEdgarProvider(FundamentalDataProvider):
    name = "sec_edgar"
    kind = "http"
    as_reported = True
    has_filing_dates = True
    keyed = False
    meters_itself = True
    periods_available = ("annual",)
    default_market = "us"
    has_lookup = True
    price_note = ("SEC EDGAR sends a company's whole history in one free request, from the company's first XBRL filing (2009 for the largest US companies, 2011 for the rest); "
                  "earlier statements exist only as documents.")
    early_note = ("SEC EDGAR's structured data starts with a company's first XBRL filing (2009 for the largest US companies, 2011 for the rest); earlier statements exist only as documents, "
                  "so nothing older can be read here.")

    def __init__(self, pool=None, transport=None, timeout: float = 60.0, sleep=time.sleep, clock=time.monotonic, user_agent: str | None = None, cache_path: str | Path | None = None,
                 base_url: str = BASE, ticker_urls: tuple = TICKER_URLS, today=None):
        self.pool = pool
        self.transport = transport or _urllib_transport
        self.timeout, self.sleep, self.clock = timeout, sleep, clock
        self.user_agent = " ".join(str(user_agent or "").split())
        self.cache_path = Path(cache_path) if cache_path else None
        self.base_url = base_url.rstrip("/")
        self.ticker_urls = tuple(ticker_urls)
        self._today = today or (lambda: datetime.now(timezone.utc).date())
        self._last = -1e9
        self._lock = threading.Lock()
        self._tickers: TickerList | None = None
        self.notes: list[str] = []

    # ------------------------------------------------------------------------------------------------------------ availability
    def available(self) -> tuple[bool, str]:
        ua = self.user_agent
        if not ua:
            return False, ("the SEC asks every program that reads EDGAR to say who runs it: set SEC_EDGAR_USER_AGENT to your name and an email address (for example 'Jane Doe jane@example.com') "
                           "as a secret (on Replit: Tools > Secrets), then restart the app. Nothing else is needed: there is no key and no limit")
        if len(ua) < 8 or len(ua) > 250 or any(ord(c) < 32 or ord(c) == 127 for c in ua):
            return False, "SEC_EDGAR_USER_AGENT must be 8 to 250 characters of plain text: a name and an email address"
        return True, ""

    def describe(self) -> dict:
        out = super().describe()
        out.update({"free": True, "contact": bool(self.user_agent), "contact_has_address": "@" in self.user_agent, "markets": ["us"]})
        return out

    def cost(self, request: str, period: str = "annual") -> int:
        return 0 if request == "search" else 1

    # ------------------------------------------------------------------------------------------------------------ http
    def _pace(self) -> None:
        with self._lock:
            wait = MIN_INTERVAL - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = self.clock()

    def _fetch(self, url: str):
        """One GET with the contact in the User-Agent, paced, retried twice on a server error; the JSON answer. A refusal (403, 429) is not retried."""
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)
        headers = {"User-Agent": self.user_agent, "Accept": "application/json", "Accept-Encoding": "gzip"}
        for attempt in range(3):
            self._pace()
            status, _, body = self.transport(url, headers, self.timeout)
            if status >= 500 and attempt < 2:
                self.sleep(1.0 + 2.0 * attempt)
                continue
            return self._interpret(status, body)
        raise ProviderError("SEC EDGAR is temporarily unavailable; try again in a few minutes")           # unreachable: the loop returns or raises

    @staticmethod
    def _interpret(status: int, body: bytes):
        if status == 200:
            try:
                return json.loads(body.decode("utf-8", "replace"))
            except ValueError:
                raise ProviderError("SEC EDGAR answered with something that is not JSON") from None
        text = body[:4000].decode("utf-8", "replace") if body else ""
        if status == 404:
            raise NotFound("SEC EDGAR has no XBRL financial data for this company: it may not file with the SEC, or be too small or too new to have filed with XBRL")
        if status == 429 or (status == 403 and "rate threshold" in text.lower()):
            raise RateLimited("SEC EDGAR says this network sent too many requests (its limit is ten a second for everything on one address): wait ten minutes and try again", 600.0, "burst")
        if status == 403:
            raise ProviderError("SEC EDGAR refused the request (HTTP 403). It requires the contact in SEC_EDGAR_USER_AGENT (a name and an email address) and refuses a program it cannot identify")
        if status >= 500:
            raise ProviderError(f"SEC EDGAR is temporarily unavailable (HTTP {status}); try again in a few minutes")
        raise ProviderError(f"SEC EDGAR answered HTTP {status}")

    def _get(self, url: str, endpoint: str, symbol: str = ""):
        if self.pool is None:
            return self._fetch(url)
        return self.pool.call(lambda _key: self._fetch(url), endpoint=endpoint, symbol=symbol, market="us", credits=1)

    def test(self) -> dict:
        """One request for a small answer (Apple's filing index): proves the contact is accepted and the network reaches the SEC."""
        self._get(f"{self.base_url}/submissions/CIK0000320193.json", "test", "cik:320193")
        return {"ok": True, "plan": None, "markets": ["us"]}

    # ------------------------------------------------------------------------------------------------------------ the ticker list
    def tickers(self, refresh: bool = False) -> TickerList:
        """The ticker list: the saved copy while it is under a week old, else a fresh one from the SEC (a stale copy is used, and said to be, when the SEC cannot be reached)."""
        today = self._today()
        if self._tickers is not None and not refresh:
            return self._tickers
        saved = None
        if self.cache_path is not None:
            try:
                saved = TickerList.from_json(json.loads(self.cache_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                saved = None
        if saved is not None and not refresh:
            day = _day(saved.fetched)
            if day is not None and 0 <= (today - day).days < TICKER_KEEP_DAYS:
                self._tickers = saved
                return saved
        error = None
        for url in self.ticker_urls:
            try:
                fresh = TickerList.parse(self._get(url, "tickers", "company_tickers"), today.isoformat())
            except RateLimited:
                raise                                                                     # asked to be left alone: the other file would only be refused too
            except ProviderError as err:
                error = err
                continue
            if fresh.rows:
                if self.cache_path is not None:
                    try:
                        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                        self.cache_path.write_text(json.dumps(fresh.to_json(), separators=(",", ":")), encoding="utf-8")
                    except OSError:
                        self.notes.append("the SEC's ticker list could not be saved next to the database")
                self._tickers = fresh
                return fresh
        if saved is not None:
            self.notes.append("the SEC's ticker list could not be renewed; the copy from " + (saved.fetched or "an earlier day") + " is used")
            self._tickers = saved
            return saved
        raise ProviderError("the SEC's list of tickers could not be read" + (f" ({error})" if error else "") + ". A company can still be added by its CIK number, for example cik:320193")

    # ------------------------------------------------------------------------------------------------------------ companies
    def _split(self, symbol: str, market: str | None = None) -> tuple[str, int | None]:
        """``(ticker, cik)`` from ``us:AAPL``, ``AAPL``, ``cik:320193`` or ``us:CIK0000320193``: ``cik`` is set only when the text named one."""
        text = str(symbol).strip()
        head, sep, tail = text.partition(":")
        if sep and head and tail:
            if head.strip().lower() == "cik":
                found = _CIK.match(tail.strip())
                if not found or int(found.group(1)) == 0:
                    raise ProviderError(f"{text!r} is not a CIK number (digits, for example cik:320193)")
                cik = int(found.group(1))
                return f"CIK{cik:010d}", cik
            if head.strip().lower() != "us":
                raise ProviderError(f"SEC EDGAR covers companies that file with the US SEC (us:{tail.strip()}); for {head.strip().lower()} use another provider")
            text = tail.strip()
        elif market and str(market).strip().lower() not in ("us", ""):
            raise ProviderError(f"SEC EDGAR covers companies that file with the US SEC, not market {str(market).strip().lower()!r}")
        found = _CIK.match(text)
        if found and (text.upper().startswith("CIK") or text.isdecimal()):
            cik = int(found.group(1))
            return f"CIK{cik:010d}", cik
        if not _TICKER.match(text):
            raise ProviderError(f"{text!r} is not a ticker: letters and digits, for example AAPL or BRK-B")
        return text.upper(), None

    def _kept_tickers(self) -> TickerList | None:
        """The ticker list if it is in memory or saved beside the database, whatever its age: nothing is asked of the SEC."""
        if self._tickers is not None:
            return self._tickers
        if self.cache_path is not None:
            try:
                return TickerList.from_json(json.loads(self.cache_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                return None
        return None

    def normalise_symbol(self, symbol: str, market: str | None = None) -> str:
        ticker, cik = self._split(symbol, market)
        known = self._kept_tickers() if cik is not None else None                              # a CIK is known by its ticker when the list is at hand
        row = next(iter(known.of_cik(cik)), None) if known is not None else None
        return f"us:{row[1]}" if row is not None else f"us:{ticker}"

    def offline_profile(self, symbol: str, market: str | None = None) -> CompanyProfile:
        return _profile(self.normalise_symbol(symbol, market).partition(":")[2])

    def _cik_of(self, symbol: str, market: str | None = None) -> int:
        ticker, cik = self._split(symbol, market)
        if cik is not None:
            return cik
        match = _CIK.match(ticker)
        if ticker.startswith("CIK") and match:
            return int(match.group(1))
        row = self.tickers().find(ticker)
        if row is None:
            raise NotFound(f"the SEC's ticker list has no company with the ticker {ticker}: check the spelling, or use its CIK (cik:320193)")
        return row[0]

    def search_plan(self, text: str, kind: str = "auto", market: str | None = None) -> dict:
        text = " ".join(str(text or "").split())
        if kind not in ("auto", "name", "symbol"):
            raise ProviderError("kind must be auto, name or symbol: the SEC's list has no ISIN or LEI")
        if kind == "auto":
            kind = "symbol" if _TICKER.match(text) and (text.isupper() or any(c.isdigit() for c in text) or "." in text) and len(text) <= 6 else "name"
        return {"kind": kind, "value": text, "market": "us", "params": {}, "alternatives": ["name"] if kind == "symbol" else []}

    def search_detail(self, query: str, limit: int = 10, kind: str = "auto", market: str | None = None) -> dict:
        plan = self.search_plan(query, kind, market)
        hits = self.tickers().search(plan["value"], limit)
        profiles = []
        for score, (cik, ticker, name, exchange) in hits:
            also = [r[1] for r in self.tickers().of_cik(cik) if r[1] != ticker][:6]
            profiles.append(_profile(ticker, name, extra={"also_listed": [f"us:{t}" for t in also], "score": score, "registry": "sec", "hint": exchange, "primary": True, "has_financials": True,
                                                         "cik": cik}))
        message = "" if profiles else "the SEC's ticker list has no company with that name or ticker (it lists the companies that file with the US SEC)"
        return {**{k: plan[k] for k in ("kind", "value", "market", "alternatives")}, "profiles": profiles, "message": message, "ambiguous": len(profiles) > 1, "markets_skipped": [], "credits": 0}

    def search(self, query: str, limit: int = 10, kind: str = "auto", market: str | None = None) -> list[CompanyProfile]:
        return self.search_detail(query, limit, kind, market)["profiles"]

    def get_company(self, symbol: str, market: str | None = None) -> CompanyProfile:
        ticker, _ = self._split(symbol, market)
        cik = self._cik_of(symbol, market)
        data = self._get(f"{self.base_url}/submissions/CIK{cik:010d}.json", "lookup", f"us:{ticker}")
        data = data if isinstance(data, dict) else {}
        lei = data.get("lei") if isinstance(data.get("lei"), str) and len(data["lei"]) == 20 else None
        extra = {"cik": cik, "sic": str(data.get("sic") or ""), "exchanges": [str(e) for e in (data.get("exchanges") or []) if isinstance(e, str)][:4],
                 "tickers": [str(t) for t in (data.get("tickers") or []) if isinstance(t, str)][:8], "fiscal_year_end": str(data.get("fiscalYearEnd") or ""), "category": str(data.get("category") or "")}
        return _profile(ticker, " ".join(str(data.get("name") or "").split())[:200], None, sector_of(data.get("sic")), _title(data.get("sicDescription")) or None, lei, extra)

    def get_financials(self, symbol: str, market: str | None = None, start_year: int | None = None, end_year: int | None = None, period: str = "annual",
                       limit: int | None = None) -> FinancialsResult:
        if period != "annual":
            raise PlanError("SEC EDGAR quarterly figures are not read by this application yet; the annual statements are")
        ticker, _ = self._split(symbol, market)
        cik = self._cik_of(symbol, market)
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik:010d}.json"
        data = self._get(url, "financials", f"us:{ticker}")
        if not isinstance(data, dict):
            raise ProviderError("SEC EDGAR answered with something that is not a set of company facts")
        periods, info = parse_companyfacts(data)
        trimmed = trim_companyfacts(data)
        trimmed["periods"] = info["lines"]                                                    # the periods read, so that the stored answer says what it held (the coverage view reads this)
        asked = {"cik": cik, "fiscal_year_from": int(start_year) if start_year else None, "fiscal_year_to": int(end_year) if end_year else None, "fiscal_period": "fy", "message": ""}
        if not periods:
            error = NotFound(f"SEC EDGAR has facts for CIK {cik} but no annual statement with figures in them (it may report only on forms this application does not read)")
            error.raw, error.charged, error.request = trimmed, 1, asked
            raise error
        earliest = min(p.fiscal_year for p in periods if p.fiscal_year is not None)
        kept = [p for p in periods if (start_year is None or (p.fiscal_year or 0) >= int(start_year)) and (end_year is None or (p.fiscal_year or 0) <= int(end_year))]
        profile = _profile(ticker, info["name"], info["currency"])
        years = [p.fiscal_year for p in kept if p.fiscal_year is not None]
        answer = {"entries": len(kept), "labels": {"FY": len(kept)} if kept else {}, "annual": len(kept), "first_year": min(years) if years else None, "last_year": max(years) if years else None, "echoed": {}}
        return FinancialsResult(kept, trimmed, asked, profile, answer=answer, exhausted=start_year is None or earliest > int(start_year), requests=1, cost=1)
