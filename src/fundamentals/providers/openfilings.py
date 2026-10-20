"""OpenFilings (https://openfilings.org): as-filed financial statements for listed companies in about twenty-five markets, read from each regulator's own filings, through its hosted REST API.

What it is. A key-authenticated service (``X-API-Key: of_...``, base ``https://api.openfilings.org``). One call to ``GET /api/v1/companies/financials`` returns a company's whole multi-year series: for
each fiscal period its end date, the *filing date* (when the figures became public), the reporting currency and a ``values`` map of canonical line items (``revenue``, ``net_income``, ``total_assets``,
``capex``, ...). The figures are as filed, not restated, which is exactly what a point-in-time study needs. Markets with live filings: us jp hk tw kr cn eu fr uk br mx au in sg ae sa il ca th my id
ph vn tr ch (``eu`` for the pan-European ESEF filers); the others in its directory have no financials.

What it costs. Every call is metered in **credits** against a daily and a monthly allowance shared by the REST API and the MCP server: a search or lookup costs 1, a financials series 10. The free
plan is 100 credits a day and 1,000 a month, so about nine companies a day (a lookup and a series each), which is why the application keeps everything it downloads and asks again only for what is
missing or new. A failed call is refunded. Daily allowances reset at 00:00 UTC, monthly ones on the 1st; prepaid top-up credits never reset. When the allowance is spent the API answers HTTP 429
(``quota_exceeded``, with the dimension and, in the body, the day's and month's counts) or, once the included credits are gone and the balance cannot cover a call, HTTP 402 (``credits_exhausted``).

Several keys. See :mod:`keys`: the first valid key is used, the others are standbys for a revoked key, and the pool never rotates keys to get past a quota (OpenFilings' usage policy forbids it).

What is not supported yet, and why. Annual statements only: the documentation does not list the ``fiscal_period`` values for quarterly or half-year figures, and a guess would spend 10 credits per try.
The sign of cash outflows is not documented either, so the outflow lines (capital expenditure, dividends, buybacks, acquisitions, interest) are read as magnitudes. Some filings in Japan and Korea are
collected on demand: the first request can answer HTTP 202 ("still backfilling"), which is treated as "ask again later", not as a failure.

Written from the published OpenAPI description and documentation (and the official Python client's published source for its error handling) and tested against stand-in answers of the documented
shape; it has not been run against a live key.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from urllib import error as urlerror
from urllib import parse, request

from ...webapp.providers import redact
from .. import canonical
from ..errors import CreditsExhausted, InvalidKey, MissingKey, NotFound, PlanError, ProviderError, RateLimited, SyncPending
from ..keys import KeyPool
from ..store import CompanyProfile, PeriodStatement
from .base import FinancialsResult, FundamentalDataProvider

BASE = "https://api.openfilings.org"
USER_AGENT = "QuantLab/1.0 (fundamentals client)"
CREDITS = {"markets": 1, "search": 1, "lookup": 1, "financials": 10}
MARKETS_WITH_FILINGS = ("us", "jp", "hk", "tw", "kr", "cn", "eu", "fr", "uk", "br", "mx", "au", "in", "sg", "ae", "sa", "il", "ca", "th", "my", "id", "ph", "vn", "tr", "ch")
COUNTRY = {"uk": "GB", "eu": None}                                             # market ids that are not ISO country codes (the issuer's country then comes from its ISIN)
SEARCH_KINDS = ("auto", "name", "symbol", "isin", "lei")
_ISIN = re.compile(r"^[A-Za-z]{2}[A-Za-z0-9]{9}[0-9]$")
_LEI = re.compile(r"^[A-Za-z0-9]{18}[0-9]{2}$")
_MARKET_PREFIX = re.compile(r"^([A-Za-z]{2,4}):\s*(\S+)$")
_TICKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$")
BURST_RETRIES = 2
MAX_BURST_WAIT = 5.0
DEFAULT_HISTORY_YEARS = 25


class _NoRedirect(request.HTTPRedirectHandler):
    """The API key is a header; a redirect would carry it to wherever it points. None is expected, so none is followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = request.build_opener(_NoRedirect)


def _urllib_transport(url: str, headers: dict, timeout: float) -> tuple[int, dict, bytes]:
    req = request.Request(url, headers=headers, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as response:                      # noqa: S310  (https only; the address is built here)
            return response.status, dict(response.headers.items()), response.read(32_000_000)
    except urlerror.HTTPError as err:
        return err.code, dict(err.headers.items()) if err.headers else {}, err.read(200_000)
    except (urlerror.URLError, TimeoutError, OSError) as err:
        raise ProviderError(f"could not reach OpenFilings ({type(err).__name__}); check the network") from None


def _day(value) -> str | None:
    text = str(value or "")[:10]
    return text if len(text) == 10 and text[4] == "-" and text[7] == "-" else None


def split_symbol(symbol: str, market: str | None = None, default_market: str | None = "us") -> tuple[str, str]:
    """``(market_id, ticker)`` from ``us:AAPL``, ``AAPL`` with a market, or ``AAPL`` alone (the default market)."""
    text = str(symbol).strip()
    if ":" in text:
        head, _, tail = text.partition(":")
        if head and tail:
            return head.strip().lower(), tail.strip()
    chosen = (market or default_market or "").strip().lower()
    if not chosen:
        raise ProviderError(f"say which market {text!r} trades in (for example us:{text} or jp:{text})")
    return chosen, text


def search_plan(text: str, kind: str = "auto", market: str | None = None) -> dict:
    """What a search for ``text`` asks OpenFilings, and what else the text could have meant.

    The directory is searched by ``query`` (a company name), ``ticker`` (an exchange symbol, with ``market_id`` to say which exchange), ``isin`` or ``lei``. With ``kind`` left at ``auto`` the text
    decides, and only when its shape is plain: ``us:NVDA`` is a ticker on a market; two letters, nine letters or digits and a digit are an ISIN; eighteen letters or digits and two digits a LEI; one
    short word typed in capitals (NVDA) or holding a digit or a dot (7203, BRK.B) is a ticker; anything else (Nvidia, Toyota Motor) is a name. A short single word could be either, so the other
    reading is listed in ``alternatives`` and is never tried unasked: every try uses a credit.
    """
    text = " ".join(str(text or "").split())
    if kind not in SEARCH_KINDS:
        raise ProviderError(f"kind must be one of {', '.join(SEARCH_KINDS)}")
    mkt = str(market).strip().lower() if market else None
    if mkt is not None and not re.match(r"^[a-z]{2,4}$", mkt):
        raise ProviderError("market is a short code such as us, jp or uk")
    value = text
    prefixed = _MARKET_PREFIX.match(text)
    if kind == "auto":
        if prefixed:
            kind, mkt, value = "symbol", prefixed.group(1).lower(), prefixed.group(2)
        elif _ISIN.match(text):
            kind = "isin"
        elif _LEI.match(text):
            kind = "lei"
        elif _TICKER.match(text) and (any(c.isdigit() for c in text) or "." in text or (text.isalpha() and text.isupper() and len(text) <= 5)):
            kind = "symbol"
        else:
            kind = "name"
    elif kind == "symbol" and prefixed:
        mkt, value = prefixed.group(1).lower(), prefixed.group(2)
    if kind in ("isin", "lei", "symbol"):
        value = value.upper()
    params = {"query": value, "market_id": mkt} if kind == "name" else {"ticker": value, "market_id": mkt} if kind == "symbol" else {kind: value}
    single_word = " " not in text and bool(_TICKER.match(value))
    alternatives = ["symbol"] if kind == "name" and single_word and len(text) <= 8 else ["name"] if kind == "symbol" and not value.replace(".", "").isdigit() else []
    return {"kind": kind, "value": value, "market": mkt, "params": params, "alternatives": alternatives}


def _search_profiles(data: dict, market: str | None = None) -> list[CompanyProfile]:
    """The companies in a search answer: ``entity`` (the best match) and ``entities`` (the others when the answer is ambiguous), each with its listings. One company is one row, shown under
    its primary listing (or the one on the market asked for); the other listings are kept in ``also_listed`` so the same company is not added twice by accident."""
    out: list[CompanyProfile] = []
    seen: set[str] = set()
    for item in [data.get("entity")] + list(data.get("entities") or []):
        if isinstance(item, dict) and "listings" not in item and isinstance(item.get("entity"), dict):
            item = item["entity"]
        if not isinstance(item, dict):
            continue
        listings = [x for x in (item.get("listings") or []) if isinstance(x, dict) and x.get("market_id") and x.get("ticker")]
        if not listings and isinstance(item.get("listing"), str) and ":" in item["listing"]:
            head, _, tail = item["listing"].partition(":")
            listings = [{"market_id": head, "ticker": tail, "primary": True}] if head and tail else []
        if not listings:
            continue
        listings.sort(key=lambda x: (not (market and str(x["market_id"]).lower() == market), not x.get("primary")))
        chosen = listings[0]
        market_id, ticker = str(chosen["market_id"]).lower(), str(chosen["ticker"]).strip().upper()
        symbol = f"{market_id}:{ticker}"
        if symbol in seen:
            continue
        seen.add(symbol)
        names = [n for n in (item.get("display_names") or []) if isinstance(n, str) and n.strip()]
        name = str(chosen.get("company_name") or (names[0] if names else "")).strip()[:200]
        also = [f"{str(x['market_id']).lower()}:{str(x['ticker']).strip().upper()}" for x in listings[1:7]]
        score = chosen.get("score")
        extra = {"also_listed": [a for a in also if a != symbol], "score": score if isinstance(score, (int, float)) else None, "registry": str(chosen.get("registry") or "")[:40],
                 "hint": str(chosen.get("hint") or "")[:200], "primary": bool(chosen.get("primary")), "has_financials": market_id in MARKETS_WITH_FILINGS}
        out.append(_profile(market_id, ticker, name, chosen.get("isin") or item.get("isin"), chosen.get("lei") or item.get("lei"), extra=extra))
    return out


def _country(market_id: str, isin: str | None) -> str | None:
    if market_id in COUNTRY:
        fixed = COUNTRY[market_id]
        if fixed:
            return fixed
        return isin[:2].upper() if isin and len(isin) >= 2 and isin[:2].isalpha() else None
    return market_id.upper()


def _profile(market_id: str, ticker: str, name: str = "", isin: str | None = None, lei: str | None = None, currency: str | None = None, sector: str | None = None,
             industry: str | None = None, extra: dict | None = None) -> CompanyProfile:
    return CompanyProfile("openfilings", f"{market_id}:{ticker}", ticker, name, exchange=market_id.upper(), market=market_id.upper(), country=_country(market_id, isin), currency=currency,
                          sector=sector, industry=industry, isin=isin or None, lei=lei or None, extra=extra or {})


class OpenFilingsProvider(FundamentalDataProvider):
    name = "openfilings"
    kind = "http"
    as_reported = True
    has_filing_dates = True
    keyed = True
    periods_available = ("annual",)
    default_market = "us"
    has_lookup = True

    def __init__(self, pool: KeyPool | None = None, transport=None, timeout: float = 30.0, sleep=time.sleep, base_url: str = BASE, line_items: str | None = None):
        self.pool = pool
        self.transport = transport or _urllib_transport
        self.timeout = timeout
        self.sleep = sleep
        self.base_url = base_url.rstrip("/")
        self.line_items = line_items if line_items is not None else ",".join(canonical.OPENFILINGS_KEYS)

    def available(self) -> tuple[bool, str]:
        if self.pool is None or len(self.pool) == 0:
            return False, "no OPENFILINGS_API_KEY is set: create a key at https://openfilings.org (Dashboard > API keys) and add it as a secret (on Replit: Tools > Secrets), then restart the app"
        return True, ""

    def cost(self, request: str, period: str = "annual") -> int:
        return CREDITS.get(request, 1)

    # ------------------------------------------------------------------------------------------------------------ http
    def _get(self, path: str, params: dict, endpoint: str, symbol: str = "", market: str = "", force: bool = False):
        if self.pool is None or len(self.pool) == 0:
            raise MissingKey(self.available()[1])
        url = f"{self.base_url}{path}"
        query = parse.urlencode({k: v for k, v in params.items() if v is not None})
        if query:
            url += "?" + query

        def call(key: str):
            headers = {"X-API-Key": key, "Accept": "application/json", "User-Agent": USER_AGENT}
            for attempt in range(BURST_RETRIES + 1):
                try:
                    status, response_headers, body = self.transport(url, headers, self.timeout)
                except ProviderError as err:
                    raise ProviderError(redact(str(err), key)) from None
                try:
                    return self._interpret(status, response_headers, body, key)
                except RateLimited as err:
                    if err.dimension != "burst" or attempt >= BURST_RETRIES:
                        raise
                    self.sleep(min(err.retry_after if err.retry_after is not None else 2.0 * (attempt + 1), MAX_BURST_WAIT))
            raise ProviderError("OpenFilings kept answering too many requests; try again in a minute")             # unreachable: the loop returns or raises

        return self.pool.call(call, endpoint=endpoint, symbol=symbol, market=market, credits=CREDITS.get(endpoint, 1), force=force)

    @staticmethod
    def _interpret(status: int, headers: dict, body: bytes, key: str):
        text = body.decode("utf-8", "replace") if body else ""
        try:
            data = json.loads(text) if text.strip() else None
        except ValueError:
            data = None
        detail = data.get("detail", data) if isinstance(data, dict) else data
        lowered = {str(k).lower(): v for k, v in (headers or {}).items()}

        def say(limit: int = 200) -> str:
            if isinstance(detail, dict):
                return redact(str(detail.get("message") or detail.get("error") or json.dumps(detail, default=str)), key)[:limit]
            return redact(str(detail if detail is not None else text), key)[:limit]

        def seconds(name: str) -> float | None:
            try:
                return float(lowered.get(name))
            except (TypeError, ValueError):
                return None

        if status == 202:
            retry = (detail.get("retry_after") if isinstance(detail, dict) else None) or seconds("retry-after")
            raise SyncPending("OpenFilings is still collecting this company's filings (on-demand sync); ask again in a few minutes", float(retry) if retry else None)
        if 200 <= status < 300:
            if data is None:
                raise ProviderError("OpenFilings answered with something that is not JSON")
            return data
        if status == 401:
            raise InvalidKey("OpenFilings rejected the API key (HTTP 401): it is missing, wrong or revoked")
        if status == 402:
            info = detail if isinstance(detail, dict) else {}
            balance = info.get("credit_balance", seconds("x-credits-balance"))
            required = seconds("x-credits-required")
            raise CreditsExhausted("OpenFilings: the included credits are spent and the prepaid balance cannot cover this call (HTTP 402)",
                                   float(balance) if isinstance(balance, (int, float)) else None, required)
        if status == 403:
            info = detail if isinstance(detail, dict) else {}
            if info.get("error") == "upgrade_required":
                raise PlanError(f"your OpenFilings plan does not include {info.get('feature') or 'this'}" + (f" (it needs the {info['min_plan']} plan)" if info.get("min_plan") else ""),
                                info.get("feature"), info.get("min_plan"))
            raise PlanError(f"OpenFilings refused the request (HTTP 403): {say()}")
        if status == 404:
            raise NotFound(f"OpenFilings has nothing for this request: {say()}")
        if status == 429:
            info = detail if isinstance(detail, dict) else {}
            retry = seconds("retry-after")
            if info.get("error") == "quota_exceeded":
                dimension = str(info.get("dimension") or "daily").lower()
                kind = "monthly" if "month" in dimension else ("daily" if "day" in dimension or "daily" in dimension else "burst")
                counts = {k: info[k] for k in ("daily_limit", "daily_used", "daily_remaining", "monthly_limit", "monthly_used", "monthly_remaining") if isinstance(info.get(k), (int, float))}
                note = ", ".join(f"{k.replace('_', ' ')} {int(v)}" for k, v in counts.items())
                if kind == "burst":
                    raise RateLimited("OpenFilings asked to slow down", retry, "burst", counts)
                raise RateLimited(f"OpenFilings {kind} allowance is spent" + (f" ({note})" if note else ""), retry, kind, counts)
            raise RateLimited("OpenFilings asked to slow down", retry, "burst", {})
        if status in (400, 422):
            raise ProviderError(f"OpenFilings rejected the request (HTTP {status}): {say()}")
        if status >= 500:
            raise ProviderError(f"OpenFilings is temporarily unavailable (HTTP {status}); try again in a few minutes")
        raise ProviderError(f"OpenFilings answered HTTP {status}: {say(120)}")

    # ------------------------------------------------------------------------------------------------------------ the account
    def test(self) -> dict:
        """One credit: list the markets on the account's plan. Proves the key works and reports the plan, which sets the allowances the quota manager enforces."""
        data = self._get("/api/v1/markets", {}, "markets", force=True)
        markets = [m.get("id") for m in data.get("markets", []) if isinstance(m, dict) and m.get("id")]
        allowed = [str(m) for m in data.get("allowed_market_ids", [])] or markets
        return {"plan": str(data.get("plan") or "").lower() or None, "markets": markets, "allowed_markets": allowed, "all_markets": bool(data.get("all_markets")),
                "with_filings": [m for m in MARKETS_WITH_FILINGS if m in allowed or not allowed]}

    # ------------------------------------------------------------------------------------------------------------ companies
    def normalise_symbol(self, symbol: str, market: str | None = None) -> str:
        market_id, ticker = split_symbol(symbol, market, self.default_market)
        return f"{market_id}:{ticker.upper()}"

    def offline_profile(self, symbol: str, market: str | None = None) -> CompanyProfile:
        market_id, ticker = split_symbol(symbol, market, self.default_market)
        return _profile(market_id, ticker.upper())

    search_plan = staticmethod(search_plan)

    def search_detail(self, query: str, limit: int = 10, kind: str = "auto", market: str | None = None) -> dict:
        """One search (one credit): ``{"profiles", "kind", "value", "market", "alternatives", "message", "ambiguous", "credits"}``. A company the directory does not know is an empty list, not an error,
        and costs nothing (a failed call is refunded)."""
        plan = search_plan(query, kind, market)
        secrets = self.pool.secrets() if self.pool else []
        try:
            data = self._get("/api/v1/entities/search", plan["params"], "search", plan["value"], plan["market"] or "")
        except NotFound as err:
            return {**{k: plan[k] for k in ("kind", "value", "market", "alternatives")}, "profiles": [], "message": redact(str(err), *secrets)[:200], "ambiguous": False, "markets_skipped": [], "credits": 0}
        data = data if isinstance(data, dict) else {}
        profiles = _search_profiles(data, plan["market"])[:limit]
        message = redact(str(data.get("message") or ""), *secrets).strip()[:300]
        skipped = []                                                                                  # markets the directory did not search (the answer's own list; its items are not described)
        for item in (data.get("markets_skipped") or [])[:20]:
            name = item if isinstance(item, str) else (item.get("market_id") or item.get("id") or item.get("market")) if isinstance(item, dict) else None
            if isinstance(name, str) and name.strip():
                skipped.append(name.strip().lower()[:12])
        return {**{k: plan[k] for k in ("kind", "value", "market", "alternatives")}, "profiles": profiles, "message": message, "ambiguous": bool(data.get("ambiguous")), "markets_skipped": skipped,
                "credits": CREDITS["search"]}

    def search(self, query: str, limit: int = 10, kind: str = "auto", market: str | None = None) -> list[CompanyProfile]:
        return self.search_detail(query, limit, kind, market)["profiles"]

    def get_company(self, symbol: str, market: str | None = None) -> CompanyProfile:
        market_id, ticker = split_symbol(symbol, market, self.default_market)
        data = self._get("/api/v1/entities/lookup", {"ticker": ticker, "market_id": market_id}, "lookup", f"{market_id}:{ticker}", market_id)
        ids = data.get("identifiers") if isinstance(data.get("identifiers"), dict) else {}
        industry = data.get("industry") if isinstance(data.get("industry"), dict) else {}
        isin = data.get("isin") or ids.get("isin")
        sector = industry.get("group") or industry.get("sic")
        detail = industry.get("damodaran") or industry.get("industry_local") or industry.get("sic")
        return _profile(market_id, ticker.upper(), str(data.get("display_name") or ""), isin, data.get("lei") or ids.get("lei"), None, sector, detail,
                        {"yahoo_symbol": ids.get("yahoo_symbol"), "sec_cik": ids.get("sec_cik"), "industry": industry})

    def get_financials(self, symbol: str, market: str | None = None, start_year: int | None = None, end_year: int | None = None, period: str = "annual",
                       limit: int | None = None) -> FinancialsResult:
        if period != "annual":
            raise PlanError("OpenFilings quarterly figures are not supported by this application yet: its documentation does not list the fiscal_period values, and a guess would spend 10 credits")
        market_id, ticker = split_symbol(symbol, market, self.default_market)
        this_year = datetime.now(timezone.utc).year
        first = start_year if start_year else (this_year - int(limit) if limit else this_year - DEFAULT_HISTORY_YEARS)
        params = {"ticker": ticker, "market_id": market_id, "fiscal_year_from": int(first), "fiscal_year_to": int(end_year) if end_year else None, "fiscal_period": "fy",
                  "line_items": self.line_items or "ALL"}
        try:
            data = self._get("/api/v1/companies/financials", params, "financials", f"{market_id}:{ticker}", market_id)
        except ProviderError as err:
            if str(err).startswith("OpenFilings rejected the request") and "line_item" in str(err).lower() and params["line_items"] != "ALL":      # a key this market's taxonomy does not know: a rejected call is refunded, so ask for everything
                params["line_items"] = "ALL"
                data = self._get("/api/v1/companies/financials", params, "financials", f"{market_id}:{ticker}", market_id)
            else:
                raise
        periods, profile, skipped = parse_financials(data, market_id, ticker)
        if not periods:
            message = str(data.get("message") or "").strip()
            raise NotFound(f"OpenFilings returned no annual statements for {market_id}:{ticker}" + (f" ({redact(message, *(self.pool.secrets() if self.pool else []))[:160]})" if message else ""))
        earliest = min(p.fiscal_year for p in periods if p.fiscal_year is not None) if any(p.fiscal_year is not None for p in periods) else None
        request_info = {"ticker": ticker, "market_id": market_id, "fiscal_year_from": params["fiscal_year_from"], "fiscal_year_to": params["fiscal_year_to"], "fiscal_period": "fy",
                        "line_items": "ALL" if params["line_items"] == "ALL" else f"{len(canonical.OPENFILINGS_KEYS)} mapped keys", "skipped": skipped}
        return FinancialsResult(periods, data, request_info, profile, exhausted=earliest is not None and earliest > params["fiscal_year_from"], requests=1, cost=CREDITS["financials"])


def _label(raw) -> str | None:
    """OpenFilings' period label as the application's: ``FY``, ``Q1``..``Q4``, ``H1`` or ``H2``; ``None`` for anything else (year-to-date and trailing figures are not stored as periods)."""
    text = str(raw or "").strip().upper().replace(" ", "").replace("-", "").replace("_", "")
    if text in ("FY", "ANNUAL", "A", "YEAR", "FULLYEAR", "FISCALYEAR"):
        return "FY"
    if text in ("Q1", "Q2", "Q3", "Q4", "H1", "H2"):
        return text
    return None


def parse_financials(payload: dict, market_id: str = "", ticker: str = "") -> tuple[list[PeriodStatement], CompanyProfile | None, int]:
    """``(periods, profile, skipped)`` from a ``CompanyFinancialsOut`` answer.

    Each entry of ``periods`` is one fiscal period with the date of the filing the figures come from. When several entries describe the same period (a figure repeated in a later filing), the earliest
    filing wins: that is when the figures first became public. Null values are dropped, never turned into zeros; entries labelled with anything but an annual period are counted in ``skipped``.
    """
    entity = payload.get("entity") if isinstance(payload.get("entity"), dict) else {}
    market = str(entity.get("market_id") or market_id or "").lower()
    symbol = str(entity.get("ticker") or ticker or "")
    chosen: dict[tuple[str, str], PeriodStatement] = {}
    skipped = 0
    for entry in payload.get("periods") or []:
        if not isinstance(entry, dict):
            continue
        label = _label(entry.get("fiscal_period")) if entry.get("fiscal_period") not in (None, "") else "FY"
        end = _day(entry.get("period_end"))
        values = {}
        for key, value in (entry.get("values") or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values[str(key)] = float(value)
        if label != "FY" or end is None or not values:
            skipped += 1
            continue
        filed = _day(entry.get("filing_date"))
        statement = PeriodStatement("FY", end, values, fiscal_year=entry.get("fiscal_year") if isinstance(entry.get("fiscal_year"), int) else None, filing_date=filed,
                                    reporting_currency=entry.get("currency") or payload.get("currency_latest"), provider_record_id=str(entry.get("filing_id") or "") or None)
        key = ("FY", end)
        old = chosen.get(key)
        if old is None or (filed and (old.filing_date is None or filed < old.filing_date)):
            chosen[key] = statement
    periods = sorted(chosen.values(), key=lambda p: p.period_end)
    profile = None
    if market and symbol:
        profile = _profile(market, symbol.upper(), str(entity.get("name") or ""), payload.get("isin") or entity.get("isin"), payload.get("lei") or entity.get("lei"), payload.get("currency_latest"))
    return periods, profile, skipped
