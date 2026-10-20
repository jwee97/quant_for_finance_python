"""Financial Modeling Prep (https://financialmodelingprep.com): company profiles and the three statements, annual or quarterly, with the filing and accepted dates the point-in-time engine needs.

A company costs **three requests per period type** (income statement, balance sheet, cash flow) plus one for the profile, so a free plan's few hundred requests a day is why the application stores
everything and asks only for what is missing. Keys come from the environment and rotate (:mod:`keys`): ``FMP_API_KEY``, ``FMP_API_KEYS`` or ``FMP_API_KEY_1`` ... ``_20``.

FMP serves the latest *restated* figures with the original filing date, so ``as_reported`` is False: a restatement is a small look-ahead that this provider cannot rule out, and the Data Manager says so.
The adapter was written from FMP's documented stable endpoints and its known error answers (HTTP 401 with ``{"Error Message": "Invalid API KEY..."}``); it has not been run against a paid plan here.
"""

from __future__ import annotations

import json
from urllib import error as urlerror
from urllib import parse, request

from ...webapp.providers import redact
from ..errors import InvalidKey, NotFound, PlanError, ProviderError, RateLimited
from ..keys import KeyPool
from ..store import CompanyProfile, PeriodStatement
from .base import FinancialsResult, FundamentalDataProvider

BASE = "https://financialmodelingprep.com/stable"
USER_AGENT = "QuantLab/1.0 (fundamentals client)"
STATEMENTS = (("income", "income-statement"), ("balance", "balance-sheet-statement"), ("cashflow", "cash-flow-statement"))
# FMP's exchange short names -> the application's market codes
EXCHANGES = {"NASDAQ": "NASDAQ", "NYSE": "NYSE", "AMEX": "NYSE", "NYSEARCA": "NYSE", "SES": "SGX", "SGX": "SGX", "LSE": "LSE", "JPX": "TSE", "TSE": "TSE", "HKSE": "HKEX", "HKEX": "HKEX",
             "ASX": "ASX", "XETRA": "XETRA", "TSX": "TSX", "NZE": "NZX", "KSC": "KRX", "KOE": "KRX", "NSE": "NSE", "SHH": "SSE", "SHZ": "SSE", "TAI": "TWSE", "SAO": "B3", "MCX": "BMV",
             "JNB": "JSE", "STO": "STO", "CPH": "CPH", "OSL": "OSL", "HEL": "HEL", "MIL": "MIL", "SIX": "SIX", "PAR": "EPA", "AMS": "AMS", "MCE": "BME"}


def _urllib_transport(url: str, timeout: float) -> tuple[int, bytes]:
    req = request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:                      # noqa: S310  (https only; the address is built here)
            return response.status, response.read(16_000_000)
    except urlerror.HTTPError as err:
        return err.code, err.read(200_000)
    except (urlerror.URLError, TimeoutError, OSError) as err:
        raise ProviderError(f"could not reach Financial Modeling Prep ({type(err).__name__}); check the network") from None


class FMPProvider(FundamentalDataProvider):
    name = "fmp"
    kind = "http"
    as_reported = False
    has_filing_dates = True
    keyed = True

    def __init__(self, pool: KeyPool | None = None, transport=None, timeout: float = 20.0, annual_limit: int = 40, quarterly_limit: int = 120):
        self.pool = pool
        self.transport = transport or _urllib_transport
        self.timeout = timeout
        self.annual_limit, self.quarterly_limit = annual_limit, quarterly_limit

    def available(self) -> tuple[bool, str]:
        if self.pool is None or len(self.pool) == 0:
            return False, "no FMP_API_KEY is set (add FMP_API_KEY, or FMP_API_KEY_1, _2, ... one per account, as a secret)"
        return True, ""

    # ------------------------------------------------------------------------------------------------------------ http
    def _get(self, path: str, params: dict, symbol: str = ""):
        if self.pool is None:
            from ..errors import MissingKey
            raise MissingKey("no FMP_API_KEY is set (add it as a secret and restart the app)")

        def call(key: str):
            query = parse.urlencode({**{k: v for k, v in params.items() if v is not None}, "apikey": key})
            try:
                status, body = self.transport(f"{BASE}/{path}?{query}", self.timeout)
            except ProviderError as err:
                raise ProviderError(redact(str(err), key)) from None
            return self._interpret(status, body, key)

        return self.pool.call(call, endpoint=path, symbol=symbol)

    @staticmethod
    def _interpret(status: int, body: bytes, key: str):
        text = redact(body[:400].decode("utf-8", "replace"), key)
        try:
            data = json.loads(body.decode("utf-8", "replace")) if body else None
        except ValueError:
            data = None
        message = ""
        if isinstance(data, dict):
            message = str(data.get("Error Message") or data.get("error") or data.get("message") or "")
        low = (message or text).lower()
        if status == 401 or "invalid api key" in low:
            raise InvalidKey("FMP rejected the API key (HTTP 401)")
        if status == 429 or "limit reach" in low or "limit exceeded" in low:
            raise RateLimited("FMP says this key's allowance is spent (HTTP 429)")
        if status in (402, 403) or "premium" in low or "subscription" in low or "upgrade your plan" in low:
            raise PlanError(f"FMP's plan does not include this ({message or 'HTTP ' + str(status)}); the free plan covers fewer exchanges and shorter histories")
        if status == 404:
            raise NotFound("FMP has nothing for that symbol")
        if status >= 400:
            raise ProviderError(f"FMP answered HTTP {status}: {message or text[:120]}")
        if isinstance(data, dict) and message:
            raise ProviderError(f"FMP: {message}")
        return data

    # ------------------------------------------------------------------------------------------------------------ the interface
    @staticmethod
    def _profile(row: dict) -> CompanyProfile:
        symbol = str(row.get("symbol") or "")
        exchange = str(row.get("exchange") or row.get("exchangeShortName") or "").upper()
        market = EXCHANGES.get(exchange, exchange or None)
        return CompanyProfile("fmp", symbol, symbol.split(".")[0], str(row.get("companyName") or row.get("name") or ""), exchange=market or "", market=market,
                              country=(str(row["country"]).upper() if row.get("country") else None), currency=row.get("currency"), sector=row.get("sector") or None,
                              industry=row.get("industry") or None, isin=row.get("isin") or None, active=bool(row.get("isActivelyTrading", True)),
                              extra={"cik": row.get("cik"), "market_cap": row.get("marketCap"), "exchange_name": row.get("exchangeFullName")})

    def search(self, query: str, limit: int = 10) -> list[CompanyProfile]:
        rows = list(self._get("search-symbol", {"query": query, "limit": limit}) or [])
        if len(rows) < limit:                                               # a name search finds what a symbol search does not, at the price of one more request
            rows += list(self._get("search-name", {"query": query, "limit": limit}) or [])
        out, seen = [], set()
        for row in rows:
            if row.get("symbol") in seen:
                continue
            seen.add(row.get("symbol"))
            exchange = str(row.get("exchange") or "").upper()
            out.append(CompanyProfile("fmp", str(row.get("symbol")), str(row.get("symbol")).split(".")[0], str(row.get("name") or ""), exchange=EXCHANGES.get(exchange, exchange),
                                      market=EXCHANGES.get(exchange, exchange) or None, currency=row.get("currency")))
        return out[:limit]

    def get_company(self, symbol: str, market: str | None = None) -> CompanyProfile:
        rows = self._get("profile", {"symbol": symbol}, symbol)
        if not rows:
            raise NotFound(f"FMP has no profile for {symbol}")
        return self._profile(rows[0] if isinstance(rows, list) else rows)

    def get_financials(self, symbol: str, market: str | None = None, start_year: int | None = None, end_year: int | None = None, period: str = "annual",
                       limit: int | None = None) -> FinancialsResult:
        """The merged income statement, balance sheet and cash flow of ``symbol``: ``limit`` periods (default 40 annual or 120 quarterly), newest first as FMP gives them."""
        per = "quarter" if period in ("quarter", "quarterly") else "annual"
        n = int(limit or (self.quarterly_limit if per == "quarter" else self.annual_limit))
        raw: dict[str, list] = {}
        for label, path in STATEMENTS:
            raw[label] = self._get(path, {"symbol": symbol, "period": per, "limit": n}, symbol) or []
        periods = self._merge(raw)
        if start_year is not None:
            periods = [p for p in periods if int(p.period_end[:4]) >= start_year]
        if end_year is not None:
            periods = [p for p in periods if int(p.period_end[:4]) <= end_year]
        got = max((len(v) for v in raw.values()), default=0)
        return FinancialsResult(periods, raw, {"symbol": symbol, "period": per, "limit": n}, exhausted=got < n, requests=len(STATEMENTS))

    @staticmethod
    def _merge(raw: dict[str, list]) -> list[PeriodStatement]:
        """One record per ``(fiscal period, period end)`` from the three statements. Each source field is prefixed with its statement so a word that means a level on one and a change on another is
        never confused."""
        merged: dict[tuple, dict] = {}
        for label, rows in raw.items():
            for row in rows:
                if not isinstance(row, dict) or not row.get("date"):
                    continue
                key = (str(row.get("period") or "FY").upper(), str(row["date"])[:10])
                rec = merged.setdefault(key, {"meta": row, "values": {}})
                for field, value in row.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        rec["values"][f"{label}:{field}"] = value
        out = []
        for (fp, end), rec in merged.items():
            meta = rec["meta"]
            year = meta.get("fiscalYear") or meta.get("calendarYear")
            out.append(PeriodStatement(fp if fp in ("FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2") else "FY", end, rec["values"], fiscal_year=int(year) if str(year).isdigit() else None,
                                       filing_date=meta.get("filingDate") or meta.get("fillingDate"), accepted_date=(str(meta.get("acceptedDate"))[:10] if meta.get("acceptedDate") else None),
                                       reporting_currency=meta.get("reportedCurrency"), provider_record_id=f"{meta.get('symbol')}-{end}-{fp}"))
        out.sort(key=lambda p: (p.period_end, p.fiscal_period))
        return out
