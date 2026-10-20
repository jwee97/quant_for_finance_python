"""OpenFilings (https://github.com/ryanrodrigues25200525-svg/Openfilings): IFRS statements for companies in about twenty-five non-US markets, read from each regulator's own structured filings.

What it is, because it is not what the word "API" suggests: an MIT-licensed *local* Python library, command line and MCP server that fetches from the regulators' public sites and keeps a cache on this
machine. It has **no hosted service, no key and no daily quota** (Japan's EDINET and Korea's DART need their own free keys, set as ``EDINET_API_KEY`` and ``DART_API_KEY``); it needs Python 3.11 or
later; and it covers neither the United States nor Hong Kong. The requests this provider counts are runs of the bridge, kept under a politeness budget (``OPENFILINGS_DAILY_LIMIT``, default 300) so a
sync does not hammer the regulators.

How it is used. The library is never imported into the application (it needs a newer Python than a Replit workspace may use, and one of its dependencies is AGPL-licensed). The bridge script
(:mod:`openfilings_bridge`) runs under ``OPENFILINGS_PYTHON`` (a Python that has OpenFilings installed: ``uv venv --python 3.11 && uv pip install git+https://github.com/ryanrodrigues25200525-svg/Openfilings``)
and prints JSON that is parsed here. Its answers are untrusted data and are treated as such.

What it returns. OpenFilings normalises to a small set of line items (revenue, cost of revenue, operating income, profit before tax, net income, EPS, cash, receivables, inventory, property,
goodwill, intangibles, current and total assets and liabilities, equity, operating/investing/financing cash flow, capital expenditure, dividends). It has no debt split, no depreciation, no SG&A or R&D
and no share counts, so factors that need those are missing for a company only OpenFilings covers; the Data Manager shows the coverage. Annual accounts only: the period is a financial year and the
filing date is the date of the filing that first reported it (a prior-year comparative takes the date of the later filing that carried it, which is conservative).

Not yet run end to end: written against the library's documented models (version 0.22) and tested against a stand-in bridge.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from ..errors import NotFound, ProviderError
from ..store import CompanyProfile, PeriodStatement
from .base import FinancialsResult, FundamentalDataProvider

BRIDGE = Path(__file__).with_name("openfilings_bridge.py")
KEEP = ("EDINET_API_KEY", "DART_API_KEY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def clean_environment(env: dict | None = None) -> dict:
    """The environment the bridge runs in: the user's own, without the secrets of other services (the FMP keys, the iTick key, ...) that a third-party library has no business reading."""
    env = dict(os.environ if env is None else env)
    return {k: v for k, v in env.items() if k in KEEP or k.startswith("OPENFILINGS_") or not any(w in k.upper() for w in SECRET_WORDS)}


def _run(command: list[str], timeout: float) -> tuple[int, str, str]:
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=clean_environment(), check=False)         # noqa: S603  (our own script, a fixed argument list)
    return done.returncode, done.stdout, done.stderr


def _day(value) -> str | None:
    return str(value)[:10] if value else None


class OpenFilingsProvider(FundamentalDataProvider):
    name = "openfilings"
    kind = "local"
    as_reported = True
    has_filing_dates = True
    keyed = False
    periods_available = ("annual",)

    def __init__(self, python: str | None = None, runner=None, timeout: float = 600.0, source: str = "all", filings: int = 8):
        self.python = python or os.environ.get("OPENFILINGS_PYTHON", "").strip() or None
        self.runner = runner or _run
        self.timeout, self.source, self.filings = timeout, source, filings
        self._checked: tuple[bool, str] | None = None

    def _interpreter(self) -> str:
        return self.python or sys.executable

    def available(self) -> tuple[bool, str]:
        if self._checked is not None:
            return self._checked
        try:
            code, out, err = self.runner([self._interpreter(), str(BRIDGE), "probe"], 120.0)
        except (OSError, subprocess.SubprocessError) as error:
            self._checked = (False, f"could not run {self._interpreter()}: {type(error).__name__}")
            return self._checked
        try:
            data = json.loads(out) if out.strip() else {}
        except ValueError:
            data = {}
        if code == 0 and "service" in data:
            self._checked = (True, "")
        else:
            why = data.get("error") or (err.strip().splitlines() or ["no answer"])[-1]
            self._checked = (False, f"OpenFilings is not available ({why}). Install it for a Python 3.11+ interpreter (uv venv --python 3.11 && uv pip install "
                                    "git+https://github.com/ryanrodrigues25200525-svg/Openfilings) and set OPENFILINGS_PYTHON to that interpreter")
        return self._checked

    # ------------------------------------------------------------------------------------------------------------ the bridge
    def _bridge(self, *args: str) -> dict:
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)
        try:
            code, out, err = self.runner([self._interpreter(), str(BRIDGE), *args], self.timeout)
        except subprocess.TimeoutExpired:
            raise ProviderError(f"OpenFilings took longer than {self.timeout:.0f} seconds") from None
        except (OSError, subprocess.SubprocessError) as error:
            raise ProviderError(f"could not run OpenFilings ({type(error).__name__})") from None
        try:
            data = json.loads(out)
        except ValueError:
            raise ProviderError("OpenFilings printed something that is not JSON: " + (err.strip().splitlines() or [out[:100]])[-1][:200]) from None
        if not isinstance(data, dict):
            raise ProviderError("OpenFilings answered with an unexpected shape")
        if data.get("error"):
            raise ProviderError(f"OpenFilings: {data['error']} {str(data.get('detail') or '')[:200]}".strip())
        return data

    @staticmethod
    def _profile(row: dict) -> CompanyProfile:
        ident = str(row.get("id") or row.get("source_id") or "")
        ticker = row.get("ticker") or row.get("local_code") or row.get("source_id") or ident
        market = row.get("market")
        return CompanyProfile("openfilings", ident, str(ticker), str(row.get("english_name") or row.get("name") or ""), exchange=str(market or ""), market=str(market) if market else None,
                              country=str(row.get("country_code")).upper() if row.get("country_code") else None, lei=row.get("lei") or None,
                              active=str(row.get("status") or "").lower() not in ("inactive", "delisted", "dissolved"), extra={"sources": row.get("sources")})

    def search(self, query: str, limit: int = 10) -> list[CompanyProfile]:
        data = self._bridge("search", query, "--source", self.source, "--limit", str(int(limit)))
        return [self._profile(c) for c in data.get("companies", []) if isinstance(c, dict)][:limit]

    def get_company(self, symbol: str, market: str | None = None) -> CompanyProfile:
        found = self.search(symbol, 5)
        for profile in found:
            if profile.provider_symbol == symbol:
                return profile
        if not found:
            raise NotFound(f"OpenFilings found no company for {symbol!r}")
        return found[0]

    def get_financials(self, symbol: str, market: str | None = None, start_year: int | None = None, end_year: int | None = None, period: str = "annual",
                       limit: int | None = None) -> FinancialsResult:
        n = int(limit or self.filings)
        data = self._bridge("financials", symbol, "--source", self.source, "--filings", str(n))
        periods = parse_financials(data)
        if start_year is not None:
            periods = [p for p in periods if int(p.period_end[:4]) >= start_year]
        if end_year is not None:
            periods = [p for p in periods if int(p.period_end[:4]) <= end_year]
        return FinancialsResult(periods, data, {"company_id": symbol, "filings": n, "source": self.source}, exhausted=len(data.get("filings", [])) < n, requests=1)


def parse_financials(payload: dict) -> list[PeriodStatement]:
    """Annual :class:`PeriodStatement` s from the bridge's ``financials`` answer.

    Each filing carries its own year and the prior-year comparatives. A year takes its values from the filing that is about that year when we have it, otherwise from the earliest filing that
    reported it; its filing date is that filing's date, so a comparative is never treated as public before the filing that carried it.
    """
    periods: dict[str, dict] = {}
    for filing in payload.get("filings", []):
        fin = filing.get("financials") if isinstance(filing, dict) else None
        if not fin:
            continue
        filed = _day(filing.get("filing_date")) or _day(filing.get("published_at"))
        own_end = _day(filing.get("period_end"))
        for statement in fin.get("statements", []):
            currency = statement.get("currency")
            for item in statement.get("line_items", []):
                code = item.get("code")
                for v in item.get("values", []):
                    per = v.get("period") or {}
                    end = _day(per.get("end_date"))
                    if not code or not end:
                        continue
                    if per.get("kind") == "duration":
                        start = _day(per.get("start_date"))
                        try:
                            span = (date.fromisoformat(end) - date.fromisoformat(start)).days if start else 0
                        except ValueError:
                            span = 0
                        if not 330 <= span <= 400:                                  # annual flows only
                            continue
                    try:
                        value = float(v.get("value"))
                    except (TypeError, ValueError):
                        continue
                    rec = periods.setdefault(end, {"values": {}, "filed": None, "original": False, "currency": currency, "start": None})
                    is_own = own_end == end
                    if is_own and not rec["original"]:
                        rec.update(values={}, filed=filed, original=True)         # the filing about this year replaces anything taken from a later comparative
                    if rec["original"] and not is_own:
                        continue
                    rec["values"].setdefault(code, value)
                    rec["currency"] = rec["currency"] or currency
                    if per.get("kind") == "duration":
                        rec["start"] = rec["start"] or _day(per.get("start_date"))
                    if not rec["original"]:
                        rec["filed"] = min([d for d in (rec["filed"], filed) if d], default=None)
    out = []
    for end, rec in sorted(periods.items()):
        if rec["values"]:
            out.append(PeriodStatement("FY", end, rec["values"], fiscal_year=int(end[:4]), period_start=rec["start"], filing_date=rec["filed"], reporting_currency=rec["currency"],
                                       accounting_standard="IFRS", provider_record_id=f"openfilings-{end}"))
    return out
