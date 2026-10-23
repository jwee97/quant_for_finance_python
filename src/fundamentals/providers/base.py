"""The interface every fundamentals provider implements, so that nothing else in the application knows which one it is talking to."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..store import CompanyProfile, PeriodStatement


@dataclass
class FinancialsResult:
    """One answer from a provider: the periods, the exact payload it returned (kept once in the database), and what the request was."""

    periods: list[PeriodStatement]
    raw: object                                  # JSON-able; stored compressed and deduplicated by hash
    request: dict = field(default_factory=dict)
    profile: CompanyProfile | None = None        # what the answer itself said about the company (a free update of the security master)
    answer: dict | None = None                   # what the answer held within the years asked, when the provider cuts the window itself (:func:`coverage.summarise_answer` otherwise reads it from ``raw``)
    exhausted: bool = False                      # True when the provider has nothing older than the earliest period returned
    requests: int = 1                            # how many external requests this answer cost
    cost: int = 1                                # what they cost in the provider's unit (credits for OpenFilings)


class FundamentalDataProvider:
    """Providers differ in what they know; this is what the research needs from any of them.

    ``as_reported`` says whether the figures are as first filed (``True``) or the latest restated ones (``False``): a restated figure with its original filing date is a small look-ahead, and the Data
    Manager says so. ``has_filing_dates`` says whether the periods come with a real filing or publication date; without one the point-in-time engine assumes a conservative lag and flags it.
    """

    name = "base"
    kind = "http"                                # http | simulated
    as_reported = False
    has_filing_dates = True
    keyed = False
    periods_available = ("annual", "quarter")
    default_market: str | None = None
    meters_itself = False                        # True when the provider counts its own requests (the service then does not count them again)
    price_note = "Credits pay for the request, not for the years: the same price would have brought back a longer or a shorter series."
    early_note = ("either it holds no statements that early for this company or it limits how long a series it returns (its documentation says neither). "
                  "Asking for just those years tells which, for the same price.")

    def available(self) -> tuple[bool, str]:
        """``(usable, reason)``: whether this provider can be called now, and if not, what is missing."""
        return True, ""

    has_lookup = False                           # whether a separate call can add sector, industry and identifiers to a company already registered

    def describe(self) -> dict:
        ok, why = self.available()
        return {"name": self.name, "kind": self.kind, "available": ok, "reason": why, "as_reported": self.as_reported, "has_filing_dates": self.has_filing_dates, "keyed": self.keyed,
                "periods": list(self.periods_available)}

    def cost(self, request: str, period: str = "annual") -> int:
        """What one request of this kind costs in the provider's unit (``profile``, ``financials``)."""
        return 1

    def normalise_symbol(self, symbol: str, market: str | None = None) -> str:
        """The one spelling the database keeps for a symbol, so that ``AAPL`` and ``us:AAPL`` are the same company."""
        return str(symbol).strip()

    def offline_profile(self, symbol: str, market: str | None = None) -> CompanyProfile:
        """The security-master entry for ``symbol`` without asking the provider anything (the ticker and market the user typed), to be completed by the first financials answer."""
        raise NotImplementedError

    def search(self, query: str, limit: int = 10) -> list[CompanyProfile]:
        raise NotImplementedError

    def search_plan(self, text: str, kind: str = "auto", market: str | None = None) -> dict:
        """What a search for ``text`` would ask, and what else it could have meant; a provider with only one way to search answers with that one."""
        return {"kind": "name", "value": " ".join(str(text).split()), "market": market, "params": {}, "alternatives": []}

    def search_detail(self, query: str, limit: int = 10, kind: str = "auto", market: str | None = None) -> dict:
        """``search`` with the details a page needs: how the text was read, a message from the provider, whether the answer is ambiguous and what it cost."""
        plan = self.search_plan(query, kind, market)
        profiles = self.search(query, limit)
        return {**{k: plan[k] for k in ("kind", "value", "market", "alternatives")}, "profiles": profiles, "message": "", "ambiguous": len(profiles) > 1, "markets_skipped": [], "credits": self.cost("search")}

    def get_company(self, symbol: str, market: str | None = None) -> CompanyProfile:
        raise NotImplementedError

    def get_financials(self, symbol: str, market: str | None = None, start_year: int | None = None, end_year: int | None = None, period: str = "annual",
                       limit: int | None = None) -> FinancialsResult:
        raise NotImplementedError

    def get_company_metrics(self, symbol: str, market: str | None = None) -> dict:
        """Optional extras a provider offers (market capitalisation, share count): nothing the research cannot compute from prices and statements."""
        return {}
