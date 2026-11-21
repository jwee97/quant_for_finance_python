"""SEC EDGAR as a country connector: the same filing list the importer already uses, in the shape the other countries' connectors have, so the backfill and the pages treat the United States like any market."""

from __future__ import annotations


from ..errors import FundamentalsError
from .base import Access, Connector, FilingRef, registry


@registry.register
class Sec(Connector):
    name = "sec_edgar"
    label = "SEC EDGAR (United States)"
    countries = ("US",)
    hosts = ("sec.gov",)
    formats = ("ixbrl", "xbrl", "json")
    languages = ("en",)
    access = Access("api", "SEC_EDGAR_USER_AGENT", "https://www.sec.gov/os/accessing-edgar-data", "Free; every request must say who is asking (a name and an e-mail address in the User-Agent).")
    provider = None

    def _provider(self):
        if self.provider is None:
            from ..providers.sec_edgar import SecEdgarProvider
            self.provider = SecEdgarProvider(user_agent=self.key())
        return self.provider

    def list_filings(self, company: dict, since: str | None = None, until: str | None = None, kinds=None, limit: int = 100) -> dict:
        self._need()
        symbol = str(company.get("ticker") or company.get("cik") or "").strip()
        if not symbol:
            raise FundamentalsError("SEC EDGAR needs a ticker or a CIK")
        forms = ("10-K", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A")
        out = self._provider().filings(symbol if symbol.lower().startswith("cik:") or not symbol.isdigit() else f"cik:{symbol}", forms, limit)
        refs = []
        for f in out["filings"]:
            kind = "FY" if f["form"].startswith(("10-K", "20-F", "40-F")) else "Q"
            if (since and f["filing_date"] < since) or (until and f["filing_date"] > until) or (kinds and kind not in kinds and not (kind == "Q" and any(k.startswith("Q") for k in kinds))):
                continue
            refs.append(FilingRef(self.name, "US", str(out["cik"]), f["accession"], f["form"], kind, f["period"], f["filing_date"], f["url"], "ixbrl" if f["xbrl"] else "html", f"{f['form']} {f['period'] or ''}", "en", 0, f["form"].endswith("/A")))
        return {"filings": refs, "complete": True, "next_since": None, "note": out.get("note", "")}

    def get(self, ref: FilingRef) -> dict:
        self._need()
        if self.fetcher is None:
            from ..importer.fetch import Fetcher
            self.fetcher = Fetcher(user_agent=self.key())
        got = self._fetch(ref.url)
        return {"kind": "file", "name": ref.url.rsplit("/", 1)[-1], "data": got.data, "hint": {"accession": ref.doc_id, "filing_date": ref.filed_at}}
