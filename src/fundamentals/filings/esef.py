"""Europe's ESEF filings (annual financial reports in inline XBRL) through filings.xbrl.org, the open index of them.

France, Spain, Germany and the other EU markets require listed issuers' annual reports in ESEF format: an inline-XBRL report tagged with IFRS elements, delivered as a ZIP package. filings.xbrl.org
publishes an open JSON index of the packages (no key). A filing's report is read by the importer's inline-XBRL reader through the IFRS mapping already used for the SEC's IFRS filers. Half-year reports are not ESEF
and are not in the index. The request shape follows the index's published JSON:API description; not exercised against the live service from here.
"""

from __future__ import annotations

import io
import json
import zipfile
from urllib import parse

from ..errors import FundamentalsError
from .base import Access, Connector, FilingRef, registry

SITE = "https://filings.xbrl.org"


@registry.register
class Esef(Connector):
    name = "esef"
    label = "ESEF reports (filings.xbrl.org)"
    countries = ("FR", "ES", "DE", "IT", "NL", "BE", "IE", "PT", "AT", "FI", "SE", "DK", "NO", "PL", "LU", "GR")
    hosts = ("filings.xbrl.org",)
    formats = ("ixbrl",)
    languages = ("en", "fr", "es", "de")
    access = Access("open_data", None, "https://filings.xbrl.org/", "An open index of ESEF filings published for reuse; no key.")

    def list_filings(self, company: dict, since: str | None = None, until: str | None = None, kinds=None, limit: int = 100) -> dict:
        lei = str(company.get("lei") or "").strip().upper()
        country = str(company.get("country") or "").strip().upper()
        if not lei and not country:
            raise FundamentalsError("ESEF needs the company's LEI (20 characters) or at least its country")
        filters = ([{"name": "fxo_id", "op": "like", "val": f"{lei}%"}] if lei else []) + ([{"name": "country", "op": "eq", "val": country}] if country and not lei else [])
        query = parse.urlencode({"filter": json.dumps(filters), "sort": "-period_end", "page[size]": min(limit, 200)})
        data = self._json(f"{SITE}/api/filings?{query}")
        found = []
        for item in data.get("data") or []:
            a = item.get("attributes") or {}
            period = str(a.get("period_end") or "")[:10]
            if (since and period < since) or (until and period > until) or not a.get("package_url"):
                continue
            found.append(FilingRef(self.name, str(a.get("country") or country), lei or str(a.get("fxo_id") or "")[:20], str(a.get("fxo_id") or item.get("id")), "ESEF", "FY", period or None, str(a.get("date_added") or "")[:10] or None,
                                   SITE + str(a["package_url"]), "ixbrl", f"Annual financial report {period[:4]}", None, 0, False, {"report_url": a.get("report_url")}))
        return {"filings": found[:limit], "complete": True, "next_since": None, "note": "ESEF covers annual reports; half-year reports are not in the index."}

    def get(self, ref: FilingRef) -> dict:
        got = self._fetch(ref.url)
        try:
            zf = zipfile.ZipFile(io.BytesIO(got.data))
        except zipfile.BadZipFile:
            raise FundamentalsError("the ESEF package is not a ZIP") from None
        reports = [i for i in zf.infolist() if "/reports/" in "/" + i.filename and i.filename.lower().endswith((".xhtml", ".html", ".htm"))]
        if not reports:
            raise FundamentalsError("the ESEF package holds no inline-XBRL report")
        main = max(reports, key=lambda i: i.file_size)
        return {"kind": "file", "name": main.filename.rsplit("/", 1)[-1], "data": zf.read(main), "hint": {"accession": ref.doc_id, "filing_date": ref.filed_at}}
