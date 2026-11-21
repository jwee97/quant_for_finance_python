"""EDINET, Japan's Financial Services Agency disclosure system, through its documented API (v2).

EDINET has no "filings of this company" call: its document list is by *date* (every filing of one day), so a company's filings are found by scanning days (business days only, at most ``MAX_DAYS`` a call,
resumable from ``next_since``) and keeping those whose security code or EDINET code match. A key is required (``EDINET_API_KEY``, free on the FSA's site): without one nothing is requested. The document types read are
the securities reports: annual (120), quarterly (140), half-year (160) and their corrections (130, 150, 170). A filing is fetched as the submitted ZIP and its XBRL instance is handed to the importer, which reads
the Japanese taxonomies (``jppfs_cor``, ``jpigp_cor``, ``jpdei_cor``). Built from the published API description; not exercised against the live service from here.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta

from ..errors import FundamentalsError
from . import calendar as cal
from .base import Access, Connector, FilingRef, registry

BASE = "https://api.edinet-fsa.go.jp/api/v2"
MAX_DAYS = 60


@registry.register
class Edinet(Connector):
    name = "edinet"
    label = "EDINET (Japan FSA)"
    countries = ("JP",)
    hosts = ("api.edinet-fsa.go.jp", "disclosure.edinet-fsa.go.jp", "disclosure2.edinet-fsa.go.jp")
    formats = ("xbrl", "pdf")
    languages = ("ja", "en")
    access = Access("api", "EDINET_API_KEY", "https://disclosure2dl.edinet-fsa.go.jp/guide/static/disclosure/WZEK0110.html", "Official API for programs; a free key is required and every request carries it.")

    def _documents(self, day: date) -> list[dict]:
        data = self._json(f"{BASE}/documents.json?date={day.isoformat()}&type=2&Subscription-Key={self.key()}")
        lower = {str(k).lower(): v for k, v in data.items()}                                         # a refused key is answered as {"StatusCode": 401, "message": ...} (a capital S), checked against the live service
        status = str((data.get("metadata") or {}).get("status", lower.get("statuscode", "200")))
        if status == "401":
            raise FundamentalsError("EDINET refused the key (HTTP 401): check EDINET_API_KEY")
        if status not in ("200", "404"):
            raise FundamentalsError(f"EDINET answered status {status}: {(data.get('metadata') or {}).get('message') or lower.get('message') or ''}")
        return data.get("results") or []

    def list_filings(self, company: dict, since: str | None = None, until: str | None = None, kinds=None, limit: int = 100) -> dict:
        self._need()
        sec = str(company.get("sec_code") or company.get("ticker") or "").strip()
        edinet = str(company.get("edinet_code") or "").strip().upper()
        if not sec and not edinet:
            raise FundamentalsError("EDINET needs the company's securities code (for example 7203) or its EDINET code (E02144)")
        end = date.fromisoformat(until) if until else date.today()
        start = date.fromisoformat(since) if since else end - timedelta(days=365)
        found: list[FilingRef] = []
        day, scanned = start, 0
        while day <= end and scanned < MAX_DAYS and len(found) < limit:
            if day.weekday() < 5:
                scanned += 1
                for r in self._documents(day):
                    code = str(r.get("docTypeCode") or "")
                    if code not in cal.EDINET_DOC_TYPES or (kinds and cal.EDINET_DOC_TYPES[code][0] not in kinds):
                        continue
                    if not ((sec and str(r.get("secCode") or "")[:4] == sec[:4]) or (edinet and str(r.get("edinetCode") or "").upper() == edinet)):
                        continue
                    kind, correction = cal.EDINET_DOC_TYPES[code]
                    found.append(FilingRef(self.name, "JP", str(r.get("edinetCode") or ""), str(r.get("docID")), code, kind, r.get("periodEnd"), str(r.get("submitDateTime") or "")[:19],
                                           f"{BASE}/documents/{r.get('docID')}", "xbrl" if r.get("xbrlFlag") in (1, "1", True) else "pdf", str(r.get("docDescription") or ""), "ja", 0, correction,
                                           {"filer": r.get("filerName"), "period_start": r.get("periodStart")}))
            day += timedelta(days=1)
        complete = day > end
        return {"filings": found, "complete": complete, "next_since": None if complete else day.isoformat(), "scanned_days": scanned,
                "note": "EDINET lists filings by day: the next call carries on from next_since." if not complete else ""}

    def get(self, ref: FilingRef) -> dict:
        self._need()
        got = self._fetch(f"{BASE}/documents/{ref.doc_id}?type=1&Subscription-Key={self.key()}")
        try:
            zf = zipfile.ZipFile(io.BytesIO(got.data))
        except zipfile.BadZipFile:
            raise FundamentalsError("EDINET did not send a ZIP for this document (it may be withdrawn, or the key was refused)") from None
        inst = [i for i in zf.infolist() if i.filename.upper().startswith("XBRL/PUBLICDOC/") and i.filename.lower().endswith(".xbrl")]
        if not inst:
            raise FundamentalsError("the EDINET package holds no XBRL instance (a filing with no tagged statements): use its PDF through Report import")
        main = max(inst, key=lambda i: i.file_size)
        return {"kind": "file", "name": f"{ref.doc_id}.xbrl", "data": zf.read(main), "hint": {"accession": ref.doc_id, "filing_date": (ref.filed_at or "")[:10] or None}}
