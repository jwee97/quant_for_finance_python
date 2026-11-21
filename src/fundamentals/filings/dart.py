"""OpenDART, the Korean Financial Supervisory Service's disclosure API: reports listed by company, and the statements of a report as structured accounts.

A key is required (``OPENDART_API_KEY``). ``list.json`` lists a company's regular reports (annual 사업보고서, half-year 반기보고서, quarterly 분기보고서); ``fnlttSinglAcntAll.json`` gives the full statements of
one report as accounts, each with its IFRS element id, so no PDF is read: the figures are mapped through the same IFRS table as the SEC's. Consolidated statements are asked for first and the separate ones only
when there are none. The report's own scope of periods applies: the income statement of a quarterly report has the quarter (``thstrm_amount``) and the year to date (``thstrm_add_amount``); the cash flow
statement has only the year to date, so it is kept for Q1 (which is the quarter) and the half-year (which is six months) and left out of Q3. The daily request limit is the source's (20,000); a refusal for the
limit (status 020) is reported as such. Built from the published API description; not exercised against the live service from here.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import date

from .. import canonical
from ..errors import FundamentalsError
from . import _util
from . import calendar as cal
from .base import Access, Connector, FilingRef, registry

BASE = "https://opendart.fss.or.kr/api"
OUTFLOWS = ("capital_expenditure", "dividends_paid", "share_repurchases", "acquisitions")
STATUS = {"000": "", "010": "the key is not registered", "011": "the key is not usable", "012": "this address is not allowed for the key", "013": "no data", "014": "no file", "020": "the daily request limit was exceeded",
          "100": "a request field is not valid", "800": "the service is down for maintenance", "900": "an unknown error"}


@registry.register
class Dart(Connector):
    name = "dart"
    label = "OpenDART (Korea FSS)"
    countries = ("KR",)
    hosts = ("opendart.fss.or.kr", "dart.fss.or.kr")
    formats = ("json", "xbrl", "html")
    languages = ("ko", "en")
    access = Access("api", "OPENDART_API_KEY", "https://opendart.fss.or.kr/intro/main.do", "Official API for programs; a free key is required (20,000 requests a day).")

    def _call(self, name: str, **params) -> dict:
        query = "&".join(f"{k}={v}" for k, v in {"crtfc_key": self.key(), **params}.items() if v not in (None, ""))
        data = self._json(f"{BASE}/{name}.json?{query}")
        status = str(data.get("status", "000"))
        if status not in ("000", "013"):
            raise FundamentalsError(f"OpenDART: {STATUS.get(status) or data.get('message') or status} (status {status})")
        return data

    def corp_code(self, stock_code: str) -> str:
        """The eight-digit corporation code of a listed company from its six-digit stock code, from OpenDART's code list (a ZIP of one XML)."""
        got = self._fetch(f"{BASE}/corpCode.xml?crtfc_key={self.key()}")
        try:
            xml = zipfile.ZipFile(io.BytesIO(got.data)).read("CORPCODE.xml")
        except (zipfile.BadZipFile, KeyError):
            raise FundamentalsError("OpenDART did not send its list of corporation codes (the key may have been refused)") from None
        for el in ET.fromstring(xml).iter("list"):
            if (el.findtext("stock_code") or "").strip() == stock_code:
                return el.findtext("corp_code").strip()
        raise FundamentalsError(f"no listed company has the stock code {stock_code} on OpenDART")

    def list_filings(self, company: dict, since: str | None = None, until: str | None = None, kinds=None, limit: int = 100) -> dict:
        self._need()
        corp = str(company.get("corp_code") or "").strip()
        if not corp:
            stock = str(company.get("stock_code") or company.get("ticker") or "").strip()
            if not stock:
                raise FundamentalsError("OpenDART needs the company's corp_code or its six-digit stock code")
            corp = self.corp_code(stock.zfill(6))
        end = (until or date.today().isoformat()).replace("-", "")
        start = (since or f"{date.today().year - 8}-01-01").replace("-", "")
        found: list[FilingRef] = []
        for page in range(1, 6):
            data = self._call("list", corp_code=corp, bgn_de=start, end_de=end, pblntf_ty="A", page_no=page, page_count=100)
            for r in data.get("list") or []:
                ref = self._ref(r)
                if ref and (not kinds or ref.fiscal_period in kinds):
                    found.append(ref)
            if page >= int(data.get("total_page") or 1) or len(found) >= limit:
                break
        return {"filings": found[:limit], "complete": True, "next_since": None, "note": ""}

    @staticmethod
    def _ref(r: dict) -> FilingRef | None:
        name = str(r.get("report_nm") or "")
        kind = next((k for w, k in cal.DART_NAME_WORDS if w in name), None)
        m = re.search(r"\((\d{4})\.(\d{2})\)", name)
        if kind is None or not m:
            return None
        year, month = int(m.group(1)), int(m.group(2))
        if kind == "Q":
            kind = "Q1" if month in (3, 4) else "Q3" if month in (9, 10) else "Q"
        end = _util.month_end(year, month)
        return FilingRef("dart", "KR", str(r.get("corp_code") or ""), str(r.get("rcept_no")), "A", kind, end.isoformat(), str(r.get("rcept_dt") or ""), f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={r.get('rcept_no')}",
                         "json", name, "ko", 0, "정정" in name, {"corp_code": r.get("corp_code"), "stock_code": r.get("stock_code"), "reprt_code": {"FY": "11011", "H1": "11012", "Q1": "11013", "Q3": "11014"}.get(kind),
                                                              "fy_end_month": None})

    def get(self, ref: FilingRef) -> dict:
        self._need()
        code = ref.extra.get("reprt_code")
        if not code or not ref.period_end:
            raise FundamentalsError("this filing is a quarter whose number is not known (OpenDART names it by its month): choose a report of Q1, the half-year, Q3 or the year")
        end = date.fromisoformat(ref.period_end)
        fy_end_month = int(ref.extra.get("fy_end_month") or 12)
        fiscal_year = end.year if ref.fiscal_period == "FY" else (end.year if end.month <= fy_end_month else end.year + 1)
        for scope in ("CFS", "OFS"):
            data = self._call("fnlttSinglAcntAll", corp_code=ref.company_id or ref.extra.get("corp_code"), bsns_year=fiscal_year, reprt_code=code, fs_div=scope)
            if data.get("list"):
                break
        else:
            raise FundamentalsError("OpenDART has no statements for this report")
        import json
        return {"kind": "draft", "name": f"{ref.doc_id}.json", "raw": json.dumps(data, ensure_ascii=False).encode(), "draft": self.draft(data, ref, scope, end)}

    @staticmethod
    def draft(data: dict, ref: FilingRef, scope: str, end: date) -> dict:
        kind = ref.fiscal_period
        fields: dict[str, float] = {}
        tags: dict[str, dict] = {}
        ytd_only: list[str] = []
        for row in data.get("list") or []:
            aid, sj = str(row.get("account_id") or ""), str(row.get("sj_div") or "")
            if not aid or aid.startswith("-"):
                continue
            field = "ifrs-full:" + aid.split("_", 1)[1] if aid.startswith("ifrs-full_") else None
            if field is None:
                continue
            raw = row.get("thstrm_add_amount") if (sj in ("IS", "CIS") and kind == "H1") else row.get("thstrm_amount")
            v = _util.number(raw)
            if v is None:
                continue
            if sj == "CF" and kind == "Q3":
                ytd_only.append(row.get("account_nm") or aid)
                continue
            if field not in fields:
                fields[field] = v
                tags[field] = {"account": row.get("account_nm"), "statement": sj, "raw": raw}
        obs, unmapped = canonical.normalise_values("sec_edgar", fields)
        values, sources = {}, {}
        for code, o in obs.items():
            if o["flag"] == "derived":
                continue
            v = abs(o["value"]) if code in OUTFLOWS else o["value"]
            values[code] = v
            t = tags.get(o["source_metric"], {})
            sources[code] = {"tag": o["source_metric"], "reported": v, "unit": "KRW", "method": "dart-api", "account": t.get("account"), "text": f"{t.get('statement')}: {t.get('account')} {t.get('raw')}"}
        if not values:
            raise FundamentalsError("OpenDART sent statements but none of the lines this application reads (accounts without an IFRS element id are not mapped)")
        period = {"fiscal_period": kind, "period_end": end.isoformat(), "fiscal_year": end.year, "filing_date": f"{ref.filed_at[:4]}-{ref.filed_at[4:6]}-{ref.filed_at[6:8]}" if ref.filed_at and len(ref.filed_at) == 8 else ref.filed_at, "currency": "KRW",
                  "values": values, "sources": sources}
        notes = [f"Statements are {'consolidated' if scope == 'CFS' else 'separate (the company has no consolidated statements)'}, in won, as reported to OpenDART."]
        if ytd_only:
            notes.append("A Q3 report gives its cash flow only for nine months: those lines were left out (" + ", ".join(sorted(set(ytd_only))[:5]) + ").")
        if kind == "H1":
            notes.append("The half-year period holds six months: its income statement is the year-to-date column, its cash flow the six months reported.")
        return {"periods": [period], "units": {"money": "units", "shares": "units"}, "currency": "KRW", "company": {"name": None, "ticker": ref.extra.get("stock_code")}, "notes": notes, "ytd_only": sorted(set(ytd_only)),
                "unmapped": unmapped[:60], "method": "dart-api", "accession": ref.doc_id, "accounting_standard": "K-IFRS", "units_stated": True}
