"""Brazil's CVM open data: the standardised financial statements (DFP, annual) and quarterly information (ITR) of every listed company, published as yearly ZIP files of CSV.

One file holds every company of a year, so listing is cheap (the files that exist) and a *get* downloads a year's file and keeps the rows of one company. The statements are numbered accounts
(``3.01`` net revenue, ``1`` total assets, see :mod:`maps`); the scale of the amounts is stated per row (thousands or units) and applied; consolidated statements are used where the company has them. The file's
``VERSAO`` column numbers each re-filing of the same period: the latest is read and the earlier versions are listed in the notes, with their dates of receipt, so a restatement is visible. A quarterly
income statement has the quarter and the year to date (told apart by the start date of the period); the cash flow statement has the year to date only, so it is kept for the first quarter and left out of the
others. Built from the published description of the files; not exercised against the live service from here.
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import date

from ..errors import FundamentalsError
from . import _util
from . import calendar as cal
from .base import Access, Connector, FilingRef, registry
from .maps import CVM_ACCOUNTS, CVM_POSITIVE

BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC"


def _rows(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as fh:
        yield from csv.DictReader(io.TextIOWrapper(fh, encoding="latin-1", newline=""), delimiter=";")


@registry.register
class Cvm(Connector):
    name = "cvm"
    label = "CVM open data (Brazil)"
    countries = ("BR",)
    hosts = ("dados.cvm.gov.br",)
    formats = ("csv",)
    languages = ("pt",)
    access = Access("open_data", None, "https://dados.cvm.gov.br/", "Open data published for reuse; no key. One ZIP per year holds every company.")

    def list_filings(self, company: dict, since: str | None = None, until: str | None = None, kinds=None, limit: int = 100) -> dict:
        y1 = int((until or date.today().isoformat())[:4])
        y0 = int((since or f"{y1 - 8}-01-01")[:4])
        found = []
        for year in range(y1, y0 - 1, -1):
            for doc, kind in (("DFP", "FY"), ("ITR", "Q")):
                if kinds and kind not in kinds and not (kind == "Q" and any(k.startswith("Q") for k in kinds)):
                    continue
                url = f"{BASE}/{doc}/DADOS/{doc.lower()}_cia_aberta_{year}.zip"
                found.append(FilingRef(self.name, "BR", str(company.get("cd_cvm") or company.get("cnpj") or ""), f"{doc}-{year}", doc, kind, None, None, url, "csv", f"{doc} {year}", "pt", 0, False, {"year": year}))
        return {"filings": found[:limit], "complete": True, "next_since": None, "note": "Each file holds every listed company for the year; reading one keeps the rows of the company asked for."}

    def get(self, ref: FilingRef) -> dict:
        company = ref.company_id.strip()
        if not company:
            raise FundamentalsError("CVM needs the company's CVM code or its CNPJ")
        got = self._fetch(ref.url)
        try:
            zf = zipfile.ZipFile(io.BytesIO(got.data))
        except zipfile.BadZipFile:
            raise FundamentalsError("CVM did not send a ZIP for this year") from None
        doc, year = ref.doc_type, ref.extra["year"]
        digits = "".join(c for c in company if c.isdigit())
        by_period: dict[str, dict] = {}
        versions: dict[str, set] = {}
        filed: dict[tuple, str] = {}
        main = f"{doc.lower()}_cia_aberta_{year}.csv"
        if main in zf.namelist():
            for r in _rows(zf, main):
                if self._mine(r, digits):
                    filed[(r.get("DT_REFER"), r.get("VERSAO"))] = r.get("DT_RECEB") or ""
        for stmt, accounts in CVM_ACCOUNTS.items():
            name = next((n for scope in ("con", "ind") for n in zf.namelist() if n.lower() == f"{doc.lower()}_cia_aberta_{stmt}_{scope}_{year}.csv".lower()), None)
            if name is None:
                continue
            for r in _rows(zf, name):
                if not self._mine(r, digits) or not (r.get("ORDEM_EXERC") or "").upper().startswith(("ÚLT", "ULT", "\xdaLT")):
                    continue
                code = accounts.get((r.get("CD_CONTA") or "").strip())
                if code is None:
                    continue
                end = date.fromisoformat(r["DT_REFER"])
                start = date.fromisoformat(r["DT_INI_EXERC"]) if r.get("DT_INI_EXERC") else None
                span = (end - start).days if start else None
                if span is not None:
                    if doc == "DFP" and not 350 <= span <= 380:
                        continue
                    if doc == "ITR" and not 80 <= span <= 100:
                        if stmt in ("DFC_MD", "DFC_MI") and not (start and start.month == 1 and span <= 100):
                            continue
                        if stmt not in ("DFC_MD", "DFC_MI"):
                            continue
                v = _util.number(r.get("VL_CONTA"))
                if v is None:
                    continue
                v = abs(v) if code in CVM_POSITIVE else v
                scale = 1000.0 if (r.get("ESCALA_MOEDA") or "").upper().startswith("MIL") else 1.0
                ver = int(r.get("VERSAO") or 0)
                versions.setdefault(r["DT_REFER"], set()).add(ver)
                slot = by_period.setdefault(r["DT_REFER"], {})
                if code not in slot or ver >= slot[code][2]:
                    slot[code] = (v * scale, f"{stmt} {r.get('CD_CONTA')} {r.get('DS_CONTA')}", ver)
        if not by_period:
            raise FundamentalsError(f"CVM's {doc} file for {year} holds no statements for this company")
        periods, notes = [], []
        for dt, slot in sorted(by_period.items()):
            end = date.fromisoformat(dt)
            kind = "FY" if doc == "DFP" else cal.quarter_of(end, 12)
            if kind == "Q":
                continue
            if "current_liabilities" in slot and "noncurrent_liabilities" in slot:                         # total liabilities as the sum of the two lines CVM reports (2.01 + 2.02)
                cl, nl = slot["current_liabilities"], slot["noncurrent_liabilities"]
                slot["total_liabilities"] = (cl[0] + nl[0], f"BPP 2.01 + 2.02 {cl[1]} + {nl[1]}", min(cl[2], nl[2]))
            slot.pop("noncurrent_liabilities", None)
            ver = max(v for _, _, v in slot.values())
            received = filed.get((dt, str(ver))) or next((v for (d, _), v in filed.items() if d == dt), "")
            periods.append({"fiscal_period": kind, "period_end": dt, "fiscal_year": end.year, "filing_date": received or None, "currency": "BRL", "values": {c: v for c, (v, _, _) in slot.items()},
                            "sources": {c: {"tag": f"CVM {t}", "reported": v, "unit": "BRL", "method": "cvm-csv", "text": t} for c, (v, t, _) in slot.items()}})
            if len(versions.get(dt, ())) > 1:
                notes.append(f"{dt}: CVM holds versions {sorted(versions[dt])} of this report; the latest ({ver}) was read, so an earlier figure that was restated is not in this draft.")
        if not periods:
            raise FundamentalsError("the statements of this file are for periods the application does not hold (a quarter whose number cannot be told); nothing was read")
        return {"kind": "draft", "name": f"{ref.doc_id}-{digits}.json", "raw": f"{doc} {year} {digits}".encode(), "draft": {"periods": periods, "units": {"money": "units", "shares": "units"}, "currency": "BRL", "company": {},
                "notes": notes + ["Consolidated statements where the company files them, otherwise individual; amounts are scaled to units. A cash flow line is kept only for the full year or the first quarter."], "ytd_only": [],
                "unmapped": [], "method": "cvm-csv", "accession": f"{doc}-{year}", "accounting_standard": "BR GAAP (CPC) / IFRS", "units_stated": True}}

    @staticmethod
    def _mine(row: dict, digits: str) -> bool:
        cnpj = "".join(c for c in (row.get("CNPJ_CIA") or "") if c.isdigit())
        cvm = (row.get("CD_CVM") or "").lstrip("0")
        return bool(digits) and (cnpj == digits or cvm == digits.lstrip("0"))
