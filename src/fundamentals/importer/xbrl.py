"""Figures from an XBRL instance or an inline-XBRL (iXBRL) filing, read exactly as tagged.

Why not guess from the page: a filing's numbers carry their own labels. Each fact names its element (``us-gaap:Revenues``), the context it belongs to (a period, and whether it is for the whole company or
for a segment) and a unit; this module reads those and nothing else, and maps the elements to the application's canonical lines with the table the SEC provider uses (:data:`canonical.SEC_MAP`), so a
filing read here and the same company read from the SEC's company facts give the same lines.

What it does and does not do. It reads the filing's own period (the document period end, and the fiscal period focus when the filing says one) and, with ``comparatives``, the earlier columns of the same
filing. A fact in a segment or scenario context (a business unit, a product line) is not the company's figure and is skipped. A 10-Q's cash flow lines are reported only for the year to date: those lines
are left out of the draft and listed (``ytd_only``), never stored as if they were a quarter. It does not download taxonomies, so it cannot validate a filing against its schema; the arithmetic checks in
:mod:`reconcile` stand in for that, and the filing's accession is kept as the figure's source. Arelle (the XBRL consortium's processor) is the tool for schema validation; it needs the taxonomy packages
of every filer and a network, which this application does not assume.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date

from lxml import etree

from .. import canonical
from ..errors import FundamentalsError

NS_XBRLI = "http://www.xbrl.org/2003/instance"
IX_NAMESPACES = ("http://www.xbrl.org/2013/inlineXBRL", "http://www.xbrl.org/2008/inlineXBRL")
NS_LINK = "http://www.xbrl.org/2003/linkbase"
MAX_FACTS = 400_000
YEAR = (350, 380)
QUARTER = (80, 100)
TAXONOMY_HINTS = (("jppfs", "jppfs_cor"), ("jpigp", "jpigp_cor"), ("jpdei", "jpdei_cor"), ("jpcrp", "jpcrp_cor"), ("us-gaap", "us-gaap"), ("ifrs-full", "ifrs-full"), ("ifrs", "ifrs-full"), ("/dei/", "dei"), ("fasb.org/us-gaap", "us-gaap"))
JP_COVER = ("CurrentPeriodEndDateDEI", "TypeOfCurrentPeriodDEI", "AccountingStandardsDEI", "FilerNameInEnglishDEI", "FilerNameInJapaneseDEI", "SecurityCodeDEI", "EDINETCodeDEI", "CurrentFiscalYearStartDateDEI", "CurrentFiscalYearEndDateDEI")
COVER_FIELDS = JP_COVER + ("DocumentType", "DocumentPeriodEndDate", "DocumentFiscalYearFocus", "DocumentFiscalPeriodFocus", "EntityRegistrantName", "TradingSymbol", "EntityCentralIndexKey", "AmendmentFlag")


def _taxonomy(uri: str) -> str | None:
    low = (uri or "").lower()
    for hint, name in TAXONOMY_HINTS:
        if hint in low:
            return name
    return None


def _day(text) -> date | None:
    try:
        return date.fromisoformat(str(text or "").strip()[:10])
    except ValueError:
        return None


def looks_like_xbrl(data: bytes) -> bool:
    head = data[:200_000].lower()
    return b"xbrli:xbrl" in head or b"<xbrl" in head or b"inlinexbrl" in head or b"ix:nonfraction" in head or (b"xbrl.org/2003/instance" in head)


def _number(text: str, scale: int, sign: str | None, fmt: str | None) -> float | None:
    """The value of an inline fact: its text under the transformation the filing names. French and German reports group digits with a non-breaking space (``15 170`` with U+00A0, or U+202F), which is
    whitespace for this purpose; ``ixt:num-comma-decimal`` and its siblings write ``1.234,5`` or ``1 234,5``; ``ixt:zerodash`` and ``fixed-zero`` mean zero."""
    t = re.sub(r"\s+", "", (text or "").strip())
    f = (fmt or "").lower().replace("-", "").replace("_", "")
    if "zerodash" in f or "fixedzero" in f:
        value = 0.0
    else:
        if "commadecimal" in f or "spacecomma" in f or "numcomma" in f:                            # 1.234,56 or 1 234,56
            t = t.replace(".", "").replace("'", "").replace(",", ".")
        else:
            t = t.replace(",", "").replace("'", "")
        negative = t.startswith("(") and t.endswith(")")
        t = t.strip("()$€£¥")
        try:
            value = float(t)
        except ValueError:
            return None
        if negative:
            value = -value
    value *= 10.0 ** scale
    if sign == "-":
        value = -value
    return value if math.isfinite(value) else None


def _local(tag: str) -> tuple[str, str]:
    if tag.startswith("{"):
        uri, _, name = tag[1:].partition("}")
        return uri, name
    return "", tag


def read_facts(data: bytes) -> dict:
    """The raw content of a filing: ``contexts`` (id -> period, dimensional), ``units`` (id -> text), ``facts`` (list of dicts: field, value, context, unit, decimals) and ``cover`` (dei text facts)."""
    if len(data) > 60_000_000:
        raise FundamentalsError("this file is too large to read as a filing (over 60 MB)")
    parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(data, parser)
    if root is None:
        raise FundamentalsError("this file is not readable as XML, so it cannot be a filing in XBRL or inline XBRL")
    contexts: dict[str, dict] = {}
    units: dict[str, str] = {}
    facts: list[dict] = []
    cover: dict[str, str] = {}
    inline = False
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        uri, name = _local(el.tag)
        if uri == NS_XBRLI and name == "context":
            period = el.find(f"{{{NS_XBRLI}}}period")
            instant = period.findtext(f"{{{NS_XBRLI}}}instant") if period is not None else None
            start = period.findtext(f"{{{NS_XBRLI}}}startDate") if period is not None else None
            end = period.findtext(f"{{{NS_XBRLI}}}endDate") if period is not None else None
            entity = el.find(f"{{{NS_XBRLI}}}entity")
            dimensional = el.find(f"{{{NS_XBRLI}}}scenario") is not None or (entity is not None and entity.find(f"{{{NS_XBRLI}}}segment") is not None)
            members = sorted({(m.text or "").strip() for m in el.iter("{http://xbrl.org/2006/xbrldi}explicitMember")})
            contexts[el.get("id")] = {"instant": _day(instant), "start": _day(start), "end": _day(end) or _day(instant), "dimensional": dimensional, "members": members}
        elif uri == NS_XBRLI and name == "unit":
            measures = [m.text.strip() for m in el.iter(f"{{{NS_XBRLI}}}measure") if m.text]
            units[el.get("id")] = "/".join(m.split(":")[-1] for m in measures) if len(measures) > 1 else (measures[0].split(":")[-1] if measures else "")
        elif uri in IX_NAMESPACES and name in ("nonFraction", "nonNumeric"):
            inline = True
            field = el.get("name") or ""
            prefix, _, local = field.partition(":")
            tax = _taxonomy(el.nsmap.get(prefix, "")) or (prefix if prefix in ("us-gaap", "ifrs-full", "dei") else None)
            if tax is None:
                continue
            if name == "nonNumeric":
                if tax in ("dei", "jpdei_cor") and local in COVER_FIELDS:
                    cover.setdefault(local, " ".join("".join(el.itertext()).split()))
                continue
            if el.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
                continue
            value = _number("".join(el.itertext()), int(el.get("scale") or 0), el.get("sign"), el.get("format"))
            if value is not None and len(facts) < MAX_FACTS:
                facts.append({"field": f"{tax}:{local}", "value": value, "context": el.get("contextRef"), "unit": el.get("unitRef"), "decimals": el.get("decimals")})
    if not inline:                                                                                     # a plain XBRL instance: the facts are the elements that carry a contextRef
        for el in root.iter():
            if not isinstance(el.tag, str) or el.get("contextRef") is None:
                continue
            uri, name = _local(el.tag)
            tax = _taxonomy(uri)
            if tax is None:
                continue
            text = " ".join("".join(el.itertext()).split())
            if tax in ("dei", "jpdei_cor") and name in COVER_FIELDS:
                cover.setdefault(name, text)
                if el.get("unitRef") is None:
                    continue
            if el.get("unitRef") is None:
                continue
            try:
                value = float(text.replace(",", ""))
            except ValueError:
                continue
            if math.isfinite(value) and len(facts) < MAX_FACTS:
                facts.append({"field": f"{tax}:{name}", "value": value, "context": el.get("contextRef"), "unit": el.get("unitRef"), "decimals": el.get("decimals")})
    if not facts:
        raise FundamentalsError("no tagged numbers were found: this is not an XBRL instance or an inline-XBRL filing (or it has no financial facts)")
    return {"contexts": contexts, "units": units, "facts": facts, "cover": cover, "inline": inline}


def extract(data: bytes, name: str = "", accession: str | None = None, comparatives: bool = False) -> dict:
    """A draft from one filing: ``{"periods": [...], "units", "currency", "company", "notes", "ytd_only", "unmapped", "method", "accession"}`` (see :mod:`pipeline`)."""
    raw = read_facts(data)
    contexts, units, facts, cover = raw["contexts"], raw["units"], raw["facts"], raw["cover"]
    money = Counter()
    for f in facts:
        u = units.get(f["unit"], "")
        if re.fullmatch(r"[A-Z]{3}", u) and f["field"].split(":")[1] in ("Assets", "Revenues", "NetIncomeLoss", "Revenue", "ProfitLoss", "Equity", "StockholdersEquity", "NetSales", "AssetsIFRS", "RevenueIFRS", "ProfitLossIFRS"):
            money[u] += 1
    currency = money.most_common(1)[0][0] if money else None
    end = _day(cover.get("DocumentPeriodEndDate")) or _day(cover.get("CurrentPeriodEndDateDEI"))
    if end is None:                                                                                  # the latest period end among the company-wide facts
        ends = [c["end"] for c in contexts.values() if c["end"] and not c["dimensional"]]
        end = max(ends) if ends else None
    if end is None:
        raise FundamentalsError("the filing names no reporting period")
    doc_type = (cover.get("DocumentType") or "").upper()
    focus = (cover.get("DocumentFiscalPeriodFocus") or "").upper()
    jp_type = (cover.get("TypeOfCurrentPeriodDEI") or "").upper()
    if jp_type:                                                                                      # Japan's own cover fact: FY, Q1, Q2, Q3 or HY (half-year)
        focus = {"HY": "H1"}.get(jp_type, jp_type)
        doc_type = doc_type or ("10-K" if jp_type == "FY" else "")
    standard = cover.get("AccountingStandardsDEI")
    notes: list[str] = []
    per_period: dict[tuple, dict] = {}                                                               # (end, kind) -> {field: (value, unit, context)}
    jp_facts = [f for f in facts if f["field"].startswith(("jppfs_cor:", "jpigp_cor:"))]
    parent_only = bool(jp_facts) and not any(contexts.get(f["context"], {}).get("dimensional") is False for f in jp_facts)                # only the parent's own statements are filed
    for f in facts:
        c = contexts.get(f["context"])
        if c is None or c["end"] is None:
            continue
        if c["dimensional"] and not (parent_only and c.get("members") and all("NonConsolidatedMember" in m for m in c["members"])):
            continue
        u = units.get(f["unit"], "")
        if currency and re.fullmatch(r"[A-Z]{3}", u) and u != currency:
            continue
        span = None if c["start"] is None else (c["end"] - c["start"]).days
        key = (c["end"], "instant" if c["start"] is None else ("year" if YEAR[0] <= span <= YEAR[1] else "quarter" if QUARTER[0] <= span <= QUARTER[1] else "half" if 160 <= span <= 200 else "ytd" if span > QUARTER[1] else "short"))
        per_period.setdefault(key, {}).setdefault(f["field"], (f["value"], u, f["context"]))
    ends = sorted({k[0] for k in per_period}, reverse=True)
    wanted = [end] if not comparatives else [e for e in ends if (end - e).days < 800]
    periods: list[dict] = []
    ytd_only: list[str] = []
    unmapped: set[str] = set()
    for e in wanted:
        flows_year = per_period.get((e, "year"), {})
        flows_q = per_period.get((e, "quarter"), {})
        flows_ytd = {**per_period.get((e, "ytd"), {}), **per_period.get((e, "half"), {})}
        flows_half = per_period.get((e, "half"), {})
        stocks = per_period.get((e, "instant"), {})
        is_year = bool(flows_year) and (doc_type.startswith("10-K") or doc_type.startswith("20-F") or doc_type.startswith("40-F") or focus == "FY" or not flows_q)
        fiscal_period = "FY" if is_year else (focus if focus in ("Q1", "Q2", "Q3", "Q4", "H1") and e == end else None)
        source: dict[str, tuple] = {}
        for field, (value, unit, ctx) in stocks.items():
            source[field] = (value, unit, ctx)
        flows = flows_year if is_year else (flows_half if fiscal_period == "H1" and flows_half else flows_q)
        for field, item in flows.items():
            source.setdefault(field, item)
        if not is_year:
            for field in flows_ytd:
                if field not in flows_q and canonical.SEC_FIELDS.get(field) == "duration":
                    ytd_only.append(field)
        for field, item in per_period.get((e, "instant"), {}).items():                                # cover figures (shares outstanding) may sit on any date
            source.setdefault(field, item)
        if not source:
            continue
        plain = {f: v[0] for f, v in source.items()}
        obs, un = canonical.normalise_values("sec_edgar", plain)
        un = [u for u in un if not u.startswith(("jppfs_cor:", "jpigp_cor:"))]
        from ..filings.maps import jp_values
        jp, jp_un = jp_values(plain)
        for code, o in jp.items():
            obs.setdefault(code, o)
        unmapped.update(un + jp_un)
        values, sources = {}, {}
        for code, o in obs.items():
            if o["flag"] == "derived":
                continue                                                                              # the application derives these itself, and marks them so
            tag = o["source_metric"]
            values[code] = o["value"]
            sources[code] = {"tag": tag, "reported": plain.get(tag), "unit": source[tag][1] if tag in source else None, "method": "xbrl", "context": source[tag][2] if tag in source else None}
        if not values:
            continue
        if fiscal_period is None:
            fiscal_period = "Q" + str((e.month - 1) // 3 + 1) if not is_year else "FY"
            notes.append(f"the period ending {e.isoformat()} is not the filing's own period; its fiscal quarter number was taken from the calendar month and may differ from the company's own numbering: check it")
        periods.append({"fiscal_period": fiscal_period, "period_end": e.isoformat(), "fiscal_year": int(cover["DocumentFiscalYearFocus"]) if e == end and str(cover.get("DocumentFiscalYearFocus", "")).isdigit() else e.year,
                        "filing_date": None, "currency": currency, "values": values, "sources": sources})
    if not periods:
        raise FundamentalsError("the filing has tagged numbers but none of the lines this application reads (revenue, net income, total assets and the others) for its reporting period")
    if ytd_only:
        notes.append("Some cash flow and income lines are reported only for the year to date in this filing (" + ", ".join(sorted({f.split(':')[1] for f in ytd_only})[:6]) + "); they were left out rather than stored as a quarter. Import the earlier quarters of the year too, or enter the quarter.")
    company = {"name": cover.get("EntityRegistrantName") or cover.get("FilerNameInEnglishDEI") or cover.get("FilerNameInJapaneseDEI"), "ticker": (cover.get("TradingSymbol") or cover.get("SecurityCodeDEI") or "").upper() or None,
               "cik": cover.get("EntityCentralIndexKey") or cover.get("EDINETCodeDEI")}
    return {"periods": periods, "units": {"money": "units", "shares": "units"}, "currency": currency, "company": company, "notes": notes, "ytd_only": sorted(set(ytd_only)), "unmapped": sorted(unmapped)[:60],
            "method": "ixbrl" if raw["inline"] else "xbrl", "accession": accession or _accession_from(name), "document_type": doc_type or None, "accounting_standard": standard,
            "parent_only": bool(parent_only)}


def _accession_from(name: str) -> str | None:
    m = re.search(r"(\d{10}-\d{2}-\d{6})", name or "") or re.search(r"\b(\d{18})\b", name or "")
    if not m:
        return None
    a = m.group(1)
    return a if "-" in a else f"{a[:10]}-{a[10:12]}-{a[12:]}"
