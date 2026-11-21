"""The extraction engine: bytes -> IR -> statements -> columns -> candidates -> validation, with **targeted, quality-driven fallback**.

``extract(data, name, kind, hint)`` returns ``(draft, document)``: the importer's draft (so the review, approval and storage code is unchanged) with extra optional keys (``statements``, ``candidates``,
``presentations``, ``checks``, ``validation``, ``confidence``, ``rejections``, ``standard``, ``fallbacks``, ``versions``, ``assessment``).

**When is a reading good enough?** Not when it found enough numbers: when :func:`assess` finds nothing unreliable. It looks at the statements recognised, the critical lines found, the columns resolved
(role, period, currency, unit), one unit per statement, the periods inferred rather than read, reconciliation failures, unresolved conflicts between candidates and missing provenance.

**What happens when it is not?** A chain, run on the *affected pages only*, each step re-validated, and a step is accepted only if validation improved (fewer failures, more critical lines confirmed,
fewer unreliable components), never because it returned more numbers:

    1. an independent second table reading of those pages (pdfplumber words, looser grouping)
    2. OCR of those pages, if an engine is installed
    3. an AI reading, only if switched on, only for critical lines still missing, only figures found verbatim in the document, stored unverified
    4. otherwise the draft goes to manual review with every reason

Results are cached by document hash and page set, so the same page is never re-read twice.
"""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

from ..errors import FundamentalsError
from . import checks, matcher, taxonomy
from .ir import Document
from .layout import html_layout, pdf_layout, pdfplumber_layout, xlsx_layout

MIN_LINES = 4
LEGACY_CONFIDENCE = 0.6
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_CACHE_MAX = 32


def read(data: bytes, kind: str, hint: dict | None = None) -> Document:
    hint = hint or {}
    language = hint.get("language") or "en"
    if kind == "pdf":
        return pdf_layout.extract(data, hint.get("ocr_engine"), language)
    if kind == "html":
        return html_layout.extract(data, language)
    if kind == "xlsx":
        return xlsx_layout.from_xlsx(data)
    if kind == "csv":
        from .tabular import decode
        return xlsx_layout.from_text(decode(data))
    raise FundamentalsError(f"the layout engine reads PDF, HTML, workbooks and delimited text, not {kind!r}")


def cap_legacy(draft: dict) -> dict:
    """A draft from the line-by-line readers matched labels without knowing which statement they were in: every figure gets a confidence below the review threshold, so it is stored unverified until a person confirms it."""
    for p in draft.get("periods", []):
        for code in p["values"]:
            s = p.setdefault("sources", {}).setdefault(code, {"tag": None, "method": draft.get("method")})
            s.setdefault("confidence", LEGACY_CONFIDENCE)
            s["confidence"] = min(s["confidence"], LEGACY_CONFIDENCE)
            s.setdefault("statement", "unknown (label match only)")
    draft.setdefault("notes", []).append("Read by the line-by-line fallback reader, which does not know which statement a line is in: every figure is stored unverified until you confirm it.")
    return draft


def lines_found(draft: dict) -> int:
    return len({c for p in draft["periods"] for c in p["values"]})


# ------------------------------------------------------------------------------------------------------------------------ is the reading reliable?
def assess(draft: dict) -> dict:
    """The unreliable components of a reading, each with the pages it concerns. ``reliable`` is true only when there are none. The count of extracted fields is deliberately not a criterion."""
    bad: list[dict] = []
    stm = draft.get("statements")
    if stm is None:                                                                                    # a legacy reading: no structure to judge
        return {"reliable": False, "unreliable": [{"component": "structure", "detail": "read without statement structure", "pages": []}], "pages": []}
    pages_of = {"IS": [], "BS": [], "CF": []}
    for s in stm:
        if s["kind"] in pages_of:
            pages_of[s["kind"]] += s.get("pages") or [s.get("page")]
    for k, name in (("IS", "income statement"), ("BS", "balance sheet")):
        if not any(s["kind"] == k and s["confidence"] > 0 for s in stm):
            bad.append({"component": "statement_identification", "detail": f"no {name} recognised", "pages": []})
    for s in stm:
        if s["kind"] in ("IS", "BS", "CF"):
            weak = [c for c in s["columns"] if c["role"] == "OTHER" and not c["period_end"]]
            if s["confidence"] <= 0 or weak or any(c["confidence"] and c["confidence"] < 0.6 for c in s["columns"] if c["role"] in ("CURRENT_PERIOD", "PRIOR_PERIOD", "HISTORICAL_PERIOD")):
                bad.append({"component": "column_resolution", "detail": f"{s['kind']} table on page {s['page']}: columns not resolved with confidence", "pages": s.get("pages") or [s["page"]]})
            if s["confidence"] and s["confidence"] < 0.6:
                bad.append({"component": "statement_identification", "detail": f"{s['kind']} on page {s['page']} classified with low confidence ({s['why']})", "pages": s.get("pages") or [s["page"]]})
    v = draft.get("validation") or {}
    for r in v.get("reasons", []):
        k = r["kind"]
        if k in ("unit_mismatch", "unit_unstated", "currency_unknown"):
            bad.append({"component": "currency_unit", "detail": r["message"], "pages": sum(pages_of.values(), [])})
        elif k == "period_unresolved":
            bad.append({"component": "period", "detail": r["message"], "pages": sum(pages_of.values(), [])})
        elif k == "check_failed":
            bad.append({"component": "reconciliation", "detail": r["message"], "pages": sum(pages_of.values(), [])})
        elif k == "critical_missing":
            bad.append({"component": "critical_coverage", "detail": r["message"], "pages": sum(pages_of.values(), [])})
    if draft.get("conflicts_between_candidates"):
        bad.append({"component": "candidate_conflict", "detail": f"{len(draft['conflicts_between_candidates'])} conflicting candidate(s)", "pages": sum(pages_of.values(), [])})
    missing_prov = [1 for p in draft.get("periods", []) for s in (p.get("sources") or {}).values() if s.get("page") is None and s.get("confidence") is not None]
    if missing_prov:
        bad.append({"component": "provenance", "detail": f"{len(missing_prov)} figure(s) without a source page", "pages": []})
    pages = sorted({int(p) for b in bad for p in b["pages"] if p})
    return {"reliable": not bad, "unreliable": bad, "pages": pages}


def score(draft: dict) -> tuple:
    """How good a reading is, for comparing two readings of the same pages: fewer failed checks, more critical lines confirmed, fewer unreliable components, more passed checks, higher mean confidence.
    The number of extracted values is not part of it."""
    v = draft.get("validation") or {}
    crit_ok = sum(1 for c in (v.get("critical") or {}).values() if c.get("validated"))
    a = assess(draft)
    fails = sum(1 for c in draft.get("checks", []) if c["status"] == "FAIL") + len(draft.get("conflicts_between_candidates") or [])
    passes = sum(1 for c in draft.get("checks", []) if c["status"] == "PASS")
    conf = (draft.get("confidence") or {}).get("mean") or 0
    return (-fails, crit_ok, -len(a["unreliable"]), passes, conf)


def _replace_pages(doc: Document, replacement: dict[int, list]) -> Document:
    """A copy of the document in which the tables of the given pages are replaced by another reader's tables."""
    out = copy.copy(doc)
    keep = [t for t in doc.tables if not (t.page in replacement or any(p in replacement for p in (t.pages or [])))]
    new = [t for ts in replacement.values() for t in ts]
    out.tables = sorted(keep + new, key=lambda t: (t.page or 0, t.table_id))
    out.reader = {**doc.reader, "second_reading_pages": sorted(replacement)}
    return out


def extract(data: bytes, name: str, kind: str, hint: dict | None = None) -> tuple[dict, Document]:
    hint = hint or {}
    doc = read(data, kind, hint)
    doc.source_name = name
    fallbacks: list[dict] = []
    if doc.format == "pdf-scan" and not doc.tables:
        raise FundamentalsError(doc.warnings[0] if doc.warnings else "scanned PDF with no OCR engine")
    draft = matcher.build(doc, hint)
    recognised = any(st["confidence"] > 0 and st["kind"] in ("IS", "BS", "CF") for st in draft.get("statements", []))
    if not recognised:                                                                                 # nothing structured: the line-by-line readers, capped
        fallbacks.append({"step": "layout", "reason": "no statement was recognised by the table reader"})
        legacy = _legacy(data, name, kind, hint)
        if legacy is not None and lines_found(legacy) > 0:
            fallbacks.append({"step": "legacy-reader", "reason": "the line-by-line reader found lines"})
            legacy["fallbacks"] = fallbacks
            draft = cap_legacy(legacy)
    elif not hint.get("no_fallback"):
        draft, doc = _targeted(data, kind, doc, draft, hint, fallbacks)
    draft["fallbacks"] = fallbacks
    draft["assessment"] = assess(draft) if draft.get("statements") is not None else {"reliable": False, "unreliable": [], "pages": []}
    draft["versions"] = {"extractor": taxonomy.EXTRACTOR_VERSION, "mapping": (draft.get("standard") or {}).get("mapping_version"), "taxonomy": taxonomy.TAXONOMY_VERSION, "reader": doc.reader.get("name")}
    if not draft["periods"]:
        raise FundamentalsError("no statement could be recognised in this document (no table was classified as an income statement, balance sheet or cash flow statement with readable period columns)")
    return draft, doc


def _targeted(data: bytes, kind: str, doc: Document, draft: dict, hint: dict, fallbacks: list) -> tuple[dict, Document]:
    a = assess(draft)
    if a["reliable"]:
        return draft, doc
    pages = a["pages"] or sorted({p for s in draft.get("statements", []) if s["kind"] in ("IS", "BS", "CF") for p in (s.get("pages") or [s["page"]]) if p})
    if hint.get("force_alt_pages"):
        pages = sorted(set(pages) | {int(p) for p in hint["force_alt_pages"]})
    fallbacks.append({"step": "assessment", "reason": "; ".join(b["detail"] for b in a["unreliable"][:4]), "pages": pages})
    if kind != "pdf" or not pages:
        return draft, doc
    key = (hashlib.sha256(data).hexdigest(), tuple(pages), "second-reading")
    if pdfplumber_layout.available():
        if key in _CACHE:
            alt = _CACHE[key]
        else:
            tables = pdfplumber_layout.tables_for_pages(data, pages)
            alt_doc = _replace_pages(doc, tables)
            alt = {"doc": alt_doc, "draft": matcher.build(alt_doc, hint)}
            _CACHE[key] = alt
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
        before, after = score(draft), score(alt["draft"])
        take = after > before
        fallbacks.append({"step": "second-table-reading", "pages": pages, "reason": "re-read the affected pages with an independent word splitter and looser grouping", "before": list(before), "after": list(after), "accepted": take})
        if take:
            draft, doc = alt["draft"], alt["doc"]
            if assess(draft)["reliable"]:
                return draft, doc
    else:
        fallbacks.append({"step": "second-table-reading", "pages": pages, "reason": "pdfplumber is not installed", "accepted": False})
    # OCR of the affected pages only, if an engine is there
    from . import ocr as ocr_module
    if hint.get("ocr_engine") is not None or ocr_module.available():
        try:
            words = ocr_module.pages_subset(data, hint.get("language") or "en", hint.get("ocr_engine"), pages)
            scale = 72.0 / ocr_module.RENDER_DPI
            counter = [20_000]
            tables = {n: pdf_layout._build_page(n, [(w[0] * scale, w[1] * scale, w[2] * scale, w[3]) for w in ws], "pdf-ocr", counter)[0] for n, ws in words.items()}
            ocr_doc = _replace_pages(doc, tables)
            ocr_draft = matcher.build(ocr_doc, hint)
            before, after = score(draft), score(ocr_draft)
            take = after > before
            fallbacks.append({"step": "ocr-pages", "pages": pages, "reason": "OCR of the affected pages only", "before": list(before), "after": list(after), "accepted": take})
            if take:
                draft, doc = ocr_draft, ocr_doc
        except FundamentalsError as error:
            fallbacks.append({"step": "ocr-pages", "pages": pages, "reason": str(error)[:120], "accepted": False})
    else:
        fallbacks.append({"step": "ocr-pages", "pages": pages, "reason": "no OCR engine is installed", "accepted": False})
    # an AI reading, opt-in: only for critical lines still missing, only what the text itself contains, never trusted
    from . import ai_extract
    if ai_extract.enabled() and (draft.get("validation") or {}).get("missing_critical"):
        try:
            text = " ".join(t.text() for t in doc.tables if (t.page in pages or any(p in pages for p in (t.pages or []))))
            got = ai_extract.extract(text, hint.get("ai_client"), hint.get("language") or "en")
            added = _merge_ai(draft, got)
            fallbacks.append({"step": "ai", "pages": pages, "reason": "critical lines still missing", "accepted": bool(added), "added": added})
            if added:
                draft["checks"] = checks.run(draft, None, draft.get("conflicts_between_candidates"))
                draft["confidence"] = checks.confidence_summary(draft)
                draft["validation"] = checks.validation(draft)
        except FundamentalsError as error:
            fallbacks.append({"step": "ai", "pages": pages, "reason": str(error)[:120], "accepted": False})
    return draft, doc


def _merge_ai(draft: dict, ai: dict) -> list[str]:
    """Lines the AI found (each verified verbatim in the document by :mod:`ai_extract`) that the deterministic reading lacks, for periods it has; stored with confidence 0.5, so unverified."""
    added = []
    by_end = {p["period_end"]: p for p in draft["periods"]}
    for ap in ai.get("periods", []):
        p = by_end.get(ap["period_end"])
        if not p:
            continue
        for code, v in ap["values"].items():
            if code in checks.CRITICAL and code not in p["values"]:
                p["values"][code] = v
                p.setdefault("sources", {})[code] = {**(ap.get("sources") or {}).get(code, {}), "confidence": 0.5, "method": "ai-assisted", "statement": "unknown (AI reading)", "page": None}
                added.append(f"{code} {ap['period_end']}")
    return added


def _legacy(data: bytes, name: str, kind: str, hint: dict):
    try:
        if kind == "pdf":
            from . import pdf as legacy_pdf
            return legacy_pdf.extract(data, hint.get("ocr_engine"), hint.get("language") or "en")
        if kind == "html":
            from . import html as legacy_html
            return legacy_html.extract(data)
    except FundamentalsError:
        return None
    return None


__all__ = ["extract", "read", "lines_found", "assess", "score", "cap_legacy"]
