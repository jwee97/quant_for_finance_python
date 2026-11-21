"""The extraction engine: bytes -> IR -> statements -> taxonomy -> checked draft, deterministic first, with the fallbacks marked.

``extract(data, name, kind, hint)`` returns ``(draft, document)``. The draft is the importer's existing draft (so the review, approval and storage code is unchanged) with extra, optional keys:
``statements``, ``checks``, ``confidence``, ``rejections``, ``standard``, ``fallbacks`` and ``versions``. The IR is returned so it can be stored beside the original bytes.

Chain: layout-aware table reader (PDF text layer, HTML, workbook) -> OCR for a scanned PDF -> if too little is found, the legacy line reader and Docling's tables -> an AI reading only if switched on,
and only figures found in the document's own text. Every step that was needed is listed in ``fallbacks`` with the reason.
"""

from __future__ import annotations

from ..errors import FundamentalsError
from . import matcher, taxonomy
from .ir import Document
from .layout import html_layout, pdf_layout, xlsx_layout

MIN_LINES = 4


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


LEGACY_CONFIDENCE = 0.6


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


def extract(data: bytes, name: str, kind: str, hint: dict | None = None) -> tuple[dict, Document]:
    hint = hint or {}
    doc = read(data, kind, hint)
    doc.source_name = name
    fallbacks: list[dict] = []
    if doc.format == "pdf-scan" and not doc.tables:
        raise FundamentalsError(doc.warnings[0] if doc.warnings else "scanned PDF with no OCR engine")
    draft = matcher.build(doc, hint)
    recognised = any(st["confidence"] > 0 and st["kind"] in ("IS", "BS", "CF") for st in draft.get("statements", []))
    if lines_found(draft) < MIN_LINES and not recognised:                                             # a recognised statement is trusted even when short: the line-by-line reader has no statement context
        fallbacks.append({"step": "layout", "reason": f"only {lines_found(draft)} statement line(s) found by the table reader"})
        legacy = _legacy(data, name, kind, hint)
        if legacy is not None and lines_found(legacy) > lines_found(draft):
            fallbacks.append({"step": "legacy-reader", "reason": "the line-by-line reader found more"})
            legacy["fallbacks"] = fallbacks
            draft = cap_legacy(legacy)
    draft["fallbacks"] = fallbacks
    draft["versions"] = {"extractor": taxonomy.EXTRACTOR_VERSION, "mapping": (draft.get("standard") or {}).get("mapping_version"), "taxonomy": taxonomy.TAXONOMY_VERSION,
                         "reader": doc.reader.get("name")}
    if not draft["periods"]:
        raise FundamentalsError("no statement could be recognised in this document (no table was classified as an income statement, balance sheet or cash flow statement with readable period columns)")
    return draft, doc


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


__all__ = ["extract", "read", "lines_found"]
