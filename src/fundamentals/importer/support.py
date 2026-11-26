"""What the importer supports, and what has been shown about how well it works.

Two different things are kept apart on purpose:

* **supported**  the code path exists and its dependencies are installed here. Nothing is claimed about accuracy.
* **demonstrated**  a benchmark with known answers was run and its results are stored. The benchmark's *kind* matters: documents made up for the benchmark measure layouts, wording and number formats; only
  real filings with answers checked by a person against the original report measure accuracy on real filings. This repository has no such run, so no format is reported as demonstrated on real filings.

The status of a format is the weakest honest sentence the evidence supports:

``demonstrated_on_real_filings``   a stored run on at least ``MIN_REAL`` real, person-verified held-out filings of that format;
``demonstrated_on_synthetic_only`` a stored run on synthetic documents with known answers, none on real filings;
``supported_unbenchmarked``        usable here, no stored benchmark of any kind;
``blocked``                        a dependency or key it needs is missing (and which);
``not_implemented``                no code for it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from .diagnostics import RESULTS

ROOT = Path(__file__).resolve().parents[3]
MIN_REAL = 5

# id, label, benchmark format key, module (relative to src/fundamentals), python packages needed, executables needed, note
FORMATS = [
    ("pdf_text", "PDF with a text layer", "pdf", "importer/pdf.py", ["pymupdf"], [], ""),
    ("pdf_ocr", "Scanned PDF (OCR)", None, "importer/ocr.py", ["pymupdf"], ["tesseract"], "The benchmark's OCR was simulated, not run."),
    ("pdf_docling", "PDF read by Docling", None, "importer/docling_adapter.py", ["docling"], [], "An optional second reader; its output goes through the same checks."),
    ("html", "HTML report", "html", "importer/html.py", [], [], ""),
    ("xlsx", "Excel workbook", "xlsx", "importer/tabular.py", ["openpyxl"], [], ""),
    ("csv", "CSV table", "csv", "importer/tabular.py", [], [], ""),
    ("xbrl", "XBRL / inline XBRL", None, "importer/xbrl.py", [], [], "Tagged facts are read as filed; the taxonomy mapping is checked by the reconciliations."),
    ("ai_assisted", "AI-assisted extraction", None, "importer/ai_extract.py", [], [], "A second, unverified source of numbers: every value must appear verbatim in the document and is reviewed by a person."),
    ("sec_edgar", "SEC EDGAR company facts", None, "providers/sec_edgar.py", [], [], "Needs network access and a contact string in the request header."),
    ("edinet", "EDINET (Japan)", None, "filings/edinet.py", [], [], "Needs an EDINET API key."),
    ("opendart", "OpenDART (Korea)", None, "filings/dart.py", [], [], "Needs an OpenDART API key."),
    ("cvm", "CVM (Brazil)", None, "filings/cvm.py", [], [], ""),
    ("esef", "ESEF (European Union)", None, "filings/esef.py", [], [], ""),
]


def _real_runs() -> list[dict]:
    """Stored benchmark runs on real filings: ``results/real_*.json`` with ``"kind": "real"`` and, per case, ``"verified_by"`` (the person who checked the answers against the original report)."""
    out = []
    for f in sorted(RESULTS.glob("real_*.json")):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("kind") == "real":
            out.append({"file": f.name, **r})
    return out


def _synthetic() -> dict:
    f = RESULTS / "heldout_post_fix.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())["summary"].get("by_format", {})
    except (ValueError, KeyError):
        return {}


def matrix() -> dict:
    synth, real = _synthetic(), _real_runs()
    rows = []
    for key, label, bench_key, module, packages, tools, note in FORMATS:
        path = ROOT / "src" / "fundamentals" / module
        missing = [p for p in packages if importlib.util.find_spec(p) is None] + [t for t in tools if shutil.which(t) is None]
        s = synth.get(bench_key) if bench_key else None
        r_cases = [c for run in real for c in run.get("cases", []) if c.get("format") == bench_key and c.get("verified_by") and c.get("split") == "heldout"] if bench_key else []
        if not path.exists():
            status = "not_implemented"
        elif missing:
            status = "blocked"
        elif len(r_cases) >= MIN_REAL:
            status = "demonstrated_on_real_filings"
        elif s:
            status = "demonstrated_on_synthetic_only"
        else:
            status = "supported_unbenchmarked"
        rows.append({"id": key, "label": label, "status": status, "blocked_by": missing, "note": note,
                     "synthetic": ({k: s.get(k) for k in ("cases", "recall", "critical_accuracy", "wrong", "confident_errors", "statement_coverage")} if s else None),
                     "real_verified_heldout_filings": len(r_cases), "claim": _claim(status, s, len(r_cases), missing)})
    return {"formats": rows, "min_real": MIN_REAL, "real_runs": [r["file"] for r in real],
            "note": "No format is reported as demonstrated on real filings unless a stored run on real, person-verified held-out filings exists. Synthetic documents measure layouts, wording and number formats, not real filings."}


def _claim(status: str, synth, n_real: int, missing: list[str]) -> str:
    if status == "not_implemented":
        return "Not implemented."
    if status == "blocked":
        return f"Cannot run here: needs {', '.join(missing)}."
    if status == "demonstrated_on_real_filings":
        return f"Accuracy shown on {n_real} verified held-out real filings."
    if status == "demonstrated_on_synthetic_only":
        return f"Measured on {synth.get('cases')} synthetic documents (recall {synth.get('recall'):.0%}); not measured on any real filing, so its accuracy on real reports is unverified."
    return "Supported but never benchmarked: its accuracy is unverified."


__all__ = ["matrix", "FORMATS", "MIN_REAL"]
