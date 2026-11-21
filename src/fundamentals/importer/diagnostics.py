"""Importer diagnostics: how well is the importer doing, from what it has actually read and from the benchmark.

* **live** numbers come from the importer's own tables: every item read, every extraction version, every draft's statements, checks, fallbacks and rejections, and the provenance of stored figures.
  They show *success*, *coverage*, *reconciliation outcomes*, *fallback usage*, *rejection reasons* and how often a person had to correct a figure (a proxy for accuracy: the corrected share of figures,
  and of the critical ones). They cannot show true accuracy: nothing here knows the right answer.
* **benchmark** numbers come from the last run of the benchmark harness (``benchmarks/fundamentals_importer/results``), which does know the right answer for its (synthetic) documents: line recall,
  critical-line accuracy, period / unit / currency accuracy, confident errors.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import checks, taxonomy

RESULTS = Path(__file__).resolve().parents[3] / "benchmarks" / "fundamentals_importer" / "results"


def live(store) -> dict:
    with store.db.transaction() as c:
        items = [dict(r) for r in c.execute("SELECT item_id, source_kind, status, method, error, draft FROM import_items WHERE status <> 'duplicate'").fetchall()]
        prov = c.execute("SELECT COUNT(*), SUM(edited), SUM(CASE WHEN verified = 0 THEN 1 ELSE 0 END), SUM(CASE WHEN confidence IS NOT NULL THEN 1 ELSE 0 END) FROM observation_provenance").fetchone()
        crit = c.execute(f"SELECT COUNT(*), SUM(edited) FROM observation_provenance WHERE metric_code IN ({','.join('?' * len(checks.CRITICAL))})", checks.CRITICAL).fetchone()
        unverified = c.execute("SELECT COUNT(*) FROM fundamental_observations WHERE quality_flag = 'unverified'").fetchone()[0]
        versions = [dict(r) for r in c.execute("SELECT extractor_version, mapping_version, standard, COUNT(*) n FROM import_extractions GROUP BY 1, 2, 3 ORDER BY n DESC").fetchall()]
    total = len(items)
    drafts = []
    for it in items:
        if it["draft"]:
            try:
                drafts.append((it, json.loads(it["draft"])))
            except ValueError:
                pass
    failed = [it for it in items if it["status"] == "failed"]
    by_status = Counter(it["status"] for it in items)
    by_format = Counter(it["source_kind"] for it in items)
    ok_by_format = Counter(it["source_kind"] for it, _ in drafts)
    fallbacks = Counter(f["step"] for _, d in drafts for f in d.get("fallbacks", []))
    methods = Counter(d.get("method") or it["method"] for it, d in drafts)
    stmt_found = Counter()
    stmt_n = 0
    checks_c: Counter = Counter()
    check_ids: dict[str, Counter] = {}
    rejections: Counter = Counter()
    unit_stated = guessed = 0
    conf_low = conf_n = 0
    for it, d in drafts:
        kinds = {s["kind"] for s in d.get("statements", []) if s.get("confidence", 0) > 0}
        if d.get("statements") is not None and "statements" in d:
            stmt_n += 1
            for k in ("IS", "BS", "CF"):
                stmt_found[k] += k in kinds
        for ch in d.get("checks", []):
            checks_c[ch["status"]] += 1
            check_ids.setdefault(ch["id"], Counter())[ch["status"]] += 1
        for r in d.get("rejections", []):
            rejections[r["reason"].split(":")[0][:70]] += 1
        unit_stated += bool(d.get("units_stated", True))
        for p in d.get("periods", []):
            guessed += bool(p.get("guessed"))
            for s in (p.get("sources") or {}).values():
                if s.get("confidence") is not None:
                    conf_n += 1
                    conf_low += s["confidence"] < checks.LOW_CONFIDENCE
    return {
        "items": total, "by_status": dict(by_status), "extraction_success": round(len(drafts) / total, 3) if total else None,
        "success_by_format": {k: round(ok_by_format[k] / v, 3) for k, v in by_format.items()},
        "failure_reasons": dict(Counter((it["error"] or "unknown")[:80] for it in failed).most_common(10)),
        "statement_coverage": {k: round(stmt_found[k] / stmt_n, 3) for k in ("IS", "BS", "CF")} if stmt_n else None, "documents_with_statements": stmt_n,
        "reconciliation": dict(checks_c), "reconciliation_by_check": {k: dict(v) for k, v in check_ids.items()},
        "fallback_usage": dict(fallbacks), "fallback_rate": round(sum(1 for _, d in drafts if d.get("fallbacks")) / len(drafts), 3) if drafts else None, "methods": dict(methods),
        "rejection_reasons": dict(rejections.most_common(12)),
        "unit_stated_rate": round(unit_stated / len(drafts), 3) if drafts else None, "periods_with_guessed_dates": guessed,
        "low_confidence_share": round(conf_low / conf_n, 3) if conf_n else None, "figures_scored": conf_n,
        "stored_figures": prov[0] or 0, "corrected_by_a_person": int(prov[1] or 0), "correction_rate": round((prov[1] or 0) / prov[0], 4) if prov[0] else None,
        "critical_stored": crit[0] or 0, "critical_corrected": int(crit[1] or 0), "critical_correction_rate": round((crit[1] or 0) / crit[0], 4) if crit[0] else None,
        "unverified_figures_held_out_of_research": unverified, "extractions_by_version": versions, "extractor_version": taxonomy.EXTRACTOR_VERSION, "taxonomy_version": taxonomy.TAXONOMY_VERSION,
    }


def benchmark() -> dict:
    out: dict = {"available": False}
    for name in ("dev", "validation", "heldout_first_run", "heldout_post_fix"):
        f = RESULTS / f"{name}.json"
        if f.exists():
            r = json.loads(f.read_text())
            out[name] = {"generated": r["generated"], "extractor_version": r["extractor_version"], **r["summary"]["overall"]}
            out["available"] = True
            if name == "heldout_post_fix":
                out["by_market"] = r["summary"]["by_market"]
                out["by_format"] = r["summary"]["by_format"]
                out["rejection_reasons"] = r["summary"]["rejection_reasons"]
    out["note"] = "synthetic documents: they measure layouts, wording and number formats, not real filings; no real filing was benchmarked; OCR was simulated"
    return out


def report(store) -> dict:
    return {"live": live(store), "benchmark": benchmark(), "threshold": checks.LOW_CONFIDENCE}


__all__ = ["live", "benchmark", "report"]
