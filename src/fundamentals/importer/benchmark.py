"""Benchmark runner for real filings: a directory of PDFs and a separate, verified reference file in; results, failures and a summary out.

    python -m src.fundamentals.importer.benchmark --filings ./test_filings --expected ./test_filings/expected.json --output ./benchmark_results

``expected.json`` (verified by a person against the original report; never read by the importer)::

    {"filings": [{"file": "samsung_fy2020.pdf", "issuer": "Samsung Electronics", "split": "development",
                  "hint": {"country": "KR"},                         # what the importer would know: country, optionally language / standard
                  "expected": {"currency": "KRW", "unit_multiplier": 1000000, "period_end": "2020-12-31", "period_type": "FY",
                               "scope": "consolidated", "standard": "k-ifrs",
                               "values": {"revenue": 236806988, "total_assets": 378235718},     # as printed in the report's unit
                               "other_periods": {"2019-12-31": {"revenue": 230400881}}}}]}       # optional, further periods

Metric names are the importer's (revenue, operating_income, net_income, net_income_attributable, total_assets, total_liabilities, shareholders_equity, temporary_equity, operating_cash_flow, ...).
Observations are compared by **metric, period, period type, currency, unit and scope**, not by value alone. A filing's ``split`` is ``development`` or ``heldout``; the summary reports them separately.
Nothing is claimed about a filing that was not processed: a missing file is reported as ``not_processed``.

Outputs in the output directory: ``results.json`` (everything), ``results.csv`` (one row per expected observation), ``failures.json`` (each wrong or missing observation with its root-cause trail: every
candidate that matched, the table and column it came from, why it was rejected or ranked lower, and the reconciliation and fallback record) and ``summary.html``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
import tracemalloc
from decimal import Decimal
from pathlib import Path

from ..errors import FundamentalsError
from . import checks, engine

UNIT = {"units": 1, "thousands": 1_000, "millions": 1_000_000, "billions": 1_000_000_000}
SIGN_FLEX = ("cost_of_revenue", "income_tax", "capital_expenditure")                                  # printed negative by some companies, positive by others
PER_SHARE = ("eps_basic", "eps_diluted", "shares_outstanding", "weighted_avg_shares_diluted")


def _kind(path: Path) -> str:
    return {".pdf": "pdf", ".htm": "html", ".html": "html", ".xlsx": "xlsx", ".csv": "csv"}.get(path.suffix.lower(), "pdf")


def _observed(draft: dict) -> dict[tuple, dict]:
    """``{(metric, period_end, period_type): {value (absolute, Decimal), currency, unit_multiplier, scope, confidence, source}}`` for every selected observation of a draft."""
    out = {}
    f = UNIT.get(draft["units"]["money"], 1)
    for p in draft["periods"]:
        ptype = {"ANNUAL": "FY", "QUARTER": "Q", "HALF_YEAR": "H1", "INSTANT": "FY"}.get(p.get("period_type"), "FY" if p["fiscal_period"] == "FY" else p["fiscal_period"][:1])
        for code, v in p["values"].items():
            src = (p.get("sources") or {}).get(code, {})
            mult = 1 if code in PER_SHARE else src.get("unit_multiplier", f)
            raw = Decimal(src["raw_value"]) if src.get("raw_value") else Decimal(str(v))
            out[(code, p["period_end"], "FY" if p["fiscal_period"] == "FY" else p["fiscal_period"])] = {"value": raw * Decimal(str(mult)), "currency": src.get("currency") or p.get("currency") or draft.get("currency"), "unit_multiplier": mult,
                                                                                                         "scope": src.get("scope", "unspecified"), "confidence": src.get("confidence"), "source": {k: src.get(k) for k in ("page", "table", "row", "col", "label", "statement", "column_role", "column_header")}}
    return out


def _periods_of(entry: dict) -> list[tuple[str, str, dict]]:
    """``[(period_end, period_type, {metric: value})]``: the main period, ``other_periods`` (annual comparatives) and ``other_observations`` (for example a quarter inside an annual report)."""
    e = entry["expected"]
    out = [(e["period_end"], e.get("period_type", "FY"), e.get("values") or {})]
    for end, vals in (e.get("other_periods") or {}).items():
        out.append((end, "FY", vals))
    for o in e.get("other_observations") or []:
        out.append((o["period_end"], o["period_type"], o["values"]))
    return out


def run_filing(path: Path, entry: dict, trace: bool = False) -> dict:
    e = entry["expected"]
    res: dict = {"file": entry["file"], "issuer": entry.get("issuer"), "split": entry.get("split", "development"), "processed": False, "error": None, "observations": [], "spurious_periods": []}
    if not path.exists():
        res["status"] = "not_processed"
        res["error"] = "file not found"
        return res
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        draft, _doc = engine.extract(path.read_bytes(), path.name, _kind(path), {**(entry.get("hint") or {}), "trace": trace})
    except FundamentalsError as error:
        res.update(status="extraction_failed", error=str(error)[:300], seconds=round(time.perf_counter() - t0, 3))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        res["peak_memory_mb"] = round(peak / 1e6, 1)
        for end, pt, vals in _periods_of(entry):
            for m, v in vals.items():
                res["observations"].append({"metric": m, "period_end": end, "period_type": pt, "expected": v * e["unit_multiplier"], "status": "missing", "reason": "extraction failed"})
        return res
    res["seconds"] = round(time.perf_counter() - t0, 3)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    res["peak_memory_mb"] = round(peak / 1e6, 1)
    res.update(processed=True, status="processed")
    got = _observed(draft)
    ptype = e.get("period_type", "FY")
    stm = {s["kind"] for s in draft.get("statements", []) if s.get("confidence", 0) > 0}
    res["statements_found"] = sorted(stm & {"IS", "BS", "CF"})
    res["statement_coverage"] = round(len(stm & {"IS", "BS", "CF"}) / 3, 3)
    unit_found = UNIT.get(draft["units"]["money"], 1)
    res["currency_ok"] = draft.get("currency") == e["currency"]
    res["unit_ok"] = unit_found == e["unit_multiplier"]
    res["currency_found"], res["unit_found"] = draft.get("currency"), unit_found
    res["standard_found"] = (draft.get("standard") or {}).get("basis")
    res["standard_ok"] = (e.get("standard") is None) or res["standard_found"] == e.get("standard")
    exp_ends = {end for end, _, _ in _periods_of(entry)}
    found_ends = {p["period_end"] for p in draft["periods"]}
    res["period_ok"] = e["period_end"] in found_ends and any(p["period_end"] == e["period_end"] and (p["fiscal_period"] == ptype or ptype == "FY" and p["fiscal_period"] == "FY") for p in draft["periods"])
    res["spurious_periods"] = sorted(found_ends - exp_ends) if e.get("strict_periods") else []
    res["validation"] = draft.get("validation", {}).get("status")
    res["validation_reasons"] = [r["kind"] for r in (draft.get("validation") or {}).get("reasons", [])]
    res["checks"] = {s: sum(1 for c in draft.get("checks", []) if c["status"] == s) for s in ("PASS", "FAIL", "INSUFFICIENT_DATA", "NOT_APPLICABLE")}
    res["fallbacks"] = [f["step"] for f in draft.get("fallbacks", []) if f["step"] != "assessment"]
    res["assessment"] = (draft.get("assessment") or {}).get("unreliable", [])
    cands = draft.get("candidates") or []
    res["raw"] = raw_coverage(draft, entry)
    for end, pt, vals in _periods_of(entry):
        for m, v in vals.items():
            exp = Decimal(str(v)) * Decimal(e["unit_multiplier"])
            g = got.get((m, end, pt))
            obs = {"metric": m, "period_end": end, "period_type": pt, "expected": float(exp), "expected_currency": e["currency"], "expected_unit_multiplier": e["unit_multiplier"], "expected_scope": e.get("scope", "consolidated")}
            if g is None:
                obs.update(status="missing", extracted=None)
            else:
                a, b = (abs(g["value"]), abs(exp)) if m in SIGN_FLEX else (g["value"], exp)
                same_value = abs(a - b) <= Decimal("1e-6") * max(Decimal(1), abs(b))
                same_ctx = g["currency"] == e["currency"] and g["unit_multiplier"] == (1 if m in PER_SHARE else e["unit_multiplier"]) and (e.get("scope", "consolidated") in (g["scope"], "unspecified") or g["scope"] == "unspecified")
                obs.update(extracted=float(g["value"]), extracted_currency=g["currency"], extracted_unit_multiplier=g["unit_multiplier"], extracted_scope=g["scope"], confidence=g["confidence"], source=g["source"],
                           status="correct" if same_value and same_ctx else "wrong")
                if same_value and not same_ctx:
                    obs["reason"] = "right number, wrong currency / unit / scope"
            if obs["status"] != "correct":
                obs["candidates"] = [c for c in cands if c["metric"] == m][:12]
                obs["root_cause_hint"] = _hint(obs, draft)
            res["observations"].append(obs)
    return res


def raw_coverage(draft: dict, entry: dict) -> dict:
    """Raw extraction coverage, kept apart from canonical coverage: how many of the expected figures appear **anywhere** among the observations the scanner archived (whatever line it guessed, whatever the
    column), and how many archived observations are unmapped. A figure found here but missing from the canonical result was read correctly and mislabelled or mis-ranked; a figure missing here was never read."""
    e = entry["expected"]
    raw = draft.get("raw_observations") or []
    scale = {"units": 1, "thousands": 1e3, "millions": 1e6, "billions": 1e9, "percent": 1}
    found = {(round(abs(o["value"]) * scale.get(o.get("unit") or "units", 1), 2)) for o in raw if not o.get("is_percent")}
    exp = [(m, end, abs(v) * e["unit_multiplier"]) for end, pt, vals in _periods_of(entry) for m, v in vals.items()]
    hit = [x for x in exp if round(x[2], 2) in found or round(x[2] / e["unit_multiplier"], 2) in {round(abs(o["value"]), 2) for o in raw}]
    return {"observations": len(raw), "unmapped": sum(1 for o in raw if not o.get("metric_guess")), "selected": sum(1 for o in raw if o.get("selected")), "expected": len(exp), "expected_in_archive": len(hit),
            "raw_recall": round(len(hit) / len(exp), 4) if exp else None, "groups": {g: sum(1 for o in raw if o.get("group") == g) for g in sorted({o.get("group") for o in raw})},
            "missing_from_archive": [{"metric": m, "period_end": end, "expected": x} for m, end, x in exp if (m, end, x) not in hit][:50]}


def workflow(path: Path, entry: dict) -> dict:
    """The review workflow on this filing in a throw-away database: nothing is stored before a person confirms; confirming stores exactly the confirmed figures; a decision survives reading the file again;
    unmapped observations remain available; the research tables never contain an unconfirmed figure."""
    import tempfile
    from ..providers.base import CompanyProfile
    from ..store import FundamentalsStore
    from .pipeline import ImportService
    out: dict = {"ran": True}
    try:
        with tempfile.TemporaryDirectory() as d:
            store = FundamentalsStore(Path(d) / "bench.db")
            store.upsert_security(CompanyProfile("manual", "bench:1", "BENCH", entry.get("issuer") or "Benchmark", exchange="X", market="US", country="US", currency=entry["expected"]["currency"]))
            sid = int(store.securities().iloc[0]["security_id"])
            svc = ImportService(store)
            batch = svc.create_batch("benchmark", sid)
            item = svc.add_file(batch, path.name, path.read_bytes())[0]["item_id"]
            svc.process(batch, hint={**(entry.get("hint") or {}), "filing_date": "2021-03-01"})
            rows = svc.observations(item, limit=1000)
            out["archived"], out["unmapped_archived"] = rows["total"], svc.observations(item, unmapped=True)["total"]
            out["nothing_stored_before_confirmation"] = store.observations([sid]).empty
            out["all_start_unconfirmed"] = all(o["status"] == "UNCONFIRMED" for o in rows["rows"])
            svc.review(item, {"confirm_periods": True})
            some = [o["obs_id"] for o in rows["rows"] if o["selected"]][:3]
            svc.decide(item, [{"action": "confirm", "ids": some}])
            svc.reprocess(item, entry.get("hint") or {})
            out["confirmation_persists_after_reprocessing"] = all(svc.observation(i)["status"] in ("CONFIRMED", "CORRECTED_AND_CONFIRMED") for i in some)
            plan = svc.plan(item)
            if plan["approvable"]:
                svc.approve(item, security={"security_id": sid})
                stored = store.observations([sid])
                direct = set(stored[stored["quality_flag"] != "derived"]["metric_code"])                       # exact arithmetic on confirmed figures is marked derived and may be added
                out["stored_only_confirmed"] = direct == {l["metric"] for l in plan["will_commit"]} and len(plan["will_commit"]) <= len(some)
                out["research_gate_holds"] = bool((stored["quality_flag"] != "unverified").all()) if len(stored) else True
            else:
                out["stored_only_confirmed"] = out["research_gate_holds"] = None
                out["note"] = "the confirmed figures could not be published (see the plan)"
    except Exception as error:                                                                            # a workflow failure is a finding, not a crash of the benchmark
        out["error"] = f"{type(error).__name__}: {str(error)[:200]}"
    return out


def _hint(obs: dict, draft: dict) -> str:
    """The first place the expected value could have been lost, from the candidates the engine kept."""
    cands = [c for c in draft.get("candidates") or [] if c["metric"] == obs["metric"]]
    if not cands:
        return "no row mapped to this metric: no label matched in a table classified as a statement (check the taxonomy and statement classification)"
    for c in cands:
        if abs(float(c["raw_value"]) * (UNIT.get(c.get("unit") or "units", 1)) - obs["expected"]) <= 1e-6 * max(1, abs(obs["expected"])):
            return f"the expected value was found as a candidate (table {c['table_id']}, page {c['page']}, column {c['column_role']}) with status {c['status']}: {'; '.join(c['reasons']) or 'ranked'}"
    return "the expected value is not among the candidates: the cell was not read as that number (table reconstruction or column resolution) or the row label did not map"


def summarize(results: list[dict]) -> dict:
    def block(rs):
        ex = [o for r in rs for o in r["observations"]]
        crit = [o for o in ex if o["metric"] in checks.CRITICAL or o["metric"] in ("net_income_attributable", "operating_income", "total_liabilities")]
        def rate(a, b):
            return round(a / b, 4) if b else None
        auto = [o for r in rs if r.get("validation") == "VALIDATED" for o in r["observations"] if o["metric"] in checks.CRITICAL and o["status"] != "missing"]
        return {"filings": len(rs), "processed": sum(1 for r in rs if r["processed"]), "processing_success": rate(sum(1 for r in rs if r["processed"]), len(rs)),
                "statement_coverage": rate(sum(r.get("statement_coverage", 0) for r in rs if r["processed"]), sum(1 for r in rs if r["processed"])),
                "expected_observations": len(ex), "correct": sum(1 for o in ex if o["status"] == "correct"), "wrong": sum(1 for o in ex if o["status"] == "wrong"), "missing": sum(1 for o in ex if o["status"] == "missing"),
                "recall": rate(sum(1 for o in ex if o["status"] == "correct"), len(ex)), "coverage": rate(sum(1 for o in ex if o["status"] != "missing"), len(ex)),
                "precision_of_extracted": rate(sum(1 for o in ex if o["status"] == "correct"), sum(1 for o in ex if o["status"] != "missing")),
                "critical_metric_accuracy": rate(sum(1 for o in crit if o["status"] == "correct"), len(crit)),
                "auto_approvable_critical": len(auto), "auto_approvable_critical_precision": rate(sum(1 for o in auto if o["status"] == "correct"), len(auto)),
                "review_required_filings": sum(1 for r in rs if r.get("validation") == "REVIEW_REQUIRED"), "rejection_or_review_rate": rate(sum(1 for r in rs if r["processed"] and r.get("validation") != "VALIDATED"), sum(1 for r in rs if r["processed"])),
                "currency_accuracy": rate(sum(1 for r in rs if r.get("currency_ok")), sum(1 for r in rs if r["processed"])), "unit_accuracy": rate(sum(1 for r in rs if r.get("unit_ok")), sum(1 for r in rs if r["processed"])),
                "period_accuracy": rate(sum(1 for r in rs if r.get("period_ok")), sum(1 for r in rs if r["processed"])), "standard_accuracy": rate(sum(1 for r in rs if r.get("standard_ok")), sum(1 for r in rs if r["processed"])),
                "false_positive_observations": sum(1 for o in ex if o["status"] == "wrong") + sum(len(r["spurious_periods"]) for r in rs),
                "fallback_usage": {k: sum(1 for r in rs if k in (r.get("fallbacks") or [])) for k in sorted({f for r in rs for f in (r.get("fallbacks") or [])})},
                "median_seconds": sorted(r.get("seconds", 0) for r in rs)[len(rs) // 2] if rs else None, "max_peak_memory_mb": max((r.get("peak_memory_mb", 0) for r in rs), default=None)}
    def wf(rs, key):
        v = [r["workflow"][key] for r in rs if r.get("workflow") and r["workflow"].get(key) is not None]
        return round(sum(1 for x in v if x) / len(v), 4) if v else None

    def block_plus(rs):
        b = block_core(rs)
        raw = [r["raw"] for r in rs if r.get("raw")]
        b.update(raw_observations=sum(r["observations"] for r in raw), unmapped_observations=sum(r["unmapped"] for r in raw), raw_recall=(round(sum(r["expected_in_archive"] for r in raw) / max(1, sum(r["expected"] for r in raw)), 4) if raw else None),
                 canonical_recall=b["recall"], workflow_filings=sum(1 for r in rs if r.get("workflow")), workflow_errors=sum(1 for r in rs if (r.get("workflow") or {}).get("error")),
                 workflow_nothing_stored_before_confirmation=wf(rs, "nothing_stored_before_confirmation"), workflow_confirmation_persists=wf(rs, "confirmation_persists_after_reprocessing"),
                 workflow_stored_only_confirmed=wf(rs, "stored_only_confirmed"), workflow_research_gate_holds=wf(rs, "research_gate_holds"))
        return b
    block_core = block
    block = block_plus
    out = {"all": block(results)}
    for split in sorted({r["split"] for r in results}):
        out[split] = block([r for r in results if r["split"] == split])
    return out


def write_outputs(results: list[dict], summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "filings": results}, indent=1, default=str, ensure_ascii=False))
    (out / "failures.json").write_text(json.dumps([{"file": r["file"], "issuer": r.get("issuer"), "status": r["status"], "error": r.get("error"), "validation_reasons": r.get("validation_reasons"), "fallbacks": r.get("fallbacks"),
                                                     "assessment": r.get("assessment"), "failed_observations": [o for o in r["observations"] if o["status"] != "correct"]} for r in results
                                                    if r["status"] != "processed" or any(o["status"] != "correct" for o in r["observations"]) or r.get("spurious_periods")], indent=1, default=str, ensure_ascii=False))
    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "issuer", "split", "metric", "period_end", "status", "expected", "extracted", "expected_currency", "extracted_currency", "expected_unit_multiplier", "extracted_unit_multiplier", "scope", "confidence", "page", "reason"])
        for r in results:
            for o in r["observations"]:
                w.writerow([r["file"], r.get("issuer"), r["split"], o["metric"], o["period_end"], o["status"], o.get("expected"), o.get("extracted"), o.get("expected_currency"), o.get("extracted_currency"),
                            o.get("expected_unit_multiplier"), o.get("extracted_unit_multiplier"), o.get("extracted_scope"), o.get("confidence"), (o.get("source") or {}).get("page"), o.get("reason") or o.get("root_cause_hint", "")])
    rows = "".join(f"<tr><td>{html.escape(str(r['file']))}</td><td>{html.escape(str(r.get('status')))}</td><td>{html.escape(str(r.get('validation')))}</td><td>{sum(1 for o in r['observations'] if o['status'] == 'correct')}/{len(r['observations'])}</td>"
                   f"<td>{html.escape(', '.join(r.get('fallbacks') or []))}</td><td>{r.get('seconds', '')}</td></tr>" for r in results)
    srows = "".join(f"<tr><td>{html.escape(k)}</td>" + "".join(f"<td>{html.escape(str(v.get(m)))}</td>" for m in ("filings", "processing_success", "critical_metric_accuracy", "raw_recall", "recall", "precision_of_extracted", "auto_approvable_critical_precision", "rejection_or_review_rate", "false_positive_observations")) + "</tr>" for k, v in summary.items())
    (out / "summary.html").write_text(f"<!doctype html><meta charset=utf-8><title>Importer benchmark</title><style>body{{font:14px system-ui;margin:24px}}td,th{{border:1px solid #8886;padding:4px 8px}}table{{border-collapse:collapse;margin:12px 0}}</style>"
                                      f"<h1>Importer benchmark</h1><p>{len(results)} filing(s). A filing that was not processed is not counted as passed.</p><h2>Summary</h2><table><tr><th>split</th><th>filings</th><th>processed</th><th>critical accuracy</th><th>raw recall</th><th>recall</th><th>precision (extracted)</th><th>precision (auto-approvable critical)</th><th>review rate</th><th>false positives</th></tr>{srows}</table>"
                                      f"<h2>Filings</h2><table><tr><th>file</th><th>status</th><th>validation</th><th>correct</th><th>fallbacks</th><th>seconds</th></tr>{rows}</table><p>See failures.json for the root-cause trail of every wrong or missing observation.</p>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.fundamentals.importer.benchmark", description="Run the importer over a directory of filings and compare with verified reference values.")
    ap.add_argument("--filings", required=True, help="directory holding the filings")
    ap.add_argument("--expected", required=True, help="the reference file (see the module docstring)")
    ap.add_argument("--output", required=True, help="directory for results.json, results.csv, failures.json, summary.html")
    ap.add_argument("--no-workflow", action="store_true", help="skip the review-workflow checks (confirmation, persistence, research gating) that run on every processed filing")
    ap.add_argument("--trace", action="store_true", help="keep the structured decision trace of every filing in results.json")
    a = ap.parse_args(argv)
    expected = json.loads(Path(a.expected).read_text())
    results = [run_filing(Path(a.filings) / f["file"], f, a.trace) for f in expected["filings"]]
    if not a.no_workflow:
        for r, f in zip(results, expected["filings"]):
            if r["processed"]:
                r["workflow"] = workflow(Path(a.filings) / f["file"], f)
    summary = summarize(results)
    write_outputs(results, summary, Path(a.output))
    allb = summary["all"]
    print(json.dumps({k: allb[k] for k in ("filings", "processed", "critical_metric_accuracy", "recall", "precision_of_extracted", "auto_approvable_critical_precision", "rejection_or_review_rate", "false_positive_observations")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
