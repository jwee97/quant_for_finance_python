"""Benchmark harness: run every case through the extraction engine, score against the truth, aggregate.

    python benchmarks/fundamentals_importer/harness.py [--split dev|validation|heldout|all] [--out results/latest.json]

Scores per case: extraction success, recall of the printed lines (and of the five critical lines), wrong values, spurious values, **confident errors** (wrong or spurious values that carry a confidence at
or above the review threshold: the failure that matters, because such a value would be stored as verified), whether wrong values were caught (low confidence or a failed reconciliation), statement
coverage, period accuracy, unit and currency accuracy, reconciliation outcomes, fallback usage and rejection reasons.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src.fundamentals.errors import FundamentalsError  # noqa: E402
from src.fundamentals.importer import checks, engine  # noqa: E402

UNIT = {"units": 1, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
COUNTRY = {"US": "US", "SG": "SG", "JP": "JP", "KR": "KR", "CN": "CN", "HK": "HK"}            # EU reports get no country: the standard is then inferred from the document
REL_TOL = 0.005
PER_SHARE = {"eps_basic", "eps_diluted"}
EQUIVALENT = {"cash": "cash_and_equivalents"}                                             # a label that reads "cash and deposits" or "cash and cash equivalents" is one line for this score


class NoisyOcr:
    """A stand-in for an OCR engine: returns the words of the case's text-layer original, in the pixels of a 200 dpi render, with random digit errors. It measures how the importer copes with OCR *errors*;
    it says nothing about how accurate a real OCR engine is (none is installed in the build environment)."""
    name = "simulated"

    def __init__(self, src_pdf: bytes, noise: float, seed: int = 5):
        import pymupdf
        self.pages = [[(w[0], (w[1] + w[3]) / 2, w[2], w[4]) for w in p.get_text("words")] for p in pymupdf.open(stream=src_pdf, filetype="pdf")]
        self.noise, self.rng, self.n = noise, random.Random(seed), 0

    def words(self, image_png: bytes, language: str):
        words = self.pages[self.n]
        self.n += 1
        k = 200 / 72
        out = []
        for x0, y, x1, t in words:
            if any(ch.isdigit() for ch in t) and self.rng.random() < self.noise:
                i = self.rng.choice([i for i, ch in enumerate(t) if ch.isdigit()])
                t = t[:i] + str((int(t[i]) + self.rng.randint(1, 8)) % 10) + t[i + 1:]
            out.append((x0 * k, y * k, x1 * k, t, 0.9))
        return out


def run_case(case: dict, data_dir: Path) -> dict:
    truth = json.loads((data_dir / case["truth"]).read_text())
    path = data_dir / case["file"]
    data = path.read_bytes()
    hint = {"country": COUNTRY.get(case["market"]), "language": case["language"] if case["language"] in ("ja", "ko", "zh") else "en"}
    if case["scanned"]:
        hint["ocr_engine"] = NoisyOcr((data_dir / case["file"].replace(".pdf", ".src.pdf")).read_bytes(), noise=0.04)
    kind = {"pdf": "pdf", "html": "html", "xlsx": "xlsx"}[case["format"]]
    t0 = time.perf_counter()
    result = {"id": case["id"], "split": case["split"], "market": case["market"], "format": case["format"], "scanned": case["scanned"], "restated": case["restated"], "traps": case["traps"], "language": case["language"],
              "period_type": case["period_type"], "layout": case["layout"], "wording": case["wording"]}
    try:
        draft, _doc = engine.extract(data, path.name, kind, hint)
    except FundamentalsError as error:
        return {**result, "success": False, "error": str(error)[:160], "seconds": round(time.perf_counter() - t0, 3), "expected": sum(len(p["values"]) for p in truth["periods"]), "correct": 0, "wrong": 0, "spurious": 0,
                "confident_errors": 0, "critical_expected": sum(1 for p in truth["periods"] for c in p["values"] if c in checks.CRITICAL), "critical_correct": 0, "statements_expected": 3, "statements_found": 0,
                "periods_expected": len(truth["periods"]), "periods_found": 0, "unit_ok": False, "currency_ok": False, "wrong_caught": 0, "fallbacks": [], "rejections": [], "checks": {}}
    seconds = round(time.perf_counter() - t0, 3)
    factor = UNIT.get(draft["units"]["money"], 1.0)
    got: dict[tuple, tuple] = {}
    for p in draft["periods"]:
        for code, v in p["values"].items():
            money = code not in PER_SHARE and code not in ("shares_outstanding", "weighted_avg_shares_diluted")
            conf = (p.get("sources") or {}).get(code, {}).get("confidence")
            got[(p["period_end"], EQUIVALENT.get(code, code))] = (v * (factor if money else 1.0), conf, p["fiscal_period"])
    expected = {(p["period_end"], EQUIVALENT.get(c, c)): v for p in truth["periods"] for c, v in p["values"].items()}
    kinds = {p["period_end"]: p["fiscal_period"] for p in truth["periods"]}
    correct = wrong = confident_errors = caught = critical_correct = 0
    for key, v in expected.items():
        if key in got:
            g, conf, _fp = got[key]
            if abs(g - v) <= REL_TOL * max(abs(v), 1e-9) or (key[1] in PER_SHARE and abs(g - v) < 0.011):
                correct += 1
                critical_correct += key[1] in checks.CRITICAL
            else:
                wrong += 1
                if conf is None or conf >= checks.LOW_CONFIDENCE:
                    confident_errors += 1
                else:
                    caught += 1
    spurious = [k for k in got if k not in expected]
    for k in spurious:
        if got[k][1] is None or got[k][1] >= checks.LOW_CONFIDENCE:
            confident_errors += 1
    found_periods = {(p["period_end"]) for p in draft["periods"]}
    exp_periods = {p["period_end"] for p in truth["periods"]}
    # a period is right when its end date is found and, for a flow period, its length class (FY / H1 / Q) is too
    kind_ok = 0
    for p in draft["periods"]:
        if p["period_end"] in kinds and (kinds[p["period_end"]][0] == p["fiscal_period"][0]):
            kind_ok += 1
    stmts = {s["kind"] for s in draft.get("statements", []) if s["confidence"] > 0}
    chk = Counter(c["status"] for c in draft.get("checks", []))
    reasons = Counter(r["reason"].split(":")[0][:60] for r in draft.get("rejections", []))
    return {**result, "success": True, "seconds": seconds, "expected": len(expected), "correct": correct, "wrong": wrong, "spurious": len(spurious), "confident_errors": confident_errors, "wrong_caught": caught,
            "critical_expected": sum(1 for k in expected if k[1] in checks.CRITICAL), "critical_correct": critical_correct,
            "statements_expected": 3, "statements_found": len(stmts & {"IS", "BS", "CF"}), "periods_expected": len(exp_periods), "periods_found": len(found_periods & exp_periods), "period_kind_ok": kind_ok,
            "unit_ok": draft["units"]["money"] == truth["unit"], "currency_ok": draft.get("currency") == truth["currency"], "fallbacks": [f["step"] for f in draft.get("fallbacks", [])],
            "rejections": dict(reasons), "checks": dict(chk), "standard": (draft.get("standard") or {}).get("standard"), "standard_ok": (draft.get("standard") or {}).get("standard") == truth["standard"],
            "mean_confidence": (draft.get("confidence") or {}).get("mean"), "lines_found": len(got)}


def aggregate(rows: list[dict]) -> dict:
    def agg(rs):
        n = len(rs)
        if not n:
            return {}
        s = lambda k: sum(r.get(k, 0) for r in rs)  # noqa: E731
        return {"cases": n, "extraction_success": round(sum(r["success"] for r in rs) / n, 3), "recall": round(s("correct") / max(s("expected"), 1), 3),
                "critical_accuracy": round(s("critical_correct") / max(s("critical_expected"), 1), 3), "wrong": s("wrong"), "spurious": s("spurious"), "confident_errors": s("confident_errors"),
                "confident_error_rate": round(s("confident_errors") / max(s("correct") + s("wrong") + s("spurious"), 1), 4), "wrong_caught_rate": round(s("wrong_caught") / max(s("wrong"), 1), 3) if s("wrong") else None,
                "statement_coverage": round(s("statements_found") / max(s("statements_expected"), 1), 3), "period_accuracy": round(s("periods_found") / max(s("periods_expected"), 1), 3),
                "unit_accuracy": round(sum(r["unit_ok"] for r in rs) / n, 3), "currency_accuracy": round(sum(r["currency_ok"] for r in rs) / n, 3), "standard_accuracy": round(sum(r.get("standard_ok", False) for r in rs) / n, 3),
                "fallback_rate": round(sum(1 for r in rs if r["fallbacks"]) / n, 3), "median_seconds": sorted(r["seconds"] for r in rs)[n // 2]}
    out = {"overall": agg(rows)}
    for dim in ("split", "format", "market", "language", "period_type", "layout"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[dim]].append(r)
        out[f"by_{dim}"] = {k: agg(v) for k, v in sorted(groups.items())}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["market"], r["format"] + ("-scan" if r["scanned"] else ""))].append(r)
    out["by_split_market_format"] = {"|".join(k): agg(v) for k, v in sorted(groups.items())}
    out["rejection_reasons"] = dict(Counter(k for r in rows for k, n in r["rejections"].items() for _ in range(n)).most_common(15))
    out["failures"] = [{"id": r["id"], "error": r.get("error")} for r in rows if not r["success"]]
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all")
    ap.add_argument("--out", default=str(HERE / "results" / "latest.json"))
    ap.add_argument("--data", default=str(HERE / "data"))
    a = ap.parse_args(argv)
    data_dir = Path(a.data)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rows = [run_case(c, data_dir) for c in manifest if a.split in ("all", c["split"])]
    from src.fundamentals.importer import taxonomy
    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "extractor_version": taxonomy.EXTRACTOR_VERSION, "taxonomy_version": taxonomy.TAXONOMY_VERSION, "threshold": checks.LOW_CONFIDENCE,
              "summary": aggregate(rows), "cases": rows}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return report


if __name__ == "__main__":
    r = main()
    o = r["summary"]["overall"]
    print(json.dumps(o, indent=1))
    for k, v in r["summary"]["by_split"].items():
        print(k, {kk: v[kk] for kk in ("cases", "extraction_success", "recall", "critical_accuracy", "confident_errors", "statement_coverage", "period_accuracy", "unit_accuracy")})
