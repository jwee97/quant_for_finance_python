"""Runs the synthetic look-alikes of the seven real reports through the extraction engine and compares with the expected values: the *reproduction* of the failure classes (see tests/fixtures/real_like.py).

    python benchmarks/fundamentals_importer/real_structures.py [--out results/real_structures_after.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.fundamentals.errors import FundamentalsError  # noqa: E402
from src.fundamentals.importer import engine  # noqa: E402
from tests.fixtures import real_like  # noqa: E402

HINTS = {"toyota": {"country": "JP", "standard": "us-gaap"}, "samsung": {"country": "KR"}, "apple": {"country": "US"}, "singtel": {"country": "SG"}, "byd": {"country": "CN"}, "tencent": {"country": "HK"}, "asml": {}}
FACTOR = {"units": 1, "thousands": 1e3, "millions": 1e6, "billions": 1e9}


def run(name: str) -> dict:
    data, truth = real_like.BUILDERS[name]()
    out = {"issuer": name, "currency_expected": truth["currency"], "metrics": {}, "periods_found": [], "error": None}
    try:
        draft, _ = engine.extract(data, f"{name}.pdf", "pdf", HINTS.get(name, {}))
    except FundamentalsError as error:
        out["error"] = str(error)[:200]
        for m, v in truth["values"].items():
            out["metrics"][m] = {"expected": v * truth["unit"], "extracted": None, "status": "missing"}
        return out
    out["currency_found"] = draft.get("currency")
    out["periods_found"] = sorted({(p["fiscal_period"], p["period_end"]) for p in draft["periods"]})
    f = FACTOR.get(draft["units"]["money"], 1.0)
    for m, v in truth["values"].items():
        exp = v * truth["unit"]
        got = None
        for p in draft["periods"]:
            if p["period_end"] == truth["period_end"] and m in p["values"] and p["fiscal_period"] in ("FY",):
                got = p["values"][m] * f
        flex = m in ("cost_of_revenue", "income_tax", "capital_expenditure")                              # a cost is printed negative by some companies and positive by others
        a, b = (abs(got), abs(exp)) if got is not None and flex else (got, exp)
        status = "missing" if got is None else "correct" if abs(a - b) <= 1e-9 * max(1, abs(b)) else "wrong"
        out["metrics"][m] = {"expected": exp, "extracted": got, "status": status}
    ends = truth.get("period_ends")
    if ends:
        found = {p["period_end"] for p in draft["periods"]}
        out["spurious_periods"] = sorted(found - ends)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "real_structures_after.json"))
    a = ap.parse_args(argv)
    rows = [run(n) for n in real_like.BUILDERS]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1, default=str))
    return {"rows": rows}


if __name__ == "__main__":
    r = main()["rows"]
    for row in r:
        c = {s: sum(1 for m in row["metrics"].values() if m["status"] == s) for s in ("correct", "wrong", "missing")}
        print(f"{row['issuer']:9s} {c} periods={row['periods_found'][:4]} spurious={row.get('spurious_periods')} err={row['error']}")
        for m, v in row["metrics"].items():
            if v["status"] != "correct":
                print(f"     {m}: expected {v['expected']:,.0f} got {v['extracted']}")
