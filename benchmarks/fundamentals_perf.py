"""Timing of the research-side hot spots on a simulated world of N companies: ``python benchmarks/fundamentals_perf.py [N]``. Prints one line per step; nothing is asserted here (tests/test_fund_perf.py
asserts generous ceilings)."""

from __future__ import annotations

import sys
import tempfile
import time

sys.path.insert(0, ".")
from src.fundamentals import corporate_actions, datahealth, factors, screen, snapshot, sources  # noqa: E402
from src.fundamentals.research import ResearchContext, UniverseSpec  # noqa: E402
from src.fundamentals.service import FundamentalsService  # noqa: E402


def timed(label, fn, out):
    t = time.perf_counter()
    r = fn()
    out[label] = round(time.perf_counter() - t, 3)
    return r


def main(n: int = 300) -> dict:
    out: dict = {"companies": n}
    svc = FundamentalsService(tempfile.mkdtemp())
    timed("load_simulated", lambda: svc.load_simulated(n_companies=n, start_year=2008, end_year=2018, seed=3), out)
    ctx = timed("research_context", lambda: ResearchContext(svc.store, UniverseSpec()), out)
    timed("market_cap_with_split_detection", ctx.market_cap, out)
    timed("30_factors", lambda: [factors.compute(ctx, f) for f in list(factors.FACTORS)[:30]], out)
    timed("screen_one_date", lambda: screen.run(ctx, [{"factor": "earnings_yield", "op": ">", "value": 0}, {"factor": "roe", "op": "top_pct", "value": 50}]), out)
    timed("screen_history_24_dates", lambda: screen.history(ctx, [{"factor": "earnings_yield", "op": "top_pct", "value": 30}], step=4), out)
    timed("snapshot_fingerprint", lambda: snapshot.fingerprint(svc.store), out)
    timed("data_health_all_companies", lambda: datahealth.overview(svc.store), out)
    sid = int(ctx.ids[0])
    timed("sources_compare_one_company", lambda: sources.compare(svc.store, sid), out)
    timed("custom_panel_empty", lambda: __import__("src.fundamentals.sectors", fromlist=["x"]).custom_panel(ctx, "custom:cet1_ratio"), out)
    timed("corporate_events_mask", lambda: corporate_actions.unreliable_mask(ctx), out)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main(int(sys.argv[1]) if len(sys.argv) > 1 else 300), indent=1))
