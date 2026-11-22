"""Screening: rules, honest exclusions, point-in-time history."""

from __future__ import annotations

import pytest

from src.fundamentals import factors, screen
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("screen"))
    svc.load_simulated(n_companies=30, start_year=2010, end_year=2016, seed=4)
    return svc


@pytest.fixture(scope="module")
def ctx(world):
    return ResearchContext(world.store, UniverseSpec())


def test_a_screen_passes_ranks_and_says_why_every_other_company_is_out(ctx):
    rules = [{"factor": "earnings_yield", "op": ">", "value": 0.0}, {"factor": "roe", "op": "top_pct", "value": 50}]
    out = screen.run(ctx, rules, rank_by="earnings_yield", top=5)
    ey = factors.compute(ctx, "earnings_yield").loc[out["as_of"]]
    assert len(out["rows"]) <= 5 and [r["rank"] for r in out["rows"]] == list(range(1, len(out["rows"]) + 1))
    assert all(ey[r["security_id"]] > 0 for r in out["rows"]) and [r["values"]["earnings_yield"] for r in out["rows"]] == sorted((r["values"]["earnings_yield"] for r in out["rows"]), reverse=True)
    c = out["counts"]
    assert sum(c.values()) == out["universe"] and c["pass"] >= len(out["rows"])
    assert {e["status"] for e in out["excluded"]} <= {"failed", "missing", "not applicable", "not investable"} and all(e["why"] for e in out["excluded"])


def test_not_applicable_and_missing_are_not_failures_and_never_pass(ctx, world):
    base = ResearchContext(world.store, UniverseSpec())
    bank = base.ids[0]
    with world.store.db.transaction() as c:
        c.execute("UPDATE securities SET sector = 'Financials', industry = 'Banks' WHERE security_id = ?", (int(bank),))
    ctx2 = ResearchContext(world.store, UniverseSpec())
    out = screen.run(ctx2, [{"factor": "gross_margin", "op": ">", "value": -10.0}])
    st = {e["security_id"]: e["status"] for e in out["excluded"]}
    assert st.get(bank) == "not applicable" and bank not in {r["security_id"] for r in out["rows"]}
    with world.store.db.transaction() as c:
        c.execute("UPDATE securities SET sector = NULL, industry = NULL WHERE security_id = ?", (int(bank),))


def test_a_screen_on_a_past_date_uses_only_what_was_known_then_and_history_reports_turnover(ctx):
    rules = [{"factor": "earnings_yield", "op": "top_pct", "value": 30}]
    early = screen.run(ctx, rules, as_of=str(ctx.dates[30])[:10])
    late = screen.run(ctx, rules)
    assert early["as_of"] < late["as_of"]
    h = screen.history(ctx, rules, start=str(ctx.dates[24])[:10], end=str(ctx.dates[30])[:10])
    assert len(h["dates"]) == 7 and h["dates"][0]["turnover_vs_previous"] is None and all(0 <= d["turnover_vs_previous"] <= 1 for d in h["dates"][1:])


def test_bad_rules_are_refused_with_the_reason(ctx):
    for rule, text in (({"factor": "nope", "op": ">", "value": 1}, "no factor"), ({"factor": "roe", "op": "~", "value": 1}, "op is"), ({"factor": "roe", "op": "between", "value": [3, 1]}, "between"),
                       ({"factor": "roe", "op": "top_pct", "value": 150}, "percentage"), ({"factor": "roe", "op": ">", "value": "x"}, "number")):
        with pytest.raises(FundamentalsError, match=text):
            screen.run(ctx, [rule])
    with pytest.raises(FundamentalsError, match="at least one"):
        screen.run(ctx, [])
