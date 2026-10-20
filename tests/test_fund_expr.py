"""Expression trees: parsing, the canonical form and its hash, the type rules, evaluation against hand calculations, and a generator that only builds formulas that mean something."""

from __future__ import annotations

import numpy as np
import pytest

from src.fundamentals import expr as X
from src.fundamentals import factors as F
from src.fundamentals import transforms as T
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("expr"))
    svc.load_simulated(n_companies=48, start_year=2009, end_year=2016, seed=17)
    return ResearchContext(svc.store, UniverseSpec())


# ------------------------------------------------------------------------------------------------------------------------ text, JSON and validation
def test_a_formula_is_read_from_text_printed_back_and_survives_json():
    text = "sub(rank(div(operating_income,total_assets)),rank(delta(roic,12)))"
    e = X.parse(text)
    assert X.to_string(e) == text and e["op"] == "sub" and e["args"][1]["args"][0]["args"][0] == {"f": "roic"} and e["args"][1]["args"][0]["k"] == 12
    assert X.loads(X.dumps(e)) == X.canonical(e) and X.parse("  div( revenue , 2.5 )  ") == X.op("div", X.field("revenue"), X.const(2.5))
    assert X.parse("neg(log(market_cap))")["op"] == "neg" and X.parse("rolling_std(roic,24)")["k"] == 24 and X.to_string(X.const(0.5)) == "0.5" and X.to_string(X.const(3)) == "3"


@pytest.mark.parametrize("bad, why", [("explode(roic)", "unknown operator"), ("div(roic)", "takes 2 arguments"), ("delta(roic)", "needs a lag"), ("delta(roic,0)", "needs a lag"),
                                        ("rank(roic,3)", "takes 1 argument"), ("div(roic,revenue", "ends too soon"), ("div(roic revenue)", "expected"), ("roic revenue", "after the formula"),
                                        ("div(1+2,3)", "cannot read"), ("__import__('os')", "cannot read")])
def test_a_malformed_formula_is_refused_with_a_reason(bad, why):
    with pytest.raises(X.ExprError, match=why):
        X.parse(bad)


def test_a_formula_that_is_too_large_or_carries_code_is_refused():
    big = X.field("roic")
    for _ in range(40):
        big = X.op("neg", big)
    with pytest.raises(X.ExprError, match="too large"):
        X.validate(big)
    with pytest.raises(X.ExprError, match="bad field name"):
        X.validate({"f": "roic; import os"})
    with pytest.raises(X.ExprError, match="finite"):
        X.validate({"c": float("inf")})
    with pytest.raises(X.ExprError, match="takes no parameter"):
        X.validate({"op": "rank", "args": [X.field("roic")], "k": 3})
    with pytest.raises(X.ExprError):
        X.loads("{not json")


# ------------------------------------------------------------------------------------------------------------------------ identity
def test_the_same_idea_has_one_canonical_form_and_one_hash():
    a, b = X.parse("add(roic,gross_margin)"), X.parse("add(gross_margin,roic)")
    assert X.canonical_string(a) == X.canonical_string(b) and X.expression_hash(a) == X.expression_hash(b)
    assert X.canonical_string(X.parse("neg(neg(roic))")) == "roic" and X.canonical_string(X.parse("rank(rank(roic))")) == "rank(roic)"
    assert X.canonical_string(X.parse("add(roic,neg(gross_margin))")) == "sub(roic,gross_margin)"
    assert X.expression_hash(X.parse("sub(roic,gross_margin)")) != X.expression_hash(X.parse("sub(gross_margin,roic)"))                 # order matters where it should
    assert X.expression_hash(X.parse("delta(roic,12)")) != X.expression_hash(X.parse("delta(roic,6)"))
    with pytest.raises(X.ExprError, match="degenerate"):
        X.canonical(X.parse("sub(roic,roic)"))
    with pytest.raises(X.ExprError, match="degenerate"):
        X.canonical(X.parse("div(roic,roic)"))
    with pytest.raises(X.ExprError, match="constants only"):
        X.canonical(X.parse("add(1,2)"))


def test_a_rank_a_zscore_or_a_sign_on_the_outside_is_the_same_signal_but_inside_a_formula_it_is_not():
    base = X.parse("div(operating_income,total_assets)")
    keys = {X.signal_key(X.op(w, base)) for w in ("rank", "zscore", "neg")} | {X.signal_key(base), X.signal_key(X.op("neg", X.op("rank", base)))}
    assert len(keys) == 1
    assert X.signal_key(X.parse("add(rank(roic),rank(gross_margin))")) != X.signal_key(X.parse("add(roic,gross_margin)"))
    assert X.size(X.parse("sub(rank(roic),delta(roic,12))")) == 5 and X.depth(X.parse("rank(delta(roic,12))")) == 3 and X.fields_used(X.parse("div(revenue,total_assets)")) == ["revenue", "total_assets"]


# ------------------------------------------------------------------------------------------------------------------------ the type rules
def test_only_formulas_that_mean_something_are_well_typed(ctx):
    kinds = {k: v.kind for k, v in X.catalogue(ctx).items()}
    assert kinds["operating_income"] == "amount_rep" and kinds["market_cap"] == "amount_trade" and kinds["roic"] == "ratio" and kinds["shares_outstanding"] == "count" and kinds["mom_12_1"] == "ratio"
    good = ["div(operating_income,total_assets)", "div(net_income,market_cap)", "delta(roic,12)", "sub(rank(roic),rank(gross_margin))", "add(operating_income,market_cap)", "div(shares_outstanding,shares_outstanding)"]
    ok = [X.well_typed(X.parse(g), kinds)[0] for g in good[:5]]
    assert ok == [True] * 5
    for bad, why in (("add(revenue,roic)", "cannot add"), ("div(roic,revenue)", "no clear meaning"), ("mul(revenue,total_assets)", "no clear meaning"), ("add(roic,2)", "bare constant"),
                     ("div(nonsense,revenue)", "unknown field")):
        good_or_not, reason = X.well_typed(X.parse(bad), kinds)
        assert not good_or_not and why in reason, bad
    for bad in ("add(roic,gross_margin)", "sub(roic,rank(gross_margin))", "sub(growth(revenue,12),roic)"):
        ok_, reason = X.well_typed(X.parse(bad), kinds)
        assert not ok_ and ("different scales" in reason or "bare constant" in reason), bad
    for fine in ("sub(roic,rolling_mean(roic,12))", "add(delta(roic,12),roic)", "sub(rank(roic),rank(gross_margin))", "add(rank(delta(roic,12)),rank(growth(revenue,12)))", "sub(delta(roic,12),delta(roic,6))"):
        assert X.well_typed(X.parse(fine), kinds)[0], fine
    assert X.scale_tag(X.parse("delta(roic,12)")) == "roic" and X.scale_tag(X.parse("rank(roic)")) == "norm" and X.scale_tag(X.parse("div(revenue,total_assets)")) == "(revenue/total_assets)"
    assert X.infer_kind(X.parse("div(net_income,market_cap)"), kinds) == "ratio" and X.infer_kind(X.parse("div(operating_income,roic)"), kinds) == "amount_rep"
    assert X.infer_kind(X.parse("growth(revenue,12)"), kinds) == "ratio" and X.infer_kind(X.parse("delta(revenue,12)"), kinds) == "amount_rep"
    info = X.describe(X.parse("div(net_income,market_cap)"), ctx)
    assert info["well_typed"] and info["kind"] == "ratio" and info["fields"] == ["market_cap", "net_income"] and len(info["hash"]) == 16
    assert not X.describe(X.parse("add(revenue,roic)"), ctx)["well_typed"]


# ------------------------------------------------------------------------------------------------------------------------ evaluation
def test_an_amount_against_a_price_is_converted_to_the_trading_currency_first(ctx):
    panel, kind = X.evaluate(X.parse("div(net_income,market_cap)"), ctx)
    library = F.compute(ctx, "earnings_yield")
    both = panel.notna() & library.notna()
    assert kind == "ratio" and both.to_numpy().sum() > 1000 and np.allclose(panel.to_numpy()[both.to_numpy()], library.to_numpy()[both.to_numpy()])
    own, _ = X.evaluate(X.parse("div(net_income,total_assets)"), ctx)
    roa = ctx.panel("roa")
    same = own.notna() & roa.notna()
    assert np.allclose(own.to_numpy()[same.to_numpy()], roa.to_numpy()[same.to_numpy()])             # a ratio of the company's own figures is never converted


def test_the_operators_do_what_they_say(ctx):
    x = ctx.panel("roic")
    ev = lambda text: X.evaluate_panel(X.parse(text), ctx)                                           # noqa: E731
    assert np.allclose(ev("delta(roic,12)").to_numpy(), (x - x.shift(12)).to_numpy(), equal_nan=True)
    assert np.allclose(ev("rolling_mean(roic,12)").to_numpy(), x.rolling(12, min_periods=6).mean().to_numpy(), equal_nan=True)
    assert np.allclose(ev("rolling_std(roic,12)").to_numpy(), x.rolling(12, min_periods=6).std().to_numpy(), equal_nan=True)
    g = T.pct_change(ctx.panel("revenue"), 12)
    assert np.allclose(ev("growth(revenue,12)").to_numpy(), g.to_numpy(), equal_nan=True)
    assert np.allclose(ev("accel(revenue,12)").to_numpy(), (g - g.shift(12)).to_numpy(), equal_nan=True)
    assert np.allclose(ev("neg(roic)").to_numpy(), -x.to_numpy(), equal_nan=True) and np.allclose(ev("abs(roic)").to_numpy(), x.abs().to_numpy(), equal_nan=True)
    r = ev("rank(roic)")
    inv = ctx.investable().reindex_like(r).fillna(False)
    assert r.where(~inv).isna().all().all() and (r.max(axis=1).dropna() == 1.0).all()                # ranked among the investable only
    sector = ctx.group("sector")
    sr = ev("sector_rank(roic)")
    biggest = sector.value_counts().index[0]
    members = sector.index[sector == biggest]
    row = sr.iloc[60][members].dropna()
    assert len(row) >= 5 and row.max() == pytest.approx(1.0) and row.min() >= 1 / len(members) - 1e-9
    z = ev("zscore(roic)")
    assert np.allclose(z.mean(axis=1).dropna(), 0, atol=1e-9)


def test_a_denominator_that_is_not_positive_and_a_log_of_a_non_positive_are_missing(ctx):
    ni = ctx.panel("net_income")
    d = X.evaluate_panel(X.parse("div(revenue,net_income)"), ctx)
    assert d[ni <= 0].isna().all().all() and d[ni > 0].notna().to_numpy().sum() > 500
    lg = X.evaluate_panel(X.parse("log(net_income)"), ctx)
    assert lg[ni <= 0].isna().all().all() and np.allclose(lg[ni > 0].to_numpy()[ni.to_numpy() > 0], np.log(ni[ni > 0]).to_numpy()[ni.to_numpy() > 0])
    c = X.evaluate_panel(X.parse("div(revenue,2)"), ctx)
    assert np.allclose(c.to_numpy(), (ctx.panel("revenue") / 2).to_numpy(), equal_nan=True)
    with pytest.raises(X.ExprError, match="unknown field"):
        X.evaluate_panel(X.parse("rank(vibes)"), ctx)


def test_a_sub_formula_is_computed_once(ctx):
    cache: dict = {}
    X.evaluate(X.parse("add(rank(delta(roic,12)),rank(delta(roic,6)))"), ctx, cache)
    assert "delta(roic,12)" in cache and "roic" in cache and len(cache) == 6                          # roic once, two deltas, two ranks and the sum
    before = dict(cache)
    X.evaluate(X.parse("rank(delta(roic,12))"), ctx, cache)
    assert all(cache[k][0] is before[k][0] for k in before)                                           # reused, not rebuilt


# ------------------------------------------------------------------------------------------------------------------------ the generator
def test_the_generator_builds_only_valid_well_typed_formulas_and_repeats_itself_for_a_seed(ctx):
    kinds = {k: v.kind for k, v in X.catalogue(ctx).items()}
    a = [X.to_string(X.Generator(ctx, np.random.default_rng(5)).random()) for _ in range(1)]
    g1, g2 = X.Generator(ctx, np.random.default_rng(5)), X.Generator(ctx, np.random.default_rng(5))
    s1, s2 = [X.to_string(g1.random()) for _ in range(150)], [X.to_string(g2.random()) for _ in range(150)]
    assert s1 == s2 and a[0] == s1[0]
    assert len(set(s1)) > 100
    for text in s1:
        e = X.parse(text)
        assert X.well_typed(e, kinds)[0] and X.size(e) <= 15 and X.canonical_string(e) == text
    families = {"delta", "growth", "accel", "rolling_mean", "rolling_std", "sector_rank", "rank", "div"}
    assert families <= {w for t in s1 for w in families if w + "(" in t}


def test_mutation_and_crossover_stay_valid_and_change_the_formula(ctx):
    kinds = {k: v.kind for k, v in X.catalogue(ctx).items()}
    g = X.Generator(ctx, np.random.default_rng(9))
    parents = [g.random() for _ in range(40)]
    for p in parents:
        child = g.mutate(p)
        assert X.well_typed(child, kinds)[0] and X.size(child) <= 15
    changed = sum(X.expression_hash(g.mutate(p)) != X.expression_hash(p) for p in parents)
    assert changed >= 35
    for a, b in zip(parents[:20], parents[20:]):
        child = g.cross(a, b)
        assert X.well_typed(child, kinds)[0] and X.size(child) <= 15
    restricted = X.Generator(ctx, np.random.default_rng(1), only_fields=["operating_income", "total_assets", "revenue", "roic"])
    assert {f for _ in range(50) for f in X.fields_used(restricted.random())} <= {"operating_income", "total_assets", "revenue", "roic"}
    with pytest.raises(X.ExprError, match="too few fields"):
        X.Generator(ctx, np.random.default_rng(1), only_fields=["revenue"])
