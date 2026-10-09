"""Contextual alpha (different weights in different buckets) and nonlinear terms (squares, products, conditional effects), each checked against a world with the effect planted."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity import alpha_model as am
from src.equity import contextual as cx
from tests.test_equity_fundamentals import IDX, T, build_table, firm
from src.equity.fundamentals import FactorInputs, Fundamentals


def panel(rng, n_assets, n_periods, k=2):
    idx = pd.date_range("2005-01-01", periods=n_periods, freq="MS")
    cols = [f"S{i:03d}" for i in range(n_assets)]
    return idx, cols, [pd.DataFrame(rng.normal(size=(n_periods, n_assets)), index=idx, columns=cols) for _ in range(k)]


# ------------------------------------------------------------------------------------------------------------------ buckets
def test_buckets_split_every_date_into_even_ordered_groups_and_leave_missing_missing():
    rng = np.random.default_rng(0)
    v = pd.DataFrame(rng.normal(size=(5, 30)))
    b = cx.buckets(v, 3)
    assert (b.apply(lambda r: r.value_counts().sort_index().tolist(), axis=1).tolist() == [[10, 10, 10]] * 5)
    for t in range(5):
        assert v.iloc[t][b.iloc[t] == 0].max() < v.iloc[t][b.iloc[t] == 1].min() and v.iloc[t][b.iloc[t] == 1].max() < v.iloc[t][b.iloc[t] == 2].min()
    holes = v.copy()
    holes.iloc[:, :6] = np.nan
    assert cx.buckets(holes, 3).iloc[:, :6].isna().all().all() and cx.buckets(holes, 3).iloc[0].dropna().value_counts().tolist() == [8, 8, 8]
    inv = pd.DataFrame(True, index=v.index, columns=v.columns)
    inv.iloc[:, 6:12] = False
    assert cx.buckets(v, 2, inv).iloc[:, 6:12].isna().all().all()
    ties = pd.DataFrame([[1.0] * 9])
    assert sorted(cx.buckets(ties, 3).iloc[0].value_counts().tolist()) == [3, 3, 3]                  # all equal: split by order, not dumped in one bucket
    with pytest.raises(ValueError, match="at least 2"):
        cx.buckets(v, 1)


def test_the_conditioning_variables_are_what_they_say():
    prices = pd.DataFrame({t: np.full(len(IDX), p) for t, p in {"A": 10.0, "B": 15.0, "C": 40.0}.items()}, index=IDX)
    x = FactorInputs(prices, Fundamentals.from_table(build_table(), lag_days=60))
    assert cx.context_variable("growth", x).loc[T].tolist() == pytest.approx([0.1, 0.1, 0.1])
    roa = [0.15 * 2000 * 1.1 ** y / (1500 * 1.05 ** y) for y in (5, 4, 3, 2)]                       # net income over assets, one observation a year
    assert cx.context_variable("earnings_variability", x).loc[T, "A"] == pytest.approx(np.std(roa, ddof=1))
    s = firm("A")[5]
    assert cx.context_variable("value", x).loc[T, "A"] == pytest.approx(s["book_equity"] / (10.0 * s["shares_outstanding"]))
    assert cx.context_variable("size", x).loc[T, "A"] == pytest.approx(10.0 * s["shares_outstanding"])
    with pytest.raises(KeyError, match="unknown context"):
        cx.context_variable("beauty", x)


# ------------------------------------------------------------------------------------------------------------------ contextual alpha
def split_world(seed, n_assets=240, n_periods=150, premium=0.006, idio=0.06):
    """Factor 1 pays in bucket 0 only and factor 2 in bucket 1 only; the buckets are persistent and even."""
    rng = np.random.default_rng(seed)
    idx, cols, (f1, f2) = panel(rng, n_assets, n_periods)
    label = np.tile([0, 1], n_assets // 2)
    context = pd.DataFrame(np.tile(label, (n_periods, 1)).astype(float), index=idx, columns=cols)
    forward = premium * (f1 * (context == 0) + f2 * (context == 1)) + idio * pd.DataFrame(rng.normal(size=f1.shape), index=idx, columns=cols)
    return {"f1": f1, "f2": f2}, forward, context


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_each_bucket_learns_the_factor_that_works_there_and_the_whole_beats_one_set_of_weights(seed):
    factors, forward, context = split_world(seed)
    result = cx.contextual_alpha(factors, forward, context, window=36, min_obs=12, shrink=0.3, min_assets=20)
    w0, w1 = result.weights[0].iloc[60:].mean(), result.weights[1].iloc[60:].mean()
    assert w0["f1"] > 0.7 and abs(w0["f2"]) < 0.3 and w1["f2"] > 0.7 and abs(w1["f1"]) < 0.3
    single = am.optimal_alpha(factors, forward, window=36, min_obs=12, shrink=0.3).alpha
    ic = lambda a: am.information_coefficients({"a": a}, forward, "pearson", 20)["a"].iloc[48:]
    assert ic(result.alpha).mean() > ic(single).mean() + 0.02                                    # the same factors, weighed where they work, find more
    assert ic(result.alpha).mean() / ic(result.alpha).std() > ic(single).mean() / ic(single).std()


def test_the_composite_is_standardised_inside_each_bucket_or_across_all_stocks():
    factors, forward, context = split_world(3, n_periods=60)
    by_bucket = cx.contextual_alpha(factors, forward, context, min_obs=12, min_assets=20, scale="context").alpha
    row = by_bucket.iloc[40]
    assert row[context.iloc[40] == 0].std() == pytest.approx(1.0) and row[context.iloc[40] == 1].std() == pytest.approx(1.0)
    overall = cx.contextual_alpha(factors, forward, context, min_obs=12, min_assets=20, scale="global").alpha
    assert overall.iloc[40].std() == pytest.approx(1.0) and abs(overall.iloc[40].mean()) < 1e-12
    with pytest.raises(ValueError, match="scale"):
        cx.contextual_alpha(factors, forward, context, scale="nope")
    with pytest.raises(ValueError, match="no buckets"):
        cx.contextual_alpha(factors, forward, context * np.nan)


def test_until_enough_ics_have_matured_a_bucket_uses_equal_weights_and_nothing_sees_the_future():
    factors, forward, context = split_world(4, n_periods=80)
    base = cx.contextual_alpha(factors, forward, context, min_obs=12, min_assets=20)
    assert (base.weights[0].iloc[:12] == 0.5).all().all()
    j = 50
    changed = forward.copy()
    changed.iloc[j:] = -changed.iloc[j:]
    other = cx.contextual_alpha(factors, changed, context, min_obs=12, min_assets=20)
    assert base.alpha.iloc[: j + 1].equals(other.alpha.iloc[: j + 1])                            # the alpha on date j uses the ICs of dates before j, whose windows had ended
    assert np.abs(base.alpha.iloc[j + 5:] - other.alpha.iloc[j + 5:]).max().max() > 0.1


# ------------------------------------------------------------------------------------------------------------------ nonlinear terms
def test_the_square_the_product_and_the_conditional_term_are_what_they_say():
    rng = np.random.default_rng(0)
    idx, cols, (a, b) = panel(rng, 300, 12)
    za, zb = am.standardize(a), am.standardize(b)
    q = cx.quadratic(a)
    assert np.abs(q.mean(axis=1)).max() < 1e-12 and np.abs(q.std(axis=1) - 1).max() < 1e-12
    assert np.mean([np.corrcoef(q.iloc[t], za.iloc[t] ** 2)[0, 1] for t in range(12)]) > 0.999
    assert abs(np.mean([np.corrcoef(q.iloc[t], za.iloc[t])[0, 1] for t in range(12)])) < 0.15       # a square carries what a straight line cannot
    ia = cx.interaction(a, b)
    assert np.mean([np.corrcoef(ia.iloc[t], (za * zb).iloc[t])[0, 1] for t in range(12)]) > 0.999
    top = cx.conditional(a, b, "high", 3)
    bucket = cx.buckets(b, 3)
    assert (top.where(bucket != 2).dropna(how="all").abs().sum().sum() == 0) and np.abs(top.where(bucket == 2) - za.where(bucket == 2)).max().max() < 1e-12
    low = cx.conditional(a, b, "low", 3)
    assert (low.where(bucket != 0).fillna(0.0) == 0).all().all() and (low.where(bucket == 0).dropna(how="all").abs().sum().sum() > 0)
    with pytest.raises(ValueError, match="high"):
        cx.conditional(a, b, "middle")


def test_named_terms_are_added_to_the_linear_factors_and_unknown_terms_are_refused():
    rng = np.random.default_rng(1)
    _, _, (a, b) = panel(rng, 80, 10)
    out = cx.nonlinear_features({"a": a, "b": b}, [("quadratic", "a"), ("interaction", "a", "b"), ("conditional", "a", "b", "low"), ("conditional", "b", "a")])
    assert list(out) == ["a", "b", "a^2", "a*b", "a|b=low", "b|a=high"]
    with pytest.raises(ValueError, match="unknown term"):
        cx.nonlinear_features({"a": a}, [("cubic", "a")])


def planted(kind, seed, n_assets=300, n_periods=200, size=0.004, idio=0.06):
    rng = np.random.default_rng(seed)
    idx, cols, (a, b) = panel(rng, n_assets, n_periods)
    za, zb = am.standardize(a), am.standardize(b)
    noise = idio * pd.DataFrame(rng.normal(size=a.shape), index=idx, columns=cols)
    if kind == "square":
        forward = -size * (za ** 2 - 1) + noise                                                      # starved and gorged both do worse
    elif kind == "product":
        forward = size * za * zb + noise
    else:
        forward = size * 1.5 * za.where(cx.buckets(zb, 3) == 2, 0.0) + noise                        # a effect that exists only where b is high
    return {"a": a, "b": b}, forward


def test_a_u_shaped_effect_has_no_linear_ic_but_a_strong_quadratic_one():
    factors, forward = planted("square", 0)
    ic = lambda f: am.information_coefficients({"x": f}, forward, "pearson", 20)["x"]
    assert abs(ic(am.standardize(factors["a"])).mean()) < 0.015                                       # a straight line sees nothing
    quad = ic(cx.quadratic(factors["a"]))
    assert quad.mean() < -0.04 and quad.mean() / quad.std() * np.sqrt(len(quad)) < -6
    lin = cx.fama_macbeth_forecast(cx.nonlinear_features({"a": factors["a"]}, []), forward, window=60, min_obs=24)
    non = cx.fama_macbeth_forecast(cx.nonlinear_features({"a": factors["a"]}, [("quadratic", "a")]), forward, window=60, min_obs=24)
    assert abs(non.average.loc["a", "t"]) < 3 and non.average.loc["a^2", "t"] < -6 and non.average.loc["a^2", "slope"] == pytest.approx(-0.004, abs=0.0015)
    skill = lambda f: am.information_coefficients({"x": f.forecast}, forward, "pearson", 20)["x"].iloc[60:]
    assert skill(non).mean() > skill(lin).mean() + 0.04 and skill(non).mean() > 0.04


@pytest.mark.parametrize("kind,term,name", [("product", ("interaction", "a", "b"), "a*b"), ("conditional", ("conditional", "a", "b", "high"), "a|b=high")])
def test_products_and_conditional_effects_are_found_by_their_own_terms(kind, term, name):
    factors, forward = planted(kind, 1)
    result = cx.fama_macbeth_forecast(cx.nonlinear_features(factors, [term]), forward, window=60, min_obs=24)
    assert result.average.loc[name, "t"] > 6                                                       # the term earns
    other = [k for k in result.average.index if k not in (name,)]
    linear_only = cx.fama_macbeth_forecast(cx.nonlinear_features(factors, []), forward, window=60, min_obs=24)
    ic = lambda f: am.information_coefficients({"x": f.forecast}, forward, "pearson", 20)["x"].iloc[60:].mean()
    assert ic(result) > ic(linear_only) + 0.03 and other


def test_fama_macbeth_slopes_recover_an_exact_linear_world_and_the_forecast_waits_for_matured_slopes():
    rng = np.random.default_rng(5)
    idx, cols, (x1, x2) = panel(rng, 60, 40)
    forward = 0.01 + 0.02 * x1 - 0.03 * x2
    r = cx.fama_macbeth_forecast({"x1": x1, "x2": x2}, forward, horizon=1, window=20, min_obs=10)
    assert np.abs(r.slopes["x1"] - 0.02).max() < 1e-12 and np.abs(r.slopes["x2"] + 0.03).max() < 1e-12
    assert r.forecast.iloc[:10].isna().all().all() and r.forecast.iloc[10:].notna().all().all()     # ten matured slopes are needed
    exact = 0.02 * x1 - 0.03 * x2
    assert np.abs(r.forecast.iloc[10:] - exact.iloc[10:]).max().max() < 1e-12
    assert r.average.loc["x1", "slope"] == pytest.approx(0.02) and set(r.average.columns) == {"slope", "t", "ir"}
    # nothing after a date can change the forecast for it
    noisy = forward + 0.05 * pd.DataFrame(rng.normal(size=forward.shape), index=idx, columns=cols)
    base = cx.fama_macbeth_forecast({"x1": x1, "x2": x2}, noisy, horizon=1, window=20, min_obs=10)
    cut = noisy.copy()
    cut.iloc[25:] = 0.0
    other = cx.fama_macbeth_forecast({"x1": x1, "x2": x2}, cut, horizon=1, window=20, min_obs=10)
    assert base.forecast.iloc[:26].equals(other.forecast.iloc[:26]) and not base.forecast.iloc[30:].equals(other.forecast.iloc[30:])
    longer = cx.fama_macbeth_forecast({"x1": x1, "x2": x2}, noisy, horizon=3, window=20, min_obs=10)
    assert longer.forecast.iloc[:12].isna().all().all()                                            # a three-period horizon matures slopes three periods later
