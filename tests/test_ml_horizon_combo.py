"""Release 8, P3-05: short, medium and long signals combined in per-session units, with the dependence between them reported and no exposure counted twice.

The units and the tranche accounting are worked by hand on panels whose answer is known; the horizon weights are checked against the rule that only settled decisions may inform them; the turnover cap is
checked step by step; and a synthetic market shows the combination scored with the single-horizon books through one engine."""

import numpy as np
import pandas as pd
import pytest

from src.ml import horizon_combo as hc
from src.ml.signals import build_weights

CFG = {"method": "top_n_long_short", "n": 2, "gross": 1.0, "net": 0.0, "max_weight": 0.5, "holding_period": 1}
BT = {"rebalance": "daily", "cost_bps": 2.0, "slippage_bps": 0.0, "target_vol": None, "max_leverage": 1.5, "extra_lag": 0}
ASSETS = [f"A{i}" for i in range(8)]


def idx(n=400):
    return pd.bdate_range("2019-01-01", periods=n)


def panel(values, n=400):
    return pd.DataFrame(np.tile(np.asarray(values, dtype=float), (n, 1)), index=idx(n), columns=ASSETS)


# ------------------------------------------------------------------------------------------------------------------ units and dependence
def test_forecasts_of_different_horizons_are_put_on_one_per_session_scale():
    x = panel(np.linspace(-1, 1, 8) * 0.001)
    ps = hc.per_session({1: x, 5: 5 * x, 20: 20 * x})
    for h in (1, 5, 20):
        pd.testing.assert_frame_equal(ps[h], x)
    for bad in ({}, {0: x}):
        with pytest.raises(ValueError, match="at least one signal"):
            hc.per_session(bad)


def test_identical_signals_count_as_one_and_independent_ones_as_many():
    rng = np.random.default_rng(0)
    base = pd.DataFrame(rng.normal(size=(300, 8)), index=idx(300), columns=ASSETS)
    same = hc.dependence({1: base, 5: 5 * base, 20: 20 * base})
    assert same["effective_independent_signals"] == pytest.approx(1.0, abs=1e-6) and np.allclose(same["correlation"].to_numpy(), 1.0)
    indep = hc.dependence({1: base, 5: pd.DataFrame(rng.normal(size=(300, 8)), index=base.index, columns=ASSETS), 20: pd.DataFrame(rng.normal(size=(300, 8)), index=base.index, columns=ASSETS)})
    assert indep["effective_independent_signals"] > 2.6 and abs(indep["correlation"].iloc[0, 1]) < 0.15 and indep["horizons"] == [1, 5, 20]


# ------------------------------------------------------------------------------------------------------------------ horizon weights use settled decisions only
def test_horizon_weights_follow_the_settled_ic_and_are_flat_when_no_horizon_has_an_edge():
    n, lag = 500, 1
    ix = idx(n)
    ic = {1: pd.Series(0.05, index=ix), 5: pd.Series(np.where(np.arange(n) < 300, 0.20, -0.50), index=ix)}
    eq = hc.horizon_weights(ic, lag, "equal")
    assert (eq == 0.5).all().all()
    w = hc.horizon_weights(ic, lag, "ic", window=50, min_dates=20)
    early = w.iloc[100]
    assert early[5] == pytest.approx(0.20 / (0.20 + 0.05)) and early[1] == pytest.approx(0.05 / 0.25) and w.sum(axis=1).iloc[100] == pytest.approx(1.0)
    assert w[5].iloc[300 + lag + 5 - 1] > 0.5, "the day before the first bad decision has settled, horizon 5 still has its old weight"
    assert w[5].iloc[300 + lag + 5 + 60] == 0.0 and w[1].iloc[300 + lag + 5 + 60] == pytest.approx(1.0), "once the bad IC has settled into the window, horizon 5 is dropped and horizon 1 takes the book"
    none = hc.horizon_weights({1: pd.Series(-0.1, index=ix), 5: pd.Series(-0.2, index=ix)}, lag, "ic", window=50, min_dates=20)
    assert (none.iloc[100:].sum(axis=1) == 0.0).all(), "no horizon with a positive settled IC: weight zero, not equal weights by default"
    assert w.iloc[:10].sum(axis=1).eq(0.0).all(), "before there are enough settled decisions nothing is weighted"
    with pytest.raises(ValueError, match="weighting is one of"):
        hc.horizon_weights(ic, lag, "vibes")


# ------------------------------------------------------------------------------------------------------------------ the book cannot count an exposure twice
def test_sleeves_with_capital_shares_never_exceed_the_shares_weighted_gross_and_identical_exposures_are_not_tripled():
    prices = pd.DataFrame(100.0, index=idx(), columns=ASSETS)
    score = panel(np.linspace(-1, 1, 8) * 0.001)
    out = hc.build({1: score, 5: 5 * score, 20: 20 * score}, CFG, prices=prices, book="sleeves", weighting="equal")
    gross = out["final"].abs().sum(axis=1)
    single = build_weights(score, CFG).abs().sum(axis=1)
    assert gross.max() == pytest.approx(single.max()) and gross.max() <= 1.0 + 1e-12, "three identical views held in three sleeves are one book, not three"
    assert out["flat_dates"] == 0 and out["dependence"]["effective_independent_signals"] == pytest.approx(1.0, abs=1e-6)
    w = out["weights"].iloc[0]
    assert w.sum() == pytest.approx(1.0) and set(w.index) == {1, 5, 20}
    blended = hc.build({1: score, 5: 5 * score, 20: 20 * score}, CFG, prices=prices, book="score", weighting="equal")
    assert blended["holding_period"] == 9 and blended["final"].abs().sum(axis=1).max() <= 1.0 + 1e-12 and blended["sleeves"] is None, "the blended book has one holding period, the weighted average horizon"
    with pytest.raises(ValueError, match="book is one of"):
        hc.build({1: score}, CFG, prices=prices, book="vibes")


def test_a_sleeve_is_the_average_of_its_last_h_daily_books():
    n = 60
    flip = np.arange(n) % 2
    score = pd.DataFrame(np.where(flip[:, None] == 0, np.linspace(-1, 1, 8), np.linspace(1, -1, 8)) * 0.001, index=idx(n), columns=ASSETS)
    prices = pd.DataFrame(100.0, index=idx(n), columns=ASSETS)
    out = hc.build({2: 2 * score}, CFG, prices=prices, book="sleeves", weighting="equal")
    daily = build_weights(score, {**CFG, "holding_period": 1})
    expected = daily.rolling(2, min_periods=1).mean()
    pd.testing.assert_frame_equal(out["sleeves"][2], expected, check_exact=False, atol=1e-12)
    assert out["final"].abs().sum(axis=1).iloc[5:].max() < 0.2, "a book that flips every day, held for two sessions, nets to almost nothing: the tranches overlap and cancel"


def test_the_turnover_cap_limits_each_step_and_a_loose_cap_changes_nothing():
    ix = idx(40)
    target = pd.DataFrame(np.where((np.arange(40) % 2 == 0)[:, None], np.array([0.5, -0.5, 0, 0]), np.array([-0.5, 0.5, 0, 0])), index=ix, columns=list("ABCD"))
    capped = hc.cap_turnover(target, 0.2)
    steps = 0.5 * capped.diff().abs().sum(axis=1).iloc[1:]
    assert (steps <= 0.2 + 1e-12).all() and steps.iloc[:5].max() == pytest.approx(0.2) and capped.abs().sum(axis=1).max() <= 1.0 + 1e-12
    pd.testing.assert_frame_equal(hc.cap_turnover(target, 5.0), target)
    assert hc.cap_turnover(target, None) is target
    with pytest.raises(ValueError, match="positive"):
        hc.cap_turnover(target, 0.0)


def test_the_weights_on_a_date_do_not_depend_on_prices_after_it():
    rng = np.random.default_rng(3)
    n = 500
    ix = idx(n)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 8)), axis=0)), index=ix, columns=ASSETS)
    scores = {1: pd.DataFrame(rng.normal(size=(n, 8)), index=ix, columns=ASSETS), 5: pd.DataFrame(rng.normal(size=(n, 8)), index=ix, columns=ASSETS)}
    base = hc.build(scores, CFG, prices=prices, weighting="ic", window=60)
    later = prices.copy()
    later.iloc[350:] = later.iloc[350:].to_numpy()[::-1] * 1.3
    other = hc.build(scores, CFG, prices=later, weighting="ic", window=60)
    np.testing.assert_allclose(base["weights"].iloc[:350].to_numpy(), other["weights"].iloc[:350].to_numpy(), atol=1e-12, err_msg="a weight moved when only later prices changed")
    assert not np.allclose(base["weights"].iloc[400:].to_numpy(), other["weights"].iloc[400:].to_numpy())


# ------------------------------------------------------------------------------------------------------------------ the comparison with single horizons
def test_the_combination_is_scored_with_each_single_horizon_book_through_one_engine():
    rng = np.random.default_rng(4)
    n, m = 900, 8
    ix = idx(n)
    slow = np.zeros((n, m))
    for t in range(1, n):
        slow[t] = 0.97 * slow[t - 1] + 0.0002 * rng.normal(size=m)
    mu = slow + 0.0003 * rng.normal(size=(n, m))                                                      # the true per-session expected return of each asset on each day
    rets = pd.DataFrame(mu + rng.normal(0, 0.008, (n, m)), index=ix, columns=ASSETS)
    prices = (1 + rets).cumprod() * 100.0

    def view(h, noise):                                                                               # a noisy forecast of the h-session return: h times the per-session mean, plus noise
        return pd.DataFrame(h * mu + rng.normal(0, noise, (n, m)), index=ix, columns=ASSETS)

    scores = {1: view(1, 0.0008), 5: view(5, 0.004), 20: view(20, 0.016)}
    out = hc.evaluate(scores, CFG, rets, prices, BT, lag=1, book="sleeves", weighting="equal", n_boot=200, turnover_cap=0.5)
    names = [r["name"] for r in out["rows"]]
    assert names == ["combination", "horizon_1", "horizon_5", "horizon_20"] and set(out["paired_vs_single"]) == {"horizon_1", "horizon_5", "horizon_20"} and out["best_single"] in names[1:]
    sh = {r["name"]: r["sharpe"] for r in out["rows"]}
    assert all(v is not None for v in sh.values()) and sh["combination"] > np.mean([sh["horizon_1"], sh["horizon_5"], sh["horizon_20"]]), "averaging three independent noisy views of one signal beats the average view"
    assert all(d["eligibility"] in ("applied", "insufficient") for d in out["paired_vs_single"].values()) and out["dependence"] > 2.0 and "paired difference whose interval includes zero is not a win" in out["note"]
