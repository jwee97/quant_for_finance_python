"""The headline statistics against definitions written out here, and against each other: a return stream must get the same Sortino ratio and the same drawdown whichever part of the platform reports it.

Two defects were found by writing these and are pinned here:

* the Sortino ratio of ``backtest.metrics`` averaged the squared shortfalls over the losing days only, where the definition (Sortino and van der Meer 1991; Sortino and Price 1994) averages over every day.
  Two other modules of the platform used the definition, so one return stream had two Sortino ratios;
* a drawdown was measured on a growth curve whose first point is already after the first return, so the starting capital was never a peak and an account that lost from its first day reported too small
  a drawdown. The out-of-sample segment of a walk-forward is exactly such a series.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtest import metrics as bm
from src.features import returns as fr
from src.ml import comparison as ml_comparison, model_selection as ms
from src.probability import ruin
from src.risk import performance as rp
from src.scenarios import backtest as scenario_backtest
from src.stats import inference as si

warnings.filterwarnings("ignore")
ANN = 252


def series(values, start="2022-01-03"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(start, periods=len(values)))


@pytest.fixture(scope="module")
def r():
    rng = np.random.default_rng(11)
    return series(rng.standard_t(5, 750) * 0.009 + 0.0004)


# ---------------------------------------------------------------------------------------------------------------------- definitions
def test_sharpe_volatility_and_growth_are_the_textbook_formulas(r):
    x = r.to_numpy()
    assert bm.annualised_volatility(r) == pytest.approx(x.std(ddof=1) * math.sqrt(ANN), rel=1e-12)
    assert bm.sharpe_ratio(r) == pytest.approx(x.mean() / x.std(ddof=1) * math.sqrt(ANN), rel=1e-12)
    rf = 0.03
    assert bm.sharpe_ratio(r, rf) == pytest.approx((x - rf / ANN).mean() / x.std(ddof=1) * math.sqrt(ANN), rel=1e-12)       # an annual rate is spread over the periods
    years = (r.index[-1] - r.index[0]).days / 365.25
    assert bm.cagr(r) == pytest.approx(np.prod(1 + x) ** (1 / years) - 1, rel=1e-3)
    assert bm.cagr(series([-1.0, 0.1, 0.1])) == -1.0 and math.isnan(bm.sharpe_ratio(series([0.001] * 50)))


def test_the_sortino_ratio_averages_the_squared_shortfalls_over_every_day(r):
    x = r.to_numpy()
    downside = math.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    assert bm.downside_deviation(r) == pytest.approx(downside, rel=1e-12)
    assert bm.sortino_ratio(r) == pytest.approx(x.mean() / downside * math.sqrt(ANN), rel=1e-12)
    losing_days_only = math.sqrt(np.mean(x[x < 0] ** 2))                                   # what the function divided by before
    assert losing_days_only / downside == pytest.approx(math.sqrt(len(x) / (x < 0).sum()), rel=1e-12) and losing_days_only > 1.3 * downside
    by_hand = series([0.02, -0.01, 0.03, -0.02, 0.01, 0.0])
    assert bm.sortino_ratio(by_hand) == pytest.approx((0.03 / 6) / math.sqrt((0.01 ** 2 + 0.02 ** 2) / 6) * math.sqrt(ANN), rel=1e-12)
    target = 0.001
    assert bm.sortino_ratio(r, target) == pytest.approx((x.mean() - target) / math.sqrt(np.mean(np.minimum(x - target, 0.0) ** 2)) * math.sqrt(ANN), rel=1e-12)
    assert math.isnan(bm.sortino_ratio(series([0.001] * 40))) and bm.sortino_ratio(series([0.01] * 39 + [-0.01])) > 0       # no downside: not a number; one losing day is enough


def test_one_return_stream_has_one_sortino_ratio_whoever_reports_it(r):
    ours = bm.sortino_ratio(r)
    assert rp.risk_metrics(r)["sortino"] == pytest.approx(ours, rel=1e-9)
    assert ms.metrics_of(r)["net_sortino"] == pytest.approx(ours, rel=1e-9)
    assert bm.performance_summary(r)["sortino"] == pytest.approx(ours, rel=1e-12)
    described = fr.describe_returns(r.to_frame("A")).loc["A"]
    assert described["downside_vol_ann"] == pytest.approx(bm.downside_deviation(r) * math.sqrt(ANN), rel=1e-12)


def test_tracking_error_and_information_ratio(r):
    rng = np.random.default_rng(2)
    bench = series(r.to_numpy() * 0.8 + rng.normal(0, 0.004, len(r)))
    out = bm.performance_summary(r, benchmark=bench)
    active = (r - bench).to_numpy()
    assert out["tracking_error"] == pytest.approx(active.std(ddof=1) * math.sqrt(ANN), rel=1e-12)
    assert out["information_ratio"] == pytest.approx(active.mean() * ANN / (active.std(ddof=1) * math.sqrt(ANN)), rel=1e-12)


# ---------------------------------------------------------------------------------------------------------------------- drawdown
def reference_drawdown(x: np.ndarray) -> np.ndarray:
    """The account's value over its high-water mark, minus one, by a loop: the mark starts at the starting capital."""
    value, peak, out = 1.0, 1.0, []
    for step in x:
        value *= 1.0 + step
        peak = max(peak, value)
        out.append(value / peak - 1.0)
    return np.array(out)


LOSES_FIRST = [-0.05, -0.05, 0.10, 0.02, 0.02, 0.01]


def test_an_account_that_loses_from_its_first_day_is_under_water_from_its_starting_capital():
    x = series(LOSES_FIRST)
    want = reference_drawdown(x.to_numpy())
    assert want.min() == pytest.approx(-0.0975)                                             # 0.95 * 0.95 - 1: before the fix this was reported as -0.05
    assert np.allclose(fr.returns_drawdown(x).to_numpy(), want, atol=1e-15)
    assert fr.max_drawdown(fr.cumulative_returns(x), 1.0) == pytest.approx(-0.0975)
    assert fr.max_drawdown(fr.cumulative_returns(x)) == pytest.approx(-0.05)                 # a bare curve (a price) has no capital before its first point: unchanged
    table = fr.drawdown_table(fr.cumulative_returns(x), initial=1.0)
    assert table.loc[0, "depth"] == pytest.approx(-0.0975) and table.loc[0, "trough"] == x.index[1]


def test_every_report_of_a_strategys_drawdown_agrees_with_the_loop(r):
    for x in (r, series(LOSES_FIRST * 8), series([-0.01] * 40)):
        want = float(reference_drawdown(x.to_numpy()).min())
        assert bm.performance_summary(x)["max_drawdown"] == pytest.approx(want, abs=1e-12)
        assert ms.max_drawdown(x) == pytest.approx(-want, abs=1e-12)
        assert ml_comparison._max_drawdown(x) == pytest.approx(want, abs=1e-12)
        assert float(rp.drawdown_series(x).min()) == pytest.approx(want, abs=1e-12)
        assert float(ruin.drawdown_series(x).min()) == pytest.approx(want, abs=1e-12)
        assert float(ruin.max_drawdown_of_paths(x.to_numpy()[None, :])[0]) == pytest.approx(want, abs=1e-12)
        assert scenario_backtest.summarise(x.to_numpy())["max_drawdown"] == pytest.approx(want, abs=1e-12)
        assert rp.risk_metrics(x)["recovery_factor"] == pytest.approx((np.prod(1 + x.to_numpy()) - 1.0) / -want, rel=1e-9)        # total return over the deepest drawdown
        if abs(want) > 1e-12 and np.prod(1 + x.to_numpy()) > 0:
            assert bm.calmar_ratio(x) == pytest.approx(bm.cagr(x) / abs(want), rel=1e-9)


def test_a_drawdown_is_never_positive_is_zero_at_a_new_high_and_does_not_depend_on_the_starting_capital(r):
    dd = fr.returns_drawdown(r)
    assert dd.max() <= 0.0 and dd.min() >= -1.0
    highs = fr.cumulative_returns(r) >= np.maximum(fr.cumulative_returns(r).cummax(), 1.0) - 1e-15
    assert (dd[highs].abs() < 1e-15).all()
    scaled = fr.drawdown(fr.cumulative_returns(r, initial=250_000.0), 250_000.0)
    assert np.allclose(scaled.to_numpy(), dd.to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------------------------------------------------- Sharpe ratio inference
def test_the_probabilistic_sharpe_ratio_is_bailey_and_lopez_de_prados_formula(r):
    x = r.to_numpy()
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    for benchmark in (0.0, 0.5):
        want = stats.norm.cdf((sr - benchmark / math.sqrt(ANN)) * math.sqrt(len(x) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
        assert bm.probabilistic_sharpe_ratio(sr * math.sqrt(ANN), benchmark, len(x), g3, g4) == pytest.approx(want, rel=1e-12)
        assert si.probabilistic_sharpe_ratio(x, benchmark, ANN) == pytest.approx(want, rel=1e-6)               # the two implementations of the platform agree
    # the worked example of Bailey and Lopez de Prado (2012): a monthly Sharpe ratio of 0.458 over 24 months, skewness -2.448 and kurtosis 10.164, against a benchmark of zero
    paper = stats.norm.cdf(0.458 * math.sqrt(23) / math.sqrt(1 + 2.448 * 0.458 + (10.164 - 1) / 4 * 0.458 ** 2))
    assert bm.probabilistic_sharpe_ratio(0.458 * math.sqrt(12), 0.0, 24, -2.448, 10.164, periods_per_year=12) == pytest.approx(paper, rel=1e-12)
    assert paper == pytest.approx(0.913, abs=0.002)                                          # a normal return with the same Sharpe ratio would give 0.982: the fat left tail costs seven points
    assert stats.norm.cdf(0.458 * math.sqrt(23) / math.sqrt(1 + 0.458 ** 2 / 2)) == pytest.approx(0.982, abs=0.002)


def test_the_expected_maximum_sharpe_ratio_of_skill_less_trials_matches_simulation():
    rng = np.random.default_rng(0)
    for n_trials, tolerance in ((10, 0.035), (100, 0.012), (1000, 0.008)):                   # the paper's formula is an approximation for many trials: about 2% too high at ten, under 1% at a hundred
        simulated = rng.standard_normal((20_000, n_trials)).max(axis=1).mean()
        assert si.expected_max_sharpe(n_trials, 1.0) == pytest.approx(simulated, rel=tolerance)
        assert si.expected_max_sharpe(n_trials, 1.0) > simulated * 0.995                     # and it errs on the side of a higher bar
    assert si.expected_max_sharpe(50, 0.04) == pytest.approx(0.2 * si.expected_max_sharpe(50, 1.0), rel=1e-12) and si.expected_max_sharpe(1) == 0.0


def test_the_deflated_sharpe_ratio_is_the_probabilistic_one_against_the_expected_maximum(r):
    x = r.to_numpy()
    n, sr = len(x), x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    annual = sr * math.sqrt(ANN)
    variance = 0.03 ** 2                                                                    # the trials' per-period Sharpe ratios spread by 0.03
    exact = bm.deflated_sharpe_ratio(annual, 40, n, g3, g4, trial_sharpe_variance=variance)
    bar = si.expected_max_sharpe(40, variance)
    assert exact == pytest.approx(bm.probabilistic_sharpe_ratio(annual, bar * math.sqrt(ANN), n, g3, g4), rel=1e-12)
    rng = np.random.default_rng(3)
    trials = rng.normal(0.0, 0.03, 40)
    full = si.deflated_sharpe_ratio(x, trial_sharpes=trials)
    assert full["probability"] == pytest.approx(bm.deflated_sharpe_ratio(annual, 40, n, g3, g4, trial_sharpe_variance=full["trial_variance"]), rel=1e-6)       # the platform's two implementations agree
    assert full["benchmark"] == pytest.approx(si.expected_max_sharpe(40, full["trial_variance"]), rel=1e-12)
    approximate = bm.deflated_sharpe_ratio(annual, 40, n, g3, g4)                           # no trial Sharpe ratios: their variance is taken to be 1 / n
    assert approximate == pytest.approx(bm.deflated_sharpe_ratio(annual, 40, n, g3, g4, trial_sharpe_variance=1.0 / n), rel=1e-12)
    assert bm.deflated_sharpe_ratio(annual, 400, n, g3, g4) < approximate < bm.deflated_sharpe_ratio(annual, 2, n, g3, g4)       # more trials, a higher bar
    assert bm.deflated_sharpe_ratio(annual, 1, n, g3, g4) == pytest.approx(bm.probabilistic_sharpe_ratio(annual, 0.0, n, g3, g4), rel=1e-12)


def test_the_standard_errors_of_a_sharpe_ratio_cover_at_their_stated_rate():
    """Simulated: 400 samples of 500 fat-tailed returns with a true per-period Sharpe ratio of 0.05. A 90% interval from the standard error that allows for skewness and kurtosis must hold the truth about
    90% of the time."""
    rng = np.random.default_rng(5)
    true, hits, runs = 0.05, 0, 400
    for _ in range(runs):
        x = rng.standard_t(6, 500) / math.sqrt(6 / 4) + true
        sr = x.mean() / x.std(ddof=1)
        hits += abs(sr - true) <= 1.645 * si.sharpe_standard_error(x, "mertens")
    assert 0.85 <= hits / runs <= 0.95
    x = rng.standard_normal(4000) * 0.01
    assert si.sharpe_standard_error(x, "iid") == pytest.approx(math.sqrt((1 + 0.5 * (x.mean() / x.std(ddof=1)) ** 2) / len(x)), rel=1e-12)


def test_autocorrelated_returns_get_a_wider_interval_than_independent_ones():
    rng = np.random.default_rng(8)
    e = rng.standard_normal(3000) * 0.01
    ar = np.empty_like(e)
    ar[0] = e[0]
    for t in range(1, len(e)):
        ar[t] = 0.4 * ar[t - 1] + e[t]
    assert si.sharpe_standard_error(ar, "lo") > 1.3 * si.sharpe_standard_error(ar, "iid")   # positive autocorrelation: the naive error is too small (Lo 2002)
    assert si.sharpe_standard_error(e, "lo") == pytest.approx(si.sharpe_standard_error(e, "iid"), rel=0.15)
