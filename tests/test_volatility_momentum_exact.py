"""Volatility estimators, forecast scores and momentum features, checked against numbers worked out by hand.

A planted-bug audit (``experiments/plant_bugs.py``) found these two modules almost untested: of the places where one operator, comparison or number was changed in ``features/volatility.py`` the
tests noticed 1 of 146, and in ``features/momentum.py`` 2 of 35. Position sizing divides by these volatilities and every trend signal is one of these momenta, so what each estimator computes,
from which window and with how many observations, is stated here with the formula written out in numpy. The audit also found that a window shorter than ten days (five, for the range estimators)
stopped the estimators with an error from pandas, because the minimum number of observations was larger than the window; they now use the whole window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import momentum as mom
from src.features import volatility as vol

SQRT252 = np.sqrt(252.0)


def make_returns(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": rng.normal(0.0002, 0.01, n), "B": rng.normal(0.0, 0.02, n)}, index=pd.bdate_range("2020-01-01", periods=n))


def first_valid(frame: pd.DataFrame, col: str = "A") -> int:
    return int(np.argmax(frame[col].notna().to_numpy()))


# ------------------------------------------------------------------------------------------------------------------ close-to-close estimators
def test_rolling_volatility_is_the_sample_deviation_of_the_window_scaled_to_a_year():
    r = make_returns()
    out = vol.rolling_volatility(r)
    k = 80
    assert out["A"].iloc[k] == pytest.approx(np.std(r["A"].iloc[k - 62:k + 1], ddof=1) * SQRT252) and out["B"].iloc[k] == pytest.approx(np.std(r["B"].iloc[k - 62:k + 1], ddof=1) * SQRT252)
    assert first_valid(out) == 36, "63 days, 60% of them (37) are the least that make an estimate"
    assert vol.rolling_volatility(r, annualise=False)["A"].iloc[k] == pytest.approx(np.std(r["A"].iloc[k - 62:k + 1], ddof=1))
    assert first_valid(vol.rolling_volatility(r, window=20)) == 11 and vol.rolling_volatility(r, window=20)["A"].iloc[30] == pytest.approx(np.std(r["A"].iloc[11:31], ddof=1) * SQRT252)
    assert first_valid(vol.rolling_volatility(r, window=10)) == 9 and first_valid(vol.rolling_volatility(r, window=5)) == 4, "a window under ten days uses the whole window"
    assert first_valid(vol.rolling_volatility(r, window=15)) == 9 and first_valid(vol.rolling_volatility(r, window=18)) == 9, "a window of 11 to 18 days: 60% of it is under ten, so ten days are needed"
    assert first_valid(vol.rolling_volatility(r, window=63, min_periods=15)) == 14 and first_valid(vol.rolling_volatility(r, window=100)) == 59, "100 days, 60 needed"
    assert vol.rolling_volatility(r, 63, True).equals(vol.rolling_volatility(r)), "63 days, annualised, unless told otherwise"


def test_expanding_volatility_uses_everything_so_far_once_it_has_enough():
    r = make_returns(150)
    out = vol.expanding_volatility(r)
    assert first_valid(out) == 62 and out["A"].iloc[100] == pytest.approx(np.std(r["A"].iloc[:101], ddof=1) * SQRT252)
    assert first_valid(vol.expanding_volatility(r, min_periods=10)) == 9 and vol.expanding_volatility(r, annualise=False)["B"].iloc[120] == pytest.approx(np.std(r["B"].iloc[:121], ddof=1))


def test_ewma_volatility_weights_the_squared_returns_by_halflife_and_is_not_centred():
    r = make_returns()
    out = vol.ewma_volatility(r)
    k, half = 60, 40.0
    alpha = 1.0 - 0.5 ** (1.0 / half)
    weights = (1.0 - alpha) ** np.arange(k, -1, -1)                                         # the newest return weighs most
    expected = np.sqrt(np.sum(weights * r["A"].iloc[:k + 1].to_numpy() ** 2) / weights.sum())
    assert out["A"].iloc[k] == pytest.approx(expected * SQRT252) and first_valid(out) == 29, "thirty observations are the least"
    quick = vol.ewma_volatility(r, halflife=10.0, annualise=False, min_periods=5)
    w10 = (1.0 - (1.0 - 0.5 ** (1.0 / 10.0))) ** np.arange(k, -1, -1)
    assert quick["A"].iloc[k] == pytest.approx(np.sqrt(np.sum(w10 * r["A"].iloc[:k + 1].to_numpy() ** 2) / w10.sum())) and first_valid(quick) == 4
    assert vol.lambda_to_halflife(0.94) == pytest.approx(np.log(0.5) / np.log(0.94)) and vol.lambda_to_halflife(0.94) == pytest.approx(11.2, abs=0.05) and vol.lambda_to_halflife(0.5) == pytest.approx(1.0)
    spike = r.copy()
    spike.iloc[k, 0] = 0.2
    assert vol.ewma_volatility(spike)["A"].iloc[k] > 2 * out["A"].iloc[k], "a shock today lifts the estimate at once"


def test_realised_volatility_is_the_root_mean_square_and_the_forward_one_looks_ahead_by_the_horizon():
    r = make_returns()
    out = vol.realised_volatility(r)
    k = 50
    assert out["A"].iloc[k] == pytest.approx(np.sqrt(np.mean(r["A"].iloc[k - 20:k + 1].to_numpy() ** 2)) * SQRT252), "21 days, the mean not taken off"
    assert first_valid(out) == 9 and first_valid(vol.realised_volatility(r, window=4)) == 3 and first_valid(vol.realised_volatility(r, window=8)) == 4, "half the window, at least five, never more than the window"
    assert vol.realised_volatility(r, annualise=False)["A"].iloc[k] == pytest.approx(np.sqrt(np.mean(r["A"].iloc[k - 20:k + 1].to_numpy() ** 2)))
    ahead = vol.forward_realised_volatility(r, horizon=10)
    assert ahead["A"].iloc[40] == pytest.approx(np.sqrt(np.mean(r["A"].iloc[41:51].to_numpy() ** 2)) * SQRT252), "the ten days after day 40, stamped on day 40"
    assert ahead["A"].iloc[-10:].isna().all() and ahead["A"].iloc[:-10].notna().sum() > 80
    assert vol.forward_realised_volatility(r, annualise=False)["A"].iloc[30] == pytest.approx(np.sqrt(np.mean(r["A"].iloc[31:52].to_numpy() ** 2))), "21 days ahead by default"
    changed = r.copy()
    changed.iloc[45] = 0.5
    assert vol.forward_realised_volatility(changed, 10)["A"].iloc[:35].equals(ahead["A"].iloc[:35]), "a return on day 45 does not reach a stamp before day 35"


# ------------------------------------------------------------------------------------------------------------------ range estimators
def test_the_parkinson_estimator_is_the_mean_squared_log_range_over_four_ln_two():
    n = 40
    idx = pd.bdate_range("2020-01-01", periods=n)
    low = pd.DataFrame({"A": np.full(n, 100.0)}, index=idx)
    high = low * np.exp(0.02)
    daily = np.sqrt(0.02 ** 2 / (4 * np.log(2.0)))
    assert vol.parkinson_volatility(high, low)["A"].iloc[30] == pytest.approx(daily * SQRT252) and vol.parkinson_volatility(high, low, annualise=False)["A"].iloc[30] == pytest.approx(daily)
    ranges = np.linspace(0.01, 0.05, n)
    wide = low * np.exp(ranges[:, None])
    out = vol.parkinson_volatility(wide, low, window=10)
    assert out["A"].iloc[25] == pytest.approx(np.sqrt(np.mean(ranges[16:26] ** 2) / (4 * np.log(2.0))) * SQRT252) and first_valid(out) == 4, "ten days and the half of them, five, are the least"
    assert first_valid(vol.parkinson_volatility(high, low)) == 9 and first_valid(vol.parkinson_volatility(high, low, window=3)) == 2, "21 days by default; a window of three uses all three"


def test_the_garman_klass_estimator_adds_the_open_to_close_move_and_never_goes_below_zero():
    n = 40
    idx = pd.bdate_range("2020-01-01", periods=n)
    open_ = pd.DataFrame({"A": np.full(n, 100.0)}, index=idx)
    low = open_ * 0.98
    high = low * np.exp(0.03)
    close = open_ * np.exp(0.01)
    daily = 0.5 * 0.03 ** 2 - (2 * np.log(2.0) - 1.0) * 0.01 ** 2
    out = vol.garman_klass_volatility(open_, high, low, close)
    assert out["A"].iloc[30] == pytest.approx(np.sqrt(daily) * SQRT252) and vol.garman_klass_volatility(open_, high, low, close, annualise=False)["A"].iloc[30] == pytest.approx(np.sqrt(daily))
    assert first_valid(out) == 9 and first_valid(vol.garman_klass_volatility(open_, high, low, close, window=3)) == 2
    assert first_valid(vol.garman_klass_volatility(open_, high, low, close, window=8)) == 4, "half of eight is four: five days are the least"
    quiet_range = open_ * np.exp(0.001)
    assert vol.garman_klass_volatility(open_, quiet_range, open_, open_ * np.exp(0.02))["A"].iloc[30] == 0.0, "a big close-to-open move in a narrow range would be a negative variance: it is zero"
    moving = np.linspace(0.02, 0.06, n)
    wide = open_ * np.exp(moving[:, None])
    expected = np.sqrt(np.mean(0.5 * moving[11:31] ** 2 - (2 * np.log(2.0) - 1.0) * 0.0 ** 2)) * SQRT252
    assert vol.garman_klass_volatility(open_, wide, open_, open_, window=20)["A"].iloc[30] == pytest.approx(expected), "an unchanged close leaves the range term alone"


class Market:
    def __init__(self, n: int = 90):
        rng = np.random.default_rng(8)
        idx = pd.bdate_range("2020-01-01", periods=n)
        self.close = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=idx)
        self.open_ = self.close.shift(1).fillna(self.close.iloc[0])
        self.high = pd.concat([self.open_, self.close]).groupby(level=0).max() * 1.004
        self.low = pd.concat([self.open_, self.close]).groupby(level=0).min() * 0.996

    def returns(self) -> pd.DataFrame:
        return self.close.pct_change()


def test_the_volatility_panel_dispatches_each_method_to_its_estimator_with_the_settings_given():
    m = Market()
    r = m.returns()
    pd.testing.assert_frame_equal(vol.volatility_panel(m, "rolling", window=30), vol.rolling_volatility(r, 30, True))
    pd.testing.assert_frame_equal(vol.volatility_panel(m, "expanding", window=20, annualise=False), vol.expanding_volatility(r, 20, False))
    pd.testing.assert_frame_equal(vol.volatility_panel(m, "ewma", halflife=15.0), vol.ewma_volatility(r, 15.0, True))
    pd.testing.assert_frame_equal(vol.volatility_panel(m), vol.ewma_volatility(r, 40.0, True), obj="ewma with a half-life of 40 days is the default")
    pd.testing.assert_frame_equal(vol.volatility_panel(m, "rolling"), vol.rolling_volatility(r, 63, True), obj="and 63 days is the default window")
    pd.testing.assert_frame_equal(vol.volatility_panel(m, "parkinson", window=15), vol.parkinson_volatility(m.high, m.low, 15, True))
    pd.testing.assert_frame_equal(vol.volatility_panel(m, "garman_klass", window=15, annualise=False), vol.garman_klass_volatility(m.open_, m.high, m.low, m.close, 15, False))
    pd.testing.assert_frame_equal(vol.volatility_panel(m, "realised", window=25), vol.realised_volatility(r, 25, True))
    with pytest.raises(ValueError, match="unknown volatility method 'garch'"):
        vol.volatility_panel(m, "garch")


# ------------------------------------------------------------------------------------------------------------------ scoring forecasts
def test_forecast_scores_are_the_correlation_the_errors_and_the_qlike_loss_by_hand():
    idx = pd.bdate_range("2020-01-01", periods=40)
    realised = pd.Series(np.linspace(0.10, 0.30, 40), index=idx)
    table = vol.evaluate_forecasts({"high": realised * 1.1, "low": realised * 0.8, "wide": realised * 1.25}, realised)
    assert list(table.index) == ["high", "wide", "low"], "best score first"
    assert list(table.columns) == ["n_obs", "corr", "rmse", "mae", "mean_bias", "mean_ratio", "qlike", "r2"]
    r = realised.to_numpy()
    for name, k in (("high", 1.1), ("low", 0.8), ("wide", 1.25)):
        f = k * r
        ratio = r ** 2 / f ** 2
        row = table.loc[name]
        assert row["n_obs"] == 40 and row["corr"] == pytest.approx(1.0) and row["rmse"] == pytest.approx(np.sqrt(np.mean((f - r) ** 2))) and row["mae"] == pytest.approx(np.mean(np.abs(f - r)))
        assert row["mean_bias"] == pytest.approx(np.mean(f - r)) and row["mean_ratio"] == pytest.approx(k) and row["qlike"] == pytest.approx(np.mean(ratio - np.log(ratio) - 1.0))
        assert row["r2"] == pytest.approx(1.0 - np.sum((r - f) ** 2) / np.sum((r - r.mean()) ** 2))
    assert table.loc["low", "qlike"] == pytest.approx(1 / 0.64 - np.log(1 / 0.64) - 1) and table.loc["high", "qlike"] == pytest.approx(1 / 1.21 - np.log(1 / 1.21) - 1)
    assert table["qlike"].is_monotonic_increasing and table.index[0] == "high" and table.index[-1] == "low"
    assert table.loc["low", "qlike"] > table.loc["wide", "qlike"], "twenty percent too low costs more than twenty-five percent too high: under-forecasting is what a position sizer cannot afford"
    assert table.loc["low", "mean_bias"] < 0 < table.loc["high", "mean_bias"]


def test_forecasts_with_too_few_or_unusable_observations_are_left_out_of_the_table():
    idx = pd.bdate_range("2020-01-01", periods=60)
    realised = pd.Series(np.linspace(0.1, 0.3, 60), index=idx)
    forecast = realised * 1.1
    assert vol.evaluate_forecasts({"x": forecast.iloc[:29]}, realised).empty and len(vol.evaluate_forecasts({"x": forecast.iloc[:30]}, realised)) == 1, "thirty days are the least"
    messy = forecast.copy()
    messy.iloc[:5] = [0.0, -0.2, np.nan, 0.0, -1.0]
    row = vol.evaluate_forecasts({"x": messy}, realised).loc["x"]
    assert row["n_obs"] == 55, "a forecast of zero, a negative one or none at all is not scored"
    zero_realised = realised.copy()
    zero_realised.iloc[10:13] = [0.0, 0.0, -0.1]
    assert vol.evaluate_forecasts({"x": forecast}, zero_realised).loc["x", "n_obs"] == 57
    assert vol.evaluate_forecasts({}, realised).empty


def test_the_forecast_of_a_garch_horizon_decays_to_the_long_run_or_grows_if_integrated():
    assert vol._horizon_variance(2.0, 0.1, 0.9, 3) == pytest.approx(np.mean([1.0 + 0.9 ** k * (2.0 - 1.0) for k in range(3)]))
    assert vol._horizon_variance(2.0, 0.1, 0.9, 1) == pytest.approx(2.0) and vol._horizon_variance(0.5, 0.1, 0.9, 400) == pytest.approx(1.0 + (0.5 - 1.0) * (1 - 0.9 ** 400) / (1 - 0.9) / 400), "one step is next period's variance; the average over a long horizon tends to the long-run one"
    assert vol._horizon_variance(2.0, 0.1, 1.0, 3) == pytest.approx(np.mean([2.0, 2.1, 2.2])) and vol._horizon_variance(2.0, 0.1, 1.2, 3) == pytest.approx(2.1), "no long-run variance to decay to: it grows by omega each step"
    assert vol._horizon_variance(2.0, 0.1, 1.0 - 1e-6, 3) == pytest.approx(2.1, abs=1e-4), "a persistence a millionth under one is as good as integrated: the two forecasts join without a jump"


# ------------------------------------------------------------------------------------------------------------------ GARCH (needs the optional arch package)
arch = pytest.importorskip("arch")


def garch_series(n: int = 600, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 4e-6, 0.08, 0.90
    sigma2, eps = omega / (1 - alpha - beta), 0.0
    out = []
    for _ in range(n):
        sigma2 = omega + alpha * eps ** 2 + beta * sigma2
        eps = np.sqrt(sigma2) * rng.standard_normal()
        out.append(eps)
    return pd.Series(out, index=pd.bdate_range("2015-01-01", periods=n))


class Fitted:
    def __init__(self, params, vols, resid):
        self.params = pd.Series(params, dtype=float)
        self.conditional_volatility = np.asarray(vols, dtype=float)
        self.resid = np.asarray(resid, dtype=float)


def test_a_fit_is_read_back_in_return_units_from_its_scaled_parameters():
    fit = vol.GarchFit(Fitted({"mu": 0.1, "omega": 0.05, "alpha[1]": 0.08, "alpha[2]": 0.02, "beta[1]": 0.85, "nu": 6.0}, [1.0, 2.0, 4.0], [0.5, -1.0, 3.0]), scale=100.0)
    assert fit.unpack() == pytest.approx((0.05, 0.10, 0.85, 0.1)), "the alpha terms and the beta terms are summed"
    assert fit.params["nu"] == 6.0 and fit.conditional_volatility().tolist() == pytest.approx([0.01, 0.02, 0.04]) and fit.last_state() == pytest.approx((16.0, 3.0))
    bare = vol.GarchFit(Fitted({"omega": 0.05, "alpha[1]": 0.08, "beta[1]": 0.9}, [1.0], [0.0]), scale=1.0)
    assert bare.unpack()[3] == 0.0 and np.isnan(vol.GarchFit(Fitted({"alpha[1]": 0.1}, [1.0], [0.0]), 1.0).unpack()[0]), "no mean is a mean of zero, and no omega is not-a-number"


def test_a_garch_fit_needs_a_hundred_observations_and_scales_the_series_to_unit_deviation():
    s = garch_series()
    with pytest.raises(ValueError, match="need at least 100 observations to fit GARCH, got 99"):
        vol.fit_garch(s.iloc[:99])
    assert vol.fit_garch(s.iloc[:100]).scale == pytest.approx(1.0 / s.iloc[:100].std(ddof=1))
    gappy = s.iloc[:120].copy()
    gappy.iloc[:30] = np.nan
    with pytest.raises(ValueError, match="got 90"):
        vol.fit_garch(gappy)                                                                    # missing days are not observations
    fixed = vol.fit_garch(s, rescale=5.0)
    assert fixed.scale == 5.0 and vol.fit_garch(s, p=1, q=1, dist="normal").unpack()[1] > 0


def test_the_conditional_volatility_and_the_parameter_table_agree_with_the_fit():
    s = garch_series()
    fit = vol.fit_garch(s)
    cond = vol.garch_conditional_volatility(s, annualise=False)
    assert list(cond.index) == list(s.index) and cond.to_numpy() == pytest.approx(fit.conditional_volatility()) and (cond > 0).all()
    assert vol.garch_conditional_volatility(s).to_numpy() == pytest.approx(cond.to_numpy() * SQRT252)
    assert vol.garch_conditional_volatility(s.where(np.arange(len(s)) != 10), annualise=False).index.equals(s.drop(s.index[10]).index), "a missing day is not a day"
    table = vol.garch_parameters(s)
    omega, alpha, beta, _ = fit.unpack()
    persistence = alpha + beta
    assert table["alpha"] == pytest.approx(alpha) and table["beta"] == pytest.approx(beta) and table["persistence"] == pytest.approx(persistence)
    assert table["omega"] == pytest.approx(omega / fit.scale ** 2) and table["integrated"] is False
    assert table["half_life_days"] == pytest.approx(np.log(0.5) / np.log(persistence)) and table["long_run_vol_ann"] == pytest.approx(np.sqrt(omega / fit.scale ** 2 / (1 - persistence)) * SQRT252)
    assert table["loglikelihood"] == pytest.approx(float(fit.result.loglikelihood)) and table["aic"] == pytest.approx(float(fit.result.aic)) and table["bic"] == pytest.approx(float(fit.result.bic)) and table["nu"] == pytest.approx(float(fit.params["nu"]))
    assert 0.5 < persistence < 1.0 and 5.0 < table["half_life_days"] < 500.0 and 0.05 < table["long_run_vol_ann"] < 0.6


def test_a_walk_forward_garch_forecast_stamps_only_days_after_the_training_window_and_never_looks_ahead(monkeypatch):
    s = garch_series(420)
    kw = dict(horizon=5, refit_every=40, min_train=300)
    out = vol.garch_rolling_forecast(s, annualise=False, **kw)
    assert out.index[0] == s.index[300] and out.index[-1] == s.index[-1] and len(out) == 120 and (out > 0).all()
    assert vol.garch_rolling_forecast(s, **kw).to_numpy() == pytest.approx(out.to_numpy() * SQRT252)
    changed = s.copy()
    changed.iloc[-1] = 0.5
    again = vol.garch_rolling_forecast(changed, annualise=False, **kw)
    assert again.iloc[:-1].equals(out.iloc[:-1]), "a return on the last day reaches no earlier forecast, and not even its own stamp"
    assert again.iloc[-1] == out.iloc[-1]
    short = vol.garch_rolling_forecast(s.iloc[:304], horizon=5, refit_every=40, min_train=300)
    assert len(short) == 304 and short.isna().all(), "too little data after the training window to forecast ahead: nothing, on the days given"
    assert vol.garch_rolling_forecast(s.iloc[:305], horizon=5, refit_every=40, min_train=300, annualise=False).notna().sum() == 5, "five days after the training window are enough"
    capped = vol.garch_rolling_forecast(s, annualise=False, max_vol_multiple=0.1, **kw)
    assert capped.iloc[:40].to_numpy() == pytest.approx(0.1 * s.iloc[:300].std(ddof=1)), "a cap of a tenth of the training deviation binds on every day of the first block"
    assert capped.iloc[40:80].to_numpy() == pytest.approx(0.1 * s.iloc[:340].std(ddof=1)) and (out > capped).all(), "the next block is capped at a tenth of the deviation of the longer training window"
    inside = s.copy()
    inside.iloc[350] = 0.4
    moved = vol.garch_rolling_forecast(inside, annualise=False, **kw)
    assert moved.loc[:s.index[350]].equals(out.loc[:s.index[350]]) and not moved.loc[s.index[351]:].equals(out.loc[s.index[351]:]), "a return on day 350 changes forecasts from day 351 and none before"

    def refuse(*args, **kwargs):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(vol, "fit_garch", refuse)
    with pytest.raises(RuntimeError, match=r"GARCH walk-forward produced no forecasts \(3 refits failed\)"):
        vol.garch_rolling_forecast(s, **kw)


class Scored(Fitted):
    loglikelihood, aic, bic = -100.0, 210.0, 220.0


def test_a_persistence_within_a_millionth_of_one_is_integrated_and_has_no_long_run_volatility(monkeypatch):
    def table(persistence: float) -> dict:
        fitted = Scored({"mu": 0.0, "omega": 0.05, "alpha[1]": 0.1, "beta[1]": persistence - 0.1}, [1.0], [0.0])
        monkeypatch.setattr(vol, "fit_garch", lambda *args, **kwargs: vol.GarchFit(fitted, scale=1.0))
        return vol.garch_parameters(pd.Series(dtype=float))

    inside, outside = table(1.0 - 0.8e-6), table(1.0 - 1.2e-6)
    assert inside["integrated"] is True and np.isnan(inside["long_run_vol_ann"]), "0.8 millionths short of one is as good as one"
    assert outside["integrated"] is False and outside["long_run_vol_ann"] == pytest.approx(np.sqrt(0.05 / 1.2e-6) * SQRT252, rel=1e-3), "1.2 millionths short has a long-run volatility, a very high one"
    explosive = table(1.2)
    assert table(1.0)["integrated"] is True and explosive["integrated"] is True and np.isnan(explosive["long_run_vol_ann"])
    assert table(1.0)["half_life_days"] == float("inf") and explosive["half_life_days"] == float("inf") and table(0.9)["half_life_days"] == pytest.approx(np.log(0.5) / np.log(0.9))


def stub_the_optimiser(monkeypatch, sigma: float = 1.0, resid: float = 2.0, scale: float = 100.0) -> list[int]:
    """Swap the optimiser for a fit whose parameters and last state are known, so the walk-forward recursion can be checked against numbers worked out by hand.

    The fit has mu 0.3, omega 0.1, alpha 0.1 and beta 0.8 in scaled units (persistence 0.9, long-run variance 1.0); the list returned fills with the length of every training window asked for."""
    asked: list[int] = []
    fitted = Fitted({"mu": 0.3, "omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8}, [sigma], [resid])

    def fake(train, *args, **kwargs):
        asked.append(len(train))
        return vol.GarchFit(fitted, scale=scale)

    monkeypatch.setattr(vol, "fit_garch", fake)
    return asked


def test_the_walk_forward_forecast_rolls_the_variance_forward_with_each_realised_shock(monkeypatch):
    asked = stub_the_optimiser(monkeypatch)
    returns = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005, 0.01, -0.01, 0.02, -0.005, 0.0, 0.03, -0.015], index=pd.bdate_range("2020-01-01", periods=12))
    out = vol.garch_rolling_forecast(returns, horizon=2, refit_every=4, min_train=5, annualise=False)
    assert asked == [5, 9] and list(out.index) == list(returns.index[5:]), "refit on the first five days and again four days later"
    # day 5: sigma2 = 0.1 + 0.1 * 2.0^2 + 0.8 * 1.0 = 1.3; the next two days average 1.3 and 1.0 + 0.9 * 0.3 = 1.27
    assert out.iloc[0] == pytest.approx(np.sqrt((1.3 + 1.27) / 2) / 100.0)
    # the shock of day 5 is 0.01 * 100 - mu 0.3 = 0.7, so day 6: sigma2 = 0.1 + 0.1 * 0.7^2 + 0.8 * 1.3 = 1.189; then 1.189 and 1.0 + 0.9 * 0.189 = 1.1701
    assert out.iloc[1] == pytest.approx(np.sqrt((1.189 + 1.1701) / 2) / 100.0)
    # day 6: -0.01 * 100 - 0.3 = -1.3, so day 7: sigma2 = 0.1 + 0.1 * 1.69 + 0.8 * 1.189 = 1.2202; then 1.2202 and 1.0 + 0.9 * 0.2202
    assert out.iloc[2] == pytest.approx(np.sqrt((1.2202 + 1.19818) / 2) / 100.0)
    assert out.iloc[4] == out.iloc[0], "a refit starts again from the state of the new fit, not from the days rolled forward before it"
    assert vol.garch_rolling_forecast(returns, horizon=2, refit_every=4, min_train=5).to_numpy() == pytest.approx(out.to_numpy() * SQRT252)


def test_the_walk_forward_cap_limits_the_variance_the_next_day_starts_from(monkeypatch):
    stub_the_optimiser(monkeypatch, sigma=1000.0)                                                 # a variance of a million, far past any cap
    returns = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005, 0.01, -0.01, 0.02, -0.005], index=pd.bdate_range("2020-01-01", periods=9))
    train_std = returns.iloc[:5].std(ddof=1)

    def two_day_average(start: float) -> float:
        return float(np.mean([start, 1.0 + 0.9 * (start - 1.0)]))                                  # the day and the next one, falling towards the long-run variance of 1.0

    cap2 = (2.0 * train_std * 100.0) ** 2
    out = vol.garch_rolling_forecast(returns, horizon=2, refit_every=4, min_train=5, annualise=False, max_vol_multiple=2.0)
    assert out.iloc[0] == pytest.approx(np.sqrt(two_day_average(cap2)) / 100.0), "the variance of a million is held to the cap, twice the training deviation squared"
    # the capped variance, not the million, is what day 6 starts from (its shock is 0.01 * 100 - 0.3 = 0.7): 0.1 + 0.1 * 0.7^2 + 0.8 * cap2
    assert out.iloc[1] == pytest.approx(np.sqrt(two_day_average(0.1 + 0.1 * 0.7 ** 2 + 0.8 * cap2)) / 100.0) and out.iloc[1] < out.iloc[0]
    default = vol.garch_rolling_forecast(returns, horizon=2, refit_every=4, min_train=5, annualise=False)
    assert default.iloc[0] == pytest.approx(np.sqrt(two_day_average((10.0 * train_std * 100.0) ** 2)) / 100.0), "ten times the training deviation unless told otherwise"


def test_the_walk_forward_forecast_trains_for_750_days_and_looks_21_ahead_refitting_every_63_unless_told_otherwise(monkeypatch):
    asked = stub_the_optimiser(monkeypatch)
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 900), index=pd.bdate_range("2015-01-01", periods=900))
    assert vol.garch_rolling_forecast(returns.iloc[:770]).isna().all() and asked == [], "750 days to train on and 21 to forecast: 770 are too few"
    assert len(vol.garch_rolling_forecast(returns.iloc[:771])) == 21 and asked == [750]
    del asked[:]
    out = vol.garch_rolling_forecast(returns)
    assert asked == [750, 813, 876] and out.index[0] == returns.index[750] and len(out) == 150


# ------------------------------------------------------------------------------------------------------------------ momentum
def make_prices(n: int = 300, seed: int = 2, columns=("A", "B", "C")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, len(columns))), axis=0)), index=pd.bdate_range("2020-01-01", periods=n), columns=list(columns))


def test_momentum_is_the_return_over_the_lookback_ending_skip_days_ago():
    p = make_prices()
    m = mom.total_return_momentum(p)
    k = 200
    assert m["A"].iloc[k] == pytest.approx(p["A"].iloc[k] / p["A"].iloc[k - 126] - 1.0) and m["A"].iloc[:126].isna().all() and m["A"].iloc[126:].notna().all(), "126 days by default"
    s = mom.total_return_momentum(p, lookback=20, skip=5)
    assert s["B"].iloc[k] == pytest.approx(p["B"].iloc[k - 5] / p["B"].iloc[k - 25] - 1.0) and s["B"].iloc[:25].isna().all() and s["B"].iloc[25:].notna().all()
    assert mom.total_return_momentum(p, 1)["A"].iloc[5] == pytest.approx(p["A"].iloc[5] / p["A"].iloc[4] - 1.0)
    with pytest.raises(ValueError, match="lookback must be >= 1"):
        mom.total_return_momentum(p, 0)
    lg = mom.log_momentum(p, lookback=30, skip=2)
    assert lg["C"].iloc[k] == pytest.approx(np.log(p["C"].iloc[k - 2]) - np.log(p["C"].iloc[k - 32])) and lg["C"].iloc[:32].isna().all()
    assert mom.log_momentum(p)["A"].iloc[k] == pytest.approx(np.log(p["A"].iloc[k] / p["A"].iloc[k - 126])) and mom.log_momentum(p)["A"].iloc[:126].isna().all()
    sign = mom.time_series_momentum_sign(p, lookback=60, skip=1)
    assert sign["A"].iloc[k] == np.sign(p["A"].iloc[k - 1] / p["A"].iloc[k - 61] - 1.0) and sign["A"].iloc[:61].isna().all() and set(sign["A"].dropna().unique()) <= {-1.0, 0.0, 1.0}
    assert mom.time_series_momentum_sign(p)["A"].iloc[:252].isna().all() and mom.time_series_momentum_sign(p)["A"].iloc[252:].notna().all(), "a year by default"


def test_momentum_per_unit_of_risk_divides_by_the_volatility_known_at_the_start_of_the_window():
    p = make_prices()
    r = p.pct_change()
    k = 200
    scaled = mom.volatility_scaled_momentum(p, r, lookback=60, skip=3)
    raw = mom.total_return_momentum(p, 60, 3)
    sigma = vol.rolling_volatility(r, window=63).shift(3)
    assert scaled["A"].iloc[k] == pytest.approx(raw["A"].iloc[k] / sigma["A"].iloc[k]), "rolling volatility over 63 days, as it was three days ago"
    wide = mom.volatility_scaled_momentum(p, r, lookback=60, skip=0, vol_lookback=20)
    assert wide["B"].iloc[k] == pytest.approx(mom.total_return_momentum(p, 60, 0)["B"].iloc[k] / vol.rolling_volatility(r, window=20)["B"].iloc[k])
    smooth = mom.volatility_scaled_momentum(p, r, lookback=60, skip=0, vol_method="ewma", halflife=15.0)
    assert smooth["C"].iloc[k] == pytest.approx(mom.total_return_momentum(p, 60, 0)["C"].iloc[k] / vol.ewma_volatility(r, halflife=15.0)["C"].iloc[k])
    assert not np.isclose(smooth["C"].iloc[k], mom.volatility_scaled_momentum(p, r, lookback=60, skip=0)["C"].iloc[k]), "the method makes a difference"
    forty = mom.volatility_scaled_momentum(p, r, lookback=60, skip=0, vol_method="ewma")
    assert forty["C"].iloc[k] == pytest.approx(mom.total_return_momentum(p, 60, 0)["C"].iloc[k] / vol.ewma_volatility(r, halflife=40.0)["C"].iloc[k]), "a half-life of 40 days unless told otherwise"
    flat = p.copy()
    flat["A"] = 100.0
    assert mom.volatility_scaled_momentum(flat, flat.pct_change(), lookback=60)["A"].iloc[k:].isna().all(), "a price that never moves has no volatility to divide by"
    default = mom.volatility_scaled_momentum(p, r)
    assert default["A"].iloc[k] == pytest.approx(mom.total_return_momentum(p, 126, 0)["A"].iloc[k] / vol.rolling_volatility(r, window=63)["A"].iloc[k]), "126 days over a 63-day volatility, by default"


def test_ranks_keep_the_order_centre_it_and_leave_missing_values_missing():
    f = pd.DataFrame({"A": [3.0, 1.0], "B": [1.0, 2.0], "C": [2.0, np.nan], "D": [np.nan, 3.0]}, index=pd.bdate_range("2020-01-01", periods=2))
    pct = mom.cross_sectional_rank(f)
    assert pct.iloc[0].dropna().to_dict() == pytest.approx({"A": 1.0, "B": 1 / 3, "C": 2 / 3}) and np.isnan(pct.iloc[0]["D"]) and pct.iloc[1].dropna().to_dict() == pytest.approx({"A": 1 / 3, "B": 2 / 3, "D": 1.0})
    assert mom.cross_sectional_rank(f, pct=False).iloc[0].dropna().to_dict() == {"A": 3.0, "B": 1.0, "C": 2.0}
    p = make_prices(columns=("A", "B", "C", "D"))
    plain = mom.ranked_momentum(p, lookback=40, skip=2, centred=False)
    centred = mom.ranked_momentum(p, lookback=40, skip=2)
    k = 100
    raw = mom.total_return_momentum(p, 40, 2).iloc[k]
    assert plain.iloc[k].tolist() == pytest.approx((raw.rank(pct=True)).tolist()) and centred.iloc[k].tolist() == pytest.approx(((raw.rank(pct=True) - 0.5) * 2.0).tolist())
    assert mom.ranked_momentum(p).iloc[:126].isna().all().all() and mom.ranked_momentum(p).equals(mom.ranked_momentum(p, 126, 0, True)), "a six-month lookback, centred, by default"
    assert centred.iloc[k].max() == pytest.approx(1.0) and centred.iloc[k].min() == pytest.approx(-0.5), "four assets: percentiles a quarter to one, centred"


def test_the_moving_average_crossover_is_the_fast_average_over_the_slow_less_one():
    p = make_prices()
    out = mom.moving_average_crossover(p, fast=10, slow=30)
    k = 100
    assert out["A"].iloc[k] == pytest.approx(p["A"].iloc[k - 9:k + 1].mean() / p["A"].iloc[k - 29:k + 1].mean() - 1.0)
    assert first_valid(out) == 14, "the slow average needs half its window, fifteen days (and never fewer than ten), the fast one five"
    assert first_valid(mom.moving_average_crossover(p, fast=20, slow=100)) == 49, "half of a hundred"
    assert first_valid(mom.moving_average_crossover(p)) == 99 and mom.moving_average_crossover(p)["A"].iloc[150] == pytest.approx(p["A"].iloc[101:151].mean() / p["A"].iloc[:151].iloc[-200:].mean() - 1.0), "50 and 200 days by default"
    assert first_valid(mom.moving_average_crossover(p, fast=3, slow=8)) == 7, "windows shorter than the usual minimum use the whole window"
    assert first_valid(mom.moving_average_crossover(p, fast=4, slow=16)) == 9, "half of sixteen is eight: the slow average needs ten days at the least"
    assert mom.moving_average_crossover(p)["A"].iloc[250] == pytest.approx(p["A"].iloc[201:251].mean() / p["A"].iloc[51:251].mean() - 1.0), "once there are 200 days the slow average is the last 200, not 201"
    with pytest.raises(ValueError, match="fast window must be shorter than slow window"):
        mom.moving_average_crossover(p, fast=30, slow=30)
    with pytest.raises(ValueError, match="fast window must be shorter"):
        mom.moving_average_crossover(p, fast=50, slow=20)


def test_the_momentum_family_has_a_raw_a_risk_scaled_and_a_ranked_version_of_every_lookback():
    p = make_prices()
    r = p.pct_change()
    family = mom.momentum_family(p, r, lookbacks=(21, 63))
    assert list(family) == ["raw_21", "vol_scaled_21", "ranked_21", "raw_63", "vol_scaled_63", "ranked_63"]
    pd.testing.assert_frame_equal(family["raw_63"], mom.total_return_momentum(p, 63, 1))
    pd.testing.assert_frame_equal(family["vol_scaled_21"], mom.volatility_scaled_momentum(p, r, 21, 1, 63))
    pd.testing.assert_frame_equal(family["ranked_63"], mom.ranked_momentum(p, 63, 1))
    assert list(mom.momentum_family(p, r)) == [f"{v}_{k}" for k in (21, 63, 126, 252) for v in ("raw", "vol_scaled", "ranked")], "four lookbacks by default"
    assert list(mom.momentum_family(p, r, lookbacks=(21,), vol_lookback=20)) == ["raw_21", "vol_scaled_21", "ranked_21"]
    pd.testing.assert_frame_equal(mom.momentum_family(p, r, lookbacks=(21,), vol_lookback=20)["vol_scaled_21"], mom.volatility_scaled_momentum(p, r, 21, 1, 20))
    pd.testing.assert_frame_equal(mom.momentum_family(p, r, lookbacks=(21,), skip=0)["raw_21"], mom.total_return_momentum(p, 21, 0))
