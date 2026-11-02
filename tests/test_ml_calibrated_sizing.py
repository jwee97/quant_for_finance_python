"""Release 7, P2-06: sizing from calibrated forecasts.

A score is calibrated against what followed it, so the fixtures plant a known relation between score and return: the slope that comes back must be the planted one, a score with no relation must be sized at
zero, an uncertain calibration must shrink the bet, and nothing on a date may depend on anything after it."""

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, contracts as ct, signals as sg, sizing as sz

K = 12
COLS = [f"S{i:02d}" for i in range(K)]


def world(n=700, beta=0.004, noise=0.01, seed=1, sigma=0.01, signal_from=0, persist=0.0):
    """Scores z[t, i]; the return of day t + 2 (the first day of the label of decision t when the lag is 1 and the horizon 1) carries ``beta * z[t, i]``. ``persist`` makes the score an AR(1) in time."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    z = pd.DataFrame(rng.normal(size=(n, K)), index=idx, columns=COLS)
    if persist:
        for i in range(1, n):
            z.iloc[i] = persist * z.iloc[i - 1] + np.sqrt(1.0 - persist ** 2) * z.iloc[i]
    r = pd.DataFrame(rng.normal(0.0, sigma, (n, K)), index=idx, columns=COLS)
    planted = z.shift(2) * beta + rng.normal(0.0, noise, (n, K))
    r = r * 0.0 + planted.fillna(0.0)
    z.iloc[:signal_from] = np.nan
    return z, r


def cfg(**over):
    base = {"method": "calibrated_sizing", "gross": 1.0, "net": 0.0, "max_weight": 0.35, "holding_period": 1, "window_days": 1, "label_offset": 2, "risk_aversion": 5.0, "calib_window": 300, "calib_min_t": 2.0,
            "stability_window": 21}
    base.update(over)
    return base


def test_the_cross_sectional_slope_is_the_ols_slope_of_the_demeaned_return_on_the_demeaned_score():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=30)
    x = pd.DataFrame(rng.normal(size=(30, 8)), index=idx)
    y = 0.5 * x + pd.DataFrame(rng.normal(size=(30, 8)), index=idx)
    x.iloc[4, :5] = np.nan                                                                        # three assets left: fewer than the minimum
    got = sz.cross_sectional_slopes(x, y, min_assets=5)
    for i in (0, 11, 29):
        xi, yi = x.iloc[i].to_numpy(), y.iloc[i].to_numpy()
        assert got.iloc[i] == pytest.approx(np.polyfit(xi, yi, 1)[0], abs=1e-12)
    assert np.isnan(got.iloc[4]), "too few assets: no slope, not zero"
    const = x.copy()
    const.iloc[7] = 1.0
    assert np.isnan(sz.cross_sectional_slopes(const, y).iloc[7]), "a constant score has no slope"


def test_the_calibration_recovers_the_planted_slope_and_only_uses_settled_outcomes():
    z, r = world(beta=0.004, noise=0.01)
    cal = sz.calibration(z, r, 1, 1, window=300, min_dates=60)
    late = cal.iloc[450:]
    assert late["slope"].mean() == pytest.approx(0.004, abs=0.0007) and (late["t"] > 4).all() and cal.iloc[:60]["slope"].isna().all(), "nothing is calibrated before 60 outcomes have settled"
    changed = r.copy()
    changed.iloc[500:] = np.random.default_rng(9).normal(0, 0.05, changed.iloc[500:].shape)           # a different future
    again = sz.calibration(z, changed, 1, 1, window=300, min_dates=60)
    assert np.allclose(cal.iloc[:500].to_numpy(), again.iloc[:500].to_numpy(), equal_nan=True), "the calibration at t does not move when what follows t changes"
    assert (cal["se"].dropna() > 0).all() and cal["n"].dropna().max() <= 300


def test_the_standard_error_of_the_calibration_widens_with_the_horizon_because_labels_overlap():
    rng = np.random.default_rng(4)
    n = 800
    idx = pd.bdate_range("2019-01-02", periods=n)
    z = pd.DataFrame(rng.normal(size=(n, K)), index=idx, columns=COLS)
    z = z.rolling(5, min_periods=1).mean()                                                        # a persistent score: its forward returns overlap in what they explain
    r = pd.DataFrame(rng.normal(0, 0.01, (n, K)), index=idx, columns=COLS) + 0.002 * z.shift(2).fillna(0.0)
    short = sz.calibration(z, r, 1, 1, window=300, min_dates=60)
    long = sz.calibration(z, r, 10, 1, window=300, min_dates=60)
    assert long["se"].dropna().iloc[100:].mean() > short["se"].dropna().iloc[100:].mean()


def test_the_book_is_the_calibrated_forecast_over_risk_scaled_by_the_confidence_and_the_stability():
    z, r = world(n=700, beta=0.004, noise=0.01, seed=2, persist=0.9)
    c = cfg(max_weight=1e6, gross=1e9)                                                            # limits out of the way, to see the formula
    w, log = sz.apply(z, r, None, c)
    i = 600
    cal = sz.calibration(z, r, 1, 1, 300)
    b, t = cal.iloc[i]["slope"], cal.iloc[i]["t"]
    g = 1.0 - 1.0 / t ** 2
    x = z.iloc[i] - z.iloc[i].mean()
    var = r.rolling(63, min_periods=40).var(ddof=1).iloc[i] * 1
    stab = sz.rank_stability(z, 21).iloc[i]
    expected = g * b * x / (5.0 * var) * stab
    expected = expected - expected.sum() / len(expected)                                         # net zero: the same shift for every name
    assert np.allclose(w.iloc[i].to_numpy(), expected.to_numpy(), atol=1e-10), (w.iloc[i].to_numpy()[:3], expected.to_numpy()[:3])
    assert stab > 0.5 and np.abs(expected).sum() > 0, "the fixture has a persistent signal, so the formula is checked at a date where the book is not empty"
    assert abs(w.iloc[i].sum()) < 1e-12 and log["sized"] > 400 and log["flat_no_edge"] == 0 and log["mean_slope"] == pytest.approx(0.004, abs=0.0007) and log["mean_shrink"] > 0.9


def test_with_no_relation_between_score_and_return_the_book_is_flat():
    z, r = world(beta=0.0, noise=0.01, seed=5)
    w, log = sz.apply(z, r, None, cfg())
    assert log["flat_no_edge"] + log["sized"] + log["flat_not_calibrated"] + log["flat_zero_size"] == log["dates"]
    assert log["flat_no_edge"] > 0.9 * (log["dates"] - log["flat_not_calibrated"]), "a score that earns nothing is not sized (at the two-sigma bar a few chance crossings are expected)"
    assert w.abs().sum(axis=1).iloc[:60].max() == 0.0 and w.abs().sum(axis=1).mean() < 0.05
    wrong_way, _ = sz.apply(z, world(beta=-0.004, noise=0.01, seed=5)[1], None, cfg())
    assert wrong_way.abs().sum().sum() == 0.0, "an edge of the wrong sign is no edge: nothing is shorted on the strength of a negative calibration"


def test_a_noisier_calibration_makes_a_smaller_bet_and_enough_noise_makes_none():
    gross = {}
    for noise in (0.01, 0.03, 0.2):
        z, r = world(beta=0.004, noise=noise, seed=6, persist=0.9)
        w, log = sz.apply(z, r, None, cfg(max_weight=1.0, gross=100.0))
        gross[noise] = (float(w.abs().sum(axis=1).iloc[400:].mean()), log["mean_shrink"])
    assert gross[0.01][0] > gross[0.03][0] > gross[0.2][0], gross
    assert gross[0.2][0] == pytest.approx(0.0, abs=1e-12) or gross[0.2][0] < 0.02 * gross[0.01][0], "with a calibration that cannot be told from zero the book is (almost) empty"


def test_the_limits_hold_and_the_gross_is_never_scaled_up_to_a_target():
    z, r = world(beta=0.004, noise=0.01, seed=7, persist=0.9)
    w, _ = sz.apply(z, r, None, cfg(max_weight=0.1, gross=0.8, net=0.0))
    live = w[w.abs().sum(axis=1) > 0]
    assert live.abs().max().max() <= 0.1 + 1e-12 and live.abs().sum(axis=1).max() <= 0.8 + 1e-12 and live.sum(axis=1).abs().max() < 1e-9 and len(live) > 250, "caps, gross and net hold together on every date"
    small, _ = sz.apply(z, r, None, cfg(risk_aversion=100.0, max_weight=0.35, gross=5.0))
    big, _ = sz.apply(z, r, None, cfg(risk_aversion=1.0, max_weight=0.35, gross=5.0))
    assert small.abs().sum(axis=1).mean() < big.abs().sum(axis=1).mean(), "more risk aversion, a smaller book: the gross is a result, not an input"
    tilted, _ = sz.apply(z, r, None, cfg(net=0.2, max_weight=1.0, gross=100.0))
    assert tilted[tilted.abs().sum(axis=1) > 0].sum(axis=1).sub(0.2).abs().max() < 1e-9, "the configured net exposure"


def test_nothing_on_a_date_depends_on_anything_after_it():
    z, r = world(beta=0.004, noise=0.01, seed=8, persist=0.9)
    w, _ = sz.apply(z, r, None, cfg())
    z2, r2 = z.copy(), r.copy()
    rng = np.random.default_rng(1)
    r2.iloc[550:] = rng.normal(0, 0.05, r2.iloc[550:].shape)
    z2.iloc[550:] = rng.normal(size=z2.iloc[550:].shape)
    w2, _ = sz.apply(z2, r2, None, cfg())
    assert np.allclose(w.iloc[:550].to_numpy(), w2.iloc[:550].to_numpy()), "the book up to a date is the same whatever the future holds"
    assert not np.allclose(w.iloc[560:].to_numpy(), w2.iloc[560:].to_numpy())


def test_the_signal_stability_is_the_rank_autocorrelation_of_consecutive_scores():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=120)
    steady = pd.DataFrame(np.tile(rng.normal(size=K), (120, 1)) + rng.normal(0, 0.05, (120, K)), index=idx, columns=COLS)
    shuffled = pd.DataFrame(rng.normal(size=(120, K)), index=idx, columns=COLS)
    assert sz.rank_stability(steady, 21).iloc[-1] > 0.9 and sz.rank_stability(shuffled, 21).iloc[-1] < 0.25
    ranks = shuffled.rank(axis=1).to_numpy()
    ref = np.mean([np.corrcoef(ranks[i], ranks[i - 1])[0, 1] for i in range(120 - 21, 120)])      # the mean of the last 21 consecutive-date rank correlations, from the definition
    assert sz.rank_stability(shuffled, 21).iloc[-1] == pytest.approx(min(max(ref, 0.0), 1.0), abs=1e-12)
    reversal = pd.DataFrame(np.tile(np.r_[np.arange(K), np.arange(K)[::-1]].reshape(2, K), (60, 1)), index=idx, columns=COLS)
    assert sz.rank_stability(reversal, 21).iloc[-1] == 0.0, "a signal that reverses every day is trusted at zero, not at minus one"


def test_projection_reaches_the_net_inside_the_limits_or_gets_as_close_as_it_can():
    w = np.array([0.5, 0.2, 0.1, -0.1, -0.1, -0.1])
    out = sz.project(w, 0.0, 0.3)
    assert abs(out.sum()) < 1e-12 and np.abs(out).max() <= 0.3 + 1e-12 and out[0] == pytest.approx(0.3), "the name above its limit sits at it and the rest absorb the net"
    far = sz.project(np.zeros(4), 2.0, 0.1)
    assert far.sum() == pytest.approx(0.4) and np.all(far == 0.1), "a net that the limits cannot reach: the closest reachable book"


def test_the_configuration_and_the_contract_for_calibrated_sizing():
    c = mc.experiment_config({"signal": {"method": "calibrated_sizing"}, "dataset": {"horizon": 10, "target": "fwd_ret_10"}}, defaults=mc.DEFAULTS)
    assert c["signal"]["window_days"] == 10 and c["signal"]["label_offset"] == 11 and set(sg.DERIVED_KEYS) <= set(c["signal"])
    for bad, msg in (({"calib_window": 10}, "calib_window"), ({"calib_min_t": -1.0}, "calib_min_t"), ({"stability_window": 2}, "stability_window")):
        with pytest.raises(mc.MLConfigError, match=msg):
            mc.experiment_config({"signal": {"method": "calibrated_sizing", **bad}}, defaults=mc.DEFAULTS)
    sigcfg = {"method": "calibrated_sizing"}
    assert ct.check_signal_use("ridge", "regression", "fwd_ret_5", sigcfg) == "expected_return"
    assert ct.check_signal_use("lgbm_ranker", "regression", "fwd_ret_5", sigcfg) == "ranking_score", "a ranker's score is exactly what calibration is for"
    with pytest.raises(ct.OutputContractError, match="calibrates a score"):
        ct.check_signal_use("logistic", "classification", "direction_5", sigcfg)
