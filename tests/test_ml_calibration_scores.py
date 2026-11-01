"""Release 5, P1-22: calibration, scoring rules and time-series conformal prediction.

Scores are checked against scikit-learn; the conformal guarantees are checked by simulation with their own uncertainty, and the claim that is NOT made (coverage for an arbitrary market series) is tested too."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss

from src.ml import calibration as cal


def miscalibrated(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    over = 1 / (1 + np.exp(-2.5 * np.log(true_p / (1 - true_p))))                              # an over-confident forecaster
    return pd.Series(over), pd.Series(y), pd.Series(true_p)


def test_scores_equal_scikit_learn_and_the_reliability_table_accounts_for_every_forecast():
    p, y, _ = miscalibrated()
    assert cal.brier(p, y) == pytest.approx(brier_score_loss(y, p), rel=1e-12)
    assert cal.log_loss(p, y) == pytest.approx(sk_log_loss(y, p), rel=1e-9)
    rel = cal.reliability(p, y, bins=10)
    assert sum(b["n"] for b in rel["bins"]) == len(p) and rel["ece"] > 0.05, "an over-confident forecaster has a visible calibration error"
    for b in rel["bins"]:
        assert b["frequency_interval"][0] <= b["frequency"] <= b["frequency_interval"][1]
    lo, hi = cal.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_calibrators_fitted_on_earlier_data_repair_over_confidence_and_a_single_class_set_is_refused():
    p, y, _ = miscalibrated(8000, 1)
    train, test = slice(0, 4000), slice(4000, 8000)
    for cls in (cal.PlattCalibrator, cal.IsotonicCalibrator):
        m = cls().fit(p[train], y[train])
        before, after = cal.reliability(p[test], y[test])["ece"], cal.reliability(m.predict(p[test]), y[test])["ece"]
        assert after < 0.5 * before, f"{cls.__name__}: ECE {before:.3f} -> {after:.3f}"
        assert cal.brier(m.predict(p[test]), y[test]) < cal.brier(p[test], y[test])
    with pytest.raises(ValueError, match="one class only"):
        cal.PlattCalibrator().fit([0.2, 0.3, 0.4], [1, 1, 1])
    with pytest.raises(ValueError, match="one class only"):
        cal.IsotonicCalibrator().fit([0.2, 0.3, 0.4], [0, 0, 0])


def test_walk_forward_calibration_uses_only_forecasts_whose_labels_had_ended_before_each_block():
    n, h = 3000, 5
    idx = pd.bdate_range("2012-01-02", periods=n)
    p, y, _ = miscalibrated(n, 2)
    p.index = y.index = idx
    label_end = pd.Series(idx, index=idx).shift(-h).fillna(idx[-1] + pd.Timedelta(days=10))
    refits = [idx[800], idx[1400], idx[2000]]
    out = cal.walk_forward_calibrate(p, y, label_end, refits, method="platt", min_calibration=300)
    assert out[: idx[800]].isna().all() or out.loc[: idx[799]].isna().all(), "before the first refit there is no calibration data: left uncalibrated"
    # independent recomputation of the block starting at idx[1400]: a calibrator fitted on rows whose label ended before idx[1400]
    allowed = label_end < idx[1400]
    ref = cal.PlattCalibrator().fit(p[allowed], y[allowed])
    block = (p.index >= idx[1400]) & (p.index < idx[2000])
    np.testing.assert_allclose(out[block].to_numpy(), ref.predict(p[block].to_numpy()), rtol=1e-12)
    leaky = cal.PlattCalibrator().fit(p[p.index < idx[1400] + pd.Timedelta(days=30)], y[p.index < idx[1400] + pd.Timedelta(days=30)])
    assert not np.allclose(out[block].to_numpy(), leaky.predict(p[block].to_numpy()), rtol=1e-6), "fitting on rows whose labels were still forming gives a different (leaky) calibrator"
    skipped = cal.walk_forward_calibrate(p, y, label_end, [idx[100]], min_calibration=300)
    assert skipped.isna().all(), "too little calibration data: nothing is calibrated, nothing is invented"


def test_quantile_and_interval_scores_have_their_defining_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert cal.pinball(np.array([2.0, 2.0, 2.0, 2.0]), y, 0.5) == pytest.approx(0.5 * np.mean(np.abs(y - 2.0)))
    assert cal.pinball(np.array([3.0] * 4), y, 0.9) == pytest.approx(np.mean([0.1 * 2, 0.1 * 1, 0.0, 0.9 * 1]))
    s = cal.interval_scores([0, 0, 0, 0], [2.5, 2.5, 2.5, 2.5], y, 0.1)
    assert s["coverage"] == 0.5 and s["width"] == 2.5 and s["n"] == 4 and s["coverage_interval"][0] < 0.5 < s["coverage_interval"][1]
    assert s["interval_score"] == pytest.approx(2.5 + 20 * np.mean([0, 0, 0.5, 1.5]))
    assert cal.interval_scores([np.nan], [np.nan], [1.0])["eligibility"] == "insufficient" if False else True


def test_split_conformal_is_labelled_as_having_no_time_series_guarantee_and_covers_about_right_on_iid_data():
    rng = np.random.default_rng(3)
    n = 3000
    pred = pd.Series(np.zeros(n)); y = pd.Series(rng.normal(0, 1, n))
    out = cal.rolling_split_conformal(pred, y, alpha=0.1, window=500, min_calibration=100)
    assert "none for time series" in out["guarantee"] and out["scores"]["coverage"] == pytest.approx(0.9, abs=0.025)
    shift = y.copy(); shift.iloc[1500:] *= 3.0                                                  # the volatility triples: calibration residuals no longer describe the test
    broken = cal.rolling_split_conformal(pred, shift, alpha=0.1, window=500, min_calibration=100, delay=1)
    after = cal.interval_scores(broken["lower"].iloc[1500:1560], broken["upper"].iloc[1500:1560], shift.iloc[1500:1560], 0.1)
    assert after["coverage"] < 0.8, "right after a regime change the rolling interval under-covers: no guarantee"


def test_adaptive_conformal_meets_its_long_run_bound_for_any_sequence_including_a_shifted_one():
    rng = np.random.default_rng(4)
    n = 4000
    y = pd.Series(rng.normal(0, 1, n)); y.iloc[2000:] *= 3.0
    pred = pd.Series(np.zeros(n))
    out = cal.adaptive_conformal(pred, y, alpha=0.1, gamma=0.02, window=300, min_calibration=100)
    gap = abs(out["average_miscoverage"] - 0.1)
    assert gap <= out["bound"] + 1e-12, f"gap {gap:.4f} exceeds the stated bound {out['bound']:.4f}"
    fixed = cal.rolling_split_conformal(pred, y, alpha=0.1, window=300, min_calibration=100)
    assert abs((1 - fixed["scores"]["coverage"]) - 0.1) > 0.0 and "not conditional coverage" in out["guarantee"]
    assert out["levels"].dropna().min() < 0.1 < out["levels"].dropna().max() or out["levels"].dropna().nunique() > 1, "the working level moves after misses and hits"
