"""Probabilistic forecasting: scoring rules, calibration, features, walk-forward causality, sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats
from scipy.special import expit, logit

from src.features.sleeves import month_end_dates
from src.models.probabilistic import (
    PlattCalibrator,
    auc,
    brier_terms,
    build_panel,
    calibration_slope_intercept,
    crps_gaussian,
    edge_book,
    expected_calibration_error,
    kelly_book,
    log_loss_terms,
    log_score_gaussian,
    paired_ece_test,
    pit_gaussian,
    pit_uniformity,
    price_features,
    reliability_table,
    rsi,
    scores_by_origin,
    walk_forward_probabilistic,
)


# ----------------------------------------------------------------- proper scoring rules
def test_gaussian_crps_matches_its_known_value_and_the_defining_integral():
    assert float(crps_gaussian(0.0, 0.0, 1.0)) == pytest.approx(2 * stats.norm.pdf(0) - 1 / np.sqrt(np.pi), abs=1e-12)
    mu, sigma, y = 0.3, 1.7, -0.9
    integral, _ = integrate.quad(lambda x: (stats.norm.cdf(x, mu, sigma) - (x >= y)) ** 2, -40, 40, points=[y], limit=200)
    assert float(crps_gaussian(y, mu, sigma)) == pytest.approx(integral, abs=1e-8)


def test_crps_tends_to_the_absolute_error_as_the_forecast_sharpens():
    assert float(crps_gaussian(1.5, 0.5, 1e-9)) == pytest.approx(1.0, abs=1e-6)


def test_log_score_is_minus_the_log_density_and_pit_is_the_cdf():
    y, mu, sigma = 0.4, -0.1, 0.8
    assert float(log_score_gaussian(y, mu, sigma)) == pytest.approx(-stats.norm.logpdf(y, mu, sigma), abs=1e-12)
    assert float(pit_gaussian(y, mu, sigma)) == pytest.approx(stats.norm.cdf(y, mu, sigma), abs=1e-12)


def test_brier_and_log_loss_by_hand():
    p, y = np.array([0.8, 0.3]), np.array([1, 0])
    assert brier_terms(p, y).tolist() == pytest.approx([0.04, 0.09])
    assert log_loss_terms(p, y).tolist() == pytest.approx([-np.log(0.8), -np.log(0.7)])


def test_both_rules_are_proper_the_true_probability_scores_best():
    rng = np.random.default_rng(0)
    y = (rng.random(200000) < 0.7).astype(int)
    for scorer in (log_loss_terms, brier_terms):
        honest = scorer(np.full(len(y), 0.7), y).mean()
        assert honest < scorer(np.full(len(y), 0.5), y).mean()
        assert honest < scorer(np.full(len(y), 0.9), y).mean()


def test_crps_is_proper_the_true_distribution_scores_best():
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 100000)
    truth = crps_gaussian(y, 0.0, 1.0).mean()
    assert truth < crps_gaussian(y, 0.0, 0.5).mean() and truth < crps_gaussian(y, 0.0, 2.0).mean()
    assert truth < crps_gaussian(y, 0.5, 1.0).mean()


def test_pit_is_uniform_for_the_true_forecast_and_piles_in_the_tails_when_overconfident():
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 1.0, 5000)
    assert pit_uniformity(pit_gaussian(y, 0.0, 1.0))["ks_p_value"] > 0.01
    bold = pit_uniformity(pit_gaussian(y, 0.0, 0.5))
    assert bold["share_below_10pct"] > 0.15 and bold["share_above_90pct"] > 0.15 and bold["ks_p_value"] < 1e-6


# --------------------------------------------------------------------------- calibration
def _outcomes(p: np.ndarray, seed: int = 3) -> np.ndarray:
    return (np.random.default_rng(seed).random(len(p)) < p).astype(int)


def test_reliability_table_counts_everything_and_ece_is_small_for_calibrated_forecasts():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, 30000)
    y = _outcomes(p)
    table = reliability_table(p, y, 10)
    assert table["count"].sum() == len(p)
    assert expected_calibration_error(p, y, 10) < 0.02


def test_overconfidence_shows_as_a_large_ece_and_a_calibration_slope_below_one():
    rng = np.random.default_rng(5)
    true_p = rng.uniform(0.2, 0.8, 30000)
    y = _outcomes(true_p)
    bold = expit(2.0 * logit(true_p))                                       # twice as extreme as the truth
    assert expected_calibration_error(bold, y, 10) > 0.05
    assert expected_calibration_error(true_p, y, 10) < 0.02
    assert calibration_slope_intercept(bold, y)["slope"] == pytest.approx(0.5, abs=0.06)
    assert calibration_slope_intercept(true_p, y)["slope"] == pytest.approx(1.0, abs=0.06)


def test_platt_scaling_undoes_a_known_distortion():
    rng = np.random.default_rng(6)
    true_p = rng.uniform(0.15, 0.85, 40000)
    y = _outcomes(true_p)
    raw = expit(2.0 * logit(true_p) + 0.4)
    platt = PlattCalibrator().fit(raw, y)
    assert platt.a == pytest.approx(0.5, abs=0.05) and platt.b == pytest.approx(-0.2, abs=0.05)
    assert expected_calibration_error(platt.transform(raw), y) < 0.02


def test_auc_by_hand():
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    rng = np.random.default_rng(7)
    assert auc(rng.random(20000), (rng.random(20000) < 0.5).astype(int)) == pytest.approx(0.5, abs=0.02)


def test_paired_ece_test_sees_that_calibration_helped():
    rng = np.random.default_rng(8)
    n_origins, per = 80, 12
    true_p = rng.uniform(0.2, 0.8, n_origins * per)
    y = _outcomes(true_p, seed=9)
    raw = expit(2.0 * logit(true_p))
    calibrated = PlattCalibrator().fit(raw, y).transform(raw)
    pred = pd.DataFrame({"origin": np.repeat(pd.bdate_range("2012-01-31", periods=n_origins, freq="21B"), per),
                         "up": y, "p_raw": raw, "p_cal": calibrated})
    result = paired_ece_test(pred, n_samples=300, block_length=3, seed=1)
    assert result["difference"] < 0 and result["p_value_not_improved"] < 0.05


# -------------------------------------------------------------------------------- features
def _prices(n=1600, k=4, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2008-01-02", periods=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, k)), axis=0)), index=index,
                        columns=[f"A{i}" for i in range(k)])


def test_price_features_use_only_data_up_to_the_day():
    prices = _prices()
    full = price_features(prices)
    cut = 900
    altered = prices.copy()
    altered.iloc[cut + 1:] *= 3.0
    changed = price_features(altered)
    for name, frame in full.items():
        pd.testing.assert_frame_equal(frame.iloc[:cut + 1], changed[name].iloc[:cut + 1])
    assert set(full) >= {"mom_21", "mom_63", "mom_126", "mom_252", "zscore_5", "zscore_21", "zscore_63",
                         "vol_21", "vol_63", "vol_ratio", "drawdown_252", "rsi_14"}


def test_momentum_and_drawdown_by_hand_and_rsi_is_bounded():
    index = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"X": np.linspace(100, 200, 300)}, index=index)
    features = price_features(prices)
    assert features["mom_21"]["X"].iloc[100] == pytest.approx(prices["X"].iloc[100] / prices["X"].iloc[79] - 1)
    assert features["drawdown_252"]["X"].iloc[-1] == pytest.approx(0.0)
    values = rsi(_prices()).to_numpy().ravel()
    values = values[np.isfinite(values)]
    assert len(values) > 1000 and ((values >= 0.0) & (values <= 1.0)).all()
    assert rsi(prices)["X"].iloc[-1] == pytest.approx(1.0)                 # only gains: strength is maximal


# --------------------------------------------------------------- walk-forward causality
def _panel(prices: pd.DataFrame):
    origins = month_end_dates(prices.index)[13:]
    common = pd.DataFrame({"macro_a": np.sin(np.arange(len(prices)) / 200.0)}, index=prices.index)
    sleeve_of = {c: ("equity" if i < 2 else "bonds") for i, c in enumerate(prices.columns)}
    return build_panel(prices, origins, 21, 40.0, common, sleeve_of)


SETTINGS = dict(min_train=756, refit_every=252, horizon=21, embargo=21)


@pytest.mark.parametrize("variant", ["price_only", "price_macro"])
def test_predictions_do_not_depend_on_prices_after_the_origin(variant):
    prices = _prices(seed=1)
    base = walk_forward_probabilistic(_panel(prices), variant, **SETTINGS)
    cut = 1250
    altered = prices.copy()
    altered.iloc[cut + 1:] = altered.iloc[cut + 1:] * np.random.default_rng(4).lognormal(0, 0.3, altered.iloc[cut + 1:].shape)
    changed = walk_forward_probabilistic(_panel(altered), variant, **SETTINGS)
    index = pd.DatetimeIndex(prices.index)
    early = base[base["origin"] <= index[cut]].sort_values(["origin", "asset"]).reset_index(drop=True)
    early_changed = changed[changed["origin"] <= index[cut]].sort_values(["origin", "asset"]).reset_index(drop=True)
    assert len(early) > 20
    for column in ("p_raw", "p_cal", "mu", "sigma", "base_rate", "mu_benchmark"):
        np.testing.assert_allclose(early[column], early_changed[column], atol=1e-12)
    assert len(changed) > len(early)                                        # the later origins exist and may differ


def test_every_training_label_is_realised_before_the_refit():
    prices = _prices(seed=2)
    panel = _panel(prices)
    predictions = walk_forward_probabilistic(panel, "price_only", **SETTINGS)
    assert (predictions["refit_day"] >= SETTINGS["min_train"]).all()
    positions = predictions["origin"].map(panel.positions)
    assert (positions >= predictions["refit_day"]).all()
    assert (positions < predictions["refit_day"] + SETTINGS["refit_every"]).all()
    assert predictions["p_cal"].between(0, 1).all() and predictions["base_rate"].between(0, 1).all()


def test_scores_by_origin_has_one_row_per_month_and_the_model_matches_the_benchmark_when_it_is_the_benchmark():
    prices = _prices(seed=3)
    predictions = walk_forward_probabilistic(_panel(prices), "price_only", **SETTINGS)
    predictions = predictions.assign(p_cal=predictions["base_rate"], p_raw=predictions["base_rate"],
                                     mu=predictions["mu_benchmark"])
    scores = scores_by_origin(predictions)
    assert scores.index.is_unique
    np.testing.assert_allclose(scores["logloss_model"], scores["logloss_benchmark"])
    np.testing.assert_allclose(scores["crps_model"], scores["crps_benchmark"])


# ------------------------------------------------------------------------------- sizing
def _origin_frame():
    return pd.DataFrame({"p_cal": [0.70, 0.40, 0.52, 0.55, 0.30], "base_rate": [0.60, 0.60, 0.50, 0.50, 0.50],
                         "sigma": [0.04, 0.02, 0.05, 0.03, 0.08], "mu": [0.01, -0.01, 0.002, 0.004, -0.02]},
                        index=list("ABCDE"))


def test_edge_books_drop_small_edges_scale_to_gross_one_and_differ_only_in_sizing():
    frame = _origin_frame()
    direction = edge_book(frame, "direction_only", 0.03, 1.0)
    sized = edge_book(frame, "probability_sized", 0.03, 1.0)
    assert direction["C"] == 0.0 and sized["C"] == 0.0                   # |edge| 0.02 < 0.03: no trade
    assert direction["A"] > 0 > direction["B"] and np.sign(direction).equals(np.sign(sized))
    assert direction.abs().sum() == pytest.approx(1.0) and sized.abs().sum() == pytest.approx(1.0)
    assert abs(sized["E"]) > abs(direction["E"]) * 0.5                    # a 20-point edge gets more than a 10-point one relative to direction
    capped = edge_book(frame, "direction_only", 0.03, 0.30)
    assert capped.abs().max() <= 0.30 + 1e-12


def test_kelly_book_is_proportional_to_mu_over_variance_and_respects_caps():
    frame = _origin_frame()
    w = kelly_book(frame, kelly_fraction=0.25, max_weight=10.0, max_gross=10.0)
    expected = 0.25 * frame["mu"] / frame["sigma"] ** 2
    np.testing.assert_allclose(w, expected)
    capped = kelly_book(frame, 0.25, max_weight=0.20, max_gross=0.5)
    assert capped.abs().max() <= 0.20 + 1e-12 and capped.abs().sum() <= 0.5 + 1e-12
