"""The covariance estimators every optimiser-based book rests on, checked against independent calculations.

A covariance matrix is an input to mean-variance, risk parity, hierarchical risk parity, the volatility target and every risk figure on the page, so a slip in it is in all of them and shows in none. Each estimator
is recomputed here by the plainest means (loops over dates, scikit-learn's Ledoit-Wolf for the identity target) on data small enough to work through, and the cases that real data produces and a textbook does not
(a missing day, an asset that never moved, a matrix that is not positive definite) are stated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from src.features.pca import marchenko_pastur_bounds
from src.portfolio.covariance import (condition_number, constant_correlation_target, estimate_covariance, evaluate_covariance_forecasts, ewma_covariance, is_positive_definite, ledoit_wolf_shrinkage,
                                      nearest_positive_definite, pca_denoised_covariance, portfolio_volatility, sample_covariance)


def returns_of(days: int = 120, assets: int = 4, seed: int = 0, factor: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.01, (days, 1))
    return pd.DataFrame(factor * common * np.linspace(0.6, 1.4, assets) + rng.normal(0.0, 0.01, (days, assets)), index=pd.bdate_range("2021-01-04", periods=days), columns=list("ABCDEFGH")[:assets])


# ------------------------------------------------------------------------------------------------------------------ the point estimators
def test_the_sample_covariance_is_the_unbiased_one_a_year_long_and_drops_a_day_with_a_missing_return():
    x = returns_of(40, 3)
    assert sample_covariance(x).to_numpy() == pytest.approx(np.cov(x.to_numpy().T, ddof=1) * 252.0)
    assert sample_covariance(x, annualise=False).to_numpy() == pytest.approx(np.cov(x.to_numpy().T, ddof=1))
    gappy = x.copy()
    gappy.iloc[5, 1] = np.nan
    gappy.iloc[9, 0] = np.nan
    kept = x.drop(x.index[[5, 9]])
    assert sample_covariance(gappy, annualise=False).to_numpy() == pytest.approx(np.cov(kept.to_numpy().T, ddof=1)), "a day on which any asset has no return is left out whole"
    assert list(sample_covariance(x).index) == list("ABC") and list(sample_covariance(x).columns) == list("ABC")


def ewma_by_hand(x: np.ndarray, halflife: float) -> np.ndarray:
    n = len(x)
    w = np.array([0.5 ** ((n - 1 - t) / halflife) for t in range(n)])
    w = w / w.sum()
    centred = x - (w[:, None] * x).sum(axis=0)
    cov = sum(w[t] * np.outer(centred[t], centred[t]) for t in range(n))
    return cov / (1.0 - (w ** 2).sum())


def test_the_exponentially_weighted_covariance_halves_the_weight_every_halflife_and_corrects_the_bias_of_the_weighted_mean():
    x = returns_of(80, 3, seed=1)
    for halflife in (5.0, 40.0):
        assert ewma_covariance(x, halflife=halflife, annualise=False).to_numpy() == pytest.approx(ewma_by_hand(x.to_numpy(), halflife))
    assert ewma_covariance(x, annualise=False).to_numpy() == pytest.approx(ewma_by_hand(x.to_numpy(), 60.0)) and ewma_covariance(x).to_numpy() == pytest.approx(ewma_by_hand(x.to_numpy(), 60.0) * 252.0)
    flat = ewma_covariance(x, halflife=1e9, annualise=False).to_numpy()
    assert flat == pytest.approx(np.cov(x.to_numpy().T, ddof=1)), "with a very long halflife every day counts the same and it is the sample covariance"
    with pytest.raises(ValueError, match="need at least 60 complete observations"):
        ewma_covariance(x.iloc[:59])
    assert ewma_covariance(x.iloc[:59], min_periods=59).shape == (3, 3)
    gappy = x.copy()
    gappy.iloc[10, 2] = np.nan
    assert ewma_covariance(gappy, halflife=20.0, annualise=False).to_numpy() == pytest.approx(ewma_by_hand(x.drop(x.index[10]).to_numpy(), 20.0))


# ------------------------------------------------------------------------------------------------------------------ shrinkage
def test_the_constant_correlation_target_keeps_every_variance_and_gives_every_pair_the_average_correlation():
    x = returns_of(150, 4, seed=2)
    target = constant_correlation_target(x).to_numpy()
    s = np.cov(x.to_numpy().T, ddof=1)
    vol = np.sqrt(np.diag(s))
    corr = s / np.outer(vol, vol)
    average = corr[~np.eye(4, dtype=bool)].mean()
    expected = average * np.outer(vol, vol)
    np.fill_diagonal(expected, np.diag(s))
    assert target == pytest.approx(expected) and np.allclose(target, target.T) and not np.allclose(target, s)
    assert list(constant_correlation_target(x).columns) == list("ABCD")


def test_an_asset_that_never_moved_has_a_variance_of_zero_and_no_correlation_instead_of_breaking_every_estimator():
    x = returns_of(150, 4, seed=2)
    x["D"] = 0.0
    target = constant_correlation_target(x).to_numpy()
    assert np.isfinite(target).all() and (target[3] == 0.0).all() and (target[:, 3] == 0.0).all(), "nothing is shrunk towards a correlation with an asset that has none"
    others = constant_correlation_target(x[list("ABC")]).to_numpy()
    assert target[:3, :3] == pytest.approx(others), "and the average correlation is that of the assets that moved"
    for method in ("sample", "shrinkage", "pca_denoised", "ewma"):
        cov = estimate_covariance(x, method).to_numpy()
        assert np.isfinite(cov).all() and (cov[3] == 0.0).all() and (cov[:, 3] == 0.0).all() and (np.diag(cov)[:3] > 0.0).all(), method
    only = x.copy()
    only[["A", "B", "C"]] = 0.0
    assert (constant_correlation_target(only).to_numpy() == 0.0).all() and np.isfinite(ledoit_wolf_shrinkage(returns_of(60, 3).assign(B=0.0).assign(C=0.0))[0].to_numpy()).all(), "and with no pair that moved, no correlation"


def test_the_identity_target_is_scikit_learns_ledoit_wolf_and_the_intensity_can_be_given_or_is_clipped():
    x = returns_of(120, 5, seed=3)
    shrunk, delta = ledoit_wolf_shrinkage(x, target="identity")
    reference = LedoitWolf().fit(x.to_numpy())
    assert shrunk.to_numpy() == pytest.approx(reference.covariance_ * 252.0) and delta == pytest.approx(reference.shrinkage_)
    assert ledoit_wolf_shrinkage(x, target="identity", annualise=False)[0].to_numpy() == pytest.approx(reference.covariance_)
    sample = np.cov(x.to_numpy().T, ddof=1)
    target = constant_correlation_target(x).to_numpy()
    for given in (0.0, 0.3, 1.0):
        mixed, used = ledoit_wolf_shrinkage(x, delta=given, annualise=False)
        assert used == given and mixed.to_numpy() == pytest.approx(given * target + (1.0 - given) * sample)
    assert ledoit_wolf_shrinkage(x, delta=1.5)[1] == 1.0 and ledoit_wolf_shrinkage(x, delta=-0.5)[1] == 0.0, "an intensity is between none and all"
    diagonal, _ = ledoit_wolf_shrinkage(x, target="diagonal", delta=1.0, annualise=False)
    assert diagonal.to_numpy() == pytest.approx(np.diag(np.diag(sample))), "all the way to the diagonal: the variances and nothing else"
    half, _ = ledoit_wolf_shrinkage(x, target="diagonal", delta=0.5, annualise=False)
    assert half.to_numpy() == pytest.approx(0.5 * np.diag(np.diag(sample)) + 0.5 * sample)
    with pytest.raises(ValueError, match="unknown shrinkage target 'oracle'"):
        ledoit_wolf_shrinkage(x, target="oracle")
    with pytest.raises(ValueError, match="not enough complete observations"):
        ledoit_wolf_shrinkage(x.iloc[:29])
    assert ledoit_wolf_shrinkage(x.iloc[:30])[0].shape == (5, 5)
    gappy = x.copy()
    gappy.iloc[3, 0] = np.nan
    assert ledoit_wolf_shrinkage(gappy, delta=0.2)[0].to_numpy() == pytest.approx(ledoit_wolf_shrinkage(x.drop(x.index[3]), delta=0.2)[0].to_numpy())


def test_the_analytic_intensity_is_the_estimation_error_of_the_sample_matrix_net_of_its_diagonal_over_its_distance_from_the_target():
    x = returns_of(90, 4, seed=4)
    a = x.to_numpy()
    n, p = a.shape
    s = np.cov(a.T, ddof=1)
    target = constant_correlation_target(x).to_numpy()
    c = a - a.mean(axis=0)
    pi = np.array([[np.mean((c[:, i] * c[:, j] - s[i, j]) ** 2) for j in range(p)] for i in range(p)])        # the variance of each entry of the sample covariance matrix
    rho = np.trace(pi)                                                                                     # the diagonal terms only, as the implementation says
    gamma = ((target - s) ** 2).sum()
    expected = float(np.clip((pi.sum() - rho) / gamma / n, 0.0, 1.0))
    delta = ledoit_wolf_shrinkage(x)[1]
    assert 0.0 < expected < 1.0 and delta == pytest.approx(expected)
    shrunk = ledoit_wolf_shrinkage(x, annualise=False)[0].to_numpy()
    assert shrunk == pytest.approx(delta * target + (1.0 - delta) * s)
    two = returns_of(90, 2, seed=4)
    both, full = ledoit_wolf_shrinkage(two)
    assert full == 1.0 and both.to_numpy() == pytest.approx(np.cov(two.to_numpy().T, ddof=1) * 252.0), "with two assets the target is the sample matrix: nothing to shrink, and the intensity says all of it"
    longer = returns_of(900, 4, seed=4)
    assert ledoit_wolf_shrinkage(longer)[1] < delta, "more days, less shrinkage"


# ------------------------------------------------------------------------------------------------------------------ denoising and repair
def denoised_by_hand(x: pd.DataFrame, n_components: int | None) -> np.ndarray:
    s = np.cov(x.to_numpy().T, ddof=1)
    vol = np.sqrt(np.diag(s))
    values, vectors = np.linalg.eigh(s / np.outer(vol, vol))
    values, vectors = values[::-1], vectors[:, ::-1]
    if n_components is None:
        n_components = max(int((values > marchenko_pastur_bounds(len(x), x.shape[1])[1]).sum()), 1)
    kept = values.copy()
    kept[n_components:] = values[n_components:].mean()
    rebuilt = vectors @ np.diag(kept) @ vectors.T
    rebuilt = rebuilt / np.sqrt(np.outer(np.diag(rebuilt), np.diag(rebuilt)))
    return rebuilt * np.outer(vol, vol)


def test_denoising_averages_the_eigenvalues_that_do_not_stand_out_of_the_noise_and_the_volatilities_are_unchanged():
    rng = np.random.default_rng(5)
    days, assets = 400, 6
    common = rng.normal(0.0, 0.01, (days, 1))
    x = pd.DataFrame(common * np.array([1.0, 1.2, 0.8, 1.0, 1.1, 0.9]) + rng.normal(0.0, 0.008, (days, assets)), columns=list("ABCDEF"))
    s = np.cov(x.to_numpy().T, ddof=1)
    vol = np.sqrt(np.diag(s))
    values = np.sort(np.linalg.eigvalsh(s / np.outer(vol, vol)))[::-1]
    assert (values > marchenko_pastur_bounds(days, assets)[1]).sum() == 1, "one factor stands out of the noise"
    out = pca_denoised_covariance(x, annualise=False).to_numpy()
    assert out == pytest.approx(denoised_by_hand(x, None)) and np.diag(out) == pytest.approx(np.diag(s)), "every asset keeps its own variance"
    assert pca_denoised_covariance(x).to_numpy() == pytest.approx(out * 252.0)
    assert pca_denoised_covariance(x, n_components=2, annualise=False).to_numpy() == pytest.approx(denoised_by_hand(x, 2)) and not np.allclose(denoised_by_hand(x, 2), out)
    assert pca_denoised_covariance(x, n_components=assets, annualise=False).to_numpy() == pytest.approx(s), "all of them kept is the sample matrix"
    assert pca_denoised_covariance(x, n_components=assets + 3, annualise=False).to_numpy() == pytest.approx(s), "more than there are is all of them"
    corr = out / np.outer(vol, vol)
    assert np.diag(corr) == pytest.approx(np.ones(assets)) and is_positive_definite(corr) and np.allclose(out, out.T)
    off = ~np.eye(assets, dtype=bool)
    assert np.abs(corr[off] - (s / np.outer(vol, vol))[off]).max() > 0.01, "it is not the sample matrix"
    noise = returns_of(400, 6, seed=6, factor=0.0)
    assert pca_denoised_covariance(noise, annualise=False).to_numpy() == pytest.approx(denoised_by_hand(noise, None)), "pure noise still keeps one eigenvalue"


def test_a_matrix_is_positive_definite_when_its_smallest_eigenvalue_is_above_the_tolerance():
    assert is_positive_definite(np.eye(3)) and is_positive_definite(pd.DataFrame([[2.0, 0.5], [0.5, 1.0]]))
    assert not is_positive_definite(np.array([[1.0, 1.0], [1.0, 1.0]])) and not is_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert not is_positive_definite(np.eye(2) * 0.5, tolerance=0.5) and is_positive_definite(np.eye(2) * 0.5, tolerance=0.4), "above the tolerance, not at it"
    assert condition_number(pd.DataFrame(np.diag([1.0, 4.0, 2.0]))) == pytest.approx(4.0) and condition_number(np.array([[1.0, 1.0], [1.0, 1.0]])) == float("inf")
    assert condition_number(np.diag([1.0, 1e-18])) == float("inf") and condition_number(np.diag([1.0, 1.2e-18])) == pytest.approx(1.0 / 1.2e-18), "a smallest eigenvalue of 1e-18 or less is none"


def test_a_matrix_with_a_negative_eigenvalue_is_repaired_by_clipping_and_keeps_its_variances():
    broken = pd.DataFrame([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]], columns=list("ABC"), index=list("ABC"))
    assert not is_positive_definite(broken)
    repaired = nearest_positive_definite(broken)
    assert is_positive_definite(repaired) and np.diag(repaired.to_numpy()) == pytest.approx(np.ones(3)) and np.allclose(repaired, repaired.T), "a real covariance matrix again, with the variances it started with"
    assert list(repaired.index) == list("ABC") and list(repaired.columns) == list("ABC")
    values, vectors = np.linalg.eigh(broken.to_numpy())
    clipped = vectors @ np.diag(np.clip(values, 1e-10, None)) @ vectors.T
    scale = np.sqrt(np.diag(broken.to_numpy()) / np.diag(clipped))
    assert repaired.to_numpy() == pytest.approx(clipped * np.outer(scale, scale)), "the eigenvalues below the floor lifted to it, then the variances restored"
    scaled = broken * np.array([4.0, 1.0, 9.0])[:, None] ** 0.5 * np.array([4.0, 1.0, 9.0])[None, :] ** 0.5
    assert np.diag(nearest_positive_definite(scaled).to_numpy()) == pytest.approx([4.0, 1.0, 9.0]), "whatever the scale of each asset"
    fine = pd.DataFrame([[1.0, 0.2], [0.4, 1.0]], columns=list("AB"), index=list("AB"))
    assert nearest_positive_definite(fine).to_numpy() == pytest.approx(np.array([[1.0, 0.3], [0.3, 1.0]])), "a matrix that is fine is only made symmetric"
    nearly = pd.DataFrame(np.diag([1.0, 1e-12]), columns=list("AB"), index=list("AB"))
    assert nearest_positive_definite(nearly, min_eigenvalue=1e-10).to_numpy()[1, 1] == pytest.approx(1e-12), "the variance of the thin asset is kept, though its eigenvalue is lifted first"


# ------------------------------------------------------------------------------------------------------------------ the entry point
def test_one_entry_point_chooses_the_estimator_the_window_and_repairs_the_result():
    x = returns_of(400, 4, seed=7)
    tail = x.tail(100)
    assert estimate_covariance(x, "sample", lookback=100).to_numpy() == pytest.approx(sample_covariance(tail).to_numpy())
    assert estimate_covariance(x, "rolling", lookback=100).to_numpy() == pytest.approx(sample_covariance(tail).to_numpy())
    assert estimate_covariance(x, "sample", lookback=0).to_numpy() == pytest.approx(sample_covariance(x).to_numpy()) and estimate_covariance(x, "sample", lookback=None).shape == (4, 4), "no lookback is every day"
    assert estimate_covariance(x, "sample").to_numpy() == pytest.approx(sample_covariance(x.tail(252)).to_numpy()), "a year by default"
    assert estimate_covariance(x, "ewma", lookback=100, halflife=30.0).to_numpy() == pytest.approx(ewma_covariance(x, 30.0).to_numpy()), "the decay is the window: the lookback does not apply"
    assert estimate_covariance(x, "ewma").to_numpy() == pytest.approx(ewma_covariance(x, 60.0).to_numpy()), "a halflife of sixty days by default"
    assert estimate_covariance(x, "shrinkage", lookback=100, delta=0.4).to_numpy() == pytest.approx(ledoit_wolf_shrinkage(tail, "constant_correlation", 0.4)[0].to_numpy())
    assert estimate_covariance(x, "shrinkage", lookback=100, target="diagonal", delta=0.4, annualise=False).to_numpy() == pytest.approx(ledoit_wolf_shrinkage(tail, "diagonal", 0.4, False)[0].to_numpy())
    assert estimate_covariance(x, "pca_denoised", lookback=100).to_numpy() == pytest.approx(pca_denoised_covariance(tail).to_numpy())
    assert estimate_covariance(x, "sample", lookback=100, annualise=False).to_numpy() == pytest.approx(sample_covariance(tail, False).to_numpy())
    with pytest.raises(ValueError, match="unknown covariance method 'garch'"):
        estimate_covariance(x, "garch")
    short = returns_of(10, 3, seed=8)
    singular = estimate_covariance(short, "sample")                                                        # ten days of three assets is fine; five assets is not
    assert is_positive_definite(singular)
    wide = returns_of(8, 8, seed=9)
    assert is_positive_definite(estimate_covariance(wide, "sample", min_eigenvalue=1e-8), tolerance=0.0) and not is_positive_definite(sample_covariance(wide), tolerance=1e-8), "eight days of eight assets: the sample matrix is singular, and the result is not"


def test_the_volatility_of_a_portfolio_is_the_square_root_of_its_variance_and_never_the_root_of_a_negative():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert portfolio_volatility([0.5, 0.5], cov) == pytest.approx(np.sqrt(0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.01))
    assert portfolio_volatility(pd.Series([1.0, 0.0]), pd.DataFrame(cov)) == pytest.approx(0.2) and portfolio_volatility(np.array([[1.0], [-1.0]]), cov) == pytest.approx(np.sqrt(0.04 + 0.09 - 0.02))
    assert portfolio_volatility([1.0, -1.0], np.array([[1.0, 2.0], [2.0, 1.0]])) == 0.0, "an indefinite matrix can give a negative variance (1 + 1 - 4), which is no volatility"
    assert portfolio_volatility([1.0, 1.0], np.array([[1.0, 2.0], [2.0, 1.0]])) == pytest.approx(np.sqrt(6.0))


# ------------------------------------------------------------------------------------------------------------------ scoring an estimator by the risk it forecast
def test_an_estimator_is_scored_by_the_portfolio_risk_it_forecast_against_the_risk_that_followed():
    x = returns_of(220, 3, seed=10)
    lookback, horizon, step = 60, 10, 10
    forecasts, realised, conditions = [], [], []
    w = np.full(3, 1.0 / 3.0)
    for end in range(lookback, len(x) - horizon, step):
        window, future = x.iloc[end - lookback:end], x.iloc[end:end + horizon]
        cov = np.cov(window.to_numpy().T, ddof=1) * 252.0
        forecasts.append(np.sqrt(w @ cov @ w))
        realised.append((future.to_numpy() @ w).std(ddof=1) * np.sqrt(252.0))
        conditions.append(np.linalg.cond(cov))
    f, r = np.array(forecasts), np.array(realised)
    ratio = r ** 2 / f ** 2
    table = evaluate_covariance_forecasts(x, methods=("sample",), lookback=lookback, horizon=horizon, step=step)
    row = table.loc["sample"]
    assert row["n_windows"] == len(range(lookback, len(x) - horizon, step)) == 15
    assert row["corr"] == pytest.approx(np.corrcoef(f, r)[0, 1]) and row["rmse"] == pytest.approx(np.sqrt(np.mean((f - r) ** 2))) and row["mean_bias"] == pytest.approx(np.mean(f - r))
    assert row["mean_ratio"] == pytest.approx(np.mean(f / r)) and row["qlike"] == pytest.approx(np.mean(ratio - np.log(ratio) - 1.0)) and row["forecast_stability"] == pytest.approx(np.std(np.diff(f)))
    assert row["mean_condition_number"] == pytest.approx(np.mean(conditions)) and np.isnan(row["mean_shrinkage_delta"]), "an intensity only for the estimator that has one"
    several = evaluate_covariance_forecasts(x, methods=("sample", "shrinkage"), lookback=lookback, horizon=horizon, step=step)
    assert set(several.index) == {"sample", "shrinkage"} and several["qlike"].is_monotonic_increasing and 0.0 < several.loc["shrinkage", "mean_shrinkage_delta"] <= 1.0, "best first"
    given = pd.DataFrame(np.tile([1.0, 0.0, 0.0], (len(x), 1)), index=x.index, columns=x.columns)
    one = evaluate_covariance_forecasts(x, methods=("sample",), lookback=lookback, horizon=horizon, step=step, weights=given).loc["sample"]
    assert one["mean_ratio"] != row["mean_ratio"] and one["n_windows"] == 15, "the book that was held, when there is one"
    nothing = evaluate_covariance_forecasts(x, methods=("sample",), lookback=lookback, horizon=horizon, step=step, weights=given * 0.0)
    assert nothing.empty, "a book that held nothing says nothing about the risk"
    assert evaluate_covariance_forecasts(x.iloc[:50], lookback=60).empty


def test_the_scoring_has_a_year_of_history_a_month_of_horizon_and_a_month_between_windows_unless_told_otherwise():
    x = returns_of(420, 3, seed=11)
    table = evaluate_covariance_forecasts(x, methods=("sample", "ewma"))
    assert table["n_windows"].tolist() == [7, 7] == [len(range(252, 420 - 21, 21))] * 2, "windows ending on the 253rd day and every 21st after it, each with 21 days to be judged on"
    equal = np.full(3, 1.0 / 3.0)
    ends = range(252, 420 - 21, 21)
    forecasts = np.array([np.sqrt(equal @ (np.cov(x.iloc[end - 252:end].to_numpy().T, ddof=1) * 252.0) @ equal) for end in ends])
    realised = np.array([(x.iloc[end:end + 21].to_numpy() @ equal).std(ddof=1) * np.sqrt(252.0) for end in ends])
    sample = table.loc["sample"]
    assert sample["rmse"] == pytest.approx(np.sqrt(np.mean((forecasts - realised) ** 2))) and sample["mean_bias"] == pytest.approx(np.mean(forecasts - realised)), "the defaults are the 252, 21 and 21 days"
    ewma = evaluate_covariance_forecasts(x, methods=("ewma",), lookback=100, horizon=10, step=50).loc["ewma"]
    ends = range(100, 420 - 10, 50)
    forecasts = np.array([np.sqrt(equal @ ewma_covariance(x.iloc[end - 100:end], 60.0).to_numpy() @ equal) for end in ends])
    realised = np.array([(x.iloc[end:end + 10].to_numpy() @ equal).std(ddof=1) * np.sqrt(252.0) for end in ends])
    assert ewma["n_windows"] == len(forecasts) == 7 and ewma["rmse"] == pytest.approx(np.sqrt(np.mean((forecasts - realised) ** 2))), "the same sixty-day halflife as the estimator itself"


def test_a_correlation_between_forecast_and_realised_risk_needs_three_windows_and_a_window_with_no_risk_after_it_is_not_scored():
    x = returns_of(100, 3, seed=12)
    three = evaluate_covariance_forecasts(x, methods=("sample",), lookback=60, horizon=10, step=10).loc["sample"]
    assert three["n_windows"] == 3 and np.isfinite(three["corr"])
    two = evaluate_covariance_forecasts(x.iloc[:90], methods=("sample",), lookback=60, horizon=10, step=10).loc["sample"]
    assert two["n_windows"] == 2 and np.isnan(two["corr"]) and np.isfinite(two["rmse"]), "two points always correlate perfectly, which says nothing"
    quiet = x.copy()
    quiet.iloc[60:70] = 0.0                                                                                # the first window is followed by ten days on which nothing moved
    scored = evaluate_covariance_forecasts(quiet, methods=("sample",), lookback=60, horizon=10, step=10).loc["sample"]
    assert scored["n_windows"] == 2 and np.isfinite(scored["qlike"]), "the window that was followed by no risk is not scored: its ratio would be zero"
    dead = x.copy()
    dead.iloc[:60] = 0.0                                                                                   # and the first window is of ten weeks in which nothing moved: it forecast no risk
    forecast_nothing = evaluate_covariance_forecasts(dead, methods=("sample",), lookback=60, horizon=10, step=10).loc["sample"]
    assert forecast_nothing["n_windows"] == 2 and np.isfinite(forecast_nothing["qlike"]) and np.isfinite(forecast_nothing["mean_ratio"]), "a forecast of nothing is not scored either"


def test_the_book_that_was_held_on_the_last_day_of_each_window_is_the_one_scored_against_what_followed():
    x = returns_of(150, 3, seed=13)
    lookback, horizon, step = 60, 10, 10
    held = pd.DataFrame(0.0, index=x.index, columns=x.columns)
    held.iloc[:100, 0] = 1.0                                                                               # asset A is the book up to the 100th day, B after it
    held.iloc[100:, 1] = 1.0
    row = evaluate_covariance_forecasts(x, methods=("sample",), lookback=lookback, horizon=horizon, step=step, weights=held).loc["sample"]
    forecasts, realised = [], []
    for end in range(lookback, len(x) - horizon, step):
        w = held.iloc[end - 1].to_numpy()
        cov = np.cov(x.iloc[end - lookback:end].to_numpy().T, ddof=1) * 252.0
        forecasts.append(np.sqrt(w @ cov @ w))
        realised.append((x.iloc[end:end + horizon].to_numpy() @ w).std(ddof=1) * np.sqrt(252.0))
    f, r = np.array(forecasts), np.array(realised)
    assert row["n_windows"] == len(f) == 8 and row["mean_bias"] == pytest.approx(np.mean(f - r)) and row["rmse"] == pytest.approx(np.sqrt(np.mean((f - r) ** 2)))
    assert row["mean_ratio"] == pytest.approx(np.mean(f / r)), "decided by the weights of the day before each window began being judged, not of the days after it"
