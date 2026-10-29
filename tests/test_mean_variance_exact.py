"""The mean-variance optimiser against the closed forms it must reproduce.

With every weight strictly inside its bounds the three classic problems have formulas, and an optimiser that returns anything else on them is wrong whatever it does on real data: the minimum-variance book is
``S^-1 1 / 1'S^-1 1``, the book that maximises ``w'mu - (lambda/2) w'S w`` with the weights adding to one is ``S^-1 (mu - g 1) / lambda`` and the tangency book is proportional to ``S^-1 (mu - rf)``. The same
inputs then check what the optimiser reports, its limits, its frontier and the experiments built on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.constraints import Constraints
from src.portfolio.mean_variance import (OptimisationResult, efficient_frontier, estimation_error_experiment, maximum_sharpe_weights, mean_variance_weights, minimum_variance_weights, resampled_weights,
                                         robust_weights)

ASSETS = ["A", "B", "C"]
COV = pd.DataFrame([[0.040, 0.006, 0.002], [0.006, 0.0225, 0.004], [0.002, 0.004, 0.010]], index=ASSETS, columns=ASSETS)
MU = pd.Series([0.060, 0.050, 0.045], index=ASSETS)
INVERSE = np.linalg.inv(COV.to_numpy())
ONES = np.ones(3)
LOOSE = Constraints(min_weight=0.0, max_weight=1.0, group_limits={}, group_map={}, net_exposure=1.0)


def mean_variance_by_formula(mu: pd.Series, risk_aversion: float) -> np.ndarray:
    gamma = (ONES @ INVERSE @ mu.to_numpy() - risk_aversion) / (ONES @ INVERSE @ ONES)
    return INVERSE @ (mu.to_numpy() - gamma * ONES) / risk_aversion


def test_the_minimum_variance_book_is_the_inverse_covariance_over_its_sum():
    expected = INVERSE @ ONES / (ONES @ INVERSE @ ONES)
    assert (expected > 0).all()
    for constraints in (None, LOOSE):
        result = minimum_variance_weights(COV, constraints)
        assert result.success and result.weights.to_numpy() == pytest.approx(expected, abs=1e-5) and list(result.weights.index) == ASSETS, "to the accuracy of a solver whose objective is flat at its minimum"
        assert result.expected_volatility == pytest.approx(np.sqrt(expected @ COV.to_numpy() @ expected), abs=1e-6) and result.expected_return == 0.0
        assert result.objective == pytest.approx(-(expected @ COV.to_numpy() @ expected), abs=1e-9) and result.sharpe == 0.0
        assert result.weights.sum() == pytest.approx(1.0) and result.iterations > 0 and isinstance(result, OptimisationResult)


def test_the_mean_variance_book_is_the_formula_and_takes_less_risk_the_more_averse_it_is():
    previous_volatility = None
    for risk_aversion in (2.0, 5.0, 12.0):
        expected = mean_variance_by_formula(MU, risk_aversion)
        assert (expected > 0).all()
        result = mean_variance_weights(MU, COV, risk_aversion, LOOSE)
        assert result.success and result.weights.to_numpy() == pytest.approx(expected, abs=1e-6)
        assert result.expected_return == pytest.approx(expected @ MU.to_numpy(), abs=1e-6) and result.expected_volatility == pytest.approx(np.sqrt(expected @ COV.to_numpy() @ expected), abs=1e-6)
        assert result.objective == pytest.approx(expected @ MU.to_numpy() - 0.5 * risk_aversion * expected @ COV.to_numpy() @ expected, abs=1e-8), "the utility the optimiser maximised"
        assert result.sharpe == pytest.approx(result.expected_return / result.expected_volatility)
        assert previous_volatility is None or result.expected_volatility < previous_volatility
        previous_volatility = result.expected_volatility
    default = mean_variance_weights(MU, COV, constraints=LOOSE)
    assert default.weights.to_numpy() == pytest.approx(mean_variance_by_formula(MU, 5.0), abs=1e-6), "a risk aversion of five by default"
    assert mean_variance_weights(MU, COV).weights.to_numpy() == pytest.approx(mean_variance_by_formula(MU, 5.0), abs=1e-6), "and with no constraints given, weights between nothing and everything that add to one"
    shuffled = mean_variance_weights(MU.reindex(["C", "A", "B"]), COV, 5.0, LOOSE)
    assert shuffled.weights.to_numpy() == pytest.approx(default.weights.to_numpy(), abs=1e-9) and list(shuffled.weights.index) == ASSETS, "expected returns are matched to the covariance by name"
    partial = mean_variance_weights(MU.drop("C"), COV, 5.0, LOOSE)
    zero_mu = MU.copy()
    zero_mu["C"] = 0.0
    assert partial.weights.to_numpy() == pytest.approx(mean_variance_weights(zero_mu, COV, 5.0, LOOSE).weights.to_numpy(), abs=1e-9), "an asset with no expected return is expected to return nothing"


def test_the_maximum_sharpe_book_is_proportional_to_the_inverse_covariance_times_the_excess_return():
    for risk_free in (0.0, 0.02):
        raw = INVERSE @ (MU.to_numpy() - risk_free)
        expected = raw / raw.sum()
        assert (expected > 0).all()
        result = maximum_sharpe_weights(MU, COV, risk_free, LOOSE)
        assert result.success and result.weights.to_numpy() == pytest.approx(expected, abs=1e-5)
        sharpe = (expected @ (MU.to_numpy() - risk_free)) / np.sqrt(expected @ COV.to_numpy() @ expected)
        assert result.objective == pytest.approx(sharpe, abs=1e-6) and result.expected_return == pytest.approx(expected @ MU.to_numpy(), abs=1e-6)
    assert maximum_sharpe_weights(MU, COV, constraints=LOOSE).weights.to_numpy() == pytest.approx(INVERSE @ MU.to_numpy() / (INVERSE @ MU.to_numpy()).sum(), abs=1e-5), "no risk-free rate by default"


def test_a_cap_on_a_weight_is_kept_and_the_rest_is_the_best_that_remains():
    capped = Constraints(min_weight=0.0, max_weight=0.35, group_limits={}, group_map={}, net_exposure=1.0)
    unconstrained = mean_variance_by_formula(MU, 2.0)
    assert unconstrained.max() > 0.35, "the cap binds"
    result = mean_variance_weights(MU, COV, 2.0, capped)
    assert result.success and result.weights.max() <= 0.35 + 1e-9 and result.weights.min() >= -1e-9 and result.weights.sum() == pytest.approx(1.0)
    grid = np.array([(a, b, 1.0 - a - b) for a in np.arange(0.0, 0.3501, 0.005) for b in np.arange(0.0, 0.3501, 0.005) if 0.0 <= 1.0 - a - b <= 0.35])
    best = (grid @ MU.to_numpy() - 1.0 * np.einsum("ij,jk,ik->i", grid, COV.to_numpy(), grid)).max()                       # the utility with a risk aversion of two, searched on a grid
    assert result.objective >= best - 1e-9 and result.objective <= best + 1e-3, "no grid point does better, and the grid is close"
    floor = Constraints(min_weight=0.2, max_weight=1.0, group_limits={}, group_map={}, net_exposure=1.0)
    assert mean_variance_weights(MU, COV, 5.0, floor).weights.min() >= 0.2 - 1e-9
    group = Constraints(min_weight=0.0, max_weight=1.0, group_limits={"g": 0.3}, group_map={"A": "g", "B": "g"}, net_exposure=1.0)
    limited = mean_variance_weights(MU, COV, 2.0, group).weights
    assert limited[["A", "B"]].sum() <= 0.3 + 1e-9 and limited.sum() == pytest.approx(1.0)
    away = Constraints(min_weight=0.0, max_weight=1.0, group_limits={}, group_map={}, net_exposure=1.0, max_turnover=0.1)
    start = pd.Series([0.6, 0.2, 0.2], index=ASSETS)
    optimum = mean_variance_by_formula(MU, 5.0)
    moved = mean_variance_weights(MU, COV, 5.0, away, previous=start).weights
    assert (moved - start).abs().sum() == pytest.approx(0.1, abs=1e-5), "a turnover limit holds the book within a tenth of the one it had, and uses all of it"
    assert np.abs(moved.to_numpy() - optimum).sum() < np.abs(start.to_numpy() - optimum).sum() and not np.allclose(moved, optimum, atol=1e-2), "on the way to the optimum and not there"
    assert (mean_variance_weights(MU, COV, 5.0, away).weights.to_numpy() == pytest.approx(mean_variance_by_formula(MU, 5.0), abs=1e-6)), "and with no previous book there is nothing to stay near"


def test_the_frontier_is_swept_by_risk_aversion_from_a_tenth_to_three_hundred_and_returns_less_as_it_asks_for_less_risk():
    frontier = efficient_frontier(MU, COV, n_points=8, constraints=LOOSE)
    assert list(frontier.columns) == ["risk_aversion", "expected_return", "expected_volatility", "sharpe", "max_weight", "effective_n"] and len(frontier) == 8
    assert frontier["risk_aversion"].tolist() == pytest.approx(list(np.logspace(-1.0, 2.5, 8)))
    assert frontier["expected_return"].is_monotonic_decreasing and frontier["expected_volatility"].is_monotonic_decreasing
    first, last = frontier.iloc[0], frontier.iloc[-1]
    assert first["max_weight"] == pytest.approx(1.0, abs=1e-6) and first["effective_n"] == pytest.approx(1.0, abs=1e-4), "hardly averse at all: everything in the asset with the highest expected return"
    expected = mean_variance_by_formula(MU, float(last["risk_aversion"]))
    assert last["max_weight"] == pytest.approx(expected.max(), abs=1e-5) and last["effective_n"] == pytest.approx(1.0 / (expected ** 2).sum(), rel=1e-4)
    assert last["sharpe"] == pytest.approx(last["expected_return"] / last["expected_volatility"]) and len(efficient_frontier(MU, COV, n_points=3)) == 3


def test_the_experiment_on_estimation_error_repeats_the_optimisation_on_noisy_expected_returns_and_the_minimum_variance_book_ignores_them():
    table = estimation_error_experiment(MU, COV, n_trials=12, methods=("unconstrained", "min_variance"))
    assert list(table.index) == ["unconstrained", "min_variance"] and table["n_trials"].tolist() == [12, 12]
    assert list(table.columns) == ["baseline_max_weight", "baseline_effective_n", "baseline_n_holdings", "mean_abs_weight_change", "mean_turnover_from_noise", "p95_turnover_from_noise", "max_turnover_from_noise",
                                   "mean_max_weight", "mean_effective_n", "n_trials"]
    base = mean_variance_by_formula(MU, 5.0)
    row = table.loc["unconstrained"]
    assert row["baseline_max_weight"] == pytest.approx(base.max(), abs=1e-5) and row["baseline_effective_n"] == pytest.approx(1.0 / (base ** 2).sum(), rel=1e-4) and row["baseline_n_holdings"] == 3
    rng = np.random.default_rng(42)                                                                                  # the same noise the experiment draws: ten per cent of the spread of the expected returns
    dispersion = MU.std(ddof=1)
    changes = []
    for _ in range(12):
        noisy = MU + pd.Series(rng.normal(0.0, 0.10 * dispersion, 3), index=ASSETS)
        changes.append(mean_variance_by_formula(noisy, 5.0) - base)
    assert row["mean_abs_weight_change"] == pytest.approx(np.mean([np.abs(c).mean() for c in changes]), rel=1e-3)
    assert row["mean_turnover_from_noise"] == pytest.approx(np.mean([np.abs(c).sum() for c in changes]), rel=1e-3) and row["max_turnover_from_noise"] == pytest.approx(max(np.abs(c).sum() for c in changes), rel=1e-3)
    assert row["p95_turnover_from_noise"] == pytest.approx(np.percentile([np.abs(c).sum() for c in changes], 95), rel=1e-3)
    still = table.loc["min_variance"]
    assert still["mean_abs_weight_change"] == pytest.approx(0.0, abs=1e-9) and still["max_turnover_from_noise"] == pytest.approx(0.0, abs=1e-9), "no expected returns, no error in them"
    assert still["baseline_max_weight"] == pytest.approx((INVERSE @ ONES / (ONES @ INVERSE @ ONES)).max(), abs=1e-5)
    again = estimation_error_experiment(MU, COV, n_trials=12, methods=("unconstrained",))
    other = estimation_error_experiment(MU, COV, n_trials=12, methods=("unconstrained",), seed=1)
    assert again.loc["unconstrained", "mean_abs_weight_change"] == pytest.approx(row["mean_abs_weight_change"]) and other.loc["unconstrained", "mean_abs_weight_change"] != row["mean_abs_weight_change"]
    shrunk = estimation_error_experiment(MU, COV, n_trials=12, methods=("shrunk_mu", "constrained"))
    assert shrunk.loc["shrunk_mu", "mean_turnover_from_noise"] < row["mean_turnover_from_noise"], "half the expected returns shrunk to their mean: less movement for the same noise"
    assert shrunk.loc["constrained", "baseline_max_weight"] == pytest.approx(row["baseline_max_weight"]), "with no constraints given the constrained pipeline is the unconstrained one"
    capped = estimation_error_experiment(MU, COV, n_trials=4, constraints=Constraints(min_weight=0.0, max_weight=0.4, group_limits={}, group_map={}, net_exposure=1.0), methods=("constrained",))
    assert capped.loc["constrained", "baseline_max_weight"] <= 0.4 + 1e-9 and capped.loc["constrained", "mean_max_weight"] <= 0.4 + 1e-9


def test_a_book_resampled_from_the_history_adds_to_one_is_repeatable_and_needs_sixty_days():
    rng = np.random.default_rng(3)
    history = pd.DataFrame(rng.multivariate_normal(MU.to_numpy() / 252.0, COV.to_numpy() / 252.0, size=300), columns=ASSETS)
    book = resampled_weights(MU, history, n_resamples=6, constraints=LOOSE)
    assert book.sum() == pytest.approx(1.0) and (book >= -1e-9).all() and list(book.index) == ASSETS
    assert resampled_weights(MU, history, n_resamples=6, constraints=LOOSE).to_numpy() == pytest.approx(book.to_numpy()) and not np.allclose(resampled_weights(MU, history, n_resamples=6, constraints=LOOSE, seed=8), book)
    with pytest.raises(ValueError, match="not enough observations to resample"):
        resampled_weights(MU, history.iloc[:59])
    assert resampled_weights(MU, history.iloc[:60], n_resamples=2, constraints=LOOSE).sum() == pytest.approx(1.0)
    gappy = history.copy()
    gappy.iloc[:241, 0] = np.nan
    with pytest.raises(ValueError, match="not enough observations to resample"):
        resampled_weights(MU, gappy)                                                                           # a day with a missing return is not a day: 59 are left


def test_the_robust_book_charges_for_the_uncertainty_of_the_assets_it_holds():
    plain = mean_variance_weights(MU, COV, 5.0, LOOSE)
    free = robust_weights(MU, COV, risk_aversion=5.0, kappa=0.0, constraints=LOOSE)
    assert free.success and free.weights.to_numpy() == pytest.approx(plain.weights.to_numpy(), abs=1e-5), "with no price on uncertainty it is mean-variance"
    unsure = pd.Series([0.30, 0.01, 0.01], index=ASSETS)                                                          # we are far less sure of the first asset's expected return
    charged = robust_weights(MU, COV, uncertainty=unsure, risk_aversion=5.0, kappa=1.0, constraints=LOOSE)
    assert charged.weights["A"] < plain.weights["A"] - 0.05 and charged.weights.sum() == pytest.approx(1.0)
    assert robust_weights(MU, COV, uncertainty=unsure, risk_aversion=5.0, kappa=3.0, constraints=LOOSE).weights["A"] < charged.weights["A"], "the more it is charged, the less of it"
    default = robust_weights(MU, COV, risk_aversion=5.0, kappa=1.0, constraints=LOOSE)
    omega = np.diag(np.diag(COV.to_numpy())) / 3.0

    def utility(w):
        return w @ MU.to_numpy() - np.sqrt(w @ omega @ w) - 2.5 * w @ COV.to_numpy() @ w

    assert default.objective == pytest.approx(utility(default.weights.to_numpy()), abs=1e-9) and utility(default.weights.to_numpy()) >= utility(plain.weights.to_numpy()) - 1e-9, "uncertainty proportional to each asset's own variance by default"
    assert default.expected_return == pytest.approx(default.weights.to_numpy() @ MU.to_numpy())


def test_a_sharpe_ratio_needs_some_volatility_and_the_default_frontier_has_twenty_five_points():
    book = pd.Series([1.0], index=["A"])
    assert OptimisationResult(book, 0.05, 1.2e-12, 0.0, True).sharpe == pytest.approx(0.05 / 1.2e-12) and np.isnan(OptimisationResult(book, 0.05, 1e-12, 0.0, True).sharpe) and np.isnan(OptimisationResult(book, 0.05, 0.0, 0.0, True).sharpe)
    assert len(efficient_frontier(MU, COV, constraints=LOOSE)) == 25


def test_the_robust_book_charges_kappa_times_the_worst_case_over_the_uncertainty_given_and_one_by_default():
    unsure = pd.Series([0.30, 0.05, 0.02], index=ASSETS)
    for kappa in (0.5, 2.0):
        result = robust_weights(MU, COV, uncertainty=unsure, risk_aversion=5.0, kappa=kappa, constraints=LOOSE)
        w = result.weights.to_numpy()
        penalty = kappa * np.sqrt(w @ np.diag(unsure.to_numpy() ** 2) @ w)
        assert result.objective == pytest.approx(w @ MU.to_numpy() - penalty - 2.5 * w @ COV.to_numpy() @ w, abs=1e-9), "the expected return, less the price of its uncertainty, less the price of the risk"
    default = robust_weights(MU, COV, uncertainty=unsure, constraints=LOOSE)
    assert default.weights.to_numpy() == pytest.approx(robust_weights(MU, COV, uncertainty=unsure, risk_aversion=5.0, kappa=1.0, constraints=LOOSE).weights.to_numpy(), abs=1e-9), "a risk aversion of five and a price of one by default"
    unknown = robust_weights(MU, COV, uncertainty=unsure.drop("B"), risk_aversion=5.0, kappa=1.0, constraints=LOOSE)
    certain = robust_weights(MU, COV, uncertainty=unsure.where(unsure.index != "B", 0.0), risk_aversion=5.0, kappa=1.0, constraints=LOOSE)
    assert unknown.weights.to_numpy() == pytest.approx(certain.weights.to_numpy(), abs=1e-9), "an asset with no uncertainty given is taken as certain"


def test_shrinking_the_expected_returns_halfway_to_their_mean_moves_the_book_less_for_the_same_noise_and_a_corner_has_fewer_holdings():
    table = estimation_error_experiment(MU, COV, n_trials=10, methods=("unconstrained", "shrunk_mu"))
    rng = np.random.default_rng(42)
    dispersion = MU.std(ddof=1)

    def shrink(v: pd.Series) -> pd.Series:
        return 0.5 * v + 0.5 * v.mean()

    trials = [MU + pd.Series(rng.normal(0.0, 0.10 * dispersion, 3), index=ASSETS) for _ in range(10)]
    row = table.loc["unconstrained"]
    full = [mean_variance_by_formula(m, 5.0) for m in trials]
    assert row["mean_max_weight"] == pytest.approx(np.mean([w.max() for w in full]), rel=1e-3) and row["mean_effective_n"] == pytest.approx(np.mean([1.0 / (w ** 2).sum() for w in full]), rel=1e-3)
    rng3 = np.random.default_rng(42)                                                                                    # the method run second draws its noise after the first: the same stream continues
    for _ in range(10):
        rng3.normal(0.0, 0.10 * dispersion, 3)
    later = [MU + pd.Series(rng3.normal(0.0, 0.10 * dispersion, 3), index=ASSETS) for _ in range(10)]
    base = mean_variance_by_formula(shrink(MU), 5.0)
    shrunk = [mean_variance_by_formula(shrink(m), 5.0) for m in later]
    row = table.loc["shrunk_mu"]
    assert row["baseline_max_weight"] == pytest.approx(base.max(), abs=1e-5) and row["mean_abs_weight_change"] == pytest.approx(np.mean([np.abs(w - base).mean() for w in shrunk]), rel=1e-3)
    assert row["mean_turnover_from_noise"] == pytest.approx(np.mean([np.abs(w - base).sum() for w in shrunk]), rel=1e-3) and row["mean_max_weight"] == pytest.approx(np.mean([w.max() for w in shrunk]), rel=1e-3)
    assert row["mean_effective_n"] == pytest.approx(np.mean([1.0 / (w ** 2).sum() for w in shrunk]), rel=1e-3)
    assert len(estimation_error_experiment(MU, COV, methods=("min_variance",))) == 1 and estimation_error_experiment(MU, COV, methods=("min_variance",)).loc["min_variance", "n_trials"] == 200, "two hundred draws by default"
    corner = pd.Series([0.06, 0.05, -0.30], index=ASSETS)
    assert estimation_error_experiment(corner, COV, n_trials=3, methods=("unconstrained",)).loc["unconstrained", "baseline_n_holdings"] == 2, "a book that holds nothing in an asset with a dismal outlook"


def resampled_by_hand(mu: pd.Series, returns: pd.DataFrame, risk_aversion: float, n_resamples: int, constraints, seed: int = 7) -> pd.Series:
    from src.portfolio.covariance import estimate_covariance

    rng = np.random.default_rng(seed)
    clean = returns.dropna(how="any")
    assets, n = list(clean.columns), len(clean)
    total, done = np.zeros(len(assets)), 0
    for _ in range(n_resamples):
        sample = clean.iloc[rng.integers(0, n, size=n)]
        cov = estimate_covariance(sample, "shrinkage", len(sample), annualise=True)
        result = mean_variance_weights(mu.reindex(assets).fillna(0.0) * (1.0 + rng.normal(0.0, 0.10, len(assets))), cov, risk_aversion, constraints)
        if result.success:
            total, done = total + result.weights.reindex(assets).to_numpy(), done + 1
    book = pd.Series(total / done, index=assets)
    return book / book.sum()


def test_a_resampled_book_is_the_average_of_the_books_of_bootstrapped_covariances_and_perturbed_expected_returns():
    history = pd.DataFrame(np.random.default_rng(3).multivariate_normal(MU.to_numpy() / 252.0, COV.to_numpy() / 252.0, size=300), columns=ASSETS)
    assert resampled_weights(MU, history, risk_aversion=3.0, n_resamples=5, constraints=LOOSE).to_numpy() == pytest.approx(resampled_by_hand(MU, history, 3.0, 5, LOOSE).to_numpy(), abs=1e-9)
    assert resampled_weights(MU, history, risk_aversion=3.0, n_resamples=5, constraints=LOOSE, seed=4).to_numpy() == pytest.approx(resampled_by_hand(MU, history, 3.0, 5, LOOSE, seed=4).to_numpy(), abs=1e-9)
    assert resampled_weights(MU, history, constraints=LOOSE).to_numpy() == pytest.approx(resampled_by_hand(MU, history, 5.0, 100, LOOSE).to_numpy(), abs=1e-9), "a hundred resamples and a risk aversion of five by default"
    assert resampled_weights(MU, history, n_resamples=3, constraints=LOOSE, covariance_method="sample").to_numpy() != pytest.approx(resampled_weights(MU, history, n_resamples=3, constraints=LOOSE).to_numpy(), abs=1e-9)
