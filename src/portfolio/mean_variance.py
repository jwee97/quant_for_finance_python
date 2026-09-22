"""Mean-variance optimisation -- models M6/M7 (Ch. 19 §19.2-§19.3, spec §28-§30).

    max_w  w'mu - (lambda / 2) w' Sigma w      s.t.  1'w = 1, constraints

and the special cases: global minimum variance, maximum Sharpe ratio, and the
efficient frontier.

The module also contains the experiment that matters more than the optimiser
itself. ``estimation_error_experiment`` perturbs mu slightly and measures how
far the optimal weights move (spec §30). Ch. 19 §19.3's point is that MVO is
an error-maximising machine: it loads on whichever asset's expected return was
most over-estimated. The remedies implemented here -- shrinking mu, shrinking
Sigma, constraining weights, resampling -- are each measured against that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import Constraints
from .covariance import nearest_positive_definite, portfolio_volatility


@dataclass
class OptimisationResult:
    weights: pd.Series
    expected_return: float
    expected_volatility: float
    objective: float
    success: bool
    message: str = ""
    iterations: int = 0

    @property
    def sharpe(self) -> float:
        return self.expected_return / self.expected_volatility if self.expected_volatility > 1e-12 else np.nan


def _prepare(mu: pd.Series | None, covariance: pd.DataFrame):
    assets = list(covariance.columns)
    cov = nearest_positive_definite(covariance).to_numpy(dtype=float)
    if mu is None:
        return assets, cov, np.zeros(len(assets))
    return assets, cov, mu.reindex(assets).fillna(0.0).to_numpy(dtype=float)


def _solve(objective, gradient, assets, constraints: Constraints | None,
           x0: np.ndarray | None = None, previous: np.ndarray | None = None,
           max_iter: int = 400) -> tuple[np.ndarray, bool, str, int]:
    n = len(assets)
    bounds = constraints.bounds(assets) if constraints else [(0.0, 1.0)] * n
    cons = (constraints.scipy_constraints(assets, previous) if constraints
            else [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}])
    start = x0 if x0 is not None else np.full(n, 1.0 / n)
    result = minimize(objective, start, jac=gradient, method="SLSQP", bounds=bounds,
                      constraints=cons, options={"maxiter": max_iter, "ftol": 1e-12})
    return result.x, bool(result.success), str(result.message), int(result.nit)


def mean_variance_weights(mu: pd.Series, covariance: pd.DataFrame, risk_aversion: float = 5.0,
                          constraints: Constraints | None = None,
                          previous: pd.Series | None = None,
                          max_iter: int = 400) -> OptimisationResult:
    """Maximise ``w'mu - (lambda/2) w'Sigma w`` subject to the constraint set."""
    assets, cov, mu_vec = _prepare(mu, covariance)
    prev = previous.reindex(assets).fillna(0.0).to_numpy() if previous is not None else None

    def objective(w):
        return float(-(w @ mu_vec) + 0.5 * risk_aversion * (w @ cov @ w))

    def gradient(w):
        return -mu_vec + risk_aversion * (cov @ w)

    x, success, message, iterations = _solve(objective, gradient, assets, constraints, None, prev, max_iter)
    weights = pd.Series(x, index=assets)
    return OptimisationResult(
        weights=weights,
        expected_return=float(weights @ mu.reindex(assets).fillna(0.0)),
        expected_volatility=portfolio_volatility(x, cov),
        objective=-objective(x),
        success=success, message=message, iterations=iterations,
    )


def minimum_variance_weights(covariance: pd.DataFrame, constraints: Constraints | None = None,
                             max_iter: int = 400) -> OptimisationResult:
    """Global minimum-variance portfolio: needs no expected returns at all.

    Which is exactly why it is worth reporting alongside full MVO -- it
    isolates how much of any improvement came from the covariance matrix
    rather than from the far noisier mu.
    """
    assets, cov, _ = _prepare(None, covariance)

    def objective(w):
        return float(w @ cov @ w)

    def gradient(w):
        return 2.0 * (cov @ w)

    x, success, message, iterations = _solve(objective, gradient, assets, constraints, None, None, max_iter)
    weights = pd.Series(x, index=assets)
    return OptimisationResult(
        weights=weights, expected_return=0.0, expected_volatility=portfolio_volatility(x, cov),
        objective=-objective(x), success=success, message=message, iterations=iterations,
    )


def maximum_sharpe_weights(mu: pd.Series, covariance: pd.DataFrame, risk_free: float = 0.0,
                           constraints: Constraints | None = None, max_iter: int = 400) -> OptimisationResult:
    assets, cov, mu_vec = _prepare(mu, covariance)
    excess = mu_vec - risk_free

    def objective(w):
        vol = np.sqrt(max(w @ cov @ w, 1e-18))
        return float(-(w @ excess) / vol)

    x, success, message, iterations = _solve(objective, None, assets, constraints, None, None, max_iter)
    weights = pd.Series(x, index=assets)
    return OptimisationResult(
        weights=weights, expected_return=float(weights @ mu.reindex(assets).fillna(0.0)),
        expected_volatility=portfolio_volatility(x, cov), objective=-objective(x),
        success=success, message=message, iterations=iterations,
    )


def efficient_frontier(mu: pd.Series, covariance: pd.DataFrame, n_points: int = 25,
                       constraints: Constraints | None = None) -> pd.DataFrame:
    """Frontier traced by sweeping the risk-aversion parameter."""
    rows = []
    for risk_aversion in np.logspace(-1.0, 2.5, n_points):
        result = mean_variance_weights(mu, covariance, risk_aversion, constraints)
        if not result.success:
            continue
        rows.append(
            {
                "risk_aversion": float(risk_aversion),
                "expected_return": result.expected_return,
                "expected_volatility": result.expected_volatility,
                "sharpe": result.sharpe,
                "max_weight": float(result.weights.max()),
                "effective_n": float(1.0 / (result.weights ** 2).sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Estimation error (Ch. 19 §19.3, spec §30)
# ---------------------------------------------------------------------------
def estimation_error_experiment(mu: pd.Series, covariance: pd.DataFrame, risk_aversion: float = 5.0,
                                perturbation: float = 0.10, n_trials: int = 200,
                                constraints: Constraints | None = None,
                                seed: int = 42,
                                methods=("unconstrained", "constrained", "shrunk_mu",
                                         "shrunk_mu_constrained", "min_variance")) -> pd.DataFrame:
    """How far do optimal weights move when mu moves a little? (spec §30)

    ``mu' = mu + eps``, with ``eps`` drawn as a fraction of the cross-sectional
    dispersion of mu. Each "remedy" is a complete pipeline from a raw mu to a
    weight vector, and the *perturbed raw mu* is pushed through that same
    pipeline -- so shrinkage is applied to the noisy input, as it would be in
    production. Comparing a shrunk baseline against unshrunk perturbations
    would measure the shrinkage, not the sensitivity.

    Reported per remedy: the concentration of the baseline book (this is the
    error-maximising behaviour itself) and its sensitivity to noise (the mean
    absolute weight change and the turnover a re-optimisation would trigger).

    One caveat the results make unavoidable: turnover sensitivity on its own
    is a misleading stability metric. An unconstrained solution sitting in a
    three-asset corner barely moves under perturbation *because it is pinned
    against its bounds*, not because it is well estimated -- and it is the
    least diversified book in the table. Read the concentration columns
    (``baseline_effective_n``, ``baseline_n_holdings``) together with the
    sensitivity columns; the out-of-sample walk-forward in Stage 11 is the
    test that settles it.
    """
    rng = np.random.default_rng(seed)
    assets = list(covariance.columns)
    mu_vec = mu.reindex(assets).fillna(0.0)
    dispersion = float(mu_vec.std(ddof=1)) or float(abs(mu_vec.mean())) or 0.01

    loose = Constraints(min_weight=0.0, max_weight=1.0, group_limits={}, group_map={}, net_exposure=1.0)
    tight = constraints or loose

    def shrink(vector: pd.Series, intensity: float = 0.5) -> pd.Series:
        return (1.0 - intensity) * vector + intensity * float(vector.mean())

    pipelines = {
        "unconstrained": lambda m: mean_variance_weights(m, covariance, risk_aversion, loose),
        "constrained": lambda m: mean_variance_weights(m, covariance, risk_aversion, tight),
        "shrunk_mu": lambda m: mean_variance_weights(shrink(m), covariance, risk_aversion, loose),
        "shrunk_mu_constrained": lambda m: mean_variance_weights(shrink(m), covariance, risk_aversion, tight),
        # Uses no expected returns at all: the limiting case of shrinking mu
        # entirely away, and therefore completely immune to errors in it.
        "min_variance": lambda m: minimum_variance_weights(covariance, tight),
    }

    rows = []
    for name in methods:
        pipeline = pipelines[name]
        baseline = pipeline(mu_vec)
        if not baseline.success:
            continue
        deltas, turnovers, max_weights, effective = [], [], [], []
        for _ in range(n_trials):
            noise = rng.normal(0.0, perturbation * dispersion, size=len(assets))
            trial = pipeline(mu_vec + pd.Series(noise, index=assets))
            if not trial.success:
                continue
            difference = trial.weights - baseline.weights
            deltas.append(float(difference.abs().mean()))
            turnovers.append(float(difference.abs().sum()))
            max_weights.append(float(trial.weights.max()))
            effective.append(float(1.0 / (trial.weights ** 2).sum()))
        if not deltas:
            continue
        rows.append(
            {
                "method": name,
                "baseline_max_weight": float(baseline.weights.max()),
                "baseline_effective_n": float(1.0 / (baseline.weights ** 2).sum()),
                "baseline_n_holdings": int((baseline.weights > 1e-4).sum()),
                "mean_abs_weight_change": float(np.mean(deltas)),
                "mean_turnover_from_noise": float(np.mean(turnovers)),
                "p95_turnover_from_noise": float(np.percentile(turnovers, 95)),
                "max_turnover_from_noise": float(np.max(turnovers)),
                "mean_max_weight": float(np.mean(max_weights)),
                "mean_effective_n": float(np.mean(effective)),
                "n_trials": int(len(deltas)),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def resampled_weights(mu: pd.Series, returns: pd.DataFrame, risk_aversion: float = 5.0,
                      n_resamples: int = 100, constraints: Constraints | None = None,
                      covariance_method: str = "shrinkage", seed: int = 7) -> pd.Series:
    """Michaud resampling -- Extension B (Ch. 19 §19.6).

    Bootstrap the return history, re-estimate mu and Sigma on each sample,
    optimise, and average the resulting weights. Averaging over the
    *estimation uncertainty* rather than optimising once on a point estimate
    produces a far more stable book at the cost of a slightly worse in-sample
    objective.
    """
    from .covariance import estimate_covariance

    rng = np.random.default_rng(seed)
    clean = returns.dropna(how="any")
    assets = list(clean.columns)
    n_obs = len(clean)
    if n_obs < 60:
        raise ValueError("not enough observations to resample")

    accumulated = np.zeros(len(assets))
    succeeded = 0
    for _ in range(n_resamples):
        draw = rng.integers(0, n_obs, size=n_obs)
        sample = clean.iloc[draw]
        try:
            cov = estimate_covariance(sample, covariance_method, len(sample), annualise=True)
        except (ValueError, np.linalg.LinAlgError):
            continue
        sample_mu = mu.reindex(assets).fillna(0.0) * (1.0 + rng.normal(0.0, 0.10, len(assets)))
        result = mean_variance_weights(sample_mu, cov, risk_aversion, constraints)
        if result.success:
            accumulated += result.weights.reindex(assets).to_numpy()
            succeeded += 1
    if succeeded == 0:
        raise RuntimeError("no resampled optimisation succeeded")
    weights = pd.Series(accumulated / succeeded, index=assets)
    return weights / weights.sum()


def robust_weights(mu: pd.Series, covariance: pd.DataFrame, uncertainty: pd.Series | None = None,
                   risk_aversion: float = 5.0, kappa: float = 1.0,
                   constraints: Constraints | None = None) -> OptimisationResult:
    """Robust optimisation -- Extension C (Ch. 19 §19.7).

        max_w  w'mu - kappa * ||Omega^(1/2) w|| - (lambda/2) w'Sigma w

    The middle term is the worst case over an ellipsoidal uncertainty set
    around mu. Instead of pretending mu is known, it charges the portfolio for
    concentrating in assets whose expected return is most uncertain -- which
    is the honest response to the estimation-error experiment above.
    """
    assets, cov, mu_vec = _prepare(mu, covariance)
    if uncertainty is None:
        # Default: uncertainty proportional to each asset's own volatility.
        omega = np.diag(np.diag(cov)) / max(len(assets), 1)
    else:
        omega = np.diag(uncertainty.reindex(assets).fillna(0.0).to_numpy() ** 2)

    def objective(w):
        penalty = kappa * np.sqrt(max(w @ omega @ w, 1e-18))
        return float(-(w @ mu_vec) + penalty + 0.5 * risk_aversion * (w @ cov @ w))

    x, success, message, iterations = _solve(objective, None, assets, constraints)
    weights = pd.Series(x, index=assets)
    return OptimisationResult(
        weights=weights, expected_return=float(weights @ mu.reindex(assets).fillna(0.0)),
        expected_volatility=portfolio_volatility(x, cov), objective=-objective(x),
        success=success, message=message, iterations=iterations,
    )
