"""Bayesian decision theory for portfolios: choose the weights that maximise *expected* utility under the posterior predictive distribution, not the utility at the posterior mean.

A plug-in optimiser estimates the parameters, treats them as true and optimises. A Bayesian one asks what return distribution it should expect, parameter uncertainty included (draw the parameters from
their posterior, then a return from each), and picks the action with the highest expected utility over those draws. The two agree only for quadratic utility with the posterior mean and the *predictive*
covariance; for the utilities investors actually have (power, exponential, shortfall, CVaR) they differ, and the Bayesian choice is the one that is optimal for an investor who knows what she does not know.

Utilities ``u(w; R)`` take a (draws, assets) array of period returns and return the (draws,) utilities of holding ``w``: ``crra`` (power utility of terminal wealth, gamma = 1 is log), ``cara`` (exponential),
``mean_variance`` (the quadratic approximation), ``cvar`` (minus the average of the worst tail of the draws, so maximising it minimises the tail loss) and ``shortfall`` (a loss below a target, with
``kappa`` times the weight of a gain). :func:`bayes_weights` maximises the sample average by sequential least squares on the long-only, fully-invested simplex (or any bounds you give);
:func:`regret` and :func:`value_of_information` measure what ignoring parameter uncertainty costs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize


def crra(gamma: float = 5.0) -> Callable:
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    def u(w, R):
        wealth = np.maximum(1.0 + R @ w, 1e-9)
        return np.log(wealth) if abs(gamma - 1.0) < 1e-12 else (wealth ** (1.0 - gamma) - 1.0) / (1.0 - gamma)
    return u


def cara(a: float = 5.0) -> Callable:
    if a <= 0:
        raise ValueError("a must be positive")
    return lambda w, R: -np.exp(-a * (R @ w)) / a


def mean_variance(risk_aversion: float = 5.0) -> Callable:
    """The quadratic utility whose expectation is ``mean - (risk_aversion / 2) * variance`` of the portfolio return across draws (which is what the sample average of ``r - (a/2)(r - m)^2`` gives)."""
    def u(w, R):
        r = R @ w
        return r - 0.5 * risk_aversion * (r - r.mean()) ** 2
    return u


def cvar(alpha: float = 0.95, weight: float = 1.0) -> Callable:
    """Mean return less ``weight`` times the expected loss in the worst ``1 - alpha`` of the draws (a utility, so larger is better; it is not additive across draws, and is evaluated on the whole sample)."""
    if not 0.5 <= alpha < 1:
        raise ValueError("0.5 <= alpha < 1")

    def u(w, R):
        r = R @ w
        k = max(int(np.ceil((1 - alpha) * len(r))), 1)
        tail = np.partition(r, k - 1)[:k].mean()
        return np.full(len(r), r.mean() + weight * tail)
    return u


def shortfall(target: float = 0.0, kappa: float = 3.0) -> Callable:
    """A gain above ``target`` counts once, a shortfall below it ``kappa`` times (loss aversion)."""
    return lambda w, R: np.where(R @ w >= target, R @ w - target, kappa * (R @ w - target))


def expected_utility(w, R, utility: Callable) -> float:
    return float(np.mean(utility(np.asarray(w, dtype=float), R)))


def bayes_weights(R: np.ndarray, utility: Callable, lb=0.0, ub=1.0, budget: float | None = 1.0, w0=None) -> np.ndarray:
    """The weights maximising the average utility over the predictive draws ``R`` (draws x assets): within ``lb <= w <= ub`` and, if ``budget`` is not None, ``sum(w) = budget``."""
    R = np.asarray(R, dtype=float)
    n = R.shape[1]
    w0 = np.full(n, (budget or 1.0) / n) if w0 is None else np.asarray(w0, dtype=float)
    bounds = [(float(np.broadcast_to(lb, n)[i]), float(np.broadcast_to(ub, n)[i])) for i in range(n)]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - budget}] if budget is not None else []
    res = minimize(lambda w: -expected_utility(w, R, utility), w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    return res.x


def predictive_draws(mu_draws: np.ndarray, cov_draws: np.ndarray, per_parameter: int = 1, seed: int = 0) -> np.ndarray:
    """Predictive returns: for every parameter draw ``(mu, Sigma)`` ``per_parameter`` returns from ``N(mu, Sigma)``, so parameter uncertainty is part of the spread."""
    rng = np.random.default_rng(seed)
    out = []
    for mu, S in zip(mu_draws, cov_draws):
        L = np.linalg.cholesky(0.5 * (S + S.T) + 1e-14 * np.eye(len(mu)))
        out.append(mu + rng.normal(size=(per_parameter, len(mu))) @ L.T)
    return np.vstack(out)


def regret(w, mu_draws, cov_draws, utility: Callable, lb=0.0, ub=1.0, budget: float | None = 1.0, per_parameter: int = 400, seed: int = 0, limit: int = 40) -> dict:
    """The expected regret of ``w``: over parameter draws, the expected utility of the best weights *for those parameters* less that of ``w``. Returns the mean regret (the Bayes risk of ``w``) and the mean
    utility of the oracle, which is what a perfectly informed investor would get (:func:`value_of_information` is the difference between that and the Bayes decision's)."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(mu_draws), size=min(limit, len(mu_draws)), replace=False)
    losses, best = [], []
    for k, i in enumerate(pick):
        R = predictive_draws(mu_draws[i:i + 1], cov_draws[i:i + 1], per_parameter, seed + k)
        star = bayes_weights(R, utility, lb, ub, budget)
        best.append(expected_utility(star, R, utility))
        losses.append(expected_utility(star, R, utility) - expected_utility(w, R, utility))
    return {"regret": float(np.mean(losses)), "oracle_utility": float(np.mean(best))}


def value_of_information(mu_draws, cov_draws, utility: Callable, lb=0.0, ub=1.0, budget: float | None = 1.0, per_parameter: int = 400, seed: int = 0, limit: int = 40) -> dict:
    """The expected value of perfect information: the average utility of the best weights for each parameter draw less the utility of the single best decision under the posterior predictive. It is what knowing
    the parameters would be worth, in utility, and an upper bound on what any amount of extra data can add."""
    R = predictive_draws(mu_draws, cov_draws, max(per_parameter // 20, 5), seed)
    w = bayes_weights(R, utility, lb, ub, budget)
    r = regret(w, mu_draws, cov_draws, utility, lb, ub, budget, per_parameter, seed, limit)
    return {"evpi": r["regret"], "bayes_weights": w}
