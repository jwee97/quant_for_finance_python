"""Mean-CVaR optimisation -- Extension D / model M9 (Ch. 19 §19.8, spec §52).

Variance punishes upside and downside identically. On a universe whose return
distributions have excess kurtosis between 2.8 and 59 (Stage 2), that is not a
harmless simplification. Mean-CVaR optimises the average loss in the worst
``1 - alpha`` of scenarios instead.

Rockafellar-Uryasev's key result makes this tractable: CVaR is the minimum
over an auxiliary variable ``z`` of

    z + 1/((1-alpha) S) * sum_s max(-r_s'w - z, 0)

which is convex in ``(w, z)`` jointly and is a linear program when the
scenarios are historical returns. No distributional assumption is required --
the tail of the scenario set *is* the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .constraints import Constraints


def scenario_cvar(weights: np.ndarray, scenarios: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR of a weight vector over a scenario matrix, as a positive loss."""
    losses = -(scenarios @ weights)
    cutoff = np.quantile(losses, alpha)
    tail = losses[losses >= cutoff]
    return float(tail.mean()) if len(tail) else float(cutoff)


def mean_cvar_weights(scenarios: pd.DataFrame, mu: pd.Series | None = None, alpha: float = 0.95,
                      target_return: float | None = None, risk_aversion: float = 1.0,
                      constraints: Constraints | None = None,
                      mu_is_annualised: bool = True, periods_per_year: int = 252) -> pd.Series:
    """Minimise CVaR (optionally net of an expected-return term) by linear programming.

    Decision vector: ``[w (n), z (1), u (S)]`` with ``u_s >= -r_s'w - z`` and
    ``u_s >= 0``, minimising ``z + mean(u)/(1-alpha) - (1/lambda) mu'w``.

    **Units.** The CVaR term is measured in the units of ``scenarios`` -- daily
    returns, if the scenario set is daily history. An annualised ``mu`` is
    therefore ~252x too large relative to it, and the expected-return term
    silently swamps the tail term: the optimiser degenerates into
    "maximise mu" and returns the same corner solution as mean-variance.
    ``mu_is_annualised=True`` (the default) converts mu to per-period units so
    the two halves of the objective are commensurable. Pass ``False`` only if
    mu is already stated per scenario period.
    """
    assets = list(scenarios.columns)
    n = len(assets)
    matrix = scenarios.dropna(how="any").to_numpy(dtype=float)
    s = len(matrix)
    if s < 50:
        raise ValueError(f"need at least 50 scenarios, got {s}")

    scale = 1.0 / ((1.0 - alpha) * s)
    mu_period = None
    if mu is not None:
        mu_period = mu.reindex(assets).fillna(0.0).astype(float)
        if mu_is_annualised:
            mu_period = mu_period / periods_per_year

    cost = np.concatenate([np.zeros(n), [1.0], np.full(s, scale)])
    if mu_period is not None and risk_aversion > 0:
        cost[:n] -= mu_period.to_numpy(dtype=float) / risk_aversion

    # -r_s'w - z - u_s <= 0
    A_ub = np.hstack([-matrix, -np.ones((s, 1)), -np.eye(s)])
    b_ub = np.zeros(s)

    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(s)]).reshape(1, -1)
    net = 1.0 if constraints is None or constraints.net_exposure is None else constraints.net_exposure
    b_eq = np.array([net])

    if constraints is not None:
        for key, members in constraints.groups(assets).items():
            row = np.zeros(n + 1 + s)
            row[np.array(members)] = 1.0
            A_ub = np.vstack([A_ub, row])
            b_ub = np.append(b_ub, constraints.group_limits[key])
        bounds = constraints.bounds(assets) + [(None, None)] + [(0.0, None)] * s
    else:
        bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * s

    if target_return is not None and mu_period is not None:
        # target_return is stated in the same annualised convention as mu.
        target_period = target_return / periods_per_year if mu_is_annualised else target_return
        row = np.zeros(n + 1 + s)
        row[:n] = -mu_period.to_numpy(dtype=float)
        A_ub = np.vstack([A_ub, row])
        b_ub = np.append(b_ub, -target_period)

    result = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"mean-CVaR linear program failed: {result.message}")
    return pd.Series(result.x[:n], index=assets)


def compare_variance_and_cvar(scenarios: pd.DataFrame, covariance: pd.DataFrame,
                              mu: pd.Series, alpha: float = 0.95,
                              risk_aversion: float = 5.0,
                              constraints: Constraints | None = None,
                              periods_per_year: int = 252) -> pd.DataFrame:
    """Do mean-variance and mean-CVaR actually produce different books?

    The research question of spec §52. The answer is usually "less than you
    would hope, but the difference concentrates exactly in the assets with the
    most negative skew", which this table makes visible.
    """
    from .mean_variance import mean_variance_weights

    mv = mean_variance_weights(mu, covariance, risk_aversion, constraints)
    cv = mean_cvar_weights(scenarios, mu, alpha, None, risk_aversion, constraints,
                           mu_is_annualised=True, periods_per_year=periods_per_year)
    assets = list(covariance.columns)
    matrix = scenarios[assets].dropna(how="any")
    from scipy import stats as sp_stats

    return pd.DataFrame(
        {
            "mean_variance_weight": mv.weights.reindex(assets),
            "mean_cvar_weight": cv.reindex(assets),
            "difference": cv.reindex(assets) - mv.weights.reindex(assets),
            "asset_skew": pd.Series(sp_stats.skew(matrix.to_numpy(), axis=0), index=assets),
            "asset_cvar_95": -matrix.apply(lambda c: c[c <= c.quantile(1 - alpha)].mean()),
        }
    )
