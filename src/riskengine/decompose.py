"""Where a portfolio's risk comes from.

With ``h`` the holdings, ``b = B' h`` the factor exposures, ``Sigma = B F B' + D``:

    variance  =  b' F b   (factor)   +   sum_i D_i h_i^2   (specific)

and the *contribution* of factor ``k`` to volatility is ``b_k (F b)_k / sigma``: these, and the specific part ``sum D h^2 / sigma``, add exactly to ``sigma``. A contribution is the change in volatility
for a proportional change in that exposure, not a quantity that can be removed without moving the rest: hedging one factor changes the others' contributions unless they are uncorrelated.

Factors are reported in groups (market, sector, style, statistical); the asset contributions ``h_i (Sigma h)_i / sigma`` add to ``sigma`` too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .factor_model import FactorModel


def _h(weights: pd.Series, model: FactorModel) -> pd.Series:
    return weights.reindex(model.assets).fillna(0.0)


def exposures(weights: pd.Series, model: FactorModel) -> pd.Series:
    """The portfolio's exposure to each factor: ``B' h``."""
    return pd.Series(model.exposures.T.to_numpy(dtype=float) @ _h(weights, model).to_numpy(dtype=float), index=model.factors)


def risk_decomposition(weights: pd.Series, model: FactorModel, annualise: float | None = None) -> dict:
    """Total volatility and its parts: ``factors`` (each factor's contribution), ``groups`` (the sums by group, plus ``specific``), ``assets`` (component contributions), shares of variance, and the
    marginal contribution of each asset (``d sigma / d h_i``). The contributions of factors plus specific equal the total, to rounding."""
    h = _h(weights, model).to_numpy(dtype=float)
    B = model.exposures.to_numpy(dtype=float)
    F = model.factor_cov.to_numpy(dtype=float)
    D = model.specific_var.reindex(model.assets).to_numpy(dtype=float)
    b = B.T @ h
    fv, sv = float(b @ F @ b), float((D * h ** 2).sum())
    var = fv + sv
    scale = np.sqrt(annualise) if annualise else 1.0
    sigma = np.sqrt(var)
    if sigma <= 0:
        z = pd.Series(0.0, index=model.factors)
        return {"volatility": 0.0, "factor_share": float("nan"), "specific_share": float("nan"), "exposures": exposures(weights, model), "factors": z, "specific": 0.0, "groups": pd.Series(dtype=float),
                "assets": pd.Series(0.0, index=model.assets), "marginal": pd.Series(0.0, index=model.assets)}
    fc = pd.Series(b * (F @ b) / sigma * scale, index=model.factors)
    spec = sv / sigma * scale
    groups = fc.groupby(pd.Series(model.groups).reindex(model.factors).fillna("other")).sum()
    groups["specific"] = spec
    S = B @ F @ B.T + np.diag(D)
    sh = S @ h
    return {"volatility": sigma * scale, "factor_share": fv / var, "specific_share": sv / var, "exposures": pd.Series(b, index=model.factors), "factors": fc, "specific": spec, "groups": groups,
            "assets": pd.Series(h * sh / sigma * scale, index=model.assets), "marginal": pd.Series(sh / sigma * scale, index=model.assets)}


def active_risk(weights: pd.Series, benchmark: pd.Series, model: FactorModel, annualise: float | None = None) -> dict:
    """The decomposition of the risk of the active holdings ``h - benchmark`` (tracking error), plus the active exposures."""
    active = _h(weights, model) - _h(benchmark, model)
    out = risk_decomposition(active, model, annualise)
    out["active_weights"] = active
    return out


def factor_risk_budget_weights(model: FactorModel, budget: dict, long_only: bool = True, max_weight: float = 1.0, start: pd.Series | None = None) -> dict:
    """Weights whose risk contributions follow a budget over groups: ``budget`` maps group names (``market``, ``sector``, ``style``, ``statistical`` and ``specific``) to the share of volatility each should
    contribute. Solved by minimising the squared distance of the realised shares from the budget with the weights summing to one; a budget that the universe cannot reach (a style group with no style
    factors) is refused. Returns the weights, the realised shares and whether the optimiser converged."""
    present = set(pd.Series(model.groups).reindex(model.factors).dropna()) | {"specific"}
    missing = [g for g in budget if g not in present]
    if missing:
        raise ValueError(f"the model has no factors in group(s) {', '.join(missing)}")
    target = pd.Series(budget, dtype=float)
    if (target < 0).any() or target.sum() <= 0:
        raise ValueError("budget shares must be non-negative and not all zero")
    target = target / target.sum()
    n = len(model.assets)
    x0 = (start.reindex(model.assets).fillna(0.0).to_numpy(dtype=float) if start is not None else np.full(n, 1.0 / n))

    def shares(h):
        d = risk_decomposition(pd.Series(h, index=model.assets), model)
        if d["volatility"] <= 0:
            return pd.Series(0.0, index=target.index)
        return (d["groups"].reindex(target.index).fillna(0.0) / d["volatility"])

    def loss(h):
        return float(((shares(h) - target) ** 2).sum())

    bounds = [(0.0 if long_only else -max_weight, max_weight)] * n
    res = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=[{"type": "eq", "fun": lambda h: float(h.sum() - 1.0)}], options={"maxiter": 500, "ftol": 1e-14})
    w = pd.Series(res.x, index=model.assets)
    return {"weights": w, "shares": shares(res.x), "target": target, "loss": float(res.fun), "converged": bool(res.success)}
