"""Neutralising and hedging.

**Factor-neutral holdings** (:func:`neutralise`): the change ``x`` to the holdings that sets the chosen exposures to zero, ``B' (h + x) = 0``. With the ``risk`` metric it is the change of smallest
variance, ``x'Sigma x``, which keeps the neutral portfolio as close to the original, in risk terms, as the constraints allow: ``x = -Sigma^-1 B (B' Sigma^-1 B)^-1 B' h``. With the ``norm`` metric it is the
smallest in squared weights. Either leaves exactly zero exposure to the neutralised factors. Exposures to the other factors change only as far as the adjustment, which trades assets that carry them too, requires.

**Minimum-variance hedge** (:func:`min_variance_hedge`): positions in named hedge instruments (an index future, a sector ETF) that minimise the variance of the whole, ``h_H = -Sigma_HH^-1 Sigma_Hp``; the
reduction in variance is the hedge effectiveness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_model import FactorModel


def neutralise(weights: pd.Series, model: FactorModel, factors: list | None = None, metric: str = "risk") -> dict:
    """Holdings with zero exposure to ``factors`` (all of them when ``None``), changed as little as possible in the ``metric``: ``risk`` (the variance of the change) or ``norm`` (its squared weights)."""
    names = list(factors) if factors is not None else model.factors
    bad = [f for f in names if f not in model.factors]
    if bad:
        raise ValueError(f"unknown factor(s): {', '.join(map(str, bad))}")
    if metric not in ("risk", "norm"):
        raise ValueError("metric must be 'risk' or 'norm'")
    h = weights.reindex(model.assets).fillna(0.0).to_numpy(dtype=float)
    Bn = model.exposures[names].to_numpy(dtype=float)
    M = model.covariance().to_numpy(dtype=float) if metric == "risk" else np.eye(len(h))
    Mi_B = np.linalg.solve(M + 1e-12 * np.trace(M) / len(h) * np.eye(len(h)), Bn)
    x = -Mi_B @ np.linalg.solve(Bn.T @ Mi_B + 1e-12 * np.eye(len(names)), Bn.T @ h)
    new = pd.Series(h + x, index=model.assets)
    S = model.covariance().to_numpy(dtype=float)
    return {"weights": new, "trade": pd.Series(x, index=model.assets), "exposures_before": pd.Series(model.exposures.T.to_numpy() @ h, index=model.factors),
            "exposures_after": pd.Series(model.exposures.T.to_numpy() @ new.to_numpy(), index=model.factors), "volatility_before": float(np.sqrt(h @ S @ h)), "volatility_after": float(np.sqrt(new.to_numpy() @ S @ new.to_numpy())),
            "turnover": float(np.abs(x).sum())}


def min_variance_hedge(weights: pd.Series, cov: pd.DataFrame, hedge_assets: list, ridge: float = 1e-10) -> dict:
    """The positions in ``hedge_assets`` (columns of ``cov``) that minimise the variance of ``weights`` plus the hedge. ``hedge_assets`` may be in the portfolio already: the hedge is then added to what
    is held."""
    assets = list(cov.columns)
    bad = [a for a in hedge_assets if a not in assets]
    if bad or not hedge_assets:
        raise ValueError("hedge assets must be a non-empty list of names in the covariance matrix" + (f" (missing: {', '.join(map(str, bad))})" if bad else ""))
    S = cov.to_numpy(dtype=float)
    h = weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    idx = [assets.index(a) for a in hedge_assets]
    S_HH = S[np.ix_(idx, idx)]
    S_Hp = S[idx] @ h
    pos = -np.linalg.solve(S_HH + ridge * np.trace(S_HH) / len(idx) * np.eye(len(idx)), S_Hp)
    after = h.copy()
    after[idx] += pos
    v0, v1 = float(h @ S @ h), float(after @ S @ after)
    return {"hedge": pd.Series(pos, index=hedge_assets), "weights": pd.Series(after, index=assets), "variance_before": v0, "variance_after": v1, "effectiveness": 1.0 - v1 / v0 if v0 > 0 else float("nan")}
