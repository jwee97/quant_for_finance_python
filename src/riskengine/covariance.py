"""Two covariance estimators the repository did not have: detoned correlation and the graphical lasso.

**Detoning** (López de Prado): the first eigenvector of an equity correlation matrix is the market, and it dominates every distance computed from it, so that clusters are the 'market' and everything else.
Remove the leading ``n`` components, ``C - V_n L_n V_n'``, and rescale to a unit diagonal. The result is no longer positive definite: use it to *cluster* (as in HERC and NCO), not to invert.

**Graphical lasso** (Friedman, Hastie and Tibshirani 2008): the covariance whose *inverse* is sparse, found by maximising the Gaussian likelihood minus ``alpha`` times the absolute off-diagonal of the
precision matrix. A zero in the precision matrix is a conditional independence: two assets that move together only through a third have no edge. Estimated on the correlation scale, so that the penalty
treats every asset alike, then rescaled to the original volatilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detone_correlation(corr: pd.DataFrame, n_market: int = 1) -> pd.DataFrame:
    """The correlation matrix with its ``n_market`` leading eigencomponents removed and the diagonal restored to one."""
    c = corr.to_numpy(dtype=float)
    values, vectors = np.linalg.eigh(c)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    n_market = int(min(max(n_market, 0), len(values) - 1))
    out = c - (vectors[:, :n_market] * values[:n_market]) @ vectors[:, :n_market].T
    d = np.sqrt(np.clip(np.diag(out), 1e-12, None))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return pd.DataFrame(0.5 * (out + out.T), index=corr.index, columns=corr.columns)


def detoned_covariance(returns: pd.DataFrame, n_market: int = 1, annualise: bool = True) -> pd.DataFrame:
    """The detoned correlation scaled back by the assets' own volatilities (rank-deficient by construction)."""
    clean = returns.dropna(how="any")
    cov = clean.cov(ddof=1)
    vol = np.sqrt(np.diag(cov.to_numpy()))
    corr = pd.DataFrame(cov.to_numpy() / np.outer(vol, vol), index=cov.index, columns=cov.columns)
    out = detone_correlation(corr, n_market)
    out = pd.DataFrame(out.to_numpy() * np.outer(vol, vol), index=cov.index, columns=cov.columns)
    return out * 252.0 if annualise else out


def graphical_lasso_covariance(returns: pd.DataFrame, alpha: float | None = None, annualise: bool = True, max_iter: int = 200, cv: int = 3) -> dict:
    """The graphical-lasso covariance. ``alpha=None`` picks the penalty by cross-validated likelihood. Returns the ``covariance``, the sparse ``precision``, the ``alpha`` used, and the share of
    off-diagonal precision entries that are zero (``sparsity``)."""
    from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

    clean = returns.dropna(how="any")
    if len(clean) < 30 or clean.shape[1] < 2:
        raise ValueError("need at least 30 complete rows and 2 assets")
    sd = clean.std(ddof=1).to_numpy()
    z = (clean - clean.mean()).to_numpy() / sd
    model = GraphicalLassoCV(cv=cv, max_iter=max_iter) if alpha is None else GraphicalLasso(alpha=float(alpha), max_iter=max_iter)
    model.fit(z)
    cov = model.covariance_ * np.outer(sd, sd)
    prec = model.precision_ / np.outer(sd, sd)
    off = ~np.eye(len(sd), dtype=bool)
    scale = 252.0 if annualise else 1.0
    idx = clean.columns
    return {"covariance": pd.DataFrame(cov * scale, index=idx, columns=idx), "precision": pd.DataFrame(prec / scale, index=idx, columns=idx), "alpha": float(getattr(model, "alpha_", alpha)),
            "sparsity": float((np.abs(model.precision_[off]) < 1e-8).mean())}
