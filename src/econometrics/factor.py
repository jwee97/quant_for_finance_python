"""Dynamic factor models and Bayesian VARs.

**Dynamic factor model** (Stock & Watson 2002; Doz, Giannone & Reichlin 2011): many series ``x_t`` (``N`` of them) are driven by a few common factors ``f_t``
(``k << N``) that follow a VAR, plus idiosyncratic noise::

    x_t = Lambda f_t + e_t,   e_t ~ N(0, diag(R)),      f_t = A_1 f_{t-1} + ... + A_p f_{t-p} + u_t,   u_t ~ N(0, Q)

``fit_dynamic_factor`` uses the two-step estimator: principal components give the loadings and a first factor estimate, a VAR on the factors gives ``A, Q``, the
residuals give ``R``, and the Kalman filter then re-estimates the factors, which also copes with missing data and a ragged edge (some series published later than
others) that PCA cannot. ``filtered_factors`` use data up to ``t`` only. ``bai_ng_criteria`` chooses ``k`` (Bai & Ng 2002).

**Minnesota BVAR** (Litterman 1986; Banbura, Giannone & Reichlin 2010): a VAR with many parameters and few observations overfits, so the prior shrinks every equation toward
a random walk (own first lag near one, other coefficients and longer lags near zero, with tightness ``lam``). It is implemented with dummy observations, so the
posterior mean is OLS on the data augmented with the prior; ``lam -> infinity`` recovers OLS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .statespace import StateSpace
from .var import VARResult


# ---------------------------------------------------------------------------------------------------------------- dynamic factors
def bai_ng_criteria(data: pd.DataFrame, k_max: int = 8) -> pd.DataFrame:
    """Bai-Ng (2002) information criteria IC1-IC3 and the share of variance explained by each number of principal components of the standardised data."""
    X = data.dropna()
    Z = (X - X.mean()) / X.std()
    T, N = Z.shape
    u, s, vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    total = float((s ** 2).sum())
    rows = []
    for k in range(1, k_max + 1):
        V = float(((s[k:]) ** 2).sum()) / (N * T)
        pen = (N + T) / (N * T)
        rows.append({"k": k, "ic1": np.log(V) + k * pen * np.log(N * T / (N + T)), "ic2": np.log(V) + k * pen * np.log(min(N, T)),
                     "ic3": np.log(V) + k * np.log(min(N, T)) / min(N, T), "variance_share": float((s[:k] ** 2).sum() / total)})
    return pd.DataFrame(rows).set_index("k")


@dataclass
class DynamicFactorResult:
    columns: list
    n_factors: int
    factor_order: int
    loadings: pd.DataFrame
    transition: np.ndarray
    factor_cov: np.ndarray
    idio_var: pd.Series
    mean: pd.Series
    std: pd.Series
    filtered_factors: pd.DataFrame
    smoothed_factors: pd.DataFrame
    pca_factors: pd.DataFrame
    model: StateSpace
    loglik: float

    def forecast_factors(self, steps: int = 1) -> pd.DataFrame:
        """Iterate the factor VAR forward from the last filtered factors."""
        k = self.n_factors
        hist = [self.filtered_factors.iloc[-1 - j].to_numpy() for j in range(self.factor_order)]
        out = []
        for _ in range(steps):
            nxt = sum(self.transition[:k, j * k: (j + 1) * k] @ hist[j] for j in range(self.factor_order))
            hist = [nxt] + hist[:-1]
            out.append(nxt)
        return pd.DataFrame(out, columns=self.filtered_factors.columns, index=range(1, steps + 1))

    def forecast(self, steps: int = 1) -> pd.DataFrame:
        """Forecast of each series: loadings times forecast factors, in the original units."""
        f = self.forecast_factors(steps)
        z = f.to_numpy() @ self.loadings.to_numpy().T
        return pd.DataFrame(z * self.std.to_numpy() + self.mean.to_numpy(), columns=self.columns, index=f.index)

    def common_component(self) -> pd.DataFrame:
        z = self.filtered_factors.to_numpy() @ self.loadings.to_numpy().T
        return pd.DataFrame(z * self.std.to_numpy() + self.mean.to_numpy(), index=self.filtered_factors.index, columns=self.columns)


def fit_dynamic_factor(data: pd.DataFrame, n_factors: int = 2, factor_order: int = 1) -> DynamicFactorResult:
    """Two-step dynamic factor model with Kalman-filtered factors (missing values allowed; the PCA step uses the complete rows, or a mean-filled panel when none exist)."""
    X = data.astype(float)
    mean, std = X.mean(), X.std()
    Z = (X - mean) / std
    complete = Z.dropna()
    base = complete if len(complete) >= max(50, 5 * n_factors) else Z.fillna(0.0)
    u, s, vt = np.linalg.svd(base.to_numpy(), full_matrices=False)
    k, N = n_factors, Z.shape[1]
    F = u[:, :k] * np.sqrt(len(base))
    Lam = vt[:k].T * s[:k] / np.sqrt(len(base))
    # rotate so that factors have unit variance (identification): F = u sqrt(T), Lambda = V s / sqrt(T)
    pca_f = pd.DataFrame(F, index=base.index, columns=[f"f{i+1}" for i in range(k)])
    # VAR on the factors
    p = factor_order
    Y = F
    Xl = np.column_stack([Y[p - j: len(Y) - j] for j in range(1, p + 1)])
    B = np.linalg.lstsq(Xl, Y[p:], rcond=None)[0]
    A = [B[j * k: (j + 1) * k].T for j in range(p)]
    Qm = np.cov((Y[p:] - Xl @ B).T).reshape(k, k)
    resid = base.to_numpy() - F @ Lam.T
    Rv = np.clip(resid.var(axis=0), 1e-4, None)
    # companion form
    r = k * p
    T = np.zeros((r, r))
    T[:k, :] = np.hstack(A)
    if p > 1:
        T[k:, :-k] = np.eye(k * (p - 1))
    Zmat = np.zeros((N, r))
    Zmat[:, :k] = Lam
    Rsel = np.zeros((r, k))
    Rsel[:k] = np.eye(k)
    model = StateSpace(T, Zmat, np.diag(Rv), Qm, R=Rsel)
    z = Z.to_numpy()
    f = model.filter(z)
    sm = model.smooth(z)
    idx = Z.index
    cols = [f"f{i+1}" for i in range(k)]
    return DynamicFactorResult(list(X.columns), k, p, pd.DataFrame(Lam, index=X.columns, columns=cols), T, Qm, pd.Series(Rv, index=X.columns), mean, std,
                               pd.DataFrame(f.filtered_state[:, :k], index=idx, columns=cols), pd.DataFrame(sm["smoothed_state"][:, :k], index=idx, columns=cols), pca_f, model,
                               f.loglik)


def nowcast(result: DynamicFactorResult, series: str, data: pd.DataFrame) -> pd.Series:
    """Fill the missing recent values of ``series`` from the common component (the ragged-edge nowcast): the filtered factors at each date times that series' loadings."""
    cc = result.common_component()[series]
    s = data[series]
    return s.where(s.notna(), cc)


# ---------------------------------------------------------------------------------------------------------------------- BVAR
def fit_bvar(data: pd.DataFrame, p: int = 2, lam: float = 0.2, theta: float = 1.0, decay: float = 1.0, sum_coef_mu: float = 0.0, co_persistence: float = 0.0,
             random_walk_mean: float = 1.0) -> VARResult:
    """Minnesota-prior Bayesian VAR(p) with dummy observations (Banbura, Giannone & Reichlin 2010).

    ``lam`` overall tightness (smaller = more shrinkage toward the prior; ``lam -> inf`` is OLS), ``theta`` cross-variable tightness (1 = same), ``decay`` lag decay
    exponent, ``sum_coef_mu`` the sum-of-coefficients prior weight (inertia: large = more unit-root-like) and ``co_persistence`` the dummy-initial-observation weight.
    ``random_walk_mean`` is the prior mean of the own first lag (1 for levels / prices, 0 for returns). Scale is set from AR(1) residual variances.
    """
    d = data.dropna()
    Y = d.to_numpy(float)
    n, k = Y.shape
    sig = np.array([np.std(np.diff(Y[:, i])) if random_walk_mean == 1.0 else np.std(Y[:, i]) for i in range(k)])
    sig = np.where(sig > 0, sig, 1.0)
    y, X = Y[p:], np.column_stack([np.ones(n - p)] + [Y[p - j: n - j] for j in range(1, p + 1)])
    Yd, Xd = [], []
    if np.isfinite(lam):
        # shrinkage dummies on the lag coefficients
        for j in range(1, p + 1):
            blk = np.diag(sig) * (j ** decay) / lam
            Yd_j = np.zeros((k, k))
            if j == 1:
                Yd_j = np.diag(sig * random_walk_mean) / lam
            Xd_j = np.zeros((k, 1 + k * p))
            Xd_j[:, 1 + (j - 1) * k: 1 + j * k] = blk
            if theta != 1.0:
                off = np.ones((k, k)) * (1 / theta) - (1 / theta - 1) * np.eye(k)
                Xd_j[:, 1 + (j - 1) * k: 1 + j * k] = blk * off
            Yd.append(Yd_j)
            Xd.append(Xd_j)
        # residual-covariance dummies (keep Sigma at its scale)
        Yd.append(np.diag(sig))
        Xd.append(np.zeros((k, 1 + k * p)))
        # intercept: loose
        Yd.append(np.zeros((1, k)))
        Xd.append(np.array([[1e-6] + [0.0] * (k * p)]))
    if sum_coef_mu > 0:
        ybar = Y[:p].mean(axis=0)
        Yd.append(np.diag(ybar) * sum_coef_mu)
        Xd.append(np.hstack([np.zeros((k, 1))] + [np.diag(ybar) * sum_coef_mu for _ in range(p)]))
    if co_persistence > 0:
        ybar = Y[:p].mean(axis=0)
        Yd.append((ybar * co_persistence)[None, :])
        Xd.append(np.hstack([[co_persistence], np.tile(ybar * co_persistence, p)])[None, :])
    if Yd:
        ya, xa = np.vstack([y] + Yd), np.vstack([X] + Xd)
    else:
        ya, xa = y, X
    B = np.linalg.lstsq(xa, ya, rcond=None)[0]
    resid = y - X @ B
    sigma = resid.T @ resid / (len(y) - X.shape[1])
    coefs = np.stack([B[1 + j * k: 1 + (j + 1) * k].T for j in range(p)])
    sigma_ml = resid.T @ resid / len(y)
    ll = -0.5 * len(y) * (k * np.log(2 * np.pi) + np.linalg.slogdet(sigma_ml)[1] + k)
    npar = k * (k * p + 1)
    cols = ["const"] + [f"{c}.L{j}" for j in range(1, p + 1) for c in d.columns]
    se = pd.DataFrame(np.nan, index=cols, columns=d.columns)
    return VARResult(list(d.columns), p, coefs, B[0], sigma, pd.DataFrame(resid, index=d.index[p:], columns=d.columns), len(y), float(ll), -2 * ll + 2 * npar,
                     -2 * ll + np.log(len(y)) * npar, -2 * ll + 2 * np.log(np.log(len(y))) * npar, se, Y, d.index)
