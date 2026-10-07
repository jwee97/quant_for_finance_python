"""Vector autoregression and cointegration: VAR, impulse responses, variance decompositions, Johansen, VECM and the single-equation error-correction model.

``fit_var`` estimates ``y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t`` equation by equation with least squares (identical to ML for the unrestricted VAR).
``select_var_order`` compares AIC, BIC and HQ. ``VARResult`` gives forecasts, the stability check (all companion-matrix eigenvalues inside the unit circle),
Cholesky-orthogonalised impulse responses (the ORDER of the variables matters: the first is treated as the most exogenous), the forecast-error variance
decomposition and block Granger-causality Wald tests.

``johansen`` is the Johansen (1988, 1991) reduced-rank test: concentrate out the short-run dynamics, solve the generalised eigenproblem for the squared canonical
correlations, and form the trace and maximum-eigenvalue statistics with Osterwald-Lenum / MacKinnon-Haug-Michelis critical values from ``statsmodels``. ``fit_vecm``
estimates ``dy_t = alpha beta' y_{t-1} + sum Gamma_i dy_{t-i} + c + u_t`` with Johansen's maximum likelihood: ``beta`` spans the cointegrating relationships and
``alpha`` is how fast each variable corrects toward them. ``error_correction_model`` is the Engle-Granger single-equation ECM for a pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


def _lag_design(Y: np.ndarray, p: int, trend: str = "c"):
    n, k = Y.shape
    X = np.column_stack([Y[p - j: n - j] for j in range(1, p + 1)])
    if trend == "c":
        X = np.column_stack([np.ones(len(X)), X])
    elif trend == "ct":
        X = np.column_stack([np.ones(len(X)), np.arange(1, len(X) + 1), X])
    return Y[p:], X


@dataclass
class VARResult:
    names: list
    p: int
    coefs: np.ndarray           # (p, k, k): A_1 .. A_p, rows = equation
    intercept: np.ndarray
    sigma_u: np.ndarray
    resid: pd.DataFrame
    nobs: int
    loglik: float
    aic: float
    bic: float
    hqic: float
    se: pd.DataFrame
    _Y: np.ndarray
    _index: pd.Index

    @property
    def k(self) -> int:
        return len(self.names)

    def companion(self) -> np.ndarray:
        k, p = self.k, self.p
        top = np.hstack(list(self.coefs))
        if p == 1:
            return top
        return np.vstack([top, np.hstack([np.eye(k * (p - 1)), np.zeros((k * (p - 1), k))])])

    def is_stable(self) -> bool:
        return bool(np.max(np.abs(np.linalg.eigvals(self.companion()))) < 1.0)

    def forecast(self, steps: int = 1) -> pd.DataFrame:
        hist = list(self._Y[-self.p:])
        out = []
        for _ in range(steps):
            y = self.intercept + sum(self.coefs[j] @ hist[-1 - j] for j in range(self.p))
            hist.append(y)
            out.append(y)
        return pd.DataFrame(out, columns=self.names, index=range(1, steps + 1))

    def forecast_cov(self, steps: int) -> list[np.ndarray]:
        """MSE matrices of the 1..steps-ahead forecasts from the MA(infinity) weights: ``Sigma_h = sum_{i<h} Phi_i Sigma_u Phi_i'``."""
        phi = self.ma_representation(steps)
        out, acc = [], np.zeros((self.k, self.k))
        for i in range(steps):
            acc = acc + phi[i] @ self.sigma_u @ phi[i].T
            out.append(acc.copy())
        return out

    def ma_representation(self, n: int) -> np.ndarray:
        """``Phi_0 = I, Phi_i = sum_{j=1}^{min(i,p)} A_j Phi_{i-j}``: the response of ``y_{t+i}`` to a unit reduced-form shock at ``t``."""
        phi = np.zeros((n, self.k, self.k))
        phi[0] = np.eye(self.k)
        for i in range(1, n):
            phi[i] = sum(self.coefs[j - 1] @ phi[i - j] for j in range(1, min(i, self.p) + 1))
        return phi

    def irf(self, steps: int = 10, orthogonal: bool = True) -> np.ndarray:
        """Impulse responses, array ``(steps, response, shock)``. ``orthogonal`` uses the Cholesky factor of ``Sigma_u`` (variable order = causal order)."""
        phi = self.ma_representation(steps)
        if not orthogonal:
            return phi
        P = np.linalg.cholesky(self.sigma_u)
        return np.einsum("hij,jk->hik", phi, P)

    def fevd(self, steps: int = 10) -> np.ndarray:
        """Forecast-error variance decomposition: array ``(steps, variable, shock)`` of the share of each variable's forecast variance due to each orthogonal shock."""
        theta = self.irf(steps)
        sq = np.cumsum(theta ** 2, axis=0)
        return sq / sq.sum(axis=2, keepdims=True)

    def granger(self, caused: str, causing: list[str] | str) -> dict:
        """Wald test that all lags of ``causing`` are zero in the equation for ``caused`` (small-sample F version)."""
        causing = [causing] if isinstance(causing, str) else causing
        i = self.names.index(caused)
        js = [self.names.index(c) for c in causing]
        n_c = int(self._X_has_const())
        n_obs = len(self._Y) - self.p
        _, X = _lag_design(self._Y, self.p, "c" if n_c else "n")
        beta = np.concatenate([[self.intercept[i]] if n_c else [], np.concatenate([self.coefs[l][i] for l in range(self.p)])])
        XtXi = np.linalg.inv(X.T @ X)
        idx = [n_c + l * self.k + j for l in range(self.p) for j in js]
        V = self.sigma_u[i, i] * XtXi[np.ix_(idx, idx)]
        b = beta[idx]
        wald = float(b @ np.linalg.solve(V, b))
        q = len(idx)
        dof = n_obs - X.shape[1]
        return {"chi2": wald, "chi2_pvalue": float(sps.chi2.sf(wald, q)), "F": wald / q, "F_pvalue": float(sps.f.sf(wald / q, q, dof)), "df": q}

    def _X_has_const(self) -> bool:
        return True


def fit_var(data: pd.DataFrame, p: int = 1) -> VARResult:
    """Unrestricted VAR(p) with a constant, estimated by OLS on every equation."""
    d = data.dropna()
    Y = d.to_numpy(float)
    k = Y.shape[1]
    y, X = _lag_design(Y, p, "c")
    B = np.linalg.lstsq(X, y, rcond=None)[0]              # (1 + k p, k)
    resid = y - X @ B
    n = len(y)
    sigma = resid.T @ resid / (n - X.shape[1])             # degrees-of-freedom corrected, as statsmodels
    sigma_ml = resid.T @ resid / n
    coefs = np.stack([B[1 + j * k: 1 + (j + 1) * k].T for j in range(p)])
    ll = -0.5 * n * (k * np.log(2 * np.pi) + np.linalg.slogdet(sigma_ml)[1] + k)
    npar = k * (k * p + 1)
    se = np.sqrt(np.outer(np.diag(np.linalg.inv(X.T @ X)), np.diag(sigma)))
    cols = ["const"] + [f"{c}.L{j}" for j in range(1, p + 1) for c in d.columns]
    return VARResult(list(d.columns), p, coefs, B[0], sigma, pd.DataFrame(resid, index=d.index[p:], columns=d.columns), n, float(ll), -2 * ll + 2 * npar,
                     -2 * ll + np.log(n) * npar, -2 * ll + 2 * np.log(np.log(n)) * npar, pd.DataFrame(se, index=cols, columns=d.columns), Y, d.index)


def select_var_order(data: pd.DataFrame, max_lag: int = 8) -> pd.DataFrame:
    """AIC / BIC / HQ for lag orders 1..max_lag, all computed on the SAME sample (the last ``n - max_lag`` rows) so they are comparable."""
    d = data.dropna()
    Y = d.to_numpy(float)
    k = Y.shape[1]
    rows = []
    for p in range(1, max_lag + 1):
        y = Y[max_lag:]
        X = np.column_stack([np.ones(len(y))] + [Y[max_lag - j: len(Y) - j] for j in range(1, p + 1)])
        B = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ B
        n = len(y)
        sig = np.linalg.slogdet(r.T @ r / n)[1]
        npar = p * k * k + k
        rows.append({"p": p, "aic": sig + 2 * npar / n, "bic": sig + np.log(n) * npar / n, "hqic": sig + 2 * np.log(np.log(n)) * npar / n})
    return pd.DataFrame(rows).set_index("p")


# --------------------------------------------------------------------------------------------------------------------- Johansen
def _johansen_core(Y: np.ndarray, k_ar_diff: int, det: str):
    """Concentrated regressions and the eigenproblem. ``det``: ``n`` none, ``co`` an unrestricted constant (a drift in the differences, so linear trends in levels),
    ``ci`` a constant restricted to the cointegrating relation (no trends in levels)."""
    n, k = Y.shape
    dY = np.diff(Y, axis=0)
    p = k_ar_diff
    Z0 = dY[p:]
    Zl = Y[p:-1]
    cols = [dY[p - j: len(dY) - j] for j in range(1, p + 1)]
    W = np.column_stack(cols) if cols else np.empty((len(Z0), 0))
    if det == "co":
        W = np.column_stack([np.ones(len(Z0)), W]) if W.size else np.ones((len(Z0), 1))
    Z1 = np.column_stack([Zl, np.ones(len(Zl))]) if det == "ci" else Zl
    if W.shape[1]:
        M = np.eye(len(W)) - W @ np.linalg.pinv(W)
        R0, R1 = M @ Z0, M @ Z1
    else:
        R0, R1 = Z0 - Z0.mean(axis=0) * 0, Z1
    T = len(R0)
    S00, S11, S01 = R0.T @ R0 / T, R1.T @ R1 / T, R0.T @ R1 / T
    S10 = S01.T
    L = np.linalg.cholesky(S11)
    Li = np.linalg.inv(L)
    A = Li @ S10 @ np.linalg.inv(S00) @ S01 @ Li.T
    A = 0.5 * (A + A.T)
    eigval, eigvec = np.linalg.eigh(A)
    order = np.argsort(-eigval)
    eigval = eigval[order]
    V = Li.T @ eigvec[:, order]                                # cointegrating vectors, normalised V' S11 V = I
    return eigval, V, S00, S11, S01, R0, R1, T, W, Z0, Z1


def johansen(data: pd.DataFrame, k_ar_diff: int = 1, det_order: int = 0) -> dict:
    """Johansen trace and max-eigenvalue tests for the cointegration rank of the columns of ``data``.

    ``k_ar_diff`` lagged differences; ``det_order`` follows ``statsmodels``: ``-1`` no deterministic term, ``0`` a constant (unrestricted: drift in the
    differences). Returns the eigenvalues, the statistics with their 90/95/99% critical values (from ``statsmodels``), the selected rank at 5% (trace), and the
    cointegrating vectors ``beta`` normalised on the first variable.
    """
    from statsmodels.tsa.coint_tables import c_sja, c_sjt

    Y = data.dropna().to_numpy(float)
    n, k = Y.shape
    if det_order not in (-1, 0):
        raise ValueError("det_order must be -1 (none) or 0 (constant)")
    eigval, V, *_ = _johansen_core(Y, k_ar_diff, "n" if det_order == -1 else "co")
    eigval, V = eigval[:k], V[:, :k]
    T = n - k_ar_diff - 1
    lr1 = np.array([-T * np.sum(np.log(1 - eigval[i:])) for i in range(k)])
    lr2 = np.array([-T * np.log(1 - eigval[i]) for i in range(k)])
    cvt = np.array([c_sjt(k - i, det_order) for i in range(k)])
    cvm = np.array([c_sja(k - i, det_order) for i in range(k)])
    rank = 0
    for i in range(k):
        if lr1[i] > cvt[i, 1]:
            rank = i + 1
        else:
            break
    beta = V[:k, :] / V[0, :]
    return {"eigenvalues": eigval, "trace": lr1, "trace_cv": cvt, "max_eig": lr2, "max_eig_cv": cvm, "rank": rank, "beta": pd.DataFrame(beta, index=data.columns),
            "n_obs": T}


@dataclass
class VECMResult:
    names: list
    rank: int
    alpha: pd.DataFrame
    beta: pd.DataFrame
    gamma: np.ndarray
    sigma_u: np.ndarray
    loglik: float
    resid: pd.DataFrame
    det: str
    k_ar_diff: int
    _Y: np.ndarray

    @property
    def pi(self) -> np.ndarray:
        return self.alpha.to_numpy() @ self.beta.to_numpy().T

    def error_correction_term(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """The cointegrating relation(s) ``beta' y_t`` (the spreads): stationary if the rank is right."""
        Y = self._Y if data is None else data.to_numpy(float)
        B = self.beta.to_numpy()
        ect = Y @ B[: len(self.names)]
        if self.det == "ci":
            ect = ect + B[len(self.names):]
        idx = data.index if data is not None else None
        return pd.DataFrame(ect, index=idx, columns=self.beta.columns)

    def to_var(self) -> np.ndarray:
        """The levels-VAR coefficient matrices ``A_1..A_{p+1}`` implied by the VECM."""
        k = len(self.names)
        G = [self.gamma[i] for i in range(self.k_ar_diff)]
        A = []
        pi = self.pi[:, :k] if self.pi.shape[1] > k else self.pi
        A.append(np.eye(k) + pi + (G[0] if G else 0))
        for i in range(1, self.k_ar_diff):
            A.append(G[i] - G[i - 1])
        if self.k_ar_diff:
            A.append(-G[-1])
        return np.stack(A)


def fit_vecm(data: pd.DataFrame, rank: int = 1, k_ar_diff: int = 1, deterministic: str = "ci") -> VECMResult:
    """Johansen maximum-likelihood VECM. ``beta`` (normalised so its top ``rank x rank`` block is the identity) and ``alpha`` (adjustment speeds) are returned
    with the short-run matrices ``Gamma_i`` and the residual covariance. ``deterministic``: ``n`` none, ``co`` unrestricted constant, ``ci`` constant inside the
    cointegrating relation (the natural choice for price levels with no trend)."""
    d = data.dropna()
    Y = d.to_numpy(float)
    n, k = Y.shape
    eigval, V, S00, S11, S01, R0, R1, T, W, Z0, Z1 = _johansen_core(Y, k_ar_diff, deterministic)
    beta = V[:, :rank]
    top = beta[:rank]
    beta = beta @ np.linalg.inv(top)
    alpha = S01 @ beta @ np.linalg.inv(beta.T @ S11 @ beta)
    # short-run terms by regressing the residual dy - alpha beta' y_lag on the lagged differences (and unrestricted constant)
    ect = Z1 @ beta
    target = Z0 - ect @ alpha.T
    if W.shape[1]:
        coef = np.linalg.lstsq(W, target, rcond=None)[0]
        resid = target - W @ coef
    else:
        coef, resid = np.zeros((0, k)), target
    off = 1 if deterministic == "co" else 0
    gamma = np.stack([coef[off + j * k: off + (j + 1) * k].T for j in range(k_ar_diff)]) if k_ar_diff else np.zeros((0, k, k))
    sigma = resid.T @ resid / len(resid)
    ll = -0.5 * len(resid) * (k * np.log(2 * np.pi) + np.linalg.slogdet(sigma)[1] + k)
    bnames = [f"beta{i+1}" for i in range(rank)]
    rows = list(d.columns) + (["const"] if deterministic == "ci" else [])
    return VECMResult(list(d.columns), rank, pd.DataFrame(alpha, index=d.columns, columns=bnames), pd.DataFrame(beta, index=rows, columns=bnames), gamma, sigma, float(ll),
                      pd.DataFrame(resid, index=d.index[k_ar_diff + 1:]), deterministic, k_ar_diff, Y)


def error_correction_model(y: pd.Series, x: pd.Series, lags: int = 1) -> dict:
    """Engle-Granger single-equation ECM for a cointegrated pair: ``dy_t = a + g ECT_{t-1} + sum b_i dx_{t-i} + sum c_i dy_{t-i} + e``, ``ECT = y - h x - c`` from the
    levels regression. ``g`` should be NEGATIVE (the pair corrects toward equilibrium); the half-life of a deviation is ``ln 0.5 / ln(1 + g)``."""
    from ..stats.regression import ols

    f = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    lev = ols(f["y"], f[["x"]])
    ect = f["y"] - lev.params["const"] - lev.params["x"] * f["x"]
    dy, dx = f["y"].diff(), f["x"].diff()
    X = pd.DataFrame({"ect_lag": ect.shift(1)})
    for i in range(lags + 1):
        if i:
            X[f"dy_l{i}"] = dy.shift(i)
        X[f"dx_l{i}"] = dx.shift(i)
    res = ols(dy, X, "HAC")
    g = res.params["ect_lag"]
    return {"hedge_ratio": float(lev.params["x"]), "intercept": float(lev.params["const"]), "speed": float(g), "t_speed": float(res.tvalues["ect_lag"]),
            "half_life": float(np.log(0.5) / np.log(1.0 + g)) if -1 < g < 0 else float("inf"), "ect": ect, "model": res}
