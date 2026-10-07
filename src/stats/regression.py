"""Linear regression with the covariance estimators finance needs, written out so every formula is visible.

``ols`` is the workhorse. The coefficients are always ``(X'X)^-1 X'y``; what differs is the standard error, because returns are heteroskedastic,
overlapping forecast horizons make residuals autocorrelated, and observations on one date or one asset are not independent:

==============  ======================================================================  ===============================
``cov``         estimator                                                               use when
==============  ======================================================================  ===============================
``nonrobust``   ``s^2 (X'X)^-1``                                                        textbook i.i.d. errors
``HC0..HC3``    White (1980) sandwich, with the MacKinnon-White leverage corrections    heteroskedasticity only
``HAC``         Newey-West (1987): Bartlett-weighted autocovariances of the scores      overlapping returns, autocorrelation
``cluster``     Liang-Zeger one-way cluster-robust                                      groups with within-group correlation
``cluster2``    Cameron-Gelbach-Miller (2011) two-way cluster                           correlation by asset AND by date
==============  ======================================================================  ===============================

``wls`` and ``gls`` handle known heteroskedasticity or correlation; ``feasible_gls_ar1`` is Prais-Winsten for AR(1) errors. ``rolling_ols`` fits a window
at every date in one batched solve. ``diagnostics`` collects the residual tests (Durbin-Watson, Breusch-Godfrey, Breusch-Pagan, White, Jarque-Bera,
Ramsey RESET, variance inflation factors).

All results are validated against ``statsmodels`` in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps


def newey_west_lag(n: int) -> int:
    """The Newey-West (1994) rule of thumb ``floor(4 (n/100)^(2/9))``."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _prepare(y, X, add_constant: bool, extra=None):
    """Align, drop incomplete rows, return (names, index, y, X, extra rows)."""
    y = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float).ravel())
    if isinstance(X, pd.Series):
        X = X.to_frame()
    elif not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X, dtype=float).reshape(len(y), -1), index=y.index, columns=[f"x{i}" for i in range(np.asarray(X).reshape(len(y), -1).shape[1])])
    frame = pd.concat([y.rename("__y__"), X], axis=1)
    if extra is not None:
        frame = frame.join(pd.Series(extra, index=y.index, name="__extra__") if not isinstance(extra, pd.Series) else extra.rename("__extra__"), how="left")
    frame = frame.dropna(subset=[c for c in frame.columns if c != "__extra__"])
    xf = frame[X.columns]
    names = list(X.columns)
    xv = xf.to_numpy(dtype=float)
    if add_constant:
        xv = np.column_stack([np.ones(len(xv)), xv])
        names = ["const"] + names
    return names, frame.index, frame["__y__"].to_numpy(dtype=float), xv, (frame["__extra__"].to_numpy() if "__extra__" in frame else None)


@dataclass
class OLSResult:
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    cov: pd.DataFrame
    resid: pd.Series
    fitted: pd.Series
    r2: float
    adj_r2: float
    nobs: int
    df_resid: int
    sigma2: float
    cov_type: str
    detail: dict = field(default_factory=dict)
    _X: np.ndarray = field(default=None, repr=False)
    _y: np.ndarray = field(default=None, repr=False)

    @property
    def names(self) -> list[str]:
        return list(self.params.index)

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        q = sps.norm.ppf(1 - alpha / 2) if self.detail.get("asymptotic", False) else sps.t.ppf(1 - alpha / 2, self.df_resid)
        return pd.DataFrame({"lower": self.params - q * self.bse, "upper": self.params + q * self.bse})

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.params, "se": self.bse, "t": self.tvalues, "p": self.pvalues})

    def wald_test(self, R, r=None) -> dict:
        """``H0: R beta = r``. Returns the chi-square (asymptotic) Wald statistic and the small-sample F version."""
        R = np.atleast_2d(np.asarray(R, dtype=float))
        r = np.zeros(R.shape[0]) if r is None else np.atleast_1d(np.asarray(r, dtype=float))
        diff = R @ self.params.to_numpy() - r
        mid = R @ self.cov.to_numpy() @ R.T
        stat = float(diff @ np.linalg.solve(mid, diff))
        q = R.shape[0]
        return {"chi2": stat, "chi2_pvalue": float(sps.chi2.sf(stat, q)), "F": stat / q, "F_pvalue": float(sps.f.sf(stat / q, q, self.df_resid)), "df": q}

    def f_test_all(self) -> dict:
        """All slope coefficients jointly zero (constant excluded when present)."""
        k = len(self.params)
        start = 1 if "const" in self.params.index else 0
        R = np.eye(k)[start:]
        return self.wald_test(R)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, len(self.params) - ("const" in self.params.index))
        if "const" in self.params.index:
            X = np.column_stack([np.ones(len(X)), X])
        return X @ self.params.to_numpy()


def _bartlett_meat(scores: np.ndarray, lags: int) -> np.ndarray:
    n = len(scores)
    meat = scores.T @ scores
    for ell in range(1, min(lags, n - 1) + 1):
        gamma = scores[ell:].T @ scores[:-ell]
        meat += (1.0 - ell / (lags + 1.0)) * (gamma + gamma.T)
    return meat


def _cluster_meat(scores: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(pd.Series(groups))
    sums = np.zeros((len(uniq), scores.shape[1]))
    np.add.at(sums, codes, scores)
    return sums.T @ sums, len(uniq)


def covariance(X: np.ndarray, resid: np.ndarray, cov: str = "nonrobust", lags: int | None = None, groups=None, use_correction: bool = True,
               sigma2: float | None = None) -> tuple[np.ndarray, dict]:
    """The sandwich ``(X'X)^-1 meat (X'X)^-1`` for the chosen estimator. Returns ``(V, detail)``."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    detail: dict = {}
    if cov == "nonrobust":
        s2 = float(resid @ resid) / (n - k) if sigma2 is None else sigma2
        return s2 * xtx_inv, detail
    scores = X * resid[:, None]
    if cov in ("HC0", "HC1", "HC2", "HC3"):
        if cov in ("HC2", "HC3"):
            h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
            scores = X * (resid / (1.0 - h) ** (0.5 if cov == "HC2" else 1.0))[:, None]
        meat = scores.T @ scores
        if cov == "HC1":
            meat *= n / (n - k)
    elif cov == "HAC":
        lags = newey_west_lag(n) if lags is None else int(lags)
        meat = _bartlett_meat(scores, lags)
        if use_correction:
            meat *= n / (n - k)
        detail["lags"] = lags
    elif cov == "cluster":
        if groups is None:
            raise ValueError("cov='cluster' needs groups")
        meat, g = _cluster_meat(scores, np.asarray(groups))
        if use_correction:
            meat *= g / (g - 1.0) * (n - 1.0) / (n - k)
        detail["n_clusters"] = g
    elif cov == "cluster2":
        if groups is None or np.asarray(groups).ndim != 2 or np.asarray(groups).shape[1] != 2:
            raise ValueError("cov='cluster2' needs groups with two columns")
        g = np.asarray(groups)
        pieces = []
        counts = []
        for labels in (g[:, 0], g[:, 1], pd.Series(list(zip(g[:, 0], g[:, 1]))).astype(str).to_numpy()):
            m, c = _cluster_meat(scores, labels)
            pieces.append(m * (c / (c - 1.0) * (n - 1.0) / (n - k) if use_correction else 1.0))
            counts.append(c)
        meat = pieces[0] + pieces[1] - pieces[2]
        detail["n_clusters"] = counts[:2]
        v = xtx_inv @ meat @ xtx_inv
        w, vec = np.linalg.eigh((v + v.T) / 2)               # the two-way estimator can be indefinite: project onto the PSD cone
        return (vec * np.clip(w, 0.0, None)) @ vec.T, detail
    else:
        raise ValueError(f"unknown covariance estimator '{cov}'")
    return xtx_inv @ meat @ xtx_inv, detail


def _result(names, index, y, X, beta, cov_type, V, detail, sigma_y=None, resid=None, df_resid=None, asymptotic=None) -> OLSResult:
    n, k = X.shape
    fitted = X @ beta
    resid = y - fitted if resid is None else resid
    df_resid = n - k if df_resid is None else df_resid
    has_const = bool(np.allclose(X[:, 0], 1.0)) if k else False
    tss = float(((y - y.mean()) ** 2).sum()) if has_const else float((y ** 2).sum())
    rss = float(resid @ resid)
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * ((n - (1 if has_const else 0)) / df_resid) if df_resid > 0 else float("nan")
    bse = np.sqrt(np.clip(np.diag(V), 0, None))
    t = beta / np.where(bse > 0, bse, np.nan)
    asymptotic = cov_type != "nonrobust" if asymptotic is None else asymptotic
    p = 2.0 * (sps.norm.sf(np.abs(t)) if asymptotic else sps.t.sf(np.abs(t), df_resid))
    detail = dict(detail, asymptotic=asymptotic)
    return OLSResult(pd.Series(beta, index=names), pd.Series(bse, index=names), pd.Series(t, index=names), pd.Series(p, index=names),
                     pd.DataFrame(V, index=names, columns=names), pd.Series(resid, index=index), pd.Series(fitted, index=index), r2, adj, n, df_resid,
                     rss / df_resid if df_resid > 0 else float("nan"), cov_type, detail, X, y)


def ols(y, X, cov: str = "nonrobust", lags: int | None = None, groups=None, add_constant: bool = True, use_correction: bool = True) -> OLSResult:
    """Ordinary least squares with the chosen covariance estimator (see the module table). Rows with a missing value are dropped.

    ``groups`` is one label per row for ``cluster`` (aligned with ``y`` before the drop) or a two-column array/frame for ``cluster2``.
    ``asymptotic`` inference (normal p-values) is used for every robust estimator, the t distribution for ``nonrobust`` (as ``statsmodels`` does).
    """
    if groups is not None:
        g = np.asarray(groups)
        yy = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float).ravel())
        names, index, yv, Xv, _ = _prepare(y, X, add_constant)
        groups = g[yy.index.get_indexer(index)]
    else:
        names, index, yv, Xv, _ = _prepare(y, X, add_constant)
    n, k = Xv.shape
    if n <= k:
        raise ValueError(f"{n} observations cannot identify {k} parameters")
    beta = np.linalg.solve(Xv.T @ Xv, Xv.T @ yv)
    resid = yv - Xv @ beta
    V, detail = covariance(Xv, resid, cov, lags, groups, use_correction)
    return _result(names, index, yv, Xv, beta, cov, V, detail)


def wls(y, X, weights, cov: str = "nonrobust", add_constant: bool = True, **kwargs) -> OLSResult:
    """Weighted least squares: minimises ``sum w_i e_i^2``. ``weights`` are inverse error variances (up to scale)."""
    w = pd.Series(np.asarray(weights, dtype=float), index=y.index) if not isinstance(weights, pd.Series) else weights
    names, index, yv, Xv, wv = _prepare(y, X, add_constant, extra=w)
    if np.any(wv <= 0):
        raise ValueError("weights must be positive")
    root = np.sqrt(wv)
    Xt, yt = Xv * root[:, None], yv * root
    beta = np.linalg.solve(Xt.T @ Xt, Xt.T @ yt)
    resid_t = yt - Xt @ beta
    V, detail = covariance(Xt, resid_t, cov, kwargs.get("lags"), kwargs.get("groups"), kwargs.get("use_correction", True))
    res = _result(names, index, yt, Xt, beta, cov, V, detail, resid=resid_t)
    res.resid = pd.Series(yv - Xv @ beta, index=index)               # residuals on the original scale
    res.fitted = pd.Series(Xv @ beta, index=index)
    # R^2 on the weighted scale, centred at the weighted mean
    if "const" in names:
        ybar = np.average(yv, weights=wv)
        res.r2 = 1.0 - float(wv @ res.resid.to_numpy() ** 2) / float(wv @ (yv - ybar) ** 2)
    res.detail["weights_used"] = True
    return res


def gls(y, X, sigma: np.ndarray, add_constant: bool = True) -> OLSResult:
    """Generalised least squares with a KNOWN error covariance ``sigma`` (n x n): ``beta = (X' S^-1 X)^-1 X' S^-1 y``."""
    names, index, yv, Xv, _ = _prepare(y, X, add_constant)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (len(yv), len(yv)):
        raise ValueError("sigma must be n x n for the n complete rows")
    L = np.linalg.cholesky(np.linalg.inv(sigma))
    Xt, yt = L.T @ Xv, L.T @ yv
    beta = np.linalg.solve(Xt.T @ Xt, Xt.T @ yt)
    resid_t = yt - Xt @ beta
    n, k = Xt.shape
    V = float(resid_t @ resid_t) / (n - k) * np.linalg.inv(Xt.T @ Xt)
    res = _result(names, index, yt, Xt, beta, "gls", V, {}, resid=resid_t)
    res.resid = pd.Series(yv - Xv @ beta, index=index)
    res.fitted = pd.Series(Xv @ beta, index=index)
    return res


def feasible_gls_ar1(y, X, add_constant: bool = True, max_iter: int = 50, tol: float = 1e-8) -> OLSResult:
    """Prais-Winsten feasible GLS for AR(1) errors ``u_t = rho u_{t-1} + e_t``; iterates ``rho`` to convergence and keeps the first observation."""
    names, index, yv, Xv, _ = _prepare(y, X, add_constant)
    n, k = Xv.shape
    beta = np.linalg.solve(Xv.T @ Xv, Xv.T @ yv)
    rho = 0.0
    for _ in range(max_iter):
        u = yv - Xv @ beta
        new = float(u[1:] @ u[:-1] / (u[:-1] @ u[:-1]))
        yt = np.concatenate([[np.sqrt(1 - new ** 2) * yv[0]], yv[1:] - new * yv[:-1]])
        Xt = np.vstack([np.sqrt(1 - new ** 2) * Xv[:1], Xv[1:] - new * Xv[:-1]])
        beta = np.linalg.solve(Xt.T @ Xt, Xt.T @ yt)
        if abs(new - rho) < tol:
            rho = new
            break
        rho = new
    resid_t = yt - Xt @ beta
    V = float(resid_t @ resid_t) / (n - k) * np.linalg.inv(Xt.T @ Xt)
    res = _result(names, index, yt, Xt, beta, "prais_winsten", V, {"rho": rho}, resid=resid_t)
    res.resid = pd.Series(yv - Xv @ beta, index=index)
    res.fitted = pd.Series(Xv @ beta, index=index)
    return res


def rolling_ols(y: pd.Series, X, window: int, expanding: bool = False, min_obs: int | None = None, add_constant: bool = True) -> dict[str, pd.DataFrame]:
    """OLS on every trailing window in one batched solve. Returns ``{'params', 'bse', 'tvalues'}`` frames indexed like ``y``.

    The value on date ``t`` uses data up to and including ``t`` only (so a rolling beta is causal). Standard errors are the non-robust ones;
    for HAC errors on a window use ``ols`` on the slice. Rows with a missing value are excluded from every window (the window is counted in rows).
    """
    names, index, yv, Xv, _ = _prepare(y, X, add_constant)
    n, k = Xv.shape
    min_obs = window if (min_obs is None and not expanding) else (min_obs or max(k + 2, 20))
    XtX = np.einsum("ti,tj->tij", Xv, Xv)
    Xty = Xv * yv[:, None]
    yty = yv ** 2
    c_xx = np.concatenate([np.zeros((1, k, k)), np.cumsum(XtX, axis=0)])
    c_xy = np.concatenate([np.zeros((1, k)), np.cumsum(Xty, axis=0)])
    c_yy = np.concatenate([[0.0], np.cumsum(yty)])
    ends = np.arange(1, n + 1)
    starts = np.zeros(n, dtype=int) if expanding else np.maximum(ends - window, 0)
    cnt = ends - starts
    A = c_xx[ends] - c_xx[starts]
    b = c_xy[ends] - c_xy[starts]
    yy = c_yy[ends] - c_yy[starts]
    ok = cnt >= min_obs
    beta = np.full((n, k), np.nan)
    se = np.full((n, k), np.nan)
    for t in np.flatnonzero(ok):
        try:
            inv = np.linalg.inv(A[t])
        except np.linalg.LinAlgError:
            continue
        bt = inv @ b[t]
        rss = max(float(yy[t] - bt @ b[t]), 0.0)
        beta[t] = bt
        se[t] = np.sqrt(np.clip(np.diag(inv), 0, None) * rss / max(cnt[t] - k, 1))
    out = {"params": pd.DataFrame(beta, index=index, columns=names), "bse": pd.DataFrame(se, index=index, columns=names)}
    out["tvalues"] = out["params"] / out["bse"]
    return {key: frame.reindex(y.index) for key, frame in out.items()}


# ----------------------------------------------------------------------------------------------------------------- diagnostics
def durbin_watson(resid) -> float:
    e = np.asarray(resid, dtype=float)
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


def jarque_bera(resid) -> dict:
    e = np.asarray(resid, dtype=float)
    n = len(e)
    s, kurt = sps.skew(e), sps.kurtosis(e, fisher=False)
    stat = n / 6.0 * (s ** 2 + (kurt - 3.0) ** 2 / 4.0)
    return {"statistic": float(stat), "pvalue": float(sps.chi2.sf(stat, 2)), "skew": float(s), "kurtosis": float(kurt)}


def _aux_r2(z: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    e = z - X @ beta
    tss = float(((z - z.mean()) ** 2).sum())
    return 1.0 - float(e @ e) / tss, X.shape[1]


def breusch_pagan(result: OLSResult, robust: bool = True) -> dict:
    """Breusch-Pagan LM test for heteroskedasticity (Koenker's studentised version by default): regress ``e^2`` on ``X``."""
    X, e = result._X, result.resid.to_numpy()
    n = len(e)
    if robust:
        r2, k = _aux_r2(e ** 2, X)
        stat = n * r2
    else:
        z = e ** 2 / (e @ e / n)
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        fitted = X @ beta
        stat = float(((fitted - z.mean()) ** 2).sum() / 2.0)
        k = X.shape[1]
    df = k - 1
    return {"statistic": float(stat), "pvalue": float(sps.chi2.sf(stat, df)), "df": df}


def white_test(result: OLSResult) -> dict:
    """White (1980) test: ``e^2`` on the regressors, their squares and cross-products."""
    X, e = result._X, result.resid.to_numpy()
    n, k = X.shape
    cols = [X[:, i] * X[:, j] for i in range(k) for j in range(i, k)]
    Z = np.column_stack(cols)
    keep = [j for j in range(Z.shape[1]) if np.ptp(Z[:, j]) > 0 or j == 0]            # drop constants other than the first
    Z = Z[:, keep]
    rank = np.linalg.matrix_rank(Z)
    r2, _ = _aux_r2(e ** 2, Z)
    stat = n * r2
    return {"statistic": float(stat), "pvalue": float(sps.chi2.sf(stat, rank - 1)), "df": int(rank - 1)}


def breusch_godfrey(result: OLSResult, nlags: int = 4) -> dict:
    """Breusch-Godfrey LM test for serial correlation up to ``nlags``: regress the residual on ``X`` and its own lags (missing lags set to zero)."""
    X, e = result._X, result.resid.to_numpy()
    n = len(e)
    lagged = np.column_stack([np.concatenate([np.zeros(l), e[:-l]]) for l in range(1, nlags + 1)])
    Z = np.column_stack([X, lagged])
    beta, *_ = np.linalg.lstsq(Z, e, rcond=None)
    u = e - Z @ beta
    r2 = 1.0 - float(u @ u) / float(e @ e)
    stat = n * r2
    return {"statistic": float(stat), "pvalue": float(sps.chi2.sf(stat, nlags)), "df": nlags}


def ramsey_reset(result: OLSResult, powers=(2, 3)) -> dict:
    """RESET: do powers of the fitted values add explanatory power? (F test of the added terms.)"""
    X, y = result._X, result._y
    fit = result.fitted.to_numpy()
    Z = np.column_stack([X] + [fit ** p for p in powers])
    rss0 = float(result.resid.to_numpy() @ result.resid.to_numpy())
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    u = y - Z @ beta
    rss1 = float(u @ u)
    q, n, k1 = len(powers), len(y), Z.shape[1]
    f = ((rss0 - rss1) / q) / (rss1 / (n - k1))
    return {"F": float(f), "pvalue": float(sps.f.sf(f, q, n - k1)), "df": (q, n - k1)}


def variance_inflation(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R_j^2) from regressing column j on the others (with a constant). Above ~10 signals collinearity."""
    Xv = X.dropna().to_numpy(dtype=float)
    out = {}
    for j, name in enumerate(X.columns):
        others = np.delete(Xv, j, axis=1)
        Z = np.column_stack([np.ones(len(Xv)), others])
        r2, _ = _aux_r2(Xv[:, j], Z)
        out[name] = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
    return pd.Series(out)


def diagnostics(result: OLSResult, nlags: int = 4) -> pd.DataFrame:
    """One table of residual tests. A small p-value rejects the assumption named in the first column."""
    jb = jarque_bera(result.resid)
    rows = {
        "normal residuals (Jarque-Bera)": (jb["statistic"], jb["pvalue"]),
        "homoskedastic (Breusch-Pagan, Koenker)": (lambda r: (r["statistic"], r["pvalue"]))(breusch_pagan(result)),
        "homoskedastic (White)": (lambda r: (r["statistic"], r["pvalue"]))(white_test(result)),
        f"no serial correlation (Breusch-Godfrey, {nlags} lags)": (lambda r: (r["statistic"], r["pvalue"]))(breusch_godfrey(result, nlags)),
        "correct functional form (RESET)": (lambda r: (r["F"], r["pvalue"]))(ramsey_reset(result)),
    }
    table = pd.DataFrame(rows, index=["statistic", "pvalue"]).T
    table.loc["Durbin-Watson (2 = no autocorrelation)"] = (durbin_watson(result.resid), np.nan)
    return table
