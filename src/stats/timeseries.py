"""Tests on time series: unit roots, stationarity, mean reversion, autocorrelation, cointegration and Granger causality.

The unit-root and stationarity tests answer opposite questions and should be read together:

``adf_test``         H0: the series has a unit root (is a random walk). Rejecting H0 is evidence of mean reversion.
``phillips_perron``  The same H0, with a non-parametric correction for serial correlation instead of lagged differences.
``kpss_test``        H0: the series is (trend-)stationary. Rejecting H0 is evidence of a unit root.
``variance_ratio``   Lo & MacKinlay (1988): under a random walk the variance of q-period returns is q times the one-period variance. VR < 1 means
                     mean reversion, VR > 1 momentum. The heteroskedasticity-robust z* is the one to use for returns.
``hurst_exponent``   Rescaled-range and detrended-fluctuation estimates; 0.5 is a random walk, below it mean reverting, above it trending.
``engle_granger``    Two-step cointegration test: regress one price on another, test the residual for a unit root with the MacKinnon (1991/2010)
                     cointegration critical values.
``granger_causality``F test that lags of ``x`` improve a regression of ``y`` on its own lags. Predictive, not causal in the structural sense.

The ADF regression, KPSS statistic, Phillips-Perron correction and the variance-ratio statistics are computed here; MacKinnon critical values and
p-values come from ``statsmodels.tsa.adfvalues``. Tests compare against ``statsmodels`` and ``arch`` in the suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp



@dataclass
class UnitRootResult:
    statistic: float
    pvalue: float
    lags: int
    nobs: int
    critical_values: dict
    regression: str
    null: str

    def reject(self, level: float = 0.05) -> bool:
        return bool(self.pvalue < level)

    def __repr__(self) -> str:
        return f"UnitRootResult(stat={self.statistic:.3f}, p={self.pvalue:.4f}, lags={self.lags}, n={self.nobs}, H0: {self.null})"


def _lag_matrix(x: np.ndarray, lags: int) -> np.ndarray:
    return np.column_stack([x[lags - j: len(x) - j] for j in range(1, lags + 1)]) if lags else np.empty((len(x) - lags, 0))


def _adf_regression(y: np.ndarray, lags: int, regression: str, trim_to: int | None = None):
    """Regress dy_t on y_{t-1}, deterministic terms and ``lags`` lagged differences. Returns (t-stat on y_{t-1}, rss, nobs, k, aic, bic)."""
    dy = np.diff(y)
    start = lags if trim_to is None else trim_to
    yl = y[:-1][start:]
    dep = dy[start:]
    cols = [yl]
    if lags:
        cols += [dy[start - j: len(dy) - j] for j in range(1, lags + 1)]
    n = len(dep)
    trend = np.arange(1, n + 1, dtype=float)
    if regression in ("c", "ct", "ctt"):
        cols.append(np.ones(n))
    if regression in ("ct", "ctt"):
        cols.append(trend)
    if regression == "ctt":
        cols.append(trend ** 2)
    Z = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Z, dep, rcond=None)
    resid = dep - Z @ beta
    rss = float(resid @ resid)
    k = Z.shape[1]
    s2 = rss / (n - k)
    se0 = np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[0, 0])
    llf = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1.0)
    return beta[0] / se0, rss, n, k, -2 * llf + 2 * k, -2 * llf + np.log(n) * k, beta, Z


def adf_test(x, regression: str = "c", maxlag: int | None = None, autolag: str | None = "aic") -> UnitRootResult:
    """Augmented Dickey-Fuller test (H0: unit root). ``regression``: ``n`` none, ``c`` constant, ``ct`` constant+trend, ``ctt`` plus quadratic trend.

    ``autolag`` ('aic', 'bic', 'tstat' or None) picks the number of lagged differences up to ``maxlag`` (Schwert: ``12 (n/100)^(1/4)``); every
    candidate is fitted on the same sample so the criteria are comparable.
    """
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
        maxlag = min(maxlag, n // 2 - 3)
    if autolag is None:
        lags = maxlag
    elif autolag in ("aic", "bic"):
        scores = {}
        for p in range(maxlag + 1):
            out = _adf_regression(y, p, regression, trim_to=maxlag)
            scores[p] = out[4] if autolag == "aic" else out[5]
        lags = min(scores, key=scores.get)
    elif autolag == "tstat":
        lags = maxlag
        while lags > 0:
            out = _adf_regression(y, lags, regression, trim_to=lags)
            beta, Z = out[6], out[7]
            resid = np.diff(y)[lags:] - Z @ beta
            s2 = float(resid @ resid) / (len(resid) - Z.shape[1])
            se = np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[lags, lags])      # column index of the last lagged difference
            if abs(beta[lags] / se) > 1.6448536269514722:
                break
            lags -= 1
    else:
        raise ValueError("autolag must be 'aic', 'bic', 'tstat' or None")
    stat, _, nobs, *_ = _adf_regression(y, lags, regression, trim_to=lags)
    p = float(mackinnonp(stat, regression=regression, N=1))
    crit = dict(zip(["1%", "5%", "10%"], mackinnoncrit(N=1, regression=regression, nobs=nobs)))
    return UnitRootResult(float(stat), p, int(lags), int(nobs), crit, regression, "unit root")


def phillips_perron(x, regression: str = "c", lags: int | None = None, test_type: str = "tau") -> UnitRootResult:
    """Phillips-Perron (1988): the Dickey-Fuller regression with a Newey-West correction to the t statistic (``tau``) or to the coefficient (``rho``)."""
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if regression not in ("c", "ct", "n"):
        raise ValueError("regression must be n, c or ct")
    n_total = len(y)
    lags = int(np.ceil(12.0 * (n_total / 100.0) ** 0.25)) if lags is None else lags
    dep = y[1:]
    cols = [y[:-1]]
    n = len(dep)
    if regression in ("c", "ct"):
        cols.append(np.ones(n))
    if regression == "ct":
        cols.append(np.arange(1, n + 1, dtype=float))
    Z = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Z, dep, rcond=None)
    u = dep - Z @ beta
    k = Z.shape[1]
    s2 = float(u @ u) / (n - k)
    gamma0 = float(u @ u) / n
    lam = gamma0
    for j in range(1, lags + 1):
        gj = float(u[j:] @ u[:-j]) / n
        lam += 2.0 * (1.0 - j / (lags + 1.0)) * gj
    se = np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[0, 0])
    tstat = (beta[0] - 1.0) / se
    if test_type == "tau":
        stat = np.sqrt(gamma0 / lam) * tstat - 0.5 * ((lam - gamma0) / np.sqrt(lam)) * (n * se / np.sqrt(s2))
        p = float(mackinnonp(stat, regression=regression, N=1))
        crit = dict(zip(["1%", "5%", "10%"], mackinnoncrit(N=1, regression=regression, nobs=n)))
    else:
        stat = n * (beta[0] - 1.0) - 0.5 * (n ** 2 * se ** 2 / s2) * (lam - gamma0)
        p, crit = float("nan"), {}
    return UnitRootResult(float(stat), p, int(lags), int(n), crit, regression, "unit root")


_KPSS_CRIT = {"c": {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}, "ct": {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216}}


def kpss_test(x, regression: str = "c", lags: int | None = None) -> UnitRootResult:
    """KPSS (Kwiatkowski-Phillips-Schmidt-Shin 1992). H0: level- (``c``) or trend- (``ct``) stationary. The p-value is interpolated in the published table
    and is capped to the range 0.01 to 0.10."""
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if regression == "c":
        resid = y - y.mean()
    elif regression == "ct":
        Z = np.column_stack([np.ones(n), np.arange(1, n + 1)])
        resid = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    else:
        raise ValueError("regression must be c or ct")
    lags = int(np.ceil(12.0 * (n / 100.0) ** 0.25)) if lags is None else lags
    lags = min(lags, n - 1)
    s2 = float(resid @ resid) / n
    for j in range(1, lags + 1):
        s2 += 2.0 * (1.0 - j / (lags + 1.0)) * float(resid[j:] @ resid[:-j]) / n
    S = np.cumsum(resid)
    stat = float((S ** 2).sum() / (n ** 2 * s2))
    crit = _KPSS_CRIT[regression]
    xs = np.array([0.0] + [crit[k] for k in ("10%", "5%", "2.5%", "1%")])
    ps = np.array([1.0, 0.10, 0.05, 0.025, 0.01])
    p = float(np.interp(stat, xs[1:], ps[1:])) if stat > xs[1] else 0.10
    p = max(p, 0.01) if stat < crit["1%"] else 0.01
    return UnitRootResult(stat, p, lags, n, crit, regression, "stationary")


def stationarity_summary(x, regression: str = "c") -> dict:
    """ADF, PP and KPSS together with a verdict: ``stationary``, ``unit root``, ``trend/difference-stationary`` or ``inconclusive``."""
    adf, pp, kp = adf_test(x, regression), phillips_perron(x, regression if regression != "ctt" else "ct"), kpss_test(x, "c" if regression == "c" else "ct")
    unit_root_rejected = adf.reject() and pp.reject()
    stationary_rejected = kp.reject()
    verdict = ("stationary" if unit_root_rejected and not stationary_rejected else
               "unit root" if not unit_root_rejected and stationary_rejected else "inconclusive")
    return {"adf": adf, "pp": pp, "kpss": kp, "verdict": verdict}


# ---------------------------------------------------------------------------------------------------------------- mean reversion
def variance_ratio(x, q: int = 5, log_prices: bool = True, robust: bool = True) -> dict:
    """Lo-MacKinlay (1988) variance ratio of the q-period to the 1-period return variance, with overlapping observations and the unbiased estimators.

    ``x`` is a price series (``log_prices=True`` takes logs) or, with ``log_prices=False``, a level series that is already a random-walk candidate.
    ``z`` assumes homoskedastic increments, ``z_robust`` (Lo-MacKinlay z*) allows heteroskedasticity.
    """
    p = np.asarray(x, dtype=float)
    p = p[np.isfinite(p)]
    if log_prices:
        p = np.log(p)
    r = np.diff(p)
    nq = len(r)
    mu = (p[-1] - p[0]) / nq
    s1 = float(((r - mu) ** 2).sum()) / (nq - 1)
    m = q * (nq - q + 1) * (1.0 - q / nq)
    rq = p[q:] - p[:-q]
    sq = float(((rq - q * mu) ** 2).sum()) / m
    vr = sq / s1
    phi1 = 2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * nq)
    z = (vr - 1.0) / np.sqrt(phi1)
    d = (r - mu) ** 2
    delta = np.array([(nq * (d[j:] * d[:-j]).sum()) / (d.sum() ** 2) for j in range(1, q)])
    weights = np.array([(2.0 * (q - j) / q) ** 2 for j in range(1, q)])
    theta = float((weights * delta).sum())
    z_robust = np.sqrt(nq) * (vr - 1.0) / np.sqrt(theta) if theta > 0 else float("nan")
    return {"vr": float(vr), "z": float(z), "z_robust": float(z_robust), "p_z": float(2 * sps.norm.sf(abs(z))), "p_robust": float(2 * sps.norm.sf(abs(z_robust))), "q": q, "n": nq}


def hurst_exponent(x, method: str = "rs", min_window: int = 10, n_windows: int = 20) -> float:
    """Hurst exponent of a RETURN series. ``rs``: rescaled range with the Anis-Lloyd small-sample correction removed from the slope's intercept via the
    expected value E[R/S]; ``dfa``: detrended fluctuation analysis. 0.5 random walk, > 0.5 persistent, < 0.5 anti-persistent."""
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    sizes = np.unique(np.logspace(np.log10(min_window), np.log10(n // 2), n_windows).astype(int))
    if method == "rs":
        stats_ = []
        for s in sizes:
            k = n // s
            seg = r[: k * s].reshape(k, s)
            dev = np.cumsum(seg - seg.mean(axis=1, keepdims=True), axis=1)
            R = dev.max(axis=1) - dev.min(axis=1)
            S = seg.std(axis=1, ddof=0)
            ok = S > 0
            stats_.append(np.mean(R[ok] / S[ok]))
        # Anis-Lloyd expected R/S of an i.i.d. series, subtracted so that i.i.d. noise gives H = 0.5
        def expected(s):
            i = np.arange(1, s)
            core = np.sum(np.sqrt((s - i) / i))
            from scipy.special import gamma
            c = gamma((s - 1) / 2) / (np.sqrt(np.pi) * gamma(s / 2)) if s <= 340 else 1.0 / np.sqrt(s * np.pi / 2)
            return c * core
        corr = np.array([expected(s) for s in sizes])
        h = np.polyfit(np.log(sizes), np.log(stats_), 1)[0] - np.polyfit(np.log(sizes), np.log(corr), 1)[0] + 0.5
        return float(h)
    if method == "dfa":
        profile = np.cumsum(r - r.mean())
        fl = []
        for s in sizes:
            k = len(profile) // s
            seg = profile[: k * s].reshape(k, s)
            t = np.arange(s)
            res = []
            for row in seg:
                coef = np.polyfit(t, row, 1)
                res.append(np.mean((row - np.polyval(coef, t)) ** 2))
            fl.append(np.sqrt(np.mean(res)))
        return float(np.polyfit(np.log(sizes), np.log(fl), 1)[0])
    raise ValueError("method must be 'rs' or 'dfa'")


def half_life(x) -> float:
    """Half-life of mean reversion from the AR(1) coefficient of ``x_t - x_{t-1}`` on ``x_{t-1}`` (Ornstein-Uhlenbeck): ``-ln 2 / ln(1 + b)``. Infinite if b >= 0."""
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    dy, yl = np.diff(y), y[:-1] - y[:-1].mean()
    b = float(yl @ dy / (yl @ yl))
    return float(-np.log(2.0) / np.log1p(b)) if -1.0 < b < 0 else float("inf")


# ----------------------------------------------------------------------------------------------------------- residual autocorrelation
def acf(x, nlags: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    d = float(x @ x)
    return np.array([x[k:] @ x[: len(x) - k] / d for k in range(nlags + 1)])


def ljung_box(x, lags: int = 10, model_df: int = 0) -> dict:
    """Ljung-Box Q: are the first ``lags`` autocorrelations jointly zero? ``model_df`` subtracts fitted ARMA parameters."""
    r = acf(x, lags)[1:]
    n = int(np.isfinite(np.asarray(x, dtype=float)).sum())
    q = n * (n + 2) * float(np.sum(r ** 2 / (n - np.arange(1, lags + 1))))
    return {"statistic": q, "pvalue": float(sps.chi2.sf(q, lags - model_df)), "lags": lags}


def arch_lm(x, lags: int = 5) -> dict:
    """Engle's ARCH-LM test: regress squared (demeaned) values on their own lags; ``n R^2`` is chi-square(lags) under no ARCH effect."""
    e = np.asarray(x, dtype=float)
    e = e[np.isfinite(e)]
    e2 = (e - e.mean()) ** 2
    y = e2[lags:]
    Z = np.column_stack([np.ones(len(y))] + [e2[lags - j: len(e2) - j] for j in range(1, lags + 1)])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    u = y - Z @ beta
    r2 = 1.0 - float(u @ u) / float(((y - y.mean()) ** 2).sum())
    stat = len(y) * r2
    return {"statistic": float(stat), "pvalue": float(sps.chi2.sf(stat, lags)), "lags": lags}


# --------------------------------------------------------------------------------------------------------------- cointegration
def engle_granger(y, x, trend: str = "c", maxlag: int | None = None, autolag: str | None = "aic") -> dict:
    """Engle-Granger (1987) two-step test. Regress ``y`` on ``x`` (and a constant if ``trend='c'``), ADF-test the residual with MacKinnon critical
    values for ``1 + k`` variables. H0: no cointegration. Returns the hedge ratio, the spread and the test."""
    ys = (y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float))).rename("y")
    if isinstance(x, pd.Series):
        xs = x.to_frame(x.name if x.name not in (None, "y") else "x")
    elif isinstance(x, pd.DataFrame):
        xs = x
    else:
        arr = np.asarray(x, dtype=float).reshape(len(ys), -1)
        xs = pd.DataFrame(arr, index=ys.index, columns=["x"] if arr.shape[1] == 1 else [f"x{i}" for i in range(arr.shape[1])])
    xs.index = ys.index if not isinstance(x, (pd.Series, pd.DataFrame)) else xs.index
    frame = pd.concat([ys, xs], axis=1).dropna()
    yv = frame.iloc[:, 0].to_numpy()
    Xv = frame.iloc[:, 1:].to_numpy()
    Z = np.column_stack([np.ones(len(yv)), Xv]) if trend == "c" else Xv
    beta = np.linalg.lstsq(Z, yv, rcond=None)[0]
    spread = yv - Z @ beta
    n_vars = 1 + Xv.shape[1]
    reg = "n" if trend == "n" else "c"
    res = adf_test(spread, regression="n", maxlag=maxlag, autolag=autolag)
    stat = res.statistic
    p = float(mackinnonp(stat, regression=reg, N=n_vars))
    crit = dict(zip(["1%", "5%", "10%"], mackinnoncrit(N=n_vars, regression=reg, nobs=res.nobs)))
    out_beta = pd.Series(beta, index=(["const"] if trend == "c" else []) + list(frame.columns[1:]))
    return {"statistic": float(stat), "pvalue": p, "critical_values": crit, "lags": res.lags, "hedge_ratio": out_beta, "spread": pd.Series(spread, index=frame.index),
            "half_life": half_life(spread)}


# --------------------------------------------------------------------------------------------------------------------- Granger
def granger_causality(y, x, maxlag: int = 4) -> pd.DataFrame:
    """Does ``x`` Granger-cause ``y``? For each lag order ``p`` up to ``maxlag``: F test that the ``p`` lags of ``x`` add nothing to an AR(``p``) of ``y``.

    Statistical predictability only: an omitted common driver, or ``y`` anticipating ``x``, produces the same result.
    """
    frame = pd.concat([pd.Series(np.asarray(y, dtype=float)) if not isinstance(y, pd.Series) else y, pd.Series(np.asarray(x, dtype=float)) if not isinstance(x, pd.Series) else x],
                      axis=1).dropna().to_numpy()
    rows = {}
    for p in range(1, maxlag + 1):
        yy = frame[p:, 0]
        ylags = np.column_stack([frame[p - j: len(frame) - j, 0] for j in range(1, p + 1)])
        xlags = np.column_stack([frame[p - j: len(frame) - j, 1] for j in range(1, p + 1)])
        R = np.column_stack([np.ones(len(yy)), ylags])
        U = np.column_stack([R, xlags])
        rss_r = float(np.sum((yy - R @ np.linalg.lstsq(R, yy, rcond=None)[0]) ** 2))
        rss_u = float(np.sum((yy - U @ np.linalg.lstsq(U, yy, rcond=None)[0]) ** 2))
        dof = len(yy) - U.shape[1]
        f = ((rss_r - rss_u) / p) / (rss_u / dof)
        rows[p] = {"F": f, "pvalue": float(sps.f.sf(f, p, dof)), "chi2": len(yy) * (rss_r - rss_u) / rss_u, "chi2_pvalue": float(sps.chi2.sf(len(yy) * (rss_r - rss_u) / rss_u, p))}
    return pd.DataFrame(rows).T.rename_axis("lags")
