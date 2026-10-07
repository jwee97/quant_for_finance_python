"""Panel regression and cross-sectional asset-pricing regressions.

* ``fama_macbeth``: Fama & MacBeth (1973). Run a cross-sectional regression each date, then treat the time series of slopes as the data: the mean slope
  is the premium for the characteristic and its standard error comes from the time-series variation of the slopes (Newey-West adjusted), which is
  robust to correlation across assets on a date. Characteristics are lagged one period by default so the regression is predictive.
* ``fama_macbeth_two_pass``: the factor-model version. Betas come from time-series regressions of returns on factors; premia from a cross-sectional
  regression of average returns on the betas, with the Shanken (1992) correction for the fact that betas are estimated (Cochrane 2005, ch. 12).
* ``grs_test``: Gibbons-Ross-Shanken (1989) joint test that all pricing errors (alphas) are zero.
* ``panel_ols``: pooled, entity-, time- or two-way fixed effects by the within transformation, with unadjusted, White, one-way and two-way clustered
  and Driscoll-Kraay (1998) standard errors. ``random_effects`` (Swamy-Arora) and ``hausman_test`` choose between random and fixed effects.

Data layout: a ``MultiIndex`` Series/DataFrame whose levels are (entity, time), or wide ``dates x assets`` frames for the Fama-MacBeth functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .regression import _bartlett_meat, newey_west_lag


# --------------------------------------------------------------------------------------------------------------- Fama-MacBeth
@dataclass
class FamaMacBethResult:
    gammas: pd.DataFrame          # one row per date: the cross-sectional slopes
    mean: pd.Series
    se: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    avg_r2: float
    n_periods: int
    avg_assets: float
    lags: int

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"premium": self.mean, "se": self.se, "t": self.tvalues, "p": self.pvalues})


def _nw_mean(x: np.ndarray, lags: int) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    n = len(x)
    c = x - x.mean()
    var = float(c @ c) / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * float(c[k:] @ c[:-k]) / n
    return float(x.mean()), float(np.sqrt(var / n))


def fama_macbeth(returns: pd.DataFrame, characteristics: dict[str, pd.DataFrame], lag: int = 1, lags: int | None = None, min_assets: int | None = None,
                 weights: pd.DataFrame | None = None, add_constant: bool = True) -> FamaMacBethResult:
    """Fama-MacBeth regression of ``returns[t]`` on ``characteristics[t - lag]`` for every date ``t`` with enough assets.

    ``lag=1`` makes each characteristic known one period before the return it is asked to explain. ``lags`` is the Newey-West truncation for the
    standard error of the mean slope (default: the rule of thumb). ``weights`` (dates x assets) makes each cross-section a WLS.
    """
    names = list(characteristics)
    k = len(names) + int(add_constant)
    min_assets = min_assets or (k + 5)
    X = {n: characteristics[n].shift(lag).reindex_like(returns) for n in names}
    rows, r2s, counts, dates = [], [], [], []
    for t in returns.index:
        y = returns.loc[t].to_numpy(dtype=float)
        cols = np.column_stack([X[n].loc[t].to_numpy(dtype=float) for n in names])
        ok = np.isfinite(y) & np.isfinite(cols).all(axis=1)
        if weights is not None:
            w = weights.loc[t].to_numpy(dtype=float)
            ok &= np.isfinite(w) & (w > 0)
        if ok.sum() < min_assets:
            continue
        Z = np.column_stack([np.ones(ok.sum()), cols[ok]]) if add_constant else cols[ok]
        yy = y[ok]
        if weights is not None:
            root = np.sqrt(w[ok])
            beta, *_ = np.linalg.lstsq(Z * root[:, None], yy * root, rcond=None)
        else:
            beta, *_ = np.linalg.lstsq(Z, yy, rcond=None)
        resid = yy - Z @ beta
        tss = float(((yy - yy.mean()) ** 2).sum())
        r2s.append(1.0 - float(resid @ resid) / tss if tss > 0 else np.nan)
        rows.append(beta)
        counts.append(int(ok.sum()))
        dates.append(t)
    cols_out = (["const"] if add_constant else []) + names
    gammas = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=cols_out)
    lags = newey_west_lag(len(gammas)) if lags is None else lags
    mean, se = {}, {}
    for c in cols_out:
        mean[c], se[c] = _nw_mean(gammas[c].to_numpy(), lags)
    mean, se = pd.Series(mean), pd.Series(se)
    t = mean / se
    p = pd.Series(2 * sps.t.sf(np.abs(t), len(gammas) - 1), index=t.index)
    return FamaMacBethResult(gammas, mean, se, t, p, float(np.nanmean(r2s)), len(gammas), float(np.mean(counts)), lags)


def fama_macbeth_two_pass(returns: pd.DataFrame, factors: pd.DataFrame, add_constant: bool = False) -> dict:
    """Two-pass factor-model estimate with Shanken (1992) standard errors (Cochrane 2005, eq. 12.19-12.20, OLS cross-section).

    Pass 1: ``r_i,t = a_i + b_i' f_t + e``. Pass 2: ``mean(r_i) = lambda' b_i + alpha_i``. Returns the premia ``lambda`` with the naive
    (Fama-MacBeth) and the Shanken-corrected standard errors, the pricing errors ``alpha`` and the cross-sectional R-squared.
    """
    frame = pd.concat([returns, factors], axis=1).dropna()
    R, F = frame[returns.columns].to_numpy(), frame[factors.columns].to_numpy()
    T, N = R.shape
    K = F.shape[1]
    Fc = np.column_stack([np.ones(T), F])
    coef, *_ = np.linalg.lstsq(Fc, R, rcond=None)
    beta = coef[1:].T                                      # N x K
    mean_r = R.mean(axis=0)
    B = np.column_stack([np.ones(N), beta]) if add_constant else beta
    lam, *_ = np.linalg.lstsq(B, mean_r, rcond=None)
    alpha = mean_r - B @ lam
    Sigma = np.cov(R - Fc @ coef, rowvar=False).reshape(N, N)
    Sf = np.atleast_2d(np.cov(F, rowvar=False))
    BtBinv = np.linalg.inv(B.T @ B)
    lam_f = lam[1:] if add_constant else lam
    shanken = 1.0 + float(lam_f @ np.linalg.solve(Sf, lam_f))
    core = BtBinv @ B.T @ Sigma @ B @ BtBinv
    var_naive = core / T
    pad = np.zeros_like(core)
    pad[-K:, -K:] = Sf
    var_shanken = (core * shanken + pad) / T
    cov_alpha = (np.eye(N) - B @ BtBinv @ B.T) @ Sigma @ (np.eye(N) - B @ BtBinv @ B.T).T * (1.0 + (lam_f @ np.linalg.solve(Sf, lam_f))) / T
    names = (["const"] if add_constant else []) + list(factors.columns)
    se_n, se_s = np.sqrt(np.diag(var_naive)), np.sqrt(np.diag(var_shanken))
    tss = float(((mean_r - mean_r.mean()) ** 2).sum())
    return {"lambda": pd.Series(lam, index=names), "se_naive": pd.Series(se_n, index=names), "se_shanken": pd.Series(se_s, index=names),
            "t_shanken": pd.Series(lam / se_s, index=names), "alpha": pd.Series(alpha, index=returns.columns), "betas": pd.DataFrame(beta, index=returns.columns, columns=factors.columns),
            "cross_sectional_r2": 1.0 - float(alpha @ alpha) / tss if tss > 0 else float("nan"), "shanken_factor": shanken, "n_periods": T,
            "alpha_chi2": float(alpha @ np.linalg.pinv(cov_alpha) @ alpha), "alpha_df": N - K - int(add_constant)}


def grs_test(returns: pd.DataFrame, factors: pd.DataFrame) -> dict:
    """Gibbons-Ross-Shanken: ``F = (T-N-K)/N * (1 + mu' S^-1 mu)^-1 * alpha' Sigma^-1 alpha`` with factor mean ``mu`` and covariance ``S``.

    The null is that all N intercepts of the time-series regressions on the K factors are zero (the factors price the assets). Exact in finite
    samples if returns are i.i.d. normal.
    """
    frame = pd.concat([returns, factors], axis=1).dropna()
    R, F = frame[returns.columns].to_numpy(), frame[factors.columns].to_numpy()
    T, N = R.shape
    K = F.shape[1]
    if T - N - K <= 0:
        raise ValueError("GRS needs more periods than assets plus factors")
    Fc = np.column_stack([np.ones(T), F])
    coef, *_ = np.linalg.lstsq(Fc, R, rcond=None)
    alpha = coef[0]
    resid = R - Fc @ coef
    Sigma = resid.T @ resid / T                                   # MLE covariance, as in the paper
    mu = F.mean(axis=0)
    Sf = np.atleast_2d(np.cov(F, rowvar=False, ddof=0))
    quad = float(mu @ np.linalg.solve(Sf, mu))
    stat = (T - N - K) / N * float(alpha @ np.linalg.solve(Sigma, alpha)) / (1.0 + quad)
    return {"F": float(stat), "pvalue": float(sps.f.sf(stat, N, T - N - K)), "df": (N, T - N - K), "alpha": pd.Series(alpha, index=returns.columns)}


# ------------------------------------------------------------------------------------------------------------------- panel OLS
def _codes(index: pd.MultiIndex, entity_level: int, time_level: int):
    e, _ = pd.factorize(index.get_level_values(entity_level))
    t, _ = pd.factorize(index.get_level_values(time_level))
    return e, t


def _demean(a: np.ndarray, e: np.ndarray, t: np.ndarray, entity: bool, time: bool, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """Project out entity and/or time means (alternating projections for the unbalanced two-way case)."""
    out = a.astype(float).copy()
    if out.ndim == 1:
        out = out[:, None]
    if entity and not time:
        for j in range(out.shape[1]):
            out[:, j] -= (np.bincount(e, out[:, j]) / np.bincount(e))[e]
        return out
    if time and not entity:
        for j in range(out.shape[1]):
            out[:, j] -= (np.bincount(t, out[:, j]) / np.bincount(t))[t]
        return out
    ne, nt = np.bincount(e), np.bincount(t)
    for _ in range(max_iter):
        before = out.copy()
        for j in range(out.shape[1]):
            out[:, j] -= (np.bincount(e, out[:, j]) / ne)[e]
            out[:, j] -= (np.bincount(t, out[:, j]) / nt)[t]
        if np.max(np.abs(out - before)) < tol:
            break
    return out


@dataclass
class PanelResult:
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    cov: pd.DataFrame
    resid: pd.Series
    r2_within: float
    nobs: int
    n_entities: int
    n_periods: int
    df_resid: int
    cov_type: str
    entity_effects: pd.Series | None = None
    time_effects: pd.Series | None = None

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.params, "se": self.bse, "t": self.tvalues, "p": self.pvalues})


def panel_ols(y: pd.Series, X: pd.DataFrame, entity_effects: bool = False, time_effects: bool = False, cov: str = "unadjusted", lags: int | None = None,
              entity_level: int = 0, time_level: int = 1) -> PanelResult:
    """Pooled / fixed-effects OLS on a (entity, time) MultiIndex.

    ``cov``: ``unadjusted``, ``robust`` (White), ``cluster_entity``, ``cluster_time``, ``cluster_two`` (Cameron-Gelbach-Miller) or ``driscoll_kraay``
    (Newey-West on the date-aggregated scores: robust to cross-sectional dependence and to autocorrelation, needs a long time dimension).
    Degrees of freedom subtract the absorbed effects except where the effect is nested in the cluster (the ``xtreg`` convention).
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()
    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    e, t = _codes(frame.index, entity_level, time_level)
    names = list(X.columns)
    yv, Xv = frame["__y__"].to_numpy(float), frame[names].to_numpy(float)
    n, N, T = len(yv), int(e.max()) + 1, int(t.max()) + 1
    absorbed = int(entity_effects) * N + int(time_effects) * (T - int(entity_effects))
    if entity_effects or time_effects:
        Z = _demean(np.column_stack([yv, Xv]), e, t, entity_effects, time_effects)
        yd, Xd = Z[:, 0], Z[:, 1:]
        full_names = names
    else:
        yd, Xd = yv, np.column_stack([np.ones(n), Xv])
        full_names = ["const"] + names
    k = Xd.shape[1]
    beta = np.linalg.solve(Xd.T @ Xd, Xd.T @ yd)
    u = yd - Xd @ beta
    df = n - k - absorbed
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    scores = Xd * u[:, None]
    if cov == "unadjusted":
        V = float(u @ u) / df * xtx_inv
    elif cov == "robust":
        V = xtx_inv @ (scores.T @ scores) @ xtx_inv * n / df
    elif cov in ("cluster_entity", "cluster_time"):
        codes = e if cov == "cluster_entity" else t
        sums = np.zeros((codes.max() + 1, k))
        np.add.at(sums, codes, scores)
        g = len(sums)
        nested = (cov == "cluster_entity" and entity_effects)
        dof = (n - 1.0) / (n - k - (absorbed if not nested else 0)) if not nested else 1.0
        V = xtx_inv @ (sums.T @ sums * g / (g - 1.0) * dof) @ xtx_inv
    elif cov == "cluster_two":
        def meat(codes):
            sums = np.zeros((codes.max() + 1, k))
            np.add.at(sums, codes, scores)
            g = len(sums)
            return sums.T @ sums * g / (g - 1.0), g
        m1, _ = meat(e)
        m2, _ = meat(t)
        m3, _ = meat(e * (T + 1) + t)
        V = xtx_inv @ (m1 + m2 - m3) @ xtx_inv
        w, vec = np.linalg.eigh((V + V.T) / 2)
        V = (vec * np.clip(w, 0, None)) @ vec.T
    elif cov == "driscoll_kraay":
        by_time = np.zeros((T, k))
        np.add.at(by_time, t, scores)
        # re-order the date totals chronologically
        labels = pd.Index(frame.index.get_level_values(time_level))
        chrono = pd.Series(np.arange(T), index=pd.factorize(labels)[1]).sort_index().to_numpy()
        by_time = by_time[chrono]
        lag = newey_west_lag(T) if lags is None else lags
        V = xtx_inv @ _bartlett_meat(by_time, lag) @ xtx_inv
    else:
        raise ValueError(f"unknown panel covariance '{cov}'")
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    tv = beta / se
    pv = 2.0 * (sps.t.sf(np.abs(tv), df) if cov == "unadjusted" else sps.norm.sf(np.abs(tv)))
    tss_w = float(((yd - yd.mean()) ** 2).sum()) if entity_effects or time_effects else float(((yv - yv.mean()) ** 2).sum())
    out = PanelResult(pd.Series(beta, index=full_names), pd.Series(se, index=full_names), pd.Series(tv, index=full_names), pd.Series(pv, index=full_names),
                      pd.DataFrame(V, index=full_names, columns=full_names), pd.Series(u, index=frame.index), 1.0 - float(u @ u) / tss_w, n, N, T, df, cov)
    if entity_effects:
        resid_raw = yv - Xv @ beta[: len(names)] if not time_effects else None
        if resid_raw is not None:
            alpha_i = pd.Series(resid_raw).groupby(e).mean()
            out.entity_effects = pd.Series(alpha_i.to_numpy(), index=pd.factorize(frame.index.get_level_values(entity_level))[1])
    return out


def random_effects(y: pd.Series, X: pd.DataFrame, entity_level: int = 0, time_level: int = 1) -> dict:
    """Swamy-Arora random-effects GLS: quasi-demean each variable by ``theta_i = 1 - sqrt(s_e^2 / (s_e^2 + T_i s_u^2))``."""
    if isinstance(X, pd.Series):
        X = X.to_frame()
    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    e, t = _codes(frame.index, entity_level, time_level)
    names = list(X.columns)
    yv, Xv = frame["__y__"].to_numpy(float), frame[names].to_numpy(float)
    n, N = len(yv), int(e.max()) + 1
    Ti = np.bincount(e)
    k = Xv.shape[1]
    within = _demean(np.column_stack([yv, Xv]), e, t, True, False)
    bw = np.linalg.lstsq(within[:, 1:], within[:, 0], rcond=None)[0]
    s2e = float(((within[:, 0] - within[:, 1:] @ bw) ** 2).sum()) / (n - N - k)
    ybar, Xbar = np.bincount(e, yv) / Ti, np.column_stack([np.bincount(e, Xv[:, j]) / Ti for j in range(k)])
    Zb = np.column_stack([np.ones(N), Xbar])
    bb = np.linalg.lstsq(Zb, ybar, rcond=None)[0]
    s2b = float(((ybar - Zb @ bb) ** 2).sum()) / (N - k - 1)
    Tbar = N / np.sum(1.0 / Ti)
    s2u = max(s2b - s2e / Tbar, 0.0)
    theta = 1.0 - np.sqrt(s2e / (s2e + Ti * s2u))
    th = theta[e]
    Yt = yv - th * ybar[e]
    Xt = np.column_stack([1.0 - th] + [Xv[:, j] - th * Xbar[e, j] for j in range(k)])
    beta = np.linalg.lstsq(Xt, Yt, rcond=None)[0]
    u = Yt - Xt @ beta
    V = float(u @ u) / (n - k - 1) * np.linalg.inv(Xt.T @ Xt)
    return {"params": pd.Series(beta, index=["const"] + names), "bse": pd.Series(np.sqrt(np.diag(V)), index=["const"] + names),
            "cov": pd.DataFrame(V, index=["const"] + names, columns=["const"] + names), "sigma_e2": s2e, "sigma_u2": s2u, "theta_mean": float(theta.mean()),
            "within_params": pd.Series(bw, index=names), "sigma2_gls": float(u @ u) / (n - k - 1), "xtx_inv": np.linalg.inv(Xt.T @ Xt)}


def hausman_test(y: pd.Series, X: pd.DataFrame) -> dict:
    """Hausman (1978): are the random-effects and fixed-effects slopes the same? A small p-value rejects RE (the effects correlate with X)."""
    fe = panel_ols(y, X, entity_effects=True)
    re = random_effects(y, X)
    names = list(fe.params.index)
    d = fe.params.to_numpy() - re["params"][names].to_numpy()
    # both covariances use the fixed-effects error variance ("sigmamore"), which keeps V_fe - V_re positive semi-definite in finite samples
    s2_fe = float(fe.resid.to_numpy() @ fe.resid.to_numpy()) / fe.df_resid
    V_re = re["xtx_inv"][1:, 1:] * s2_fe
    V = fe.cov.to_numpy() - V_re
    stat = float(d @ np.linalg.pinv(V) @ d)
    return {"chi2": stat, "df": len(names), "pvalue": float(sps.chi2.sf(stat, len(names))), "fe": fe.params, "re": re["params"][names]}


def long_from_wide(**frames: pd.DataFrame) -> pd.DataFrame:
    """Stack ``dates x assets`` frames into an (asset, date) MultiIndex frame, one column per keyword."""
    out = {name: f.stack(future_stack=True) if hasattr(f, "stack") else f for name, f in frames.items()}
    long = pd.DataFrame(out)
    long.index = long.index.set_names(["date", "asset"])
    return long.swaplevel().sort_index()
