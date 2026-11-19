"""Three ways to a factor model of returns, all giving the same object: ``Sigma = B F B' + diag(D)``.

* **Time series** (:func:`fit_time_series`): the factors are given (market, rates, a long-short portfolio) and each asset is regressed on them: ``B`` are the slopes, ``D`` the residual variances.
* **Cross-sectional / fundamental** (:func:`fit_cross_sectional`): the exposures are given (sector membership, value, size, quality: characteristics known at the start of each period) and on every date the
  cross-section of returns is regressed on them: the slopes are that period's *factor returns*. ``F`` is their covariance, ``D`` the variance of the residuals. Exposures are lagged one period, so a
  characteristic never explains the return that produced it.
* **Statistical** (:func:`fit_statistical`): the factors are the leading principal components of the returns; their number is the count of correlation-matrix eigenvalues above the Marchenko-Pastur bound.

Everything is per period (daily if the returns are daily): ``annualise`` multiplies by the periods in a year.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class FactorModelError(ValueError):
    pass


@dataclass
class FactorModel:
    kind: str
    exposures: pd.DataFrame            # assets x factors
    factor_returns: pd.DataFrame       # dates x factors
    factor_cov: pd.DataFrame           # factors x factors (per period)
    specific_var: pd.Series            # per asset, per period
    groups: dict = field(default_factory=dict)       # factor -> "market" | "sector" | "style" | "statistical" | "given"
    r2: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))     # per asset
    n_obs: int = 0

    @property
    def assets(self) -> list:
        return list(self.exposures.index)

    @property
    def factors(self) -> list:
        return list(self.exposures.columns)

    def covariance(self, annualise: float | None = None) -> pd.DataFrame:
        B = self.exposures.to_numpy(dtype=float)
        S = B @ self.factor_cov.to_numpy(dtype=float) @ B.T + np.diag(self.specific_var.reindex(self.assets).to_numpy(dtype=float))
        out = pd.DataFrame(0.5 * (S + S.T), index=self.assets, columns=self.assets)
        return out * annualise if annualise else out

    def volatility(self, annualise: float | None = None) -> pd.Series:
        v = pd.Series(np.sqrt(np.diag(self.covariance().to_numpy())), index=self.assets)
        return v * np.sqrt(annualise) if annualise else v

    def summary(self) -> dict:
        return {"kind": self.kind, "assets": len(self.assets), "factors": len(self.factors), "n_obs": self.n_obs, "median_r2": float(self.r2.median()) if len(self.r2) else float("nan"),
                "systematic_share": float(1.0 - self.specific_var.sum() / max(np.trace(self.covariance().to_numpy()), 1e-300))}


def _weights(n: int, halflife: float | None) -> np.ndarray:
    if not halflife:
        return np.ones(n)
    return 0.5 ** (np.arange(n)[::-1] / float(halflife))


def _wcov(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    m = w @ x
    xc = x - m
    return (xc * w[:, None]).T @ xc / (1.0 - (w ** 2).sum())


def _r2(returns: pd.DataFrame, resid: pd.DataFrame) -> pd.Series:
    tv = returns.var(ddof=1)
    return (1.0 - resid.var(ddof=1) / tv.where(tv > 0)).clip(lower=0.0, upper=1.0)


def fit_time_series(returns: pd.DataFrame, factors: pd.DataFrame, window: int | None = None, halflife: float | None = None, groups: dict | None = None, min_obs: int = 60) -> FactorModel:
    """Regress each asset's returns on the given factor returns (with an intercept). ``window`` keeps the last rows; ``halflife`` weights recent rows more (for the betas, the factor covariance and the
    residual variance alike)."""
    r, f = returns.align(factors, join="inner", axis=0)
    if window:
        r, f = r.tail(window), f.tail(window)
    f = f.dropna(how="any")
    r = r.loc[f.index]
    if len(f) < min_obs:
        raise FactorModelError(f"need at least {min_obs} dates in common, have {len(f)}")
    K = f.shape[1]
    X = np.column_stack([np.ones(len(f)), f.to_numpy(dtype=float)])
    w = _weights(len(f), halflife)
    B, spec = {}, {}
    resid = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for a in r.columns:
        y = r[a].to_numpy(dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < min_obs:
            continue
        sw = np.sqrt(w[ok])
        beta, *_ = np.linalg.lstsq(X[ok] * sw[:, None], y[ok] * sw, rcond=None)
        e = y[ok] - X[ok] @ beta
        B[a] = beta[1:]
        spec[a] = float((w[ok] * e ** 2).sum() / w[ok].sum() * ok.sum() / max(ok.sum() - K - 1, 1))
        resid.loc[r.index[ok], a] = e
    if not B:
        raise FactorModelError("no asset has enough history")
    kept = list(B)
    exposures = pd.DataFrame(B, index=f.columns).T.loc[kept]
    Fcov = pd.DataFrame(_wcov(f.to_numpy(dtype=float), w), index=f.columns, columns=f.columns)
    return FactorModel("time_series", exposures, f, Fcov, pd.Series(spec).loc[kept], dict(groups or {c: "given" for c in f.columns}), _r2(r[kept], resid[kept]), int(len(f)))


def _sector_dummies(sectors: pd.Series, assets: list) -> pd.DataFrame:
    s = sectors.reindex(assets).fillna("unclassified")
    return pd.get_dummies(s).astype(float).reindex(assets)


def fit_cross_sectional(returns: pd.DataFrame, styles: dict | None = None, sectors: pd.Series | None = None, window: int | None = None, halflife: float | None = None,
                        weights: pd.DataFrame | None = None, min_assets: int = 8, market: bool = True) -> FactorModel:
    """The fundamental factor model by cross-sectional regression on every date.

    ``styles``: ``{name: frame of dates x assets}`` of characteristics (already standardised; this function does not scale them) as of the END of each period: the return of date ``t`` is regressed on the
    exposures of date ``t-1``. ``sectors``: a series asset -> label; the sector dummies and the market intercept would be collinear, so the sector returns are constrained to sum to zero (weighted by sector
    size) and the market factor is the average return. ``weights``: optional dates x assets regression weights (typically the square root of market value); equal otherwise.

    ``B`` is the exposure on the last date with a full set of exposures; ``F`` the covariance of the daily factor returns over the window; ``D`` the variance of each asset's residuals."""
    styles = dict(styles or {})
    if not styles and sectors is None and not market:
        raise FactorModelError("give style exposures, sectors, or the market factor")
    r = returns.copy()
    if window:
        r = r.tail(window)
    assets = list(r.columns)
    sec = _sector_dummies(sectors, assets) if sectors is not None else pd.DataFrame(index=assets)
    names = (["market"] if market else []) + list(sec.columns) + list(styles)
    groups = {**({"market": "market"} if market else {}), **{c: "sector" for c in sec.columns}, **{k: "style" for k in styles}}
    lagged = {k: v.reindex(r.index.union(v.index)).sort_index().shift(1).reindex(r.index) for k, v in styles.items()}
    f_rows, e_rows = {}, {}
    for t in r.index:
        y = r.loc[t]
        cols = []
        if market:
            cols.append(pd.Series(1.0, index=assets, name="market"))
        cols += [sec[c].rename(c) for c in sec.columns]
        cols += [lagged[k].loc[t].reindex(assets).rename(k) for k in styles]
        X = pd.concat(cols, axis=1)
        ok = y.notna() & X.notna().all(axis=1)
        if ok.sum() < max(min_assets, len(names) + 2):
            continue
        Xo, yo = X[ok].to_numpy(dtype=float), y[ok].to_numpy(dtype=float)
        sw = np.ones(len(yo)) if weights is None else np.sqrt(np.clip(weights.loc[t].reindex(assets)[ok].fillna(0.0).to_numpy(dtype=float), 0.0, None))
        if market and len(sec.columns):
            c = np.zeros((1, Xo.shape[1]))
            idx = [names.index(s) for s in sec.columns]
            c[0, idx] = sec[sec.columns][ok].sum().to_numpy(dtype=float)                       # sector returns, weighted by how many names each has, sum to zero
            Xo, yo, sw = np.vstack([Xo, 1e3 * c]), np.append(yo, 0.0), np.append(sw, 1.0)
        beta, *_ = np.linalg.lstsq(Xo * sw[:, None], yo * sw, rcond=None)
        f_rows[t] = pd.Series(beta, index=X.columns)
        e_rows[t] = pd.Series(y[ok].to_numpy(dtype=float) - X[ok].to_numpy(dtype=float) @ beta, index=y[ok].index)
    if len(f_rows) < 30:
        raise FactorModelError(f"only {len(f_rows)} dates had enough assets and exposures to fit")
    fr = pd.DataFrame(f_rows).T[names]
    er = pd.DataFrame(e_rows).T.reindex(columns=assets)
    w = _weights(len(fr), halflife)
    Fcov = pd.DataFrame(_wcov(fr.to_numpy(dtype=float), w), index=names, columns=names)
    ew = _weights(len(er), halflife)[:, None] * er.notna().to_numpy()
    spec = pd.Series(np.where(ew.sum(axis=0) > 0, (ew * np.nan_to_num(er.to_numpy(dtype=float)) ** 2).sum(axis=0) / np.where(ew.sum(axis=0) > 0, ew.sum(axis=0), 1.0), np.nan), index=assets)
    last = {}
    for k in styles:
        s = styles[k].dropna(how="all")
        last[k] = s.iloc[-1].reindex(assets) if len(s) else pd.Series(np.nan, index=assets)
    B = pd.concat(([pd.Series(1.0, index=assets, name="market")] if market else []) + [sec[c].rename(c) for c in sec.columns] + [last[k].rename(k) for k in styles], axis=1)[names]
    keep = B.notna().all(axis=1) & spec.notna()
    if keep.sum() < 3:
        raise FactorModelError("fewer than three assets have both exposures and residuals")
    return FactorModel("cross_sectional", B[keep], fr, Fcov, spec[keep], groups, _r2(r.loc[:, keep], er.reindex(columns=r.columns[keep.to_numpy()])), int(len(fr)))


def fit_statistical(returns: pd.DataFrame, n_factors: int | None = None, window: int | None = None, halflife: float | None = None, min_obs: int = 60) -> FactorModel:
    """Principal-component factors: ``n_factors`` leading components of the covariance (the number of correlation eigenvalues above the Marchenko-Pastur bound when ``None``; at least one)."""
    from ..features.pca import marchenko_pastur_bounds

    r = returns.tail(window) if window else returns
    r = r.dropna(axis=1, thresh=min_obs).fillna(0.0)
    if len(r) < min_obs or r.shape[1] < 3:
        raise FactorModelError(f"need at least {min_obs} dates and 3 assets")
    x = r.to_numpy(dtype=float)
    w = _weights(len(x), halflife)
    S = _wcov(x, w)
    values, vectors = np.linalg.eigh(S)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if n_factors is None:
        vol = np.sqrt(np.clip(np.diag(S), 1e-300, None))
        corr = S / np.outer(vol, vol)
        _, upper = marchenko_pastur_bounds(len(x), x.shape[1])
        n_factors = max(int((np.linalg.eigvalsh(corr) > upper).sum()), 1)
    k = int(max(1, min(n_factors, x.shape[1] - 1)))
    V = vectors[:, :k]
    names = [f"PC{i + 1}" for i in range(k)]
    f = pd.DataFrame((x - w @ x / w.sum()) @ V, index=r.index, columns=names)
    resid = (x - w @ x / w.sum()) - f.to_numpy() @ V.T
    spec = pd.Series((w[:, None] * resid ** 2).sum(axis=0) / w.sum(), index=r.columns)
    return FactorModel("statistical", pd.DataFrame(V, index=r.columns, columns=names), f, pd.DataFrame(_wcov(f.to_numpy(), w), index=names, columns=names), spec, {n: "statistical" for n in names},
                       _r2(r, pd.DataFrame(resid, index=r.index, columns=r.columns)), int(len(x)))
