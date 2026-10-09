"""Alpha model construction: standardise the factors, make them independent, weigh them by what they have earned, and find out what each adds to the others.

A factor is a table of dates by assets, higher meaning "expect more". Four steps turn a handful of them into one alpha.

1. **Standardise** (:func:`standardize`): winsorise, subtract the cross-sectional mean and divide by the cross-sectional standard deviation on every date, so that every factor has the same
   scale and a weight means the same thing for each. Without it the factor with the widest dispersion decides the answer.
2. **Orthogonalise** (:func:`gram_schmidt`, :func:`symmetric_orthogonalize`): value and earnings yield, momentum and revisions say much the same thing. Gram-Schmidt keeps the first factor, removes from
   the second what the first explains (the residual of a cross-sectional regression on each date), from the third what the first two explain, and so on, so each factor carries only information
   the earlier ones did not. The order is a decision: the factor placed first keeps all it shares.
3. **Weigh** (:func:`max_ir_weights`): the information coefficient (IC) of a factor on a date is the correlation of the factor with the next period's returns across assets. If the factors are
   orthonormal on every date, the composite's IC is exactly ``w'IC / ||w||`` and its information ratio (mean over standard deviation of the composite IC) is ``w'mu / sqrt(w'Sigma w)``, where ``mu`` is
   the mean and ``Sigma`` the covariance of the factors' ICs over time. That ratio is maximised by ``w = Sigma^-1 mu`` (Qian, Hua and Sorensen 2007, ch. 8) and the best achievable ratio is
   ``sqrt(mu' Sigma^-1 mu)``. Weights follow the covariance as well as the means: a factor whose IC moves against another's is worth more than its own IC says.
4. **Attribute** (:func:`marginal_contributions`): a Fama-MacBeth regression of the next period's returns on all the factors at once gives each factor's return after controlling for the others,
   which is what tells apart a factor that earns from one that merely proxies for another.

:func:`optimal_alpha` runs the steps walk-forward on a grid of rebalance dates. The weights used on date ``t`` come from ICs that were known on ``t`` (an IC measured for ``s`` needs the returns of the
``horizon`` periods after ``s``), never from the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.regression import cross_sectional_ic
from ..stats.panel import fama_macbeth
from ..signals.transform import cross_sectional_demean, cross_sectional_zscore, winsorize
from .qp import solve_qp


# ------------------------------------------------------------------------------------------------------------------ standardise
def standardize(factor: pd.DataFrame, investable: pd.DataFrame | None = None, winsorize_quantile: float = 0.01) -> pd.DataFrame:
    """Winsorise, demean and divide by the cross-sectional standard deviation on every date. Assets that are not investable (or missing) stay missing."""
    work = factor.where(investable.reindex_like(factor).fillna(False)) if investable is not None else factor.copy()
    if winsorize_quantile:
        work = winsorize(work, winsorize_quantile)
    return cross_sectional_zscore(cross_sectional_demean(work))


def standardize_all(factors: dict[str, pd.DataFrame], investable: pd.DataFrame | None = None, winsorize_quantile: float = 0.01) -> dict[str, pd.DataFrame]:
    return {k: standardize(v, investable, winsorize_quantile) for k, v in factors.items()}


# ------------------------------------------------------------------------------------------------------------------ orthogonalise
def _stack(factors: dict[str, pd.DataFrame], missing: str):
    """The factors as one dates x assets x factors cube, the mask of assets seen on a date (at least one factor observed) and the filled cube.

    ``missing='zero'`` fills a missing value with zero (neutral after standardising); ``'drop'`` keeps the NaN so the asset is dropped from that date's orthogonalisation."""
    if missing not in ("zero", "drop"):
        raise ValueError("missing must be 'zero' (a missing factor is neutral) or 'drop'")
    names = list(factors)
    first = factors[names[0]]
    index, columns = first.index, first.columns
    raw = np.stack([factors[n].reindex(index=index, columns=columns).to_numpy(dtype=float) for n in names], axis=-1)       # dates x assets x factors
    cube = np.where(np.isfinite(raw), raw, 0.0) if missing == "zero" else raw
    seen = np.isfinite(raw).all(axis=-1) if missing == "drop" else np.isfinite(raw).any(axis=-1)
    return names, index, columns, cube, seen


def _unstack(names, index, columns, cube, observed) -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame(np.where(observed, cube[:, :, k], np.nan), index=index, columns=columns) for k, name in enumerate(names)}


def gram_schmidt(factors: dict[str, pd.DataFrame], order: list[str] | None = None, missing: str = "zero", rescale: bool = True) -> dict[str, pd.DataFrame]:
    """Sequential cross-sectional orthogonalisation (modified Gram-Schmidt, date by date).

    ``order`` lists the factors from the one that keeps everything it shares to the one that keeps only what is new. On each date a factor is replaced by the residual of a cross-sectional regression
    (with an intercept) on the factors before it. ``missing='zero'`` treats a missing value as neutral (zero after standardising) so an asset with one missing factor is still ranked on the others;
    ``'drop'`` leaves an asset missing in every output if it is missing in any input. With ``rescale`` each residual is divided by its cross-sectional standard deviation, so the outputs are orthogonal
    and standardised. A factor that adds nothing (an exact combination of the earlier ones) comes out as zeros, not as noise."""
    order = list(order) if order is not None else list(factors)
    if set(order) != set(factors) or len(order) != len(factors):
        raise ValueError("order must name every factor exactly once")
    names, index, columns, cube, seen = _stack({k: factors[k] for k in order}, missing)
    out = np.full_like(cube, np.nan)
    for t in range(cube.shape[0]):
        ok = seen[t]
        if ok.sum() < len(names) + 2:
            continue
        X = cube[t][ok]
        Q = np.empty_like(X)
        for k in range(X.shape[1]):
            v = X[:, k] - X[:, k].mean()
            for j in range(k):
                denom = Q[:, j] @ Q[:, j]
                if denom > 1e-12:
                    v = v - (Q[:, j] @ v) / denom * Q[:, j]
            if rescale:
                sd = v.std(ddof=1)
                v = v / sd if sd > 1e-9 * max(X[:, k].std(ddof=1), 1e-300) else np.zeros_like(v)
            Q[:, k] = v
        out[t, ok] = Q
    return _unstack(names, index, columns, out, seen)


def symmetric_orthogonalize(factors: dict[str, pd.DataFrame], missing: str = "zero") -> dict[str, pd.DataFrame]:
    """Loewdin (symmetric) orthogonalisation: the orthonormal set closest to the originals, with no factor privileged. ``F (F'F)^(-1/2)`` on every date (after removing each factor's mean), scaled to
    unit variance. Unlike Gram-Schmidt the answer does not depend on the order of the factors, and every output stays as close as it can to its own input."""
    names, index, columns, cube, seen = _stack(factors, missing)
    out = np.full_like(cube, np.nan)
    for t in range(cube.shape[0]):
        ok = seen[t]
        if ok.sum() < len(names) + 2:
            continue
        X = cube[t][ok]
        X = X - X.mean(axis=0)
        vals, vecs = np.linalg.eigh(X.T @ X)
        vals = np.clip(vals, 1e-12 * max(vals.max(), 1e-300), None)
        W = (vecs / np.sqrt(vals)) @ vecs.T
        out[t, ok] = X @ W * np.sqrt(len(X) - 1)
    return _unstack(names, index, columns, out, seen)


def factor_correlation(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The average over dates of the cross-sectional correlation matrix of the factors (how much of what they say is the same)."""
    names = list(factors)
    index = factors[names[0]].index
    total, count = np.zeros((len(names), len(names))), 0
    cube = np.stack([factors[n].reindex(index).to_numpy(dtype=float) for n in names], axis=-1)
    for t in range(cube.shape[0]):
        ok = np.isfinite(cube[t]).all(axis=-1)
        if ok.sum() > len(names) + 2:
            c = np.corrcoef(cube[t][ok], rowvar=False)
            if np.isfinite(c).all():
                total += c
                count += 1
    return pd.DataFrame(total / max(count, 1), index=names, columns=names)


# ------------------------------------------------------------------------------------------------------------------ weigh
def information_coefficients(factors: dict[str, pd.DataFrame], forward: pd.DataFrame, method: str = "spearman", min_assets: int = 10) -> pd.DataFrame:
    """The IC of every factor on every date against ``forward`` (dates x factors); ``forward[t]`` is the return AFTER ``t``."""
    return pd.DataFrame({k: cross_sectional_ic(v, forward, method=method, min_assets=min_assets) for k, v in factors.items()})


def ic_moments(ic: pd.DataFrame, shrink: float = 0.0) -> tuple[pd.Series, pd.DataFrame]:
    """The mean of each factor's IC and the covariance of the ICs across time, the second shrunk toward its diagonal by ``shrink`` (0 = sample, 1 = no cross terms): with a few years of monthly
    ICs the off-diagonal terms are mostly noise, and the inverse of a noisy covariance puts large offsetting weights on them."""
    clean = ic.dropna()
    mean = clean.mean()
    cov = pd.DataFrame(np.atleast_2d(np.cov(clean.to_numpy(), rowvar=False)), index=ic.columns, columns=ic.columns)
    if shrink:
        cov = (1.0 - shrink) * cov + shrink * pd.DataFrame(np.diag(np.diag(cov)), index=cov.index, columns=cov.columns)
    return mean, cov


def max_ir_weights(mean: pd.Series, cov: pd.DataFrame, nonnegative: bool = False, ridge: float = 1e-10) -> pd.Series:
    """The weights on standardised, orthogonal factors that maximise ``w'mean / sqrt(w'cov w)``, scaled to sum of absolute weights one: ``cov^-1 mean`` or, with ``nonnegative`` (a factor may be left
    out but never traded the other way), the same problem with ``w >= 0``."""
    mu = mean.to_numpy(dtype=float)
    S = cov.to_numpy(dtype=float) + ridge * np.eye(len(mu)) * max(np.trace(cov.to_numpy()) / len(mu), 1e-12)
    if nonnegative:
        w = np.zeros_like(mu)
        if (mu > 0).any():                                    # with no positive mean there is nothing to buy: zero weights, not a rescaled rounding error
            r = solve_qp(S, -mu, lb=0.0)
            w = np.clip(r.x, 0.0, None) if r.ok else w
            w[w < 1e-8 * max(w.max(), 1e-300)] = 0.0
    else:
        w = np.linalg.solve(S, mu)
    gross = np.abs(w).sum()
    return pd.Series(w / gross if gross > 0 else w, index=mean.index)


def composite_ir(weights: pd.Series, mean: pd.Series, cov: pd.DataFrame) -> float:
    """``w'mean / sqrt(w'cov w)``: the information ratio of the IC of the composite (per period, not annualised)."""
    w = weights.reindex(mean.index).to_numpy(dtype=float)
    var = float(w @ cov.to_numpy(dtype=float) @ w)
    return float(w @ mean.to_numpy(dtype=float) / np.sqrt(var)) if var > 0 else float("nan")


def best_possible_ir(mean: pd.Series, cov: pd.DataFrame) -> float:
    """``sqrt(mean' cov^-1 mean)``, the information ratio the optimal weights reach."""
    mu = mean.to_numpy(dtype=float)
    return float(np.sqrt(max(mu @ np.linalg.solve(cov.to_numpy(dtype=float) + 1e-12 * np.eye(len(mu)), mu), 0.0)))


def combine_factors(factors: dict[str, pd.DataFrame], weights: pd.Series | pd.DataFrame, renormalize: bool = True) -> pd.DataFrame:
    """The weighted sum of the factors, standardised across assets on every date so the composite has the same scale whatever the weights. ``weights`` is a Series (one weight per factor, constant) or a
    frame (dates x factors, forward-filled onto the dates of the factors)."""
    names = list(factors)
    index = factors[names[0]].index
    if isinstance(weights, pd.Series):
        w = pd.DataFrame([weights.reindex(names).to_numpy()] * len(index), index=index, columns=names)
    else:
        w = weights.reindex(columns=names).reindex(index.union(weights.index)).ffill().reindex(index)
    total = sum(factors[n].reindex(index).fillna(0.0) * w[n].to_numpy()[:, None] for n in names)
    seen = np.logical_or.reduce([factors[n].reindex(index).notna().to_numpy() for n in names])
    total = total.where(seen & w.notna().any(axis=1).to_numpy()[:, None])
    return cross_sectional_zscore(total) if renormalize else total


# ------------------------------------------------------------------------------------------------------------------ walk-forward
@dataclass
class AlphaModel:
    alpha: pd.DataFrame          # the composite alpha at each rebalance date (standardised across assets)
    weights: pd.DataFrame        # the weights on the (orthogonalised) factors used at each date
    ic: pd.DataFrame             # each factor's IC for the period AFTER each date (only usable once that period has ended)
    expected_ir: pd.Series       # the best information ratio the data known at each date support
    factors: dict                # the standardised (and orthogonalised) factors the weights apply to


def optimal_alpha(factors: dict[str, pd.DataFrame], forward: pd.DataFrame, horizon: int = 1, window: int = 36, min_obs: int = 12, shrink: float = 0.3, orthogonalize: bool | str = False,
                  order: list[str] | None = None, nonnegative: bool = False, investable: pd.DataFrame | None = None, method: str = "spearman", min_assets: int = 10) -> AlphaModel:
    """The optimal alpha model, walk-forward, on a grid of periods (one row per rebalance date).

    ``forward[t]`` is the return over the ``horizon`` periods after ``t``. The weights at date ``t`` use only the ICs of dates ``s <= t - horizon`` (their forward windows have ended by ``t``), the last
    ``window`` of them and at least ``min_obs``; before that the composite is the equal-weighted sum. ``orthogonalize`` is ``False``, ``'gram_schmidt'`` (or ``True``) or ``'symmetric'``."""
    std = standardize_all(factors, investable)
    if orthogonalize is True or orthogonalize == "gram_schmidt":
        std = gram_schmidt(std, order)
    elif orthogonalize == "symmetric":
        std = symmetric_orthogonalize(std)
    elif orthogonalize not in (False, None):
        raise ValueError("orthogonalize must be False, True/'gram_schmidt' or 'symmetric'")
    names = list(std)
    index = std[names[0]].index
    ic = information_coefficients(std, forward.reindex(index), method, min_assets).reindex(index)
    rows, ir = {}, {}
    equal = pd.Series(1.0 / len(names), index=names)
    for i, t in enumerate(index):
        known = ic.iloc[: max(i - horizon + 1, 0)].dropna().iloc[-window:]
        if len(known) < min_obs:
            rows[t], ir[t] = equal, np.nan
            continue
        mean, cov = ic_moments(known, shrink)
        rows[t] = max_ir_weights(mean, cov, nonnegative)
        ir[t] = best_possible_ir(mean, cov)
    weights = pd.DataFrame(rows).T.reindex(columns=names)
    alpha = combine_factors(std, weights)
    return AlphaModel(alpha, weights, ic, pd.Series(ir), std)


# ------------------------------------------------------------------------------------------------------------------ attribute
def marginal_contributions(factors: dict[str, pd.DataFrame], realized: pd.DataFrame, lag: int = 1, orthogonalize: bool | str = False, order: list[str] | None = None,
                           investable: pd.DataFrame | None = None) -> pd.DataFrame:
    """What each factor earns with the others held fixed: a Fama-MacBeth regression of ``realized[t]`` (the return of the period ending at ``t``) on all the factors as of ``t - lag``.

    Returns one row per factor with the mean slope (the return of a portfolio with unit exposure to that factor and none to the others), its Newey-West t-statistic, the information ratio of the slope
    (mean over standard deviation across dates) and the share of dates on which it was positive, next to the factor's stand-alone mean IC for comparison: a factor with a good IC and a small
    slope is a proxy for another."""
    std = standardize_all(factors, investable)
    if orthogonalize is True or orthogonalize == "gram_schmidt":
        std = gram_schmidt(std, order)
    elif orthogonalize == "symmetric":
        std = symmetric_orthogonalize(std)
    fm = fama_macbeth(realized, {k: v.fillna(0.0) for k, v in std.items()}, lag=lag, add_constant=True)
    slopes = fm.gammas.drop(columns="const")
    alone = information_coefficients({k: v.shift(lag) for k, v in std.items()}, realized, "spearman")
    return pd.DataFrame({"slope": fm.mean.drop("const"), "t": fm.tvalues.drop("const"), "ir": slopes.mean() / slopes.std(ddof=1), "hit_rate": (slopes > 0).mean(),
                         "mean_ic_alone": alone.mean()})
