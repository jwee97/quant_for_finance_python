"""How much capital a strategy can run, whether strategies that look different are the same bet, and how much to lever.

**Capacity** (Grinold and Kahn, *Advances*, ch. 31). A trade of ``Q`` units in an asset with daily volatility ``sigma`` and average daily volume ``ADV`` moves the price against you by about
``c sigma sqrt(Q / ADV)`` (the square-root law; ``c`` of order one). Scaling capital ``A`` scales every trade, so the cost per unit of capital grows with ``sqrt(A)`` and the strategy's net return falls
with capital: its *capacity* is the capital at which the net Sharpe ratio is half the one at zero impact.

**Crowding** (ch. 30). Two books can hold different names and the same risk. :func:`crowding_report` compares them on positions, on factor exposures, and on the correlation of their returns.

**Gearing** (ch. 29). For a return stream with mean ``mu`` and variance ``sigma^2`` per period, the leverage that maximises the long-run growth ``L mu - L^2 sigma^2 / 2`` is ``mu / sigma^2`` (Kelly); estimation error
makes full Kelly too much, and a fraction is used. For Brownian motion with drift ``L mu`` and volatility ``L sigma`` the drawdown from the running peak is a reflected Brownian motion whose long-run
distribution is exponential: the share of time spent at least ``D`` below the peak is ``exp(-2 mu D / (L sigma^2))``. (It is not the chance that a drawdown *ever* reaches ``D``: over a long enough history
that is close to one.) A limit on that share limits the leverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def impact_cost(trades: pd.DataFrame, aum: float, sigma: pd.DataFrame, adv: pd.DataFrame, c: float = 1.0, linear_bps: float = 0.0) -> pd.Series:
    """The cost per day as a fraction of capital: ``sum_i |dw_i| (c sigma_i sqrt(|dw_i| A / ADV_i) + linear)`` for trades ``dw`` as fractions of capital, ``sigma`` the daily volatility (as a fraction) and ``ADV``
    the average daily volume in currency."""
    q = trades.abs()
    adv_ = adv.reindex_like(trades).where(lambda x: x > 0)
    per = c * sigma.reindex_like(trades) * np.sqrt(q * float(aum) / adv_)
    cost = (q * (per.fillna(0.0) + linear_bps * 1e-4)).sum(axis=1)
    return cost


def capacity_curve(gross: pd.Series, trades: pd.DataFrame, grid, sigma: pd.DataFrame, adv: pd.DataFrame, c: float = 1.0, linear_bps: float = 0.0) -> pd.DataFrame:
    """The net annualised return and Sharpe ratio at each capital in ``grid`` (``gross`` is the daily return before costs, ``trades`` the daily change in holdings as fractions of capital)."""
    rows = {}
    for a in grid:
        net = gross.reindex(trades.index).fillna(0.0) - impact_cost(trades, a, sigma, adv, c, linear_bps)
        sd = net.std(ddof=1)
        rows[float(a)] = {"ann_return": float(net.mean() * ANN), "sharpe": float(net.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan")}
    return pd.DataFrame(rows).T.rename_axis("aum")


def capacity(curve: pd.DataFrame, floor: float = 0.5) -> dict:
    """The capital at which the net Sharpe ratio falls to ``floor`` times the smallest-capital Sharpe, by log-linear interpolation along the curve. ``capacity`` is a number, ``'below grid'`` when even the
    smallest capital is under the floor, ``'above grid'`` when the largest is not, or ``'n/a'`` when the zero-impact Sharpe is not positive."""
    s = curve["sharpe"].dropna()
    if s.empty or s.iloc[0] <= 0:
        return {"capacity": "n/a", "sharpe_at_start": float(s.iloc[0]) if len(s) else float("nan")}
    level = floor * float(s.iloc[0])
    if s.iloc[-1] > level:
        return {"capacity": "above grid", "sharpe_at_start": float(s.iloc[0]), "level": level}
    below = np.where(s.to_numpy() <= level)[0][0]
    if below == 0:
        return {"capacity": "below grid", "sharpe_at_start": float(s.iloc[0]), "level": level}
    x0, x1 = np.log(s.index[below - 1]), np.log(s.index[below])
    y0, y1 = s.iloc[below - 1], s.iloc[below]
    return {"capacity": float(np.exp(x0 + (level - y0) / (y1 - y0) * (x1 - x0))), "sharpe_at_start": float(s.iloc[0]), "level": level}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def crowding_report(books: dict, exposures: pd.DataFrame | None = None, returns: pd.DataFrame | None = None) -> dict:
    """Pairwise similarity of books (each a Series of holdings by asset): ``positions`` the cosine of the holdings, ``factors`` (if ``exposures``, assets x factors, is given) the cosine of the factor-exposure vectors
    ``B' h``, and ``returns`` (if the return panel is given) the correlation of the books' returns ``returns @ h``. Books that hold different names but share factor exposure score low on the first and high
    on the others: the diversification between them is less than it looks."""
    names = list(books)
    if len(names) < 2:
        raise ValueError("need at least two books")
    assets = sorted(set().union(*[set(b.index) for b in books.values()]))
    H = pd.DataFrame({k: pd.Series(b).reindex(assets).fillna(0.0) for k, b in books.items()})
    out = {"positions": pd.DataFrame([[_cos(H[a].to_numpy(), H[b].to_numpy()) for b in names] for a in names], index=names, columns=names)}
    if exposures is not None:
        X = exposures.reindex(assets).fillna(0.0)
        F = pd.DataFrame({k: X.T.to_numpy() @ H[k].to_numpy() for k in names}, index=X.columns)
        out["factors"] = pd.DataFrame([[_cos(F[a].to_numpy(), F[b].to_numpy()) for b in names] for a in names], index=names, columns=names)
        out["factor_exposures"] = F
    if returns is not None:
        P = pd.DataFrame({k: returns.reindex(columns=assets).fillna(0.0).to_numpy() @ H[k].to_numpy() for k in names}, index=returns.index)
        out["returns"] = P.corr()
        v = np.clip(np.linalg.eigvalsh(P.corr().to_numpy()), 0.0, None)
        out["effective_independent_books"] = float(v.sum() ** 2 / (v ** 2).sum()) if (v ** 2).sum() > 0 else float("nan")
    return out


def growth_rate(leverage, mu: float, sigma: float, rf: float = 0.0):
    """The expected log growth of a stream with mean excess return ``mu`` and volatility ``sigma`` per period at ``leverage``: ``rf + L mu - L^2 sigma^2 / 2``."""
    L = np.asarray(leverage, dtype=float)
    return rf + L * mu - 0.5 * L ** 2 * sigma ** 2


def kelly_leverage(returns: pd.Series, fraction: float = 0.5, cap: float | None = None) -> dict:
    """The growth-optimal leverage ``mu / sigma^2`` of the stream, scaled by ``fraction`` (half Kelly by default) and optionally capped. Also the standard error of the mean, which says how much of the
    estimate is noise: the leverage at one standard error below the mean is what a cautious reading of the same data gives."""
    r = returns.dropna()
    mu, var = float(r.mean()), float(r.var(ddof=1))
    se = float(np.sqrt(var / len(r)))
    full = mu / var if var > 0 else float("nan")
    L = full * fraction
    if cap is not None and np.isfinite(L):
        L = float(min(L, cap))
    return {"full_kelly": full, "leverage": L, "low": (mu - se) / var * fraction if var > 0 else float("nan"), "growth_at_leverage": float(growth_rate(L, mu, np.sqrt(var))) if np.isfinite(L) else float("nan"),
            "growth_at_full": float(growth_rate(full, mu, np.sqrt(var))) if np.isfinite(full) else float("nan"), "mu_t": mu / se if se > 0 else float("nan")}


def leverage_for_drawdown(mu: float, sigma: float, depth: float, probability: float) -> float:
    """The leverage at which the long-run share of time spent at least ``depth`` below the running peak equals ``probability``, for Brownian motion with per-unit-leverage drift ``mu`` and volatility ``sigma``
    (same time unit): ``L = -2 mu D / (sigma^2 ln p)``. ``mu`` must be positive: with no drift there is no stationary drawdown distribution."""
    if mu <= 0 or sigma <= 0 or not 0 < probability < 1 or depth <= 0:
        raise ValueError("need mu > 0, sigma > 0, depth > 0 and a probability strictly between 0 and 1")
    return float(-2.0 * mu * depth / (sigma ** 2 * np.log(probability)))
