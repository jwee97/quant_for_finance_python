"""Breadth, the fundamental law, and the transfer coefficient.

**Fundamental law** (Grinold 1989): ``IR = TC x IC x sqrt(BR)``. The information ratio of a portfolio is the transfer coefficient (how much of the alpha the constraints let through) times the skill of the
forecasts times the square root of the number of independent forecasts a year. ``BR`` is the part that is easy to overstate: 500 stocks are not 500 bets when they share a market factor, and a signal that
barely changes from one month to the next is not 12 bets a year.

**Effective breadth.** The cross-sectional count is the participation ratio of the eigenvalues of the correlation matrix of residual returns, ``(sum l)^2 / sum l^2``: ``N`` for uncorrelated assets, 1 when one
factor drives everything. The time count scales it by the number of non-overlapping forecast horizons a year and by the share of the signal that is new at each one, ``1 - rho`` with ``rho`` the
autocorrelation of the signal across one horizon. This is an estimate, not a law of nature; it is labelled as one wherever it is shown.

**Transfer coefficient.** With ``Sigma = L L'`` the risk-adjusted alpha is ``L^-1 alpha`` and the risk-adjusted position ``L' h``; the TC is their cosine. It is exactly the ratio of the information ratio a
holding ``h`` earns, ``h'alpha / sqrt(h' Sigma h)``, to the best any holding could earn, ``sqrt(alpha' Sigma^-1 alpha)``. A long-only constraint, a turnover cap, or sector limits lower it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..equity.qp import solve_qp


def effective_cross_section(returns: pd.DataFrame) -> float:
    """The number of independent bets in the cross-section: the participation ratio of the eigenvalues of the correlation matrix of ``returns``."""
    clean = returns.dropna(how="any")
    if clean.shape[0] < 3 or clean.shape[1] < 1:
        return float("nan")
    corr = np.corrcoef(clean.to_numpy(dtype=float), rowvar=False)
    corr = np.atleast_2d(corr)
    values = np.clip(np.linalg.eigvalsh(np.nan_to_num(corr)), 0.0, None)
    return float(values.sum() ** 2 / (values ** 2).sum()) if (values ** 2).sum() > 0 else float("nan")


def signal_autocorrelation(signal: pd.DataFrame, lag: int = 1, min_assets: int = 5) -> float:
    """The mean over dates of the cross-sectional rank correlation of the signal with itself ``lag`` periods earlier."""
    from ..models.regression import cross_sectional_ic

    ac = cross_sectional_ic(signal, signal.shift(-int(lag)).rename_axis(None), "spearman", min_assets).dropna()
    return float(ac.mean()) if len(ac) else float("nan")


def effective_breadth(residuals: pd.DataFrame, signal: pd.DataFrame | None = None, horizon: int = 1, periods_per_year: float = 252.0) -> dict:
    """Independent forecasts per year: ``N_eff x (periods_per_year / horizon) x (1 - rho)``.

    Returns the pieces so that each can be argued with: ``n_assets``, ``n_effective`` (cross-section), ``forecasts_per_year`` (non-overlapping horizons), ``rho`` (signal autocorrelation over one horizon,
    0 when no signal is given) and ``breadth``."""
    n_eff = effective_cross_section(residuals)
    rho = signal_autocorrelation(signal, horizon) if signal is not None else 0.0
    rho = 0.0 if not np.isfinite(rho) else float(np.clip(rho, 0.0, 0.999))
    per_year = float(periods_per_year) / max(int(horizon), 1)
    return {"n_assets": int(residuals.shape[1]), "n_effective": n_eff, "forecasts_per_year": per_year, "rho": rho, "breadth": n_eff * per_year * (1.0 - rho)}


def fundamental_law(ic: float, breadth: float, tc: float = 1.0) -> float:
    """``IR = TC x IC x sqrt(BR)``."""
    return float(tc * ic * np.sqrt(max(breadth, 0.0)))


def implied_ic(ir: float, breadth: float, tc: float = 1.0) -> float:
    """The skill that an information ratio implies at a given breadth and transfer coefficient: ``IR / (TC sqrt(BR))``."""
    return float(ir / (tc * np.sqrt(breadth))) if tc > 0 and breadth > 0 else float("nan")


def _cov(cov) -> np.ndarray:
    s = np.asarray(cov, dtype=float)
    return 0.5 * (s + s.T)


def unconstrained_ir(alpha, cov) -> float:
    """``sqrt(alpha' Sigma^-1 alpha)``: the information ratio of the best possible portfolio for these forecasts (per period of the alpha's horizon)."""
    a = np.asarray(alpha, dtype=float)
    return float(np.sqrt(max(a @ np.linalg.solve(_cov(cov), a), 0.0)))


def transfer_coefficient(alpha, cov, holdings) -> float:
    """The cosine of ``L^-1 alpha`` and ``L' h`` for ``Sigma = L L'``: the share of the unconstrained information ratio that the holdings ``h`` earn. 1 for ``h`` proportional to ``Sigma^-1 alpha``."""
    a, h = np.asarray(alpha, dtype=float), np.asarray(holdings, dtype=float)
    s = _cov(cov)
    L = np.linalg.cholesky(s + 1e-14 * np.trace(s) / len(a) * np.eye(len(a)))
    ra, rh = np.linalg.solve(L, a), L.T @ h
    den = np.linalg.norm(ra) * np.linalg.norm(rh)
    return float(ra @ rh / den) if den > 0 else float("nan")


def constrained_transfer(alpha, cov, lb=0.0, ub=None, risk_aversion: float = 1.0, net: float | None = None) -> dict:
    """The transfer coefficient of the best portfolio under box limits (``lb``, ``ub``; long-only is ``lb = 0``) and, if given, a net-exposure sum: maximise ``alpha'h - (lambda/2) h'Sigma h``.

    Returns the holdings, the TC, and the information ratios with and without the constraint, which is the price of the constraint in information-ratio terms."""
    a = np.asarray(alpha, dtype=float)
    s = _cov(cov)
    n = len(a)
    A = np.ones((1, n)) if net is not None else None
    r = solve_qp(risk_aversion * s, -a, A=A, l=None if net is None else [net], u=None if net is None else [net], lb=lb, ub=ub)
    h = np.asarray(r.x, dtype=float)
    ok = bool(np.isfinite(h).all() and np.abs(h).sum() > 0)
    tc = transfer_coefficient(a, s, h) if ok else float("nan")
    full = unconstrained_ir(a, s)
    return {"holdings": h, "tc": tc, "ir_unconstrained": full, "ir_constrained": tc * full if ok else float("nan"), "ok": ok}
