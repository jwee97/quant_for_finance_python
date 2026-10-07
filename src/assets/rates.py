"""Fixed income: yield-curve models, bond analytics and carry / roll-down.

* ``nelson_siegel`` / ``svensson``: smooth parametric curves. ``fit_nelson_siegel`` fits ``(beta0, beta1, beta2)`` by least squares for each decay ``lambda`` on a grid; the three betas are the
  level, slope and curvature of the curve (Diebold & Li 2006).
* ``dynamic_nelson_siegel``: fit the curve every day, treat the betas as a VAR(1) and forecast the whole curve (Diebold-Li); ``synthetic_curve_panel`` simulates such a world.
* ``bond_price`` / ``yield_to_maturity`` / ``duration`` / ``convexity`` / ``dv01`` / ``key_rate_durations``: the analytics of a fixed-coupon bond.
* ``bootstrap_zero_curve``: zero rates from par yields; ``forward_rate`` between two maturities.
* ``carry_rolldown``: the expected return of holding a bond for ``horizon`` years if the curve does not move: the coupon accrual (carry) plus the price gain from rolling down a
  steep curve to a lower yield.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize


def nelson_ns_loadings(tau, lam):
    tau = np.asarray(tau, dtype=float)
    x = lam * tau
    f1 = (1.0 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(tau), f1, f1 - np.exp(-x)])


def nelson_siegel(tau, beta0, beta1, beta2, lam):
    """Zero yield at maturity ``tau`` (years): ``b0 + b1 (1 - e^{-lam t})/(lam t) + b2 [(1 - e^{-lam t})/(lam t) - e^{-lam t}]``."""
    return nelson_ns_loadings(tau, lam) @ np.array([beta0, beta1, beta2])


def svensson(tau, beta0, beta1, beta2, beta3, lam1, lam2):
    tau = np.asarray(tau, dtype=float)
    x1, x2 = lam1 * tau, lam2 * tau
    f1 = (1 - np.exp(-x1)) / x1
    f2 = (1 - np.exp(-x2)) / x2
    return beta0 + beta1 * f1 + beta2 * (f1 - np.exp(-x1)) + beta3 * (f2 - np.exp(-x2))


def fit_nelson_siegel(maturities, yields, lam_grid=None) -> dict:
    """Least-squares fit: for each ``lambda`` on the grid the betas solve a linear regression; the best ``lambda`` minimises the residual sum of squares."""
    tau, y = np.asarray(maturities, float), np.asarray(yields, float)
    grid = np.linspace(0.2, 3.0, 57) if lam_grid is None else np.asarray(lam_grid)
    best = None
    for lam in grid:
        X = nelson_ns_loadings(tau, lam)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(((y - X @ beta) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, lam, beta)
    rss, lam, beta = best
    return {"beta0": float(beta[0]), "beta1": float(beta[1]), "beta2": float(beta[2]), "lambda": float(lam), "rmse": float(np.sqrt(rss / len(y)))}


def dynamic_nelson_siegel(yields: pd.DataFrame, lam: float = 0.5) -> dict:
    """Diebold-Li: with ``lambda`` fixed, the daily betas are an OLS regression of the cross-section of yields on the loadings. ``yields`` has maturities (years) as columns. Returns the
    factor panel ``level, slope, curvature`` (sign conventions as the loadings: ``beta1 = -slope``) and the fitted curves' RMSE."""
    tau = np.asarray(yields.columns, dtype=float)
    X = nelson_ns_loadings(tau, lam)
    pinv = np.linalg.pinv(X)
    betas = yields.to_numpy() @ pinv.T
    fitted = betas @ X.T
    return {"factors": pd.DataFrame(betas, index=yields.index, columns=["level", "slope", "curvature"]), "rmse": float(np.sqrt(np.mean((yields.to_numpy() - fitted) ** 2))), "lambda": lam, "loadings": X}


def forecast_curve(factors: pd.DataFrame, maturities, lam: float = 0.5, horizon: int = 21) -> pd.Series:
    """Diebold-Li forecast: a VAR(1) on the three factors (OLS), iterated ``horizon`` steps, mapped back to a curve."""
    from ..econometrics.var import fit_var

    var = fit_var(factors, 1)
    f = var.forecast(horizon).iloc[-1].to_numpy()
    return pd.Series(nelson_ns_loadings(np.asarray(maturities, float), lam) @ f, index=maturities)


def synthetic_curve_panel(n_days: int = 2000, maturities=(0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30), lam: float = 0.5, seed: int = 0, start: str = "2010-01-04") -> pd.DataFrame:
    """Daily zero yields from VAR(1) dynamics of Nelson-Siegel factors (level persistent, slope and curvature faster) plus measurement noise of about one basis point."""
    rng = np.random.default_rng(seed)
    A = np.diag([0.9995, 0.995, 0.99])
    mean = np.array([0.04, -0.015, -0.01])
    sd = np.array([0.0004, 0.0006, 0.0010])
    f = np.empty((n_days, 3))
    f[0] = mean
    for t in range(1, n_days):
        f[t] = mean + A @ (f[t - 1] - mean) + sd * rng.standard_normal(3) + np.array([-0.2, 0.3, 0.0]) * 0.0
    X = nelson_ns_loadings(np.asarray(maturities, float), lam)
    y = f @ X.T + 0.0001 * rng.standard_normal((n_days, len(maturities)))
    return pd.DataFrame(y, index=pd.bdate_range(start, periods=n_days), columns=list(maturities))


# ------------------------------------------------------------------------------------------------------------------ bond analytics
def cashflow_schedule(maturity: float, coupon: float, freq: int = 2, face: float = 100.0):
    n = int(round(maturity * freq))
    times = np.arange(1, n + 1) / freq
    flows = np.full(n, face * coupon / freq)
    flows[-1] += face
    return times, flows


def bond_price(maturity, coupon, ytm, freq: int = 2, face: float = 100.0) -> float:
    """Clean price (at a coupon date) of a fixed-coupon bond for a yield quoted with ``freq`` compounding."""
    t, cf = cashflow_schedule(maturity, coupon, freq, face)
    return float((cf / (1.0 + ytm / freq) ** (t * freq)).sum())


def bond_price_from_curve(maturity, coupon, zero_curve, freq: int = 2, face: float = 100.0) -> float:
    """Price from a zero curve ``zero_curve(t)`` (continuously compounded zero rates)."""
    t, cf = cashflow_schedule(maturity, coupon, freq, face)
    return float((cf * np.exp(-np.asarray([zero_curve(x) for x in t]) * t)).sum())


def yield_to_maturity(price, maturity, coupon, freq: int = 2, face: float = 100.0) -> float:
    return float(optimize.brentq(lambda y: bond_price(maturity, coupon, y, freq, face) - price, -0.5, 2.0, xtol=1e-14))


def duration(maturity, coupon, ytm, freq: int = 2, kind: str = "modified") -> float:
    """Macaulay duration (PV-weighted average time of the cash flows) or modified duration ``D_mac / (1 + y/freq)`` (the percentage price change per unit yield)."""
    t, cf = cashflow_schedule(maturity, coupon, freq)
    pv = cf / (1.0 + ytm / freq) ** (t * freq)
    mac = float((t * pv).sum() / pv.sum())
    return mac if kind == "macaulay" else mac / (1.0 + ytm / freq)


def convexity(maturity, coupon, ytm, freq: int = 2) -> float:
    t, cf = cashflow_schedule(maturity, coupon, freq)
    n = t * freq
    pv = cf / (1.0 + ytm / freq) ** n
    return float((pv * n * (n + 1)).sum() / (pv.sum() * freq ** 2 * (1.0 + ytm / freq) ** 2))


def dv01(maturity, coupon, ytm, freq: int = 2, face: float = 100.0) -> float:
    """Dollar value of a basis point: the price change for a one-basis-point fall in yield, by full revaluation (so it contains convexity)."""
    return float(bond_price(maturity, coupon, ytm - 1e-4, freq, face) - bond_price(maturity, coupon, ytm, freq, face))


def key_rate_durations(maturity, coupon, zero_curve, key_maturities=(1, 2, 3, 5, 7, 10, 20, 30), bump: float = 1e-4, freq: int = 2) -> pd.Series:
    """Price sensitivity to a triangular bump of each key zero rate (the bump is spread linearly between neighbouring key maturities). The sum is the effective duration."""
    keys = np.asarray(key_maturities, float)
    base = bond_price_from_curve(maturity, coupon, zero_curve, freq)
    out = {}
    for i, k in enumerate(keys):
        lo = keys[i - 1] if i else 0.0
        hi = keys[i + 1] if i + 1 < len(keys) else keys[-1] + 1.0

        def shock(t, lo=lo, k=k, hi=hi):
            return bump * (np.clip((t - lo) / (k - lo), 0, 1) if t <= k else np.clip((hi - t) / (hi - k), 0, 1))

        bumped = bond_price_from_curve(maturity, coupon, lambda t: zero_curve(t) + shock(t), freq)
        out[k] = -(bumped - base) / (base * bump)
    return pd.Series(out, name="krd")


def bootstrap_zero_curve(par_maturities, par_yields, freq: int = 2) -> pd.Series:
    """Zero (spot) rates, continuously compounded, from par-bond yields at semi-annual maturities by forward substitution: each par bond prices at 100, which pins the discount factor
    of its final cash flow given the earlier ones. Maturities must be a multiple of ``1/freq``; intermediate par yields are linearly interpolated."""
    mats = np.asarray(par_maturities, float)
    grid = np.arange(1, int(round(mats.max() * freq)) + 1) / freq
    py = np.interp(grid, mats, np.asarray(par_yields, float))
    df = np.empty(len(grid))
    for i, (t, y) in enumerate(zip(grid, py)):
        c = y / freq
        df[i] = (1.0 - c * df[:i].sum()) / (1.0 + c)
    return pd.Series(-np.log(df) / grid, index=grid, name="zero")


def forward_rate(zero_curve, t1: float, t2: float) -> float:
    """Continuously compounded forward rate between ``t1`` and ``t2``: ``(z2 t2 - z1 t1) / (t2 - t1)``."""
    return float((zero_curve(t2) * t2 - zero_curve(t1) * t1) / (t2 - t1))


def carry_rolldown(maturity: float, coupon: float, par_curve, horizon: float = 0.25, freq: int = 2, financing: float = 0.0) -> dict:
    """Expected return over ``horizon`` years if the PAR curve does not move. ``par_curve(t)`` gives the par yield. Buy at the yield for ``maturity`` (priced at par if ``coupon`` equals
    it), hold, and revalue at the yield for the SHORTER maturity ``maturity - horizon``: ``carry`` is the coupon accrual minus ``financing``, ``rolldown`` the price gain from the
    lower yield. Returns both and the total, in percent of the starting price."""
    y0 = float(par_curve(maturity))
    y1 = float(par_curve(max(maturity - horizon, 1.0 / freq)))
    p0 = bond_price(maturity, coupon, y0, freq)
    p1 = bond_price(max(maturity - horizon, 1.0 / freq), coupon, y1, freq)
    accrual = coupon * horizon * 100.0
    carry_ = (accrual - financing * horizon * p0) / p0
    roll = (p1 - p0) / p0
    return {"carry": float(carry_), "rolldown": float(roll), "total": float(carry_ + roll), "yield_start": y0, "yield_end": y1}
