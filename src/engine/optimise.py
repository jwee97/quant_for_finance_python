"""Cost-aware portfolio optimisation, impact calibration and capacity estimates.

* :func:`cost_aware_weights` solves the one-period rebalancing problem with trading costs in the objective::

      max_w   alpha' w  -  (lambda / 2) w' Sigma w  -  sum_i c_i |w_i - w0_i|  -  sum_i k_i |w_i - w0_i|^1.5

  with ``alpha`` the expected return per period (in the same units as the weights), ``Sigma`` the covariance per period, ``c`` the proportional cost (spread, fees: from the cost schedule
  through :func:`proportional_costs`) and ``k`` the market-impact coefficient (the square-root law makes the cost of a trade grow with its size to the power 1.5). Proportional costs create
  a NO-TRADE REGION: a weight stays where it is unless the marginal alpha net of risk exceeds the cost of moving it, which is what turns a noisy signal into a low-turnover portfolio.
  Limits on gross, net and single weights are enforced.
* :func:`proportional_costs` reads a :class:`~src.engine.costs.CostSchedule` and gives each instrument's one-way cost per unit of notional (half spread plus commission).
* :func:`calibrate_sqrt_impact` fits the square-root impact coefficient ``Y`` in ``slippage = Y sigma sqrt(Q / ADV)`` to executed trades (a regression through the origin), so the
  simple model can be replaced by a calibrated one.
* :func:`capacity_curve` scales a run's trading up and down and reports how net P&L erodes: linear costs (fees, spread) scale with size, impact with size to the power 1.5, so the
  AUM at which half of the gross edge is gone is a defensible capacity estimate. It needs average daily volume and volatility per instrument (not observable in synthetic data: pass
  them in).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .costs import CostSchedule


def proportional_costs(registry, costs: CostSchedule, prices: dict, notional: float = 1_000_000.0) -> pd.Series:
    """One-way cost per unit of notional (a fraction) for each instrument id at the given prices, for a trade of ``notional`` (spread, fees and slippage; impact excluded)."""
    out = {}
    for iid, px in prices.items():
        inst = registry.get(iid)
        unit = inst.notional(px, 1.0)
        if not unit > 0:
            continue
        q = notional / unit
        c = costs.estimate(inst, q, px)
        out[iid] = (c.price_distance * abs(q) * inst.contract_multiplier + c.fee) / notional
    return pd.Series(out)


def cost_aware_weights(alpha: pd.Series, cov: pd.DataFrame, current: pd.Series | None = None, cost: pd.Series | float = 0.0, impact: pd.Series | float = 0.0, risk_aversion: float = 5.0,
                       max_gross: float = 2.0, max_net: float | None = None, max_weight: float = 1.0) -> pd.DataFrame:
    """The weights maximising alpha net of risk and trading costs (see the module docstring), as a frame: ``w0``, ``w``, ``trade``, ``alpha``, ``cost`` per instrument.

    ``cost`` and ``impact`` may be scalars or per-instrument series; units are fractions of capital. The problem is convex (an L1 cost plus a power-1.5 cost); it is solved by SLSQP on
    the split ``w = w0 + u - v`` with ``u, v >= 0``."""
    ids = list(alpha.index)
    n = len(ids)
    a = alpha.to_numpy(float)
    S = cov.reindex(index=ids, columns=ids).to_numpy(float)
    w0 = np.zeros(n) if current is None else current.reindex(ids).fillna(0.0).to_numpy(float)
    c = np.broadcast_to(np.asarray(cost.reindex(ids).fillna(0.0) if isinstance(cost, pd.Series) else cost, float), (n,)).copy()
    k = np.broadcast_to(np.asarray(impact.reindex(ids).fillna(0.0) if isinstance(impact, pd.Series) else impact, float), (n,)).copy()

    def split(x):
        return x[:n], x[n:]

    def neg(x):
        u, v = split(x)
        w = w0 + u - v
        d = u + v
        return float(-(a @ w) + 0.5 * risk_aversion * w @ S @ w + c @ d + k @ d ** 1.5)

    def grad(x):
        u, v = split(x)
        w = w0 + u - v
        g_w = -a + risk_aversion * S @ w
        d = u + v
        g_d = c + 1.5 * k * np.sqrt(d + 1e-12)
        return np.concatenate([g_w + g_d, -g_w + g_d])

    cons = [{"type": "ineq", "fun": lambda x: max_gross - np.abs(w0 + split(x)[0] - split(x)[1]).sum()}]
    if max_net is not None:
        cons += [{"type": "ineq", "fun": lambda x: max_net - (w0 + split(x)[0] - split(x)[1]).sum()}, {"type": "ineq", "fun": lambda x: max_net + (w0 + split(x)[0] - split(x)[1]).sum()}]
    bounds = [(0.0, 2 * max_weight)] * (2 * n)
    cons += [{"type": "ineq", "fun": lambda x: max_weight - (w0 + split(x)[0] - split(x)[1])}, {"type": "ineq", "fun": lambda x: max_weight + (w0 + split(x)[0] - split(x)[1])}]
    res = minimize(neg, np.zeros(2 * n), jac=grad, bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    u, v = split(res.x)
    w = w0 + u - v
    d = np.abs(w - w0)
    d[d < 1e-7] = 0.0                                                  # numerical dust is not a trade
    w = w0 + np.sign(w - w0) * d
    return pd.DataFrame({"w0": w0, "w": w, "trade": w - w0, "alpha": a, "cost": c * d + k * d ** 1.5}, index=ids)


def calibrate_sqrt_impact(trades: pd.DataFrame) -> dict:
    """Fit ``slippage_bps = Y x sigma_bps x sqrt(Q / ADV)`` through the origin. ``trades`` needs ``slippage_bps`` (the signed-adverse cost against arrival, positive = paid), ``quantity``,
    ``adv`` and ``sigma`` (daily, a fraction). Returns ``Y``, its standard error, R-squared and the number of trades."""
    df = trades.dropna(subset=["slippage_bps", "quantity", "adv", "sigma"])
    df = df[(df["adv"] > 0) & (df["sigma"] > 0) & (df["quantity"] != 0)]
    x = (df["sigma"] * 1e4 * np.sqrt(df["quantity"].abs() / df["adv"])).to_numpy(float)
    y = df["slippage_bps"].to_numpy(float)
    if len(x) < 5 or not (x @ x) > 0:
        return {"Y": float("nan"), "se": float("nan"), "r2": float("nan"), "n": int(len(x))}
    Y = float(x @ y / (x @ x))
    resid = y - Y * x
    se = float(np.sqrt((resid @ resid) / max(len(x) - 1, 1) / (x @ x)))
    r2 = float(1.0 - (resid @ resid) / (y @ y)) if (y @ y) > 0 else float("nan")
    return {"Y": Y, "se": se, "r2": r2, "n": int(len(x))}


def capacity_curve(result, adv_notional: dict | float, sigma_daily: dict | float, multiples=(0.5, 1, 2, 5, 10, 25, 50, 100), y: float = 0.5, registry=None) -> pd.DataFrame:
    """Net P&L of a run as its size is multiplied.

    Gross P&L (before costs) scales linearly with size. The run's actual fees and spreads scale linearly too. Market impact is re-estimated for every fill at the scaled size with the
    square-root law (``y x sigma x sqrt(k Q / ADV)`` per unit, so total impact grows as ``k^1.5``); the impact the run already paid is replaced by it. ``adv_notional`` and ``sigma_daily``
    are per instrument id (or one number for all). Returns a frame by multiple with gross, linear costs, impact, net and the share of gross edge kept."""
    j = result.journal
    j = j[j["category"] != "transfer"]
    costs_lin = -float(j.loc[j["category"].isin(["spread", "fee", "slippage"]), "pnl"].sum())
    paid_impact = -float(j.loc[j["category"] == "impact", "pnl"].sum())
    gross_after_costs = float(j["pnl"].sum())
    interest = float(j.loc[j["category"] == "interest", "pnl"].sum())
    strategy_pnl = gross_after_costs - interest                         # what the positions made net of the costs actually paid (idle-cash interest is not strategy P&L)
    gross = strategy_pnl + costs_lin + paid_impact
    fills = result.fills
    rows = []
    for k in multiples:
        imp = 0.0
        for f in fills.itertuples():
            adv = adv_notional.get(f.instrument_id, np.nan) if isinstance(adv_notional, dict) else adv_notional
            sig = sigma_daily.get(f.instrument_id, np.nan) if isinstance(sigma_daily, dict) else sigma_daily
            if not (np.isfinite(adv) and adv > 0 and np.isfinite(sig)):
                continue
            notional = abs(f.quantity) * f.price * (registry.get(f.instrument_id).contract_multiplier if registry is not None else 1.0)
            part = min(k * notional / adv, 1.0)
            imp += k * notional * y * sig * np.sqrt(part)
        net = k * gross - k * costs_lin - imp
        rows.append({"multiple": k, "gross": k * gross, "linear_costs": k * costs_lin, "impact": imp, "net": net, "edge_kept": net / (k * gross) if gross else float("nan")})
    return pd.DataFrame(rows).set_index("multiple")


def capacity_estimate(curve: pd.DataFrame, keep: float = 0.5) -> float:
    """The largest size multiple at which at least ``keep`` of the gross edge survives costs (interpolated; ``nan`` if even the smallest multiple fails, the last multiple if none fails)."""
    s = curve["edge_kept"].dropna()
    if s.empty or s.iloc[0] < keep:
        return float("nan")
    ok = s[s >= keep]
    if len(ok) == len(s):
        return float(s.index[-1])
    i = len(ok)
    x0, x1, y0, y1 = s.index[i - 1], s.index[i], s.iloc[i - 1], s.iloc[i]
    return float(x0 + (x1 - x0) * (y0 - keep) / (y0 - y1))
