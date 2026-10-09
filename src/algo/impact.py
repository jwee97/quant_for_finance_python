"""The cost of trading: impact while trading, impact that stays, the spread, and the pre-trade estimate.

The model is Almgren-Chriss (2000) with the power-law temporary impact Almgren, Thum, Hauptmann and Li (2005) measured. For a slice of ``q`` shares in an interval in which ``V`` shares trade

* **temporary impact**   ``eta * sigma * (q / V)^beta`` of the price, paid on this slice only (square-root-ish: ``beta`` near one half to six tenths);
* **permanent impact**   ``gamma * sigma * q / adv`` of the price, which moves the market for everything that follows; a slice pays half of its own permanent impact, so the total over a whole order is
  ``gamma sigma X^2 / (2 adv)`` whatever the schedule (the Almgren-Chriss result), which is why the optimisation below ignores it;
* **spread**             half the quoted spread on every marketable share.

``expected_cost`` and ``timing_variance`` give the expected shortfall of a SCHEDULE and the variance that price moves add to it. They use expected volumes and no randomness, and the simulator in
:mod:`src.algo.simulate` reproduces them (to Monte Carlo error) when volumes are not noisy: that agreement is tested, and it is what allows the optimiser's answers to be trusted in the simulator.

:func:`istar_estimate` is Kissell's I-Star model, the practitioner's pre-trade estimate: ``I* = a1 Size^a2 sigma^a3`` for a trade done at once and
``MI = b1 I* POV^a4 + (1 - b1) I*`` for one done at participation ``POV``. Its default coefficients are ILLUSTRATIVE (the scale is typical of large US stocks); calibrate them to your own executions.
"""

from __future__ import annotations

import numpy as np

from .market import Market, Order


def temporary_fraction(q, volume, sigma: float, eta: float, beta: float):
    """Temporary impact as a fraction of price for a slice ``q`` against ``volume`` shares of other trading (arrays broadcast)."""
    q = np.maximum(np.asarray(q, float), 0.0)
    volume = np.maximum(np.asarray(volume, float), 1e-9)
    return eta * sigma * (q / volume) ** beta


def permanent_fraction(q, adv: float, sigma: float, gamma: float):
    """Permanent impact as a fraction of price for a slice of ``q`` shares."""
    return gamma * sigma * np.maximum(np.asarray(q, float), 0.0) / adv


def remaining_after(schedule) -> np.ndarray:
    """``x_i``: the shares still to trade AFTER slice ``i`` (the last is zero)."""
    s = np.asarray(schedule, float)
    return s.sum() - np.cumsum(s)


def expected_cost(schedule, order: Order, market: Market, alpha_bps: float = 0.0, marketable: float = 1.0) -> dict:
    """The expected implementation shortfall of ``schedule`` (shares per interval of the order's window) in dollars and in bps of the order's value at the arrival price.

    ``temporary + permanent + spread`` are the costs of trading; ``alpha`` is the expected price drift (``alpha_bps`` over the window, spread evenly over its intervals) earned or paid by shares bought or sold
    while it unfolds. ``marketable`` is the share of each slice that crosses the spread. Costs are positive numbers: a larger one is worse."""
    q = np.asarray(schedule, float)
    n = len(q)
    window = order.window(market)
    volume = market.expected_volume()[window]
    ref = order.reference(market)
    if len(volume) != n:
        raise ValueError("the schedule must have one entry per interval of the order's window")
    temp = float((ref * q * temporary_fraction(q, volume, market.sigma, market.eta, market.beta)).sum())
    perm = 0.5 * ref * market.gamma * market.sigma * float(q.sum()) ** 2 / market.adv
    spread = 0.5 * ref * market.spread * marketable * float(q.sum())
    drift = np.linspace(0.0, 1.0, n + 1)[:-1] * alpha_bps * 1e-4                      # the expected move so far at the START of each interval, as a fraction
    alpha = order.side * ref * float((drift * q).sum())
    total = temp + perm + spread + alpha
    value = ref * float(q.sum())
    bps = lambda x: 1e4 * x / value if value > 0 else 0.0
    return {"temporary": temp, "permanent": perm, "spread": spread, "alpha": alpha, "total": total,
            "temporary_bps": bps(temp), "permanent_bps": bps(perm), "spread_bps": bps(spread), "alpha_bps": bps(alpha), "total_bps": bps(total)}


def timing_variance(schedule, order: Order, market: Market) -> dict:
    """The variance of the shortfall that price moves add: ``ref^2 sigma^2 sum_i w_i x_i^2`` with ``x_i`` the shares still unexecuted after slice ``i`` and ``w_i`` the interval's share of a day's variance.
    Returns the dollar variance and standard deviation and the standard deviation in bps of the order's value."""
    q = np.asarray(schedule, float)
    window = order.window(market)
    w = market.variance_weights()[window]
    ref = order.reference(market)
    x = remaining_after(q)
    var = ref ** 2 * market.sigma ** 2 * float((w * x ** 2).sum())
    value = ref * float(q.sum())
    return {"variance": var, "std": float(np.sqrt(var)), "std_bps": 1e4 * float(np.sqrt(var)) / value if value > 0 else 0.0}


def istar_estimate(shares: float, adv: float, sigma_annual: float, pov: float, a1: float = 708.0, a2: float = 0.55, a3: float = 0.71, a4: float = 0.5, b1: float = 0.98) -> dict:
    """Kissell's I-Star pre-trade cost estimate in bps. ``sigma_annual`` is the annualised volatility as a decimal (0.30 = 30%) and ``pov`` the participation rate ``Q / (Q + V)`` of the trade.

    ``I* = a1 (Q / adv)^a2 (100 sigma)^a3 / 100^a3`` is the instantaneous impact of trading everything at once; the market impact of a trade at participation ``pov`` is
    ``MI = b1 I* pov^a4 + (1 - b1) I*``: the larger part is temporary and scales with how hard the trade pushes, the rest is permanent. The default coefficients are illustrative, not estimates."""
    if shares <= 0 or adv <= 0 or sigma_annual <= 0 or not 0 < pov <= 1:
        raise ValueError("shares, adv, sigma_annual > 0 and 0 < pov <= 1")
    size = shares / adv
    istar = a1 * size ** a2 * sigma_annual ** a3
    mi = b1 * istar * pov ** a4 + (1.0 - b1) * istar
    return {"istar_bps": float(istar), "market_impact_bps": float(mi), "temporary_bps": float(b1 * istar * pov ** a4), "permanent_bps": float((1.0 - b1) * istar), "size": float(size), "pov": float(pov)}
