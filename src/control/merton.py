"""Merton's portfolio problem: the closed forms, and the same problem with proportional trading costs solved by dynamic programming on the risky fraction.

An investor with power utility of relative risk aversion ``gamma`` splits wealth between a riskless asset earning ``r`` and a risky one with drift ``mu`` and volatility ``sigma``. Without frictions the optimal
risky fraction is the constant ``pi* = (mu - r) / (gamma sigma^2)`` whatever the horizon (Merton 1969, 1971); with consumption from an infinite horizon the optimal consumption rate is the constant
``c / W = (rho - (1 - gamma)(r + (mu - r)^2 / (2 gamma sigma^2))) / gamma``.

With a proportional cost ``kappa`` on every dollar traded, the investor stops rebalancing inside a *no-trade region* around ``pi*`` and trades to its edge when the fraction drifts out (Magill and Constantinides 1976;
Davis and Norman 1990). :class:`CostDP` finds the region in discrete time by backward induction on the pre-trade risky fraction, using that power utility scales with wealth, so the fraction is the only state.
For small costs its half-width is about ``(3 pi*^2 (1 - pi*)^2 kappa / (2 gamma))^(1/3)`` (Janecek and Shreve 2004; Gerhold, Guasoni, Muhle-Karbe and Schachermayer 2014): it grows with the cube root of the cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def merton_fraction(mu: float, r: float, sigma: float, gamma: float) -> float:
    return (mu - r) / (gamma * sigma ** 2)


def merton_value_growth(mu: float, r: float, sigma: float, gamma: float) -> float:
    """The certainty-equivalent growth rate of wealth under the optimal fraction: ``r + (mu - r)^2 / (2 gamma sigma^2)``."""
    return r + (mu - r) ** 2 / (2.0 * gamma * sigma ** 2)


def merton_value(W: np.ndarray | float, tau: float, mu: float, r: float, sigma: float, gamma: float):
    """The value ``V(W, tau) = W^(1-gamma) / (1-gamma) exp((1-gamma) g tau)`` of terminal power utility ``tau`` years ahead (log utility for ``gamma = 1``: ``log W + g tau``)."""
    g = merton_value_growth(mu, r, sigma, gamma)
    W = np.asarray(W, dtype=float)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(W) + g * tau
    return W ** (1.0 - gamma) / (1.0 - gamma) * np.exp((1.0 - gamma) * g * tau)


def merton_consumption_rate(mu: float, r: float, sigma: float, gamma: float, rho: float) -> float:
    """Infinite horizon, utility ``e^{-rho t} c^(1-gamma)/(1-gamma)`` of consumption: the optimal ``c / W``. It must be positive for the problem to be well posed."""
    nu = (rho - (1.0 - gamma) * merton_value_growth(mu, r, sigma, gamma)) / gamma
    if nu <= 0:
        raise ValueError("the problem is not well posed: rho - (1 - gamma) * growth must be positive")
    return nu


def gauss_hermite_returns(mu: float, r: float, sigma: float, dt: float, n: int = 15) -> tuple[np.ndarray, np.ndarray, float]:
    """Quadrature for one period's lognormal gross risky return: ``(gross returns, weights, riskless gross return)``."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    gross = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * x)
    return gross, w / w.sum(), float(np.exp(r * dt))


@dataclass
class CostSolution:
    grid: np.ndarray            # pre-trade risky fractions
    target: np.ndarray          # the fraction to trade to, from each pre-trade fraction (first period)
    value: np.ndarray
    lower: float                # edges of the no-trade region at the first period
    upper: float
    merton: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)


class CostDP:
    """Discrete-time Merton problem with proportional cost ``kappa`` per dollar traded, power utility, ``periods`` periods of length ``dt`` and a terminal utility of wealth.

    The state is the risky fraction ``p`` before trading. Trading to ``q`` costs ``kappa |q - p|`` of wealth; the period's gross growth of wealth is ``(1 - kappa|q - p|) (q R + (1 - q) R_f)`` and the fraction
    drifts to ``q R / (q R + (1 - q) R_f)``. With power utility, ``J_t(p) = max_q E[ growth^(1-gamma) J_{t+1}(p') ]`` (a product, since utility is homothetic); with log utility the recursion is additive."""

    def __init__(self, mu: float, r: float, sigma: float, gamma: float, kappa: float, dt: float = 1 / 12, periods: int = 120, grid: int = 241, quad: int = 15, pmax: float = 1.5, pmin: float = 0.0):
        if kappa < 0 or gamma <= 0 or sigma <= 0 or periods < 1 or grid < 11:
            raise ValueError("kappa >= 0, gamma > 0, sigma > 0, periods >= 1, grid >= 11")
        self.mu, self.r, self.sigma, self.gamma, self.kappa, self.dt, self.periods = mu, r, sigma, gamma, kappa, dt, int(periods)
        self.p = np.linspace(pmin, pmax, grid)
        self.R, self.w, self.Rf = gauss_hermite_returns(mu, r, sigma, dt, quad)
        self.log = abs(gamma - 1.0) < 1e-12

    def solve(self) -> CostSolution:
        p, R, w, Rf, k, g = self.p, self.R, self.w, self.Rf, self.kappa, self.gamma
        n = len(p)
        q = p                                                                                            # candidate post-trade fractions: the same grid
        mix = q[:, None] * R[None, :] + (1.0 - q[:, None]) * Rf                                          # (q, R): gross growth before cost
        nxt = q[:, None] * R[None, :] / mix                                                              # next period's pre-trade fraction
        cost = 1.0 - k * np.abs(q[None, :] - p[:, None])                                                 # (p, q): wealth kept after trading p -> q
        if (cost <= 0).any():
            raise ValueError("kappa too large for the grid")
        J = np.zeros(n) if self.log else np.ones(n)                                                      # terminal: log W or W^(1-g)/(1-g), up to the factor 1/(1-g)
        for _ in range(self.periods):
            Jn = np.empty_like(nxt)
            for j in range(R.size):
                Jn[:, j] = np.interp(nxt[:, j], p, J)
            if self.log:
                cont = (w[None, :] * (np.log(mix) + Jn)).sum(axis=1)                                     # (q,)
                total = np.log(cost) + cont[None, :]
                J_new = total.max(axis=1)
            else:
                e = 1.0 - g
                cont = (w[None, :] * mix ** e * Jn).sum(axis=1)
                total = cost ** e * cont[None, :]
                J_new = total.max(axis=1) if e > 0 else total.min(axis=1)
            J = J_new
        target = q[total.argmax(axis=1) if (self.log or g < 1) else total.argmin(axis=1)]
        star = merton_fraction(self.mu, self.r, self.sigma, g)
        stay = np.isclose(target, p, atol=1e-12)
        inside = p[stay]
        lower, upper = (float(inside.min()), float(inside.max())) if inside.size else (star, star)
        return CostSolution(p, target, J, lower, upper, star)


def davis_norman_half_width(mu: float, r: float, sigma: float, gamma: float, kappa: float) -> float:
    """The small-cost approximation to the half-width of the no-trade region, ``(3 pi*^2 (1 - pi*)^2 kappa / (2 gamma))^(1/3)``."""
    s = merton_fraction(mu, r, sigma, gamma)
    return float((1.5 * s ** 2 * (1.0 - s) ** 2 * kappa / gamma) ** (1.0 / 3.0))
