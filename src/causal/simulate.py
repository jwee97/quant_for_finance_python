"""Simulated worlds with a known causal effect, used to validate the estimators before they touch real data (Stage 37)."""

from __future__ import annotations

import numpy as np


def confounded_plr(n: int, rng: np.random.Generator, theta: float = 0.5, p: int = 10) -> dict:
    """y = theta d + h(X) + e, d = g(X) + v, with nonlinear g and h. Regressing y on d alone is biased; the nuisance functions are nonlinear."""
    X = rng.normal(size=(n, p))
    g = np.sin(X[:, 0]) + X[:, 1] ** 2 - 1.0 + 0.5 * X[:, 2] * X[:, 3]
    h = 2.0 * np.sin(X[:, 0]) + 1.5 * (X[:, 1] ** 2 - 1.0) + X[:, 2] * X[:, 3] + 0.5 * X[:, 4]
    d = g + rng.normal(size=n)
    y = theta * d + h + rng.normal(size=n)
    return {"y": y, "d": d, "X": X, "theta": theta, "m_d": g, "m_y": theta * g + h}


def endogenous_iv(n: int, rng: np.random.Generator, theta: float = 0.5, strength: float = 0.6) -> dict:
    """An unobserved u drives both d and y; z moves d and has no direct effect on y."""
    u, z = rng.normal(size=n), rng.normal(size=n)
    d = strength * z + u + rng.normal(size=n)
    y = theta * d + 2.0 * u + rng.normal(size=n)
    return {"y": y, "d": d, "z": z, "theta": theta}


def did_panel(n_units: int, rng: np.random.Generator, effect: float = 1.0, trend_gap: float = 0.0) -> dict:
    """Two groups, two periods. ``trend_gap`` is the extra change the treated group would have had WITHOUT treatment (0 = parallel trends)."""
    unit = np.repeat(np.arange(n_units), 2)
    treated = np.repeat((np.arange(n_units) < n_units // 2).astype(float), 2)
    post = np.tile([0.0, 1.0], n_units)
    level = np.repeat(rng.normal(size=n_units), 2)
    y = level + 0.5 * treated + 0.3 * post + trend_gap * treated * post + effect * treated * post + rng.normal(size=2 * n_units)
    return {"y": y, "treated": treated, "post": post, "unit": unit, "theta": effect}


def heterogeneous_effect(n: int, rng: np.random.Generator, p: int = 5) -> dict:
    """tau(x) = 0.5 + 1.0 x0 (x0 > 0 gets a bigger effect), confounded treatment."""
    X = rng.normal(size=(n, p))
    d = 0.8 * X[:, 1] + 0.5 * np.sin(X[:, 2]) + rng.normal(size=n)
    tau = 0.5 + 1.0 * X[:, 0]
    y = tau * d + X[:, 1] + 0.5 * X[:, 2] ** 2 + rng.normal(size=n)
    return {"y": y, "d": d, "X": X, "tau": tau}
