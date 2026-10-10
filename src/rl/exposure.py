"""Choosing how much to hold by reinforcement learning, from past data only: fitted Q-iteration on an empirical model of a market.

The state of the market at a month-end is one of a few buckets (is the trend up, is volatility high) together with the exposure currently held; the action is the next exposure; the reward is the month's
mean-variance utility of the held exposure net of the cost of changing it. Because an investor's trades do not move the market, the reward of *every* action in a past month can be computed, so no exploration is
needed: the Q-function is fitted by value iteration on the empirical transitions of the months already observed, ``Q(m, p, a) = E[u(a; r_next) - cost(a, p)] + gamma E[ max_a' Q(m', a, a') ]``,
with the expectations taken over the observed months in state ``m``, shrunk toward zero reward when a state has been seen only a few times. A policy that uses it holds the exposure with the largest ``Q``.
"""

from __future__ import annotations

import numpy as np


def market_states(returns: np.ndarray, trend_window: int = 12, vol_window: int = 3) -> np.ndarray:
    """A state in ``0..3`` per month: bit 0 is a positive return over the last ``trend_window`` months, bit 1 a standard deviation over the last ``vol_window`` months above the median of the history so far.
    ``-1`` while there is not enough history. Uses only data up to and including each month."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.full(n, -1, dtype=int)
    vol = np.full(n, np.nan)
    for t in range(n):
        if t + 1 >= vol_window:
            vol[t] = np.std(r[t + 1 - vol_window:t + 1], ddof=1)
        if t + 1 < trend_window or not np.isfinite(vol[t]):
            continue
        trend = float(np.prod(1.0 + r[t + 1 - trend_window:t + 1]) - 1.0) > 0.0
        hist = vol[:t + 1][np.isfinite(vol[:t + 1])]
        high = vol[t] > np.median(hist)
        out[t] = int(trend) + 2 * int(high)
    return out


def fit_exposure_policy(states: np.ndarray, next_returns: np.ndarray, exposures=(0.0, 0.5, 1.0), cost: float = 0.001, risk_aversion: float = 5.0, gamma: float = 0.5, shrink: float = 6.0,
                        n_states: int = 4, sweeps: int = 200) -> np.ndarray:
    """``Q[m, p, a]`` fitted on the given history: ``states[s]`` is the state at the end of month ``s`` and ``next_returns[s]`` the return of the following month; entries with state ``-1`` or a non-finite return are
    ignored. Returns the array of shape ``(n_states, len(exposures), len(exposures))``."""
    e = np.asarray(exposures, dtype=float)
    k = len(e)
    s = np.asarray(states, dtype=int)
    r = np.asarray(next_returns, dtype=float)
    ok = (s >= 0) & np.isfinite(r)
    idx = np.flatnonzero(ok)
    # transitions s -> s+1 within the observed months (the next state must also be known)
    nxt = np.full(len(s), -1, dtype=int)
    nxt[:-1] = s[1:]
    reward = np.zeros((n_states, k, k))                                                       # mean reward over months in state m, for previous exposure p and new exposure a
    count = np.zeros(n_states)
    trans = np.zeros((n_states, n_states))
    for m in range(n_states):
        members = idx[s[idx] == m]
        count[m] = len(members)
        if len(members):
            rm = r[members]
            gain = (e[None, :] * rm[:, None] - 0.5 * risk_aversion * (e[None, :] * rm[:, None]) ** 2).mean(axis=0)     # (a,)
            reward[m] = gain[None, :] * (len(members) / (len(members) + shrink)) - cost * np.abs(e[None, :] - e[:, None])
            nm = nxt[members]
            for j in nm[nm >= 0]:
                trans[m, j] += 1.0
        else:
            reward[m] = -cost * np.abs(e[None, :] - e[:, None])
    row = trans.sum(axis=1, keepdims=True)
    P = np.where(row > 0, trans / np.maximum(row, 1.0), 1.0 / n_states)
    Q = np.zeros((n_states, k, k))
    for _ in range(sweeps):
        V = Q.max(axis=2)                                                                      # V[m', a]: value of arriving in state m' holding exposure a
        # Q[m, p, a] = reward[m, p, a] + gamma * sum_m' P[m, m'] * V[m', a]
        Q_new = reward + gamma * np.einsum("mj,ja->ma", P, V)[:, None, :]
        if np.abs(Q_new - Q).max() < 1e-12:
            Q = Q_new
            break
        Q = Q_new
    return Q
