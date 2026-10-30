"""A Gymnasium environment for portfolio allocation on the platform's panel, with the backtest engine's timing and cost model.

**Timing.** Row ``t`` of the arrays holds the features known at the close of day ``t``; the action chosen then is a book of target weights. As in the backtest engine (``signal_lag``), the book decided on day
``t`` is held from ``t + lag`` and earns the return of day ``t + lag + 1`` (``forward_returns`` builds that column), so the agent never trades on a return it can see. Training episodes use only rows up to the
end of the training window.

**Actions.** A vector in [-1, 1] per asset. Long-short (default): the vector is centred (dollar neutral) and scaled to ``gross``. Long-only: a softmax over the assets and a cash slot, times ``gross``. Both are
then capped at ``max_weight`` per asset (a cap that binds can leave the book below its exposure budget; a long-short book stays dollar neutral) and, when ``max_turnover`` is set, the trade from the previous book is shortened to that one-way turnover. ``project_weights`` is the single place these constraints live,
so the rollout that produces the strategy's positions and the environment the agent trained in cannot disagree.

**Costs.** One-way turnover times (commission + slippage) in basis points, the engine's ``LinearCostModel``, plus an optional quadratic impact term.

**Rewards** (percent units, so the scale does not depend on the data): ``return`` the net return; ``risk_adjusted`` the net return less ``risk_aversion / 2`` times its squared deviation from its running mean, less
``drawdown_penalty`` times any new drawdown, less ``turnover_penalty`` times turnover; ``dsr`` the differential Sharpe ratio of Moody and Saffell (2001), the marginal effect of this step on an exponentially
weighted Sharpe ratio.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

REWARDS = ("return", "risk_adjusted", "dsr")


def forward_returns(returns: np.ndarray, lag: int) -> np.ndarray:
    """``out[t]`` = the return earned by a book decided on day ``t`` (day ``t + lag + 1``); NaN where that day is not in the data."""
    out = np.full_like(returns, np.nan, dtype=np.float64)
    k = int(lag) + 1
    if k < len(returns):
        out[:-k] = returns[k:]
    return out


def project_weights(action: np.ndarray, previous: np.ndarray, gross: float = 1.0, max_weight: float = 0.3, long_only: bool = False, max_turnover: float = 0.0, mask: np.ndarray | None = None) -> np.ndarray:
    """The book an action asks for, after the constraints: exposure ``gross``, at most ``max_weight`` per asset, long-only or dollar neutral, no position in an asset that is not investable, and the one-way
    turnover from ``previous`` limited to ``max_turnover`` when that is positive."""
    a = np.asarray(action, dtype=np.float64).copy()
    n = len(previous)
    ok = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    if long_only:
        logits = np.append(np.where(ok, 3.0 * a[:n], -np.inf), 3.0 * a[n] if len(a) > n else 0.0)
        e = np.exp(logits - logits.max())
        w = gross * (e / e.sum())[:n]
    else:
        v = np.where(ok, a[:n], 0.0)
        v = v - v[ok].mean() if ok.any() else v
        v = np.where(ok, v, 0.0)
        total = np.abs(v).sum()
        w = v / total * gross if total > 1e-12 else np.zeros(n)
    w = np.clip(w, -float(max_weight), float(max_weight))                              # a cap that binds leaves the book below its exposure budget, never above it
    if not long_only:
        long, short = w[w > 0].sum(), -w[w < 0].sum()                                  # keep it dollar neutral: the heavier side is scaled down to the lighter
        if long > short > 0:
            w = np.where(w > 0, w * short / long, w)
        elif short > long > 0:
            w = np.where(w < 0, w * long / short, w)
        elif min(long, short) == 0:
            w = np.zeros(n)
    if max_turnover and max_turnover > 0:
        trade = w - previous
        moved = np.abs(trade).sum()
        if moved > max_turnover:
            w = previous + trade * (max_turnover / moved)
    return w


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, forward: np.ndarray, mask: np.ndarray | None = None, cost_bps: float = 7.0, impact: float = 0.0, gross: float = 1.0, max_weight: float = 0.3,
                 long_only: bool = False, max_turnover: float = 0.0, reward: str = "risk_adjusted", risk_aversion: float = 0.5, turnover_penalty: float = 0.0, drawdown_penalty: float = 0.0,
                 episode_days: int = 252, eta: float = 1.0 / 63.0, seed: int | None = None):
        super().__init__()
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {list(REWARDS)}")
        if features.ndim != 3 or forward.shape != features.shape[:2]:
            raise ValueError("features must be [days, assets, features] and forward [days, assets]")
        self.features = np.asarray(features, dtype=np.float32)
        self.forward = np.nan_to_num(np.asarray(forward, dtype=np.float64))
        self.mask = np.ones(forward.shape, bool) if mask is None else (np.asarray(mask, bool) & np.isfinite(forward))
        self.n_days, self.n_assets, self.n_features = self.features.shape
        self.cost, self.impact, self.gross, self.cap, self.long_only = float(cost_bps) / 1e4, float(impact), float(gross), float(max_weight), bool(long_only)
        self.max_turnover, self.reward_kind, self.lam = float(max_turnover), reward, float(risk_aversion)
        self.turnover_penalty, self.drawdown_penalty, self.eta = float(turnover_penalty), float(drawdown_penalty), float(eta)
        self.episode_days = max(2, min(int(episode_days), self.n_days))
        width = self.n_assets + (1 if self.long_only else 0)
        self.action_space = spaces.Box(-1.0, 1.0, (width,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (self.n_assets * self.n_features + self.n_assets + 1,), dtype=np.float32)
        self._t0 = self._t = 0
        self.reset(seed=seed)

    # ------------------------------------------------------------------------------------------------------------ the book
    def book(self, action: np.ndarray, previous: np.ndarray, t: int) -> np.ndarray:
        return project_weights(action, previous, self.gross, self.cap, self.long_only, self.max_turnover, self.mask[t])

    def observation(self, t: int, weights: np.ndarray, drawdown: float) -> np.ndarray:
        return np.concatenate([self.features[t].ravel(), weights, [drawdown]]).astype(np.float32)

    # ------------------------------------------------------------------------------------------------------------ gymnasium
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        room = self.n_days - self.episode_days
        self._t0 = int(self.np_random.integers(0, room + 1)) if room > 0 else 0
        self._t = self._t0
        self.w = np.zeros(self.n_assets)
        self.equity = self.peak = 1.0
        self.ema, self.var = 0.0, 0.0
        self.A, self.B = 0.0, 0.0
        return self.observation(self._t, self.w, 0.0), {}

    def step(self, action):
        t = self._t
        w_new = self.book(np.asarray(action), self.w, t)
        trade = w_new - self.w
        turnover = float(np.abs(trade).sum())
        cost = self.cost * turnover + self.impact * float((trade ** 2).sum())
        gross_ret = float((w_new * np.where(self.mask[t], self.forward[t], 0.0)).sum())
        net = gross_ret - cost
        self.equity *= 1.0 + net
        dd_before = self.equity / self.peak - 1.0 if self.peak > 0 else 0.0
        self.peak = max(self.peak, self.equity)
        dd = self.equity / self.peak - 1.0
        x = 100.0 * net
        if self.reward_kind == "return":
            reward = x
        elif self.reward_kind == "risk_adjusted":
            reward = x - 0.5 * self.lam * (x - self.ema) ** 2 - self.drawdown_penalty * 100.0 * max(dd_before - dd, 0.0) - self.turnover_penalty * turnover
            self.ema += self.eta * (x - self.ema)
        else:
            dA, dB = x - self.A, x * x - self.B
            denom = (self.B - self.A ** 2) ** 1.5
            reward = (self.B * dA - 0.5 * self.A * dB) / denom if denom > 1e-8 else 0.0
            reward = float(np.clip(reward, -10.0, 10.0))
            self.A += self.eta * dA
            self.B += self.eta * dB
        self.w = w_new
        self._t += 1
        truncated = self._t >= min(self._t0 + self.episode_days, self.n_days)
        obs = self.observation(min(self._t, self.n_days - 1), self.w, dd)
        return obs, float(reward), False, bool(truncated), {"net_return": net, "gross_return": gross_ret, "cost": cost, "turnover": turnover, "drawdown": dd}


def rollout(policy, features: np.ndarray, mask: np.ndarray, env_kwargs: dict, recurrent: bool = False) -> np.ndarray:
    """The books ``policy`` chooses on each day of ``features`` in order, starting flat: ``policy(obs, state, first) -> (action, state)``. This is how a strategy's positions are produced out of sample: the
    same constraint function and the same observation as in training, one day after another, using the rows up to that day only."""
    n_days, n_assets = features.shape[:2]
    env = PortfolioEnv(features, np.zeros((n_days, n_assets)), mask, **{**env_kwargs, "episode_days": n_days})
    w, state, out = np.zeros(n_assets), None, np.zeros((n_days, n_assets))
    for t in range(n_days):
        obs = env.observation(t, w, 0.0)
        action, state = policy(obs, state, t == 0)
        w = env.book(np.asarray(action), w, t)
        out[t] = w
    return out
