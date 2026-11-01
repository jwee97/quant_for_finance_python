"""A Gymnasium environment for portfolio allocation on the platform's panel: one chronological ledger, one state transition for training and for playing a policy forward.

**Timeline.** Row ``t`` holds what is known at the close of day ``t``; ``returns[t]`` is the return of day ``t`` (close ``t-1`` to close ``t``). One ``step`` is one session:

1. the policy has observed day ``t`` (features, the holdings it has, the order still waiting, the drawdown so far) and chooses target weights;
2. the order is queued to execute at the close of day ``t + lag`` (``lag = 0``: now, at today's close);
3. the session passes: the holdings are marked with the realised return of day ``t + 1``, financing and borrow accrue;
4. an order that is due is executed at that close, after the marks, against the holdings as they have drifted, and the trading cost is charged;
5. the reward is the realised net return of the whole session, and the next observation is built from the ledger.

So a book decided on day ``t`` is held from ``t + lag`` and earns the return of day ``t + lag + 1`` (the platform's ``signal_lag`` convention), and nothing the observation shows is a return that has not yet happened: with
``lag >= 1`` the return of a book is credited to the step in which it is earned, not to the step that chose it. A realised return updates equity, drawdown and weights only when the clock reaches it.

**Actions.** A vector in [-1, 1] per asset. Long-short (default): centred (dollar neutral) and scaled to ``gross``. Long-only: a softmax over the assets and a cash slot, times ``gross``. Then capped at ``max_weight``.
At execution, assets that are no longer eligible are liquidated whatever the budget (mandatory), and the remaining optional trade is shortened to ``max_turnover`` one-way turnover; a forced liquidation that exceeds
the budget is reported in ``info["events"]``. Wrong-sized or non-finite actions are rejected with ``ValueError``; values outside [-1, 1] are clipped and the clipping is reported.

**Costs.** One-way turnover times ``cost_bps`` plus ``impact`` times the squared weight changes, both fractions of the equity before the trade, charged when the order executes; ``financing_bps`` (annual, on
debit cash) and ``borrow_bps`` (annual, on short notional) accrue each session. ``max_leverage`` is a hard exposure limit: a book that drifts above it is cut back at that close.

**Rewards** (percent per session): ``return``; ``risk_adjusted``: the net return less ``risk_aversion / 2`` times its squared deviation from a running mean, less ``drawdown_penalty`` times the new drawdown, less
``turnover_penalty`` times the turnover; ``dsr``: the differential Sharpe ratio (``ledger.differential_sharpe``). A reward does not sum to final equity and is not meant to.

Insolvent portfolios terminate the episode and cannot resume. ``STATE_SCHEMA_VERSION`` changes whenever the observation changes.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .ledger import EPS_EQUITY, Ledger, differential_sharpe, drawdown_increase

REWARDS = ("return", "risk_adjusted", "dsr")
STATE_SCHEMA_VERSION = 2                                      # 1: weights and drawdown carried from forward-shifted returns; 2: quantity ledger, event timeline, pending order in the observation
TRADING_DAYS = 252.0


def forward_returns(returns: np.ndarray, lag: int) -> np.ndarray:
    """``out[t]`` = the return earned by a book decided on day ``t`` (day ``t + lag + 1``); NaN where that day is not in the data. A label-alignment helper: the environment itself never reads it."""
    out = np.full_like(returns, np.nan, dtype=np.float64)
    k = int(lag) + 1
    if k < len(returns):
        out[:-k] = returns[k:]
    return out


def _raw_book(action: np.ndarray, gross: float, max_weight: float, long_only: bool, ok: np.ndarray) -> np.ndarray:
    """The book an action asks for before any state is looked at: exposure ``gross``, at most ``max_weight`` per asset, long-only or dollar neutral, nothing in an asset that is not eligible."""
    n = len(ok)
    a = np.asarray(action, dtype=np.float64)
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
    return w


def enforce_limits(w: np.ndarray, previous: np.ndarray, max_turnover: float, ok: np.ndarray, cap: float | None = None, gross: float | None = None) -> tuple[np.ndarray, dict]:
    """Apply the constraints that depend on what is held. In order of precedence:

    1. eligibility: a holding in an asset that is no longer eligible is liquidated whatever the turnover budget (mandatory);
    2. the optional trade that remains is shortened to the one-way turnover budget ``max_turnover``;
    3. the hard risk limits ``cap`` (per asset) and ``gross`` (total) are then enforced on the result, because holdings drift past them and a budget must not leave a breach standing (mandatory).

    Returns the book and what happened: ``forced`` turnover of the liquidation, ``trimmed`` (weight removed by step 3), whether the budget ``shortened`` the trade, and ``exceeded`` when the mandatory
    steps alone are above the budget.
    """
    previous_legal = np.where(ok, previous, 0.0)
    forced = float(np.abs(previous - previous_legal).sum())
    w = np.where(ok, w, 0.0)
    shortened = False
    if max_turnover and max_turnover > 0:
        trade = w - previous_legal
        moved = float(np.abs(trade).sum())
        if moved > max_turnover:
            w = previous_legal + trade * (max_turnover / moved)                        # zero in every ineligible asset: both ends are zero there
            shortened = True
    trimmed = 0.0
    if cap is not None:
        clipped = np.clip(w, -cap, cap)
        trimmed += float(np.abs(w - clipped).sum())
        w = clipped
    if gross is not None and np.abs(w).sum() > gross + 1e-12:
        trimmed += float(np.abs(w).sum() - gross)
        w = w * (gross / np.abs(w).sum())
    over = bool(max_turnover and (forced > max_turnover or float(np.abs(w - previous_legal).sum()) > max_turnover + 1e-9))
    return w, {"forced": forced, "trimmed": trimmed, "shortened": shortened, "exceeded": over}


def project_weights(action: np.ndarray, previous: np.ndarray, gross: float = 1.0, max_weight: float = 0.3, long_only: bool = False, max_turnover: float = 0.0, mask: np.ndarray | None = None) -> np.ndarray:
    """The book an action asks for after every constraint, given the holdings ``previous``: ``_raw_book`` then ``enforce_limits``. The environment applies the two at different times (the first when the order
    is placed, the second when it executes); this composes them for callers that want the result at once."""
    ok = np.ones(len(previous), bool) if mask is None else np.asarray(mask, bool)
    w, _ = enforce_limits(_raw_book(action, gross, max_weight, long_only, ok), np.asarray(previous, dtype=float), max_turnover, ok, max_weight, gross)
    return w


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, returns: np.ndarray, mask: np.ndarray | None = None, cost_bps: float = 7.0, impact: float = 0.0, gross: float = 1.0, max_weight: float = 0.3,
                 long_only: bool = False, max_turnover: float = 0.0, reward: str = "risk_adjusted", risk_aversion: float = 0.5, turnover_penalty: float = 0.0, drawdown_penalty: float = 0.0,
                 episode_days: int = 252, eta: float = 1.0 / 63.0, seed: int | None = None, lag: int = 1, financing_bps: float = 0.0, borrow_bps: float = 0.0,
                 max_leverage: float = 0.0):
        super().__init__()
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {list(REWARDS)}")
        if features.ndim != 3 or returns.shape != features.shape[:2]:
            raise ValueError("features must be [days, assets, features] and returns [days, assets]")
        if int(lag) < 0:
            raise ValueError("lag must be 0 or more")
        self.features = np.asarray(features, dtype=np.float32)
        self.returns = np.nan_to_num(np.asarray(returns, dtype=np.float64))
        self.mask = np.ones(returns.shape, bool) if mask is None else np.asarray(mask, bool)
        self.n_days, self.n_assets, self.n_features = self.features.shape
        self.cost, self.impact, self.gross, self.cap, self.long_only = float(cost_bps) / 1e4, float(impact), float(gross), float(max_weight), bool(long_only)
        self.max_turnover, self.reward_kind, self.lam = float(max_turnover), reward, float(risk_aversion)
        self.turnover_penalty, self.drawdown_penalty, self.eta = float(turnover_penalty), float(drawdown_penalty), float(eta)
        self.lag, self.financing, self.borrow = int(lag), float(financing_bps) / 1e4 / TRADING_DAYS, float(borrow_bps) / 1e4 / TRADING_DAYS
        self.max_leverage = float(max_leverage)
        self.episode_days = max(1, min(int(episode_days), self.n_days - 1))
        width = self.n_assets + (1 if self.long_only else 0)
        self.action_space = spaces.Box(-1.0, 1.0, (width,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (self.n_assets * self.n_features + 2 * self.n_assets + 1,), dtype=np.float32)
        self._t0 = self.t = 0
        self.reset(seed=seed)

    # ------------------------------------------------------------------------------------------------------------ state
    @property
    def w(self) -> np.ndarray:
        """The holdings now, as weights of the equity now (they drift with the marks)."""
        return self.ledger.weights()

    def observation(self) -> np.ndarray:
        """Everything the policy may see at the close of day ``t``: the day's features, its holdings, the order still waiting (zeros if none) and the drawdown so far."""
        latest = self.pending[-1][1] if self.pending else np.zeros(self.n_assets)
        return np.concatenate([self.features[self.t].ravel(), self.ledger.weights(), latest, [self.ledger.drawdown()]]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        room = self.n_days - 1 - self.episode_days
        self._t0 = int(self.np_random.integers(0, room + 1)) if room > 0 else 0
        return self.reset_at(self._t0), {}

    def reset_at(self, t0: int) -> np.ndarray:
        """Start a flat account (cash only, equity 1) at the close of day ``t0``. Opening the first book costs what any trade costs."""
        self._t0 = self.t = int(t0)
        self.ledger = Ledger.start(self.n_assets)
        self.pending: list[tuple[int, np.ndarray]] = []
        self.ema = self.var = self.A = self.B = 0.0
        self.steps = 0
        return self.observation()

    # ------------------------------------------------------------------------------------------------------------ one session
    def check_action(self, action) -> tuple[np.ndarray, list]:
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.shape != self.action_space.shape:
            raise ValueError(f"the action has {a.size} values; this environment takes {self.action_space.shape[0]}")
        if not np.isfinite(a).all():
            raise ValueError("the action has a value that is not finite")
        events = []
        if (np.abs(a) > 1.0).any():
            events.append("action clipped to [-1, 1]")
            a = np.clip(a, -1.0, 1.0)
        return a, events

    def decide(self, action) -> tuple[np.ndarray, list]:
        """The target weights an action asks for on the current day, and what was changed to get them. This is the part of ``step`` before time moves; a policy being played to its last day uses it alone."""
        a, events = self.check_action(action)
        ok = self.mask[self.t]
        if not ok.any():
            events.append("no eligible asset: the target is cash")
        return _raw_book(a, self.gross, self.cap, self.long_only, ok), events

    def _execute(self, target: np.ndarray, day: int, events: list) -> tuple[float, float]:
        ok = self.mask[day]
        book, info = enforce_limits(target, self.ledger.weights(), self.max_turnover, ok, self.cap, self.gross)
        if info["forced"] > 0:
            events.append(f"forced liquidation of {info['forced']:.4f} of equity in assets no longer eligible")
        if info["trimmed"] > 1e-12:
            events.append(f"a position or exposure limit trimmed {info['trimmed']:.4f} of equity after the turnover budget")
        if info["exceeded"]:
            events.append("the mandatory trades alone are above the turnover budget")
        if info["shortened"]:
            events.append("the trade was shortened to the turnover budget")
        fill = self.ledger.trade_to(book, self.cost, self.impact)
        return fill["cost"], fill["turnover"]

    def step(self, action):
        if self.ledger.dead:
            raise RuntimeError("the portfolio is insolvent: reset the environment")
        if self.t + 1 >= self.n_days:
            raise RuntimeError("there is no later day to advance to")
        target, events = self.decide(action)
        before, dd_before = self.ledger.equity, self.ledger.drawdown()
        self.pending.append((self.t + self.lag, target))
        cost = turnover = 0.0
        if self.lag == 0:                                                                   # same-close execution: the order is filled before the session passes
            c, u = self._execute(self.pending.pop(0)[1], self.t, events)
            cost, turnover = cost + c, turnover + u
        self.t += 1
        self.ledger.mark(self.returns[self.t], self.financing, self.borrow)
        if not self.ledger.dead and self.max_leverage > 0 and self.ledger.gross() > self.max_leverage + 1e-9:
            cut = self.ledger.weights() * (self.max_leverage / self.ledger.gross())
            events.append(f"exposure above the {self.max_leverage:g}x limit: cut back at the close")
            fill = self.ledger.trade_to(cut, self.cost, self.impact)
            cost, turnover = cost + fill["cost"], turnover + fill["turnover"]
        while self.pending and self.pending[0][0] <= self.t and not self.ledger.dead:
            c, u = self._execute(self.pending.pop(0)[1], self.t, events)
            cost, turnover = cost + c, turnover + u
        after = self.ledger.equity
        net = after / before - 1.0 if before > EPS_EQUITY else 0.0
        gross_ret = (after / before - 1.0 + cost) if before > EPS_EQUITY else 0.0           # the session's return before the cost of this step's trades (approximately: cost is a fraction of the equity then)
        dd = self.ledger.drawdown()
        x = 100.0 * net
        if self.reward_kind == "return":
            reward = x
        elif self.reward_kind == "risk_adjusted":
            reward = x - 0.5 * self.lam * (x - self.ema) ** 2 - self.drawdown_penalty * 100.0 * drawdown_increase(dd_before, dd) - self.turnover_penalty * turnover
            self.ema += self.eta * (x - self.ema)
        else:
            reward, _, self.A, self.B = differential_sharpe(x, self.A, self.B, self.eta)
        self.steps += 1
        terminated = bool(self.ledger.dead)
        truncated = bool(self.steps >= self.episode_days or self.t >= self.n_days - 1) and not terminated
        info = {"net_return": net, "gross_return": gross_ret, "cost": cost, "turnover": turnover, "drawdown": dd, "equity": after, "target": target, "events": events, "dead": terminated}
        return self.observation(), float(reward), terminated, truncated, info


def rollout(policy, features: np.ndarray, returns: np.ndarray, mask: np.ndarray, env_kwargs: dict, recurrent: bool = False, record: bool = False):
    """The books ``policy`` chooses on each day of ``features`` in order, starting flat, through the same ``PortfolioEnv.step`` the policy was trained in: ``policy(obs, state, first) -> (action, state)``.

    Returns ``(books, trace)``. ``books[t]`` is the target decided on day ``t`` (the book the strategy would hold from ``t + lag``); ``trace`` has the equity, drawdown, cost and turnover of the ledger after each day,
    and the observations when ``record`` is set. The returns up to day ``t`` are what the ledger has seen when the policy decides on day ``t``.
    """
    n_days, n_assets = features.shape[:2]
    kwargs = {**env_kwargs, "episode_days": max(n_days - 1, 1)}
    env = PortfolioEnv(features, returns, mask, **kwargs)
    obs, state = env.reset_at(0), None
    books = np.zeros((n_days, n_assets))
    trace = {"equity": np.ones(n_days), "drawdown": np.zeros(n_days), "cost": np.zeros(n_days), "turnover": np.zeros(n_days), "observations": [] if record else None, "events": [[] for _ in range(n_days)]}
    for t in range(n_days):
        trace["equity"][t], trace["drawdown"][t] = env.ledger.equity, env.ledger.drawdown()          # the account at the close of day t, as the policy sees it
        if env.ledger.dead:
            books[t:], trace["equity"][t:], trace["drawdown"][t:] = 0.0, 0.0, -1.0
            break
        if record:
            trace["observations"].append(obs.copy())
        action, state = policy(obs, state, t == 0)
        books[t], trace["events"][t] = env.decide(action)
        if t == n_days - 1:
            break
        obs, _, _, _, info = env.step(action)
        trace["cost"][t + 1], trace["turnover"][t + 1], trace["events"][t] = info["cost"], info["turnover"], info["events"]
    return books, trace
