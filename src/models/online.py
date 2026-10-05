"""Online learning (Generation 3, Priority 10): train, trade, update, trade, update.

Three coefficient updaters for a linear forecast ``y = x'beta`` and one expert-
aggregation rule, all causal by construction: a prediction at time ``t`` uses
only updates from observations whose outcomes had fully elapsed before ``t``.

``OnlineRidge``   Recursive least squares with exponential forgetting. Kept as
                  sufficient statistics (A = sum lambda^age x x', b = sum lambda^age x y)
                  rather than a rank-one inverse update: for the small panels
                  here that is exact, stable and easy to test against batch
                  least squares.
``NLMS``          Normalised least-mean-squares (online gradient descent on the
                  squared loss with a step normalised by the row's energy).
``KalmanRW``      Random-walk coefficients: beta_t = beta_{t-1} + eta, a Kalman
                  filter whose ratio of process to observation noise plays the
                  role of the forgetting factor.
``hedge_weights`` Exponentially weighted forecaster over experts (Hedge /
                  multiplicative weights) with the regret it guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class OnlineRidge:
    def __init__(self, dim: int, forgetting: float = 0.99, ridge: float = 100.0):
        self.forgetting, self.ridge = float(forgetting), float(ridge)
        self.a = np.zeros((dim, dim))
        self.b = np.zeros(dim)
        self.dim = dim
        self._beta = np.zeros(dim)
        self._dirty = False

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """One update per period: older information is down-weighted by ``forgetting`` first."""
        x = np.atleast_2d(x)
        self.a = self.forgetting * self.a + x.T @ x
        self.b = self.forgetting * self.b + x.T @ np.asarray(y, dtype=float)
        self._dirty = True

    @property
    def beta(self) -> np.ndarray:
        if self._dirty:
            self._beta = np.linalg.solve(self.a + self.ridge * np.eye(self.dim), self.b)
            self._dirty = False
        return self._beta

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.atleast_2d(x) @ self.beta


class NLMS:
    def __init__(self, dim: int, step: float = 0.1, eps: float = 1e-8):
        self.beta = np.zeros(dim)
        self.step, self.eps = float(step), float(eps)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        for row, target in zip(np.atleast_2d(x), np.asarray(y, dtype=float)):
            error = target - row @ self.beta
            self.beta = self.beta + self.step * error * row / (row @ row + self.eps)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.atleast_2d(x) @ self.beta


class KalmanRW:
    def __init__(self, dim: int, process_to_observation: float = 1e-3, observation_variance: float = 1.0,
                 prior_variance: float = 1.0):
        self.beta = np.zeros(dim)
        self.p = np.eye(dim) * prior_variance
        self.r = float(observation_variance)
        self.q = float(process_to_observation) * self.r
        self.dim = dim

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.atleast_2d(x)
        y = np.asarray(y, dtype=float)
        p_prior = self.p + self.q * np.eye(self.dim)
        s = x @ p_prior @ x.T + self.r * np.eye(len(y))
        gain = p_prior @ x.T @ np.linalg.inv(s)
        self.beta = self.beta + gain @ (y - x @ self.beta)
        self.p = (np.eye(self.dim) - gain @ x) @ p_prior
        self.p = 0.5 * (self.p + self.p.T)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.atleast_2d(x) @ self.beta


# ---------------------------------------------------------------------------
# The online forecasting loop
# ---------------------------------------------------------------------------
def run_online_forecasts(design: np.ndarray, target: np.ndarray, origin_of_row: np.ndarray, position_of_row: np.ndarray,
                         first_test_position: int, horizon: int, factories: dict, test_origins: list) -> dict[str, pd.DataFrame]:
    """Month-by-month predictions from updaters that only ever see matured labels.

    ``design`` / ``target``: one row per (origin, asset). ``origin_of_row``: the origin label of each row;
    ``position_of_row``: its integer position on the trading calendar. At each test origin ``t`` (position
    ``p_t``) every origin ``s`` with ``p_s + horizon <= p_t`` that has not been used yet is fed to the models,
    one origin at a time in chronological order (one forgetting step per origin), and then the rows of ``t``
    are predicted.

    ``test_origins`` is ``[(origin_label, position), ...]``. Returns ``{model: frame(origin, row_index, mu)}``.
    """
    models = {name: make() for name, make in factories.items()}
    labelled = np.isfinite(target) & np.isfinite(design).all(axis=1)
    origins = sorted(np.unique(origin_of_row).tolist(), key=lambda o: position_of_row[origin_of_row == o][0])
    origin_rows = {o: np.flatnonzero((origin_of_row == o) & labelled) for o in origins}
    origin_pos = {o: int(position_of_row[origin_of_row == o][0]) for o in origins}
    used: set = set()
    out = {name: [] for name in models}
    for label, position in test_origins:
        for o in origins:
            if o in used or origin_pos[o] + horizon > position:
                continue
            rows = origin_rows[o]
            used.add(o)
            if len(rows) == 0:
                continue
            for model in models.values():
                model.update(design[rows], target[rows])
        test_rows = np.flatnonzero((origin_of_row == label) & np.isfinite(design).all(axis=1))
        for name, model in models.items():
            out[name].append(pd.DataFrame({"origin": label, "row": test_rows, "mu": model.predict(design[test_rows])}))
    return {name: pd.concat(frames, ignore_index=True) for name, frames in out.items()}


# ---------------------------------------------------------------------------
# Expert aggregation
# ---------------------------------------------------------------------------
@dataclass
class HedgeResult:
    weights: pd.DataFrame          # weight applied to each expert on each day (known BEFORE that day's return)
    returns: pd.Series             # aggregate return
    learning_rate: float
    regret: pd.Series              # cumulative (best fixed expert in hindsight - aggregate), in gain units
    regret_bound: pd.Series        # sqrt(T_eff ln N / 2)-type bound on the same scale


def hedge_aggregate(expert_returns: pd.DataFrame, gain_scale: float = 0.01, periods_per_year: int = 252) -> HedgeResult:
    """Hedge over experts on daily returns.

    On day ``t`` the weights are proportional to ``exp(eta * G_{t-1,i})`` where ``G`` is the cumulative gain
    (return / ``gain_scale``) of expert ``i`` through the previous day, so the weights used on ``t`` depend only
    on the past. ``eta = sqrt(8 ln N / T)`` with ``T`` the number of periods (theory-derived for gains in an
    interval of unit length; daily gains are clipped to [-1, 1] in the exponent, which they almost never reach).
    """
    r = expert_returns.dropna(how="any")
    n_exp = r.shape[1]
    gains = np.clip(r.to_numpy() / gain_scale, -1.0, 1.0)
    horizon = len(r)
    eta = float(np.sqrt(8.0 * np.log(n_exp) / max(horizon, 1)))
    cumulative = np.vstack([np.zeros(n_exp), np.cumsum(gains, axis=0)[:-1]])
    logits = eta * cumulative
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    agg = (w * r.to_numpy()).sum(axis=1)
    agg_gain = (w * gains).sum(axis=1)
    best = np.cumsum(gains, axis=0).max(axis=1)
    regret = best - np.cumsum(agg_gain)
    steps = np.arange(1, horizon + 1)
    bound = np.sqrt(steps * np.log(n_exp) / 2.0) * 2.0                 # an upper envelope of the standard O(sqrt(T ln N)) bound
    return HedgeResult(pd.DataFrame(w, index=r.index, columns=r.columns), pd.Series(agg, index=r.index),
                       eta, pd.Series(regret, index=r.index), pd.Series(bound, index=r.index))
