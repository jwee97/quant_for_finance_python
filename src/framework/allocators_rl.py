"""An allocator from reinforcement learning: how much of an equal-weighted book to hold, learned month by month from the months already observed (:mod:`src.rl.exposure`).

    q_learning_exposure    the exposure (0, half, or full) that fitted Q-iteration says maximises mean-variance utility net of trading costs, given the trend and volatility state of the book and the exposure held
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.sleeves import month_end_dates
from ..rl.exposure import fit_exposure_policy, market_states
from .allocation import Allocator, Context
from .registry import register_allocator


@register_allocator("q_learning_exposure", "Equal-weighted book held at 0, half or full exposure chosen by fitted Q-iteration on the trend and volatility state of the book, learned from past months only and net of trading cost")
class QLearningExposure(Allocator):
    """At each month-end the book's own history (its equal-weighted monthly return) gives a state: up or down over twelve months, calm or turbulent over three. A Q-function over (state, exposure held, new exposure) is
    fitted on every earlier month whose following month has ended, with the reward of each action computed exactly (no exploration is needed: holding less does not change the market), and the exposure with the
    largest Q is held for the coming month. Before ``min_months`` months of history the book is fully invested."""

    min_assets = 2

    def __init__(self, exposures: tuple = (0.0, 0.5, 1.0), min_months: int = 36, cost_bps: float = 10.0, risk_aversion: float = 5.0, gamma: float = 0.5, shrink: float = 6.0):
        if len(exposures) < 2 or min(exposures) < 0 or max(exposures) > 1.0 or min_months < 24 or cost_bps < 0 or risk_aversion <= 0 or not 0 <= gamma < 1 or shrink < 0:
            raise ValueError("exposures in [0, 1] (at least two), min_months >= 24, cost_bps >= 0, risk_aversion > 0, 0 <= gamma < 1, shrink >= 0")
        self.exposures, self.min_months, self.cost, self.risk_aversion, self.gamma, self.shrink = tuple(exposures), int(min_months), cost_bps / 1e4, float(risk_aversion), float(gamma), float(shrink)

    def exposure_path(self, bundle) -> pd.Series:
        """The exposure chosen at each month-end (before the weights are formed)."""
        months = month_end_dates(bundle.index)
        px = bundle.prices.loc[months]
        live = bundle.investable.loc[months]
        monthly = (px / px.shift(1) - 1.0).where(live & live.shift(1, fill_value=False))
        book = monthly.mean(axis=1).to_numpy()                                                 # the equal-weighted return from the previous to this month-end
        states = market_states(np.nan_to_num(book))
        e = np.asarray(self.exposures)
        chosen = np.full(len(months), np.nan)
        held = int(np.argmax(e))
        for t in range(len(months)):
            known = t - 1                                                                      # month t's return is the 'next' return of state t - 1; use states up to t - 1 with returns up to t
            if known < self.min_months or states[t] < 0:
                chosen[t] = e[held]
                continue
            Q = fit_exposure_policy(states[:known + 1], book[1:known + 2], e, self.cost, self.risk_aversion, self.gamma, self.shrink)
            held = int(np.argmax(Q[states[t], held]))
            chosen[t] = e[held]
        return pd.Series(chosen, index=months)

    def build(self, ctx: Context) -> pd.DataFrame:
        bundle = ctx.bundle
        exposure = self.exposure_path(bundle)
        investable = bundle.investable.loc[exposure.index]
        n = investable.sum(axis=1).replace(0, np.nan)
        base = investable.astype(float).div(n, axis=0).fillna(0.0)
        w = base.mul(exposure, axis=0)
        out = pd.DataFrame(np.nan, index=bundle.index, columns=bundle.assets)
        out.loc[w.index] = w.to_numpy()
        return out.ffill().fillna(0.0)
