"""Allocators from the Bayesian side of portfolio theory.

    hierarchical_bayes     mean-variance on expected returns from a hierarchical prior: assets shrink toward their group (asset class), groups toward the whole (:mod:`src.portfolio.hierarchical_bayes`)
    bayes_expected_utility  the weights that maximise expected utility (power, exponential, shortfall or CVaR) over the posterior predictive distribution (:mod:`src.portfolio.decision_theory`)

Both use data through the month-end only, trailing windows, and the platform's position limits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.sleeves import month_end_dates
from ..portfolio import decision_theory as dt
from ..portfolio.bayesian import niw_posterior
from ..portfolio.hierarchical_bayes import hierarchical_means
from ..portfolio.mean_variance import mean_variance_weights
from .allocation import Allocator, Context
from .registry import register_allocator

ANN = 252.0


@register_allocator("hierarchical_bayes", "Mean-variance on expected returns from a hierarchical Bayesian prior: each asset shrinks toward its group (asset class), each group toward the whole, by amounts the data choose")
class HierarchicalBayes(Allocator):
    """The sample mean return of an asset is a poor guess at its expected return; the mean of its asset class is a better one when the class members are alike, and the market's is better still when the
    classes are. The scales of those similarities are not assumed but inferred from the trailing ``lookback`` days (a posterior over a grid), and the portfolio is the long-only, capped mean-variance
    optimum for the resulting posterior mean and a covariance shrunk toward its diagonal. Groups are the bundle's groups, else its asset classes; an asset with none is its own group."""

    min_assets = 3

    def __init__(self, lookback: int = 756, risk_aversion: float = 5.0, points: int = 7):
        if lookback < 120 or risk_aversion <= 0 or points < 3:
            raise ValueError("lookback >= 120, risk_aversion > 0, points >= 3")
        self.lookback, self.risk_aversion, self.points = int(lookback), float(risk_aversion), int(points)

    def build(self, ctx: Context) -> pd.DataFrame:
        from ..portfolio.covariance import estimate_covariance

        bundle = ctx.bundle
        groups = dict(bundle.group or bundle.asset_class or {})
        cons = ctx.constraints()
        out = pd.DataFrame(np.nan, index=bundle.index, columns=bundle.assets)
        for origin in month_end_dates(bundle.index):
            pos = bundle.index.get_loc(origin)
            if pos < self.lookback:
                continue
            window = bundle.returns.iloc[pos - self.lookback + 1:pos + 1]
            live = [a for a in bundle.assets if bool(bundle.investable.loc[origin, a]) and window[a].notna().mean() > 0.95]
            if len(live) < self.min_assets:
                continue
            w = window[live].fillna(0.0)
            mu_daily, _ = hierarchical_means(w, groups, self.points)
            cov = estimate_covariance(w, "shrinkage", self.lookback, annualise=True)
            res = mean_variance_weights(pd.Series(ANN * mu_daily.to_numpy(), index=live), cov, self.risk_aversion, cons)
            if res.success:
                out.loc[origin, live] = res.weights.reindex(live).to_numpy()
        return out.ffill().fillna(0.0)


UTILITIES = {"crra": dt.crra, "cara": dt.cara, "mean_variance": dt.mean_variance, "cvar": dt.cvar, "shortfall": dt.shortfall}


@register_allocator("bayes_expected_utility", "The weights that maximise expected utility (power, exponential, mean-variance, CVaR or loss-averse) over the posterior predictive distribution of next month's returns, parameter uncertainty included")
class BayesExpectedUtility(Allocator):
    """Draw parameters ``(mu, Sigma)`` from the normal-inverse-Wishart posterior of the trailing ``lookback`` days, draw a month of returns from each, and choose the long-only, capped weights with the highest
    average ``utility`` over those draws (``crra`` with ``risk`` as the relative risk aversion, ``cara``, ``mean_variance``, ``cvar`` or ``shortfall``). Unlike a plug-in optimiser it knows that the
    parameters are uncertain; unlike mean-variance it can care about the shape of the tail."""

    min_assets = 2

    def __init__(self, utility: str = "crra", risk: float = 5.0, lookback: int = 504, draws: int = 300, nu0: float = 126.0, seed: int = 7):
        if utility not in UTILITIES or risk <= 0 or lookback < 120 or draws < 20 or nu0 <= 0:
            raise ValueError(f"utility in {tuple(UTILITIES)}, risk > 0, lookback >= 120, draws >= 20, nu0 > 0")
        self.utility, self.risk, self.lookback, self.draws, self.nu0, self.seed = utility, float(risk), int(lookback), int(draws), float(nu0), int(seed)

    def _utility(self):
        return UTILITIES[self.utility](self.risk) if self.utility in ("crra", "cara", "mean_variance") else UTILITIES[self.utility]()

    def build(self, ctx: Context) -> pd.DataFrame:
        bundle = ctx.bundle
        cons = ctx.constraints()
        out = pd.DataFrame(np.nan, index=bundle.index, columns=bundle.assets)
        rng = np.random.default_rng(self.seed)
        u = self._utility()
        for origin in month_end_dates(bundle.index):
            pos = bundle.index.get_loc(origin)
            if pos < self.lookback:
                continue
            window = bundle.returns.iloc[pos - self.lookback + 1:pos + 1]
            live = [a for a in bundle.assets if bool(bundle.investable.loc[origin, a]) and window[a].notna().mean() > 0.95]
            if len(live) < self.min_assets:
                continue
            post = niw_posterior(window[live].fillna(0.0), self.nu0)
            mus, sigmas = post.draw(self.draws, rng)
            R = dt.predictive_draws(21.0 * mus, 21.0 * sigmas, 1, int(rng.integers(1 << 30)))
            cap = max(cons.max_weight, 1.0 / len(live))
            w = dt.bayes_weights(R, u, lb=cons.min_weight, ub=cap, budget=1.0)
            out.loc[origin, live] = w
        return out.ffill().fillna(0.0)
