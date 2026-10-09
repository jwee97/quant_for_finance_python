"""Portfolio-theory allocators: the tangency portfolio with the capital market line, mean-variance under a CVaR limit, and best-K selection by annealing a QUBO.

    tangency_cml    the highest-Sharpe mix of risky assets on the model forecast, scaled to the investor's risk aversion along the capital market line (the rest in cash, or borrowed)
    mv_cvar         mean-variance with a limit on conditional value-at-risk estimated from the trailing scenarios (Rockafellar-Uryasev)
    qubo_select     the K assets with the best mean-variance trade-off as a QUBO, found by simulated annealing (the form a quantum annealer takes), held in equal weights

All three use the forecast of the month-end and a trailing covariance, and data through the origin only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..equity.cvar_portfolio import mean_variance_cvar
from ..equity.qubo import anneal_selection
from ..equity.theory import cml_allocation, tangency_portfolio
from ..features.sleeves import month_end_dates
from .allocation import Allocator, Context
from .registry import register_allocator


class _MonthlyForecastBook(Allocator):
    """Month-end weights from the forecast row and the trailing returns; subclasses implement ``solve``. The weights are held until the next month-end."""

    lookback = 252
    min_assets = 3

    def solve(self, mu: pd.Series, window: pd.DataFrame, horizon: int, ctx: Context) -> pd.Series | None:
        raise NotImplementedError

    def build(self, ctx: Context) -> pd.DataFrame:
        if ctx.forecasts is None:
            raise ValueError(f"{type(self).__name__} needs forecasts")
        fc, bundle = ctx.forecasts, ctx.bundle
        out = pd.DataFrame(np.nan, index=bundle.index, columns=bundle.assets)
        for origin in month_end_dates(bundle.index):
            position = bundle.index.get_loc(origin)
            if position < self.lookback or origin not in fc.mean.index:
                continue
            window = bundle.returns.iloc[position - self.lookback + 1:position + 1]
            mu_row = fc.mean.loc[origin]
            live = [a for a in bundle.assets if bool(bundle.investable.loc[origin, a]) and window[a].notna().mean() > 0.9 and np.isfinite(mu_row.get(a, np.nan))]
            if len(live) < self.min_assets:
                continue
            w = self.solve(mu_row[live], window[live].fillna(0.0), fc.horizon, ctx)
            if w is not None and np.isfinite(w.to_numpy()).all():
                out.loc[origin, live] = w.reindex(live).to_numpy()
        return out.ffill().fillna(0.0)


@register_allocator("tangency_cml", "Tangency portfolio on the forecast scaled along the capital market line to a risk aversion (the rest in cash, borrowing allowed up to max_leverage)")
class TangencyCML(_MonthlyForecastBook):
    """Markowitz's answer to "which risky portfolio?" when a risk-free asset exists: the one with the highest Sharpe ratio, ``Sigma^-1 (mu - rf)`` normalised, the same for every investor. How much
    of it to hold is then a separate decision, and it is the only place risk aversion enters: ``(mu_T - rf) / (gamma sigma_T^2)`` of wealth, the rest in cash (or borrowed, up to ``max_leverage``). The forecast
    is the expected return of the holding period, the covariance is the trailing shrinkage covariance scaled to the same period, and ``risk_free`` is the annual rate of the cash. ``long_only`` solves the tangency
    problem with ``w >= 0``. Estimation error is the whole difficulty (see the mean-variance guide): the tangency weights are the most error-sensitive portfolio there is, so expect extreme positions unless
    ``long_only`` is on or the forecast is shrunk."""

    def __init__(self, risk_aversion: float = 5.0, risk_free: float = 0.0, long_only: bool = True, max_leverage: float = 1.0, lookback: int = 252):
        if risk_aversion <= 0 or risk_free < -0.05 or max_leverage <= 0 or lookback < 60:
            raise ValueError("risk_aversion > 0, max_leverage > 0, lookback >= 60")
        self.risk_aversion, self.risk_free, self.long_only, self.max_leverage, self.lookback = float(risk_aversion), float(risk_free), bool(long_only), float(max_leverage), int(lookback)

    def solve(self, mu, window, horizon, ctx):
        from ..portfolio.covariance import estimate_covariance

        scale = horizon / 252.0
        cov = estimate_covariance(window, "shrinkage", self.lookback, annualise=True) * scale             # covariance over the forecast's holding period
        try:
            t = tangency_portfolio(mu, cov, self.risk_free * scale, long_only=self.long_only)
        except ValueError:
            return pd.Series(0.0, index=mu.index)                                                         # nothing is expected to beat cash: hold cash
        mix = cml_allocation(t, self.risk_free * scale, self.risk_aversion)
        share = float(np.clip(mix["risky_share"], 0.0, self.max_leverage))
        return t.weights * share


@register_allocator("mv_cvar", "Mean-variance under a CVaR limit: maximise the forecast less a variance penalty subject to the average loss in the worst 5% of the trailing scenarios staying below a limit (cash absorbs the rest)")
class MeanVarianceCVaR(_MonthlyForecastBook):
    """Variance does not see the tail, and the tail is where a portfolio is hurt. The scenarios are the overlapping returns of the forecast's holding period over the trailing ``window`` days (about
    ``window`` of them, thinned to ``scenarios``), the limit ``cvar_limit`` is the largest average loss in the worst ``1 - alpha`` of them, in the same units as the forecast (0.05 is 5% of the
    fund over the holding period), and the portfolio is the best mean-variance one that respects it. Long-only, each name at most ``max_weight``; with ``fully_invested`` off, a limit that cannot be met by
    diversification is met by holding cash. Solved by Kelley cutting planes with an interior-point method (see :mod:`src.equity.cvar_portfolio`)."""

    min_assets = 3

    def __init__(self, cvar_limit: float = 0.05, alpha: float = 0.95, risk_aversion: float = 5.0, max_weight: float = 0.4, fully_invested: bool = False, window: int = 756, scenarios: int = 400):
        if cvar_limit <= 0 or not 0.5 <= alpha < 1 or risk_aversion < 0 or not 0 < max_weight <= 1 or window < 252 or scenarios < 50:
            raise ValueError("cvar_limit > 0, 0.5 <= alpha < 1, risk_aversion >= 0, 0 < max_weight <= 1, window >= 252, scenarios >= 50")
        self.cvar_limit, self.alpha, self.risk_aversion, self.max_weight = float(cvar_limit), float(alpha), float(risk_aversion), float(max_weight)
        self.fully_invested, self.window, self.scenarios = bool(fully_invested), int(window), int(scenarios)
        self.lookback = int(window)

    def solve(self, mu, window, horizon, ctx):
        h = max(int(horizon), 1)
        daily = window.to_numpy()
        csum = np.vstack([np.zeros((1, daily.shape[1])), np.cumsum(daily, axis=0)])
        R = csum[h:] - csum[:-h]                                                                          # overlapping h-day returns (sums of daily returns)
        if len(R) < 60:
            return None
        if len(R) > self.scenarios:
            R = R[np.linspace(0, len(R) - 1, self.scenarios).astype(int)]
        cap = max(self.max_weight, 1.2 / len(mu))
        result = mean_variance_cvar(mu.to_numpy(), R, self.cvar_limit, self.alpha, self.risk_aversion, lb=0.0, ub=cap, fully_invested=self.fully_invested, names=list(mu.index))
        return result.weights if result.ok else None


@register_allocator("qubo_select", "Best K assets for the mean-variance trade-off found by simulated annealing of a QUBO (the form a quantum annealer takes), held in equal weights")
class QuboSelect(_MonthlyForecastBook):
    """Choosing the best ``k`` of ``n`` assets for a mean-variance investor is a subset problem with ``C(n, k)`` answers. Written as a QUBO, ``x'Qx`` over bits with a penalty that the number of ones is ``k``, it is
    the problem a quantum annealer is built for; here it is solved by simulated annealing with single and pair flips from ``restarts`` random starts, which for the sizes of this package matches exhaustive search
    (see the tests). The chosen assets are held in equal weights. The forecast and covariance are for the holding period."""

    def __init__(self, k: int = 5, risk_aversion: float = 5.0, sweeps: int = 200, restarts: int = 16, lookback: int = 252):
        if k < 1 or risk_aversion < 0 or sweeps < 20 or restarts < 2 or lookback < 60:
            raise ValueError("k >= 1, risk_aversion >= 0, sweeps >= 20, restarts >= 2, lookback >= 60")
        self.k, self.risk_aversion, self.sweeps, self.restarts, self.lookback = int(k), float(risk_aversion), int(sweeps), int(restarts), int(lookback)
        self.min_assets = self.k + 1

    def required_assets(self) -> int:
        return self.k + 1

    def solve(self, mu, window, horizon, ctx):
        from ..portfolio.covariance import estimate_covariance

        cov = estimate_covariance(window, "shrinkage", self.lookback, annualise=True) * (horizon / 252.0)
        mask, _ = anneal_selection(mu.to_numpy(), cov.to_numpy(), self.k, self.risk_aversion, sweeps=self.sweeps, restarts=self.restarts, seed=0)
        if mask.sum() == 0:
            return None
        return pd.Series(mask / mask.sum(), index=mu.index)
