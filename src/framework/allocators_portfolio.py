"""Allocators backed by the Generation 2/3 portfolio research: forecast-covariance minimum variance and Bayesian mean-variance.

    dynamic_cov   Global minimum variance on a month-ahead covariance FORECAST (DCC-GARCH, O-GARCH) or, as the control, a rolling shrinkage estimate
    bayesian      Constrained mean-variance under parameter uncertainty (Bayes-Stein, posterior-predictive) or the plug-in sample control
    min_variance, max_diversification   Risk-based books on a trailing shrinkage covariance
    es_policy     A linear softmax policy over price features trained by evolution strategies (the reinforcement-learning allocator)
    kelly         Fractional Kelly (growth-optimal) weights from the forecast mean and a trailing covariance, with a gross-leverage cap
    black_litterman   The model forecasts as views on an equilibrium prior, with view uncertainty from the forecast confidence (Idzorek), then mean-variance

Both stamp weights at month-end origins from data up to that origin, so the engine's execution lag keeps them causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.sleeves import month_end_dates
from .allocation import Allocator, Context
from .registry import register_allocator

DYNAMIC_MODELS = ("dcc", "ogarch", "static")
BAYES_KINDS = ("mvo_sample", "bayes_stein", "bayes_predictive")


@register_allocator("dynamic_cov", "Minimum variance on a month-ahead covariance forecast: DCC-GARCH, O-GARCH, or a rolling shrinkage control (model='static')")
class DynamicCovarianceMinVar(Allocator):
    """The universe is the assets with complete history over the first ``min_train`` days (so the choice never looks ahead); weights are zero before the
    first refit. ``refit_every`` is in trading days: the GARCH parameters are re-estimated that often, the covariance forecast every month-end."""

    def __init__(self, model: str = "dcc", min_train: int = 750, refit_every: int = 504, horizon: int = 21, n_factors: int = 3, lookback: int = 252):
        if model not in DYNAMIC_MODELS:
            raise ValueError(f"model must be one of {DYNAMIC_MODELS}")
        if min_train < 250 or refit_every < 21 or horizon < 1 or n_factors < 1 or lookback < 60:
            raise ValueError("min_train >= 250, refit_every >= 21, horizon >= 1, n_factors >= 1, lookback >= 60")
        self.model, self.min_train, self.refit_every, self.horizon, self.n_factors, self.lookback = model, min_train, refit_every, horizon, n_factors, lookback

    def _universe(self, ctx: Context) -> pd.DataFrame:
        returns = ctx.bundle.returns
        head = returns.iloc[1:self.min_train]
        columns = [c for c in returns.columns if head[c].notna().all()]
        if len(columns) < 3:
            raise ValueError("dynamic_cov needs at least 3 assets with complete history over the training window")
        return returns[columns].iloc[1:].fillna(0.0)

    def build(self, ctx: Context) -> pd.DataFrame:
        from ..portfolio.covariance import estimate_covariance
        from ..portfolio.dynamic_covariance import walk_forward_dynamic_covariance
        from ..portfolio.mean_variance import minimum_variance_weights

        returns = self._universe(ctx)
        origins = month_end_dates(returns.index)
        constraints = ctx.constraints()
        out = pd.DataFrame(np.nan, index=ctx.bundle.index, columns=ctx.bundle.assets)
        if self.model == "static":
            forecasts = {o: estimate_covariance(returns.loc[:o].iloc[-self.lookback:], "shrinkage", self.lookback, annualise=False)
                         for o in origins if returns.index.get_loc(o) >= max(self.min_train, self.lookback)}
        else:
            key = ("dynamic_cov", self.model, self.min_train, self.refit_every, self.horizon, self.n_factors, tuple(returns.columns))
            if key not in ctx.cache:
                ctx.cache[key] = walk_forward_dynamic_covariance(returns, origins, self.min_train, self.refit_every, self.horizon, self.n_factors, (self.model,))
            dyn = ctx.cache[key]
            forecasts = {o: pd.DataFrame(dyn.forecasts[self.model][k], index=dyn.columns, columns=dyn.columns) for k, o in enumerate(dyn.origins)}
        for origin, cov in forecasts.items():
            if not np.isfinite(cov.to_numpy()).all():
                continue
            result = minimum_variance_weights(cov * 252.0, constraints)
            if result.success:
                out.loc[origin, cov.columns] = result.weights.reindex(cov.columns).to_numpy()
        return out.ffill().fillna(0.0).where(ctx.bundle.investable.reindex_like(out).fillna(False), 0.0)


@register_allocator("bayesian", "Mean-variance under parameter uncertainty: kind='bayes_stein' (posterior-mean inputs) or 'bayes_predictive' (average of posterior-draw optima)")
class BayesianMeanVariance(Allocator):
    def __init__(self, kind: str = "bayes_stein", lookback: int = 252, risk_aversion: float = 5.0, nu0: float = 126.0, n_draws: int = 100, seed: int = 11):
        if kind not in BAYES_KINDS:
            raise ValueError(f"kind must be one of {BAYES_KINDS}")
        if lookback < 60 or risk_aversion <= 0 or nu0 <= 0 or n_draws < 1:
            raise ValueError("lookback >= 60, risk_aversion > 0, nu0 > 0, n_draws >= 1")
        self.kind, self.lookback, self.risk_aversion, self.nu0, self.n_draws, self.seed = kind, lookback, risk_aversion, nu0, n_draws, seed

    def build(self, ctx: Context) -> pd.DataFrame:
        from ..portfolio.bayesian import bayesian_book

        bundle = ctx.bundle
        book = bayesian_book(bundle.returns, bundle.investable, self.kind, ctx.constraints(), month_end_dates(bundle.index), self.lookback,
                             risk_aversion=self.risk_aversion, nu0=self.nu0, n_draws=self.n_draws, seed=self.seed)
        return book.reindex(index=bundle.index, columns=bundle.assets).fillna(0.0)


@register_allocator("es_policy", "A linear softmax policy over price features, trained by evolution strategies on net-of-cost mean-variance utility and refit yearly on matured months (long-only, fully invested)")
class EvolutionStrategyPolicy(Allocator):
    """Zero parameters is equal weight, so any tilt has to be earned. Weights are zero until ``min_train`` days have passed; each refit uses only months whose
    next-month-end return was complete by the refit date, and the feature standardisation comes from those same months. Reinforcement learning in the sense of
    policy search on a simulator (the backtest); it is the lowest-priority technique here and Stage 38 found no out-of-sample gain over equal weight."""

    def __init__(self, seeds: tuple = (1,), min_train: int = 1260, refit_every: int = 252, cost: float = 0.0005, risk_aversion: float = 5.0,
                 pairs: int = 32, sigma: float = 0.1, lr: float = 0.05, iterations: int = 60):
        if not seeds or min_train < 504 or refit_every < 21 or cost < 0 or risk_aversion <= 0 or pairs < 2 or sigma <= 0 or lr <= 0 or iterations < 1:
            raise ValueError("need at least one seed; min_train >= 504, refit_every >= 21, cost >= 0, risk_aversion > 0, pairs >= 2, sigma > 0, lr > 0, iterations >= 1")
        self.seeds, self.min_train, self.refit_every, self.cost, self.risk_aversion = tuple(seeds), min_train, refit_every, cost, risk_aversion
        self.pairs, self.sigma, self.lr, self.iterations = pairs, sigma, lr, iterations

    def build(self, ctx: Context) -> pd.DataFrame:
        from ..models.probabilistic import price_features
        from ..models.rl_allocation import evolution_strategies, softmax_weights, utility

        bundle = ctx.bundle
        index, assets = bundle.index, list(bundle.assets)
        months = month_end_dates(index)
        pos = np.array([index.get_loc(d) for d in months])
        pos = pos[pos >= 252]
        cube = np.stack([f.reindex(columns=assets).to_numpy(dtype=float) for f in price_features(bundle.prices).values()], axis=2)[pos]
        nxt = np.append(pos[1:], len(index) - 1)
        prices = bundle.prices.reindex(columns=assets).to_numpy(dtype=float)
        forward = prices[nxt] / prices[pos] - 1.0
        live = np.isfinite(prices[pos]) & bundle.investable.to_numpy()[pos] & np.isfinite(forward)
        live[-1] = False                                                       # the last month has no realised next month
        forward = np.nan_to_num(forward)
        n_features = cube.shape[2]
        out = pd.DataFrame(np.nan, index=index, columns=assets)
        for start in range(self.min_train, len(index), self.refit_every):
            trained = (nxt <= start) & live.any(axis=1)
            if trained.sum() < 24:
                continue
            flat = cube[trained].reshape(-1, n_features)
            mu, sd = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0, ddof=1)
            sd[~np.isfinite(sd) | (sd == 0)] = 1.0
            standardised = (cube - mu) / sd
            fitness = lambda p: utility(p, standardised[trained], forward[trained], n_features, self.risk_aversion, self.cost, live[trained])      # noqa: E731
            fits = [evolution_strategies(fitness, n_features + len(assets), int(seed), self.pairs, self.sigma, self.lr, self.iterations)[0] for seed in self.seeds]
            for i in np.flatnonzero((pos >= start) & (pos < start + self.refit_every)):
                mask = np.isfinite(prices[pos[i]]) & bundle.investable.to_numpy()[pos[i]]
                if not mask.any():
                    continue
                w = np.mean([softmax_weights(standardised[i][None], p[:n_features], p[n_features:], mask[None])[0] for p in fits], axis=0)
                out.iloc[pos[i]] = w
        return out.ffill().fillna(0.0)


class _RollingRiskBook(Allocator):
    """Month-end weights from a trailing shrinkage covariance of the investable assets; zero until ``lookback`` days of history exist."""

    solver = None

    def __init__(self, lookback: int = 252, min_assets: int = 3):
        if lookback < 60 or min_assets < 2:
            raise ValueError("lookback >= 60 and min_assets >= 2")
        self.lookback, self.min_assets = lookback, min_assets

    def build(self, ctx: Context) -> pd.DataFrame:
        from ..portfolio.covariance import estimate_covariance

        bundle = ctx.bundle
        constraints = ctx.constraints()
        out = pd.DataFrame(np.nan, index=bundle.index, columns=bundle.assets)
        for origin in month_end_dates(bundle.index):
            position = bundle.index.get_loc(origin)
            if position < self.lookback:
                continue
            window = bundle.returns.iloc[position - self.lookback + 1:position + 1]
            live = [a for a in bundle.assets if bool(bundle.investable.loc[origin, a]) and window[a].notna().mean() > 0.9]
            if len(live) < self.min_assets:
                continue
            cov = estimate_covariance(window[live].fillna(0.0), "shrinkage", self.lookback, annualise=True)
            result = self.solver(cov, constraints)
            if result.success:
                out.loc[origin, live] = result.weights.reindex(live).to_numpy()
        return out.ffill().fillna(0.0)


@register_allocator("min_variance", "Global minimum variance on a trailing shrinkage covariance, long-only within the configured weight limits, rebalanced monthly")
class MinimumVariance(_RollingRiskBook):
    @staticmethod
    def solver(cov, constraints):
        from ..portfolio.mean_variance import minimum_variance_weights
        return minimum_variance_weights(cov, constraints)


@register_allocator("max_diversification", "Maximum diversification (Choueifaty-Coignard): maximise weighted-average volatility over portfolio volatility, rebalanced monthly")
class MaximumDiversification(_RollingRiskBook):
    @staticmethod
    def solver(cov, constraints):
        from ..portfolio.diversification import maximum_diversification_weights
        return maximum_diversification_weights(cov, constraints)


class _ForecastBook(Allocator):
    """Month-end weights built from the forecast panel and a trailing shrinkage covariance; the forecast row at the origin uses data through the origin."""

    def __init__(self, lookback: int = 252, min_assets: int = 3):
        if lookback < 60 or min_assets < 2:
            raise ValueError("lookback >= 60 and min_assets >= 2")
        self.lookback, self.min_assets = lookback, min_assets

    def solve(self, mu: pd.Series, cov: pd.DataFrame, confidence: pd.Series, ctx: Context) -> pd.Series | None:
        raise NotImplementedError

    def build(self, ctx: Context) -> pd.DataFrame:
        from ..portfolio.covariance import estimate_covariance

        if ctx.forecasts is None:
            raise ValueError(f"{type(self).__name__} needs forecasts")
        fc, bundle = ctx.forecasts, ctx.bundle
        scale = 252.0 / fc.horizon
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
            cov = estimate_covariance(window[live].fillna(0.0), "shrinkage", self.lookback, annualise=True)
            w = self.solve(mu_row[live] * scale, cov, fc.confidence.loc[origin, live].fillna(0.5), ctx)
            if w is not None:
                out.loc[origin, live] = w.reindex(live).to_numpy()
        return out.ffill().fillna(0.0)


@register_allocator("kelly", "Fractional Kelly: weights proportional to the inverse covariance times the forecast mean, scaled by a Kelly fraction and capped in gross leverage")
class FractionalKelly(_ForecastBook):
    """Full Kelly maximises long-run growth but is extremely sensitive to the inputs, so ``fraction`` of 0.25 to 0.5 is the practitioner's choice (MacLean, Thorp &
    Ziemba 2010). Long-only by default, with the per-asset limit of the configured constraints."""

    def __init__(self, fraction: float = 0.25, max_gross: float = 1.0, long_only: bool = True, lookback: int = 252, confidence_weighted: bool = True):
        super().__init__(lookback)
        if not 0.0 < fraction <= 1.0 or max_gross <= 0:
            raise ValueError("fraction must be in (0, 1] and max_gross > 0")
        self.fraction, self.max_gross, self.long_only, self.confidence_weighted = fraction, max_gross, long_only, confidence_weighted

    def solve(self, mu, cov, confidence, ctx):
        from ..probability.ruin import multi_asset_kelly

        m = mu * (confidence if self.confidence_weighted else 1.0)
        w = pd.Series(self.fraction * multi_asset_kelly(m.to_numpy(), cov.to_numpy()), index=mu.index)
        if self.long_only:
            w = w.clip(lower=0.0)
        w = w.clip(-ctx.constraints().max_weight, ctx.constraints().max_weight)
        gross = w.abs().sum()
        return w * (self.max_gross / gross) if gross > self.max_gross else w


@register_allocator("black_litterman", "Black-Litterman with the model forecasts as absolute views on an equilibrium prior; view uncertainty from the forecast confidence (Idzorek); constrained mean-variance")
class BlackLittermanForecast(_ForecastBook):
    """The prior is the return the (equal-weight) market portfolio implies; each forecast moves the posterior toward itself in proportion to its confidence.
    With zero confidence the portfolio is the market; with full confidence it follows the forecasts."""

    def __init__(self, tau: float = 0.05, risk_aversion: float = 2.5, lookback: int = 252, optimiser_risk_aversion: float = 5.0):
        super().__init__(lookback)
        if tau <= 0 or risk_aversion <= 0 or optimiser_risk_aversion <= 0:
            raise ValueError("tau, risk_aversion and optimiser_risk_aversion must be positive")
        self.tau, self.risk_aversion, self.optimiser_risk_aversion = tau, risk_aversion, optimiser_risk_aversion

    def solve(self, mu, cov, confidence, ctx):
        from ..portfolio.mean_variance import mean_variance_weights

        sigma = cov.to_numpy()
        n = len(mu)
        w_mkt = np.full(n, 1.0 / n)
        pi = self.risk_aversion * sigma @ w_mkt
        c = np.clip(confidence.to_numpy(), 0.01, 0.99)
        tau_sigma = self.tau * sigma
        omega = np.diag((1.0 - c) / c * np.diag(tau_sigma))              # Idzorek: view variance = (1 - c) / c times the prior variance of the view
        ts_inv, om_inv = np.linalg.inv(tau_sigma), np.linalg.inv(omega)
        posterior = np.linalg.solve(ts_inv + om_inv, ts_inv @ pi + om_inv @ mu.to_numpy())
        result = mean_variance_weights(pd.Series(posterior, index=mu.index), cov, self.optimiser_risk_aversion, ctx.constraints())
        return result.weights if result.success else None
