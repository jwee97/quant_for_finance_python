"""Portfolio overlays: allocators that take another allocator's book and change it for a reason that is not the forecast.

    beta_neutral    market-neutral and hedged books: remove the book's exposure to the market (the equal-weight average of the investable assets), by projecting it out of every position or by
                    shorting one hedge instrument
    liquidity_cap   liquidation risk: no position larger than can be sold in a few days at a small share of the stock's dollar volume
    tca_mvo         portfolio optimisation with transaction-cost analysis: mean-variance in which the spread and the market impact of the trades that move the book from what it holds now are
                    inside the objective, so it trades only what the forecast pays for

``inner`` names the book to start from, as the other allocators that wrap allocators do: ``{"allocator": "forecast_stack", "params": {...}}`` or ``{"book": "risk_parity"}``; the default is
``sleeves`` for a single per-asset strategy and ``forecast_stack`` otherwise. Every overlay uses only data through the origin date it stamps, like the books it wraps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..features.sleeves import month_end_dates
from .allocation import Allocator, Context, book
from .allocators_portfolio import _ForecastBook
from .registry import ALLOCATORS, register_allocator

DEFAULT_INNER = {"allocator": "forecast_stack"}


def default_inner(ctx: Context | None = None) -> dict:
    """The book an overlay wraps when none is named: independent sleeves for a single per-asset rule (the way the dashboard would run it), the calibrated forecast stack otherwise."""
    models = list(getattr(ctx, "models", None) or [])
    if len(models) == 1 and getattr(models[0], "book", None) == "sleeves":
        return {"allocator": "sleeves"}
    return DEFAULT_INNER


def build_inner(spec: dict | None, ctx: Context) -> pd.DataFrame:
    """The book an overlay starts from: an allocator (with its parameters) or a named static book, laid out on the bundle's dates and assets with missing weights as zero."""
    spec = spec or default_inner(ctx)
    if "allocator" in spec:
        raw = ALLOCATORS.create(spec["allocator"], **(spec.get("params") or {})).build(ctx)
    else:
        raw = book(spec.get("book", "risk_parity"), ctx)
    return raw.reindex(index=ctx.bundle.index, columns=ctx.bundle.assets).fillna(0.0)


def inner_required(spec: dict | None) -> int:
    """How many tickers the wrapped book needs (an overlay needs at least that many)."""
    spec = spec or DEFAULT_INNER
    if "allocator" in spec:
        return int(ALLOCATORS.create(spec["allocator"], **(spec.get("params") or {})).required_assets())
    return 1 if spec.get("book") in ("equal_weight", "inverse_vol") else 2


# ------------------------------------------------------------------------------------------------------------------------------- market neutral
def rolling_beta(returns: pd.DataFrame, market: pd.Series, window: int) -> pd.DataFrame:
    """Each asset's beta to the market over the trailing ``window`` days (at least half of it observed), using only data through each date."""
    mp = max(window // 2, 30)
    return returns.rolling(window, min_periods=mp).cov(market).div(market.rolling(window, min_periods=mp).var().replace(0.0, np.nan), axis=0)


@register_allocator("beta_neutral", "Market-neutral overlay: take another allocator's book and remove its exposure to the market (the equal-weight average of the assets) by projection or by shorting one hedge ticker")
class BetaNeutral(Allocator):
    """A book that is long what it likes is also long the market, and its profit and loss mixes the stock-picking it wants to measure with the market move it does not. The overlay estimates each asset's
    beta to the equal-weight average of the investable assets over ``window`` days and removes the book's net beta (down to ``target_beta``): ``hedge=""`` subtracts the multiple of the beta vector
    that makes the beta zero with the smallest change to the weights, ``hedge="SPY"`` instead sells that much of one instrument. ``keep_gross`` rescales the hedged book to the gross exposure it had.
    Assets whose beta cannot be estimated yet get no position, and nothing is held before the beta window has filled."""

    def __init__(self, inner: dict | None = None, window: int = 252, hedge: str = "", target_beta: float = 0.0, keep_gross: bool = True, max_hedge: float = 2.0):
        if window < 60 or max_hedge <= 0:
            raise ValueError("window >= 60 and max_hedge > 0")
        self.inner, self.window, self.hedge, self.target_beta, self.keep_gross, self.max_hedge = inner, int(window), hedge, float(target_beta), bool(keep_gross), float(max_hedge)

    def required_assets(self) -> int:
        return max(2, inner_required(self.inner))                                                           # a market exposure is only worth removing from a book with something to hedge it with

    def build(self, ctx: Context) -> pd.DataFrame:
        bundle = ctx.bundle
        weights = build_inner(self.inner, ctx)
        if self.hedge and self.hedge not in bundle.assets:
            raise KeyError(f"the hedge instrument {self.hedge!r} is not in the universe {list(bundle.assets)}")
        r = bundle.returns.where(bundle.investable)
        market = r.mean(axis=1)
        beta = rolling_beta(r, market, self.window).reindex(columns=weights.columns)
        known = beta.notna().to_numpy()
        b, w = np.where(known, beta.to_numpy(), 0.0), np.where(known, weights.to_numpy(), 0.0)
        exposure = (w * b).sum(axis=1) - self.target_beta
        if self.hedge:
            h = list(weights.columns).index(self.hedge)
            with np.errstate(divide="ignore", invalid="ignore"):
                trade = np.where(known[:, h] & (np.abs(b[:, h]) > 0.1), exposure / b[:, h], 0.0)
            hedged = w.copy()
            hedged[:, h] -= np.clip(trade, -self.max_hedge, self.max_hedge)
        else:
            denom = (b * b).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                step = np.where(denom > 1e-12, exposure / denom, 0.0)
            hedged = w - step[:, None] * b
        if self.keep_gross:
            before, after = np.abs(w).sum(axis=1), np.abs(hedged).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                hedged = hedged * np.where(after > 1e-12, before / after, 1.0)[:, None]
        ready = known.any(axis=1)
        return pd.DataFrame(np.where(ready[:, None], hedged, 0.0), index=weights.index, columns=weights.columns)


# ------------------------------------------------------------------------------------------------------------------------------- liquidation risk
@register_allocator("liquidity_cap", "Liquidation overlay: limit every position to what can be sold in a few days at a small share of the asset's dollar volume (the excess stays in cash)")
class LiquidityCap(Allocator):
    """A position is only as good as your ability to leave it. With the fund worth ``aum`` dollars, an asset trading ``ADV`` dollars a day and a ``participation`` share of volume you are willing to be,
    the weight that can be sold in ``days`` days is ``participation * days * ADV / aum``. The overlay clips every weight to that limit (long or short), using the median dollar volume of the last
    ``window`` days; an asset whose volume is unknown gets no position, because there is no evidence that it can be sold. What is clipped away is held as cash, not redistributed, so the cap can
    only lower risk. Needs trading volume."""

    def __init__(self, inner: dict | None = None, aum: float = 1e8, participation: float = 0.10, days: float = 5.0, window: int = 20):
        if aum <= 0 or not 0 < participation <= 1 or days <= 0 or window < 5:
            raise ValueError("aum > 0, 0 < participation <= 1, days > 0, window >= 5")
        self.inner, self.aum, self.participation, self.days, self.window = inner, float(aum), float(participation), float(days), int(window)

    def required_assets(self) -> int:
        return inner_required(self.inner)

    def limit(self, bundle) -> pd.DataFrame:
        """The largest weight (a fraction of the fund) each asset may have on each date."""
        if bundle.volume is None:
            raise KeyError("liquidity_cap needs trading volume, which this bundle does not have")
        dollars = (bundle.prices * bundle.volume.reindex_like(bundle.prices)).rolling(self.window, min_periods=max(self.window // 2, 3)).median()
        return (self.participation * self.days * dollars / self.aum).reindex(index=bundle.index, columns=bundle.assets)

    def build(self, ctx: Context) -> pd.DataFrame:
        weights = build_inner(self.inner, ctx)
        cap = self.limit(ctx.bundle)
        return weights.clip(lower=-cap, upper=cap).where(cap.notna(), 0.0)


# ------------------------------------------------------------------------------------------------------------------------------- optimisation with TCA
@register_allocator("tca_mvo", "Mean-variance with transaction costs inside the optimisation (half-spread, power-law temporary and permanent impact on the trade from what is held now): it trades only what the forecast pays for")
class TcaMeanVariance(_ForecastBook):
    """Plain mean-variance ignores what its trades cost, so every month it chases forecast noise with expensive turnover. Here each month-end solves

        maximise  mu'w - (risk_aversion / 2) w'Sigma w - periods * sum_i cost_i(|w_i - h_i|)      subject to the configured constraints,

    where ``h`` is what the book holds after drifting with the month's returns and ``cost_i(x)`` is the cost of trading ``x`` of the fund in asset ``i``: half the spread ``a_i x``, the temporary
    impact ``eta sigma_i (x aum / ADV_i)^beta`` paid on the whole trade ``x`` and half the permanent impact ``gamma sigma_i x aum / ADV_i`` (the same model as ``src.algo``; ``ADV_i`` is the median dollar
    volume of the last 20 days). The cost is convex and its slope at zero is the half-spread, so an asset whose expected gain over the risk it adds is smaller than that is left alone (a no-trade
    region), and large trades are spread over more months than small ones. ``periods`` is the number of rebalances a year (12). Needs trading volume. Holdings in assets that have left the investable
    set are dropped without cost. A third-wave transaction-cost-aware optimiser in the vocabulary of the execution literature."""

    def __init__(self, risk_aversion: float = 5.0, aum: float = 1e8, spread_bps: float = 3.0, eta: float = 0.142, beta: float = 0.6, gamma: float = 0.30, periods: int = 12,
                 lookback: int = 252, confidence_weighted: bool = False):
        super().__init__(lookback, min_assets=2)
        if risk_aversion <= 0 or aum <= 0 or spread_bps < 0 or eta < 0 or not 0 < beta <= 2 or gamma < 0 or periods < 1:
            raise ValueError("risk_aversion, aum > 0; spread_bps, eta, gamma >= 0; 0 < beta <= 2; periods >= 1")
        self.risk_aversion, self.aum, self.spread_bps, self.eta, self.beta, self.gamma = float(risk_aversion), float(aum), float(spread_bps), float(eta), float(beta), float(gamma)
        self.periods, self.confidence_weighted = int(periods), bool(confidence_weighted)

    def cost_terms(self, sigma: np.ndarray, adv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(a, c, d)`` per asset so that trading ``x`` (a fraction of the fund) costs ``a x + c x^(1 + beta) + d x^2`` of the fund."""
        turnover = self.aum / adv                                                                          # the fund's weight of one unit of volume
        return (np.full(len(sigma), 0.5e-4 * self.spread_bps), self.eta * sigma * turnover ** self.beta, 0.5 * self.gamma * sigma * turnover)

    def optimise(self, mu: np.ndarray, cov: np.ndarray, held: np.ndarray, sigma: np.ndarray, adv: np.ndarray, assets: list, constraints) -> np.ndarray | None:
        n = len(mu)
        a, c, d = self.cost_terms(sigma, adv)
        lam, per, beta = self.risk_aversion, float(self.periods), self.beta

        def parts(z):
            return held + z[:n] - z[n:], np.maximum(z[:n], 0.0), np.maximum(z[n:], 0.0)

        def objective(z):
            w, buy, sell = parts(z)
            cost = (a * (buy + sell)).sum() + (c * (buy ** (1.0 + beta) + sell ** (1.0 + beta))).sum() + (d * (buy * buy + sell * sell)).sum()
            return float(-(w @ mu) + 0.5 * lam * (w @ cov @ w) + per * cost)

        def gradient(z):
            w, buy, sell = parts(z)
            g = -mu + lam * (cov @ w)
            return np.concatenate([g + per * (a + (1.0 + beta) * c * buy ** beta + 2.0 * d * buy), -g + per * (a + (1.0 + beta) * c * sell ** beta + 2.0 * d * sell)])

        lo, hi = constraints.min_weight, constraints.max_weight
        eye = np.eye(n)
        cons = [{"type": "ineq", "fun": lambda z: np.concatenate([parts(z)[0] - lo, hi - parts(z)[0]]), "jac": lambda z: np.vstack([np.hstack([eye, -eye]), np.hstack([-eye, eye])])}]
        for spec in constraints.scipy_constraints(assets, held):
            cons.append({"type": spec["type"], "fun": (lambda z, f=spec["fun"]: f(parts(z)[0]))})
        start = held if held.sum() > 0.5 else np.full(n, 1.0 / n)
        z0 = np.concatenate([np.maximum(start - held, 0.0), np.maximum(held - start, 0.0)])
        result = minimize(objective, z0, jac=gradient, method="SLSQP", bounds=[(0.0, None)] * (2 * n), constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
        w = parts(result.x)[0]
        return w if result.success and np.isfinite(w).all() else None

    def build(self, ctx: Context) -> pd.DataFrame:
        from ..portfolio.covariance import estimate_covariance, nearest_positive_definite

        if ctx.forecasts is None:
            raise ValueError("tca_mvo needs forecasts")
        fc, bundle = ctx.forecasts, ctx.bundle
        if bundle.volume is None:
            raise KeyError("tca_mvo needs trading volume, which this bundle does not have")
        dollars = (bundle.prices * bundle.volume.reindex_like(bundle.prices)).rolling(20, min_periods=10).median()
        constraints = ctx.constraints()
        scale = 252.0 / fc.horizon
        out = pd.DataFrame(np.nan, index=bundle.index, columns=bundle.assets)
        held, last = pd.Series(0.0, index=bundle.assets), None
        for origin in month_end_dates(bundle.index):
            position = bundle.index.get_loc(origin)
            if position < self.lookback or origin not in fc.mean.index:
                continue
            window = bundle.returns.iloc[position - self.lookback + 1:position + 1]
            mu_row = fc.mean.loc[origin]
            live = [x for x in bundle.assets if bool(bundle.investable.loc[origin, x]) and window[x].notna().mean() > 0.9 and np.isfinite(mu_row.get(x, np.nan))
                    and np.isfinite(dollars.loc[origin, x]) and dollars.loc[origin, x] > 0]
            if len(live) < self.min_assets:
                continue
            if last is not None:                                                                           # what the previous trade has grown into by now
                growth = (bundle.prices.iloc[position] / bundle.prices.iloc[last]).fillna(1.0)
                held = held * growth
                total = float(held.sum())
                held = held * (float(out.iloc[last].fillna(0.0).sum()) / total) if total > 0 else held
            mu = mu_row[live] * scale
            if self.confidence_weighted:
                mu = mu * fc.confidence.loc[origin, live].fillna(0.5)
            cov = nearest_positive_definite(estimate_covariance(window[live].fillna(0.0), "shrinkage", self.lookback, annualise=True)).to_numpy()
            sigma = window[live].std().to_numpy()
            w = self.optimise(mu.to_numpy(), cov, held.reindex(live).fillna(0.0).to_numpy(), sigma, dollars.loc[origin, live].to_numpy(dtype=float), live, constraints)
            if w is None:
                continue
            row = pd.Series(0.0, index=bundle.assets)
            row[live] = w
            out.loc[origin] = row
            held, last = row, position
        return out.ffill().fillna(0.0)
