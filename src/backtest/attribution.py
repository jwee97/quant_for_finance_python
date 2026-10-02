"""Performance attribution (Generation 2, Priority 8).

Three questions a performance number cannot answer on its own, each with an
accounting identity that is checked to machine precision rather than assumed.

**Where did the active return come from?** Brinson-Fachler attribution splits a
portfolio's return relative to a benchmark, period by period, by asset class:

    allocation_j   = (Wp_j - Wb_j)(Rb_j - Rb)       being in the right classes
    selection_j    = Wb_j (Rp_j - Rb_j)             picking the right assets inside them
    interaction_j  = (Wp_j - Wb_j)(Rp_j - Rb_j)     the two together

and the three sum, over classes, to Rp - Rb exactly in every period.

**How do monthly effects become a multi-period number?** Arithmetic effects do
not add across periods because returns compound. Carino (1999) scales each
period's effects by k_t / K, with k_t = [ln(1+Rp_t) - ln(1+Rb_t)] / (Rp_t - Rb_t)
and K the same ratio for the cumulative returns, which makes the linked effects
sum EXACTLY to the cumulative active return. The same device links contributions
to a total return (the benchmark return is then zero).

**What did the risk come from?** The Euler decomposition of volatility,
sigma_p = sum_i w_i (Sigma w)_i / sigma_p, gives each asset's component and the
components sum to the portfolio volatility.

Weights here are the weights that EARN each period's return: the book held at
the close of the previous period, which is what the engine's timing produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


# ---------------------------------------------------------------------------
# Carino linking
# ---------------------------------------------------------------------------
def carino_factor(portfolio, benchmark=0.0) -> np.ndarray:
    """k = [ln(1+Rp) - ln(1+Rb)] / (Rp - Rb); the limit 1/(1+R) when the two are equal."""
    rp = np.asarray(portfolio, dtype=float)
    rb = np.broadcast_to(np.asarray(benchmark, dtype=float), rp.shape)
    gap = rp - rb
    safe = np.where(np.abs(gap) < EPS, 1.0, gap)
    return np.where(np.abs(gap) < EPS, 1.0 / (1.0 + rp), (np.log1p(rp) - np.log1p(rb)) / safe)


def cumulative(returns) -> float:
    return float(np.prod(1.0 + np.asarray(returns, dtype=float)) - 1.0)


def link_effects(effects: pd.DataFrame, portfolio: pd.Series, benchmark: pd.Series | None = None) -> pd.Series:
    """Carino-linked effects: they sum to cumulative(portfolio) - cumulative(benchmark) exactly.

    ``effects`` is periods x components; ``portfolio`` and ``benchmark`` are the
    per-period returns the effects decompose the difference of (the benchmark
    defaults to zero, which links contributions to a total return).
    """
    portfolio = portfolio.reindex(effects.index)
    benchmark = (benchmark.reindex(effects.index) if benchmark is not None
                 else pd.Series(0.0, index=effects.index))
    k = carino_factor(portfolio.to_numpy(), benchmark.to_numpy())
    total_p, total_b = cumulative(portfolio), cumulative(benchmark)
    big_k = float(carino_factor(total_p, total_b))
    scaled = effects.mul(k / big_k, axis=0)
    return scaled.sum(axis=0)


# ---------------------------------------------------------------------------
# Brinson-Fachler
# ---------------------------------------------------------------------------
@dataclass
class BrinsonResult:
    allocation: pd.DataFrame
    selection: pd.DataFrame
    interaction: pd.DataFrame
    portfolio_return: pd.Series
    benchmark_return: pd.Series
    portfolio_sector_weight: pd.DataFrame
    benchmark_sector_weight: pd.DataFrame

    @property
    def total(self) -> pd.DataFrame:
        return self.allocation + self.selection + self.interaction

    def active_return(self) -> pd.Series:
        return self.portfolio_return - self.benchmark_return

    def identity_error(self) -> float:
        """Largest per-period gap between the sum of effects and Rp - Rb (zero up to rounding)."""
        return float((self.total.sum(axis=1) - self.active_return()).abs().max())

    def linked(self) -> pd.DataFrame:
        """Carino-linked effects by class; the grand total equals the cumulative active return."""
        out = pd.DataFrame({
            "allocation": link_effects(self.allocation, self.portfolio_return, self.benchmark_return),
            "selection": link_effects(self.selection, self.portfolio_return, self.benchmark_return),
            "interaction": link_effects(self.interaction, self.portfolio_return, self.benchmark_return),
        })
        out["total"] = out.sum(axis=1)
        return out

    def cumulative_active_return(self) -> float:
        return cumulative(self.portfolio_return) - cumulative(self.benchmark_return)


def brinson_fachler(portfolio_weights: pd.DataFrame, benchmark_weights: pd.DataFrame,
                    asset_returns: pd.DataFrame, sector_of: dict[str, str]) -> BrinsonResult:
    """Brinson-Fachler effects for every period (rows) and sector (columns).

    ``portfolio_weights`` and ``benchmark_weights`` are the weights at the START
    of each period and must each sum to one; ``asset_returns`` are the assets'
    returns over the period. Empty sectors use the other side's return so that
    no effect appears from a sector that is not held (the identity does not
    depend on that convention).
    """
    columns = list(asset_returns.columns)
    sectors = sorted(set(sector_of[c] for c in columns))
    wp = portfolio_weights.reindex(index=asset_returns.index, columns=columns).fillna(0.0)
    wb = benchmark_weights.reindex(index=asset_returns.index, columns=columns).fillna(0.0)
    for name, frame in (("portfolio", wp), ("benchmark", wb)):
        gap = (frame.sum(axis=1) - 1.0).abs().max()
        if gap > 1e-6:
            raise ValueError(f"{name} weights must sum to one in every period (largest gap {gap:.2e})")
    rp_total = (wp * asset_returns).sum(axis=1)
    rb_total = (wb * asset_returns).sum(axis=1)

    allocation = pd.DataFrame(0.0, index=asset_returns.index, columns=sectors)
    selection, interaction = allocation.copy(), allocation.copy()
    sector_wp, sector_wb = allocation.copy(), allocation.copy()
    for sector in sectors:
        members = [c for c in columns if sector_of[c] == sector]
        w_p, w_b = wp[members].sum(axis=1), wb[members].sum(axis=1)
        r_p = (wp[members] * asset_returns[members]).sum(axis=1) / w_p.where(w_p.abs() > EPS)
        r_b = (wb[members] * asset_returns[members]).sum(axis=1) / w_b.where(w_b.abs() > EPS)
        r_b = r_b.fillna(rb_total)
        r_p = r_p.fillna(r_b)
        allocation[sector] = (w_p - w_b) * (r_b - rb_total)
        selection[sector] = w_b * (r_p - r_b)
        interaction[sector] = (w_p - w_b) * (r_p - r_b)
        sector_wp[sector], sector_wb[sector] = w_p, w_b
    return BrinsonResult(allocation, selection, interaction, rp_total, rb_total, sector_wp, sector_wb)


# ---------------------------------------------------------------------------
# From a backtest to attribution inputs
# ---------------------------------------------------------------------------
def monthly_inputs(held: pd.DataFrame, daily_returns: pd.DataFrame, gross_returns: pd.Series | None = None) -> dict:
    """Start-of-month earning weights, monthly asset returns and (optionally) the portfolio's monthly return.

    ``held`` is the book in force at each close; the weights that earn day t's
    return are ``held`` on day t-1, so a month's starting weights are the ones
    held at the previous month's last close.
    """
    index = pd.DatetimeIndex(held.index)
    earning = held.shift(1)
    months = index.to_period("M")
    first = earning.groupby(months).head(1)
    first.index = months[~months.duplicated()]
    asset = (1.0 + daily_returns.reindex(index).fillna(0.0)).groupby(months).prod() - 1.0
    out = {"weights": first.reindex(asset.index), "asset_returns": asset}
    if gross_returns is not None:
        out["portfolio_return"] = (1.0 + gross_returns.reindex(index).fillna(0.0)).groupby(months).prod() - 1.0
    return out


def contribution_by_asset(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Per-period contribution w_i * R_i; across assets it sums to the period's portfolio return."""
    return weights.reindex_like(asset_returns).fillna(0.0) * asset_returns


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
def euler_risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """Component volatility of each asset: w_i (Sigma w)_i / sigma_p; the components sum to sigma_p."""
    w = weights.reindex(covariance.columns).fillna(0.0).to_numpy(dtype=float)
    cov = covariance.to_numpy(dtype=float)
    variance = float(w @ cov @ w)
    sigma = float(np.sqrt(max(variance, 0.0)))
    if sigma < EPS:
        return pd.DataFrame({"weight": w, "marginal": 0.0, "component": 0.0, "share": 0.0}, index=covariance.columns)
    marginal = cov @ w / sigma
    component = w * marginal
    return pd.DataFrame({"weight": w, "marginal": marginal, "component": component,
                         "share": component / sigma}, index=covariance.columns)


def cost_by_asset(trades: pd.DataFrame, rates: pd.Series) -> pd.DataFrame:
    """Daily transaction cost of each asset: |traded weight| times its one-way rate."""
    return trades.abs().mul(rates.reindex(trades.columns), axis=1)
