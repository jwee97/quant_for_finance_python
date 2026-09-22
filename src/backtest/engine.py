"""Backtest engine (Ch. 22 §22.2, spec §20-§23).

One class, ``BacktestEngine``, with one job: given target weights and a
market, produce gross and net return streams under an auditable timing
convention.

    R_{p,t} = w_{t-1}' r_t

The engine never computes a signal, never optimises a portfolio and never
chooses a parameter. Everything it needs arrives as an argument. That
separation is what makes the look-ahead test in ``validation.leakage``
meaningful: if the engine cannot see the future, and the weights it is given
were built causally, the result is honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.returns import cumulative_returns, drawdown
from ..utils.logging import get_logger
from .costs import LinearCostModel, breakeven_cost, cost_sensitivity, turnover_statistics
from .execution import build_held_weights, vol_target_scaling
from .metrics import performance_summary

LOGGER = get_logger(__name__)


@dataclass
class BacktestResult:
    """Everything produced by one backtest run."""

    name: str
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    weights: pd.DataFrame            # book actually held
    trades: pd.DataFrame
    turnover: pd.Series
    vol_scalar: pd.Series | None = None
    config: dict = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        return cumulative_returns(self.net_returns.dropna())

    @property
    def gross_equity_curve(self) -> pd.Series:
        return cumulative_returns(self.gross_returns.dropna())

    def drawdown(self) -> pd.Series:
        return drawdown(self.equity_curve)

    def summary(self, risk_free=0.0, benchmark: pd.Series | None = None) -> dict:
        gross = performance_summary(self.gross_returns, risk_free, benchmark=benchmark)
        net = performance_summary(self.net_returns, risk_free, self.turnover, self.costs, benchmark)
        out = {f"gross_{k}": v for k, v in gross.items() if isinstance(v, (int, float))}
        out.update(net)
        out["name"] = self.name
        out["cost_of_trading_sharpe"] = out.get("gross_sharpe", np.nan) - out.get("sharpe", np.nan)
        return out

    def slice(self, start=None, end=None) -> "BacktestResult":
        from ..utils.dates import slice_dates

        return BacktestResult(
            name=self.name,
            gross_returns=slice_dates(self.gross_returns, start, end),
            net_returns=slice_dates(self.net_returns, start, end),
            costs=slice_dates(self.costs, start, end),
            weights=slice_dates(self.weights, start, end),
            trades=slice_dates(self.trades, start, end),
            turnover=slice_dates(self.turnover, start, end),
            vol_scalar=None if self.vol_scalar is None else slice_dates(self.vol_scalar, start, end),
            config=self.config,
        )

    def cost_sensitivity(self, levels=(0.0, 5.0, 10.0, 25.0, 50.0),
                         per_asset_bps: dict | None = None) -> pd.DataFrame:
        return cost_sensitivity(self.gross_returns, self.trades, levels, per_asset_bps)

    def breakeven_cost_bps(self) -> float:
        return breakeven_cost(self.gross_returns, self.trades)

    def write(self, directory: str | Path) -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = {}
        frame = pd.DataFrame(
            {"gross": self.gross_returns, "net": self.net_returns,
             "costs": self.costs, "turnover": self.turnover}
        )
        path = directory / f"{self.name}_returns.csv"
        frame.to_csv(path, index_label="date")
        written["returns"] = path
        path = directory / f"{self.name}_weights.csv"
        self.weights.to_csv(path, index_label="date")
        written["weights"] = path
        return written


@dataclass
class BacktestEngine:
    """Config-driven backtester.

    Parameters mirror ``config/backtest.yaml`` so the engine can be built
    straight from the configuration and the run is reproducible from it.
    """

    signal_lag: int = 1
    rebalance: str = "monthly"
    weight_drift: bool = True
    annualisation: int = 252
    min_assets: int = 5
    cost_model: LinearCostModel = field(default_factory=LinearCostModel)
    target_vol: float | None = None
    vol_lookback: int = 63
    max_leverage: float = 1.5

    @classmethod
    def from_config(cls, config) -> "BacktestEngine":
        costs = LinearCostModel(
            cost_bps=float(config.get("backtest.costs.cost_bps", 10.0)),
            per_asset_bps=dict(config.get("backtest.costs.per_asset_bps", {}) or {}),
        )
        target_vol = None
        if bool(config.get("portfolio.volatility.vol_target_enabled", False)):
            target_vol = float(config.get("portfolio.volatility.target_vol", 0.10))
        return cls(
            signal_lag=int(config.get("backtest.engine.signal_lag", 1)),
            rebalance=str(config.get("backtest.engine.rebalance", "monthly")),
            weight_drift=bool(config.get("backtest.engine.weight_drift", True)),
            annualisation=int(config.get("backtest.engine.annualisation", 252)),
            min_assets=int(config.get("backtest.engine.min_assets", 5)),
            cost_model=costs,
            target_vol=target_vol,
            vol_lookback=int(config.get("portfolio.volatility.lookback", 63)),
            max_leverage=float(config.get("portfolio.volatility.max_leverage_from_vol_target", 1.5)),
        )

    # -- the run ----------------------------------------------------------
    def run(self, target_weights: pd.DataFrame, returns: pd.DataFrame, name: str = "strategy",
            investable: pd.DataFrame | None = None, apply_vol_target: bool | None = None,
            start=None, end=None) -> BacktestResult:
        """Run one backtest.

        ``target_weights`` must be stamped at the date the decision is made,
        using only information available then. The engine applies the
        execution lag itself; callers must not pre-shift.
        """
        weights = target_weights.copy()
        aligned_returns = returns.reindex(weights.index).reindex(columns=weights.columns)

        if investable is not None:
            mask = investable.reindex_like(weights).fillna(False)
            weights = weights.where(mask, 0.0)
            enough = mask.sum(axis=1) >= self.min_assets
            weights = weights.where(enough, 0.0)

        vol_scalar = None
        use_vol_target = self.target_vol is not None if apply_vol_target is None else apply_vol_target
        if use_vol_target and self.target_vol:
            weights, vol_scalar = vol_target_scaling(
                weights, aligned_returns, self.target_vol, self.vol_lookback,
                self.max_leverage, self.annualisation,
            )

        held, traded = build_held_weights(
            weights, aligned_returns, self.rebalance, self.signal_lag, self.weight_drift
        )

        # R_t = w_{t-1}' r_t. The shift here is the second half of the timing
        # convention: `held` is the book in force at the close of t, so it
        # earns the return of t+1.
        gross = (held.shift(1) * aligned_returns).sum(axis=1, skipna=True)
        valid = held.shift(1).notna().any(axis=1) & aligned_returns.notna().any(axis=1)
        gross = gross.where(valid)

        net, costs = self.cost_model.apply(gross, traded)
        turnover = traded.abs().sum(axis=1)

        if start is not None or end is not None:
            from ..utils.dates import slice_dates

            gross, net = slice_dates(gross, start, end), slice_dates(net, start, end)
            costs, turnover = slice_dates(costs, start, end), slice_dates(turnover, start, end)
            held, traded = slice_dates(held, start, end), slice_dates(traded, start, end)

        result = BacktestResult(
            name=name, gross_returns=gross.dropna(), net_returns=net.dropna(), costs=costs,
            weights=held, trades=traded, turnover=turnover, vol_scalar=vol_scalar,
            config={
                "signal_lag": self.signal_lag,
                "rebalance": self.rebalance,
                "weight_drift": self.weight_drift,
                "target_vol": self.target_vol,
                **self.cost_model.describe(),
            },
        )
        stats = turnover_statistics(held, traded, self.annualisation)
        LOGGER.info(
            "%-28s obs=%4d net CAGR=%+.2f%% vol=%.2f%% Sharpe=%+.2f turnover=%.1fx/yr costs=%.0fbp/yr",
            name, len(result.net_returns),
            100 * result.summary().get("cagr", np.nan),
            100 * result.summary().get("ann_vol", np.nan),
            result.summary().get("sharpe", np.nan),
            stats["annualised_turnover"],
            1e4 * costs.mean() * self.annualisation,
        )
        return result

    def run_many(self, books: dict[str, pd.DataFrame], returns: pd.DataFrame,
                 investable: pd.DataFrame | None = None, **kwargs) -> dict[str, BacktestResult]:
        return {name: self.run(weights, returns, name, investable, **kwargs)
                for name, weights in books.items()}


def buy_and_hold(returns: pd.DataFrame, ticker: str, name: str | None = None) -> BacktestResult:
    """Single-asset buy-and-hold reference with genuinely zero turnover."""
    series = returns[ticker].dropna()
    weights = pd.DataFrame(0.0, index=series.index, columns=returns.columns)
    weights[ticker] = 1.0
    trades = pd.DataFrame(0.0, index=series.index, columns=returns.columns)
    trades.iloc[0, trades.columns.get_loc(ticker)] = 1.0
    return BacktestResult(
        name=name or f"buy_hold_{ticker}",
        gross_returns=series, net_returns=series,
        costs=pd.Series(0.0, index=series.index),
        weights=weights, trades=trades,
        turnover=trades.abs().sum(axis=1),
        config={"type": "buy_and_hold", "ticker": ticker},
    )
