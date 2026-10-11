"""Running an ML strategy through the platform's own backtest engine.

There is no separate ML backtester. Target weights stamped at the close of ``t`` go to ``backtest.engine.BacktestEngine``, which delays them by the signal lag, trades only on the rebalance schedule, lets
weights drift with prices, charges a linear cost on the traded notional (fee plus slippage, one way) and, if asked, scales the book to a volatility target with a scalar known the day before. The
same engine runs every other strategy on the platform, so an ML Sharpe ratio and a momentum Sharpe ratio are measured by the same ruler.

Returns are cut into the segments of the timeline by the date the profit was REALISED, so a position decided in the validation segment and held into the test segment earns its test-segment days in the
test segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.costs import LinearCostModel
from ..backtest.engine import BacktestEngine, BacktestResult


def total_cost_bps(bt_cfg: dict, multiple: float = 1.0) -> float:
    """Fee plus slippage in basis points, one way, optionally scaled (the robustness tests double and triple it)."""
    return (float(bt_cfg["cost_bps"]) + float(bt_cfg["slippage_bps"])) * float(multiple)


def make_engine(bt_cfg: dict, lag: int = 1, cost_multiple: float = 1.0, extra_lag: int | None = None) -> BacktestEngine:
    """The engine for a ``backtest`` configuration section. ``lag`` is the dataset's lag (labels and trades must start at the same time); ``extra_lag`` defaults to the section's own."""
    extra = int(bt_cfg.get("extra_lag", 0)) if extra_lag is None else int(extra_lag)
    return BacktestEngine(signal_lag=int(lag) + extra, rebalance=str(bt_cfg["rebalance"]), weight_drift=True, annualisation=252, min_assets=1,
                          cost_model=LinearCostModel(cost_bps=total_cost_bps(bt_cfg, cost_multiple)), target_vol=None if bt_cfg.get("target_vol") in (None, 0) else float(bt_cfg["target_vol"]), vol_lookback=63,
                          max_leverage=float(bt_cfg.get("max_leverage", 1.5)))


def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, bt_cfg: dict, lag: int = 1, name: str = "ml_strategy", cost_multiple: float = 1.0,
                 extra_lag: int | None = None) -> BacktestResult:
    """Backtest target ``weights`` (date x asset, stamped at the decision date) against daily simple ``returns``."""
    engine = make_engine(bt_cfg, lag, cost_multiple, extra_lag)
    w = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    return engine.run(w, returns, name, investable, apply_vol_target=engine.target_vol is not None)


def trim_to_signal(result: BacktestResult, first_signal: pd.Timestamp, lag: int) -> dict:
    """The result's return streams from the first day a position could have earned anything (the decision date plus the lag), dropping the idle days before the first prediction."""
    start_pos = result.net_returns.index.searchsorted(pd.Timestamp(first_signal)) + int(lag)
    keep = result.net_returns.index[start_pos:]
    return {"net": result.net_returns.loc[keep], "gross": result.gross_returns.reindex(keep), "costs": result.costs.reindex(keep), "turnover": result.turnover.reindex(keep),
            "weights": result.weights.reindex(keep), "trades": result.trades.reindex(keep)}


def equal_weight_benchmark(returns: pd.DataFrame, investable: pd.DataFrame | None = None) -> pd.Series:
    """The daily-rebalanced equal-weight return of the assets investable that day: what holding the whole universe would have earned."""
    r = returns.where(investable.reindex_like(returns).fillna(False)) if investable is not None else returns
    return r.mean(axis=1, skipna=True).fillna(0.0)


def label_days(index: pd.DatetimeIndex, segment_labels: pd.Series) -> pd.Series:
    """The segment of every date in ``index``: the plan's labels, carried forward over the unlabelled tail (which belongs to the forward segment) and back over any lead-in."""
    s = segment_labels.reindex(index.union(segment_labels.index)).sort_index().ffill().bfill()
    return s.reindex(index)


def annualised_turnover(turnover: pd.Series) -> float:
    return float(turnover.mean() * 252.0) if len(turnover) else float("nan")


def cost_drag_bps(costs: pd.Series) -> float:
    return float(costs.mean() * 252.0 * 1e4) if len(costs) else float("nan")


def sharpe_net_of_costs(gross: pd.Series, costs: pd.Series) -> float:
    """Sharpe of gross returns less costs (a convenience for the cost-sensitivity table)."""
    net = (gross - costs.reindex(gross.index).fillna(0.0)).dropna()
    sd = net.std(ddof=1)
    return float(net.mean() / sd * np.sqrt(252.0)) if sd and sd > 0 else float("nan")
