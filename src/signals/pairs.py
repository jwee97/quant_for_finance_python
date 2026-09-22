"""Pairs trading -- Extension E (Ch. 22 §22.3.3-§22.3.4, spec §53).

Kept as a **separate research branch**, deliberately not mixed into the
multi-asset book. Pairs trading is a different animal: it is a relative-value
strategy on a spread that must first be shown to be stationary, whereas
everything in the core project is a directional allocation across assets.

The pipeline is the same discipline applied to a different object:

1. **Establish the relationship exists** -- cointegration (Engle-Granger),
   not merely correlation. Two assets can be 0.95 correlated and have a
   spread that wanders off forever.
2. **Estimate the hedge ratio** on training data only.
3. **Measure the spread's half-life.** If it is longer than the holding
   period, there is nothing to trade.
4. **Only then** build entry and exit rules, and charge them costs.

The book's worked example is gold versus gold miners; GLD/SLV is the closest
analogue available in this universe and is used as the headline pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.mean_reversion import half_life_of_reversion


@dataclass
class PairResult:
    """Everything needed to judge whether a pair is tradable."""

    left: str
    right: str
    hedge_ratio: float
    intercept: float
    adf_statistic: float
    adf_pvalue: float
    half_life: float
    spread_volatility: float
    correlation: float
    n_obs: int

    @property
    def is_cointegrated(self) -> bool:
        return bool(self.adf_pvalue < 0.05)

    @property
    def tradable(self) -> bool:
        """Cointegrated *and* reverting fast enough to trade."""
        return self.is_cointegrated and 1.0 < self.half_life < 126.0

    def to_row(self) -> dict:
        return {
            "pair": f"{self.left}/{self.right}",
            "hedge_ratio": self.hedge_ratio,
            "correlation": self.correlation,
            "adf_stat": self.adf_statistic,
            "adf_pvalue": self.adf_pvalue,
            "half_life_days": self.half_life,
            "spread_vol": self.spread_volatility,
            "cointegrated": self.is_cointegrated,
            "tradable": self.tradable,
            "n_obs": self.n_obs,
        }


def estimate_hedge_ratio(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    """OLS of ``log(left)`` on ``log(right)``: the cointegrating vector."""
    frame = pd.concat([np.log(left).rename("y"), np.log(right).rename("x")], axis=1).dropna()
    if len(frame) < 60:
        return float("nan"), float("nan")
    x = frame["x"].to_numpy()
    y = frame["y"].to_numpy()
    centred = x - x.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= 0:
        return float("nan"), float("nan")
    beta = float(np.dot(centred, y - y.mean()) / denominator)
    alpha = float(y.mean() - beta * x.mean())
    return beta, alpha


def build_spread(left: pd.Series, right: pd.Series, hedge_ratio: float,
                 intercept: float = 0.0) -> pd.Series:
    """``log(left) - beta log(right) - alpha``."""
    return (np.log(left) - hedge_ratio * np.log(right) - intercept).rename("spread")


def analyse_pair(left: pd.Series, right: pd.Series, left_name: str = "left",
                 right_name: str = "right") -> PairResult:
    """Engle-Granger: estimate the hedge ratio, then test the residual.

    Named ``analyse_pair`` rather than ``test_pair`` because pytest collects
    any module-level callable whose name starts with ``test_``, and importing
    this one into a test module made the suite try to run it as a test case.
    """
    import warnings

    from statsmodels.tsa.stattools import adfuller

    frame = pd.concat([left.rename("l"), right.rename("r")], axis=1).dropna()
    if len(frame) < 250:
        return PairResult(left_name, right_name, np.nan, np.nan, np.nan, 1.0,
                          np.nan, np.nan, np.nan, len(frame))

    beta, alpha = estimate_hedge_ratio(frame["l"], frame["r"])
    spread = build_spread(frame["l"], frame["r"], beta, alpha).dropna()
    try:
        with warnings.catch_warnings():
            # statsmodels is changing adfuller's return type in a future
            # release; the tuple indices used here are stable until then.
            warnings.simplefilter("ignore", FutureWarning)
            adf = adfuller(spread.to_numpy(), maxlag=21, regression="c", autolag="AIC")
        statistic, p_value = float(adf[0]), float(adf[1])
    except Exception:
        statistic, p_value = float("nan"), 1.0

    return PairResult(
        left=left_name, right=right_name, hedge_ratio=beta, intercept=alpha,
        adf_statistic=statistic, adf_pvalue=p_value,
        half_life=half_life_of_reversion(spread),
        spread_volatility=float(spread.std(ddof=1)),
        correlation=float(np.log(frame["l"]).diff().corr(np.log(frame["r"]).diff())),
        n_obs=int(len(frame)),
    )


def screen_pairs(prices: pd.DataFrame, candidates: list[tuple[str, str]] | None = None,
                 sample_end=None) -> pd.DataFrame:
    """Test every candidate pair, on training data only.

    ``sample_end`` restricts the cointegration test to the training window, so
    the pair is selected without seeing the period it will be traded in --
    the pairs-trading version of the look-ahead problem, and an easy one to
    get wrong.
    """
    data = prices if sample_end is None else prices.loc[prices.index <= pd.Timestamp(sample_end)]
    if candidates is None:
        columns = list(data.columns)
        candidates = [(a, b) for i, a in enumerate(columns) for b in columns[i + 1:]]

    rows = []
    for left, right in candidates:
        if left not in data.columns or right not in data.columns:
            continue
        rows.append(analyse_pair(data[left], data[right], left, right).to_row())
    frame = pd.DataFrame(rows)
    return frame.sort_values("adf_pvalue").reset_index(drop=True) if len(frame) else frame


def pairs_positions(left: pd.Series, right: pd.Series, hedge_ratio: float, intercept: float,
                    window: int = 63, entry: float = 2.0, exit_threshold: float = 0.5,
                    stop: float = 4.0) -> pd.DataFrame:
    """Entry/exit rules on the spread z-score, causally.

    Enter when the spread is ``entry`` standard deviations from its rolling
    mean, exit at ``exit_threshold``, stop out at ``stop``. The z-score uses a
    *rolling* mean and standard deviation rather than the full-sample ones,
    because full-sample moments are not knowable at the time of the trade --
    the most common way a pairs backtest cheats.
    """
    spread = build_spread(left, right, hedge_ratio, intercept)
    mean = spread.rolling(window, min_periods=window // 2).mean()
    std = spread.rolling(window, min_periods=window // 2).std(ddof=1)
    z = (spread - mean) / std.replace(0.0, np.nan)

    position = pd.Series(0.0, index=z.index)
    state = 0.0
    for stamp, value in z.items():
        if not np.isfinite(value):
            state = 0.0
        elif state == 0.0:
            if value > entry:
                state = -1.0          # spread rich: short left, long right
            elif value < -entry:
                state = 1.0
        else:
            if abs(value) < exit_threshold or abs(value) > stop:
                state = 0.0
        position.loc[stamp] = state

    return pd.DataFrame(
        {
            "spread": spread,
            "zscore": z,
            "position": position,
            "weight_left": position,
            "weight_right": -position * hedge_ratio,
        }
    )


def backtest_pair(left: pd.Series, right: pd.Series, result: PairResult, window: int = 63,
                  entry: float = 2.0, exit_threshold: float = 0.5, cost_bps: float = 10.0,
                  lag: int = 1) -> dict:
    """Backtest one pair with the project's timing and cost conventions."""
    from ..backtest.metrics import performance_summary

    signals = pairs_positions(left, right, result.hedge_ratio, result.intercept,
                              window, entry, exit_threshold)
    left_returns = left.pct_change()
    right_returns = right.pct_change()

    held_left = signals["weight_left"].shift(lag)
    held_right = signals["weight_right"].shift(lag)
    gross = (held_left * left_returns + held_right * right_returns).dropna()

    traded = (held_left.diff().abs() + held_right.diff().abs()).reindex(gross.index).fillna(0.0)
    costs = traded * cost_bps / 1e4
    net = gross - costs

    gross_leverage = (held_left.abs() + held_right.abs()).reindex(gross.index).fillna(0.0)
    summary = performance_summary(net)
    summary.update(
        {
            "pair": f"{result.left}/{result.right}",
            "gross_sharpe": performance_summary(gross).get("sharpe", np.nan),
            "ann_turnover": float(traded.mean() * 252),
            "share_of_days_in_position": float((gross_leverage > 1e-9).mean()),
            "n_round_trips": int((signals["position"].diff().abs() > 0).sum() / 2),
            "half_life_days": result.half_life,
            "adf_pvalue": result.adf_pvalue,
        }
    )
    return summary
