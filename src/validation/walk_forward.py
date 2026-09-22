"""Walk-forward validation (Ch. 22 §22.2.6-§22.2.7, spec §40).

    TRAIN 2006..2014 | TEST 2015
    TRAIN 2006..2015 | TEST 2016
    TRAIN 2006..2016 | TEST 2017        ...

For each fold: estimate everything on the training block, **freeze it**,
generate the test block's returns, and move on. Only test returns are
concatenated, so the reported track record is entirely out of sample.

The ``embargo`` is the detail that is easy to omit and fatal to leave out.
With an ``h``-day forecast horizon, the last ``h`` training observations
overlap the first test observations: their outcome is partly inside the test
block. Dropping ``h`` days between train and test removes that overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    """One train/test split."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def label(self) -> str:
        return f"fold{self.index:02d}_{self.test_start.date()}_{self.test_end.date()}"

    def describe(self) -> dict:
        return {
            "fold": self.index,
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
            "train_days": None,
            "test_days": None,
        }


@dataclass
class WalkForwardSplitter:
    """Generate expanding or rolling folds over a date index."""

    scheme: str = "expanding"
    train_years_min: float = 5.0
    test_months: int = 12
    rolling_train_years: float = 8.0
    embargo_days: int = 21

    @classmethod
    def from_config(cls, config) -> "WalkForwardSplitter":
        node = config.get("backtest.walk_forward", {}) or {}
        return cls(
            scheme=str(node.get("scheme", "expanding")),
            train_years_min=float(node.get("train_years_min", 5)),
            test_months=int(node.get("test_months", 12)),
            rolling_train_years=float(node.get("rolling_train_years", 8)),
            embargo_days=int(node.get("embargo_days", 21)),
        )

    def split(self, index: pd.DatetimeIndex) -> list[Fold]:
        index = pd.DatetimeIndex(index).sort_values()
        if len(index) < 300:
            return []
        start = index[0]
        first_test = start + pd.DateOffset(years=int(self.train_years_min),
                                           months=int((self.train_years_min % 1) * 12))
        folds: list[Fold] = []
        test_start = first_test
        i = 0
        while test_start < index[-1]:
            test_end = min(test_start + pd.DateOffset(months=self.test_months) - pd.Timedelta(days=1),
                           index[-1])
            train_end = test_start - pd.Timedelta(days=self.embargo_days + 1)
            train_start = (start if self.scheme == "expanding"
                           else max(start, train_end - pd.DateOffset(years=int(self.rolling_train_years))))
            if train_end <= train_start:
                test_start = test_end + pd.Timedelta(days=1)
                continue
            if len(index[(index >= test_start) & (index <= test_end)]) < 20:
                break
            i += 1
            folds.append(Fold(i, train_start, train_end, test_start, test_end))
            test_start = test_end + pd.Timedelta(days=1)
        return folds

    def iter_folds(self, index: pd.DatetimeIndex) -> Iterator[Fold]:
        yield from self.split(index)


@dataclass
class WalkForwardResult:
    """Concatenated out-of-sample returns plus per-fold detail."""

    name: str
    oos_returns: pd.Series
    oos_net_returns: pd.Series
    fold_table: pd.DataFrame
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self, periods_per_year: int = 252) -> dict:
        from ..backtest.metrics import performance_summary

        out = performance_summary(self.oos_net_returns, turnover=self.turnover,
                                  costs=self.costs, periods_per_year=periods_per_year)
        out["name"] = self.name
        out["n_folds"] = int(len(self.fold_table))
        if "test_sharpe" in self.fold_table:
            out["folds_positive"] = int((self.fold_table["test_return"] > 0).sum())
            out["fold_sharpe_mean"] = float(self.fold_table["test_sharpe"].mean())
            out["fold_sharpe_std"] = float(self.fold_table["test_sharpe"].std(ddof=1))
            out["worst_fold_return"] = float(self.fold_table["test_return"].min())
        return out


def run_walk_forward(
    index: pd.DatetimeIndex,
    fit_predict: Callable[[Fold], tuple[pd.Series, pd.Series | None, pd.DataFrame | None]],
    splitter: WalkForwardSplitter,
    name: str = "strategy",
    periods_per_year: int = 252,
) -> WalkForwardResult:
    """Drive a walk-forward run.

    ``fit_predict`` receives a fold and must return
    ``(gross_test_returns, net_test_returns, test_weights)``, having used only
    data inside ``fold.train_start .. fold.train_end`` to fit anything. The
    splitter guarantees the embargo; honouring the training window is the
    caller's job, and ``validation.leakage`` is what checks that they did.
    """
    folds = splitter.split(index)
    if not folds:
        raise ValueError("no walk-forward folds could be generated from this index")

    gross_parts, net_parts, weight_parts, rows = [], [], [], []
    for fold in folds:
        try:
            gross, net, weights = fit_predict(fold)
        except Exception as exc:  # a failing fold must be visible, not silent
            LOGGER.warning("%s fold %d failed: %s", name, fold.index, exc)
            rows.append({**fold.describe(), "status": f"failed: {exc}"})
            continue
        if gross is None or gross.dropna().empty:
            rows.append({**fold.describe(), "status": "empty"})
            continue
        net = gross if net is None else net
        gross_parts.append(gross.dropna())
        net_parts.append(net.dropna())
        if weights is not None and len(weights):
            weight_parts.append(weights)

        clean = net.dropna()
        vol = float(clean.std(ddof=1) * np.sqrt(periods_per_year))
        rows.append(
            {
                **fold.describe(),
                "status": "ok",
                "test_days": int(len(clean)),
                "test_return": float((1.0 + clean).prod() - 1.0),
                "test_vol": vol,
                "test_sharpe": float(clean.mean() * periods_per_year / vol) if vol > 0 else np.nan,
                "test_max_drawdown": float(
                    (lambda c: (c / c.cummax() - 1.0).min())((1.0 + clean).cumprod())
                ),
                "test_hit_rate": float((clean > 0).mean()),
            }
        )

    if not net_parts:
        raise RuntimeError(f"walk-forward for '{name}' produced no out-of-sample returns")

    oos_gross = pd.concat(gross_parts).sort_index()
    oos_net = pd.concat(net_parts).sort_index()
    oos_gross = oos_gross[~oos_gross.index.duplicated(keep="first")]
    oos_net = oos_net[~oos_net.index.duplicated(keep="first")]
    weights = pd.concat(weight_parts).sort_index() if weight_parts else pd.DataFrame()
    if len(weights):
        weights = weights[~weights.index.duplicated(keep="first")]

    return WalkForwardResult(
        name=name, oos_returns=oos_gross, oos_net_returns=oos_net,
        fold_table=pd.DataFrame(rows), weights=weights,
    )


def in_sample_vs_out_of_sample(in_sample: pd.Series, out_of_sample: pd.Series,
                               periods_per_year: int = 252) -> dict:
    """Quantify out-of-sample slippage (Ch. 22 §22.2.7).

    The gap between an in-sample and an out-of-sample Sharpe ratio is the most
    honest single measure of how much of a backtest was fitting. A ratio near
    1 means the research process generalised; a ratio near 0 means it did not.
    """
    from ..backtest.metrics import sharpe_ratio

    is_sharpe = sharpe_ratio(in_sample, 0.0, periods_per_year)
    oos_sharpe = sharpe_ratio(out_of_sample, 0.0, periods_per_year)
    return {
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "sharpe_decay": is_sharpe - oos_sharpe,
        "sharpe_retention": oos_sharpe / is_sharpe if abs(is_sharpe) > 1e-9 else np.nan,
        "is_ann_return": float(in_sample.mean() * periods_per_year),
        "oos_ann_return": float(out_of_sample.mean() * periods_per_year),
        "is_n_obs": int(len(in_sample.dropna())),
        "oos_n_obs": int(len(out_of_sample.dropna())),
    }


def purged_kfold_indices(index: pd.DatetimeIndex, n_splits: int = 5, embargo_days: int = 21,
                         horizon: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged, embargoed k-fold indices for the ML stage.

    Standard k-fold on time-series data leaks twice: the training set contains
    observations whose *forecast window* overlaps the test block, and it
    contains observations immediately after it. Purging removes the first,
    the embargo removes the second.
    """
    index = pd.DatetimeIndex(index)
    n = len(index)
    fold_bounds = np.linspace(0, n, n_splits + 1).astype(int)
    out = []
    for i in range(n_splits):
        test_slice = np.arange(fold_bounds[i], fold_bounds[i + 1])
        if len(test_slice) == 0:
            continue
        test_start, test_end = index[test_slice[0]], index[test_slice[-1]]
        purge_start = test_start - pd.Timedelta(days=int(horizon * 1.5) + embargo_days)
        purge_end = test_end + pd.Timedelta(days=embargo_days)
        train_mask = (index < purge_start) | (index > purge_end)
        train_slice = np.where(train_mask)[0]
        if len(train_slice) < 50:
            continue
        out.append((train_slice, test_slice))
    return out
