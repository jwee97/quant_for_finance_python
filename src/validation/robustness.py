"""Robustness analysis (spec §42-§43, Ch. 22 §22.2.4).

What this module is for: distinguishing a result from an artefact. Four tools.

``parameter_surface``    evaluate a whole parameter grid, not one point. A
                         broad plateau of positive results is evidence; a
                         single narrow spike is parameter mining.
``block_bootstrap``      confidence intervals for the Sharpe ratio that respect
                         serial dependence, by resampling blocks rather than
                         individual days.
``subperiod_stability``  the same statistics computed decade by decade.
``sensitivity_report``   costs, execution lag, rebalance frequency, universe
                         exclusion -- the assumptions that were chosen rather
                         than discovered.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..backtest.metrics import deflated_sharpe_ratio, sharpe_ratio
from ..utils.dates import TRADING_DAYS_PER_YEAR
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)
ANN = TRADING_DAYS_PER_YEAR


def parameter_surface(evaluate: Callable[[dict], dict], grid: dict[str, list],
                      progress: bool = False) -> pd.DataFrame:
    """Evaluate ``evaluate`` over the full Cartesian product of ``grid``."""
    import itertools

    keys = list(grid)
    combinations = list(itertools.product(*(grid[k] for k in keys)))
    rows = []
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        try:
            result = evaluate(params)
        except Exception as exc:
            LOGGER.warning("parameter set %s failed: %s", params, exc)
            continue
        if result:
            rows.append({**params, **result})
        if progress and (i + 1) % 10 == 0:
            LOGGER.info("  evaluated %d/%d parameter sets", i + 1, len(combinations))
    return pd.DataFrame(rows)


def surface_diagnostics(surface: pd.DataFrame, metric: str = "sharpe") -> dict:
    """Is the parameter surface a plateau or a spike?

    ``share_positive`` and the gap between the best result and the median are
    the numbers that matter. A strategy whose best Sharpe is 1.5 while its
    median is 0.1 has one lucky parameter, not an effect.
    """
    if surface.empty or metric not in surface:
        return {}
    values = surface[metric].dropna()
    if values.empty:
        return {}
    best, median = float(values.max()), float(values.median())
    return {
        "n_parameter_sets": int(len(values)),
        "best": best,
        "median": median,
        "worst": float(values.min()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "share_positive": float((values > 0).mean()),
        "best_minus_median": best - median,
        "best_to_median_ratio": best / median if abs(median) > 1e-9 else np.nan,
        "interquartile_range": float(values.quantile(0.75) - values.quantile(0.25)),
        "verdict": (
            "plateau: results are broadly similar across parameters"
            if len(values) > 1 and abs(best - median) < max(0.5 * abs(values.std(ddof=1)) + 0.15, 0.2)
            else "spike: the best parameter is far above the median, treat as parameter mining"
        ),
    }


def stationary_block_bootstrap(returns: pd.Series, n_samples: int = 2000, block_length: int = 21,
                               statistic: Callable[[pd.Series], float] | None = None,
                               seed: int = 7, periods_per_year: int = ANN) -> dict:
    """Bootstrap confidence intervals that respect serial dependence.

    Resampling individual days destroys volatility clustering and autocorrelation
    and produces confidence intervals that are far too narrow. Politis and
    Romano's stationary bootstrap resamples blocks of geometrically distributed
    length instead, preserving short-range dependence.
    """
    clean = returns.dropna()
    if len(clean) < 100:
        return {}
    statistic = statistic or (lambda s: sharpe_ratio(s, 0.0, periods_per_year))
    rng = np.random.default_rng(seed)
    values = clean.to_numpy()
    n = len(values)
    p = 1.0 / max(block_length, 1)

    draws = np.empty(n_samples)
    for i in range(n_samples):
        sample = np.empty(n)
        position = 0
        index = rng.integers(0, n)
        while position < n:
            sample[position] = values[index % n]
            position += 1
            index = rng.integers(0, n) if rng.random() < p else index + 1
        draws[i] = statistic(pd.Series(sample, index=clean.index))

    observed = statistic(clean)
    return {
        "observed": float(observed),
        "bootstrap_mean": float(np.mean(draws)),
        "bootstrap_std": float(np.std(draws, ddof=1)),
        "ci_lower_5pct": float(np.percentile(draws, 5)),
        "ci_upper_95pct": float(np.percentile(draws, 95)),
        "ci_lower_2_5pct": float(np.percentile(draws, 2.5)),
        "ci_upper_97_5pct": float(np.percentile(draws, 97.5)),
        "p_value_vs_zero": float((draws <= 0).mean()),
        "n_samples": int(n_samples),
        "block_length": int(block_length),
    }


def subperiod_stability(returns: pd.Series, periods: dict[str, tuple] | None = None,
                        periods_per_year: int = ANN) -> pd.DataFrame:
    """Statistics per sub-period. Defaults to calendar years."""
    from ..backtest.metrics import performance_summary
    from ..utils.dates import slice_dates

    clean = returns.dropna()
    if periods is None:
        years = sorted(set(clean.index.year))
        periods = {str(y): (f"{y}-01-01", f"{y}-12-31") for y in years}

    rows = []
    for name, (start, end) in periods.items():
        block = slice_dates(clean, start, end)
        if len(block) < 20:
            continue
        summary = performance_summary(block, periods_per_year=periods_per_year)
        rows.append(
            {
                "period": name,
                "n_obs": summary.get("n_obs"),
                "return": float((1.0 + block).prod() - 1.0),
                "vol": summary.get("ann_vol"),
                "sharpe": summary.get("sharpe"),
                "max_drawdown": summary.get("max_drawdown"),
                "hit_rate": summary.get("hit_rate"),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["positive"] = frame["return"] > 0
    return frame


def sensitivity_report(build_and_run: Callable[[dict], pd.Series], base_params: dict,
                       variations: dict[str, list], periods_per_year: int = ANN) -> pd.DataFrame:
    """Vary one assumption at a time and report what it does to the result."""
    rows = []
    baseline = build_and_run(base_params)
    base_sharpe = sharpe_ratio(baseline, 0.0, periods_per_year)
    rows.append({"assumption": "baseline", "value": "-", "sharpe": base_sharpe,
                 "ann_return": float(baseline.mean() * periods_per_year), "delta_sharpe": 0.0})

    for key, options in variations.items():
        for option in options:
            params = {**base_params, key: option}
            try:
                series = build_and_run(params)
            except Exception as exc:
                LOGGER.warning("sensitivity %s=%s failed: %s", key, option, exc)
                continue
            value = sharpe_ratio(series, 0.0, periods_per_year)
            rows.append(
                {
                    "assumption": key,
                    "value": str(option),
                    "sharpe": value,
                    "ann_return": float(series.mean() * periods_per_year),
                    "delta_sharpe": value - base_sharpe,
                }
            )
    return pd.DataFrame(rows)


def multiple_testing_penalty(best_sharpe: float, n_trials: int, n_obs: int,
                             skew: float = 0.0, kurtosis: float = 3.0,
                             periods_per_year: int = ANN) -> dict:
    """What is left of the best backtest once the search is accounted for?"""
    deflated = deflated_sharpe_ratio(best_sharpe, n_trials, n_obs, skew, kurtosis, periods_per_year)
    return {
        "observed_best_sharpe": best_sharpe,
        "n_trials": int(n_trials),
        "n_obs": int(n_obs),
        "deflated_sharpe_probability": deflated,
        "survives_at_95pct": bool(np.isfinite(deflated) and deflated > 0.95),
        "interpretation": (
            "the best result is unlikely to be selection alone"
            if np.isfinite(deflated) and deflated > 0.95
            else "the best result is within what searching this many variants produces by chance"
        ),
    }
