"""Run a registered strategy on any asset class's bundle through the platform's own pipeline and report it compactly.

The futures and crypto workspaces do not have their own backtesters: they build a ``MarketBundle`` (``assets.bundles.futures_bundle``, ``framework.crypto_bundle``) and hand a ``PipelineSpec`` to
``framework.Pipeline``, so a carry strategy on simulated oil, a funding strategy on Deribit perpetuals and an ETF momentum strategy are costed, lagged, volatility-targeted and validated by the same
code. The only things set here are the universe, the rebalance frequency, a volatility target, a regime detector and a one-way cost in basis points (fees plus slippage, which the pipeline's engine
applies to the traded notional).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..framework import Pipeline, PipelineSpec
from ..framework.data import MarketBundle

FREQUENCIES = ("daily", "weekly", "monthly")


def subset_bundle(bundle: MarketBundle, columns: list[str]) -> MarketBundle:
    """The bundle restricted to ``columns`` (assets); the macro panel is kept whole because signals are named after assets and read by name."""
    missing = [c for c in columns if c not in bundle.prices.columns]
    if missing:
        raise ValueError(f"not in this universe: {missing}; available: {list(bundle.prices.columns)}")
    return replace(bundle, prices=bundle.prices[columns], returns=bundle.returns[columns], investable=bundle.investable[columns],
                   high=None if bundle.high is None else bundle.high[columns], low=None if bundle.low is None else bundle.low[columns],
                   volume=None if bundle.volume is None else bundle.volume[columns], asset_class={c: bundle.asset_class.get(c, "unknown") for c in columns},
                   group={c: bundle.group.get(c, "unknown") for c in columns}, market=None)


def _sample(series: pd.Series, every: int = 5) -> pd.Series:
    return series.iloc[::every] if len(series) > 400 else series


def compact_result(result, top_weights: int = 8) -> dict:
    """A pipeline result as a small JSON-ready dictionary: headline metrics, the equity curve and drawdown (with the benchmarks), calendar-year returns, the latest weights and the cost drag."""
    net = result.window.dropna()
    if len(net) < 20:
        raise ValueError("the strategy has fewer than 20 evaluated days (its signal needs more history than the data has)")
    equity = (1.0 + net).cumprod()
    dd = equity / equity.cummax() - 1.0
    idx = _sample(equity).index
    years = [{"year": int(y), "value": float(((1.0 + r).prod() - 1.0)), "days": int(len(r))} for y, r in net.groupby(net.index.year)]
    benches = {}
    for name, ret in (result.benchmarks or {}).items():
        b = ret.reindex(net.index).dropna()
        if len(b) > 20:
            benches[name] = [round(float(v), 5) for v in (1.0 + b).cumprod().reindex(idx).to_numpy()]
    last = result.weights.iloc[-1].dropna() if len(result.weights) else pd.Series(dtype=float)
    last = last[last.abs() > 1e-6].reindex(last.abs().sort_values(ascending=False).index).head(top_weights)
    metrics = {k: v for k, v in result.metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)}
    metrics["hit_rate"] = float((net > 0).mean())
    metrics["gross_exposure_mean"] = float(result.weights.abs().sum(axis=1).loc[net.index[0]:].mean()) if len(result.weights) else float("nan")
    warnings = []
    ruin = net[net <= -1.0 + 1e-12]
    if len(ruin):
        warnings.append(f"The account was wiped out on {ruin.index[0].date()}: the book lost all of its capital in one day (a leveraged position can lose more than 100%). The backtest stops there and every later day is zero; "
                        "lower the volatility target or the leverage to see the strategy without the blow-up.")
    if metrics["gross_exposure_mean"] < 1e-9:
        warnings.append("The strategy held no position on any evaluated day: its signal never fired for this universe (a cross-sectional rule needs at least two assets of the same asset class, and a rule that needs "
                        "history needs more days than the data has).")
    return {"start": str(net.index[0].date()), "end": str(net.index[-1].date()), "n_days": int(len(net)), "metrics": metrics,
            "dates": [str(d.date()) for d in idx], "equity": [round(float(v), 5) for v in equity.reindex(idx).to_numpy()], "drawdown": [round(float(v), 5) for v in dd.reindex(idx).to_numpy()],
            "benchmarks": benches, "years": years, "weights": {str(k): round(float(v), 4) for k, v in last.items()},
            "deflated_sharpe_probability": result.validation.get("deflated_sharpe_probability"), "n_trials": result.validation.get("n_trials"), "warnings": warnings}


def run_strategy(bundle: MarketBundle, config, model: str, params: dict | None = None, rebalance: str | None = None, target_vol: float | None = None, regime: str | None = None,
                 cost_bps: float | None = None, label: str | None = None, allocator: str | None = None, min_assets: int = 1, n_trials: int = 1, validate: bool = False) -> dict:
    """Run ``model`` on ``bundle`` and return ``compact_result`` plus the spec that ran. ``rebalance`` daily/weekly/monthly (default: the model's own), ``target_vol`` an annual volatility target
    (default: the platform's), ``regime`` the name of a regime detector, which also switches the volatility target to the regime's, ``cost_bps`` the one-way cost of trading the whole book."""
    spec: dict = {"name": label or model, "models": [{"name": model, "params": dict(params or {})}], "execution": {"min_assets": int(min_assets)},
                  "evaluation": {"benchmarks": ["equal_weight"], "causality": bool(validate), "n_trials": int(n_trials)}}
    if rebalance:
        if rebalance not in FREQUENCIES:
            raise ValueError(f"frequency must be one of {FREQUENCIES}")
        spec["execution"]["rebalance"] = rebalance
    if cost_bps is not None:
        if not 0.0 <= float(cost_bps) <= 500.0:
            raise ValueError("fees plus slippage must be between 0 and 500 basis points")
        spec["execution"]["cost_bps"] = float(cost_bps)
    if allocator:
        spec["allocation"] = {"allocator": allocator}
    if regime:
        spec["regime"] = {"detector": regime}
        spec["risk"] = {"mode": "regime"}
    elif target_vol is not None:
        if not 0.01 <= float(target_vol) <= 1.0:
            raise ValueError("the volatility target must be between 1% and 100%")
        spec["risk"] = {"mode": "constant", "target_vol": float(target_vol)}
    result = Pipeline(PipelineSpec.from_dict(spec), config, bundle).run(validate=validate)
    out = compact_result(result)
    out["spec"] = {k: spec[k] for k in ("models", "execution", "risk", "regime", "allocation") if k in spec}
    out["universe"] = list(bundle.prices.columns)
    return out
