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
from ..risk import performance

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
    out = {"start": str(net.index[0].date()), "end": str(net.index[-1].date()), "n_days": int(len(net)), "metrics": metrics,
           "dates": [str(d.date()) for d in idx], "equity": [round(float(v), 5) for v in equity.reindex(idx).to_numpy()], "drawdown": [round(float(v), 5) for v in dd.reindex(idx).to_numpy()],
           "benchmarks": benches, "years": years, "weights": {str(k): round(float(v), 4) for k, v in last.items()},
           "deflated_sharpe_probability": result.validation.get("deflated_sharpe_probability"), "n_trials": result.validation.get("n_trials"), "warnings": warnings}
    out.update(_risk_views(result, net, idx))
    return out


def _risk_views(result, net: pd.Series, idx: pd.Index) -> dict:
    """What the Backtest page shows beside the equity curve, for a pipeline result: the risk measures, the deepest falls, the volatility through time, the months and the distribution of days, the
    benchmark's beta and alpha, what each asset earned, and how the book was spread through time."""
    rolling = performance.rolling_volatility(net, 63).reindex(idx)
    bench_returns = {name: ret.reindex(net.index) for name, ret in (result.benchmarks or {}).items() if ret.reindex(net.index).notna().sum() > 20}
    table = result.tables.get("attribution_by_asset") if getattr(result, "tables", None) else None
    contribution = [] if table is None else [{"asset": str(a), "contribution_pct": float(r.contribution_pct), "average_weight": float(r.average_weight), "asset_class": str(r.asset_class)} for a, r in table.iterrows()]
    weights = result.weights.reindex(net.index).fillna(0.0)
    weekly = weights.abs().sum(axis=1).resample("W").last().dropna()
    monthly = weights.resample("MS").mean().tail(60)
    order = weights.abs().mean().sort_values(ascending=False).index[:12]
    return {"risk": performance.risk_metrics(net), "drawdown_episodes": performance.drawdown_episodes(net, 5), "rolling_vol": [None if not np.isfinite(v) else round(float(v), 4) for v in rolling.to_numpy()],
            "calendar_months": performance.calendar_month_profile(net), "monthly": performance.monthly_returns(net), "distribution": performance.return_distribution(net),
            "benchmark_relative": {name: performance.benchmark_relative(net, b) for name, b in bench_returns.items()}, "contribution": contribution,
            "exposure": {"dates": [str(d.date()) for d in weekly.index], "gross": [round(float(v), 3) for v in weekly.to_numpy()], "net": [round(float(v), 3) for v in weights.sum(axis=1).resample("W").last().reindex(weekly.index).to_numpy()]},
            "positions": {"months": [d.strftime("%Y-%m") for d in monthly.index], "assets": [str(a) for a in order], "values": [[round(float(v), 4) for v in monthly[a].to_numpy()] for a in order]}}


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
    out["correlation"] = correlation_view(bundle.returns.loc[out["start"]:out["end"]])
    return out


def correlation_view(returns: pd.DataFrame, min_days: int = 60) -> dict | None:
    """The correlation of the assets' daily returns over the days both have one (``None`` for the pair with fewer than ``min_days`` such days), and each asset's annualised volatility, for the matrix that
    shows how alike the markets a book trades are. ``None`` for a single asset."""
    if returns.shape[1] < 2:
        return None
    corr = returns.corr(min_periods=min_days)
    return {"assets": [str(c) for c in returns.columns], "values": [[None if not np.isfinite(v) else round(float(v), 3) for v in row] for row in corr.to_numpy()],
            "volatility": [None if not np.isfinite(v) else round(float(v), 4) for v in (returns.std(ddof=1) * np.sqrt(252.0)).to_numpy()]}


STRONG = 0.25                                                                 # a signal this far from zero, in units of its own typical size, counts as a view


def signals_board(bundle: MarketBundle, rules: tuple, window: int = 504) -> dict:
    """What each rule of ``rules`` (``(label, model, settings)``) says about each asset of ``bundle`` on the last day: the rule's score, and the score over the average size of its own absolute score in the
    last ``window`` days (its *strength*: 1 is a typical position, the sign is the side, capped at 3). A rule that cannot be computed for this bundle (a cross-sectional rule on one asset, a signal that
    needs more history than there is, or series the bundle lacks) says why instead. ``agreement`` counts, for each asset, the rules that are long, short or without a view beyond ``STRONG``."""
    from ..framework import MODELS, load_library

    load_library()
    assets = list(bundle.prices.columns)
    rows = []
    for label, name, params in rules:
        row = {"id": name, "label": label}
        try:
            score = MODELS.create(name, **params).score(bundle).where(bundle.investable)
        except (KeyError, ValueError) as error:
            rows.append({**row, "error": str(error).strip("'\"")[:240]})
            continue
        live = score.dropna(how="all")
        if live.empty:
            rows.append({**row, "error": "no signal yet: the rule needs more history than these prices have, or more assets"})
            continue
        last = live.iloc[-1]
        typical = score.abs().tail(window).mean().replace(0.0, np.nan)
        strength = (last / typical).clip(-3.0, 3.0)
        rows.append({**row, "date": str(live.index[-1].date()), "score": {a: None if not np.isfinite(last[a]) else float(last[a]) for a in assets},
                     "strength": {a: None if not np.isfinite(strength[a]) else round(float(strength[a]), 3) for a in assets}})
    return {"assets": assets, "signals": rows, "agreement": signal_agreement(rows, assets), "window": int(window), "strong": STRONG, "last_day": str(bundle.index[-1].date())}


def signal_agreement(rows: list[dict], assets: list[str]) -> dict:
    """For each asset, how many of the rules that were computed are long, short or without a view (a strength within ``STRONG`` of zero, or none)."""
    shown = [r for r in rows if "strength" in r]
    return {a: {"long": sum(1 for r in shown if (r["strength"][a] or 0.0) > STRONG), "short": sum(1 for r in shown if (r["strength"][a] or 0.0) < -STRONG),
                "flat": sum(1 for r in shown if r["strength"][a] is None or abs(r["strength"][a]) <= STRONG)} for a in assets}
