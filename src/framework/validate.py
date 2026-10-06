"""The validation battery run on every pipeline result, and the causality check run on every plug-in.

``check_causality`` is the test every registered model and regime detector must pass: replace ALL data after a cutoff with noise
and require the output on or before the cutoff to be unchanged. A model that peeks fails it, whatever its backtest says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from ..backtest.metrics import deflated_sharpe_ratio, performance_summary
from ..signals.alpha_engine import forward_returns
from ..utils.dates import DateWindow
from ..validation.robustness import paired_sharpe_test

ANN = 252.0


def check_causality(model, bundle, cutoff=None, seed: int = 0, tolerance: float = 1e-9) -> dict:
    """Output on dates <= cutoff computed on the real bundle and on a bundle whose later data is noise: they must agree."""
    index = bundle.index
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else index[int(0.7 * len(index))]
    noisy = bundle.perturbed_after(cutoff, seed=seed)
    real = _output(model, bundle).loc[:cutoff]
    fake = _output(model, noisy).loc[:cutoff]
    both = real.notna() & fake.notna()
    differ = (real.where(both) - fake.where(both)).abs().max().max() if both.to_numpy().any() else 0.0
    missing_mismatch = int((real.notna() != fake.notna()).to_numpy().sum())
    ok = bool(np.isfinite(differ) and differ <= tolerance and missing_mismatch == 0)
    return {"ok": ok, "max_abs_difference": float(differ), "availability_mismatches": missing_mismatch, "cutoff": str(cutoff.date()), "compared": int(both.to_numpy().sum())}


def _output(obj, bundle) -> pd.DataFrame:
    if hasattr(obj, "detect"):
        return obj.detect(bundle).probabilities
    return obj.forecast(bundle).mean


def _perf_row(returns: pd.Series) -> dict:
    p = performance_summary(returns.dropna())
    return {k: p.get(k, np.nan) for k in ("sharpe", "cagr", "ann_vol", "max_drawdown")}


def _benchmark_returns(name: str, pipeline, engine, ctx) -> pd.Series:
    from .allocation import BOOKS, book
    key = name.replace("M0_", "").replace("M1_", "").replace("M2_", "").replace("M9_", "").replace("M11_", "").replace("M12_", "")
    if key not in BOOKS:
        raise KeyError(f"unknown benchmark '{name}'")
    return engine.run(book(key, ctx), pipeline.bundle.returns, key, pipeline.bundle.investable, apply_vol_target=False).net_returns


def forecast_quality(forecasts, returns: pd.DataFrame, start) -> dict:
    """Rank IC, hit rate, mean confidence and the reliability of P(up), pooled over assets and dates (matured labels only)."""
    fwd = forward_returns(returns, forecasts.horizon).reindex_like(forecasts.mean)
    mean, conf = forecasts.mean.loc[start:], forecasts.confidence.loc[start:]
    fwd = fwd.loc[start:]
    p_up = pd.DataFrame(norm.cdf((forecasts.mean / forecasts.std.where(forecasts.std > 0)).to_numpy()), index=forecasts.mean.index, columns=forecasts.mean.columns).loc[start:]
    ok = mean.notna() & fwd.notna() & p_up.notna()
    ic = []
    for _, (m, f) in zip(mean.index[::forecasts.horizon], ((mean.loc[d], fwd.loc[d]) for d in mean.index[::forecasts.horizon])):
        pair = pd.concat([m, f], axis=1).dropna()
        if len(pair) >= 5 and pair.iloc[:, 0].nunique() > 1:
            ic.append(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])[0])
    up = (fwd > 0).astype(float).where(ok)
    brier = float(((p_up.where(ok) - up) ** 2).stack().mean()) if ok.to_numpy().any() else np.nan
    base = float(up.stack().mean()) if ok.to_numpy().any() else np.nan
    bins = pd.cut(p_up.where(ok).stack(), np.linspace(0, 1, 11), include_lowest=True)
    rel = pd.DataFrame({"p_up": p_up.where(ok).stack(), "up": up.stack()}).groupby(bins, observed=True).agg(predicted=("p_up", "mean"), realised=("up", "mean"), n=("up", "size"))
    return {"mean_rank_ic": float(np.nanmean(ic)) if ic else np.nan, "ic_observations": len(ic),
            "hit_rate": float(((np.sign(mean) == np.sign(fwd)) & ok & (mean != 0)).sum().sum() / max(((mean != 0) & ok).sum().sum(), 1)),
            "mean_confidence": float(conf.where(ok).stack().mean()) if ok.to_numpy().any() else np.nan,
            "brier": brier, "brier_climatology": base * (1 - base) if np.isfinite(base) else np.nan, "reliability": rel}


def evaluate(out, pipeline, engine, ctx, models, validate: bool, trust) -> None:
    spec, config = pipeline.spec, pipeline.config
    start = out.start
    net = out.net_returns.loc[start:].dropna()
    turnover = out.turnover.loc[start:]
    perf = performance_summary(net, turnover=turnover, costs=out.costs.loc[start:])
    gross = out.gross_returns.loc[start:].dropna()
    out.metrics = {**{k: perf.get(k, np.nan) for k in ("sharpe", "cagr", "ann_vol", "max_drawdown", "sortino", "calmar", "ann_turnover")},
                   "gross_sharpe": float(ANN ** 0.5 * gross.mean() / gross.std(ddof=1)) if gross.std(ddof=1) > 0 else np.nan,
                   "ann_cost_bps": float(1e4 * out.costs.loc[start:].mean() * ANN), "n_days": int(len(net)), "start": str(net.index[0].date()), "end": str(net.index[-1].date()),
                   "mean_ann_turnover": float(turnover.mean() * ANN)}
    # splits
    samples = {n: DateWindow.from_config(n, s or {}) for n, s in (config.get("backtest.samples", {}) or {}).items()}
    rows = {n: {**_perf_row(w.apply(net)), "days": int(len(w.apply(net).dropna()))} for n, w in samples.items() if len(w.apply(net).dropna()) > 20}
    out.tables["by_sample"] = pd.DataFrame(rows).T
    # regimes
    if out.regimes is not None:
        labels = out.regimes.hard_labels().reindex(net.index)
        table = {}
        for name, r in net.groupby(labels):
            if len(r) > 5:
                table[name] = {"share_of_days": len(r) / len(net), "ann_return": float(r.mean() * ANN), "ann_vol": float(r.std(ddof=1) * np.sqrt(ANN)),
                               "sharpe": float(ANN ** 0.5 * r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else np.nan}
        out.tables["by_regime"] = pd.DataFrame(table).T
    # attribution: additive daily contributions of each asset's held weight, then costs
    held = out.weights.shift(1).reindex(net.index)
    contribution = (held * pipeline.bundle.returns.reindex(net.index)).sum()
    avg_weight = held.mean()
    attribution = pd.DataFrame({"contribution_pct": 100 * contribution, "average_weight": avg_weight,
                                "asset_class": [pipeline.bundle.asset_class.get(a, "unknown") for a in contribution.index]})
    out.tables["attribution_by_asset"] = attribution.sort_values("contribution_pct", ascending=False)
    by_class = attribution.groupby("asset_class")["contribution_pct"].sum()
    comp = by_class.to_dict()
    comp["costs"] = -float(100 * out.costs.loc[net.index].sum())
    comp["other (compounding, lag alignment)"] = float(100 * net.sum()) - sum(comp.values())
    out.tables["attribution"] = pd.Series(comp, name="return_points").to_frame()
    # forecast quality
    if out.forecasts is not None:
        fq = forecast_quality(out.forecasts, pipeline.bundle.returns, start)
        out.tables["reliability"] = fq.pop("reliability")
        out.metrics.update({f"forecast_{k}": v for k, v in fq.items()})
    # benchmarks
    bench_names = spec.evaluation.get("benchmarks") or config.get("framework.tearsheet.benchmarks", ["equal_weight", "risk_parity"])
    boot = (config.get("framework.tearsheet.bootstrap", {}) or {})
    rows = []
    for name in bench_names:
        try:
            b = _benchmark_returns(name, pipeline, engine, ctx).loc[start:].dropna()
        except Exception as error:
            rows.append({"benchmark": name, "error": str(error)})
            continue
        common = net.index.intersection(b.index)
        out.benchmarks[name] = b.loc[common]
        if len(common) > 250:
            t = paired_sharpe_test(net.loc[common], b.loc[common], n_samples=int(boot.get("n_samples", 2000)), block_length=int(boot.get("block_length", 21)),
                                   seed=int(boot.get("seed", 7)))
            rows.append({"benchmark": name, "sharpe_strategy": t["sharpe_a"], "sharpe_benchmark": t["sharpe_b"], "difference": t["difference"], "p_value": t["p_value"]})
    out.tables["benchmarks"] = pd.DataFrame(rows)
    # validation
    n_trials = int(spec.evaluation.get("n_trials", 1))
    out.validation = {"deflated_sharpe_probability": deflated_sharpe_ratio(float(out.metrics["sharpe"]), max(n_trials, 1), len(net)), "n_trials": n_trials}
    if validate and spec.evaluation.get("causality", True) and models:
        out.validation["causality"] = {m.name: check_causality(m, pipeline.bundle) for m in models}
    if validate and spec.evaluation.get("causality", True) and spec.regime:
        from .registry import DETECTORS
        det = DETECTORS.create(spec.regime["detector"], **(spec.regime.get("params") or {}))
        out.validation.setdefault("causality", {})[f"regime:{spec.regime['detector']}"] = check_causality(det, pipeline.bundle)
    if trust is not None:
        out.tables["trust_weights"] = trust.resample("ME").last()
