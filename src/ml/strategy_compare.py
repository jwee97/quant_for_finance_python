"""Models and baselines compared like for like (phase 5).

Several models are trained on the **same dataset and splits**, traded by the **same position rule, costs and lag**, and set beside baselines run through the same engine over the **same days**: equal weight
(daily rebalanced and held), a 12-1 momentum book, a short-term reversal book, a trend (50-day above 200-day average) book, equal weight scaled to a volatility target, and the average of random-signal books.
Predictive quality and trading quality are reported apart, because a model can rank well and still not trade profitably after costs, and the reverse.

Predictive: rank information coefficient (mean, information ratio, share of positive days), the day-by-day stability of the ordering, the error of the score as a return forecast against the error of a forecast
of the average (skill over doing nothing), and the top-minus-bottom quantile spread. Trading: CAGR, Sharpe, Sortino, maximum drawdown, turnover, cost drag, beta to equal weight, the active return over a
beta-matched benchmark, average exposure. Capacity is not estimated (it needs volume and a size of order); the table says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import benchmarks as bm
from . import controls as ct
from . import experiments as ex
from .backtest import run_backtest, trim_to_signal
from .config import experiment_config, merge
from .training import walk_forward_train

VERSION = "strategy-compare-v1"
MAX_MODELS = 6
ANN = 252.0


def _summary(net: pd.Series, bench: pd.Series | None = None) -> dict:
    x = net.dropna()
    if len(x) < 30:
        return {"n": int(len(x))}
    eq = (1 + x).cumprod()
    years = len(x) / ANN
    dd = eq / eq.cummax() - 1
    sd = x.std(ddof=1)
    down = x[x < 0].std(ddof=1)
    out = {"n": int(len(x)), "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else float("nan"), "ann_vol": float(sd * np.sqrt(ANN)), "sharpe": float(x.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan"),
           "sortino": float(x.mean() / down * np.sqrt(ANN)) if down and down > 0 else float("nan"), "max_drawdown": float(dd.min())}
    if bench is not None:
        b = bench.reindex(x.index)
        d = pd.concat([x, b], axis=1, keys=["a", "b"]).dropna()
        if len(d) > 30 and d["b"].var() > 0:
            out["beta"] = float(np.cov(d["a"], d["b"])[0, 1] / d["b"].var())
            act = d["a"] - d["b"]
            out["information_ratio"] = float(act.mean() / act.std(ddof=1) * np.sqrt(ANN)) if act.std(ddof=1) > 0 else float("nan")
            matched, _ = bm.beta_matched(x, bench)
            m_act = (x - matched).dropna()
            out["active_over_beta_matched"] = float(m_act.mean() * ANN) if len(m_act) else float("nan")
    return out


def _predictive(exp, ds) -> dict:
    """Prediction quality of one trained model, from its recorded report and its out-of-sample scores."""
    r = exp.report["ic"]["all"]
    q = exp.report["ic"].get("quantiles") or {}
    scores = exp.scores
    y = ds.Y[ds.target].unstack().reindex(index=scores.index, columns=scores.columns)
    both = pd.concat([scores.stack(), y.stack()], axis=1, keys=["s", "y"]).dropna()
    out = {"rank_ic": r.get("mean"), "ic_ir": r.get("ir"), "ic_hit_rate": r.get("hit_rate"), "ic_days": r.get("n")}
    if len(both) > 100:
        z = both["s"]
        slope = float(np.cov(z, both["y"])[0, 1] / z.var()) if z.var() > 0 else float("nan")
        out["calibration_slope"] = slope                                         # 1 means the score is, on average, the return that followed it (meaningful for a return forecast)
        out["mae_score"], out["mae_zero"] = float((both["s"] - both["y"]).abs().mean()), float(both["y"].abs().mean())
        out["mse_score"], out["mse_zero"] = float(((both["s"] - both["y"]) ** 2).mean()), float((both["y"] ** 2).mean())
    if isinstance(q, dict) and q.get("spread") is not None:
        out["top_minus_bottom"] = q.get("spread")
    return out


def baselines(ds, cfg: dict, first: pd.Timestamp, lag: int, bench: pd.Series, *, draws: int = 10, seed: int = 0) -> dict[str, pd.Series]:
    """Causal baseline books through the same engine, costs and lag, starting from the same first day."""
    R, px = ds.returns, ds.prices
    bt = cfg["backtest"]
    k = R.shape[1]
    out = {}

    def run(w: pd.DataFrame, name: str, rebalance: str, target_vol=None) -> pd.Series:
        b = {**bt, "rebalance": rebalance, "target_vol": target_vol}
        res = run_backtest(w, R, ds.investable, b, lag, name)
        return trim_to_signal(res, first, lag)["net"]

    ew = pd.DataFrame(1.0 / k, index=R.index, columns=R.columns)
    out["equal weight, rebalanced monthly"] = run(ew, "ew_monthly", "monthly")
    out["equal weight, held (no rebalance)"] = run(ew, "ew_hold", "on_change")                 # constant targets: the event-driven book trades once, then only drifts
    mom = (px.shift(21) / px.shift(252) - 1.0)
    reversal = -(px / px.shift(5) - 1.0)
    for name, score in (("12-1 momentum, top 30%", mom), ("5-day reversal, top 30%", reversal)):
        rk = score.rank(axis=1, pct=True)
        w = (rk >= 0.7).astype(float)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        out[name] = run(w, name, "monthly")
    trend = (px.rolling(50).mean() > px.rolling(200).mean()).astype(float)
    wt = trend.div(trend.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    out["trend: 50-day above 200-day average"] = run(wt, "trend", "weekly")
    out["equal weight, 10% volatility target"] = run(ew, "ew_vt", "monthly", target_vol=0.10)
    rng = np.random.default_rng(seed)
    rho = 0.97
    nets = []
    for _ in range(draws):
        sc = ct.random_scores(px.notna().astype(float).where(px.notna()), rho, rng)
        rk = sc.rank(axis=1, pct=True)
        w = (rk >= 0.7).astype(float)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        nets.append(run(w, "rand", "monthly"))
    out[f"random-signal books (mean of {draws})"] = pd.concat(nets, axis=1).mean(axis=1)
    return out


def compare(models: list[dict], overrides: dict | None = None, *, bundle=None, config=None, builder=None, dataset=None, info=None, progress=None, draws: int = 10) -> dict:
    """Train each model in ``models`` (``{"kind": ..., "params": {...}}``) on the same dataset and splits, trade them identically, and tabulate them with the baselines over their common days."""
    if not 1 <= len(models) <= MAX_MODELS:
        raise ValueError(f"compare between 1 and {MAX_MODELS} models")
    progress = progress or (lambda m: None)
    over = merge(overrides or {}, {"backtest": {"target_vol": None}})
    cfg0 = experiment_config(merge(over, {"model": models[0]}))
    ds, info = ex.prepare(cfg0, bundle, config, builder, dataset, info, progress)
    runs = {}
    for i, m in enumerate(models):
        progress(f"model {i + 1} of {len(models)}: {m['kind']}")
        cfg = experiment_config(merge(over, {"model": m, "name": f"compare_{m['kind']}"}))
        train = walk_forward_train(ds, cfg["model"], cfg["split"], cfg["seed"])
        e = ex.evaluate_run(cfg, ds, train, info, None, None, False, None, False, False)
        label = m["kind"] if m["kind"] not in runs else f"{m['kind']} #{i + 1}"
        runs[label] = (e, cfg)
    firsts = [e.first_signal for e, _ in runs.values()]
    first = max(firsts)
    lag = ds.lag
    cfgb = next(iter(runs.values()))[1]
    bench = ex.equal_weight_benchmark(ds.returns, ds.investable)
    base = baselines(ds, cfgb, first, lag, bench, draws=draws)
    nets = {k: e.net for k, (e, _) in runs.items()} | base
    common = None
    for s in nets.values():
        idx = s.dropna().index
        common = idx if common is None else common.intersection(idx)
    common = common[common >= first]
    if len(common) < 100:
        raise ValueError("the strategies share fewer than 100 days: the training window leaves too little out-of-sample history")
    b = bench.reindex(common)
    rows = []
    for k, (e, cfg) in runs.items():
        w = e.weights.loc[common[0]:common[-1]]
        rows.append({"name": k, "kind": "model", **_summary(e.net.reindex(common), b), "predictive": _predictive(e, ds), "ann_turnover": float(e.turnover.reindex(common).mean() * ANN),
                     "cost_drag": float(e.costs.reindex(common).mean() * ANN), "gross_exposure": float(w.abs().sum(axis=1).mean()), "net_exposure": float(w.sum(axis=1).mean())})
    for k, s in base.items():
        rows.append({"name": k, "kind": "control" if k.startswith("random") else "baseline", **_summary(s.reindex(common), b), "predictive": None})
    eq = {k: [float(x) for x in (1 + v.reindex(common).fillna(0)).cumprod().iloc[::5]] for k, v in nets.items()}
    corr = pd.DataFrame({k: v.reindex(common) for k, v in nets.items()}).corr()
    return {"version": VERSION, "rows": rows, "period": {"from": str(common[0].date()), "to": str(common[-1].date()), "days": int(len(common))},
            "equity": {"dates": [str(d.date()) for d in common[::5]], **eq}, "correlation": {"names": list(corr.columns), "values": [[None if not np.isfinite(x) else float(x) for x in r] for r in corr.to_numpy()]},
            "design": {"models": [m["kind"] for m in models], "position_rule": cfgb["signal"]["method"], "cost_bps": float(cfgb["backtest"]["cost_bps"]) + float(cfgb["backtest"]["slippage_bps"]),
                       "lag": lag, "target": ds.target, "min_train_days": int(cfgb["split"]["min_train_days"])},
            "notes": ["Every strategy is traded through the same engine, costs and lag over the same days; the models share the dataset and the splits.",
                      "Predictive quality (rank IC, calibration, error against a forecast of nothing) and trading quality (everything else) are different questions and are reported apart.",
                      "Beta is to the daily-rebalanced equal-weight universe; 'active over beta-matched' is the strategy's annual return less the market scaled by its own causal beta.",
                      "Capacity is not estimated here: it needs volume and an order size (see the Alpha lab's capacity study).",
                      "Several models on one history is a multiple comparison: the best row is not corrected for being the best of several."]}
