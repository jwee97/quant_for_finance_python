"""Two questions, kept apart: which model forecasts best, and which allocation policy turns a given forecast into the best risk-adjusted outcome.

``prediction_quality`` answers the first with what the forecasts did against what followed (time-series and pooled statistics: a cross-sectional rank correlation is estimated only when the universe is large
enough to have a cross-section). ``compare`` answers the second: every allocator gets the SAME forecasts, dates, limits, costs and execution (the authoritative backtest engine), and the table reports net return,
Sharpe, drawdown, turnover, concentration and the stability of the weights.

Selection discipline. The allocators are ranked, and one is selected, on the segment the caller names (validation by default) and nowhere else. The protected segments (test, forward) are not computed unless
``evaluate`` asks for them, and then every allocator is reported on them with ``used_for_selection: False``: the selected one is not chosen again there. The comparison states how many allocators were tried,
Holm-adjusts the p-value of each against the equal-weight book, and gives the deflated Sharpe probability of the best one for that number of trials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..alphalab.factors import holm
from ..stats import inference as si
from . import inference as inf
from . import portfolio_alloc as pa
from .backtest import equal_weight_benchmark, label_days, run_backtest, trim_to_signal

SEGMENTS = ("train", "validation", "test", "forward")


def _concentration(w: pd.DataFrame) -> dict:
    a = w.fillna(0.0).abs()
    gross = a.sum(axis=1)
    live = gross > 1e-12
    if not live.any():
        return {"effective_n": None, "hhi": None, "max_weight": None}
    share = a[live].div(gross[live], axis=0)
    hhi = (share ** 2).sum(axis=1)
    return {"effective_n": float((1.0 / hhi).mean()), "hhi": float(hhi.mean()), "max_weight": float(share.max(axis=1).mean())}


def _stability(w: pd.DataFrame) -> dict:
    """How much the book moves: the mean one-way change of the target weights between consecutive decision dates, and the share of dates at which it moved by more than one percentage point."""
    a = w.fillna(0.0)
    moved = a.diff().abs().sum(axis=1) / 2.0
    changes = moved[moved > 1e-9]
    return {"mean_change_per_decision": float(changes.mean()) if len(changes) else 0.0, "decisions_with_a_change": int(len(changes)), "share_moving_more_than_1pt": float((changes > 0.01).mean()) if len(changes) else 0.0}


def _segment_metrics(net: pd.Series, turnover: pd.Series, costs: pd.Series, weights: pd.DataFrame) -> dict:
    n = len(net)
    if n < 20:
        return {"n_days": int(n), "note": "fewer than 20 days: no statistics"}
    eq = (1.0 + net).cumprod()
    dd = eq / eq.cummax() - 1.0
    sd = float(net.std(ddof=1))
    return {"n_days": int(n), "net_return": float(eq.iloc[-1] - 1.0), "ann_return": float(net.mean() * 252.0), "ann_vol": sd * float(np.sqrt(252.0)), "sharpe": float(net.mean() / sd * np.sqrt(252.0)) if sd > 0 else None,
            "max_drawdown": float(dd.min()), "turnover_per_year": float(turnover.mean() * 252.0), "cost_bps_per_year": float(costs.mean() * 252.0 * 1e4), **_concentration(weights), **_stability(weights)}


def prediction_quality(score: pd.DataFrame, returns: pd.DataFrame, horizon: int, lag: int, labels: pd.Series | None = None, segments=("validation",)) -> dict:
    """What the forecasts did against what followed, on the segments asked for: the pooled correlation of forecast and outcome, the share of right-signed forecasts, the error of the forecast and of the zero forecast
    (how much a forecast beats 'no change'), and the cross-sectional rank correlation only when at least five assets are present on a date."""
    real = pa.realised_window_returns(returns.reindex(index=score.index, columns=score.columns), horizon, lag)
    out: dict = {}
    for seg in segments:
        mask = (labels.reindex(score.index) == seg) if labels is not None else pd.Series(True, index=score.index)
        s, r = score[mask], real[mask]
        pair = pd.concat([s.stack(future_stack=True), r.stack(future_stack=True)], axis=1, keys=["f", "y"]).dropna()
        if len(pair) < 30:
            out[seg] = {"n": int(len(pair)), "note": "fewer than 30 forecasts with a known outcome"}
            continue
        f, y = pair["f"].to_numpy(), pair["y"].to_numpy()
        rec = {"n": int(len(pair)), "pooled_correlation": float(np.corrcoef(f, y)[0, 1]) if f.std() > 0 and y.std() > 0 else None, "hit_rate": float((np.sign(f) == np.sign(y)).mean()),
               "rmse": float(np.sqrt(np.mean((y - f) ** 2))), "rmse_zero_forecast": float(np.sqrt(np.mean(y ** 2))), "mae": float(np.mean(np.abs(y - f)))}
        rec["skill_vs_zero"] = 1.0 - rec["rmse"] ** 2 / rec["rmse_zero_forecast"] ** 2
        counts = s.notna().sum(axis=1)
        if int(counts.max()) >= 5:
            ics = [s.loc[d].corr(r.loc[d], method="spearman") for d in s.index[counts >= 5] if r.loc[d].notna().sum() >= 5]
            rec["cross_sectional_rank_ic"] = float(np.nanmean(ics)) if ics else None
        else:
            rec["cross_sectional_rank_ic"] = None
            rec["cross_sectional_note"] = f"not estimated: a rank correlation needs at least five assets on a date and this universe has {int(counts.max())}"
        out[seg] = rec
    return out


def compare(score: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, sig: dict, bt_cfg: dict, lag: int, labels: pd.Series, groups: dict | None = None, *, allocators=pa.ALLOCATORS,
            select_on: str = "validation", evaluate: tuple = (), select_by: str = "sharpe", prices: pd.DataFrame | None = None, probability: pd.DataFrame | None = None,
            predictive_sd: pd.DataFrame | None = None, legacy_rule: dict | None = None, seed: int = 0) -> dict:
    """Run ``allocators`` on one set of forecasts. ``sig`` is the experiment's ``signal`` section with its ``allocation`` block (limits, risk model, calibration, costs); every allocator replaces only the
    ``allocator`` key. ``legacy_rule`` (a ``signal`` section) adds the experiment's existing position rule as a further row. ``evaluate`` lists protected segments to report in addition to ``select_on``."""
    if select_on in evaluate:
        raise pa.AllocationError("the segment used to select an allocator cannot also be listed as an evaluation segment")
    protected = set(evaluate) - {select_on}
    need = int(sig["window_days"])
    rows: dict[str, dict] = {}
    nets: dict[str, pd.Series] = {}
    logs: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    first_signal = score.dropna(how="all").index[0]
    seg_all = label_days(returns.index, labels)
    for a in allocators:
        try:
            w, log = pa.run(score, returns, investable, sig, groups, prices=prices, probability=probability, predictive_sd=predictive_sd, allocator=a)
        except pa.AllocationError as exc:
            skipped[a] = str(exc)
            continue
        res = run_backtest(w, returns, investable, bt_cfg, lag, a)
        parts = trim_to_signal(res, first_signal, lag)
        nets[a], logs[a] = parts["net"], log
        rows[a] = {"label": pa.DOCS[a]["label"], "kind": "allocator", "parts": parts, "weights": w}
    if legacy_rule is not None:
        from .signals import build_weights

        w = build_weights(score, legacy_rule, None, None, None, groups, investable, None, returns)
        res = run_backtest(w, returns, investable, bt_cfg, lag, "existing_rule")
        parts = trim_to_signal(res, first_signal, lag)
        nets["existing_rule"] = parts["net"]
        rows["existing_rule"] = {"label": f"Existing rule ({legacy_rule['method']})", "kind": "existing_rule", "parts": parts, "weights": w}
    if not rows:
        raise pa.AllocationError("no allocator could run: " + "; ".join(f"{k}: {v}" for k, v in skipped.items()))
    bench = equal_weight_benchmark(returns, investable)
    table: dict[str, dict] = {}
    for k, r in rows.items():
        seg = seg_all.reindex(r["parts"]["net"].index)
        table[k] = {"label": r["label"], "kind": r["kind"], "segments": {}}
        for s in [select_on, *sorted(protected)]:
            m = seg == s
            if not m.any():
                table[k]["segments"][s] = {"n_days": 0, "note": "the segment has no days"}
                continue
            wk = r["weights"].reindex(r["parts"]["net"].index)[m.to_numpy()]
            table[k]["segments"][s] = {**_segment_metrics(r["parts"]["net"][m], r["parts"]["turnover"][m], r["parts"]["costs"][m], wk),
                                       "used_for_selection": s == select_on}
    seg_b = seg_all.reindex(bench.index)
    bench_rows = {s: _segment_metrics(bench[seg_b == s], pd.Series(0.0, index=bench[seg_b == s].index), pd.Series(0.0, index=bench[seg_b == s].index), pd.DataFrame(1.0, index=bench[seg_b == s].index, columns=returns.columns))
                  for s in [select_on, *sorted(protected)]}
    # selection and multiple-testing control on the selection segment only
    sel = {k: nets[k][seg_all.reindex(nets[k].index) == select_on] for k in nets}
    ew_key = "equal_weight" if "equal_weight" in sel else None
    pvals: dict[str, float] = {}
    paired: dict[str, dict] = {}
    if ew_key is not None:
        for k, x in sel.items():
            if k == ew_key or len(x) < 60:
                continue
            res = inf.paired_difference(x, sel[ew_key], statistic="mean", label=f"{k} minus equal weight, {select_on}", n_boot=500, seed=seed, horizon=need)
            hac = (res.extras or {}).get("hac") or {}
            pv = hac.get("p_value") if hac.get("p_value") is not None else res.p_value          # the autocorrelation-robust test of the mean when it applied, else the bootstrap
            pv = None if pv is None else float(pv)
            paired[k] = {"mean_daily_difference": res.value, "interval": None if res.interval is None else [float(res.interval[0]), float(res.interval[1])], "p_value": pv, "eligibility": res.eligibility}
            if pv is not None:
                pvals[k] = pv
    adj = holm(pvals) if pvals else {}
    for k in paired:
        paired[k]["p_value_holm"] = adj.get(k)
    ranked = sorted((k for k in sel if len(sel[k]) >= 60), key=lambda k: -(table[k]["segments"][select_on].get(select_by) or -1e9))
    best = ranked[0] if ranked else None
    dsr = None
    if best is not None:
        trial_sharpes = [float(sel[k].mean() / sel[k].std(ddof=1)) for k in ranked if sel[k].std(ddof=1) > 0]         # per-period, as the deflation expects
        try:
            dsr = si.deflated_sharpe_ratio(sel[best], trial_sharpes=trial_sharpes, n_trials=len(trial_sharpes))
        except Exception as exc:                                                                       # noqa: BLE001
            dsr = {"error": str(exc)}
    notes = [f"{len(rows)} allocation policies were compared on identical forecasts, dates, limits, costs and execution; the winner is chosen on the {select_on} segment only.",
             "A difference between allocators over a few years is usually inside the noise: read the interval and the Holm-adjusted p-value, not only the ranking."]
    if protected:
        notes.append(f"The segment(s) {', '.join(sorted(protected))} are reported for every allocator and were not used to select: they are a look at protected outcomes and count as one.")
    shown = [select_on, *sorted(protected)]
    curves = {"segments": shown, "dates": {}, "equity": {}, "weights": {}}
    for sname in shown:
        days = seg_all.index[(seg_all == sname).to_numpy()]
        curves["dates"][sname] = [str(d.date()) for d in days[::max(1, len(days) // 300)]]
        curves["equity"][sname] = {}
        curves["weights"][sname] = {}
        for k in rows:
            net = nets[k].reindex(days).dropna()
            eqc = (1.0 + net).cumprod()
            curves["equity"][sname][k] = {"dates": [str(d.date()) for d in eqc.index[::max(1, len(eqc) // 300)]], "values": [float(v) for v in eqc.to_numpy()[::max(1, len(eqc) // 300)]]}
            wk = rows[k]["weights"].reindex(days).dropna(how="all")
            step = max(1, len(wk) // 120)
            curves["weights"][sname][k] = {"dates": [str(d.date()) for d in wk.index[::step]], "assets": list(wk.columns), "values": [[float(v) for v in r] for r in wk.to_numpy()[::step]]}
        b = bench[bench.index.isin(days)]
        be = (1.0 + b).cumprod()
        curves["equity"][sname]["_equal_weight_daily_rebalanced"] = {"dates": [str(d.date()) for d in be.index[::max(1, len(be) // 300)]], "values": [float(v) for v in be.to_numpy()[::max(1, len(be) // 300)]]}
    return {"version": pa.VERSION, "select_on": select_on, "select_by": select_by, "n_allocators": len(rows), "allocators": list(rows), "skipped": skipped, "table": table, "equal_weight_benchmark": bench_rows,
            "selected": best, "ranking": ranked, "paired_vs_equal_weight": paired, "deflated_sharpe_of_selected": dsr, "evaluated_segments": sorted(protected), "notes": notes, "curves": curves,
            "counts": {k: logs[k]["counts"] for k in logs}, "_returns": {k: nets[k] for k in nets}, "_weights": {k: rows[k]["weights"] for k in rows}, "_logs": logs}


def public(result: dict) -> dict:
    """The comparison without the in-memory series, weights and logs (what the API and a saved record carry)."""
    return {k: v for k, v in result.items() if not k.startswith("_")}
