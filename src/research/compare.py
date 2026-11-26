"""The Research Comparison: put saved results side by side without letting the layout say more than the evidence does.

What it does that a table of the artifacts' own metrics cannot:

* every result is measured again **over the window they all share** with one piece of arithmetic (the one of the Backtest tab's tiles and ``risk.performance``), so a ratio is never a ratio over
  different years;
* it names every **difference in assumptions** that would make a ranking misleading (different evidence, different data kind, different assets, dataset versions, targets or costs) and says whether
  the difference stops a ranking (``blocks_ranking``) or only deserves a warning. When something blocks, no ranking is produced and the reason is returned instead;
* it returns the series the page draws (rebased equity, drawdown, rolling Sharpe and volatility, the distribution of daily returns, the correlation of the returns) on that one window, so the charts
  share an x-axis;
* it writes a reproducible report: every result with its references (job or experiment, configuration hash, dataset, code version) and the assumptions and caveats that apply.

Nothing is re-run and nothing is estimated beyond what the saved series hold.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from ..risk import performance
from .extract import returns_from_equity

ANN = 252.0
MIN_DAYS = 30
RANK_BY = "sharpe"
EVIDENCE_TEXT = {"in_sample": "in-sample", "out_of_sample": "out-of-sample", "walk_forward": "walk-forward (out-of-sample)", "historical_simulation": "historical simulation of the whole sample", "simulated": "simulated data",
                 "retrospective": "retrospective (uses information not known at the time)", "paper": "paper trading", "live": "live"}


def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def _returns(art: dict) -> pd.Series | None:
    s = art.get("series") or {}
    if not s.get("dates") or not s.get("equity"):
        return None
    return returns_from_equity(s["dates"], [np.nan if v is None else v for v in s["equity"]]).dropna()


def _extra(art: dict, name: str, index: pd.DatetimeIndex) -> pd.Series | None:
    vals = ((art.get("series") or {}).get("extra") or {}).get(name)
    if vals is None:
        return None
    dates = pd.to_datetime(art["series"]["dates"])
    return pd.Series(vals, index=dates, dtype=object).reindex(index)


def measure(r: pd.Series, rf: float = 0.0) -> dict:
    """The headline numbers of a daily-return series (the arithmetic of the Backtest tab, plus a few of ``risk.performance``). ``rf`` is an annual rate taken off the mean before the ratios."""
    r = r.dropna().astype(float)
    n = len(r)
    if n < MIN_DAYS:
        return {"n_days": n}
    excess = r - rf / ANN
    sd = float(r.std(ddof=1))
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax().clip(lower=1.0) - 1.0
    cagr = float(eq.iloc[-1] ** (ANN / n) - 1.0)
    risk = performance.risk_metrics(r)
    return {"n_days": n, "cagr": cagr, "ann_vol": sd * math.sqrt(ANN), "sharpe": _num(math.sqrt(ANN) * excess.mean() / sd) if sd > 0 else None, "sortino": _num(risk.get("sortino")),
            "max_drawdown": float(dd.min()), "calmar": _num(risk.get("calmar")), "skew": _num(risk.get("skew")), "excess_kurtosis": _num(risk.get("excess_kurtosis")), "var_95": _num(risk.get("var_95")),
            "es_95": _num(risk.get("es_95")), "positive_days": _num(risk.get("positive_days")), "total_return": float(eq.iloc[-1] - 1.0)}


def _label(a: dict) -> str:
    return f"{a['title']} ({a['kind'].replace('_', ' ')})"


def issues_for(arts: list[dict], window: tuple[str, str] | None) -> list[dict]:
    """The differences between the results that matter for reading them side by side."""
    out: list[dict] = []

    def add(code, severity, message, ids=None):
        out.append({"code": code, "severity": severity, "message": message, "artifacts": ids or [a["id"] for a in arts]})

    if len(arts) < 2:
        return out
    kinds = {a["evidence"] for a in arts}
    if len(kinds) > 1:
        add("mixed_evidence", "blocks_ranking", "The results are different kinds of evidence (" + "; ".join(sorted(EVIDENCE_TEXT.get(k, k) for k in kinds)) + "). Ranking an out-of-sample run against an in-sample simulation rewards the one that saw the answer.")
    dk = {a["data_kind"] for a in arts}
    if len(dk) > 1:
        add("mixed_data", "blocks_ranking", "The results rest on different kinds of data (" + ", ".join(sorted(dk)) + "); a result on simulated data says nothing about the market.")
    if "simulated" in dk or "simulated" in kinds:
        add("simulated", "warning", "At least one result is simulated: it shows that the machinery works, not that a market pays.")
    asset_sets = {tuple(sorted(a["context"].get("assets") or [])) for a in arts if a["context"].get("assets")}
    if len(asset_sets) > 1:
        add("different_assets", "blocks_ranking", "The results trade different assets, so a difference in their performance may be a difference in what they could buy.")
    versions = {a["context"].get("dataset_version") for a in arts if a["context"].get("dataset_version")}
    if len(versions) > 1 and len(asset_sets) <= 1:
        add("different_dataset", "blocks_ranking", "The results rest on different versions of the data (" + ", ".join(sorted(versions)) + "): the same name does not mean the same prices.")
    ml = [a for a in arts if a["kind"] == "ml_experiment"]
    targets = {(a["config"].get("target"), a["config"].get("horizon")) for a in ml}
    if len(targets) > 1:
        add("different_target", "blocks_ranking", "The machine-learning experiments predict different targets or horizons; their scores and returns answer different questions.", [a["id"] for a in ml])
    costs = {str(a["context"].get("costs")) for a in arts if a["context"].get("costs")}
    if len(costs) > 1:
        add("different_costs", "warning", "The results were made with different cost assumptions; net returns are not like for like.")
    freqs = {a["context"].get("frequency") for a in arts if a["context"].get("frequency")}
    if len(freqs) > 1:
        add("different_frequency", "blocks_ranking", "The results use different data frequencies.")
    withdata = [a for a in arts if a.get("series")]
    if window:
        for a in withdata:
            first, last = a["series"]["dates"][0], a["series"]["dates"][-1]
            if first < window[0] or last > window[1]:
                add("window_cut", "note", f"{_label(a)} covers {first} to {last}; it is measured over {window[0]} to {window[1]}, the days all results share.", [a["id"]])
    studies = [a for a in arts if a["kind"] in ("factor_study", "forecast")]
    if len(studies) >= 2:                                                                       # studies without a return stream are comparable only if they asked the same question of the same sample
        ids = [a["id"] for a in studies]
        if len({a["kind"] for a in studies}) > 1:
            add("different_studies", "blocks_ranking", "The results are different kinds of study (a factor study and a forecast evaluation); their scores answer different questions.", ids)
        hz = {a["config"].get("horizon") for a in studies if a["config"].get("horizon")}
        if len(hz) > 1:
            add("different_horizon", "blocks_ranking", "The studies look at different horizons (" + ", ".join(f"{int(h)} days" for h in sorted(hz)) + "): an IC or a forecast score over 5 days is not comparable with one over 21.", ids)
        spans = {(a["context"].get("start"), a["context"].get("end")) for a in studies if a["context"].get("start")}
        if len(spans) > 1 and len(asset_sets) <= 1:
            add("different_sample_window", "blocks_ranking", "The studies cover different dates (" + "; ".join(f"{s} to {e}" for s, e in sorted(spans)) + "): they did not face the same market.", ids)
    fund = [a for a in arts if (a["config"].get("fundamental") or {})]
    if fund:
        statuses = {a["config"]["fundamental"].get("pit_status") for a in fund}
        if any(s == "ready_with_assumptions" for s in statuses) or len(statuses) > 1:
            add("pit_assumptions", "warning", "Some results use fundamental figures whose availability dates were partly assumed (" + ", ".join(sorted(str(s) for s in statuses)) + "): when a filing really became public may differ from the assumed lag.", [a["id"] for a in fund])
        if len(fund) < len([a for a in arts if a["kind"] == "factor_study"]):
            add("fundamentals_mixed", "warning", "Some factor studies use fundamental factors and some do not; the first have a smaller, slower-moving set of figures to work with.", [a["id"] for a in arts if a["kind"] == "factor_study"])
    trading = [a for a in arts if a.get("series") and a["kind"] in ("portfolio_lab", "backtest", "ml_experiment", "options", "futures", "crypto")]
    if len(trading) >= 2:
        bps = {a["context"].get("costs", {}).get("cost_bps") for a in trading if (a["context"].get("costs") or {}).get("cost_bps") is not None}
        if len(bps) > 1:
            add("different_cost_bps", "blocks_ranking", "The results charge different trading costs (" + ", ".join(f"{b:g} bp" for b in sorted(bps)) + "): net returns are not like for like. Compare them with the same cost, or compare gross figures.", [a["id"] for a in trading])
        bench = {a.get("benchmark") for a in trading if a.get("benchmark")}
        if len(bench) > 1:
            add("different_benchmark", "warning", "The results were measured against different benchmarks (" + ", ".join(sorted(map(str, bench))) + "): their benchmark-relative numbers are not comparable; the comparison below does not use them.", [a["id"] for a in trading])
        rfs = {a["context"].get("risk_free") for a in trading if a["context"].get("risk_free") is not None}
        if len(rfs) > 1:
            add("different_risk_free", "warning", "The results assumed different risk-free rates; the comparison uses the one given above for all of them.", [a["id"] for a in trading])
    pl = [a for a in arts if a["kind"] == "portfolio_lab"]
    if len(pl) >= 2:
        keys = ("lookback", "step", "cov_method", "max_weight", "risk_aversion", "mu_shrink")
        diff = [k for k in keys if len({str((a["config"].get("study") or {}).get(k)) for a in pl}) > 1]
        if diff:
            add("different_construction", "blocks_ranking", "The portfolios were constructed with different assumptions (" + ", ".join(diff) + "): a difference in performance may be a difference in how they were built, not in what they were built from.", [a["id"] for a in pl])
        mus = {a["config"].get("mu_source") or "historical mean" for a in pl}
        if len(mus) > 1:
            add("different_expected_returns", "note", "The portfolios use different sources of expected returns (" + ", ".join(sorted(mus)) + "): that is the difference being compared; everything else about their construction is the same.", [a["id"] for a in pl])
        meths = {a["config"].get("method") for a in pl}
        if len(meths) > 1 and len(mus) > 1:
            add("different_methods_and_sources", "warning", "The portfolios differ in both optimiser and source of expected returns: the comparison cannot say which of the two made the difference.", [a["id"] for a in pl])
    cur = {a["config"].get("currency") for a in arts if a["config"].get("currency")}
    if len(cur) > 1:
        add("different_currency", "blocks_ranking", "The results are in different currencies (" + ", ".join(sorted(map(str, cur))) + "): returns in one cannot be ranked against returns in another without conversion.")
    attr = [a for a in arts if a.get("attribution")]
    if len(attr) >= 2:
        kinds = {(a["attribution"].get("kind") or "") for a in attr}
        if len(kinds) > 1:
            add("different_attribution_models", "warning", "The attribution reports use different factor models (" + ", ".join(sorted(k or "unknown" for k in kinds)) + "): their factor lines are not the same quantities and are not compared line by line.", [a["id"] for a in attr])
    many = [(a["id"], a["metrics"].get("n_trials")) for a in arts if a["metrics"].get("n_trials") and a["metrics"]["n_trials"] > 1]
    if many:
        add("selection", "note", "Some results were picked from many tries (" + ", ".join(f"{i}: {int(n)} trials" for i, n in many) + "). The best of many is biased upward; the deflated Sharpe of each result says by how much.", [i for i, _ in many])
    return out


PREDICTIVE_KINDS = ("factor_study", "forecast")
PREDICTIVE_KEY = {"factor_study": ("best_combination_test_ic", "the best combination's information coefficient on the test period nothing was fitted on"),
                  "forecast": ("best_skill_common", "the best method's skill against the random walk on the dates every method forecast")}
ASSUMPTION_ROWS = (("kind", "Kind of study"), ("assets", "Instruments"), ("start", "First date"), ("end", "Last date"), ("dataset", "Data version"), ("horizon", "Horizon (days)"), ("costs", "Trading costs"),
                   ("benchmark", "Benchmark"), ("construction", "Construction"), ("mu", "Expected returns"), ("evidence", "Evidence"), ("pit", "Point-in-time status"), ("version", "Version"))


def _assumptions(a: dict) -> dict:
    c, cfg = a["context"], a["config"]
    study = cfg.get("study") or {}
    construction = ", ".join(f"{k} {study[k]}" for k in ("lookback", "step", "cov_method", "max_weight") if k in study) or ""
    pit = (cfg.get("fundamental") or {}).get("pit_status") or ("past-only" if (a.get("validation") or {}).get("point_in_time") else "")
    return {"kind": a["kind"], "assets": len(c.get("assets") or []), "start": c.get("start") or "", "end": c.get("end") or "", "dataset": c.get("dataset_version") or "", "horizon": cfg.get("horizon") or "",
            "costs": ("; ".join(f"{k} {v}" for k, v in (a.get("costs") or {}).items() if v is not None)) or "", "benchmark": a.get("benchmark") or "", "construction": construction, "mu": cfg.get("mu_source") or "",
            "evidence": EVIDENCE_TEXT.get(a["evidence"], a["evidence"]), "pit": pit, "version": a.get("version", 1)}


def study_comparison(arts: list[dict]) -> dict:
    """What the saved studies can be compared on besides a return series: the assumptions side by side (and which differ), the predictive scores of factor studies and forecast evaluations (never mixed with trading
    performance), the attribution reports, and, for every study that could not be measured, why."""
    rows = {a["id"]: _assumptions(a) for a in arts}
    differ = [k for k, _ in ASSUMPTION_ROWS if len({str(r[k]) for r in rows.values()}) > 1]
    role = {}
    for a in arts:
        role[a["id"]] = "predictive" if a["kind"] in PREDICTIVE_KINDS else "trading" if a.get("series") else "study"
    predictive = {}
    for a in arts:
        if a["kind"] in PREDICTIVE_KINDS:
            key, text = PREDICTIVE_KEY[a["kind"]]
            predictive[a["id"]] = {"kind": a["kind"], "headline": a["metrics"].get(key), "headline_means": text, "metrics": a["metrics"]}
    ranking = None
    pred_blocks = []
    pa = [a for a in arts if a["kind"] in PREDICTIVE_KINDS]
    if len(pa) >= 2:
        if len({a["kind"] for a in pa}) > 1:
            pred_blocks.append("a factor study and a forecast evaluation answer different questions")
        for field, word in (("horizon", "horizons"), ):
            if len({a["config"].get(field) for a in pa}) > 1:
                pred_blocks.append(f"different {word}")
        if len({(a["context"].get("start"), a["context"].get("end")) for a in pa}) > 1:
            pred_blocks.append("different evaluation dates")
        if len({tuple(sorted(a["context"].get("assets") or [])) for a in pa}) > 1:
            pred_blocks.append("different instruments")
        if len({a["context"].get("dataset_version") for a in pa}) > 1:
            pred_blocks.append("different data versions")
        if not pred_blocks:
            ranked = sorted(pa, key=lambda a: -(a["metrics"].get(PREDICTIVE_KEY[a["kind"]][0]) if a["metrics"].get(PREDICTIVE_KEY[a["kind"]][0]) is not None else -9))
            ranking = [a["id"] for a in ranked]
    attribution = []
    for a in arts:
        at = a.get("attribution")
        if not at:
            continue
        f, b = at.get("factor") or {}, at.get("brinson") or {}
        attribution.append({"id": a["id"], "method": at.get("method"), "kind": at.get("kind"), "factor_totals": f.get("totals"), "factor_status": (f.get("reconciliation") or {}).get("status"),
                            "brinson_totals": b.get("totals"), "brinson_status": (b.get("reconciliation") or {}).get("status"), "benchmark": at.get("benchmark")})
    unmeasured = [{"id": a["id"], "reason": ("a predictive study: its scores are shown on their own and are not trading performance" if a["kind"] in PREDICTIVE_KINDS else
                                              "no return series was saved with it, so nothing can be recomputed over a common window; only its saved summary is shown") }
                  for a in arts if not a.get("series")]
    return {"roles": role, "assumptions": {"rows": [{"key": k, "label": lab} for k, lab in ASSUMPTION_ROWS], "values": rows, "differing": differ},
            "predictive": {"studies": predictive, "ranking": ranking, "blocked_because": pred_blocks,
                           "note": "Predictive quality (IC, forecast skill) and trading performance (Sharpe, drawdown) answer different questions. A better forecast does not by itself make a better portfolio, and a better portfolio does not show a better forecast."},
            "attribution": attribution, "unmeasured": unmeasured}


def all_issues(arts: list[dict], window) -> list[dict]:
    """The issues of the results that have a return stream (these decide whether they can be ranked), of the studies that have none (these decide whether the studies can be put side by side), and a note when
    both kinds were chosen: predictive quality and trading performance are never put in one ranking."""
    trading = [a for a in arts if a.get("series")]
    studies = [a for a in arts if not a.get("series")]
    out = [{**i, "scope": "trading"} for i in issues_for(trading, window)] + [{**i, "scope": "studies"} for i in issues_for(studies, None)]
    if trading and studies:
        out.append({"code": "mixed_roles", "severity": "note", "scope": "studies", "artifacts": [a["id"] for a in arts],
                    "message": "Some chosen results are trading results (a return series) and some are studies without one (predictive scores, a signal's implementation). They are shown in separate blocks: predictive quality and trading performance are different questions and are never ranked together."})
    return out


def compare(arts: list[dict], *, benchmark: str = "", start: str = "", end: str = "", risk_free: float | None = None, roll: int | None = None) -> dict:
    """The comparison of ``arts`` (artifacts with their series). ``benchmark`` is the id of one of them to treat as the reference, or ``"equal_weight"``-style names of a benchmark curve a result carries is
    not used: only a saved result can be the reference, so that it has its own evidence label."""
    if len(arts) < 1:
        raise ValueError("choose at least one result")
    rets = {a["id"]: _returns(a) for a in arts}
    have = {k: v for k, v in rets.items() if v is not None and len(v)}
    out = {"studies": study_comparison(arts), "artifacts": [{"id": a["id"], "title": a["title"], "kind": a["kind"], "version": a.get("version", 1), "parent": a.get("parent", ""), "hypothesis": a.get("hypothesis", ""), "evidence": a["evidence"], "evidence_text": EVIDENCE_TEXT.get(a["evidence"], a["evidence"]), "data_kind": a["data_kind"],
                          "context": a["context"], "refs": a["refs"], "reported": a["metrics"], "has_series": a["id"] in have} for a in arts], "missing_series": [i for i in rets if i not in have]}
    if not have:
        out.update(window=None, issues=all_issues(arts, None), metrics={}, series={}, rankable=False, ranking=None, reason="none of the chosen results has a return series, so nothing can be recomputed over a common window; the studies are compared on their saved summaries")
        return out
    lo = max(v.index[0] for v in have.values())
    hi = min(v.index[-1] for v in have.values())
    if start:
        lo = max(lo, pd.Timestamp(start))
    if end:
        hi = min(hi, pd.Timestamp(end))
    window = (str(lo.date()), str(hi.date()))
    out["window"] = {"start": window[0], "end": window[1]}
    cut = {k: v.loc[lo:hi] for k, v in have.items()}
    days = min(len(v) for v in cut.values()) if cut else 0
    out["issues"] = all_issues(arts, window)
    rf = float(risk_free or 0.0)
    out["risk_free"] = rf
    if hi <= lo or days < MIN_DAYS:
        out.update(metrics={}, series={}, rankable=False, ranking=None, reason=f"the results share only {max(days, 0)} days ({window[0]} to {window[1]}); at least {MIN_DAYS} are needed to compare them")
        out["issues"].append({"code": "short_overlap", "severity": "blocks_ranking", "message": out["reason"], "artifacts": list(have)})
        return out
    metrics = {k: measure(v, rf) for k, v in cut.items()}
    for a in arts:                                                                      # costs and turnover, where the result saved them
        if a["id"] not in cut:
            continue
        for col, name in (("turnover", "turnover_annual"), ("costs", "cost_drag_annual")):
            s = _extra(a, col, cut[a["id"]].index)
            if s is not None and s.notna().any():
                metrics[a["id"]][name] = float(pd.to_numeric(s, errors="coerce").mean() * ANN)
    window_len = days
    rw = int(roll or (252 if window_len >= 400 else max(30, window_len // 4)))
    series = {"dates": [str(d.date()) for d in next(iter(cut.values())).index], "roll_window": rw, "equity": {}, "drawdown": {}, "rolling_sharpe": {}, "rolling_vol": {}}
    index = next(iter(cut.values())).index
    for k, r in cut.items():
        r = r.reindex(index)
        eq = (1.0 + r.fillna(0.0)).cumprod()
        series["equity"][k] = [_num(round(v, 5)) for v in eq]
        series["drawdown"][k] = [_num(round(v, 5)) for v in performance.drawdown_series(r.fillna(0.0))]
        mean, sd = r.rolling(rw, min_periods=rw).mean(), r.rolling(rw, min_periods=rw).std(ddof=1)
        series["rolling_sharpe"][k] = [_num(round(v, 3)) for v in (math.sqrt(ANN) * (mean - rf / ANN) / sd)]
        series["rolling_vol"][k] = [_num(round(v, 4)) for v in sd * math.sqrt(ANN)]
    frame = pd.DataFrame({k: v.reindex(index) for k, v in cut.items()})
    corr = frame.corr()
    out["correlation"] = {"ids": list(frame.columns), "values": [[_num(round(float(corr.iloc[i, j]), 3)) for j in range(len(corr))] for i in range(len(corr))]}
    edges = np.linspace(float(frame.min().min()), float(frame.max().max()), 31) if len(frame) else np.array([])
    out["distribution"] = {"edges": [_num(round(float(e), 5)) for e in edges], "counts": {k: [int(c) for c in np.histogram(frame[k].dropna(), bins=edges)[0]] for k in frame.columns}} if len(edges) > 2 else None
    ref = benchmark if benchmark in cut else ""
    if ref:
        out["versus_reference"] = {k: performance.benchmark_relative(cut[k], cut[ref]) for k in cut if k != ref}
        out["reference"] = ref
    elif benchmark:
        out["issues"].append({"code": "reference_missing", "severity": "note", "message": "The reference result has no return series on this window and was ignored.", "artifacts": []})
    out["metrics"], out["series"] = metrics, series
    blocks = [i for i in out["issues"] if i["severity"] == "blocks_ranking" and i.get("scope") != "studies"]
    out["rankable"] = not blocks and len(cut) > 1
    if len(cut) < 2:
        out["ranking"], out["reason"] = None, "a ranking needs at least two results"
    elif blocks:
        out["ranking"], out["reason"] = None, "not ranked: " + "; ".join(i["message"].split(".")[0] for i in blocks)
    else:
        key = RANK_BY
        out["ranking"] = [k for k, _ in sorted(((k, m.get(key)) for k, m in metrics.items()), key=lambda kv: (kv[1] is None, -(kv[1] or 0.0)))]
        out["reason"] = f"ranked by {key.replace('_', ' ')} over the shared window; the order is a description of this window, not a forecast, and a small gap is not a difference"
    return out


# ------------------------------------------------------------------------------------------------------ export
METRIC_ROWS = (("cagr", "CAGR", "pct"), ("ann_vol", "Volatility (annual)", "pct"), ("sharpe", "Sharpe", "num"), ("sortino", "Sortino", "num"), ("max_drawdown", "Max drawdown", "pct"), ("calmar", "Calmar", "num"),
               ("total_return", "Total return", "pct"), ("turnover_annual", "Turnover (annual)", "num"), ("cost_drag_annual", "Costs (annual, fraction)", "pct"), ("n_days", "Days", "int"))


def _fmt(v, kind):
    if v is None:
        return ""
    return f"{v:.2%}" if kind == "pct" else (f"{int(v)}" if kind == "int" else f"{v:.2f}")


def to_csv(cmp: dict) -> str:
    """The metrics table as CSV, one column per result, with the window and the evidence label on the first rows so the file cannot be read without them."""
    import csv
    import io

    arts = [a for a in cmp["artifacts"] if a["id"] in cmp.get("metrics", {})]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["window", (cmp.get("window") or {}).get("start", ""), (cmp.get("window") or {}).get("end", "")])
    w.writerow(["metric"] + [a["title"] for a in arts])
    w.writerow(["evidence"] + [a["evidence_text"] for a in arts])
    w.writerow(["data"] + [a["data_kind"] for a in arts])
    for key, label, kind in METRIC_ROWS:
        w.writerow([label] + [_fmt(cmp["metrics"][a["id"]].get(key), kind) for a in arts])
    for issue in cmp.get("issues", []):
        w.writerow([f"{issue['severity']}: {issue['message']}"])
    return buf.getvalue()


def to_markdown(cmp: dict, *, session: dict | None = None, group: dict | None = None, generated: str | None = None) -> str:
    """A report a colleague could use to repeat the comparison: what was compared, over which days, with which references, what differs, and the caveats."""
    gen = generated or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = (group or {}).get("name") or "Research comparison"
    lines = [f"# {title}", "", f"Generated {gen}." + (f" Session: {session['name']} ({session['id']})." if session else ""), ""]
    win = cmp.get("window")
    lines += [f"**Window shared by all results:** {win['start']} to {win['end']}." if win else "**No shared window.**", f"**Risk-free rate used for ratios:** {cmp.get('risk_free', 0.0):.2%} a year.", ""]
    lines += ["## Results", ""]
    for a in cmp["artifacts"]:
        refs = ", ".join(f"{k} `{v}`" for k, v in a["refs"].items())
        ctx = a["context"]
        lines += [f"### {a['title']}", f"- Kind: {a['kind'].replace('_', ' ')}; evidence: {a['evidence_text']}; data: {a['data_kind']}",
                  f"- Assets: {', '.join(ctx.get('assets') or []) or 'not recorded'}; source: {ctx.get('data_source') or 'not recorded'}; dates: {ctx.get('start', '?')} to {ctx.get('end', '?')}",
                  f"- References: {refs or 'none recorded'}", ""]
    if cmp.get("metrics"):
        arts = [a for a in cmp["artifacts"] if a["id"] in cmp["metrics"]]
        lines += ["## Metrics over the shared window", "", "| Metric | " + " | ".join(a["title"] for a in arts) + " |", "|---|" + "---|" * len(arts)]
        for key, label, kind in METRIC_ROWS:
            lines.append(f"| {label} | " + " | ".join(_fmt(cmp["metrics"][a["id"]].get(key), kind) for a in arts) + " |")
        lines.append("")
    lines += ["## Ranking", "", cmp.get("reason") or "", ""]
    if cmp.get("ranking"):
        by = {a["id"]: a["title"] for a in cmp["artifacts"]}
        lines += [f"{i + 1}. {by[k]}" for i, k in enumerate(cmp["ranking"])] + [""]
    lines += ["## Differences in assumptions", ""]
    lines += [f"- **{i['severity'].replace('_', ' ')}**: {i['message']}" for i in cmp.get("issues", [])] or ["- None found."]
    lines += ["", "## Caveats", "", "- Past performance, however it was measured, does not describe future returns. Nothing here is a forecast or advice.",
              "- Every result is a simulation on saved data unless its evidence says paper or live. Historical simulations of the whole sample are not out-of-sample evidence.",
              "- Returns are net of the costs each result assumed; the costs differ where a difference is listed above.", ""]
    return "\n".join(lines)
