"""The Research Comparison: put saved results side by side without letting the layout say more than the evidence does.

What it does that a table of the artifacts' own metrics cannot:

* every result is measured again **on the dates they all share** with one piece of arithmetic (``research.conventions``: elapsed-time CAGR, volatility and Sharpe annualised by the declared or inferred
  calendar of the common dates, the risk-free rate accrued by elapsed time), so a ratio is never a ratio over different years and a day one result lacks is never a zero return in another;
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
from . import conventions, metrics as metric_registry
from .extract import returns_from_equity

MIN_DAYS = 30                                                                           # fewest returns on the common dates for which a comparison is made
RANK_BY = metric_registry.DEFAULT_TRADING
EVIDENCE_TEXT = {"in_sample": "in-sample", "out_of_sample": "out-of-sample", "walk_forward": "walk-forward (out-of-sample)", "historical_simulation": "historical simulation of the whole sample", "simulated": "simulated data",
                 "retrospective": "retrospective (uses information not known at the time)", "paper": "paper trading", "live": "live"}


def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def _returns(art: dict) -> pd.Series | None:
    """The saved curve's own period returns (the first one relative to the starting capital of 1). Used for a result on its own; the comparison measures on the common dates from the levels (:func:`_levels`)."""
    s = art.get("series") or {}
    if not s.get("dates") or not s.get("equity"):
        return None
    return returns_from_equity(s["dates"], [np.nan if v is None else v for v in s["equity"]]).dropna()


def _levels(art: dict) -> pd.Series | None:
    """The dated levels that were actually observed: a missing level is absent, never filled, so a return is always the change between two observed levels."""
    s = art.get("series") or {}
    if not s.get("dates") or not s.get("equity"):
        return None
    lv = pd.Series([np.nan if v is None else float(v) for v in s["equity"]], index=pd.to_datetime(list(s["dates"]))).dropna()
    return lv if len(lv) else None


def _extra(art: dict, name: str, index: pd.DatetimeIndex) -> pd.Series | None:
    vals = ((art.get("series") or {}).get("extra") or {}).get(name)
    if vals is None:
        return None
    dates = pd.to_datetime(art["series"]["dates"])
    return pd.Series(vals, index=dates, dtype=object).reindex(index)


def measure(r: pd.Series, rf: float = 0.0) -> dict:
    """The headline numbers of a return series dated by its index, under the convention its own dates imply (see :mod:`research.conventions`). Kept for callers that hold returns; the comparison itself measures
    levels on the common dates with :func:`conventions.measure_levels`."""
    r = r.dropna().astype(float)
    if len(r) < MIN_DAYS:
        return {"n_days": len(r)}
    m = conventions.measure_returns(r, rf)
    risk = performance.risk_metrics(r)
    m.update(skew=_num(risk.get("skew")), excess_kurtosis=_num(risk.get("excess_kurtosis")), var_95=_num(risk.get("var_95")), es_95=_num(risk.get("es_95")), positive_days=_num(risk.get("positive_days")))
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in m.items()}


def _full_measure(lv: pd.Series, ann: dict, rf: float) -> dict:
    m = conventions.measure_levels(lv, ann, rf)
    if m.get("n_days", 0) < 3:
        return m
    r = lv.pct_change().dropna()
    risk = performance.risk_metrics(r)
    m.update(skew=_num(risk.get("skew")), excess_kurtosis=_num(risk.get("excess_kurtosis")), var_95=_num(risk.get("var_95")), es_95=_num(risk.get("es_95")), positive_days=_num(risk.get("positive_days")))
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in m.items()}


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
    ranking, key = None, None
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
            key = PREDICTIVE_KEY[pa[0]["kind"]][0]                                                 # one kind, so one headline metric, sorted in that metric's own direction
            unfit = metric_registry.applicability(pa, key)
            if unfit:
                pred_blocks.extend(unfit)
            else:
                ranking = metric_registry.rank({a["id"]: a["metrics"].get(key) for a in pa}, key)
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
            "predictive": {"studies": predictive, "ranking": ranking, "blocked_because": pred_blocks, "ranked_by": None if ranking is None else {"key": key, "label": metric_registry.get(key).label, "direction": metric_registry.get(key).direction},
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


def _per_year(ann: dict, n: int, elapsed: float) -> float:
    """Observations per year: the declared or inferred periods per year, or, for an irregular grid, the number of observations over the elapsed time."""
    return float(ann["periods_per_year"]) if ann["basis"] == "periods" else (n / elapsed if elapsed > 0 else float("nan"))


def convention_issues(arts: list[dict], metas: dict[str, dict], grid_ann: dict, coverage: dict[str, dict]) -> list[dict]:
    """What differs in how the results were sampled and what that does to the numbers: calendars, frequencies, return types, currencies, an irregular grid, observations dropped to reach the common dates."""
    out: list[dict] = []

    def add(code, severity, message, ids):
        out.append({"code": code, "severity": severity, "message": message, "artifacts": ids})

    ids = list(metas)
    cal = {m["calendar"] for m in metas.values()}
    if len(cal) > 1:
        add("different_calendars", "note", "The results trade on different calendars (" + ", ".join(sorted(c.replace("_", " ") for c in cal)) + "). They are measured only on the dates every one of them has a level for; a return across a day one result "
            "did not trade is the change between two levels it did record, so nothing is invented for the days it lacks.", ids)
    freq = {m["frequency"] for m in metas.values()}
    if len(freq) > 1:
        add("mixed_frequency", "warning", "The results are sampled at different frequencies (" + ", ".join(sorted(freq)) + "). They are compared on the dates they share, which is the coarser grid (" + grid_ann["frequency"] + "); detail finer than that is not used.", ids)
    rtype = {m.get("return_type") for m in metas.values() if str(m.get("source", "")).startswith("declared")}
    if len(rtype) > 1:
        add("different_return_type", "blocks_ranking", "The results declare different return types (" + ", ".join(sorted(map(str, rtype))) + "): simple and logarithmic returns are not added or annualised alike.", ids)
    cur = {m.get("currency") for m in metas.values() if m.get("currency") not in (None, "", "unspecified")}
    if len(cur) > 1:
        add("different_currency", "blocks_ranking", "The results declare different currencies (" + ", ".join(sorted(map(str, cur))) + "): returns in one cannot be ranked against returns in another without conversion.", ids)
    rfc = {m.get("risk_free") for m in metas.values() if str(m.get("source", "")).startswith("declared") and m.get("risk_free")}
    if len(rfc) > 1:
        add("different_risk_free_convention", "warning", "The results declare different risk-free conventions (" + "; ".join(sorted(map(str, rfc))) + "); the comparison uses one for all of them.", ids)
    if grid_ann["basis"] == "elapsed":
        add("irregular_spacing", "warning", "The shared dates are unevenly spaced, so volatility and Sharpe are annualised by elapsed time (variance per year of elapsed calendar time), which assumes returns scale with elapsed time. "
            "They are not multiplied by a fixed number of periods per year.", ids)
    undeclared = [i for i, m in metas.items() if not str(m.get("source", "")).startswith("declared")]
    if undeclared:
        add("annualisation_inferred", "note", f"The calendar and frequency of {len(undeclared)} result(s) were not saved with them and were inferred from their dates; the annualisation used is {grid_ann['calendar'].replace('_', ' ')} "
            + ("(elapsed time)" if grid_ann["basis"] == "elapsed" else f"({grid_ann['periods_per_year']:g} periods a year)") + ".", undeclared)
    lost = {i: c for i, c in coverage.items() if c["dropped_in_window"] > 0}
    if lost:
        worst = max(lost.values(), key=lambda c: c["dropped_in_window"] / max(c["in_window"], 1))
        share = worst["dropped_in_window"] / max(worst["in_window"], 1)
        add("sample_reduced", "warning" if share > 0.2 else "note", "To compare like with like only the dates every result observed are used; observations dropped from the window: "
            + "; ".join(f"{_title(arts, i)}: {c['dropped_in_window']} of {c['in_window']}" for i, c in lost.items()) + ". A missing date is left out for all results, never counted as a day with no return.", list(lost))
    return out


def _title(arts: list[dict], i: str) -> str:
    return next((a["title"] for a in arts if a["id"] == i), i)


def compare(arts: list[dict], *, benchmark: str = "", start: str = "", end: str = "", risk_free: float | None = None, roll: int | None = None, rank_by: str = "") -> dict:
    """The comparison of ``arts`` (artifacts with their series). ``benchmark`` is the id of one of them to treat as the reference: only a saved result can be the reference, so that it has its own evidence label.

    The results are measured on **the dates they all observed** (the intersection of their dated levels inside the window they share). Returns are the changes between consecutive common levels, so a result
    that did not trade on a date, started later, or has a hole in its history is never given a zero return there. ``rank_by`` is a key of :mod:`research.metrics`; it is sorted in that metric's own direction."""
    if len(arts) < 1:
        raise ValueError("choose at least one result")
    key = rank_by or RANK_BY
    metric_registry.get(key)
    levels = {a["id"]: _levels(a) for a in arts}
    bad = [i for i, lv in levels.items() if lv is not None and (lv <= 0).any()]
    have = {k: v for k, v in levels.items() if v is not None and len(v) >= 2 and k not in bad}
    out = {"studies": study_comparison(arts), "artifacts": [{"id": a["id"], "title": a["title"], "kind": a["kind"], "version": a.get("version", 1), "parent": a.get("parent", ""), "hypothesis": a.get("hypothesis", ""), "evidence": a["evidence"], "evidence_text": EVIDENCE_TEXT.get(a["evidence"], a["evidence"]), "data_kind": a["data_kind"],
                          "context": a["context"], "refs": a["refs"], "reported": a["metrics"], "has_series": a["id"] in have} for a in arts], "missing_series": [i for i in levels if i not in have]}
    out["metric_registry"] = metric_registry.describe()
    if not have:
        out.update(window=None, issues=all_issues(arts, None), metrics={}, series={}, rankable=False, ranking=None, reason="none of the chosen results has a return series, so nothing can be recomputed over a common window; the studies are compared on their saved summaries")
        out["issues"] += [{"code": "invalid_levels", "severity": "warning", "scope": "trading", "message": f"{_title(arts, i)} has a level at or below zero, so its returns are undefined and it was not measured.", "artifacts": [i]} for i in bad]
        return out
    lo = max(v.index[0] for v in have.values())
    hi = min(v.index[-1] for v in have.values())
    if start:
        lo = max(lo, pd.Timestamp(start))
    if end:
        hi = min(hi, pd.Timestamp(end))
    grid = None
    for v in have.values():
        ix = v.loc[lo:hi].index
        grid = ix if grid is None else grid.intersection(ix)
    grid = pd.DatetimeIndex(sorted(grid)) if grid is not None else pd.DatetimeIndex([])
    days = max(len(grid) - 1, 0)
    window = (str((grid[0] if len(grid) else lo).date()), str((grid[-1] if len(grid) else hi).date()))
    out["window"] = {"start": window[0], "end": window[1]}
    out["issues"] = all_issues(arts, window)
    out["issues"] += [{"code": "invalid_levels", "severity": "warning", "scope": "trading", "message": f"{_title(arts, i)} has a level at or below zero, so its returns are undefined and it was not measured.", "artifacts": [i]} for i in bad]
    rf = float(risk_free or 0.0)
    out["risk_free"] = rf
    metas = {a["id"]: conventions.meta_of(a.get("series")) for a in arts if a["id"] in have}
    grid_ann = conventions.grid_annualisation(grid, list(metas.values())) if len(grid) >= 4 else {"basis": "elapsed", "periods_per_year": None, "calendar": "custom", "frequency": "irregular", "source": "too few common dates"}
    coverage = {}
    for k, v in have.items():
        inside = v.loc[lo:hi]
        gaps = np.diff(inside.index.values).astype("timedelta64[D]").astype(int) if len(inside) > 1 else np.array([0])
        coverage[k] = {"first": str(v.index[0].date()), "last": str(v.index[-1].date()), "observations": int(len(v)), "in_window": int(len(inside)), "on_common_dates": int(len(grid)), "dropped_in_window": int(len(inside) - len(grid)),
                       "longest_gap_days": int(gaps.max()), "calendar": metas[k]["calendar"], "frequency": metas[k]["frequency"]}
    out["coverage"] = coverage
    out["conventions"] = {"metadata": metas, "common_dates": {**{k: grid_ann[k] for k in ("basis", "periods_per_year", "calendar", "frequency", "source")}, "n_dates": int(len(grid))},
                          "cagr": "elapsed calendar time between the first and last common date (365.25-day years)", "risk_free": conventions.RF_CONVENTION,
                          "volatility": ("sd of the per-period returns x sqrt(periods per year)" if grid_ann["basis"] == "periods" else "variance per year of elapsed calendar time, square-rooted")}
    out["issues"] += convention_issues(arts, metas, grid_ann, coverage)
    if hi <= lo or days < MIN_DAYS:
        out.update(metrics={}, series={}, rankable=False, ranking=None, reason=f"the results share only {max(days, 0)} dates with a level in common ({window[0]} to {window[1]}); at least {MIN_DAYS} returns are needed to compare them")
        out["issues"].append({"code": "short_overlap", "severity": "blocks_ranking", "message": out["reason"], "artifacts": list(have)})
        return out
    on_grid = {k: v.loc[grid] for k, v in have.items()}
    metrics = {k: _full_measure(v, grid_ann, rf) for k, v in on_grid.items()}
    out["individual"] = {}                                                               # each result on its own whole history, under its own calendar: informative, not comparable between results
    for k, v in have.items():
        own = conventions.grid_annualisation(v.index, [metas[k]]) if len(v) >= 4 else {"basis": "elapsed", "periods_per_year": None, "calendar": "custom", "frequency": "irregular", "source": "too few dates"}
        out["individual"][k] = {**_full_measure(v, own, rf), "first": str(v.index[0].date()), "last": str(v.index[-1].date())}
    for a in arts:                                                                      # costs and turnover, where the result saved them, per year of the result's own observations
        if a["id"] not in have:
            continue
        own = have[a["id"]]
        own_ann = conventions.grid_annualisation(own.loc[grid[0]:grid[-1]].index, [metas[a["id"]]])
        for col, name in (("turnover", "turnover_annual"), ("costs", "cost_drag_annual")):
            s = _extra(a, col, own.loc[grid[0]:grid[-1]].index)
            if s is None:
                continue
            s = pd.to_numeric(s, errors="coerce").iloc[1:]
            if s.notna().any():
                span = float((s.index[-1] - own.loc[grid[0]:grid[-1]].index[0]).days) / conventions.YEAR_DAYS
                metrics[a["id"]][name] = float(s.mean() * _per_year(own_ann, int(s.notna().sum()), span))
    rw = int(roll or (252 if days >= 400 else max(30, days // 4)))
    index = grid
    series = {"dates": [str(d.date()) for d in index], "roll_window": rw, "equity": {}, "drawdown": {}, "rolling_sharpe": {}, "rolling_vol": {}}
    rets = {}
    for k, v in on_grid.items():
        eq = v / v.iloc[0]
        series["equity"][k] = [_num(round(float(x), 5)) for x in eq]
        series["drawdown"][k] = [_num(round(float(x), 5)) for x in (eq / eq.cummax() - 1.0)]
        roll_m = conventions.rolling_measures(v, grid_ann, rw, rf)
        series["rolling_sharpe"][k] = [None] + [_num(round(float(x), 3)) for x in roll_m["sharpe"]]
        series["rolling_vol"][k] = [None] + [_num(round(float(x), 4)) for x in roll_m["vol"]]
        rets[k] = v.pct_change().dropna()
    frame = pd.DataFrame(rets)
    corr = frame.corr()
    out["correlation"] = {"ids": list(frame.columns), "values": [[_num(round(float(corr.iloc[i, j]), 3)) for j in range(len(corr))] for i in range(len(corr))]}
    edges = np.linspace(float(frame.min().min()), float(frame.max().max()), 31) if len(frame) else np.array([])
    out["distribution"] = {"edges": [_num(round(float(e), 5)) for e in edges], "counts": {k: [int(c) for c in np.histogram(frame[k].dropna(), bins=edges)[0]] for k in frame.columns}} if len(edges) > 2 else None
    ref = benchmark if benchmark in rets else ""
    if ref:
        out["versus_reference"] = {k: conventions.relative_to(rets[k], rets[ref], grid_ann) for k in rets if k != ref}
        out["reference"] = ref
    elif benchmark:
        out["issues"].append({"code": "reference_missing", "severity": "note", "message": "The reference result has no return series on this window and was ignored.", "artifacts": []})
    out["metrics"], out["series"] = metrics, series
    out["rank_by"] = {"key": key, "label": metric_registry.get(key).label, "direction": metric_registry.get(key).direction}
    blocks = [i for i in out["issues"] if i["severity"] == "blocks_ranking" and i.get("scope") != "studies"]
    out["rankable"] = not blocks and len(on_grid) > 1
    unfit = metric_registry.applicability([a for a in arts if a["id"] in on_grid], key)
    if len(on_grid) < 2:
        out["ranking"], out["reason"] = None, "a ranking needs at least two results"
    elif blocks:
        out["ranking"], out["reason"] = None, "not ranked: " + "; ".join(i["message"].split(".")[0] for i in blocks)
    elif unfit:
        out["rankable"], out["ranking"], out["reason"] = False, None, "not ranked: " + "; ".join(unfit)
    else:
        out["ranking"] = metric_registry.rank({k: m.get(key) for k, m in metrics.items()}, key)
        way = {"higher": "highest first", "lower": "lowest first"}[metric_registry.get(key).direction]
        out["reason"] = f"ranked by {metric_registry.get(key).label} ({way}) on the {days} returns between the dates every result observed; the order is a description of this window, not a forecast, and a small gap is not a difference"
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
    lines += [f"**Window shared by all results:** {win['start']} to {win['end']}." if win else "**No shared window.**", f"**Risk-free rate used for ratios:** {cmp.get('risk_free', 0.0):.2%} a year ({conventions.RF_CONVENTION}).", ""]
    conv = cmp.get("conventions")
    if conv:
        g = conv["common_dates"]
        lines += [f"**Measured on:** the {g['n_dates']} dates every result observed ({g['calendar'].replace('_', ' ')}, {g['frequency']}); annualisation " + (f"{g['periods_per_year']:g} periods a year" if g["basis"] == "periods" else "by elapsed calendar time") + f" ({g['source']}). CAGR: {conv['cagr']}.", ""]
        if cmp.get("coverage"):
            lines += ["| Result | Own dates | Observations in window | Left out | Longest gap (days) |", "|---|---|---|---|---|"]
            by = {a["id"]: a["title"] for a in cmp["artifacts"]}
            lines += [f"| {by.get(i, i)} | {c['first']} to {c['last']} | {c['in_window']} | {c['dropped_in_window']} | {c['longest_gap_days']} |" for i, c in cmp["coverage"].items()] + [""]
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
