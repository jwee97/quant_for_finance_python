"""Turn the outcome of a run that already exists into the artifact a session keeps.

The numbers come from the server's own records, never from the page: a backtest from its finished job (``data/user/jobs``), a machine-learning experiment from the registry. What this module adds is
the *labelling* that makes a result comparable later: what kind of evidence it is (a historical simulation of the whole sample, a walk-forward out-of-sample run ...), what kind of data it rests on, the
context it was made under, and the references (job, experiment, configuration hash, code version) a person needs to reproduce it.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

from .store import StoreError

BACKTEST_METRICS = ("cagr", "ann_return", "ann_vol", "vol", "sharpe", "sortino", "calmar", "max_drawdown", "turnover", "avg_turnover", "cost_drag", "hit_rate", "skew", "kurtosis", "total_return", "var_95", "cvar_95")


def fingerprint(*parts) -> str:
    """A short, stable name for 'this data': what was asked for and what came back. Two results with the same fingerprint rest on the same universe, dates and source."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def data_kind(label: str, provider: str | None = None) -> str:
    """The provenance badge of a universe, from the label the universe builder gave it ("platform dataset", "platform dataset + Yahoo Finance", "Yahoo Finance" ...)."""
    text = (label or "").lower()
    if "simulat" in text or "synthetic" in text:
        return "simulated"
    if "file" in text:
        return "user_file"
    if "+" in text:
        return "mixed"
    if provider or (text and "platform" not in text):
        return "downloaded"
    return "platform"


def _numbers(source: dict, keys=None) -> dict:
    out = {}
    for k, v in (source or {}).items():
        if (keys is None or k in keys) and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[k] = float(v)
    return out


def _clean_series(values) -> list:
    return [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else float(v) for v in values]


def backtest_artifact(result: dict, *, job_id: str = "", code_version: str = "unknown") -> dict:
    """The artifact of a finished Backtest-tab job (``result`` is what ``/api/job/<id>`` returns as its result)."""
    for key in ("universe", "dates", "equity", "metrics", "spec"):
        if key not in result:
            raise StoreError("this job did not produce a backtest result")
    uni, dates = result["universe"], list(result["dates"])
    equity = _clean_series(result["equity"])
    if len(equity) != len(dates):
        raise StoreError("the backtest's dates and equity curve do not match")
    metrics = _numbers(result["metrics"])
    val = result.get("validation") or {}
    for src, dst in (("deflated_sharpe_probability", "deflated_sharpe_probability"), ("n_trials", "n_trials")):
        if isinstance(val.get(src), (int, float)) and math.isfinite(val[src]):
            metrics[dst] = float(val[src])
    account = result.get("account") or {}
    ctx = {"assets": list(uni.get("tickers", [])), "start": dates[0], "end": dates[-1], "frequency": "daily", "data_source": str(uni.get("provider") or "platform"),
           "dataset_version": fingerprint(uni.get("tickers"), uni.get("first"), uni.get("last"), uni.get("days"), uni.get("source"))}
    benches = {k: _clean_series(v) for k, v in (result.get("benchmarks") or {}).items() if isinstance(v, list) and len(v) == len(dates)}
    notes = ["A backtest over the whole history: every parameter was chosen with the whole sample in view unless the Tune tab's hold-out was used. It is a historical simulation, not out-of-sample evidence."]
    notes += [str(f.get("message") or f.get("title") or f) for f in (result.get("flags") or [])][:6]
    return {"kind": "backtest", "title": str(result.get("name") or "Backtest")[:120], "evidence": "historical_simulation", "data_kind": data_kind(str(uni.get("source", "")), uni.get("provider")),
            "context": ctx, "config": {"spec": result["spec"], "capital": result.get("capital"), "account": {k: v for k, v in account.items() if isinstance(v, (int, float, str))}},
            "metrics": metrics, "series": {"dates": dates, "equity": equity, "benchmarks": benches},
            "refs": {"job": job_id, "code_version": code_version, "spec_hash": fingerprint(result["spec"]), "dataset": ctx["dataset_version"]}, "notes": notes}


def portfolio_lab_artifact(result: dict, method: str, *, job_id: str = "", code_version: str = "unknown") -> dict:
    """The artifact of ONE method of a finished Portfolio-lab study (``result`` is the job's result). Its net equity curve is the series; the equal-weight run on the same windows is the benchmark; the metrics are its
    row of the table, and ``n_trials`` is how many methods the study tried, because choosing the best of them is a selection that its Sharpe ratio does not account for."""
    for key in ("table", "equity", "config", "universe"):
        if key not in result:
            raise StoreError("this job did not produce a portfolio study")
    rows = {r["key"]: r for r in result["table"]}
    if method not in rows:
        raise StoreError(f"the study has no method {method!r}: it has {', '.join(rows)}")
    dates, series = list(result["equity"]["dates"]), result["equity"]["series"]
    equity = _clean_series(series[method])
    if len(equity) != len(dates):
        raise StoreError("the study's dates and equity curve do not match")
    row, cfg, uni = rows[method], result["config"], result["universe"]
    names = {"sharpe": "sharpe", "sharpe_se": "sharpe_se", "ann_return": "ann_return", "ann_vol": "ann_vol", "max_drawdown": "max_drawdown", "cdar_95": "cdar_95", "turnover_per_year": "turnover", "effective_n": "effective_n",
             "diversification_ratio": "diversification_ratio", "capped": "capped_weight", "fallbacks": "fallbacks", "instability": "instability"}
    metrics = {dst: float(row[src]) for src, dst in names.items() if isinstance(row.get(src), (int, float)) and not isinstance(row.get(src), bool) and math.isfinite(row[src])}
    metrics["n_trials"] = float(len(rows))
    benches = {"Equal weight": _clean_series(series["equal_weight"])} if "equal_weight" in series and method != "equal_weight" and len(series["equal_weight"]) == len(dates) else {}
    ctx = {"assets": list(uni.get("tickers", [])), "start": dates[0], "end": dates[-1], "frequency": "daily", "data_source": str(uni.get("source") or "platform"),
           "dataset_version": fingerprint(uni.get("tickers"), cfg.get("first"), cfg.get("last"), cfg.get("rows"), uni.get("source"))}
    notes = [f"One method ({row.get('method', method)}) of a study that tried {len(rows)}: the Sharpe ratio is not corrected for having chosen it from them. Each rebalance used only the trailing {cfg.get('lookback')} days, so the weights are "
             "walk-forward, but which method to look at was decided with the whole result in view; this is a historical simulation, not out-of-sample evidence."]
    if row.get("fallbacks"):
        notes.append(f"The method failed on {int(row['fallbacks'])} rebalance date(s) and equal weight was used there.")
    cost = cfg.get("cost_bps")
    if cost is not None:
        ctx["costs"] = {"cost_bps": float(cost)}
    er = result.get("expected_returns") or {}
    attr = ((result.get("attribution") or {}).get("methods") or {}).get(method) or {}
    validation = {"expected_returns_source": er.get("source", "historical mean"), "horizon_days": er.get("horizon_days"),
                  "point_in_time": "weights at each rebalance use only information before it; expected returns from forecasts or factors use records available before the decision",
                  "attribution_reconciliation": (attr.get("factor") or {}).get("reconciliation", {}).get("status"), "brinson_reconciliation": (attr.get("brinson") or {}).get("reconciliation", {}).get("status"),
                  "forecast": ({k: er.get("meta", {}).get(k) for k in ("forecast_method", "model_version", "calibrate", "max_age_days", "forecasts", "training_fingerprint_example")} if (er.get("meta") or {}).get("forecast_method") else None)}
    attribution = None
    if attr:
        attribution = {"method": method, "kind": (result.get("attribution") or {}).get("kind"), "factor": attr.get("factor"), "brinson": attr.get("brinson"), "benchmark": (result.get("attribution") or {}).get("benchmark"),
                       "methodology": (result.get("attribution") or {}).get("methodology"), "brinson_reason": (result.get("attribution") or {}).get("brinson_reason")}
    if er:
        notes.append(f"Expected returns for the optimisers that use them: {er.get('source', 'historical mean')}; every source is run with the same risk estimate, caps, costs and schedule.")
    return {"kind": "portfolio_lab", "title": f"Portfolio lab: {row.get('method', method)}"[:120], "evidence": "historical_simulation", "data_kind": data_kind(str(uni.get("source", "")), None), "context": ctx,
            "config": {"method": method, "study": {k: v for k, v in cfg.items() if isinstance(v, (int, float, str))}, "methods_tried": list(rows), "mu_source": er.get("source", "historical mean"), "mu_meta": er.get("meta")},
            "metrics": metrics, "series": {"dates": dates, "equity": equity, "benchmarks": benches}, "refs": {"job": job_id, "code_version": code_version, "dataset": ctx["dataset_version"]}, "notes": notes,
            "benchmark": "Equal weight" if benches else "", "costs": {"cost_bps": cost, "charged": "on the rebalance day, on the whole of the trade"}, "validation": validation, "attribution": attribution}


def ml_artifact(report: dict, returns: pd.DataFrame, *, code_version: str = "unknown") -> dict:
    """The artifact of a saved machine-learning experiment (``report`` is its ``report.json``, ``returns`` its ``returns.parquet``): its out-of-sample net returns, never the in-sample fit."""
    if returns is None or not len(returns) or "net" not in returns:
        raise StoreError("this experiment has no saved returns")
    net = returns["net"].astype(float).fillna(0.0)
    dates = [str(d)[:10] for d in net.index]
    equity = (1.0 + net).cumprod()
    extra = {}
    for col in ("turnover", "costs"):
        if col in returns:
            extra[col] = _clean_series(returns[col].astype(float).to_numpy())
    if "segment" in returns:
        extra["segment"] = [str(s) for s in returns["segment"].to_numpy()]
    benches = {}
    if "benchmark" in returns and returns["benchmark"].notna().any():
        benches["benchmark"] = _clean_series((1.0 + returns["benchmark"].astype(float).fillna(0.0)).cumprod())
    ds = report.get("dataset") or {}
    prov = report.get("provenance") or {}
    kind = {"simulated": "simulated", "user_file": "user_file", "downloaded": "downloaded"}.get(str(prov.get("source", "platform")), "platform")
    metrics = {}
    for k, v in (report.get("metrics", {}).get("oos") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            metrics[k] = float(v)
    trials = (report.get("trials") or {}).get("dataset_trials")
    if isinstance(trials, (int, float)):
        metrics["n_trials"] = float(trials)
    cfg = report.get("config") or {}
    ctx = {"start": dates[0], "end": dates[-1], "frequency": "daily", "data_source": str(prov.get("label") or prov.get("source") or "platform"), "dataset_version": str(report.get("dataset_key") or "")[:64]}
    notes = [f"Out-of-sample net returns of a walk-forward run (first prediction {dates[0]}); the segments column says which days are validation, test or held out."] + [str(n) for n in (prov.get("notes") or [])][:4]
    out = {"kind": "ml_experiment", "title": str(report.get("name") or "Experiment")[:120], "evidence": "walk_forward", "data_kind": kind, "context": ctx,
           "config": {"model": (cfg.get("model") or {}), "dataset": (cfg.get("dataset") or {}), "split": (cfg.get("split") or {}), "signal": (cfg.get("signal") or {}), "backtest": (cfg.get("backtest") or {}),
                      "segments": (report.get("segments") or {}).get("plan"), "n_assets": ds.get("n_assets"), "n_features": ds.get("n_features"), "target": ds.get("target"), "horizon": ds.get("horizon")},
           "metrics": metrics, "series": {"dates": dates, "equity": _clean_series(equity.to_numpy()), "benchmarks": benches, "extra": extra},
           "refs": {"ml_experiment": str(report.get("id") or ""), "config_hash": str(report.get("config_hash") or ""), "dataset": str(report.get("dataset_key") or ""), "code_version": code_version}, "notes": notes}
    if (cfg.get("signal") or {}).get("method") == "portfolio_allocation" and report.get("portfolio_allocation"):
        _allocation_fields(out, report, cfg, ds)
    return out


def _allocation_fields(out: dict, report: dict, cfg: dict, ds: dict) -> None:
    """What an allocation study adds to its artifact so that it can be compared with portfolio studies on like terms: the construction in the portfolio lab's own vocabulary, the cost it charged, the benchmark,
    the currency it was valued in, and the point-in-time statement with what the run depended on."""
    pa = report["portfolio_allocation"]
    pc = pa.get("config") or {}
    lim = pc.get("limits") or {}
    bt = cfg.get("backtest") or {}
    fx = report.get("fx") or {}
    assets = list(ds.get("universe") or []) if isinstance(ds.get("universe"), list) else []
    if not assets:                                                                  # the universe may be a preset name; the books name the stocks that were really held
        for rec in pa.get("records") or []:
            if rec.get("weights"):
                assets = sorted(rec["weights"])
                break
    out["context"].update({"assets": assets, "costs": {"cost_bps": float(bt.get("cost_bps", 0.0)) + float(bt.get("slippage_bps", 0.0))}})
    out["costs"] = {"cost_bps": float(bt.get("cost_bps", 0.0)) + float(bt.get("slippage_bps", 0.0)), "charged": "on every trade, as a fraction of the weight traded"}
    out["benchmark"] = "Equal weight"
    out["config"]["study"] = {"lookback": pc.get("cov_window"), "step": pa.get("opt_every"), "cov_method": pc.get("cov_method"), "max_weight": lim.get("max_weight"), "risk_aversion": pc.get("risk_aversion"),
                              "holding_days": pa.get("holding_window_days"), "allocator": pa.get("allocator")}
    out["config"]["mu_source"] = "ML forecast, calibrated on past outcomes" if pc.get("calibrate", "slope") == "slope" else "ML forecast, uncalibrated"
    out["config"]["allocation"] = {"allocator": pa.get("allocator"), "settings": pc, "counts": pa.get("counts"), "config_hash": pa.get("config_hash")}
    if fx:
        out["config"]["currency"] = fx.get("base")
    out["validation"] = {"point_in_time": "each weight uses only forecasts, returns and eligibility known at its decision date; the forecast is calibrated on outcomes resolved before it; the covariance is estimated on earlier returns",
                         "holdout": (report.get("holdout") or {}).get("status"), "forecast": (pa.get("records") or [{}])[-1].get("forecast") if pa.get("records") else None,
                         "fx": None if not fx else {"base": fx.get("base"), "assumption": (fx.get("forecast_conversion") or {}).get("assumption"), "exchange_accurate": fx.get("exchange_accurate"), "hedge": (fx.get("hedge") or {}).get("active")}}
    out["notes"].append(f"Portfolio allocation by {(pa.get('documentation') or {}).get('label', pa.get('allocator'))}: the books are a function of the ML forecast, a risk estimate and limits; equal weight is the benchmark. "
                        "Net returns include the backtest's costs" + (" and the currency conversion and hedge costs." if fx else "."))
    if fx and not fx.get("exchange_accurate", True):
        out["notes"].append("The currency hedge is an explicit forward overlay without forward points or financing: not exchange-accurate.")


def returns_from_equity(dates, equity) -> pd.Series:
    """Daily returns of an equity curve (the first day's return is relative to 1.0, the starting capital)."""
    eq = pd.Series(np.asarray(equity, dtype=float), index=pd.to_datetime(list(dates)))
    return eq.pct_change().fillna(eq.iloc[0] - 1.0) if len(eq) else eq


STRATEGY_LABS = {"options_backtest": "options", "futures_backtest": "futures", "crypto_backtest": "crypto"}


def strategy_artifact(result: dict, *, job_id: str = "", code_version: str = "unknown") -> dict:
    """The artifact of a rolling-strategy backtest from the Options, Futures or Crypto lab (``result`` is the job's result, an envelope with ``kind``, ``source`` and the series). The options market is
    simulated, so its evidence is ``simulated``; a futures or crypto backtest on real data is a historical simulation, on simulated data it is simulated."""
    kind = STRATEGY_LABS.get(result.get("kind"))
    if kind is None or not result.get("dates") or "equity" not in result:
        raise StoreError("this job is not a strategy backtest with an equity curve (a sweep or a comparison cannot be saved as one result)")
    dates, equity = list(result["dates"]), _clean_series(result["equity"])
    if len(equity) != len(dates):
        raise StoreError("the backtest's dates and equity curve do not match")
    source = str(result.get("source") or "simulated")
    dk = source if source in ("platform", "user_file", "downloaded", "simulated", "mixed") else "mixed"
    simulated = kind == "options" or dk == "simulated"
    metrics = _numbers(result.get("metrics") or {})
    for key in ("deflated_sharpe_probability", "n_trials"):
        if isinstance(result.get(key), (int, float)) and math.isfinite(result[key]):
            metrics[key] = float(result[key])
    benches = {k: _clean_series(v) for k, v in (result.get("benchmarks") or {}).items() if isinstance(v, list) and len(v) == len(dates)}
    assets = [str(a) for a in (result.get("assets") or result.get("roots") or ([result["underlying"]] if result.get("underlying") else []))][:40]
    ctx = {"assets": assets, "start": dates[0], "end": dates[-1], "frequency": "daily", "data_source": source, "dataset_version": fingerprint(kind, assets, dates[0], dates[-1], source)}
    notes = list(result.get("notes") or [])[:4]
    notes.append("The option market is simulated: premiums, quotes and hedging fills are model outputs, not trades that could have been made." if kind == "options"
                 else "A simulation on saved data: costs are modelled, fills are not.")
    return {"kind": kind, "title": str(result.get("strategy") or result.get("name") or f"{kind} strategy")[:120], "evidence": "simulated" if simulated else "historical_simulation", "data_kind": dk,
            "context": ctx, "config": {"strategy": result.get("strategy"), "params": result.get("params"), "settings": result.get("settings")}, "metrics": metrics,
            "series": {"dates": dates, "equity": equity, "benchmarks": benches}, "refs": {"job": job_id, "code_version": code_version, "dataset": ctx["dataset_version"]}, "notes": notes}


def factor_study_artifact(result: dict, *, job_id: str = "", code_version: str = "unknown") -> dict:
    """The artifact of a finished factor study (the Alpha research workflow, stages 1 to 3). It has no return series, so a comparison lists it and checks that it is comparable with others (same assets, dates,
    horizon, kind of factor and point-in-time assumptions) without measuring it. The numbers kept are the corrected ones: how many factors were tried, how many survive the multiple-testing correction, and how the
    best combination did on the test period nothing was fitted on."""
    for key in ("comparison", "combination", "universe", "assets"):
        if key not in result:
            raise StoreError("this job did not produce a factor study")
    cmp, comb, uni = result["comparison"], result["combination"], result["universe"]
    rows = cmp["table"]
    oos = {k: v for k, v in comb["out_of_sample"].items() if not k.startswith("single:")}
    best = max(oos.items(), key=lambda kv: kv[1]["mean"] if kv[1].get("mean") is not None else -9, default=(None, {}))
    fund = result.get("fundamental") or None
    ctx = {"assets": list(result["assets"]), "start": result["first"], "end": result["last"], "frequency": "daily", "data_source": str(uni.get("source") or "platform"),
           "dataset_version": fingerprint(result["assets"], result["first"], result["last"], uni.get("source"))}
    metrics = {"n_trials": float(cmp["n_tested"]), "n_tested": float(cmp["n_tested"]), "survive_holm": float(sum(1 for r in rows if r.get("survives_holm"))), "survive_bh": float(sum(1 for r in rows if r.get("survives_bh"))),
               "best_rank_ic": float(max((abs(r["rank_ic"]) for r in rows if r.get("rank_ic") is not None), default=0.0)), "test_days": float(comb["out_of_sample"][next(iter(comb["out_of_sample"]))].get("n_test_dates", 0))}
    if best[0] is not None and best[1].get("mean") is not None:
        metrics["best_combination_test_ic"] = float(best[1]["mean"])
        metrics["combination_improves"] = float(bool(comb["verdicts"].get(best[0], {}).get("improves_on_baselines")))
    config = {"horizon": int(result["horizon"]), "factors": [r["factor"] for r in rows], "best_combination": best[0], "uses_fundamentals": bool(fund), "min_assets_per_date": int(result["discovery"].get("min_assets", 0))}
    notes = ["Rank ICs and their intervals are over the whole period; the combinations were fitted on the first part and judged on the rest. The correction for multiple testing counts every factor tried, including the ones not kept.",
             "A study has no return series: it is kept so that it can be found again and checked for comparability, not ranked by performance."]
    if fund:
        pit = fund["pit"]
        config["fundamental"] = {"database": fund["database"], "basis": fund["basis"], "pit_status": pit["verdict"]["status"], "assumed_dates": int(pit["estimated"]), "periods": int(pit["periods"]), "pit_settings": fund["settings"]}
        notes += list(pit.get("assumptions") or [])
        notes.append(f"Fundamental factors were admitted after a point-in-time check: {pit['verdict']['status'].replace('_', ' ')}.")
    return {"kind": "factor_study", "title": f"Factor study: {len(rows)} factors, {result['horizon']}-day horizon"[:120], "evidence": "historical_simulation", "data_kind": data_kind(str(uni.get("source", "")), None), "context": ctx,
            "config": config, "metrics": metrics, "refs": {"job": job_id, "code_version": code_version, "dataset": ctx["dataset_version"]}, "notes": notes}


def forecast_artifact(result: dict, *, job_id: str = "", code_version: str = "unknown") -> dict:
    """The artifact of a finished forecast evaluation: the walk-forward scores of each method against the random walk on the dates every method forecast. Like a factor study it has no return series."""
    for key in ("evaluation", "verdict", "ticker", "horizon"):
        if key not in result:
            raise StoreError("this job did not produce a forecast evaluation")
    ev, hz = result["evaluation"], str(result["horizon"])
    common = (ev.get("common") or {}).get(hz) or {}
    methods = [m for m in (common.get("methods") or {}).items() if m[0] != ev["baseline"] and m[1].get("skill_common") is not None]
    best = max(methods, key=lambda kv: kv[1]["skill_common"], default=(None, {}))
    o, uni = ev["origins"], result.get("universe") or {}
    ctx = {"assets": [result["ticker"]], "start": o["first"], "end": o["last"], "frequency": "daily", "data_source": str(uni.get("source") or "platform"),
           "dataset_version": fingerprint([result["ticker"]], o["first"], o["last"], o["history_days"], uni.get("source"))}
    metrics = {"n_trials": float(len(methods)), "origins": float(o["count"]), "common_origins": float(common.get("n_common", 0)), "same_dates": float(bool(common.get("same_dates")))}
    if best[0] is not None:
        metrics["best_skill_common"] = float(best[1]["skill_common"])
        if best[1].get("p_common_adjusted") is not None:
            metrics["best_adjusted_p"] = float(best[1]["p_common_adjusted"])
        if best[1].get("coverage90_common") is not None:
            metrics["best_coverage90"] = float(best[1]["coverage90_common"])
    conc = result.get("conclusions") or {}
    config = {"horizon": int(result["horizon"]), "methods": [m[0] for m in methods], "effort": result.get("effort"), "best_method": best[0], "step_days": o["step_days"], "refit_days": o["refit_days"], "paths": o["paths"]}
    notes = [f"Walk-forward: every forecast used only the prices before its date. The skill is measured on the {common.get('n_common', 0)} dates all methods forecast, against the random walk with the spread of the last year.",
             "Skill is not a trading result: a forecast can describe the spread of returns well and make no money."] + [i["text"] for i in conc.get("items", []) if i.get("status") == "caution"][:4]
    return {"kind": "forecast", "title": f"Forecast evaluation: {result['ticker']}, {result['horizon']}-day horizon"[:120], "evidence": "walk_forward", "data_kind": data_kind(str(uni.get("source", "")), None), "context": ctx,
            "config": config, "metrics": metrics, "refs": {"job": job_id, "code_version": code_version, "dataset": ctx["dataset_version"]}, "notes": notes}


def trading_study_artifact(result: dict, *, job_id: str = "", code_version: str = "unknown") -> dict:
    """The artifact of a finished Alpha Lab trading study: the calibrated alpha of a signal traded under four policies, and the implementation waterfall from paper Sharpe to the realised book. It has no return
    series (the study reports the Sharpe ratios and costs of each step), so a comparison lists it and checks that it is comparable but does not measure it."""
    for key in ("waterfall", "policies", "config", "universe"):
        if key not in result:
            raise StoreError("this job did not produce a trading study")
    wf, pol, cfg, uni = result["waterfall"], result["policies"], result["config"], result["universe"]
    sharpe = lambda rows, name: next((r.get("sharpe") for r in rows if r.get("index") == name), None)
    metrics = {"n_trials": float(len(pol)), "days": float(result["days"]), "paper_sharpe": _num_or_none(wf[0].get("sharpe")) if wf else None, "realised_sharpe": _num_or_none(wf[-1].get("sharpe")) if wf else None}
    for r in pol:
        if r.get("sharpe") is not None:
            metrics[f"sharpe_{r.get('index')}"] = _num_or_none(r.get("sharpe"))
    metrics = {k: v for k, v in metrics.items() if v is not None}
    ctx = {"assets": list(uni.get("tickers", [])), "start": result["first"], "end": result["last"], "frequency": "daily", "data_source": str(uni.get("source") or "platform"),
           "costs": {"cost_bps": float(cfg.get("cost_bps") or 0.0)}, "dataset_version": fingerprint(uni.get("tickers"), result["first"], result["last"], uni.get("source"))}
    return {"kind": "trading_study", "title": f"Trading study: {result['label']}"[:120], "evidence": "historical_simulation", "data_kind": data_kind(str(uni.get("source", "")), None), "context": ctx,
            "config": {"signal": result["signal"], "horizon": int(result["horizon"]), "policies": [r.get("index") for r in pol], **{k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}},
            "metrics": metrics, "refs": {"job": job_id, "code_version": code_version, "dataset": ctx["dataset_version"]}, "costs": {k: cfg.get(k) for k in ("cost_bps", "impact", "lag")},
            "validation": {"point_in_time": "the alpha is calibrated on past data only (rolling window); trading starts after a one-year warm-up", "waterfall": wf},
            "notes": ["The study reports the Sharpe ratio of the same forecasts as each real-world step is added; a signal built from prices is an example of the interface and not a recommendation.",
                      "A study has no return series: it is listed and checked for comparability, not ranked by performance."] + list(result.get("notes", []))[:2]}


def _num_or_none(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)
