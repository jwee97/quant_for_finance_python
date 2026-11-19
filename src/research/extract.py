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
    return {"kind": "ml_experiment", "title": str(report.get("name") or "Experiment")[:120], "evidence": "walk_forward", "data_kind": kind, "context": ctx,
            "config": {"model": (cfg.get("model") or {}), "dataset": (cfg.get("dataset") or {}), "split": (cfg.get("split") or {}), "signal": (cfg.get("signal") or {}), "backtest": (cfg.get("backtest") or {}),
                       "segments": (report.get("segments") or {}).get("plan"), "n_assets": ds.get("n_assets"), "n_features": ds.get("n_features"), "target": ds.get("target"), "horizon": ds.get("horizon")},
            "metrics": metrics, "series": {"dates": dates, "equity": _clean_series(equity.to_numpy()), "benchmarks": benches, "extra": extra},
            "refs": {"ml_experiment": str(report.get("id") or ""), "config_hash": str(report.get("config_hash") or ""), "dataset": str(report.get("dataset_key") or ""), "code_version": code_version}, "notes": notes}


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
