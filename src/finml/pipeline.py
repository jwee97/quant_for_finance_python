"""The financial machine-learning research pipeline, end to end (Lopez de Prado, *Advances in Financial Machine Learning*).

    event sampling (CUSUM)  ->  triple-barrier labels with a primary side  ->  uniqueness-aware weights  ->  purged / combinatorial cross-validation
    ->  meta-labelling  ->  probability-based bet sizing  ->  cost-aware backtest  ->  evidence (PSR, DSR, CPCV distribution, untouched holdout)

On ONE daily price series. Each step is a module of this package; this file only connects them, and decides what is reported so that a flattering result cannot hide behind an unflattering one:

* The last ``holdout`` share of the events is **never** used to fit, tune or choose anything. Training events whose labels reach into it are dropped. It is scored once, after everything else.
* Cross-validation is purged by each label's own span and embargoed. CPCV gives a *distribution* of out-of-sample paths, not one path. The CPCV research evaluation is not a prospective test, and is labelled so.
* The meta-labelled strategy is compared with the primary strategy it filters, with the same signals and the same costs, with a *random filter that skips as many signals*, and with buy-and-hold of the instrument. When it does
  not do better, the report says so.
* The Sharpe ratio is deflated for the number of configurations the researcher says were tried (``n_trials``). The default of 1 means "this is the only one tried", which is rarely true; the report says what was assumed.
* Position = the average of the sizes of the bets open that day, skipped signals counting as zero, so that the primary and the filtered strategies are scaled the same way. Costs are paid on every change of the position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.finml import events as ev
from src.finml import importance as imp
from src.finml import sampling as sp
from src.finml import sizing as sz
from src.finml.cv import cpcv_event_splits, leakage_check, purged_folds
from src.ops.cv import cpcv_paths
from src.stats import inference as inf

PRIMARIES = ("momentum", "mean_reversion", "ma_cross", "always_long")
DEFAULTS = {"primary": "momentum", "lookback": 50, "fast": 10, "cusum_k": 1.0, "vol_span": 50, "pt": 1.0, "sl": 1.0, "vertical_bars": 10, "entry_lag": 1, "same_bar": "stop", "cost_bps": 5.0,
            "weighting": "uniqueness", "decay": 1.0, "model": "logistic", "bagging": "none", "n_bags": 8, "cv_splits": 5, "embargo": 5, "holdout": 0.2, "threshold": 0.55, "step": 0.0,
            "cpcv_groups": 6, "cpcv_test": 2, "n_trials": 1, "importance": True, "seed": 0, "min_events": 120}
WEIGHTINGS = ("none", "uniqueness", "attribution")
MODELS = ("logistic", "forest")
BAGGING = ("none", "sequential")


class PipelineError(ValueError):
    """The request cannot be run honestly (too few events, an unknown option, no price history)."""


# ---------------------------------------------------------------------------------------------------- pieces
def _primary_side(close: pd.Series, cfg: dict) -> pd.Series:
    k, f = int(cfg["lookback"]), int(cfg["fast"])
    if cfg["primary"] == "momentum":
        return np.sign(close / close.shift(k) - 1.0)
    if cfg["primary"] == "mean_reversion":
        return -np.sign(close / close.shift(k) - 1.0)
    if cfg["primary"] == "ma_cross":
        return np.sign(close.rolling(f).mean() - close.rolling(k).mean())
    return pd.Series(1.0, index=close.index)


def _features(close: pd.Series, side: pd.Series, vol: pd.Series, events: pd.DatetimeIndex) -> pd.DataFrame:
    """Everything a meta-model may look at, each computed from the closes up to and including the event row (the labels never look back from the future, and neither do the features)."""
    lr = np.log(close).diff()
    rsi_up = lr.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    rsi_dn = (-lr.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    f = pd.DataFrame({"ret_1": lr, "ret_5": np.log(close).diff(5), "ret_20": np.log(close).diff(20), "vol": vol, "vol_ratio": vol / vol.rolling(100, min_periods=30).mean(),
                      "dist_ma20": close / close.rolling(20).mean() - 1.0, "rsi": rsi_up / (rsi_up + rsi_dn + 1e-12), "side": side}, index=close.index)
    f["side_x_ret_5"] = f["side"] * f["ret_5"]
    f["side_x_ret_20"] = f["side"] * f["ret_20"]
    return f.loc[events].replace([np.inf, -np.inf], np.nan)


def _model(cfg: dict):
    if cfg["model"] == "forest":
        return RandomForestClassifier(n_estimators=200, max_depth=4, min_weight_fraction_leaf=0.03, class_weight="balanced_subsample", random_state=cfg["seed"], n_jobs=1)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))


def _fit_predict(cfg: dict, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, labels: pd.DataFrame, n_rows: int, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """P(the primary signal pays) for the ``test`` rows from a model fitted on the ``train`` rows alone. Medians and scaling are fitted inside the training rows. With sequential bagging the model is fitted
    on several sequential-bootstrap samples of the training labels and the probabilities are averaged."""
    base = float(np.average(y[train], weights=w[train])) if len(train) else 0.5
    if len(train) < 20 or len(np.unique(y[train])) < 2:
        return np.full(len(test), base)
    med = X.iloc[train].median()
    Xtr, Xte = X.iloc[train].fillna(med), X.iloc[test].fillna(med)
    kind = cfg["model"]
    sw_key = "logisticregression__sample_weight" if kind == "logistic" else "sample_weight"
    if cfg["bagging"] == "sequential":
        sub = labels.iloc[train]
        ps = []
        for b in range(int(cfg["n_bags"])):
            draw = sp.sequential_bootstrap(sub, n_rows, seed=int(cfg["seed"]) + b)
            ytr = y[train][draw]
            if len(np.unique(ytr)) < 2:
                continue
            m = _model(cfg)
            m.fit(Xtr.iloc[draw], ytr)                                                         # the draw already favours unique labels: no extra weights
            ps.append(m.predict_proba(Xte)[:, list(m.classes_).index(1)])
        return np.mean(ps, axis=0) if ps else np.full(len(test), base)
    m = _model(cfg)
    m.fit(Xtr, y[train], **{sw_key: w[train]})
    return m.predict_proba(Xte)[:, list(m.classes_).index(1)]


def _daily_positions(labels: pd.DataFrame, sizes: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    first = labels["pos_entry"].to_numpy()
    return sz.average_active(pd.Series(sizes), first, labels["pos1"].to_numpy(), index)


def _daily_pnl(position: pd.Series, returns: pd.Series, cost: float) -> pd.Series:
    return (position * returns - cost * position.diff().abs().fillna(position.abs())).rename("pnl")


def _perf(r: pd.Series, ppy: int = 252) -> dict:
    r = r.dropna()
    if len(r) < 5 or not r.std() > 0:
        return {"days": int(len(r)), "sharpe": None, "annual_return": None, "annual_vol": None, "max_drawdown": None, "total_return": float(r.sum()) if len(r) else 0.0}
    eq = (1 + r).cumprod()
    return {"days": int(len(r)), "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(ppy)), "annual_return": float(r.mean() * ppy), "annual_vol": float(r.std(ddof=1) * np.sqrt(ppy)),
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "total_return": float(eq.iloc[-1] - 1)}


def _evidence(r: pd.Series, n_trials: int) -> dict:
    r = r.dropna()
    if len(r) < 30 or not r.std() > 0:
        return {"psr": None, "dsr": None, "min_track_record_days": None}
    d = inf.deflated_sharpe_ratio(r.to_numpy(), n_trials=max(1, int(n_trials)))
    mtrl = inf.min_track_record_length(r.to_numpy(), 0.0, 0.95)
    return {"psr": float(inf.probabilistic_sharpe_ratio(r.to_numpy(), 0.0)), "dsr": float(d["probability"]), "dsr_benchmark_sharpe_annual": float(d["benchmark"] * np.sqrt(252)),
            "min_track_record_days": None if not np.isfinite(mtrl) else float(mtrl)}


def _block_bootstrap_gain(diff: np.ndarray, block: int = 10, reps: int = 1000, seed: int = 0) -> dict:
    """Circular block bootstrap of the mean daily difference between two return series (the filtered minus the primary): the share of resamples in which it is positive, and an interval for it, annualised."""
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2 * block:
        return {"p_positive": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    means = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n, k)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        means[i] = diff[idx.ravel()[:n]].mean()
    return {"p_positive": float((means > 0).mean()), "low": float(np.quantile(means, 0.025) * 252), "high": float(np.quantile(means, 0.975) * 252)}


def _reliability(p: np.ndarray, y: np.ndarray, bins: int = 8) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-12
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            out.append({"predicted": float(p[m].mean()), "observed": float(y[m].mean()), "n": int(m.sum())})
    return out


# ---------------------------------------------------------------------------------------------------- the run
def run_pipeline(close: pd.Series, config: dict | None = None, high: pd.Series | None = None, low: pd.Series | None = None, progress=None) -> dict:
    cfg = {**DEFAULTS, **(config or {})}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise PipelineError(f"unknown settings: {sorted(unknown)}")
    for key, allowed in (("primary", PRIMARIES), ("weighting", WEIGHTINGS), ("model", MODELS), ("bagging", BAGGING), ("same_bar", ev.SAME_BAR)):
        if cfg[key] not in allowed:
            raise PipelineError(f"{key} must be one of {allowed}")
    if not 0.0 <= float(cfg["holdout"]) < 0.5:
        raise PipelineError("holdout must be between 0 and 0.5")
    if not 0.5 <= float(cfg["threshold"]) < 1.0:
        raise PipelineError("threshold must be at least 0.5 (below it the meta-model would take signals it thinks are more likely wrong than right) and below 1")
    say = progress or (lambda *_: None)
    close = close.astype(float).dropna()
    close = close[close > 0]
    if len(close) < 400:
        raise PipelineError("at least 400 daily closes are needed")
    n = len(close)
    index = close.index
    ret = close.pct_change().fillna(0.0)
    cost = float(cfg["cost_bps"]) / 1e4
    warnings: list[str] = []

    say("Choosing events", 0.05)
    vol = ev.daily_vol(close, int(cfg["vol_span"]))
    events = ev.cusum_events(np.log(close), (vol.shift(0) * float(cfg["cusum_k"])).fillna(np.inf))
    side = _primary_side(close, cfg)
    events = events[side.reindex(events).fillna(0).ne(0).to_numpy()]
    labels = ev.triple_barrier_events(close, events, vol, pt=float(cfg["pt"]), sl=float(cfg["sl"]), vertical_bars=int(cfg["vertical_bars"]), side=side, entry_lag=int(cfg["entry_lag"]),
                                      high=high, low=low, same_bar=cfg["same_bar"], cost=2 * cost)
    if len(labels) < int(cfg["min_events"]):
        raise PipelineError(f"only {len(labels)} labelled events (at least {cfg['min_events']} are needed): lower the CUSUM threshold, use a longer history or a shorter holding limit")
    if high is not None and labels["ambiguous"].mean() > 0.02:
        warnings.append(f"{labels['ambiguous'].mean():.1%} of the labels end on a bar that touched both barriers; the {cfg['same_bar']!r} assumption decided them. Intraday data would resolve it.")
    X = _features(close, side, vol, labels.index)
    y = labels["bin"].to_numpy().astype(int)

    # the untouched holdout: the last ``holdout`` share of the HISTORY (by date, not by number of events, so that nothing that happens inside it can move the line). A development label that reaches into it is dropped,
    # and so is an event that starts before it and ends inside it.
    boundary = int(round(n * (1.0 - float(cfg["holdout"])))) if float(cfg["holdout"]) > 0 else n
    p0, p1 = labels["pos0"].to_numpy(), labels["pos1"].to_numpy()
    dev = np.where(p1 < boundary)[0]
    hold = np.where(p0 >= boundary)[0]
    k_hold = len(hold)
    if len(dev) < 80:
        raise PipelineError("too few development events once the holdout is set aside")
    if float(cfg["holdout"]) > 0 and k_hold < 20:
        raise PipelineError("too few events in the holdout: use a larger holdout share or a longer history")

    # weights: how much independent information each label carries (from the spans only, never from outcomes), normalised within the development events so that the holdout cannot influence them
    quality = sp.sample_quality(labels, n)
    u = sp.uniqueness(labels, n).to_numpy()
    w = {"none": np.ones(len(labels)), "uniqueness": u, "attribution": sp.return_attribution_weights(labels, ret).to_numpy()}[cfg["weighting"]].astype(float).copy()
    if cfg["decay"] != 1.0:
        w[dev] = sp.time_decay(pd.Series(w[dev], index=labels.index[dev]), float(cfg["decay"])).to_numpy()
    w = w / w[dev].mean()
    ldev = labels.iloc[dev]
    Xd, yd, wd = X.iloc[dev], y[dev], w[dev]

    # purged k-fold out-of-sample probabilities
    say("Cross-validating (purged)", 0.25)
    oof = np.full(len(dev), np.nan)
    leak_free = True
    for tr, te in purged_folds(ldev, int(cfg["cv_splits"]), int(cfg["embargo"])):
        leak_free &= leakage_check(ldev, tr, te)
        oof[te] = _fit_predict(cfg, Xd, yd, wd, ldev, n, tr, te)
    ok = np.isfinite(oof)
    base_rate = float(np.average(yd, weights=wd))
    valid = {"auc": float(roc_auc_score(yd[ok], oof[ok])) if len(np.unique(yd[ok])) == 2 else None,
             "log_loss": float(log_loss(yd[ok], np.clip(oof[ok], 1e-6, 1 - 1e-6), sample_weight=wd[ok], labels=[0, 1])),
             "base_rate_log_loss": float(log_loss(yd[ok], np.full(ok.sum(), base_rate), sample_weight=wd[ok], labels=[0, 1])), "brier": float(brier_score_loss(yd[ok], oof[ok], sample_weight=wd[ok])),
             "base_rate": base_rate, "reliability": _reliability(oof[ok], yd[ok]), "purging_leak_free": bool(leak_free), "folds": int(cfg["cv_splits"])}

    # strategy variants from probabilities
    thr = float(cfg["threshold"])

    def sizes_for(variant: str, p: np.ndarray, lab: pd.DataFrame, mask: np.ndarray | None = None) -> np.ndarray:
        s = lab["side"].to_numpy().astype(float)
        if variant == "primary":
            return s
        if variant == "meta_filter":
            return s * (p >= thr)
        if variant == "random_filter":
            return s * mask
        m = sz.size_from_probability(p, s, floor=thr)
        return sz.discretize(m, float(cfg["step"])) if float(cfg["step"]) > 0 else m

    def evaluate(lab: pd.DataFrame, p: np.ndarray, rows: slice) -> dict:
        out = {}
        series = {}
        take = p >= thr
        rng = np.random.default_rng(int(cfg["seed"]))
        for v in ("primary", "meta_filter", "meta_sized"):
            pos = _daily_positions(lab, sizes_for(v, p, lab), index)
            series[v] = _daily_pnl(pos, ret, cost).iloc[rows]
            out[v] = {**_perf(series[v]), "signals": int(len(lab)), "taken": int(len(lab) if v == "primary" else (take.sum() if v == "meta_filter" else (np.abs(sizes_for(v, p, lab)) > 0).sum())),
                      "turnover_per_day": float(pos.iloc[rows].diff().abs().mean())}
        rnd = []
        for _ in range(40):
            mask = np.zeros(len(lab), dtype=bool)
            mask[rng.choice(len(lab), int(take.sum()), replace=False)] = True
            rnd.append(_perf(_daily_pnl(_daily_positions(lab, sizes_for("random_filter", p, lab, mask), index), ret, cost).iloc[rows])["sharpe"] or 0.0)
        out["random_filter"] = {"sharpe": float(np.mean(rnd)), "sharpe_p05": float(np.quantile(rnd, 0.05)), "sharpe_p95": float(np.quantile(rnd, 0.95)), "taken": int(take.sum()), "signals": int(len(lab))}
        bh = ret.iloc[rows]
        out["buy_and_hold"] = _perf(bh)
        series["buy_and_hold"] = bh
        gain = _block_bootstrap_gain((series["meta_filter"] - series["primary"]).to_numpy(), seed=int(cfg["seed"]))
        beat = out["meta_filter"]["sharpe"] is not None and out["random_filter"]["sharpe_p95"] is not None and out["meta_filter"]["sharpe"] > out["random_filter"]["sharpe_p95"]
        out["comparison"] = {"filter_minus_primary_sharpe": None if None in (out["meta_filter"]["sharpe"], out["primary"]["sharpe"]) else out["meta_filter"]["sharpe"] - out["primary"]["sharpe"],
                             "bootstrap": gain, "beats_random_filters_p95": bool(beat)}
        out["evidence"] = {v: _evidence(series[v], int(cfg["n_trials"])) for v in ("primary", "meta_filter", "meta_sized")}
        return out, series

    first_dev = int(ldev["pos_entry"].iloc[0]) + 1
    last_dev = int(ldev["pos1"].max())
    say("Scoring development window", 0.45)
    res_dev, ser_dev = evaluate(ldev, oof, slice(first_dev, last_dev + 1))

    # CPCV: a distribution of out-of-sample paths (research evaluation: not a prospective test)
    say("Combinatorial purged cross-validation", 0.55)
    splits = cpcv_event_splits(ldev, int(cfg["cpcv_groups"]), int(cfg["cpcv_test"]), int(cfg["embargo"]))
    preds = {}
    for i, s in enumerate(splits):
        preds[i] = pd.Series(_fit_predict(cfg, Xd, yd, wd, ldev, n, s["train"], s["test"]), index=s["test"])
        say("Combinatorial purged cross-validation", 0.55 + 0.25 * (i + 1) / len(splits))
    edges = np.linspace(0, len(ldev), int(cfg["cpcv_groups"]) + 1).astype(int)
    path_sharpes, path_primary = [], []
    for path in cpcv_paths(splits, int(cfg["cpcv_groups"])):
        p = np.full(len(ldev), np.nan)
        for si, g in path:
            idx = np.arange(edges[g], edges[g + 1])
            p[idx] = preds[si].loc[idx].to_numpy()
        pos = _daily_positions(ldev, sizes_for("meta_filter", p, ldev), index)
        path_sharpes.append(_perf(_daily_pnl(pos, ret, cost).iloc[first_dev:last_dev + 1])["sharpe"])
        path_primary.append(res_dev["primary"]["sharpe"])
    ps = np.array([x for x in path_sharpes if x is not None], dtype=float)
    cp = {"n_paths": int(len(path_sharpes)), "n_splits": len(splits), "sharpes": [None if x is None else float(x) for x in path_sharpes], "mean": float(ps.mean()) if len(ps) else None,
          "std": float(ps.std(ddof=1)) if len(ps) > 1 else None, "prob_negative": float((ps < 0).mean()) if len(ps) else None, "primary_sharpe": res_dev["primary"]["sharpe"],
          "fraction_above_primary": float((ps > (res_dev["primary"]["sharpe"] or 0)).mean()) if len(ps) else None,
          "note": "research evaluation by combinatorial purged cross-validation of the filtered strategy; the same development events as the cross-validation above. It is not a prospective test."}

    # feature importance on the development events
    importance = None
    if cfg["importance"]:
        say("Feature importance", 0.82)
        Xi = Xd.fillna(Xd.median())
        groups = imp.feature_clusters(Xi, 0.5)
        importance = {"mdi": imp.mdi(Xi, pd.Series(yd), wd, seed=int(cfg["seed"])).reset_index(names="feature").to_dict("records"),
                      "mda": imp.mda(Xi, pd.Series(yd), ldev, wd, int(cfg["cv_splits"]), int(cfg["embargo"]), n_repeats=2, clusters=groups, seed=int(cfg["seed"])).reset_index(names="cluster").to_dict("records"),
                      "sfi": imp.sfi(Xi, pd.Series(yd), ldev, wd, int(cfg["cv_splits"]), int(cfg["embargo"]), seed=int(cfg["seed"])).reset_index(names="feature").to_dict("records"), "clusters": groups}

    # the holdout, scored once
    res_hold = None
    ser_hold = {}
    if k_hold:
        say("Scoring the untouched holdout", 0.92)
        lh = labels.iloc[hold]
        ph = _fit_predict(cfg, X, y, w, labels, n, dev, hold)
        rows = slice(int(lh["pos_entry"].iloc[0]) + 1, int(lh["pos1"].max()) + 1)
        res_hold, ser_hold = evaluate(lh, ph, rows)
        res_hold["window"] = {"start": str(index[rows.start].date()), "end": str(index[min(rows.stop - 1, n - 1)].date())}
        res_hold["auc"] = float(roc_auc_score(y[hold], ph)) if len(np.unique(y[hold])) == 2 else None

    dev_gain = res_dev["comparison"]["filter_minus_primary_sharpe"]
    hold_gain = None if res_hold is None else res_hold["comparison"]["filter_minus_primary_sharpe"]
    shown = (res_hold or res_dev)["comparison"]
    demonstrated = bool(res_hold is not None and hold_gain is not None and dev_gain is not None and hold_gain > 0 and dev_gain > 0 and (shown["bootstrap"]["p_positive"] or 0) >= 0.9 and shown["beats_random_filters_p95"])
    if demonstrated:
        verdict = "The meta-model's filter beat the primary strategy on the development folds and on the untouched holdout, and beat the random filters that skip as many signals. That is evidence for this configuration on this history; it is not a forecast."
    elif res_hold is None:
        verdict = "No holdout was set aside, so the comparison is development evidence only."
    else:
        verdict = "The meta-model's filter was not shown to improve on the primary strategy: it did not do better than it on both the development folds and the holdout with a convincing margin over random filters. A negative result is a result."
    notes = [f"A cost of {cfg['cost_bps']} bp per unit traded is charged on every change of position (and twice that on the label's round trip).",
             f"Configurations tried (n_trials) was taken as {cfg['n_trials']}: the deflated Sharpe ratio is only as honest as that number.",
             "Position is the average of the sizes of the bets open that day (skipped signals count as zero)."]
    pack = lambda s: {"dates": [str(d.date()) for d in s.index], "values": [float(x) for x in s.to_numpy()]}  # noqa: E731
    equity = lambda s: pack((1 + s).cumprod())  # noqa: E731
    return {"config": cfg, "labels": {**ev.label_summary(labels), "events_total_cusum": int(len(events)), "first": str(labels.index[0].date()), "last": str(labels.index[-1].date()), "ambiguous_share": float(labels["ambiguous"].mean())},
            "quality": quality, "validation": valid, "cpcv": cp, "development": res_dev, "holdout": res_hold, "importance": importance, "verdict": verdict, "meta_labelling_demonstrated": demonstrated, "warnings": warnings, "notes": notes,
            "split": {"development_events": int(len(dev)), "holdout_events": int(k_hold), "dropped_for_overlap": int(len(labels) - len(dev) - k_hold), "holdout_starts": None if not k_hold else str(index[boundary].date())},
            "series": {"development": {k: equity(v) for k, v in ser_dev.items()}, "holdout": {k: equity(v) for k, v in ser_hold.items()}}}
