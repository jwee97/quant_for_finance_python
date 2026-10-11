"""One experiment end to end: data, walk-forward training, predictions, positions, backtest, evaluation, overfitting diagnostics, explanation, robustness, and the record of it all.

``run_experiment`` is the single door. Nothing in the platform trains a model any other way, which is the point: a model's score here is always the score of a STRATEGY that was costed and lagged by the
platform's engine and judged on segments of time it was not fitted on, with the number of times the data has been looked at in the report next to it.

The report (a plain JSON-able dictionary) is what the dashboard shows and what is saved as ``report.json``; ``Experiment`` keeps the live objects (dataset, fold models, predictions, weights) for the
tests that need them: explanations, robustness, forward testing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import explainability as xp
from . import robustness as rb
from .backtest import equal_weight_benchmark, label_days, run_backtest, trim_to_signal
from .config import DEFAULTS, build_key, config_hash, dataset_key, experiment_config, merge
from .datasets import Dataset, build_dataset, resolve_bundle
from .models import Model
from .registry import Registry
from .signals import build_weights
from .targets import forward_return, is_classification
from .training import TrainResult, fit_final, walk_forward_train
from .visualization import equity_payload, importance_bars, ic_series_payload, jsonable, prediction_distribution, render_html


class ExperimentError(ValueError):
    """An experiment that cannot run as configured (too little data, an unavailable model); the message says why."""


@dataclass
class Experiment:
    id: str
    cfg: dict
    dataset: Dataset
    train: TrainResult
    scores: pd.DataFrame
    weights: pd.DataFrame
    net: pd.Series
    gross: pd.Series
    costs: pd.Series
    turnover: pd.Series
    benchmark: pd.Series
    insample_net: pd.Series | None
    labels: pd.Series
    first_signal: pd.Timestamp
    final_model: Model | None
    report: dict
    context: rb.Context | None = None
    metrics: dict = field(default_factory=dict)


def trailing_vol(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Annualised volatility known at each date (for the volatility-scaled signal)."""
    return returns.rolling(window, min_periods=window // 2).std(ddof=1) * np.sqrt(252.0)


def confidence_from_scores(scores: pd.DataFrame, probabilities: pd.DataFrame | None) -> pd.DataFrame:
    """A 0..1 confidence for each score: for a classifier ``|2p - 1|``; otherwise how far the score sits from the cross-section's centre in units of its spread, capped at two."""
    if probabilities is not None:
        return (probabilities.sub(0.5).abs() * 2.0).reindex_like(scores)
    z = scores.sub(scores.mean(axis=1), axis=0).div(scores.std(axis=1, ddof=0).replace(0.0, np.nan), axis=0)
    return (z.abs().clip(upper=2.0) / 2.0)


def regime_series(ds: Dataset, on_states: list) -> pd.Series | None:
    """True on the dates the dataset's regime state is one of ``on_states`` (None if the dataset has no regime feature)."""
    if "regime_state" not in ds.X.columns:
        return None
    state = ds.X["regime_state"].groupby(level=0).first().reindex(ds.dates)
    return state.isin([float(s) for s in on_states]).where(state.notna())


def prepare(cfg: dict, bundle=None, config=None, builder=None, dataset: Dataset | None = None, info: dict | None = None, progress=None) -> tuple[Dataset, dict]:
    """The dataset for a configuration: the one given, or built from ``bundle`` (or from the universe named in the configuration)."""
    if dataset is not None:
        return dataset, info or {"source": dataset.meta.get("source", "platform"), "label": dataset.meta.get("label", "dataset")}
    uni = cfg["dataset"]["universe"]
    if bundle is None:
        try:
            bundle, info = resolve_bundle(config, uni, builder)
        except (FileNotFoundError, KeyError) as exc:
            raise ExperimentError(f"the data for universe '{uni}' is not available: {exc}") from None
        except ValueError as exc:
            raise ExperimentError(str(exc)) from None
    elif info is None:
        info = {"source": "user", "label": getattr(bundle, "name", "bundle")}
    if progress:
        progress("building the dataset")
    try:
        ds = build_dataset(bundle, cfg["dataset"], cfg["name"], source=info)
    except ValueError as exc:
        raise ExperimentError(str(exc)) from None
    return ds, info


def _segments_ic(ic: pd.Series, labels: pd.Series, horizon: int) -> dict:
    lab = labels.reindex(ic.index)
    return {seg: ev.ic_summary(ic[lab == seg], horizon) for seg in ("validation", "test", "forward") if (lab == seg).sum() >= 5}


def _fold_ics(train: TrainResult, ic: pd.Series, ds: Dataset) -> list[float]:
    out = []
    for f in train.folds:
        if f.get("skipped"):
            continue
        part = ic.loc[pd.Timestamp(f["test_start"]):pd.Timestamp(f["test_end"])].dropna()
        out.append(float(part.mean()) if len(part) else float("nan"))
    return out


def run_experiment(overrides: dict | None = None, bundle=None, config=None, builder=None, registry: Registry | None = None, progress=None, save: bool = True, dataset: Dataset | None = None,
                   info: dict | None = None, parent_id: str | None = None, explain: bool = True, final_model: bool = True, robustness: bool | None = None, trials_extra: int = 0,
                   experiment_id: str | None = None, stop_after: str | None = None) -> Experiment:
    """Run one experiment. ``overrides`` is merged over the platform configuration (``ml.config.experiment_config``). With a ``registry`` and ``save`` the artifacts are written and the run recorded.

    ``explain`` computes permutation importance; ``final_model`` fits the model a live prediction would use; ``robustness`` overrides the configuration's switch for the ten stress tests;
    ``trials_extra`` adds to the recorded trial count (a search that ran many variants passes how many). ``stop_after="validation"`` is the search mode: the test and forward segments are not
    predicted at all, so nothing a search does can depend on them."""
    t0 = time.perf_counter()
    cfg = experiment_config(overrides)
    progress = progress or (lambda msg: None)
    ds, info = prepare(cfg, bundle, config, builder, dataset, info, progress)
    progress("training walk-forward")
    try:
        train = walk_forward_train(ds, cfg["model"], cfg["split"], cfg["seed"], progress=progress, stop_after=stop_after)
    except ValueError as exc:
        raise ExperimentError(str(exc)) from None
    return evaluate_run(cfg, ds, train, info, registry, progress, save, parent_id, explain, final_model, robustness, trials_extra, experiment_id, stop_after, t0)


def evaluate_run(cfg: dict, ds: Dataset, train: TrainResult, info: dict, registry: Registry | None = None, progress=None, save: bool = True, parent_id: str | None = None, explain: bool = True,
                 final_model: bool = True, robustness: bool | None = None, trials_extra: int = 0, experiment_id: str | None = None, stop_after: str | None = None, t0: float | None = None) -> Experiment:
    """Everything after the fitting: positions, backtest, segments, information coefficient, overfitting diagnostics, explanation, robustness, the report and (if a registry is given) the record. Separate
    from ``run_experiment`` so that the same predictions can be re-run under different signal and cost settings without refitting (``rebacktest``)."""
    t0 = time.perf_counter() if t0 is None else t0
    progress = progress or (lambda msg: None)
    dkey, chash = dataset_key(cfg), config_hash(cfg)
    lag, horizon = ds.lag, ds.horizon
    scores = train.scores_wide(ds)
    prob_wide = train.probabilities.unstack("asset").reindex(index=ds.dates, columns=ds.assets) if train.probabilities is not None else None
    vol = trailing_vol(ds.returns)
    conf = confidence_from_scores(scores, prob_wide)
    regime_on = regime_series(ds, cfg["signal"].get("regime_on_states", [0]))
    groups = ds.group or ds.asset_class
    progress("building positions and backtesting")
    try:
        weights = build_weights(scores, cfg["signal"], vol, conf, regime_on, groups, ds.investable)
    except ValueError as exc:
        raise ExperimentError(str(exc)) from None
    result = run_backtest(weights, ds.returns, ds.investable, cfg["backtest"], lag, cfg["name"])
    first_signal = scores.dropna(how="all").index[0]
    parts = trim_to_signal(result, first_signal, lag)
    net, gross, costs, turnover = parts["net"], parts["gross"], parts["costs"], parts["turnover"]
    if stop_after:                                                             # the positions of the last day still earn for `lag` more days; nothing later belongs to the run
        cut = net.index.searchsorted(train.plan.dates[train.plan.bounds[stop_after][1] - 1]) + lag + 1
        net, gross, costs, turnover = net.iloc[:cut], gross.iloc[:cut], costs.iloc[:cut], turnover.iloc[:cut]
    labels = label_days(net.index, train.plan.labels())
    bench = equal_weight_benchmark(ds.returns, ds.investable).reindex(net.index)
    # the train segment: the first model's in-sample positions, run through the same engine
    insample_net = None
    s_in = train.scores_wide(ds, "insample")
    if s_in.notna().any().any():
        w_in = build_weights(s_in, cfg["signal"], vol, confidence_from_scores(s_in, None), regime_on, groups, ds.investable)
        r_in = trim_to_signal(run_backtest(w_in, ds.returns, ds.investable, cfg["backtest"], lag, cfg["name"] + "_in_sample"), s_in.dropna(how="all").index[0], lag)["net"]
        last_in = s_in.dropna(how="all").index[-1]
        insample_net = r_in.iloc[: r_in.index.searchsorted(last_in) + 1]
    seg_stats = ev.segment_stats(net, labels, insample_net)
    oos_summary = ev.strategy_summary(net, gross, turnover, costs, bench)
    # information coefficient against the forward return of the target's horizon (a classifier's labels are not returns)
    fwd = forward_return(ds.prices, horizon, lag).reindex(index=ds.dates, columns=ds.assets)
    ic = ev.rank_ic_wide(scores, fwd).dropna()
    ic_all = ev.ic_summary(ic, horizon)
    ic_seg = _segments_ic(ic, label_days(ic.index, train.plan.labels()), horizon)
    fold_ics = _fold_ics(train, ic, ds)
    decay = ev.ic_decay(scores, ds.prices, lag)
    quant = ev.quantile_spread(scores.loc[ic.index], fwd.loc[ic.index])
    y_oos = ds.Y[train.target].reindex(train.predictions.index)
    pq = {}
    plabels = label_days(pd.DatetimeIndex(train.predictions.index.get_level_values(0).unique()), train.plan.labels())
    row_seg = plabels.reindex(train.predictions.index.get_level_values(0)).to_numpy()
    for seg in ("validation", "test", "forward"):
        sel = row_seg == seg
        if sel.sum() < 30:
            continue
        if train.task == "classification":
            pq[seg] = ev.classification_metrics(train.probabilities[sel], y_oos[sel])
        else:
            pq[seg] = ev.regression_metrics(train.predictions[sel], y_oos[sel])
    n_trials = (registry.trial_count(dkey) if registry is not None else 0) + 1 + int(trials_extra)
    tf_net = net[labels.isin(["test", "forward"])]
    over = ev.overfit_report(seg_stats, n_trials, fold_ics, oos_summary.get("gross_sharpe"), oos_summary.get("sharpe"), len(ds.features), len(ds.X), test_forward_returns=tf_net)
    blocks = [ev.sharpe_of(pd.Series(b)) for b in np.array_split(net[labels == "validation"].to_numpy(), 4) if len(b) >= 20]
    over["validation_blocks"] = blocks
    over["instability"] = float(np.nanstd(blocks, ddof=1)) if len(blocks) >= 2 else float("nan")
    imp_table = xp.importance_table(train.importance)
    perm = pd.DataFrame()
    if explain:
        progress("explaining the model")
        try:
            perm = xp.permutation_importance(train, ds, max_folds=5, n_repeats=1, top=20)
        except Exception:                                                              # an explanation that cannot be computed must not lose the experiment
            perm = pd.DataFrame()
    fmodel = None
    if final_model:
        try:
            fmodel = fit_final(ds, cfg["model"], cfg["seed"])
        except ValueError:
            fmodel = None
    exp_id = experiment_id or (registry.new_id(cfg["name"]) if registry is not None else "unsaved")
    ctx = rb.Context(ds, train, cfg, scores, first_signal, net, labels, vol, conf, regime_on)
    rob = None
    if (cfg["robustness"]["enabled"] if robustness is None else robustness):
        rob = rb.run_all(ctx, progress=progress)
    eq = equity_payload(net, bench, labels, insample_net)
    notes = list(info.get("notes", [])) if isinstance(info.get("notes"), list) else []
    if info.get("source") == "simulated":
        notes.append("The data is simulated: this checks the machinery, not the market.")
    report = jsonable({
        "id": exp_id, "name": cfg["name"], "config": cfg, "config_hash": chash, "dataset_key": dkey, "build_key": build_key(cfg), "dataset": ds.summary(), "provenance": {"source": info.get("source", "platform"), "label": info.get("label", ""), "notes": notes},
        "segments": {"plan": train.plan.to_dict(), "stats": seg_stats},
        "metrics": {"oos": oos_summary, "benchmark": ev.strategy_summary(bench)},
        "ic": {"all": ic_all, "by_segment": ic_seg, "decay": decay, "series": ic_series_payload(ic), "by_fold": fold_ics, "quantiles": quant},
        "prediction_quality": pq, "overfit": over,
        "importance": {"table": imp_table.head(40).reset_index().rename(columns={"index": "feature"}) if not imp_table.empty else pd.DataFrame(), "bars": importance_bars(imp_table, 20), "rank_stability": xp.rank_stability(train.importance),
                       "permutation": perm.head(20).reset_index().rename(columns={"index": "feature"}) if not perm.empty else pd.DataFrame()},
        "distribution": prediction_distribution(train.predictions, train.plan.labels()), "equity": eq,
        "positions": {"date": str(weights.index[-1].date()), "weights": {a: float(w) for a, w in weights.iloc[-1].items() if abs(w) > 1e-6}},
        "folds": {"n": len(train.folds), "n_skipped": len(train.skipped), "first": train.folds[0] if train.folds else None, "last": train.folds[-1] if train.folds else None, "refit_every": cfg["split"]["refit_every"],
                  "list": [{k: f.get(k) for k in ("fold", "test_start", "test_end", "n_train", "n_test", "skipped", "seconds")} for f in train.folds]},
        "features": {"n": len(ds.features), "skipped": ds.meta.get("skipped", {}), "names": ds.features}, "robustness": rob,
        "trials": {"dataset_trials": n_trials, "note": "Every experiment and search trial on this dataset is counted; the deflated Sharpe probability deflates the validation Sharpe by this number."},
        "warnings": list(over["warnings"]), "seconds": round(time.perf_counter() - t0, 2), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    metrics = {"oos": {k: oos_summary.get(k) for k in ("sharpe", "cagr", "ann_vol", "max_drawdown", "hit_rate", "ann_turnover", "ann_cost_drag", "gross_sharpe", "total_return", "sortino", "calmar", "n_obs")},
               "ic": {k: ic_all[k] for k in ("mean", "std", "ir", "t", "hit_rate", "n")}, "overfit": {"generalization_score": over["generalization_score"], "psr_test_forward": over["psr_test_forward"],
                                                                                                 "deflated_sharpe_probability": over["deflated_sharpe_probability"], "n_trials": n_trials},
               "robustness": {} if rob is None else {"score": rob["score"], "n_pass": rob["n_pass"], "n_warn": rob["n_warn"], "n_fail": rob["n_fail"]}}
    for seg, s in seg_stats.items():
        metrics[seg] = {k: s[k] for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "n_days", "hit_rate", "total_return")}
    exp = Experiment(exp_id, cfg, ds, train, scores, weights, net, gross, costs, turnover, bench, insample_net, labels, first_signal, fmodel, report, ctx, metrics)
    if registry is not None and save:
        progress("saving")
        save_experiment(registry, exp, info, parent_id, time.perf_counter() - t0)
    report["seconds"] = round(time.perf_counter() - t0, 2)
    return exp


def save_experiment(registry: Registry, exp: Experiment, info: dict, parent_id: str | None = None, seconds: float = 0.0) -> None:
    """Write the artifacts of an experiment and record it in the registry."""
    exp_id, cfg, ds, tr = exp.id, exp.cfg, exp.dataset, exp.train
    d = registry.experiment_dir(exp_id)
    registry.log_experiment(exp_id, cfg, exp.report["config_hash"], exp.report["dataset_key"], exp.report["trials"]["dataset_trials"] - 1, "done", seconds, str(info.get("source", "platform")), parent_id)
    seg = exp.train.plan.labels()
    preds = pd.DataFrame({"score": tr.predictions})
    preds["segment"] = label_days(preds.index.get_level_values(0).unique(), seg).reindex(preds.index.get_level_values(0)).to_numpy()
    preds["target"] = ds.Y[tr.target].reindex(preds.index).to_numpy()
    if tr.probabilities is not None:
        preds["prob"] = tr.probabilities
    preds.to_parquet(d / "predictions.parquet")
    exp.weights.loc[exp.first_signal:].to_parquet(d / "positions.parquet")
    ret = pd.DataFrame({"net": exp.net, "gross": exp.gross, "costs": exp.costs, "turnover": exp.turnover, "benchmark": exp.benchmark, "segment": exp.labels.reindex(exp.net.index)})
    ret.to_parquet(d / "returns.parquet")
    if exp.insample_net is not None:
        exp.insample_net.rename("net").to_frame().to_parquet(d / "returns_insample.parquet")
    if len(tr.insample):
        tr.insample.rename("score").to_frame().to_parquet(d / "predictions_insample.parquet")
    files = {"predictions": d / "predictions.parquet", "positions": d / "positions.parquet", "returns": d / "returns.parquet", "config": d / "config.yaml"}
    if exp.final_model is not None:
        (d / "model.pkl").write_bytes(exp.final_model.to_bytes())
        files["model"] = d / "model.pkl"
    report = jsonable(exp.report)
    import json as _json

    (d / "report.json").write_text(_json.dumps(report, allow_nan=False))
    (d / "report.html").write_text(render_html(report), encoding="utf-8")
    files.update({"report_json": d / "report.json", "report_html": d / "report.html"})
    registry.log_metrics(exp_id, exp.metrics)
    for name, path in files.items():
        registry.log_artifact(exp_id, name, path, path.suffix.lstrip("."))
    for seg_name in ("validation", "test", "forward"):
        sel = preds["segment"] == seg_name
        if sel.any():
            registry.log_prediction_summary(exp_id, seg_name, int(sel.sum()), d / "predictions.parquet", exp.report["ic"]["by_segment"].get(seg_name, {}).get("mean"))
    if exp.final_model is not None:
        registry.register_model(exp_id, exp.final_model.to_bytes(), cfg["model"]["kind"], ds.features, "candidate", cfg["name"])
    registry.register_feature_set(ds, exp.report["build_key"])


def load_report(registry: Registry, exp_id: str) -> dict:
    return registry.report(exp_id)


def is_classification_target(target: str) -> bool:
    return is_classification(target)


def load_run(registry: Registry, exp_id: str) -> tuple[dict, Dataset, TrainResult, dict]:
    """Rebuild what ``evaluate_run`` needs from a saved experiment: its configuration, the saved panel, and a ``TrainResult`` made of the saved predictions (the fitted models are not kept, so anything that
    needs them, such as permutation importance and the noisy-feature test, is unavailable for a reloaded run)."""
    import yaml

    from .training import plan_for

    rec = registry.get_experiment(exp_id)
    report = registry.report(exp_id)
    d = registry.root / "experiments" / exp_id
    saved = yaml.safe_load((d / "config.yaml").read_text())
    cfg = experiment_config(merge(DEFAULTS, saved), name=saved.get("name"))
    fsets = [f for f in registry.list_feature_sets() if f["id"].startswith(report["build_key"])]
    if not fsets:
        raise ExperimentError(f"the saved panel of experiment {exp_id} is not in the registry; re-run the experiment")
    ds = Dataset.load(fsets[0]["path"])
    preds = pd.read_parquet(d / "predictions.parquet")
    plan, _ = plan_for(ds, cfg["split"], cfg["dataset"]["target"])
    insample = pd.read_parquet(d / "predictions_insample.parquet")["score"] if (d / "predictions_insample.parquet").exists() else pd.Series(dtype=float)
    folds = [dict(f, skipped=bool(f.get("skipped"))) for f in report["folds"].get("list", [])]
    train = TrainResult(preds["score"], insample, folds, [], [], pd.DataFrame(), plan, cfg["dataset"]["target"], "classification" if "prob" in preds.columns else "regression", cfg["model"]["kind"],
                        list(ds.features), preds["prob"] if "prob" in preds.columns else None, 0.0, [])
    return cfg, ds, train, {"source": rec.get("source") or "platform", "label": report["provenance"].get("label", "")}


def rebacktest(registry: Registry, exp_id: str, overrides: dict | None = None, progress=None, save: bool = True) -> Experiment:
    """Run a saved experiment's predictions through different positions rules, rebalancing or costs, with no refitting. ``overrides`` may change the ``signal`` and ``backtest`` sections only (anything
    else would change the predictions). The new run is recorded as a child of the original and counts as another look at the data."""
    bad = set(overrides or {}) - {"signal", "backtest", "name", "notes", "robustness"}
    if bad:
        raise ExperimentError(f"a re-backtest can change only the signal, backtest and robustness settings, not {sorted(bad)}: they would change the predictions")
    cfg0, ds, train, info = load_run(registry, exp_id)
    cfg = experiment_config(merge(cfg0, overrides or {}), name=(overrides or {}).get("name") or f"{cfg0['name']}-rebacktest")
    return evaluate_run(cfg, ds, train, info, registry, progress, save, exp_id, explain=False, final_model=False, robustness=None)
