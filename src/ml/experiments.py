"""One experiment end to end: data, walk-forward training, predictions, positions, backtest, evaluation, overfitting diagnostics, explanation, robustness, and the record of it all.

``run_experiment`` is the single door. Nothing in the platform trains a model any other way, which is the point: a model's score here is always the score of a STRATEGY that was costed and lagged by the
platform's engine and judged on segments of time it was not fitted on, with the number of times the data has been looked at in the report next to it.

The report (a plain JSON-able dictionary) is what the dashboard shows and what is saved as ``report.json``; ``Experiment`` keeps the live objects (dataset, fold models, predictions, weights) for the
tests that need them: explanations, robustness, forward testing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import execution as xe
from . import explainability as xp
from . import model_report as mr
from . import provenance as pv
from . import research_controls as rc
from . import robustness as rb
from .backtest import equal_weight_benchmark, label_days, run_backtest, trim_to_signal
from .config import DEFAULTS, build_key, config_hash, dataset_key, experiment_config, merge
from .datasets import Dataset, build_dataset, resolve_bundle
from .models import Model
from .registry import Registry
from .signals import UNCERTAINTY_METHODS, build_weights
from .models import task_for
from .contracts import OutputContractError, build_contract, check_signal_use
from .targets import forward_return, is_classification
from .training import TrainResult, fit_final, plan_for, walk_forward_train
from .visualization import equity_payload, importance_bars, ic_series_payload, jsonable, prediction_distribution, render_html


RL_KINDS = ("ppo", "sac", "recurrent_ppo")


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
    execution: object | None = None                                       # the realistic run (orders, fills, ledger) when the experiment asked for that mode; None for the vectorized engine


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


def check_dataset_matches(cfg: dict, ds: Dataset) -> None:
    """Refuse a panel that was built for another question than the configuration asks. A different target, horizon, lag, set of feature groups or scaling would make the labels and features mean something other than
    what the report says they are."""
    want, got = cfg["dataset"], ds.meta
    problems = []
    for key in ("target", "horizon", "lag", "normalize", "winsor", "benchmark", "min_history"):
        if key in got and key in want and str(got[key]) != str(want[key]) and not (isinstance(want[key], float) and float(got[key]) == want[key]):
            problems.append(f"{key}: the panel has {got[key]}, the configuration asks for {want[key]}")
    if "feature_groups" in got and list(got["feature_groups"]) != list(want["feature_groups"]):
        problems.append(f"feature_groups: the panel has {list(got['feature_groups'])}, the configuration asks for {list(want['feature_groups'])}")
    if problems:
        raise ExperimentError("the dataset does not match the configuration (" + "; ".join(problems) + "): rebuild it from this configuration")


def prepare(cfg: dict, bundle=None, config=None, builder=None, dataset: Dataset | None = None, info: dict | None = None, progress=None) -> tuple[Dataset, dict]:
    """The dataset for a configuration: the one given, or built from ``bundle`` (or from the universe named in the configuration)."""
    if dataset is not None:
        check_dataset_matches(cfg, dataset)
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


def realistic_requested(cfg: dict) -> bool:
    return bool(cfg.get("execution")) and cfg["execution"].get("mode", "realistic_cash_equity") == "realistic_cash_equity"


def _market(cfg: dict, bundle=None, config=None, builder=None, info: dict | None = None, market: dict | None = None) -> dict | None:
    """The market data the realistic mode needs beyond the dataset (the traded volume), from the bundle given or resolved for the configuration's universe. ``None`` when the run did not ask for the mode;
    a missing universe is an error, not a silent fall-back to the vectorized engine."""
    if not realistic_requested(cfg):
        return None
    if market is not None:
        return market
    if bundle is None:
        try:
            bundle, info = resolve_bundle(config, cfg["dataset"]["universe"], builder)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise ExperimentError(f"realistic execution needs the market data (volume) for universe '{cfg['dataset']['universe']}': {exc}") from None
    return {"volume": getattr(bundle, "volume", None), "status": str((info or {}).get("source", "unknown"))}


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


def _latest_positions(weights: pd.DataFrame) -> dict:
    """The book on the last day that held anything: the final days of a dataset have no scores (their targets are not known yet), so the very last row of weights is empty."""
    held = weights.index[(weights.abs() > 1e-6).any(axis=1)] if len(weights) else weights.index
    if not len(held):
        return {"date": str(weights.index[-1].date()) if len(weights) else None, "weights": {}}
    day = held[-1]
    return {"date": str(day.date()), "weights": {a: float(w) for a, w in weights.loc[day].items() if abs(w) > 1e-6}}


def run_experiment(overrides: dict | None = None, bundle=None, config=None, builder=None, registry: Registry | None = None, progress=None, save: bool = True, dataset: Dataset | None = None,
                   info: dict | None = None, parent_id: str | None = None, explain: bool = True, final_model: bool = True, robustness: bool | None = None, trials_extra: int = 0,
                   experiment_id: str | None = None, stop_after: str | None = None, evaluation=None, study: str | None = None, entry_point: str = "python", kind: str = "experiment",
                   market: dict | None = None) -> Experiment:
    """Run one experiment. ``overrides`` is merged over the platform configuration (``ml.config.experiment_config``). With a ``registry`` and ``save`` the artifacts are written and the run recorded.

    ``explain`` computes permutation importance; ``final_model`` fits the model a live prediction would use; ``robustness`` overrides the configuration's switch for the stress tests;
    ``trials_extra`` adds to the recorded trial count for trials that are NOT in the registry (a search run without one passes how many; a search that logged its trials there must pass 0, or each is counted
    twice). ``stop_after="validation"`` is the development mode: the test and forward segments are not predicted at all, so nothing a development run does can depend on them.

    With a ``registry`` the run is an operation of its study (``research_controls``): registered before it starts, finished as done, failed or cancelled. A run that predicts the test and forward segments is a
    look at protected outcomes: it needs ``evaluation`` (the context of a frozen plan, from ``begin_evaluation``) when a reservation covers that history, and otherwise it is recorded as an exposure and the report
    says ``exposed_exploratory``."""
    t0 = time.perf_counter()
    cfg = experiment_config(overrides)
    progress = progress or (lambda msg: None)
    try:
        check_signal_use(cfg["model"]["kind"], task_for(is_classification(cfg["dataset"]["target"])), cfg["dataset"]["target"], cfg["signal"])        # refused before any training, not after it
    except OutputContractError as exc:
        raise ExperimentError(str(exc)) from None
    ds, info = prepare(cfg, bundle, config, builder, dataset, info, progress)
    market = _market(cfg, bundle, config, builder, info, market)
    exp_id = experiment_id or (registry.new_id(cfg["name"]) if registry is not None else "unsaved")
    with rc.experiment_op(registry, cfg, kind, parent_experiment=parent_id, study=study, experiment_id=exp_id, feature_hash=pv.digest(sorted(ds.features), 16), counts=kind not in rc.NON_COUNTING) as op:
        holdout = _holdout_block(registry, cfg, ds, stop_after, evaluation, entry_point, exp_id)
        progress("training walk-forward")
        try:
            train = walk_forward_train(ds, cfg["model"], cfg["split"], cfg["seed"], progress=progress, stop_after=stop_after)
        except ValueError as exc:
            raise ExperimentError(str(exc)) from None
        if stop_after == "validation":
            rc.guard_development(train, train.plan)
        lineage = None if op is None else {"study": op.study_id, "operation": op.op_key, "kind": op.kind, "counts": op.counts}
        exp = evaluate_run(cfg, ds, train, info, registry, progress, save, parent_id, explain, final_model, robustness, trials_extra, exp_id, stop_after, t0, holdout=holdout, lineage=lineage, market=market)
        if op is not None:
            op.experiment_id = exp.id
            op.finish_detail = {"sharpe": exp.metrics.get("oos", {}).get("sharpe"), "validation_sharpe": exp.metrics.get("validation", {}).get("sharpe"), "holdout": holdout.get("status")}
    return exp


def _holdout_block(registry, cfg: dict, ds: Dataset, stop_after: str | None, evaluation, entry_point: str, exp_id: str) -> dict:
    """The ``holdout`` block of a run's report, opened BEFORE any protected outcome is predicted: ``development`` for a run that stops at the validation segment, otherwise the access the ledger recorded."""
    if stop_after:
        return {"status": "development", "note": "predictions stop at the end of the validation segment: the test and forward outcomes were not read"}
    if registry is None:
        return {"status": "unrecorded", "note": "no registry was given, so the look at the test and forward segments is not in any ledger"}
    try:
        plan, _ = plan_for(ds, cfg["split"], cfg["dataset"]["target"])
    except ValueError as exc:
        raise ExperimentError(str(exc)) from None
    identity = rc.identity_of(ds, plan)
    try:
        block = rc.open_outcome_access(registry, cfg, identity, evaluation, entry_point=entry_point, purpose="direct_experiment" if evaluation is None else "frozen_evaluation", experiment_id=exp_id)
    except rc.HoldoutAccessError as exc:
        raise ExperimentError(str(exc)) from None
    return {**block, "identity": {k: identity[k] for k in ("assets", "start", "end")}}


def evaluate_run(cfg: dict, ds: Dataset, train: TrainResult, info: dict, registry: Registry | None = None, progress=None, save: bool = True, parent_id: str | None = None, explain: bool = True,
                 final_model: bool = True, robustness: bool | None = None, trials_extra: int = 0, experiment_id: str | None = None, stop_after: str | None = None, t0: float | None = None,
                 holdout: dict | None = None, lineage: dict | None = None, market: dict | None = None) -> Experiment:
    """Everything after the fitting: positions, backtest, segments, information coefficient, overfitting diagnostics, explanation, robustness, the report and (if a registry is given) the record. Separate
    from ``run_experiment`` so that the same predictions can be re-run under different signal and cost settings without refitting (``rebacktest``)."""
    t0 = time.perf_counter() if t0 is None else t0
    progress = progress or (lambda msg: None)
    dkey, chash = dataset_key(cfg), config_hash(cfg)
    try:
        output_kind = check_signal_use(train.kind, train.task, train.target, cfg["signal"])               # what the model's number means, and whether the position rule may read it as that
    except OutputContractError as exc:
        raise ExperimentError(str(exc)) from None
    lag, horizon = ds.lag, ds.horizon
    scores = train.scores_wide(ds)
    prob_wide = train.probabilities.unstack("asset").reindex(index=ds.dates, columns=ds.assets) if train.probabilities is not None else None
    vol = trailing_vol(ds.returns)
    conf = confidence_from_scores(scores, prob_wide)
    regime_on = regime_series(ds, cfg["signal"].get("regime_on_states", [0]))
    groups = ds.group or ds.asset_class
    dist = train.extras.get("distribution")
    unc_wide = None if dist is None else dist["std"].unstack("asset").reindex(index=ds.dates, columns=ds.assets)          # the model's own predictive standard deviation, for the uncertainty-aware rules
    progress("building positions and backtesting")
    try:
        weights = build_weights(scores, cfg["signal"], vol, conf, regime_on, groups, ds.investable, unc_wide, ds.returns)
    except ValueError as exc:
        raise ExperimentError(str(exc)) from None
    result = run_backtest(weights, ds.returns, ds.investable, cfg["backtest"], lag, cfg["name"])
    realistic = None
    if realistic_requested(cfg):                                               # the declared evaluation mode: the same weights through orders, fills and the ledger instead of a cost per unit of turnover
        mk = _market(cfg, None, None, None, info, market)
        try:
            realistic = xe.realistic_run(weights, ds.prices, ds.returns, ds.investable, cfg["backtest"], lag + int(cfg["backtest"].get("extra_lag", 0)), cfg["execution"], volume=(mk or {}).get("volume"),
                                         data_status=(mk or {}).get("status", str(info.get("source", "unknown"))), name=cfg["name"], legacy=result,
                                         **{k: (mk or {}).get(k) for k in ("membership", "actions", "borrow", "closed", "halted")})
        except ValueError as exc:
            raise ExperimentError(str(exc)) from None
        result = realistic.backtest
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
        sig_in = {**cfg["signal"], "method": "proportional"} if cfg["signal"]["method"] in UNCERTAINTY_METHODS else cfg["signal"]               # no predictive distribution in sample
        w_in = build_weights(s_in, sig_in, vol, confidence_from_scores(s_in, None), regime_on, groups, ds.investable)
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
    ctx = rb.Context(ds, train, cfg, scores, first_signal, net, labels, vol, conf, regime_on, unc_wide)
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
        "positions": _latest_positions(weights),
        "folds": {"n": len(train.folds), "n_skipped": len(train.skipped), "first": train.folds[0] if train.folds else None, "last": train.folds[-1] if train.folds else None, "refit_every": cfg["split"]["refit_every"],
                  "list": [{k: f.get(k) for k in ("fold", "test_start", "test_end", "n_train", "n_test", "skipped", "seconds", "train_start", "train_end", "fit_cutoff", "label_end_max", "train_rows_hash",
                                                  "rows_removed_by_interval_rule", "rows_added_by_interval_rule", "purge_rule", "dependence")} for f in train.folds]},
        "dependence": dependence_summary(train.folds, cfg["split"].get("sample_weighting", "none")), "audit": ds.meta.get("audit"),
        "features": {"n": len(ds.features), "skipped": ds.meta.get("skipped", {}), "names": ds.features}, "robustness": rob,
        "trials": {"dataset_trials": n_trials, "note": "Every experiment and search trial on this dataset is counted; the deflated Sharpe probability deflates the validation Sharpe by this number."},
        "warnings": list(over["warnings"]), "seconds": round(time.perf_counter() - t0, 2), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    report["output_contract"] = build_contract(train.kind, train.task, train.target, ds, cfg, chash)
    report["execution"] = jsonable(realistic.block if realistic is not None else xe.legacy_block(cfg))
    if realistic is not None and rob is not None:
        report["warnings"].append("The robustness tests re-run the vectorized engine, not the order-level execution: their cost and delay variants do not describe the realistic result.")
    report["holdout"] = jsonable(holdout if holdout is not None else {"status": "development" if stop_after else "unrecorded"})
    report["lineage"] = jsonable(lineage if lineage is not None else {"study": rc.study_of(cfg), "operation": None, "note": "not registered as an operation (this run was made through a path that records none)"})
    if registry is not None:
        report["trials"]["study"] = {"study": rc.study_of(cfg), "raw_looks": rc.raw_looks(registry, rc.study_of(cfg)),
                                     "note": "Raw look counts of the study (failed and cancelled tries included); they are counts, not estimates of independent trials."}
    if cfg["signal"].get("reinterpret_outputs") and output_kind != "expected_return":
        report["warnings"].append(f"The {train.kind} output is a {output_kind.replace('_', ' ')}, and signal.reinterpret_outputs reads it as a score for the position rule: that reading is yours, not the model's.")
    try:
        report["model_extras"] = jsonable(mr.build_report(train, ds, cfg, scores, fwd, train.plan.labels(), None, time.perf_counter() - t0, chash, fmodel.to_bytes() if fmodel is not None else None))
    except Exception:                                                              # the extras are an addition to the report; if they cannot be built the experiment still stands
        report["model_extras"] = {"kind": train.kind, "outputs": []}
    metrics = {"oos": {k: oos_summary.get(k) for k in ("sharpe", "cagr", "ann_vol", "max_drawdown", "hit_rate", "ann_turnover", "ann_cost_drag", "gross_sharpe", "total_return", "sortino", "calmar", "n_obs")},
               "ic": {k: ic_all[k] for k in ("mean", "std", "ir", "t", "hit_rate", "n")}, "overfit": {"generalization_score": over["generalization_score"], "psr_test_forward": over["psr_test_forward"],
                                                                                                 "deflated_sharpe_probability": over["deflated_sharpe_probability"], "n_trials": n_trials},
               "robustness": {} if rob is None else {"score": rob["score"], "n_pass": rob["n_pass"], "n_warn": rob["n_warn"], "n_fail": rob["n_fail"]}}
    for seg, s in seg_stats.items():
        metrics[seg] = {k: s[k] for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "n_days", "hit_rate", "total_return")}
    res = report["model_extras"].get("resources", {})
    metrics["run"] = {k: v for k, v in {"train_seconds": res.get("train_seconds"), "total_seconds": res.get("total_seconds"), "n_params": res.get("n_params"), "peak_memory_mb": res.get("peak_memory_mb"),
                                         "n_fits": float(len(train.diagnostics))}.items() if v is not None}
    exp = Experiment(exp_id, cfg, ds, train, scores, weights, net, gross, costs, turnover, bench, insample_net, labels, first_signal, fmodel, report, ctx, metrics, execution=realistic)
    if registry is not None and save:
        progress("saving")
        record = pv.provenance_record(cfg, ds, train, report["output_contract"], chash)
        report["reproducibility"] = pv.summary(record)                                    # in the report that is saved, so the page and the file carry the identity
        save_experiment(registry, exp, info, parent_id, time.perf_counter() - t0)
        pv.store(registry, exp_id, record)
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
    for name, frame in tr.extras.items():                                           # distribution, regimes, heads, weights, latent, portfolio: what the model reported beside its scores
        frame.to_parquet(d / f"extra_{name}.parquet")
        files[f"extra_{name}"] = d / f"extra_{name}.parquet"
    if exp.final_model is not None and exp.final_model.diagnostics():
        (d / "diagnostics.json").write_text(json.dumps(jsonable(exp.final_model.diagnostics()), allow_nan=False))
        files["diagnostics"] = d / "diagnostics.json"
    if exp.final_model is not None:
        (d / "model.pkl").write_bytes(exp.final_model.to_bytes())
        files["model"] = d / "model.pkl"
        from .bundle import InferenceBundle

        try:                                                                              # the same model with its schema, preprocessing description, cutoffs and label convention: what a forward run checks its data against
            InferenceBundle.from_model(exp.final_model, ds).save(d / "model.qfb")
            files["bundle"] = d / "model.qfb"
        except (ValueError, TypeError) as exc:                                            # a model with no recorded fit schema keeps working through model.pkl; the lack is said, not hidden
            (d / "bundle_error.txt").write_text(str(exc))
    if exp.execution is not None:                                                     # the orders, fills and journal of the realistic mode, with a manifest of hashes the report cites
        manifest = xe.write_artifacts(exp.execution, d)
        exp.report["execution"]["manifest"] = {"sha256": manifest["manifest_sha256"], "files": manifest["files"], "inputs": manifest["inputs"]}
        files.update({f"execution_{k}": d / v["file"] for k, v in manifest["files"].items()})
        files["execution_manifest"] = d / "execution_manifest.json"
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


def dependence_summary(folds: list, requested: str) -> dict:
    """What was done about overlapping labels, over all folds: the treatment asked for, how overlapped the labels were, and whether the model used the weights (a model that ignores per-row weights says so here)."""
    rows = [f["dependence"] for f in folds if f.get("dependence")]
    if not rows:
        return {"requested": requested, "treatment": "not recorded", "folds": 0}
    applied = [r for r in rows if r["treatment"] == "uniqueness_weights"]
    notes = sorted({r["note"] for r in rows if r.get("note")})
    return {"requested": requested, "treatment": "uniqueness_weights" if applied and all(r.get("applied") for r in applied) else ("none" if not applied else "requested, not fully applied"), "folds": len(rows),
            "mean_uniqueness": round(float(np.mean([r["mean_uniqueness"] for r in rows])), 4), "mean_concurrency": round(float(np.mean([r["mean_concurrency"] for r in rows])), 3), "notes": notes}


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
    check_dataset_matches(cfg, ds)
    preds = pd.read_parquet(d / "predictions.parquet")
    plan, _ = plan_for(ds, cfg["split"], cfg["dataset"]["target"])
    insample = pd.read_parquet(d / "predictions_insample.parquet")["score"] if (d / "predictions_insample.parquet").exists() else pd.Series(dtype=float)
    folds = [dict(f, skipped=bool(f.get("skipped"))) for f in report["folds"].get("list", [])]
    extras = {p.stem[len("extra_"):]: pd.read_parquet(p) for p in sorted(d.glob("extra_*.parquet"))}
    train = TrainResult(preds["score"], insample, folds, [], [], pd.DataFrame(), plan, cfg["dataset"]["target"], "classification" if "prob" in preds.columns else "regression", cfg["model"]["kind"],
                        list(ds.features), preds["prob"] if "prob" in preds.columns else None, 0.0, [], [], extras)
    return cfg, ds, train, {"source": rec.get("source") or "platform", "label": report["provenance"].get("label", "")}


def rebacktest(registry: Registry, exp_id: str, overrides: dict | None = None, progress=None, save: bool = True, bundle=None, config=None, builder=None, market: dict | None = None) -> Experiment:
    """Run a saved experiment's predictions through different positions rules, rebalancing or costs, with no refitting. ``overrides`` may change the ``signal`` and ``backtest`` sections only (anything
    else would change the predictions). The new run is recorded as a child of the original and counts as another look at the data: it is an operation of the parent's study, and it inherits what the parent
    read (a re-analysis of outcomes a plan or an exploratory run already exposed is recorded as one and is never untouched)."""
    bad = set(overrides or {}) - {"signal", "backtest", "name", "notes", "robustness", "execution"}
    if bad:
        raise ExperimentError(f"a re-backtest can change only the signal, backtest, execution and robustness settings, not {sorted(bad)}: they would change the predictions")
    cfg0, ds, train, info = load_run(registry, exp_id)
    cfg = experiment_config(merge(cfg0, overrides or {}), name=(overrides or {}).get("name") or f"{cfg0['name']}-rebacktest")
    over_keys = set(overrides or {})
    kind = "portfolio_variation" if "execution" in over_keys and "signal" not in over_keys else ("signal_variation" if "signal" in over_keys or "backtest" not in over_keys else "portfolio_variation")
    market = _market(cfg, bundle, config, builder, info, market)                      # a realistic rerun needs the volume of the universe; a vectorized one needs nothing more
    child_id = registry.new_id(cfg["name"])
    study = rc.study_of_experiment(registry, exp_id) or rc.study_of(cfg0)
    with rc.experiment_op(registry, cfg, kind, parent_experiment=exp_id, study=study, experiment_id=child_id, feature_hash=pv.digest(sorted(ds.features), 16), adopt_parent=False) as op:
        holdout = rc.inherit_holdout(registry, [exp_id], child_id, "rebacktest")
        exp = evaluate_run(cfg, ds, train, info, registry, progress, save, exp_id, explain=False, final_model=False, robustness=None, experiment_id=child_id, holdout=holdout,
                           lineage={"study": study, "operation": op.op_key, "kind": kind, "counts": True, "parent": exp_id}, market=market)
        op.experiment_id = exp.id
        op.finish_detail = {"parent": exp_id, "holdout": holdout.get("status")}
    return exp
