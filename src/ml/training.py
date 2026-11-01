"""Walk-forward training: how a model is actually run, and the only way this platform trains one.

On each refit date the model is fitted to the rows whose labels were COMPLETE by then (the splitters' purge: a training row at position ``i`` is used only if ``i + lag + horizon + embargo <= refit position``),
and it predicts the next ``refit_every`` days, which it has never seen. The out-of-sample predictions of all the refits, laid end to end, are the strategy's forecasts; nothing else is ever scored.

The timeline is cut into the segments of ``splitters.segment_plan``. The models are fitted from the beginning of the data and predict out of sample from the start of the validation segment. The
warm-up segment gets in-sample predictions from the first model (what the model fitted to its own training data), so the gap between in-sample and out-of-sample performance, the signature of overfitting,
is measured on the same strategy. Every fitted model is kept in memory (the robustness tests re-predict with noisy features, no refit needed) and the final model, fitted to all labelled data, is what a
live prediction uses.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .datasets import Dataset
from . import intervals as iv
from .causal import CausalView
from .models import Model, default_refit, make_model, task_for
from .splitters import SegmentPlan, segment_plan, walk_forward_splits
from .targets import is_classification

MIN_TRAIN_ROWS = 300


@dataclass
class TrainResult:
    predictions: pd.Series                         # out-of-sample scores, (date, asset), from the start of the validation segment
    insample: pd.Series                            # in-sample scores of the first model on the warm-up segment
    folds: list                                    # one dict per refit
    models: list                                   # the fitted Model of each fold (None where a fold was skipped)
    fold_rows: list                                # row positions (into the dataset) each fold predicted
    importance: pd.DataFrame                       # fold x feature
    plan: SegmentPlan
    target: str
    task: str
    kind: str
    features: list
    probabilities: pd.Series | None = None         # for a classifier: P(positive class) out of sample
    seconds: float = 0.0
    skipped: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)  # one dict per fitted fold: what the model reports about its own fit (training history, device, regimes, member weights)
    extras: dict = field(default_factory=dict)       # out-of-sample frames some models add: distribution (mean, std, intervals), regimes (probabilities by date), heads (one column per horizon)

    def producers(self) -> pd.DataFrame:
        """For every date that has a prediction: the fold that made it, the day that fold's model was fitted (its information cutoff), the last day its labels depended on and a hash of the rows it was fitted to. A later
        stage that learns from these predictions (a stacker, a meta-labeller, a combination) reads here which model produced each one and what it had seen."""
        dates = pd.DatetimeIndex(sorted(set(self.predictions.index.get_level_values(0))))
        out = pd.DataFrame(index=dates, columns=["fold", "fit_cutoff", "label_end_max", "train_rows_hash"], dtype=object)
        for f in self.folds:
            if f.get("skipped"):
                continue
            span = (dates >= pd.Timestamp(f["test_start"])) & (dates <= pd.Timestamp(f["test_end"]))
            out.loc[span, ["fold", "fit_cutoff", "label_end_max", "train_rows_hash"]] = [f["fold"], f.get("fit_cutoff"), f.get("label_end_max"), f.get("train_rows_hash")]     # None for a run saved before the lineage was kept
        return out

    @property
    def lineage_recorded(self) -> bool:
        """Whether every fitted fold says when it was fitted and what its labels reached: False for a run saved before that was kept, whose out-of-sample claim then rests on the splitter alone."""
        return all(f.get("fit_cutoff") and f.get("label_end_max") for f in self.folds if not f.get("skipped"))

    def oos_violations(self) -> list[str]:
        """Every way a prediction could fail to be out of sample, as sentences: made by a model fitted after the date it scores, by one whose labels run past its own fitting day, or by none."""
        out = []
        prod = self.producers()
        for d, row in prod.iterrows():
            if pd.isna(row["fold"]):
                out.append(f"{d.date()}: no fold produced this prediction")
            elif pd.isna(row["fit_cutoff"]) or pd.isna(row["label_end_max"]):
                continue                                                                      # saved before the lineage was kept: not a violation, but not verified either (see ``lineage_recorded``)
            elif pd.Timestamp(row["fit_cutoff"]) > d:
                out.append(f"{d.date()}: scored by a model fitted on {row['fit_cutoff']}, after the date")
            elif pd.Timestamp(row["label_end_max"]) > pd.Timestamp(row["fit_cutoff"]):
                out.append(f"{d.date()}: the model that scored it was fitted with labels that end on {row['label_end_max']}, after its own fitting day {row['fit_cutoff']}")
        return out

    def scores_wide(self, dataset: Dataset, which: str = "predictions") -> pd.DataFrame:
        """The predictions as a wide frame (date x asset) on the dataset's date axis."""
        s = self.predictions if which == "predictions" else self.insample
        if s.empty:                                                                        # e.g. no in-sample scores for a reloaded or meta-labelled run
            return pd.DataFrame(np.nan, index=dataset.dates, columns=dataset.assets)
        return s.unstack("asset").reindex(index=dataset.dates, columns=dataset.assets)


def labelled_dates(ds: Dataset, target: str | None = None) -> np.ndarray:
    """Positions (into ``ds.dates``) of the dates that have at least one row with a label: the dates a model can be trained or scored on."""
    t = target or ds.target
    has = ds.Y[t].notna().to_numpy()
    return np.unique(ds.positions()[has])


def plan_for(ds: Dataset, split_cfg: dict, target: str | None = None) -> tuple[SegmentPlan, np.ndarray]:
    """The segment plan over the labelled dates and those dates' positions in the full date axis."""
    pos = labelled_dates(ds, target)
    plan = segment_plan(ds.dates[pos], float(split_cfg["validation_fraction"]), float(split_cfg["test_fraction"]), float(split_cfg["forward_fraction"]), int(split_cfg["min_train_days"]))
    return plan, pos


def row_hash(positions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(positions, dtype=np.int64).tobytes()).hexdigest()[:16]


def _fit(kind: str, params: dict, task: str, seed: int, X: pd.DataFrame, y: pd.Series, dates: np.ndarray, ds: Dataset | None = None, until: int | None = None, info: dict | None = None) -> Model:
    """Fit one model. A model that reads history is bound to a causal view of the dataset that ends on ``until`` (default the last date) and holds no labels. ``info`` (what the fit used) is kept on the model
    as ``fit_info_``, with the schema of the features it was fitted on."""
    model = make_model(kind, params, task, seed)
    if ds is not None and model.needs_dataset:
        model.bind(CausalView(ds, until))
    model.fit(X, y, dates)
    model.fit_info_ = {**(info or {}), "schema": {"features": list(X.columns), "dtypes": [str(t) for t in X.dtypes], "hash": hashlib.sha256(json.dumps([list(X.columns), [str(t) for t in X.dtypes]]).encode()).hexdigest()[:16]}}
    return model


def collect_extras(parts: list[dict]) -> dict:
    """The per-fold ``predict_extras`` frames laid end to end (a date that appears in two folds keeps its first row)."""
    out = {}
    for name in sorted({k for part in parts for k in part}):
        frame = pd.concat([part[name] for part in parts if name in part]).sort_index()
        out[name] = frame[~frame.index.duplicated(keep="first")]
    return out


def walk_forward_train(ds: Dataset, model_cfg: dict, split_cfg: dict, seed: int = 0, target: str | None = None, progress=None, stop_after: str | None = None) -> TrainResult:
    """Fit ``model_cfg`` (``kind`` and ``params``) walk-forward on ``ds`` according to ``split_cfg`` and return the out-of-sample predictions with everything needed to audit them.

    ``stop_after="validation"`` ends the run at the end of the validation segment: the test and forward segments are never predicted, which is how a search keeps them out of its reach."""
    t0 = time.perf_counter()
    tgt = target or ds.target
    task = task_for(is_classification(tgt))
    kind, params = model_cfg["kind"], dict(model_cfg.get("params") or {})
    make_model(kind, params, task, seed)                                         # an unknown model, setting or task is refused here, with its own message, not hidden as a fold that failed to fit
    plan, upos = plan_for(ds, split_cfg, tgt)                                    # plan positions index the labelled dates; upos maps them to the full axis
    purge = ds.lag + ds.horizon
    refit = int(split_cfg.get("refit_every") or default_refit(kind))
    embargo = int(split_cfg.get("embargo", 0))
    last_test = plan.bounds[stop_after][1] if stop_after else None
    splits = walk_forward_splits(len(upos), plan.first_prediction(), refit, purge, embargo, str(split_cfg.get("window", "expanding")), int(split_cfg.get("rolling_days", 1260)),
                                 last_test)
    pos_all = ds.positions()
    # row -> position in the labelled-date axis (-1 where the date has no label at all)
    inv = np.full(len(ds.dates), -1)
    inv[upos] = np.arange(len(upos))
    lab = inv[pos_all]
    has_y = ds.Y[tgt].notna().to_numpy()
    y_all = ds.Y[tgt]
    preds, probs, folds, models, fold_rows, imps, skipped = [], [], [], [], [], [], []
    extras_parts, diagnostics = [], []
    features = list(ds.features)
    for sp in splits:
        f0 = time.perf_counter()
        t_first, t_last = int(upos[sp.test[0]]), int(upos[sp.test[-1]])                   # the first and last day the model will be asked about, on the dataset's own date axis
        in_block = (lab >= sp.train[0]) & (lab <= sp.train[-1]) & has_y                  # the walk-forward block on the list of labelled dates (what the split believed)
        in_train = (lab >= sp.train[0]) & has_y & iv.eligible_for_fit(pos_all, ds.lag, ds.horizon, t_first, embargo)           # the rule that decides: the label ended on the real calendar by the first decision it may inform
        test_rows = np.flatnonzero((lab >= sp.test[0]) & (lab <= sp.test[-1]))
        tr_idx = np.flatnonzero(in_train)
        info = {"fold": sp.fold, "train_start": str(ds.dates[upos[sp.train[0]]].date()), "train_end": str(ds.dates[upos[sp.train[-1]]].date()), "test_start": str(ds.dates[upos[sp.test[0]]].date()),
                "test_end": str(ds.dates[upos[sp.test[-1]]].date()), "n_train": int(len(tr_idx)), "n_test": int(len(test_rows))}
        if len(tr_idx):
            end_max = int(iv.label_ends(pos_all[tr_idx], ds.lag, ds.horizon).max())
            info.update(fit_cutoff=str(ds.dates[t_first].date()), label_end_max=str(ds.dates[min(end_max, len(ds.dates) - 1)].date()), train_rows_hash=row_hash(tr_idx),
                        rows_removed_by_interval_rule=int((in_block & ~in_train).sum()), rows_added_by_interval_rule=int((in_train & ~in_block).sum()), purge_rule="label ended by the first decision date, less the embargo")
        if len(tr_idx) < MIN_TRAIN_ROWS or not len(test_rows):
            skipped.append(sp.fold)
            models.append(None)
            fold_rows.append(test_rows)
            folds.append({**info, "skipped": True, "seconds": 0.0})
            continue
        Xtr, ytr = ds.X.iloc[tr_idx], y_all.iloc[tr_idx]
        try:
            model = _fit(kind, params, task, seed, Xtr, ytr, lab[tr_idx], ds, until=t_first, info={k: info[k] for k in info if k in ("fit_cutoff", "label_end_max", "train_rows_hash", "train_start", "train_end", "n_train")})
        except ValueError as exc:                                                 # e.g. a single class in a short window: record and move on
            skipped.append(sp.fold)
            models.append(None)
            fold_rows.append(test_rows)
            folds.append({**info, "skipped": True, "error": str(exc), "seconds": 0.0})
            continue
        Xte = ds.X.iloc[test_rows]
        if model.needs_dataset:
            model.bind(CausalView(ds, t_last))                                            # while it predicts the block it may read history up to the block's last day, and nothing later
        pred = pd.Series(model.predict(Xte), index=Xte.index)
        preds.append(pred)
        if task == "classification":
            probs.append(pd.Series(model.predict_proba(Xte), index=Xte.index))
        imp = model.importance()
        imps.append(imp.reindex(features).rename(sp.fold))
        extras_parts.append(model.predict_extras(Xte))
        diagnostics.append({"fold": sp.fold, **model.diagnostics()})
        models.append(model)
        fold_rows.append(test_rows)
        folds.append({**info, "skipped": False, "seconds": round(time.perf_counter() - f0, 3)})
        if progress:
            progress(f"fold {sp.fold + 1} of {len(splits)}")
    if not preds:
        raise ValueError("no fold could be fitted: the training windows are too short or the labels have a single class; use more history or a smaller min_train_days")
    first_idx = next(i for i, m in enumerate(models) if m is not None)
    first_model = models[first_idx]
    warm = np.flatnonzero((lab >= 0) & (lab < plan.bounds["train"][1]) & (lab <= splits[first_idx].train[-1]))                 # what the first model was fitted to
    insample = pd.Series(first_model.predict(ds.X.iloc[warm]), index=ds.X.index[warm]) if len(warm) else pd.Series(dtype=float)
    return TrainResult(pd.concat(preds).sort_index(), insample.sort_index(), folds, models, fold_rows, pd.DataFrame(imps), plan, tgt, task, kind, features,
                       pd.concat(probs).sort_index() if probs else None, time.perf_counter() - t0, skipped, diagnostics, collect_extras(extras_parts))


def fit_final(ds: Dataset, model_cfg: dict, seed: int = 0, target: str | None = None) -> Model:
    """The model a live prediction uses: fitted to every labelled row there is. Its labels are complete by definition (a row without a label is not used). The model remembers the last day any of its labels
    depends on (``fit_info_["label_end_max"]``): it has seen the market through that day, so it may not be used to replay a decision made before it."""
    tgt = target or ds.target
    task = task_for(is_classification(tgt))
    mask = ds.Y[tgt].notna().to_numpy()
    idx = np.flatnonzero(mask)
    pos = ds.positions()
    end_max = int(iv.label_ends(pos[idx], ds.lag, ds.horizon).max())
    info = {"fit_cutoff": str(ds.dates[-1].date()), "label_end_max": str(ds.dates[min(end_max, len(ds.dates) - 1)].date()), "train_rows_hash": row_hash(idx), "n_train": int(len(idx)),
            "train_start": str(ds.dates[pos[idx].min()].date()), "train_end": str(ds.dates[pos[idx].max()].date())}
    return _fit(model_cfg["kind"], dict(model_cfg.get("params") or {}), task, seed, ds.X.iloc[idx], ds.Y[tgt].iloc[idx], pos[idx], ds, None, info)


class ReplayError(ValueError):
    """A model asked to decide a date it had already seen the future of."""


def predict_latest(model: Model, ds: Dataset, date=None, allow_replay: bool = False) -> pd.Series:
    """The model's score for each asset on ``date`` (default the last date of the dataset, which has features but no label yet). A model fitted with information after ``date`` (see ``fit_info_``) refuses, because
    its answer would be a replay and not a decision; ``allow_replay`` says that is what is wanted, and the caller must then not present the result as a forward test."""
    d = pd.Timestamp(date) if date is not None else ds.dates[-1]
    seen = (getattr(model, "fit_info_", None) or {}).get("label_end_max")
    if seen and d < pd.Timestamp(seen) and not allow_replay:
        raise ReplayError(f"this model was fitted with labels that end on {seen}, so it has seen the market after {d.date()}: a score for {d.date()} is a replay, not a decision (allow_replay=True to run one anyway)")
    if model.needs_dataset:
        model.bind(CausalView(ds, d))
    rows = ds.X.xs(d, level=0, drop_level=False)
    return pd.Series(model.predict(rows), index=rows.index.get_level_values(1), name=str(d.date()))
