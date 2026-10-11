"""Walk-forward training: how a model is actually run, and the only way this platform trains one.

On each refit date the model is fitted to the rows whose labels were COMPLETE by then (the splitters' purge: a training row at position ``i`` is used only if ``i + lag + horizon + embargo <= refit position``),
and it predicts the next ``refit_every`` days, which it has never seen. The out-of-sample predictions of all the refits, laid end to end, are the strategy's forecasts; nothing else is ever scored.

The timeline is cut into the segments of ``splitters.segment_plan``. The models are fitted from the beginning of the data and predict out of sample from the start of the validation segment. The
warm-up segment gets in-sample predictions from the first model (what the model fitted to its own training data), so the gap between in-sample and out-of-sample performance, the signature of overfitting,
is measured on the same strategy. Every fitted model is kept in memory (the robustness tests re-predict with noisy features, no refit needed) and the final model, fitted to all labelled data, is what a
live prediction uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .datasets import Dataset
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


def _fit(kind: str, params: dict, task: str, seed: int, X: pd.DataFrame, y: pd.Series, dates: np.ndarray) -> Model:
    return make_model(kind, params, task, seed).fit(X, y, dates)


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
    last_test = plan.bounds[stop_after][1] if stop_after else None
    splits = walk_forward_splits(len(upos), plan.first_prediction(), refit, purge, int(split_cfg.get("embargo", 0)), str(split_cfg.get("window", "expanding")), int(split_cfg.get("rolling_days", 1260)),
                                 last_test)
    pos_all = ds.positions()
    # row -> position in the labelled-date axis (-1 where the date has no label at all)
    inv = np.full(len(ds.dates), -1)
    inv[upos] = np.arange(len(upos))
    lab = inv[pos_all]
    has_y = ds.Y[tgt].notna().to_numpy()
    y_all = ds.Y[tgt]
    preds, probs, folds, models, fold_rows, imps, skipped = [], [], [], [], [], [], []
    features = list(ds.features)
    for sp in splits:
        f0 = time.perf_counter()
        in_train = (lab >= sp.train[0]) & (lab <= sp.train[-1]) & has_y                  # the walk-forward train block is contiguous
        test_rows = np.flatnonzero((lab >= sp.test[0]) & (lab <= sp.test[-1]))
        tr_idx = np.flatnonzero(in_train)
        info = {"fold": sp.fold, "train_start": str(ds.dates[upos[sp.train[0]]].date()), "train_end": str(ds.dates[upos[sp.train[-1]]].date()), "test_start": str(ds.dates[upos[sp.test[0]]].date()),
                "test_end": str(ds.dates[upos[sp.test[-1]]].date()), "n_train": int(len(tr_idx)), "n_test": int(len(test_rows))}
        if len(tr_idx) < MIN_TRAIN_ROWS or not len(test_rows):
            skipped.append(sp.fold)
            models.append(None)
            fold_rows.append(test_rows)
            folds.append({**info, "skipped": True, "seconds": 0.0})
            continue
        Xtr, ytr = ds.X.iloc[tr_idx], y_all.iloc[tr_idx]
        try:
            model = _fit(kind, params, task, seed, Xtr, ytr, lab[tr_idx])
        except ValueError as exc:                                                 # e.g. a single class in a short window: record and move on
            skipped.append(sp.fold)
            models.append(None)
            fold_rows.append(test_rows)
            folds.append({**info, "skipped": True, "error": str(exc), "seconds": 0.0})
            continue
        Xte = ds.X.iloc[test_rows]
        pred = pd.Series(model.predict(Xte), index=Xte.index)
        preds.append(pred)
        if task == "classification":
            probs.append(pd.Series(model.predict_proba(Xte), index=Xte.index))
        imp = model.importance()
        imps.append(imp.reindex(features).rename(sp.fold))
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
                       pd.concat(probs).sort_index() if probs else None, time.perf_counter() - t0, skipped)


def fit_final(ds: Dataset, model_cfg: dict, seed: int = 0, target: str | None = None) -> Model:
    """The model a live prediction uses: fitted to every labelled row there is. Its labels are complete by definition (a row without a label is not used)."""
    tgt = target or ds.target
    task = task_for(is_classification(tgt))
    mask = ds.Y[tgt].notna().to_numpy()
    idx = np.flatnonzero(mask)
    return _fit(model_cfg["kind"], dict(model_cfg.get("params") or {}), task, seed, ds.X.iloc[idx], ds.Y[tgt].iloc[idx], ds.positions()[idx])


def predict_latest(model: Model, ds: Dataset, date=None) -> pd.Series:
    """The model's score for each asset on ``date`` (default the last date of the dataset, which has features but no label yet)."""
    d = pd.Timestamp(date) if date is not None else ds.dates[-1]
    rows = ds.X.xs(d, level=0, drop_level=False)
    return pd.Series(model.predict(rows), index=rows.index.get_level_values(1), name=str(d.date()))
