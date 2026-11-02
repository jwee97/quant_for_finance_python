"""Stable feature selection inside the training window (spec v3, Release 6, P2-03).

A feature chosen because it looked good on the whole sample has been selected with the test in view. Here the choice is made *inside each fit*, by the model that is fitted on a fold's training rows and by
nothing else, so a walk-forward run selects afresh in every fold with only what that fold's training window contained. The selection is a stability selection (Meinshausen and Buhlmann, 2010): the training
dates are cut into contiguous blocks, many random halves of the blocks are drawn, a simple selector is run on each half, and a feature is chosen only if it is picked in at least ``pi`` of the halves. Blocks,
not rows or single dates, are resampled because the labels of neighbouring dates overlap and a date-level resample would pretend they are independent.

Two selectors: ``ic`` (the ``top_k`` features with the largest absolute mean cross-sectional rank IC with the label; no model involved) and ``lasso`` (the features with a non-zero coefficient of an L1 fit at a
predeclared penalty). Everything the choice depends on is a setting, and every setting is recorded, because the settings are the adaptive part: change them after looking and it is another candidate that must be
counted, not a refinement of the first.

What is persisted for every fit (``StableSelect.diagnostics``): the chosen features, every rejected feature with its selection frequency and the reason, the frequencies, the settings and the seed. Across
folds, ``selection_stability`` reports how often each feature is chosen and how much consecutive folds agree. A fit with no stable feature does not fall back to "all of them" (that would defeat the
selection): it refuses, and the trainer records the fold as skipped with that reason.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from .models import Model, make_model

VERSION = "featsel-v1"
SELECTORS = ("ic", "lasso")
MIN_BLOCKS = 6


def _date_ranks(frame: pd.DataFrame, dates: np.ndarray) -> np.ndarray:
    """Ranks of every column within each date (average ranks for ties); a column that is constant within a date gets NaN there."""
    ranked = frame.groupby(dates, sort=False).rank(method="average")
    spread = frame.groupby(dates, sort=False).transform("nunique")
    return ranked.where(spread > 1).to_numpy(dtype=float)


def _ic_by_date(X: pd.DataFrame, y: pd.Series, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(unique dates, matrix of per-date rank correlations of every feature with y)``. Dates where a feature has no cross-sectional variation, or with fewer than five assets, are NaN."""
    rx = _date_ranks(X, dates)
    ry = _date_ranks(y.to_frame("y"), dates)[:, 0]
    keep = ~np.isnan(ry)
    uniq, inv = np.unique(dates, return_inverse=True)
    n_dates = len(uniq)
    out = np.full((n_dates, X.shape[1]), np.nan)
    count = np.bincount(inv, minlength=n_dates)
    for j in range(X.shape[1]):
        ok = keep & ~np.isnan(rx[:, j])
        n = np.bincount(inv[ok], minlength=n_dates).astype(float)
        sx = np.bincount(inv[ok], weights=rx[ok, j], minlength=n_dates)
        sy = np.bincount(inv[ok], weights=ry[ok], minlength=n_dates)
        mx, my = np.divide(sx, n, out=np.zeros(n_dates), where=n > 0), np.divide(sy, n, out=np.zeros(n_dates), where=n > 0)
        dx, dy = rx[ok, j] - mx[inv[ok]], ry[ok] - my[inv[ok]]
        sxy = np.bincount(inv[ok], weights=dx * dy, minlength=n_dates)
        sxx = np.bincount(inv[ok], weights=dx * dx, minlength=n_dates)
        syy = np.bincount(inv[ok], weights=dy * dy, minlength=n_dates)
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = sxy / np.sqrt(sxx * syy)
        out[:, j] = np.where((n >= 5) & (sxx > 0) & (syy > 0) & (count > 0), ic, np.nan)
    return uniq, out


def stability_selection(X: pd.DataFrame, y: pd.Series, dates: np.ndarray, *, selector: str = "ic", top_k: int = 0, pi: float = 0.6, subsamples: int = 40, block_days: int = 21, fraction: float = 0.5,
                        selector_alpha: float = 0.05, seed: int = 0) -> dict:
    """Choose features from one training window. ``dates`` holds the date (any sortable key, one per row) of every row. Returns the record described in the module docstring; ``chosen`` may be empty."""
    if selector not in SELECTORS:
        raise ValueError(f"unknown selector '{selector}'; choose from {list(SELECTORS)}")
    if not 0.0 < pi <= 1.0 or not 0.0 < fraction < 1.0 or subsamples < 10 or block_days < 1:
        raise ValueError("pi must be in (0, 1], fraction in (0, 1), subsamples at least 10 and block_days at least 1")
    dates = np.asarray(dates)
    uniq = np.unique(dates)
    blocks = np.arange(len(uniq)) // int(block_days)
    n_blocks = int(blocks.max()) + 1 if len(uniq) else 0
    p = X.shape[1]
    base = {"version": VERSION, "selector": selector, "settings": {"top_k": int(top_k), "pi": float(pi), "subsamples": int(subsamples), "block_days": int(block_days), "fraction": float(fraction),
                                                                "selector_alpha": float(selector_alpha), "seed": int(seed)}, "n_features": int(p), "n_dates": int(len(uniq)), "n_blocks": n_blocks}
    if n_blocks < MIN_BLOCKS:
        return {**base, "eligibility": "insufficient", "reason": f"{n_blocks} blocks of {block_days} dates: at least {MIN_BLOCKS} are needed to draw different halves", "chosen": [], "rejected": [], "frequencies": {}}
    k = int(top_k) if top_k else max(1, math.ceil(math.sqrt(p)))
    base["settings"]["top_k_used"] = k
    rng = np.random.default_rng(seed)
    n_pick = max(2, int(round(fraction * n_blocks)))
    hits = np.zeros(p)
    signs = np.zeros(p)
    tried = np.zeros(p)
    unassessable: dict[str, str] = {}
    if selector == "ic":
        u_dates, ic = _ic_by_date(X, y, dates)
        block_of = blocks[np.searchsorted(uniq, u_dates)]
        sums = np.stack([np.nansum(ic[block_of == b], axis=0) for b in range(n_blocks)])
        cnts = np.stack([np.sum(~np.isnan(ic[block_of == b]), axis=0) for b in range(n_blocks)])
        full = np.nansum(ic, axis=0) / np.maximum(np.sum(~np.isnan(ic), axis=0), 1)
        assessable = np.sum(~np.isnan(ic), axis=0) >= 20
        for j, name in enumerate(X.columns):
            if not assessable[j]:
                unassessable[str(name)] = "no cross-sectional variation on enough dates (a date-level feature cannot rank assets)"
        for _ in range(int(subsamples)):
            pick = rng.choice(n_blocks, size=n_pick, replace=False)
            c = cnts[pick].sum(axis=0)
            mean = np.divide(sums[pick].sum(axis=0), c, out=np.zeros(p), where=c > 0)
            score = np.where(assessable & (c > 0), np.abs(mean), -np.inf)
            top = np.argsort(-score)[:k]
            top = top[np.isfinite(score[top])]
            hits[top] += 1
            signs[top] += (np.sign(mean[top]) == np.sign(full[top])).astype(float)
            tried += 1
    else:
        Z = ((X - X.mean()) / X.std(ddof=0).replace(0.0, np.nan)).fillna(0.0).to_numpy()
        yy = y.to_numpy(dtype=float)
        yy = (yy - np.nanmean(yy)) / (np.nanstd(yy) or 1.0)
        row_block = blocks[np.searchsorted(uniq, dates)]
        reference = np.sign(Lasso(alpha=float(selector_alpha), max_iter=5000).fit(Z, yy).coef_)               # the sign on the whole training window, to see whether a feature keeps its direction
        for _ in range(int(subsamples)):
            pick = rng.choice(n_blocks, size=n_pick, replace=False)
            rows = np.isin(row_block, pick)
            coef = Lasso(alpha=float(selector_alpha), max_iter=5000).fit(Z[rows], yy[rows]).coef_
            nz = np.abs(coef) > 1e-12
            hits += nz
            signs += (nz & (np.sign(coef) == reference)).astype(float)
            tried += 1
    freq = hits / np.maximum(tried, 1)
    chosen = [str(c) for c, q in zip(X.columns, freq) if q >= pi and str(c) not in unassessable]
    rejected = []
    for c, q in zip(X.columns, freq):
        if str(c) in chosen:
            continue
        reason = unassessable.get(str(c)) or f"selected in {q:.2f} of the {int(subsamples)} subsamples, below the stability threshold {pi:.2f}"
        rejected.append({"feature": str(c), "frequency": float(q), "reason": reason})
    q_avg = float(hits.sum() / max(tried.max(), 1.0))                                              # features picked per subsample, on average
    bound = None if pi <= 0.5 else q_avg ** 2 / ((2.0 * pi - 1.0) * p)
    out = {**base, "eligibility": "applied", "chosen": chosen, "rejected": rejected, "frequencies": {str(c): float(q) for c, q in zip(X.columns, freq)},
           "error_control": {"mean_selected_per_subsample": q_avg, "expected_false_selections_bound": bound, "meaningful": bool(bound is not None and bound < 1.0),
                             "note": "Meinshausen and Buhlmann (2010): the expected number of non-signal features chosen is at most q^2 / ((2 pi - 1) p) if their selection probabilities are exchangeable and the selector is "
                                     "no worse than guessing; a bound of 1 or more means these settings give no error control, and the chosen set is a description of the training window, not a finding"}}
    out["sign_agreement"] = {str(c): (float(s / h) if h > 0 else None) for c, s, h in zip(X.columns, signs, hits)}
    return out


class StableSelect(Model):
    """A base model fitted on the features that survive a stability selection of the training rows it is given. ``params``: ``base`` (the model kind), ``alpha`` (its penalty, where it has one), and the
    selection settings of ``stability_selection``."""

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        self.base_: Model | None = None
        self.selection_: dict = {}

    def _base_model(self) -> Model:
        p = self.params
        base_params = {"alpha": float(p["alpha"])} if str(p["base"]) in ("ridge", "lasso", "elasticnet", "logistic") and p.get("alpha") is not None else {}
        for extra in ("halflife_days",):
            if p.get(extra):
                base_params[extra] = p[extra]
        return make_model(str(p["base"]), base_params, self.task, self.seed)

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "StableSelect":
        if dates is None:
            raise ValueError("stable selection needs the date of every training row")
        p = self.params
        sel = stability_selection(X, y, np.asarray(dates), selector=str(p["selector"]), top_k=int(p["top_k"]), pi=float(p["pi"]), subsamples=int(p["subsamples"]), block_days=int(p["block_days"]),
                                  fraction=float(p["fraction"]), selector_alpha=float(p["selector_alpha"]), seed=self.seed)
        self.selection_ = sel
        if sel["eligibility"] != "applied":
            raise ValueError("stable selection could not run: " + sel["reason"])
        if len(sel["chosen"]) < int(p["min_features"]):
            raise ValueError(f"no feature was selected in at least {float(p['pi']):.2f} of the subsamples (the best was {max(sel['frequencies'].values(), default=0.0):.2f}): the training window has no stable signal at these settings")
        self.features_ = list(sel["chosen"])
        self.n_train_ = len(X)
        self.base_ = self._base_model()
        given = getattr(self, "sample_weight_", None)
        if given is not None:
            self.base_.sample_weight_, self.base_.weights_applied_ = given, False
        self.base_.fit(X[self.features_], y, dates)
        if given is not None:
            self.weights_applied_ = bool(getattr(self.base_, "weights_applied_", False))
            self.base_.sample_weight_ = None
        self.y_mean_, self.y_std_ = self.base_.y_mean_, self.base_.y_std_
        return self

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_.raw_predict(X[self.features_])

    def predict_proba(self, X: pd.DataFrame):
        return None if self.base_ is None else self.base_.predict_proba(X[self.features_])

    def importance(self) -> pd.Series:
        return self.base_.importance().reindex(self.features_).fillna(0.0)

    def diagnostics(self) -> dict:
        s = self.selection_
        return {"stable_selection": {k: s[k] for k in ("version", "selector", "settings", "n_features", "n_dates", "n_blocks", "chosen", "rejected", "frequencies", "error_control") if k in s}}


def build(kind: str, task: str, params: dict, seed: int) -> StableSelect:
    if str(params.get("base")) == "stable_select":
        raise ValueError("the base model of a stable selection cannot itself be one")
    return StableSelect(kind, task, params, seed)


def selection_stability(folds: list[dict]) -> dict:
    """How steady the selection is across the folds of one run. ``folds`` holds each fold's ``stable_selection`` record (or None for a fold that selected nothing). Reported: the number of folds that selected,
    the inclusion rate of every feature, the mean size of the chosen set, and the mean Jaccard similarity of consecutive selecting folds' sets."""
    sets = [list(f["chosen"]) for f in folds if f and f.get("chosen")]
    n_folds = len(folds)
    if not sets:
        return {"eligibility": "insufficient", "reason": "no fold selected any feature", "folds": n_folds, "folds_selected": 0}
    names = sorted({c for s in sets for c in s})
    inclusion = {c: sum(c in s for s in sets) / len(sets) for c in names}
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(sets[:-1], sets[1:])]
    return {"eligibility": "applied", "folds": n_folds, "folds_selected": len(sets), "inclusion_rate": inclusion, "mean_size": float(np.mean([len(s) for s in sets])),
            "consecutive_jaccard": float(np.mean(jac)) if jac else None, "note": "a high Jaccard says the same features are chosen fold after fold; it says nothing about whether they predict. Folds that selected nothing are "
            "left out of both numbers and counted in folds"}
