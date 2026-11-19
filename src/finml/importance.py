"""Feature importance that holds up out of sample (Lopez de Prado, chapter 8).

Three views, because each answers a different question and each fails differently:

* **MDI** (mean decrease impurity): how much a forest's splits on a feature reduce impurity *in the training sample*. Fast, and biased toward features with many distinct values; it can only say what explained the sample.
* **MDA** (mean decrease accuracy): permute one feature (or one *cluster* of correlated features) in a purged out-of-sample fold and see how much the score falls. It tells what the model uses on data it has not seen.
  Correlated features share their importance, so each looks less important than the group is: ``clusters`` permutes a whole group together.
* **SFI** (single feature importance): the out-of-sample score of a model that sees one feature alone. It has no substitution effect, but it cannot see a feature that matters only in combination with others.

MDA and SFI report the mean and standard deviation across purged folds, and the share of folds in which the feature helped: a feature that helps in some folds and hurts in others is not reliable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from src.finml.cv import purged_folds


def _forest(seed: int, max_features, n_estimators: int = 150) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, min_weight_fraction_leaf=0.02, class_weight="balanced_subsample", random_state=seed, n_jobs=1)


def _score(model, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, scoring: str) -> float:
    if scoring == "accuracy":
        return float(accuracy_score(y, model.predict(X), sample_weight=w))
    return float(-log_loss(y, model.predict_proba(X), sample_weight=w, labels=model.classes_))


def mdi(X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None, n_estimators: int = 200, seed: int = 0) -> pd.DataFrame:
    """Mean decrease impurity over the trees of a forest in which every split considers ONE random feature (``max_features=1``), so that no feature masks another. ``mean`` sums to 1; ``std`` is the standard error across trees."""
    rf = _forest(seed, 1, n_estimators).fit(X, y, sample_weight=sample_weight)
    per_tree = np.array([t.feature_importances_ for t in rf.estimators_])
    return pd.DataFrame({"mean": per_tree.mean(axis=0), "std": per_tree.std(axis=0, ddof=1) / np.sqrt(len(per_tree))}, index=X.columns).sort_values("mean", ascending=False)


def feature_clusters(X: pd.DataFrame, distance: float = 0.5) -> dict[str, list[str]]:
    """Group features whose correlation distance ``sqrt((1 - rho) / 2)`` is below ``distance`` (average linkage). A feature with no close partner is its own group."""
    if X.shape[1] == 1:
        return {"c0": list(X.columns)}
    rho = X.corr().fillna(0.0).to_numpy()
    d = np.sqrt(np.clip((1.0 - rho) / 2.0, 0.0, 1.0))
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, metric="precomputed", linkage="average").fit_predict(d)
    out: dict[str, list[str]] = {}
    for k in sorted(set(lab)):
        out[f"c{k}"] = [c for c, l in zip(X.columns, lab) if l == k]
    return out


def mda(X: pd.DataFrame, y: pd.Series, labels: pd.DataFrame, sample_weight: np.ndarray | None = None, n_splits: int = 5, embargo: int = 0, scoring: str = "neg_log_loss", n_repeats: int = 3,
        clusters: dict[str, list[str]] | None = None, seed: int = 0) -> pd.DataFrame:
    """Out-of-sample permutation importance in purged folds. ``labels`` gives each row's span (``pos0``, ``pos1``) for purging. The importance of a feature (or cluster) in a fold is the fall in the score when it is shuffled
    within the test fold, averaged over ``n_repeats`` shuffles. Returns ``mean``, ``std`` (over folds), ``positive`` (share of folds in which shuffling hurt) and ``folds``."""
    groups = clusters or {c: [c] for c in X.columns}
    w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    yv = y.to_numpy()
    rng = np.random.default_rng(seed)
    drops: dict[str, list[float]] = {g: [] for g in groups}
    for train, test in purged_folds(labels, n_splits, embargo):
        if len(train) < 20 or len(np.unique(yv[train])) < 2 or len(test) == 0:
            continue
        model = _forest(seed, "sqrt").fit(X.iloc[train], yv[train], sample_weight=w[train])
        base = _score(model, X.iloc[test], yv[test], w[test], scoring)
        for g, cols in groups.items():
            fall = []
            for _ in range(n_repeats):
                shuffled = X.iloc[test].copy()
                order = rng.permutation(len(test))
                for c in cols:
                    shuffled[c] = shuffled[c].to_numpy()[order]
                fall.append(base - _score(model, shuffled, yv[test], w[test], scoring))
            drops[g].append(float(np.mean(fall)))
    rows = {g: {"mean": float(np.mean(v)) if v else np.nan, "std": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, "positive": float(np.mean(np.array(v) > 0)) if v else np.nan, "folds": len(v)} for g, v in drops.items()}
    out = pd.DataFrame(rows).T.sort_values("mean", ascending=False)
    out["members"] = [", ".join(groups[g]) for g in out.index]
    return out


def sfi(X: pd.DataFrame, y: pd.Series, labels: pd.DataFrame, sample_weight: np.ndarray | None = None, n_splits: int = 5, embargo: int = 0, scoring: str = "neg_log_loss", seed: int = 0) -> pd.DataFrame:
    """The purged out-of-sample score of a model trained on each feature alone. For ``neg_log_loss``, ``mean`` is the score itself (higher is better; compare it with ``baseline``, the score of the training class frequencies);
    ``gain`` is ``mean - baseline``."""
    w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    yv = y.to_numpy()
    rows = {}
    folds = [(tr, te) for tr, te in purged_folds(labels, n_splits, embargo) if len(tr) >= 20 and len(np.unique(yv[tr])) == 2 and len(te)]
    base = []
    for tr, te in folds:
        p = np.average(yv[tr], weights=w[tr])
        pr = np.tile([1 - p, p], (len(te), 1))
        base.append(float(-log_loss(yv[te], pr, sample_weight=w[te], labels=[0, 1])) if scoring != "accuracy" else float(accuracy_score(yv[te], np.full(len(te), p >= 0.5), sample_weight=w[te])))
    for c in X.columns:
        s = []
        for tr, te in folds:
            m = _forest(seed, 1, 100).fit(X.iloc[tr][[c]], yv[tr], sample_weight=w[tr])
            s.append(_score(m, X.iloc[te][[c]], yv[te], w[te], scoring))
        gains = np.array(s) - np.array(base)
        rows[c] = {"mean": float(np.mean(s)) if s else np.nan, "std": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan, "baseline": float(np.mean(base)) if base else np.nan,
                   "gain": float(np.mean(gains)) if s else np.nan, "positive": float(np.mean(gains > 0)) if s else np.nan}
    return pd.DataFrame(rows).T.sort_values("gain", ascending=False)
