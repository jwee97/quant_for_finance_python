"""Explaining a model: which features it uses, whether the use is stable, and why it scored one asset one way on one day.

* ``importance_table``     the model's own importance (coefficients or tree gains) across the walk-forward refits: the mean, the spread, how often the sign agrees and how stable the ranking is
* ``permutation_importance``  how much the out-of-sample rank IC falls when one feature's values are shuffled across the rows of a refit's test block (the fitted models are reused; nothing is refit)
* ``feature_ic``           each feature's own cross-sectional rank IC with the target, the univariate view that needs no model
* ``shapley_sampling``     a SHAP-style explanation of ONE prediction: Monte Carlo estimates of the Shapley value of each feature, replacing features by those of random background rows. The contributions add up
  exactly to the prediction minus the background's average prediction, whatever the model, and equal the textbook SHAP values for an additive model

Importance is a description of a model, not a causal statement: two correlated features split the credit, and a feature the model leans on in a period when it does not help is still "important".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import Dataset
from .models import Model
from .training import TrainResult


def importance_table(imp: pd.DataFrame) -> pd.DataFrame:
    """Summary over refits of the importance values (``fold x feature``): mean, standard deviation, mean absolute value, share of refits agreeing with the mean's sign, and the average rank. Sorted by
    mean absolute importance."""
    if imp.empty:
        return pd.DataFrame()
    mean, sd = imp.mean(), imp.std(ddof=1)
    sign_agree = imp.apply(lambda col: float((np.sign(col.dropna()) == np.sign(mean[col.name])).mean()) if np.sign(mean[col.name]) != 0 else float("nan"))
    ranks = imp.abs().rank(axis=1, ascending=False)
    out = pd.DataFrame({"mean": mean, "std": sd, "mean_abs": imp.abs().mean(), "sign_agreement": sign_agree, "mean_rank": ranks.mean(), "rank_std": ranks.std(ddof=1)})
    total = out["mean_abs"].sum()
    out["share"] = out["mean_abs"] / total if total > 0 else 0.0
    return out.sort_values("mean_abs", ascending=False)


def rank_stability(imp: pd.DataFrame) -> float:
    """Average Spearman correlation between the importance rankings of consecutive refits: near 1 means the model keeps relying on the same features, near 0 that it reshuffles each time."""
    if len(imp) < 3:
        return float("nan")
    ranks = imp.abs().rank(axis=1)
    corrs = [ranks.iloc[i].corr(ranks.iloc[i + 1]) for i in range(len(ranks) - 1)]
    return float(np.nanmean(corrs))


def _fast_ic(pred: np.ndarray, y: np.ndarray, codes: np.ndarray) -> float:
    """Mean over dates of the cross-sectional rank correlation, for rows grouped by integer ``codes`` (vectorised through group transforms)."""
    df = pd.DataFrame({"p": pred, "y": y, "d": codes}).dropna()
    if df.empty:
        return float("nan")
    g = df.groupby("d")
    pr, yr = g["p"].rank(), g["y"].rank()
    pc, yc = pr - pr.groupby(df["d"]).transform("mean"), yr - yr.groupby(df["d"]).transform("mean")
    num = (pc * yc).groupby(df["d"]).sum()
    den = np.sqrt((pc ** 2).groupby(df["d"]).sum() * (yc ** 2).groupby(df["d"]).sum())
    ic = (num / den.replace(0.0, np.nan))[g.size() >= 3]
    return float(ic.mean()) if len(ic) else float("nan")


def permutation_importance(train: TrainResult, ds: Dataset, max_folds: int = 8, n_repeats: int = 2, seed: int = 0, top: int | None = 30) -> pd.DataFrame:
    """Drop in out-of-sample rank IC when a feature is shuffled, averaged over up to ``max_folds`` evenly spaced refits and ``n_repeats`` shuffles. Positive means the model needs the feature. Only the ``top``
    features by the model's own importance are tested (all if ``None``)."""
    rng = np.random.default_rng(seed)
    ok = [i for i, m in enumerate(train.models) if m is not None and len(train.fold_rows[i]) > 0]
    if not ok:
        return pd.DataFrame()
    picks = [ok[j] for j in np.unique(np.linspace(0, len(ok) - 1, min(max_folds, len(ok))).round().astype(int))]
    feats = list(train.features)
    if top is not None and len(feats) > top:
        order = train.importance.abs().mean().sort_values(ascending=False).index
        feats = [f for f in order if f in feats][:top]
    y_all = ds.Y[train.target].to_numpy(dtype=float)
    pos = ds.positions()
    drops = {f: [] for f in feats}
    for i in picks:
        model, rows = train.models[i], train.fold_rows[i]
        X, y, codes = ds.X.iloc[rows], y_all[rows], pos[rows]
        base = _fast_ic(model.predict(X), y, codes)
        for f in feats:
            if f not in X.columns:
                continue
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[f] = rng.permutation(Xp[f].to_numpy())
                drops[f].append(base - _fast_ic(model.predict(Xp), y, codes))
    out = pd.DataFrame({f: {"mean_drop": float(np.nanmean(v)) if v else float("nan"), "std_drop": float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"), "n": len(v)} for f, v in drops.items()}).T
    return out.sort_values("mean_drop", ascending=False)


def feature_ic(ds: Dataset, target: str | None = None, features: list[str] | None = None) -> pd.DataFrame:
    """Each feature's mean cross-sectional rank IC with the target over all labelled dates, its IC information ratio and the share of positive dates."""
    from .evaluation import ic_summary, rank_ic_wide

    t = target or ds.target
    wide_y = ds.Y[t].unstack("asset").reindex(columns=ds.assets)
    rows = {}
    for f in features or ds.features:
        wide_x = ds.X[f].unstack("asset").reindex(index=wide_y.index, columns=ds.assets)
        if f in ds.meta.get("broadcast", []):
            continue                                                              # constant across assets: no cross-sectional rank to speak of
        s = ic_summary(rank_ic_wide(wide_x, wide_y), ds.horizon)
        rows[f] = {"mean_ic": s["mean"], "ic_ir": s["ir"], "t": s["t"], "hit_rate": s["hit_rate"]}
    return pd.DataFrame(rows).T.sort_values("mean_ic", key=lambda s: -s.abs()) if rows else pd.DataFrame()


def shapley_sampling(model: Model, x_row: pd.DataFrame, background: pd.DataFrame, n_perm: int = 40, seed: int = 0) -> dict:
    """Monte Carlo Shapley values of one prediction. ``x_row`` is a one-row frame, ``background`` a sample of rows representing "what the model sees on an ordinary day". For each of ``n_perm`` random
    orderings of the features, a random background row is turned into ``x_row`` one feature at a time and the change in the prediction is credited to the feature just switched. Returns the value of
    each feature, the base (the average prediction on the background rows used) and the prediction; ``sum(values) == prediction - base`` exactly."""
    feats = [f for f in model.features_ if f in x_row.columns]
    if not feats:
        raise ValueError("none of the model's features is in the row")
    rng = np.random.default_rng(seed)
    x = x_row[feats].iloc[0].to_numpy(dtype=float)
    bg = background[feats].to_numpy(dtype=float)
    d = len(feats)
    blocks, orders, bases = [], [], []
    for _ in range(n_perm):
        order = rng.permutation(d)
        b = bg[rng.integers(len(bg))]
        mat = np.tile(b, (d + 1, 1))
        for k in range(d):
            mat[k + 1:, order[k]] = x[order[k]]                                    # row k+1 has the first k+1 features of the ordering set to x
        blocks.append(mat)
        orders.append(order)
        bases.append(b)
    frame = pd.DataFrame(np.vstack(blocks), columns=feats)
    for c in model.features_:                                                      # a model fitted on more columns than the row names (none expected) keeps the background there
        if c not in frame.columns:
            frame[c] = float(background[c].median()) if c in background.columns else 0.0
    pred = np.asarray(model.predict(frame[model.features_]), dtype=float).reshape(n_perm, d + 1)
    phi = np.zeros(d)
    for p, order in zip(pred, orders):
        delta = np.diff(p)
        phi[order] += delta
    phi /= n_perm
    base = float(pred[:, 0].mean())
    full = float(pred[:, -1].mean())
    return {"values": pd.Series(phi, index=feats), "base": base, "prediction": full}


def explain_prediction(model: Model, ds: Dataset, date, asset: str, n_background: int = 200, n_perm: int = 40, seed: int = 0, top: int = 12) -> dict:
    """Why did ``model`` score ``asset`` as it did on ``date``? The top contributing features with their (normalised) values, the base and the score. The background is a random sample of rows from the
    ``200`` days before ``date`` so that nothing after the day is used."""
    d = pd.Timestamp(date)
    try:
        row = ds.X.loc[[(d, asset)]]
    except KeyError:
        raise ValueError(f"no features for {asset} on {d.date()}: the asset was not investable then or the date is outside the data") from None
    prior = ds.X[ds.X.index.get_level_values(0) < d]
    if prior.empty:
        raise ValueError("there are no earlier rows to use as a background")
    recent = prior.iloc[-min(len(prior), 200 * len(ds.assets)):]
    bg = recent.sample(min(n_background, len(recent)), random_state=seed)
    res = shapley_sampling(model, row, bg, n_perm, seed)
    vals = res["values"].sort_values(key=lambda s: -s.abs()).head(top)
    return {"date": str(d.date()), "asset": asset, "prediction": res["prediction"], "base": res["base"], "other": float(res["values"].drop(vals.index).sum()),
            "contributions": [{"feature": f, "value": float(row[f].iloc[0]) if np.isfinite(row[f].iloc[0]) else None, "contribution": float(v)} for f, v in vals.items()]}
