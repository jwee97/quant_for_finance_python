"""Putting alphas together: how alike they are, which belong in the same cluster, and how to weigh them into one signal, by simple rules and by machine-learning models trained without a glance ahead.

**Correlation and clusters.** Two alphas that rank the companies the same way are one idea. The correlation of two signals is the average over dates of the rank correlation across companies; the
distance is ``1 - |correlation|`` and the alphas are grouped by average-linkage hierarchical clustering. The *effective number of independent alphas* (from the eigenvalues of the correlation matrix)
says how many separate ideas a set really holds.

**Rules for combining** (each standardises the alphas on every date, then weighs them using only what was known on that date):
``equal`` weight; ``ic`` weight by the trailing mean IC; ``icir`` by the trailing information ratio of the IC; ``diversified`` the weights that maximise the information ratio of the combination given
how the ICs move together (Qian, Hua and Sorensen), after the alphas have been orthogonalised, with the covariance shrunk towards its diagonal; ``cluster`` equal weight to each cluster and then
within it, so that five copies of one idea do not outvote one different idea. Weights on date ``t`` use only the ICs whose forward windows had ended by ``t``.

**Machine-learning combination.** A model learns to predict the cross-section of next-period returns from the alphas. It is trained on an expanding or rolling window, retrained every so often, and each
prediction is made by a model that saw only dates strictly before the prediction date minus a purge as long as the forecast horizon (so that no training target overlaps the prediction window) plus an
embargo. The predictions are therefore out of sample throughout and go through the same evaluation as any signal. Linear models (ridge, lasso, elastic net) and trees (random forest, extra trees,
gradient boosting, LightGBM, XGBoost) are offered; importance comes from the fitted model, from permutation (how much the IC falls when one alpha is shuffled) and, for LightGBM and XGBoost, from
their built-in TreeSHAP contributions. A model is no better than the simple rules until it beats them out of sample, and the report puts the two side by side.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from ..equity import alpha_model
from ..models.regression import cross_sectional_ic
from . import transforms
from .errors import FundamentalsError
from .evaluate import PERIODS_PER_YEAR, ic_stats, newey_west, overlap_lags, rank_ic
from .research import FREQ_DAYS, ResearchContext, horizon_days

METHODS = ("equal", "ic", "icir", "diversified", "cluster")
MODELS = ("ridge", "lasso", "elastic_net", "random_forest", "extra_trees", "gradient_boosting", "lightgbm", "xgboost")


# ------------------------------------------------------------------------------------------------------------------ correlation and clusters
def standardise(signals: dict[str, pd.DataFrame], ctx: ResearchContext) -> dict[str, pd.DataFrame]:
    """Each signal winsorised at 1% and z-scored on every date among the investable companies."""
    return {k: transforms.apply(v, [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}], ctx) for k, v in signals.items()}


def correlation(signals: dict[str, pd.DataFrame], min_assets: int = 10) -> pd.DataFrame:
    """The average over dates of the rank correlation between every pair of signals across companies.

    Each signal is ranked among its own valid companies on each date; the correlation of two signals on a date uses the companies valid in both (so it equals Spearman's correlation when their coverage
    is the same, and is a close cousin of it when it is not), and a date with fewer than ``min_assets`` such companies is skipped."""
    names = list(signals)
    k = len(names)
    out = pd.DataFrame(np.eye(k), index=names, columns=names)
    if k < 2:
        return out
    first = signals[names[0]]
    ranks = np.stack([signals[n].reindex(index=first.index, columns=first.columns).rank(axis=1).to_numpy(float) for n in names], axis=-1)      # dates x companies x signals
    valid = np.isfinite(ranks)
    total, count = np.zeros((k, k)), np.zeros((k, k))
    for t in range(ranks.shape[0]):
        m = valid[t].astype(float)
        x = np.where(valid[t], ranks[t], 0.0)
        n = m.T @ m                                                            # companies valid in both
        sx = x.T @ m                                                           # sum of a's ranks over those companies
        sxy, sxx = x.T @ x, (x * x).T @ m
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sxy / n - sx * sx.T / n ** 2
            va = sxx / n - (sx / n) ** 2
            r = cov / np.sqrt(va * va.T)
        r = np.where((n >= min_assets) & np.isfinite(r), np.clip(r, -1.0, 1.0), np.nan)
        total += np.nan_to_num(r)
        count += np.isfinite(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    np.fill_diagonal(mean, 1.0)
    return pd.DataFrame(mean, index=names, columns=names)


def effective_number(corr: pd.DataFrame) -> float:
    """How many independent ideas a set of alphas holds: ``(sum of eigenvalues)^2 / sum of squared eigenvalues`` of the correlation matrix (one for identical alphas, k for unrelated ones)."""
    values = np.nan_to_num(corr.to_numpy(float), nan=0.0)
    np.fill_diagonal(values, 1.0)
    eig = np.clip(np.linalg.eigvalsh((values + values.T) / 2.0), 0.0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum()) if (eig ** 2).sum() > 0 else float("nan")


def cluster(corr: pd.DataFrame, cut: float = 0.5) -> dict:
    """Hierarchical clustering of alphas on the distance ``1 - |correlation|`` (average linkage). Returns the leaf order for a heatmap, the cluster each alpha falls in when the tree is cut at
    distance ``cut``, the merge steps for a dendrogram, and the effective number of independent alphas."""
    names = list(corr.index)
    if len(names) < 2:
        return {"order": names, "labels": {n: 1 for n in names}, "links": [], "effective_number": float(len(names)), "cut": cut}
    dist = 1.0 - np.abs(np.nan_to_num(corr.to_numpy(float), nan=0.0))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    link = hierarchy.linkage(squareform(np.clip(dist, 0.0, None), checks=False), method="average")
    order = [names[i] for i in hierarchy.leaves_list(link)]
    labels = hierarchy.fcluster(link, t=cut, criterion="distance")
    return {"order": order, "labels": {n: int(c) for n, c in zip(names, labels)}, "links": [[int(a), int(b), float(d), int(n)] for a, b, d, n in link], "effective_number": effective_number(corr),
            "cut": cut}


def heatmap(corr: pd.DataFrame, order: list[str]) -> dict:
    ordered = corr.loc[order, order]
    return {"names": order, "values": [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in ordered.to_numpy(float)]}


def analyse(ctx: ResearchContext, signals: dict[str, pd.DataFrame], cut: float = 0.5, min_assets: int = 10, redundant_above: float = 0.8) -> dict:
    """The correlation heatmap, the clusters and the effective number of independent alphas for a set of signals, ready for a page: the heatmap is in leaf order, each cluster lists its members and
    how alike they are, and ``redundant`` lists the pairs so alike that one of them adds nothing."""
    if len(signals) < 2:
        raise FundamentalsError("the correlation of alphas needs at least two of them")
    std = standardise(signals, ctx)
    corr = correlation(std, min_assets)
    tree = cluster(corr, cut)
    groups: dict[int, list[str]] = {}
    for name in tree["order"]:
        groups.setdefault(tree["labels"][name], []).append(name)
    clusters = []
    for label, members in groups.items():
        block = np.abs(corr.loc[members, members].to_numpy(float))
        off = block[~np.eye(len(members), dtype=bool)]
        clusters.append({"id": int(label), "members": members, "mean_abs_correlation": float(np.nanmean(off)) if off.size and np.isfinite(off).any() else None})
    names = list(corr.index)
    redundant = sorted(({"a": a, "b": b, "correlation": float(corr.loc[a, b])} for i, a in enumerate(names) for b in names[i + 1:] if abs(corr.loc[a, b]) >= redundant_above),
                       key=lambda r: -abs(r["correlation"]))
    return {"names": names, "heatmap": heatmap(corr, tree["order"]), "order": tree["order"], "clusters": clusters, "links": tree["links"], "cut": cut, "effective_number": tree["effective_number"],
            "alphas": len(names), "redundant": redundant,
            "reading": (f"{len(names)} alphas hold about {tree['effective_number']:.1f} independent ideas; {len(clusters)} clusters at distance {cut}"
                        + (f"; {len(redundant)} pair(s) are {redundant_above:.0%} or more alike, so one of each pair adds almost nothing" if redundant else ""))}


# ------------------------------------------------------------------------------------------------------------------ rules
@dataclass
class Combination:
    signal: pd.DataFrame
    weights: pd.DataFrame
    ic: pd.DataFrame
    method: str
    notes: list = field(default_factory=list)


def combine(ctx: ResearchContext, signals: dict[str, pd.DataFrame], method: str = "equal", horizon="1m", window: int = 36, min_obs: int = 12, shrink: float = 0.3,
            orthogonalize: bool | str = False, nonnegative: bool = True, cut: float = 0.5) -> Combination:
    """One signal from several, by one of :data:`METHODS`. Every signal must point the same way (higher means expect more; the library's ``build_signal`` applies each alpha's learned sign). The weights
    on each date use only ICs measured on dates whose forward windows had ended by then (``ctx.settled_lag``); before ``min_obs`` such dates the weights are equal. An alpha that is missing for a
    company counts as *no view* (zero after standardising, so it neither helps nor hurts the company) and a company with every alpha missing gets no signal; the result is z-scored across companies."""
    if method not in METHODS:
        raise FundamentalsError(f"method must be one of {METHODS}")
    if len(signals) < 2:
        raise FundamentalsError("combining needs at least two alphas")
    std = standardise(signals, ctx)
    names = list(std)
    fwd = ctx.forward(horizon)
    lag = ctx.settled_lag(horizon)
    notes: list[str] = []
    if method == "diversified":
        model = alpha_model.optimal_alpha(std, fwd, horizon=lag, window=window, min_obs=min_obs, shrink=shrink, orthogonalize=orthogonalize, nonnegative=nonnegative, investable=ctx.investable())
        notes.append(f"weights from the information-ratio maximum over the last {window} ICs, covariance shrunk {shrink:.0%} towards its diagonal" + (f", alphas orthogonalised ({orthogonalize})" if orthogonalize else ""))
        return Combination(model.alpha, model.weights.reindex(columns=names), model.ic, method, notes)
    ic = pd.DataFrame({k: cross_sectional_ic(v, fwd.reindex(v.index), "spearman", 10) for k, v in std.items()}).reindex(next(iter(std.values())).index)
    equal = pd.Series(1.0 / len(names), index=names)
    static = None
    if method == "cluster":
        labels = pd.Series(cluster(correlation(std), cut)["labels"]).reindex(names)
        static = (1.0 / (labels.nunique() * labels.map(labels.value_counts()))).astype(float)
        notes.append(f"{labels.nunique()} clusters of alphas at distance {cut} (from how alike the alphas themselves are, no returns used): "
                     + "; ".join(", ".join(members.index) for _, members in labels.groupby(labels)) + "; equal weight to each cluster, then equally within it")
    elif method == "ic":
        notes.append(f"weights proportional to the mean IC over the last {window} dates whose forward windows had ended (negative means set to zero)")
    elif method == "icir":
        notes.append(f"weights proportional to mean IC / standard deviation of IC over the last {window} dates whose forward windows had ended (negative set to zero)")
    rows = {}
    for i, t in enumerate(ic.index):
        known = ic.iloc[: max(i - lag + 1, 0)].dropna().iloc[-window:]
        if method == "cluster":
            w = static
        elif method == "equal" or len(known) < min_obs:
            w = equal
        elif method == "ic":
            w = known.mean().clip(lower=0.0)
        else:
            w = (known.mean() / known.std(ddof=1).replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
        total = float(w.sum())
        rows[t] = w / total if total > 0 else equal
    if method in ("ic", "icir"):
        notes.append(f"until {min_obs} such dates are known the weights are equal")
    weights = pd.DataFrame(rows).T.reindex(columns=names)
    return Combination(alpha_model.combine_factors(std, weights), weights, ic, method, notes)


# ------------------------------------------------------------------------------------------------------------------ machine learning
def _optional(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise FundamentalsError(f"the {module} model needs the optional package: pip install {module}") from exc


PARAM_LIMITS = {"alpha": (0.0, 1e6), "l1_ratio": (0.0, 1.0), "n_estimators": (1, 2000), "max_depth": (1, 12), "min_samples_leaf": (1, 100_000), "learning_rate": (0.001, 1.0)}


def _clean_params(params: dict | None) -> dict:
    """Model settings from a caller (a page, a file): only the known names, only numbers, only within sensible bounds."""
    out: dict = {}
    for key, value in (params or {}).items():
        if key not in PARAM_LIMITS:
            raise FundamentalsError(f"unknown model setting {key!r}; the settings are {sorted(PARAM_LIMITS)}")
        try:
            x = float(value)
        except (TypeError, ValueError):
            raise FundamentalsError(f"model setting {key!r} must be a number") from None
        low, high = PARAM_LIMITS[key]
        if not low <= x <= high:
            raise FundamentalsError(f"model setting {key!r} must be between {low:g} and {high:g}")
        out[key] = int(x) if key in ("n_estimators", "max_depth", "min_samples_leaf") else x
    return out


def _model(name: str, params: dict, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    if name == "ridge":
        return Ridge(alpha=params.get("alpha", 1000.0))
    if name == "lasso":
        return Lasso(alpha=params.get("alpha", 0.01), max_iter=5000)
    if name == "elastic_net":
        return ElasticNet(alpha=params.get("alpha", 0.01), l1_ratio=params.get("l1_ratio", 0.5), max_iter=5000)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=params.get("n_estimators", 120), max_depth=params.get("max_depth", 4), min_samples_leaf=params.get("min_samples_leaf", 200), max_features=0.7,
                                     n_jobs=-1, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=params.get("n_estimators", 150), max_depth=params.get("max_depth", 4), min_samples_leaf=params.get("min_samples_leaf", 200), max_features=0.7,
                                   n_jobs=-1, random_state=seed)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=params.get("n_estimators", 120), max_depth=params.get("max_depth", 2), learning_rate=params.get("learning_rate", 0.05), subsample=0.7,
                                         min_samples_leaf=params.get("min_samples_leaf", 200), random_state=seed)
    if name == "lightgbm":
        lightgbm = _optional("lightgbm")
        return lightgbm.LGBMRegressor(n_estimators=params.get("n_estimators", 150), max_depth=params.get("max_depth", 3), learning_rate=params.get("learning_rate", 0.05), subsample=0.7,
                                      subsample_freq=1, min_child_samples=params.get("min_samples_leaf", 200), colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1)
    if name == "xgboost":
        xgboost = _optional("xgboost")
        return xgboost.XGBRegressor(n_estimators=params.get("n_estimators", 150), max_depth=params.get("max_depth", 3), learning_rate=params.get("learning_rate", 0.05), subsample=0.7,
                                    min_child_weight=params.get("min_samples_leaf", 200) / 10.0, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=0)
    raise FundamentalsError(f"model must be one of {MODELS}")


GRIDS = {"ridge": [{"alpha": 10.0}, {"alpha": 300.0}, {"alpha": 10000.0}], "lasso": [{"alpha": 0.0005}, {"alpha": 0.005}, {"alpha": 0.05}],
         "elastic_net": [{"alpha": 0.002, "l1_ratio": 0.3}, {"alpha": 0.01, "l1_ratio": 0.7}], "random_forest": [{"max_depth": 3}, {"max_depth": 6}],
         "extra_trees": [{"max_depth": 3}, {"max_depth": 6}], "gradient_boosting": [{"max_depth": 2}, {"max_depth": 3}], "lightgbm": [{"max_depth": 2}, {"max_depth": 4}],
         "xgboost": [{"max_depth": 2}, {"max_depth": 4}]}


def _gaussian_rank(target: pd.DataFrame) -> pd.DataFrame:
    """The cross-section of next-period returns on each date turned into normal scores of its ranks: robust to outliers and about the order of companies, not the size of the moves."""
    pct = target.rank(axis=1, pct=True, method="average")
    n = target.notna().sum(axis=1).to_numpy(float)[:, None]
    adj = (pct * n - 0.5) / np.where(n > 0, n, 1.0)
    return pd.DataFrame(stats.norm.ppf(np.clip(adj, 1e-6, 1 - 1e-6)), index=target.index, columns=target.columns).where(target.notna())


@dataclass
class MLCombination:
    signal: pd.DataFrame
    folds: list
    importance: dict
    model: str
    horizon: object = "1m"
    notes: list = field(default_factory=list)


def ml_combine(ctx: ResearchContext, signals: dict[str, pd.DataFrame], model: str = "ridge", horizon="1m", min_train: int = 36, retrain_every: int = 12, window: int | None = None,
               embargo: int = 1, tune: bool = False, seed: int = 0, permutation_repeats: int = 3, max_rows: int = 400_000, params: dict | None = None) -> MLCombination:
    """Train ``model`` walk-forward to predict the cross-section of next-period returns from the alphas, and return its out-of-sample predictions as a signal.

    For each retraining date the training rows are every (date, company) at least ``purge + embargo`` periods before it (``purge`` is the number of periods a forecast window overlaps, so no
    training target reaches into the prediction period); with ``window`` the rows are the most recent ``window`` periods only (rolling), otherwise all of them (expanding). A missing alpha value is
    set to zero after standardising (no view), which is a stated rule; a company with every alpha missing is left out. ``params`` overrides the model's settings (see :func:`_model`); with ``tune`` a
    small grid is tried on the last quarter of each training window on top of them.
    """
    if model not in MODELS:
        raise FundamentalsError(f"model must be one of {MODELS}")
    names = list(signals)
    if len(names) < 2:
        raise FundamentalsError("combining needs at least two alphas")
    if min(int(min_train), int(retrain_every)) < 1 or int(embargo) < 0 or (window is not None and int(window) < 1):
        raise FundamentalsError("min_train and retrain_every must be at least 1, embargo at least 0, and window (if given) at least 1")
    std = standardise(signals, ctx)
    dates, ids = ctx.dates, ctx.ids
    cube = np.stack([std[k].reindex(index=dates, columns=ids).to_numpy(float) for k in names], axis=-1)           # dates x companies x alphas
    seen = np.isfinite(cube).any(axis=-1)
    features = np.nan_to_num(cube, nan=0.0)
    target = _gaussian_rank(ctx.forward(horizon).where(ctx.investable())).to_numpy(float)
    purge = ctx.settled_lag(horizon) + int(embargo)
    n = len(dates)
    predictions = np.full((n, len(ids)), np.nan)
    folds, notes, importance_acc = [], [], np.zeros(len(names))
    perm_acc, tree_acc, shap_acc = np.zeros(len(names)), np.zeros(len(names)), np.zeros(len(names))
    n_perm = n_tree = n_shap = 0
    base = _clean_params(params)
    grid = GRIDS[model] if tune else [{}]
    start = max(min_train + purge, 1)
    for s in range(start, n, retrain_every):
        stop = min(s + retrain_every, n)
        lo = 0 if window is None else max(0, s - purge - window)
        train_dates = np.arange(lo, s - purge)
        xs, ys = features[train_dates], target[train_dates]
        ok = seen[train_dates] & np.isfinite(ys)
        if ok.sum() < 500:
            continue
        X, y = xs[ok], ys[ok]
        if len(y) > max_rows:
            pick = np.random.default_rng(seed + s).choice(len(y), max_rows, replace=False)
            X, y = X[pick], y[pick]
        chosen: dict = {}
        if tune and len(grid) > 1:
            cut = len(train_dates) * 3 // 4
            inner_train, inner_val = train_dates[:max(cut - purge, 0)], train_dates[cut:]
            okt, okv = seen[inner_train] & np.isfinite(target[inner_train]), seen[inner_val] & np.isfinite(target[inner_val])
            if okt.sum() > 500 and okv.sum() > 200:
                best = None
                for candidate in grid:
                    m = _model(model, {**base, **candidate}, seed).fit(features[inner_train][okt], target[inner_train][okt])
                    pred = np.full(target[inner_val].shape, np.nan)
                    pred[okv] = m.predict(features[inner_val][okv])
                    score = cross_sectional_ic(pd.DataFrame(pred), pd.DataFrame(target[inner_val]), "spearman", 20).mean()
                    if best is None or (np.isfinite(score) and score > best[0]):
                        best = (score, candidate)
                chosen = best[1] if best else {}
        used = {**base, **chosen}
        fitted = _model(model, used, seed).fit(X, y)
        pred_dates = np.arange(s, stop)
        pok = seen[pred_dates]
        block = np.full((len(pred_dates), len(ids)), np.nan)
        block[pok] = fitted.predict(features[pred_dates][pok])
        predictions[pred_dates] = block
        realised = pd.DataFrame(block, index=dates[pred_dates])
        fold_ic = cross_sectional_ic(realised, pd.DataFrame(target[pred_dates], index=dates[pred_dates]), "spearman", 20)
        folds.append({"train_from": str(dates[train_dates[0]].date()), "train_to": str(dates[train_dates[-1]].date()), "predict_from": str(dates[s].date()), "predict_to": str(dates[stop - 1].date()),
                      "train_rows": int(len(y)), "params": used, "oos_ic": float(fold_ic.mean()) if len(fold_ic) else float("nan")})
        # what the fitted model says about the alphas
        if hasattr(fitted, "coef_"):
            importance_acc += np.abs(np.asarray(fitted.coef_, dtype=float).ravel())
        elif hasattr(fitted, "feature_importances_"):
            tree_acc += np.asarray(fitted.feature_importances_, dtype=float)
            n_tree += 1
        # permutation: how much the IC falls when one alpha is shuffled across companies within each date
        if permutation_repeats and len(fold_ic):
            reference = float(fold_ic.mean())
            rng = np.random.default_rng(seed + s)
            drops = np.zeros(len(names))
            for j in range(len(names)):
                for _ in range(permutation_repeats):
                    shuffled = features[pred_dates].copy()
                    for t in range(len(pred_dates)):
                        idx = np.flatnonzero(pok[t])
                        shuffled[t, idx, j] = shuffled[t, rng.permutation(idx), j]
                    again = np.full(block.shape, np.nan)
                    again[pok] = fitted.predict(shuffled[pok])
                    ic2 = cross_sectional_ic(pd.DataFrame(again, index=dates[pred_dates]), pd.DataFrame(target[pred_dates], index=dates[pred_dates]), "spearman", 20)
                    drops[j] += (reference - float(ic2.mean())) / permutation_repeats if len(ic2) else 0.0
            perm_acc += drops
            n_perm += 1
        # TreeSHAP from the boosting libraries themselves (no extra package)
        if model in ("lightgbm", "xgboost"):
            sample = features[pred_dates][pok][:2000]
            if len(sample):
                if model == "lightgbm":
                    contrib = fitted.predict(sample, pred_contrib=True)[:, :-1]
                else:
                    contrib = fitted.get_booster().predict(_optional("xgboost").DMatrix(sample), pred_contribs=True)[:, :-1]
                shap_acc += np.abs(contrib).mean(axis=0)
                n_shap += 1
    if not folds:
        raise FundamentalsError(f"not enough history to train: needs {min_train} periods before the first prediction, and {purge} more for the purge and embargo")
    importance = {"alphas": names}
    if importance_acc.sum() > 0:
        coefficient = importance_acc / len(folds)
        importance["coefficients"] = {k: float(v) for k, v in zip(names, coefficient)}
    if n_tree:
        built_in = tree_acc / n_tree
        importance["built_in"] = {k: float(v) for k, v in zip(names, built_in / built_in.sum() if built_in.sum() > 0 else built_in)}
        importance["built_in_kind"] = {"lightgbm": "share of the splits", "xgboost": "share of the average gain"}.get(model, "share of the impurity decrease")
    if n_perm:
        importance["permutation"] = {k: float(v) for k, v in zip(names, perm_acc / n_perm)}
    importance["shap"] = ({k: float(v) for k, v in zip(names, shap_acc / n_shap)} if n_shap else {"available": False, "reason": "TreeSHAP is built into the LightGBM and XGBoost models only"})
    out = pd.DataFrame(predictions, index=dates, columns=ids)
    notes.append(f"{len(folds)} walk-forward folds; every prediction comes from a model trained on dates at least {purge} periods earlier (purge {purge - int(embargo)}, embargo {int(embargo)})"
                 + ("; a small grid of settings was tried on the last quarter of each training window" if tune else ""))
    return MLCombination(out, folds, importance, model, horizon, notes)


def compare_with_rules(ctx: ResearchContext, ml: MLCombination, signals: dict[str, pd.DataFrame], horizon=None, rules: tuple = ("equal", "icir")) -> dict:
    """The model's out-of-sample rank IC against simple rules over the very same dates, and against the best single alpha. The gap to each rule is tested (Newey-West t on the difference of the two IC
    series, which allows for the overlap of forecast windows): a model that cannot beat a plain average or an IC-weighted one is not earning its complexity."""
    horizon = ml.horizon if horizon is None else horizon
    fwd = ctx.forward(horizon)
    dates = ml.signal.dropna(how="all").index
    h, step, ppy = horizon_days(horizon), FREQ_DAYS[ctx.freq], PERIODS_PER_YEAR[ctx.freq]
    model_ic = rank_ic(ml.signal.loc[dates], fwd.reindex(dates), 20)
    out = {"dates": int(len(model_ic)), "model": ic_stats(model_ic, h, step, ppy), "rules": {}}
    lags = max(overlap_lags(h, step), int(np.floor(4.0 * (max(len(model_ic), 1) / 100.0) ** (2.0 / 9.0))))
    wins = []
    for rule in rules:
        sig = combine(ctx, signals, rule, horizon).signal
        ic = rank_ic(sig.reindex(dates), fwd.reindex(dates), 20)
        gap = (model_ic - ic.reindex(model_ic.index)).dropna()
        nw = newey_west(gap, lags)
        out["rules"][rule] = {**ic_stats(ic, h, step, ppy), "model_minus_rule": nw["mean"], "model_minus_rule_t": nw["t"]}
        wins.append(bool(np.isfinite(nw["t"]) and nw["mean"] > 0 and nw["t"] > 1.96))
    singles = {k: float(rank_ic(v.reindex(dates), fwd.reindex(dates), 20).mean()) for k, v in signals.items()}
    best = max(singles, key=lambda k: abs(singles[k]))
    out["best_single"] = {"alpha": best, "mean_ic": singles[best], "note": "chosen after looking at these same dates, so a generous yardstick"}
    out["model_beats_rules"] = bool(wins) and all(wins)
    out["verdict"] = ("The model beats every simple rule out of sample by more than chance explains." if out["model_beats_rules"] else
                      "The model does not clearly beat the simple rules out of sample: its extra complexity is not earning anything that can be told from chance.")
    return out


__all__ = ["METHODS", "MODELS", "Combination", "MLCombination", "combine", "ml_combine", "compare_with_rules", "correlation", "cluster", "effective_number", "heatmap", "standardise", "analyse"]
