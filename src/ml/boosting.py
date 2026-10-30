"""Rankers and probabilistic boosters behind the ``Model`` interface.

``lgbm_ranker`` and ``xgb_ranker`` are learning-to-rank models (LambdaMART and XGBoost's ranking objectives): every date is one query, the assets of that date are its documents, and the label is the
asset's return bucket within the date, so the model learns the ORDER of assets, which is what a cross-sectional strategy trades. Their scores are standardised with the mean and spread of the training scores (a ranker's raw score has no scale of its own), so a row's
score does not depend on which other rows are scored with it.

``lgbm_quantile`` fits three LightGBM quantile regressions and ``ngboost`` a Normal predictive distribution; both report a prediction interval and a standard deviation through ``predict_extras``,
which is what the uncertainty-aware position rule and the interval chart read.

Nothing here looks at another date than the row it predicts: training uses the rows it is given and a prediction is a function of the row alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .models import Model, _import

MIN_QUERY = 3                                                # a date with fewer assets than this teaches a ranker nothing


def _date_keys(X: pd.DataFrame, dates) -> np.ndarray:
    return np.asarray(dates) if dates is not None else X.index.get_level_values(0).to_numpy()


def relevance_buckets(y: np.ndarray, keys: np.ndarray, levels: int) -> np.ndarray:
    """Integer relevance 0 .. levels-1 for each row: the return bucket of the asset among that date's assets (the best bucket is levels-1)."""
    pct = pd.Series(y).groupby(keys).rank(pct=True, method="average").to_numpy()
    return np.clip(np.ceil(pct * levels).astype(int) - 1, 0, levels - 1)


def query_order(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row order that puts each date's rows together, and the size of each date's group, in date order."""
    order = np.argsort(keys, kind="stable")
    _, counts = np.unique(keys[order], return_counts=True)
    return order, counts


class _Tabular(Model):
    """Shared plumbing: column choice, median imputation for learners that cannot take a missing value, and the walk-forward-safe prediction frame."""

    def _values(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        values = (self._select(X) if fit else X[self.features_]).to_numpy(dtype=np.float64)
        if getattr(self, "impute", False):
            if fit:
                med = np.nanmedian(values, axis=0)
                self.medians_ = np.where(np.isfinite(med), med, 0.0)
            values = np.where(np.isnan(values), self.medians_, values)
        return values


class RankerModel(_Tabular):
    """LambdaMART (LightGBM) or an XGBoost ranking objective, one query per date."""

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        self.est_ = None
        self.impute = False

    def _estimator(self, levels: int):
        p, seed = self.params, self.seed
        if self.kind == "lgbm_ranker":
            lgb = _import("lightgbm")
            return lgb.LGBMRanker(objective="lambdarank", n_estimators=int(p["n_estimators"]), num_leaves=int(p["num_leaves"]), learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
                                  subsample_freq=1, colsample_bytree=float(p["colsample_bytree"]), min_child_samples=int(p["min_child_samples"]), reg_lambda=float(p["reg_lambda"]), n_jobs=1,
                                  random_state=seed, verbosity=-1, force_row_wise=True, importance_type="gain", label_gain=[2 ** i - 1 for i in range(max(levels, 2))])
        xgb = _import("xgboost")
        return xgb.XGBRanker(objective=str(p["objective"]), n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]), learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
                             colsample_bytree=float(p["colsample_bytree"]), min_child_weight=float(p["min_child_weight"]), reg_lambda=float(p["reg_lambda"]), tree_method="hist", n_jobs=1,
                             random_state=seed, verbosity=0)

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "RankerModel":
        values = self._values(X, fit=True)
        self.features_ = list(self._select(X).columns)
        self.n_train_ = len(values)
        keys, target = _date_keys(X, dates), y.to_numpy(dtype=float)
        sizes = pd.Series(keys).map(pd.Series(keys).value_counts()).to_numpy()
        keep = np.isfinite(target) & (sizes >= MIN_QUERY)
        if keep.sum() < 50:
            raise ValueError("too few dates with enough assets to train a ranker")
        levels = 2 if self.task == "classification" else int(self.params["relevance_levels"])
        if not 2 <= levels <= 10:
            raise ValueError("relevance_levels must be between 2 and 10")
        rel = target[keep].astype(int) if self.task == "classification" else relevance_buckets(target[keep], keys[keep], levels)
        order, counts = query_order(keys[keep])
        Xk, rel = values[keep][order], rel[order]
        self.est_ = self._estimator(levels)
        if self.kind == "lgbm_ranker":
            self.est_.fit(Xk, rel, group=counts)
        else:
            self.est_.fit(Xk, rel, qid=np.repeat(np.arange(len(counts)), counts))
        self.levels_ = levels
        raw = self.est_.predict(values)
        self.score_mu_, self.score_sd_ = float(raw.mean()), float(raw.std()) or 1.0
        return self

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.est_.predict(self._values(X)) - self.score_mu_) / self.score_sd_

    def importance(self) -> pd.Series:
        return pd.Series(np.asarray(self.est_.feature_importances_, dtype=float), index=self.features_)

    def diagnostics(self) -> dict:
        return {"relevance_levels": getattr(self, "levels_", None), "n_estimators": int(self.params["n_estimators"])}


class QuantileLGBM(_Tabular):
    """Low, median and high LightGBM quantile regressions: the median is the score, the outer two are a prediction interval."""

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        lo, hi = float(self.params["alpha_low"]), float(self.params["alpha_high"])
        if not 0.0 < lo < 0.5 < hi < 1.0:
            raise ValueError("alpha_low must be below 0.5 and alpha_high above it, both between 0 and 1")
        self.alphas_ = (lo, 0.5, hi)
        self.ests_: list = []
        self.impute = False

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "QuantileLGBM":
        lgb, p = _import("lightgbm"), self.params
        values = self._values(X, fit=True)
        self.features_ = list(self._select(X).columns)
        self.n_train_ = len(values)
        target = self._prepare_y(y)
        w = self._weights(dates)
        self.ests_ = []
        for a in self.alphas_:
            est = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=int(p["n_estimators"]), num_leaves=int(p["num_leaves"]), learning_rate=float(p["learning_rate"]),
                                    subsample=float(p["subsample"]), subsample_freq=1, colsample_bytree=float(p["colsample_bytree"]), min_child_samples=int(p["min_child_samples"]),
                                    reg_lambda=float(p["reg_lambda"]), n_jobs=1, random_state=self.seed, verbosity=-1, force_row_wise=True, importance_type="gain")
            self.ests_.append(est.fit(values, target, sample_weight=w))
        return self

    def _quantiles(self, X: pd.DataFrame) -> np.ndarray:
        values = self._values(X)
        q = np.sort(np.column_stack([e.predict(values) for e in self.ests_]), axis=1)               # quantiles may cross; sorting is the standard repair
        return q * self.y_std_ + self.y_mean_

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._quantiles(X)[:, 1]

    def predict_extras(self, X: pd.DataFrame) -> dict:
        q = self._quantiles(X)
        z = stats.norm.ppf(self.alphas_[2]) - stats.norm.ppf(self.alphas_[0])
        return {"distribution": pd.DataFrame({"mean": q[:, 1], "std": (q[:, 2] - q[:, 0]) / z, "lower": q[:, 0], "upper": q[:, 2]}, index=X.index)}

    def importance(self) -> pd.Series:
        return pd.Series(np.mean([e.feature_importances_ for e in self.ests_], axis=0), index=self.features_)

    def diagnostics(self) -> dict:
        return {"quantiles": list(self.alphas_), "interval_level": self.alphas_[2] - self.alphas_[0]}


class NGBoostModel(_Tabular):
    """A Normal predictive distribution by natural-gradient boosting: mean and standard deviation per row."""

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        if not 0.0 < float(self.params["level"]) < 1.0:
            raise ValueError("level must be between 0 and 1")
        self.est_ = None
        self.impute = True

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "NGBoostModel":
        ng, p = _import("ngboost"), self.params
        from ngboost.distns import Normal
        from ngboost.scores import LogScore
        from sklearn.tree import DecisionTreeRegressor

        values = self._values(X, fit=True)
        self.features_ = list(self._select(X).columns)
        self.n_train_ = len(values)
        target = self._prepare_y(y)
        cap = int(p["max_rows"])
        rows = np.random.default_rng(self.seed).choice(len(values), size=cap, replace=False) if 0 < cap < len(values) else slice(None)
        base = DecisionTreeRegressor(criterion="friedman_mse", max_depth=int(p["max_depth"]), min_samples_leaf=50)
        self.est_ = ng.NGBRegressor(Dist=Normal, Score=LogScore, Base=base, n_estimators=int(p["n_estimators"]), learning_rate=float(p["learning_rate"]), minibatch_frac=float(p["minibatch_frac"]),
                                    col_sample=float(p["col_sample"]), natural_gradient=True, verbose=False, random_state=self.seed)
        self.est_.fit(values[rows], target[rows])
        return self

    def _dist(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        d = self.est_.pred_dist(self._values(X))
        return d.params["loc"] * self.y_std_ + self.y_mean_, d.params["scale"] * self.y_std_

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._dist(X)[0]

    def predict_extras(self, X: pd.DataFrame) -> dict:
        mean, std = self._dist(X)
        z = stats.norm.ppf(0.5 + float(self.params["level"]) / 2.0)
        return {"distribution": pd.DataFrame({"mean": mean, "std": std, "lower": mean - z * std, "upper": mean + z * std}, index=X.index)}

    def importance(self) -> pd.Series:
        imp = np.asarray(self.est_.feature_importances_, dtype=float)
        return pd.Series(imp[0] if imp.ndim == 2 else imp, index=self.features_)

    def diagnostics(self) -> dict:
        return {"interval_level": float(self.params["level"]), "n_estimators": int(self.params["n_estimators"])}


def build(kind: str, task: str, params: dict, seed: int) -> Model:
    if kind in ("lgbm_ranker", "xgb_ranker"):
        return RankerModel(kind, task, params, seed)
    if kind == "lgbm_quantile":
        return QuantileLGBM(kind, task, params, seed)
    if kind == "ngboost":
        return NGBoostModel(kind, task, params, seed)
    raise ValueError(f"unknown model '{kind}'")
