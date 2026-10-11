"""The model zoo behind one interface.

Every model is a ``Model``: ``fit(X, y, dates)`` on a frame of features and a target, ``predict(X)`` returning a score where higher means a better asset to hold (for a classifier, the probability of the
positive class minus one half), ``importance()`` returning one number per feature, and ``to_bytes`` / ``Model.from_bytes`` for the registry. Regression targets are standardised inside ``fit`` (so the
strength of a penalty means the same thing for a 1-day and a 20-day target) and the prediction is rescaled back.

Kinds: ``linear``, ``ridge``, ``lasso``, ``elasticnet`` (and ``logistic`` for a classification target), ``random_forest``, ``hist_gb`` (histogram gradient boosting), ``xgboost``, ``lightgbm`` and ``catboost``
when those packages are installed; the time-series flavours ``ar`` (a ridge on lagged returns), ``kalman`` (a regression whose coefficients drift: a Kalman filter on the coefficient vector),
``online_ridge`` (exponentially weighted ridge, updated date by date), ``online_logistic`` (stochastic-gradient logistic, updated date by date) and ``regime_ridge`` (one ridge per regime state). Meta-labelling, the
bandit allocator and the model selectors live in ``ensembles.py`` because they combine models.

Nothing here knows about the future: the models see only the rows they are given. What keeps a walk-forward run honest is ``training.py``, which gives each fit only the rows whose labels were complete.
"""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..ops.hpo import Categorical, Float, Int, LogFloat


class ModelUnavailable(RuntimeError):
    """The model needs a package that is not installed."""


AR_FEATURES = ("ret_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5")


class MedianImputer(BaseEstimator, TransformerMixin):
    """Replace missing values by the training median of their column (zero for a column that is entirely missing). scikit-learn's imputer computes the median through masked arrays, about ten times slower
    on a panel of this size, and a walk-forward run fits a hundred models."""

    def fit(self, X, y=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.fill_ = np.nan_to_num(np.nanmedian(np.asarray(X, dtype=float), axis=0), nan=0.0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(np.isnan(X), self.fill_, X) if np.isnan(X).any() else X


# ------------------------------------------------------------------------------------------------------------------ the interface
@dataclass
class Model:
    kind: str
    task: str = "regression"                       # regression | classification
    params: dict = field(default_factory=dict)
    seed: int = 0
    features_: list = field(default_factory=list)
    y_mean_: float = 0.0
    y_std_: float = 1.0
    n_train_: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "Model":
        raise NotImplementedError

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.raw_predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        return None

    def importance(self) -> pd.Series:
        return pd.Series(0.0, index=self.features_)

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        joblib.dump(self, buf, compress=3)
        return buf.getvalue()

    @staticmethod
    def from_bytes(blob: bytes) -> "Model":
        return joblib.load(io.BytesIO(blob))

    # shared helpers
    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.params.get("columns", []) if c in X.columns] if self.params.get("columns") else list(X.columns)
        if not cols:
            raise ValueError("none of the requested feature columns is in the data")
        return X[cols]

    def _prepare_y(self, y: pd.Series) -> np.ndarray:
        arr = y.to_numpy(dtype=float)
        if self.task == "regression":
            self.y_mean_, self.y_std_ = float(np.nanmean(arr)), float(np.nanstd(arr)) or 1.0
            return (arr - self.y_mean_) / self.y_std_
        return arr.astype(int)

    def _weights(self, dates: np.ndarray | None) -> np.ndarray | None:
        hl = self.params.get("halflife_days")
        if not hl or dates is None:
            return None
        d = np.asarray(dates, dtype=float)
        return np.exp(-math.log(2.0) * (d.max() - d) / float(hl))


# ------------------------------------------------------------------------------------------------------------------ scikit-learn and booster models
class SklearnModel(Model):
    """A scikit-learn style estimator (with an imputer and scaler in front when the estimator needs them)."""

    builder: str = ""

    def __init__(self, kind: str, task: str, params: dict, seed: int, estimator, needs_scaling: bool, native_nan: bool):
        super().__init__(kind, task, dict(params), seed)
        self.estimator, self.needs_scaling, self.native_nan = estimator, needs_scaling, native_nan
        self.pipe_: Pipeline | None = None

    def _pipeline(self) -> Pipeline:
        steps = []
        if not self.native_nan:
            steps.append(("impute", MedianImputer()))
        if self.needs_scaling:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", self.estimator))
        return Pipeline(steps)

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "SklearnModel":
        Xs = self._select(X)
        self.features_ = list(Xs.columns)
        target = self._prepare_y(y)
        self.n_train_ = len(Xs)
        self.pipe_ = self._pipeline()
        w = self._weights(dates)
        values = Xs.to_numpy(dtype=np.float64)
        if self.task == "classification" and len(np.unique(target)) < 2:
            raise ValueError("the training labels contain a single class")
        if w is not None:
            self.pipe_.fit(values, target, model__sample_weight=w)
        else:
            self.pipe_.fit(values, target)
        return self

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.features_].to_numpy(dtype=np.float64)
        if self.task == "classification":
            return self.pipe_.predict_proba(values)[:, 1] - 0.5
        return self.pipe_.predict(values) * self.y_std_ + self.y_mean_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if self.task != "classification":
            return None
        return self.pipe_.predict_proba(X[self.features_].to_numpy(dtype=np.float64))[:, 1]

    def importance(self) -> pd.Series:
        est = self.pipe_.named_steps["model"]
        if hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=float).reshape(-1)
            return pd.Series(coef, index=self.features_)
        if hasattr(est, "feature_importances_"):
            return pd.Series(np.asarray(est.feature_importances_, dtype=float), index=self.features_)
        if hasattr(est, "get_booster"):
            gain = est.get_booster().get_score(importance_type="gain")
            return pd.Series({f: float(gain.get(f"f{i}", 0.0)) for i, f in enumerate(self.features_)})
        return pd.Series(0.0, index=self.features_)


def _import(name: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise ModelUnavailable(f"the '{name}' package is not installed (pip install {name}); choose another model") from exc


def _build_sklearn(kind: str, task: str, p: dict, seed: int) -> SklearnModel:
    clf = task == "classification"
    if kind in ("linear", "ridge", "lasso", "elasticnet", "logistic", "ar"):
        if clf:
            penalty = {"lasso": "l1", "elasticnet": "elasticnet"}.get(kind, "l2")
            strength = float(p.get("alpha", p.get("C_inverse", 1.0 if kind != "linear" else 1e-4)))
            kw = {"penalty": penalty, "C": 1.0 / max(strength, 1e-8), "solver": "saga" if penalty != "l2" else "lbfgs", "max_iter": 2000, "random_state": seed}
            if penalty == "elasticnet":
                kw["l1_ratio"] = float(p.get("l1_ratio", 0.5))
            if penalty == "l2":
                kw.pop("penalty")
            return SklearnModel(kind, task, p, seed, LogisticRegression(**kw), True, False)
        if kind == "linear":
            return SklearnModel(kind, task, p, seed, LinearRegression(), True, False)
        if kind in ("ridge", "ar"):
            return SklearnModel(kind, task, p, seed, Ridge(alpha=float(p.get("alpha", 10.0)), random_state=seed), True, False)
        if kind == "lasso":
            return SklearnModel(kind, task, p, seed, Lasso(alpha=float(p.get("alpha", 0.02)), max_iter=5000, random_state=seed), True, False)
        if kind == "elasticnet":
            return SklearnModel(kind, task, p, seed, ElasticNet(alpha=float(p.get("alpha", 0.02)), l1_ratio=float(p.get("l1_ratio", 0.5)), max_iter=5000, random_state=seed), True, False)
        raise ValueError("logistic regression needs a classification target (direction_h or topbottom_h)")
    if kind == "random_forest":
        kw = dict(n_estimators=int(p.get("n_estimators", 200)), max_depth=int(p.get("max_depth", 6)), min_samples_leaf=int(p.get("min_samples_leaf", 200)), max_features=float(p.get("max_features", 0.5)),
                  n_jobs=1, random_state=seed)
        return SklearnModel(kind, task, p, seed, (RandomForestClassifier if clf else RandomForestRegressor)(**kw), False, False)
    if kind == "hist_gb":
        kw = dict(max_iter=int(p.get("max_iter", 200)), learning_rate=float(p.get("learning_rate", 0.05)), max_depth=int(p.get("max_depth", 3)), min_samples_leaf=int(p.get("min_samples_leaf", 200)),
                  l2_regularization=float(p.get("l2_regularization", 1.0)), early_stopping=False, random_state=seed)
        return SklearnModel(kind, task, p, seed, (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(**kw), False, True)
    if kind == "xgboost":
        xgb = _import("xgboost")
        kw = dict(n_estimators=int(p.get("n_estimators", 300)), max_depth=int(p.get("max_depth", 3)), learning_rate=float(p.get("learning_rate", 0.03)), subsample=float(p.get("subsample", 0.7)),
                  colsample_bytree=float(p.get("colsample_bytree", 0.7)), min_child_weight=float(p.get("min_child_weight", 50)), reg_lambda=float(p.get("reg_lambda", 5.0)), tree_method="hist", n_jobs=1,
                  random_state=seed, verbosity=0)
        return SklearnModel(kind, task, p, seed, (xgb.XGBClassifier if clf else xgb.XGBRegressor)(**kw), False, True)
    if kind == "lightgbm":
        lgb = _import("lightgbm")
        kw = dict(n_estimators=int(p.get("n_estimators", 300)), num_leaves=int(p.get("num_leaves", 15)), learning_rate=float(p.get("learning_rate", 0.03)), subsample=float(p.get("subsample", 0.7)),
                  subsample_freq=1, colsample_bytree=float(p.get("colsample_bytree", 0.7)), min_child_samples=int(p.get("min_child_samples", 100)), reg_lambda=float(p.get("reg_lambda", 5.0)), n_jobs=1,
                  random_state=seed, verbosity=-1, force_row_wise=True)
        return SklearnModel(kind, task, p, seed, (lgb.LGBMClassifier if clf else lgb.LGBMRegressor)(**kw), False, True)
    if kind == "catboost":
        cb = _import("catboost")
        kw = dict(iterations=int(p.get("iterations", 300)), depth=int(p.get("depth", 4)), learning_rate=float(p.get("learning_rate", 0.05)), l2_leaf_reg=float(p.get("l2_leaf_reg", 5.0)), random_seed=seed,
                  verbose=False, thread_count=1)
        return SklearnModel(kind, task, p, seed, (cb.CatBoostClassifier if clf else cb.CatBoostRegressor)(**kw), False, True)
    raise ValueError(f"unknown model '{kind}'")


# ------------------------------------------------------------------------------------------------------------------ time-series flavours
def _by_date(dates: np.ndarray | None, n: int) -> list[np.ndarray]:
    """Row indices grouped by date, in date order (every row its own date if none are given)."""
    if dates is None:
        return [np.arange(n)]
    d = np.asarray(dates)
    order = np.argsort(d, kind="stable")
    bounds = np.flatnonzero(np.diff(d[order])) + 1
    return np.split(order, bounds)


class OnlineRidge(Model):
    """Exponentially weighted ridge regression, updated one date at a time: ``A <- lam A + X'X`` and ``b <- lam b + X'y`` for each date's rows, coefficients ``(A + alpha I)^-1 b``. A forgetting factor
    ``lam`` below one lets the coefficients drift with the market, which is the point of an online model; ``lam = 1`` is ordinary ridge."""

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        self.A_ = self.b_ = self.coef_ = self.mu_ = self.sd_ = None

    def _init_state(self, d: int):
        self.A_, self.b_ = np.zeros((d, d)), np.zeros(d)

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "OnlineRidge":
        if self.task != "regression":
            raise ValueError("online_ridge is a regression; use online_logistic for a classification target")
        Xs = self._select(X)
        self.features_ = list(Xs.columns)
        v = Xs.to_numpy(dtype=np.float64)
        self.mu_ = np.nanmedian(v, axis=0)
        v = np.where(np.isnan(v), self.mu_, v)
        self.sd_ = v.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        v = (v - self.mu_) / self.sd_
        t = self._prepare_y(y)
        self.n_train_ = len(v)
        self._init_state(v.shape[1])
        lam = float(self.params.get("forgetting", 0.998))
        for idx in _by_date(dates, len(v)):
            self.A_ = lam * self.A_ + v[idx].T @ v[idx]
            self.b_ = lam * self.b_ + v[idx].T @ t[idx]
        self._solve()
        return self

    def _solve(self):
        alpha = float(self.params.get("alpha", 10.0))
        self.coef_ = np.linalg.solve(self.A_ + alpha * np.eye(len(self.b_)), self.b_)

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> "OnlineRidge":
        """One more date's rows: decay the state, add them, re-solve (the monitoring loop's daily update)."""
        v = np.where(np.isnan(X[self.features_].to_numpy(dtype=np.float64)), self.mu_, X[self.features_].to_numpy(dtype=np.float64))
        v = (v - self.mu_) / self.sd_
        t = (y.to_numpy(dtype=float) - self.y_mean_) / self.y_std_
        lam = float(self.params.get("forgetting", 0.998))
        self.A_, self.b_ = lam * self.A_ + v.T @ v, lam * self.b_ + v.T @ t
        self._solve()
        return self

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        v = X[self.features_].to_numpy(dtype=np.float64)
        v = (np.where(np.isnan(v), self.mu_, v) - self.mu_) / self.sd_
        return (v @ self.coef_) * self.y_std_ + self.y_mean_

    def importance(self) -> pd.Series:
        return pd.Series(self.coef_, index=self.features_)


class KalmanRegression(OnlineRidge):
    """A regression whose coefficient vector follows a random walk, tracked by a Kalman filter: ``beta_t = beta_{t-1} + w_t``, ``y_it = x_it' beta_t + e_it``. Each date's cross-section is one observation
    batch. ``q`` is the variance of the coefficient drift per date and ``r`` the noise variance of the (standardised) target, so ``q / r`` sets how fast the coefficients are allowed to change. To keep
    the filter well conditioned only the ``max_features`` features most related to the target in the training sample are used."""

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "KalmanRegression":
        if self.task != "regression":
            raise ValueError("the Kalman regression is for regression targets")
        Xs = self._select(X)
        v = Xs.to_numpy(dtype=np.float64)
        t = self._prepare_y(y)
        self.mu_ = np.nanmedian(v, axis=0)
        v = np.where(np.isnan(v), self.mu_, v)
        self.sd_ = v.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        v = (v - self.mu_) / self.sd_
        k = min(int(self.params.get("max_features", 12)), v.shape[1])
        corr = np.abs([np.corrcoef(v[:, j], t)[0, 1] if v[:, j].std() > 0 else 0.0 for j in range(v.shape[1])])
        keep = np.argsort(-np.nan_to_num(corr))[:k]
        self.keep_ = np.sort(keep)
        self.features_ = [Xs.columns[j] for j in self.keep_]
        self.mu_, self.sd_, v = self.mu_[self.keep_], self.sd_[self.keep_], v[:, self.keep_]
        self.n_train_ = len(v)
        q, r = float(self.params.get("q", 1e-5)), float(self.params.get("r", 1.0))
        d = v.shape[1]
        beta, P = np.zeros(d), np.eye(d) * float(self.params.get("p0", 0.01))
        path = []
        for idx in _by_date(dates, len(v)):
            P = P + q * np.eye(d)                                               # predict: the coefficients drift
            H = v[idx]
            S = H @ P @ H.T + r * np.eye(len(idx))
            K = P @ H.T @ np.linalg.inv(S)
            beta = beta + K @ (t[idx] - H @ beta)                              # update with this date's cross-section
            P = P - K @ H @ P
            path.append(beta.copy())
        self.coef_ = beta
        self.path_ = np.asarray(path)
        return self

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        v = X[self.features_].to_numpy(dtype=np.float64)
        v = (np.where(np.isnan(v), self.mu_, v) - self.mu_) / self.sd_
        return (v @ self.coef_) * self.y_std_ + self.y_mean_

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> "KalmanRegression":
        raise NotImplementedError("refit the Kalman regression on the new window instead")


class OnlineLogistic(Model):
    """A logistic regression trained by stochastic gradient steps, one date's rows at a time with a constant step size so old dates fade: the classification counterpart of ``online_ridge``."""

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        self.w_ = self.b0_ = self.mu_ = self.sd_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "OnlineLogistic":
        if self.task != "classification":
            raise ValueError("online_logistic needs a classification target (direction_h or topbottom_h)")
        Xs = self._select(X)
        self.features_ = list(Xs.columns)
        v = Xs.to_numpy(dtype=np.float64)
        self.mu_ = np.nanmedian(v, axis=0)
        v = np.where(np.isnan(v), self.mu_, v)
        self.sd_ = v.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        v = (v - self.mu_) / self.sd_
        t = y.to_numpy(dtype=float)
        self.n_train_ = len(v)
        self.w_, self.b0_ = np.zeros(v.shape[1]), 0.0
        eta, l2, epochs = float(self.params.get("eta", 0.05)), float(self.params.get("l2", 1e-3)), int(self.params.get("epochs", 2))
        groups = _by_date(dates, len(v))
        for _ in range(epochs):
            for idx in groups:
                p = 1.0 / (1.0 + np.exp(-(v[idx] @ self.w_ + self.b0_)))
                g = v[idx].T @ (p - t[idx]) / len(idx)
                self.w_ -= eta * (g + l2 * self.w_)
                self.b0_ -= eta * float(np.mean(p - t[idx]))
        return self

    def _p(self, X: pd.DataFrame) -> np.ndarray:
        v = X[self.features_].to_numpy(dtype=np.float64)
        v = (np.where(np.isnan(v), self.mu_, v) - self.mu_) / self.sd_
        return 1.0 / (1.0 + np.exp(-(v @ self.w_ + self.b0_)))

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._p(X) - 0.5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        return self._p(X)

    def importance(self) -> pd.Series:
        return pd.Series(self.w_, index=self.features_)


class RegimeRidge(Model):
    """One ridge regression per regime state, chosen by the ``regime_state`` column; a state with too few rows falls back to the pooled fit. The state is itself a feature the dataset builder adds
    from a walk-forward regime detector, so the choice of regression at each date uses only what was known on that date."""

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        self.models_: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "RegimeRidge":
        if "regime_state" not in X.columns:
            raise ValueError("regime_ridge needs the regime_state feature: set dataset.regime_detector (vol_state or hmm)")
        if self.task != "regression":
            raise ValueError("regime_ridge is a regression")
        cols = [c for c in X.columns if c != "regime_state"]
        self.features_ = cols + ["regime_state"]
        self.n_train_ = len(X)
        alpha = float(self.params.get("alpha", 10.0))
        mk = lambda: SklearnModel("ridge", "regression", {"alpha": alpha}, self.seed, Ridge(alpha=alpha), True, False)             # noqa: E731
        self.models_ = {"all": mk().fit(X[cols], y, dates)}
        state = X["regime_state"].to_numpy()
        for s in np.unique(state[~np.isnan(state)]):
            sel = state == s
            if sel.sum() >= int(self.params.get("min_rows", 500)):
                self.models_[float(s)] = mk().fit(X.loc[sel, cols], y[sel], None if dates is None else np.asarray(dates)[sel])
        self.y_mean_, self.y_std_ = 0.0, 1.0
        return self

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.features_ if c != "regime_state"]
        out = self.models_["all"].raw_predict(X[cols])
        state = X["regime_state"].to_numpy()
        for s, m in self.models_.items():
            if s == "all":
                continue
            sel = state == s
            if sel.any():
                out[sel] = m.raw_predict(X.loc[sel, cols])
        return out

    def importance(self) -> pd.Series:
        return self.models_["all"].importance().reindex(self.features_).fillna(0.0)


# ------------------------------------------------------------------------------------------------------------------ catalogue
@dataclass(frozen=True)
class ModelInfo:
    kind: str
    label: str
    family: str
    tasks: tuple
    description: str
    params: dict                         # name -> default
    refit_every: int = 21                # a sensible refit period (trees are slow, so they refit less often)
    optional_package: str | None = None


MODEL_INFO = [
    ModelInfo("linear", "Linear regression", "linear", ("regression", "classification"), "Ordinary least squares on standardised features (a near-unregularised logistic for a classification target)", {}, 21),
    ModelInfo("ridge", "Ridge", "linear", ("regression", "classification"), "L2-penalised linear model: shrinks correlated features together", {"alpha": 10.0}, 21),
    ModelInfo("lasso", "Lasso", "linear", ("regression", "classification"), "L1-penalised linear model: sets weak features to zero", {"alpha": 0.02}, 21),
    ModelInfo("elasticnet", "Elastic net", "linear", ("regression", "classification"), "A mix of L1 and L2 penalties", {"alpha": 0.02, "l1_ratio": 0.5}, 21),
    ModelInfo("logistic", "Logistic regression", "linear", ("classification",), "L2 logistic regression for direction and top-versus-bottom targets", {"alpha": 1.0}, 21),
    ModelInfo("random_forest", "Random forest", "trees", ("regression", "classification"), "Bagged deep-ish trees with large leaves (panel data is noisy)", {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 200, "max_features": 0.5}, 63),
    ModelInfo("hist_gb", "Gradient boosting", "trees", ("regression", "classification"), "scikit-learn histogram gradient boosting with shallow trees and a small learning rate", {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 200, "l2_regularization": 1.0}, 63),
    ModelInfo("xgboost", "XGBoost", "trees", ("regression", "classification"), "Gradient-boosted trees (xgboost)", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 50, "reg_lambda": 5.0}, 63, "xgboost"),
    ModelInfo("lightgbm", "LightGBM", "trees", ("regression", "classification"), "Gradient-boosted trees (lightgbm)", {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 100, "reg_lambda": 5.0}, 63, "lightgbm"),
    ModelInfo("catboost", "CatBoost", "trees", ("regression", "classification"), "Gradient-boosted trees (catboost), when installed", {"iterations": 300, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 5.0}, 63, "catboost"),
    ModelInfo("ar", "Autoregression", "time series", ("regression", "classification"), "A ridge on the last five daily returns only: does yesterday predict tomorrow?", {"alpha": 10.0, "columns": list(AR_FEATURES)}, 21),
    ModelInfo("kalman", "Kalman regression", "time series", ("regression",), "A regression whose coefficients drift as a random walk, tracked by a Kalman filter", {"q": 1e-5, "r": 1.0, "max_features": 12}, 21),
    ModelInfo("online_ridge", "Online ridge", "time series", ("regression",), "Exponentially weighted ridge updated date by date (forgets old data)", {"alpha": 10.0, "forgetting": 0.998}, 21),
    ModelInfo("online_logistic", "Online logistic", "time series", ("classification",), "Stochastic-gradient logistic regression updated date by date", {"eta": 0.05, "l2": 1e-3, "epochs": 2}, 21),
    ModelInfo("regime_ridge", "Regime ridge", "regime", ("regression",), "A ridge per regime state (needs dataset.regime_detector)", {"alpha": 10.0, "min_rows": 500}, 21),
]
_INFO = {m.kind: m for m in MODEL_INFO}


def _package_available(name: str | None) -> bool:
    if name is None:
        return True
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def model_catalog() -> list[dict]:
    """Every model kind with its tasks, defaults and whether its package is installed."""
    return [{"kind": m.kind, "label": m.label, "family": m.family, "tasks": list(m.tasks), "description": m.description, "params": dict(m.params), "refit_every": m.refit_every,
             "available": _package_available(m.optional_package), "reason": None if _package_available(m.optional_package) else f"install the {m.optional_package} package"} for m in MODEL_INFO]


def default_refit(kind: str) -> int:
    return _INFO[kind].refit_every if kind in _INFO else 21


def make_model(kind: str, params: dict | None = None, task: str = "regression", seed: int = 0) -> Model:
    """A fresh, unfitted model. ``task`` is ``regression`` or ``classification`` (chosen by the target). Unknown parameters are refused rather than ignored."""
    if kind not in _INFO:
        raise ValueError(f"unknown model '{kind}'; choose from {[m.kind for m in MODEL_INFO]}")
    info = _INFO[kind]
    if task not in info.tasks and not (task == "classification" and kind in ("linear", "ridge", "lasso", "elasticnet", "ar")):
        raise ValueError(f"{info.label} does not do {task}; it does {list(info.tasks)}")
    params = dict(params or {})
    unknown = set(params) - set(info.params) - {"columns", "halflife_days"}
    if unknown:
        raise ValueError(f"{kind} has no parameter(s) {sorted(unknown)}; its parameters are {sorted(info.params) + ['halflife_days']}")
    merged = {**info.params, **params}
    if kind in ("ridge", "lasso", "elasticnet", "online_ridge", "regime_ridge") and float(merged.get("alpha", 1.0)) < 0:
        raise ValueError("alpha must be non-negative")
    if kind == "online_ridge":
        return OnlineRidge(kind, task, merged, seed)
    if kind == "kalman":
        return KalmanRegression(kind, task, merged, seed)
    if kind == "online_logistic":
        return OnlineLogistic(kind, task, merged, seed)
    if kind == "regime_ridge":
        return RegimeRidge(kind, task, merged, seed)
    return _build_sklearn(kind, task, merged, seed)


def search_space(kind: str) -> dict:
    """The hyperparameter search space of a model kind, for ``ops.hpo.Study``."""
    spaces = {
        "ridge": {"alpha": LogFloat(0.1, 1000.0)}, "lasso": {"alpha": LogFloat(0.001, 1.0)}, "elasticnet": {"alpha": LogFloat(0.001, 1.0), "l1_ratio": Float(0.1, 0.9)},
        "logistic": {"alpha": LogFloat(0.01, 100.0)}, "linear": {}, "ar": {"alpha": LogFloat(0.1, 1000.0)},
        "random_forest": {"max_depth": Int(2, 8), "min_samples_leaf": Categorical(50, 100, 200, 400), "max_features": Float(0.2, 0.9)},
        "hist_gb": {"max_depth": Int(2, 5), "learning_rate": LogFloat(0.01, 0.15), "max_iter": Int(50, 300), "l2_regularization": LogFloat(0.1, 20.0)},
        "xgboost": {"max_depth": Int(2, 5), "learning_rate": LogFloat(0.01, 0.1), "n_estimators": Int(100, 400), "reg_lambda": LogFloat(1.0, 50.0)},
        "lightgbm": {"num_leaves": Int(4, 31), "learning_rate": LogFloat(0.01, 0.1), "n_estimators": Int(100, 400), "reg_lambda": LogFloat(1.0, 50.0)},
        "kalman": {"q": LogFloat(1e-7, 1e-3), "max_features": Int(4, 20)}, "online_ridge": {"alpha": LogFloat(0.1, 1000.0), "forgetting": Float(0.99, 0.9999)},
        "online_logistic": {"eta": LogFloat(0.005, 0.2), "l2": LogFloat(1e-5, 1e-1)}, "regime_ridge": {"alpha": LogFloat(0.1, 1000.0)},
    }
    if kind not in spaces:
        raise ValueError(f"no search space for '{kind}'")
    return spaces[kind]


def task_for(target_is_classification: bool) -> str:
    return "classification" if target_is_classification else "regression"
