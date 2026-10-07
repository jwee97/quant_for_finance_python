"""Machine-learning forecast: a walk-forward ridge regression of the next 21-day return on price (and optionally macro) features.

This is the "ML" and "macro" box of the research pipeline, deliberately small: ridge is the model that survived the Generation 2
forecasting study against heavier ones. Forecasts are made at month-end origins, from a model refit once a year on rows whose
labels (21 days ahead, plus a 21-day embargo) were realised before the refit, and held until the next origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.macro import expanding_zscore
from ..features.sleeves import month_end_dates
from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..framework.types import ForecastPanel
from ..models.probabilistic import ewma_sigma, price_features


@register_model("ml_ridge", "machine learning", "Walk-forward ridge regression of the next 21-day return on price features, optionally plus macro features")
class MLRidge(ForecastModel):
    """Features that describe an asset's recent history (momentum, z-scores, volatility, drawdown, RSI) carry a little forecasting information."""

    name, family = "ml_ridge", "machine learning"

    def __init__(self, features: str = "price", min_train: int = 1260, refit_every: int = 252, embargo: int = 21, alpha: float = 100.0):
        if features not in ("price", "price_macro"):
            raise ValueError("features must be 'price' or 'price_macro'")
        self.features, self.min_train, self.refit_every, self.embargo, self.alpha = features, min_train, refit_every, embargo, alpha

    def _rows(self, data, origins: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        h = self.horizon
        feats = price_features(data.prices)
        frames = {n: f.loc[origins].stack(future_stack=True) for n, f in feats.items()}
        X = pd.concat(frames, axis=1)
        if self.features == "price_macro":
            z = expanding_zscore(data.macro.dropna(axis=1, how="all"), 504, 4.0).reindex(origins)
            for c in z.columns:
                X[f"macro_{c}"] = X.index.get_level_values(0).map(z[c])
        assets = pd.get_dummies(X.index.get_level_values(1)).astype(float)
        assets.index = X.index
        X = pd.concat([X, assets], axis=1)
        forward = (data.prices.shift(-h) / data.prices - 1.0).loc[origins].stack(future_stack=True)
        sigma = ewma_sigma(data.returns, 40.0, h).loc[origins].stack(future_stack=True)
        return X, forward, sigma

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """The learner: anything with ``predict``. Subclasses swap this to run the same walk-forward protocol with a different model."""
        from sklearn.linear_model import Ridge

        return Ridge(alpha=self.alpha).fit(X, y)

    def forecast(self, data, min_observations: int = 504, allow_negative: bool = False, halflife: float = 40.0) -> ForecastPanel:
        self.require(data)
        h = self.horizon
        index = data.index
        origins = month_end_dates(index)
        X, y, sigma = self._rows(data, origins)
        position = pd.Series(np.arange(len(index)), index=index)
        origin_pos = position.reindex(X.index.get_level_values(0)).to_numpy()
        valid_x = X.notna().all(axis=1).to_numpy()
        z = (y / sigma.where(sigma > 0)).clip(-5, 5)
        mean_at_origin = pd.Series(np.nan, index=X.index)
        for start in range(self.min_train, len(index), self.refit_every):
            train = (origin_pos + h + self.embargo <= start) & valid_x & z.notna().to_numpy()
            test = (origin_pos >= start) & (origin_pos < start + self.refit_every) & valid_x
            if train.sum() < 100 or not test.any():
                continue
            mu, sd = X[train].mean(), X[train].std(ddof=1).replace(0.0, 1.0)
            model = self._fit(((X[train] - mu) / sd).to_numpy(), z[train].to_numpy())
            mean_at_origin[test] = model.predict(((X[test] - mu) / sd).to_numpy()) * sigma[test].to_numpy()
        monthly = mean_at_origin.unstack()
        mean = monthly.reindex(index).ffill().reindex(columns=data.assets)
        mean = mean.where(data.investable)
        std = ewma_sigma(data.returns, halflife, h).reindex_like(mean)
        return ForecastPanel.from_mean_std(mean, std.where(mean.notna()), h, self.name)

    def explain(self, data, n_repeats: int = 10, recent_years: int = 5) -> pd.DataFrame:
        """Permutation importance of the FINAL model (fit on every row whose label was complete by the last origin), measured on the most recent years.

        Descriptive, not a performance claim: it says which features the model leans on, and the rows it is measured on are partly in its training set.
        """
        from ..models.explain import permutation_importance

        self.require(data)
        h, index = self.horizon, data.index
        origins = month_end_dates(index)
        X, y, sigma = self._rows(data, origins)
        position = pd.Series(np.arange(len(index)), index=index)
        origin_pos = position.reindex(X.index.get_level_values(0)).to_numpy()
        z = (y / sigma.where(sigma > 0)).clip(-5, 5)
        ok = (X.notna().all(axis=1) & z.notna()).to_numpy() & (origin_pos + h + self.embargo <= len(index) - 1)
        if ok.sum() < 100:
            raise ValueError("not enough matured rows to explain the model")
        mu, sd = X[ok].mean(), X[ok].std(ddof=1).replace(0.0, 1.0)
        Xs, target = ((X[ok] - mu) / sd).to_numpy(), z[ok].to_numpy()
        model = self._fit(Xs, target)
        recent = X.index.get_level_values(0)[ok] >= index[-1] - pd.DateOffset(years=recent_years)
        sample = recent if recent.sum() >= 50 else np.ones(len(target), dtype=bool)
        importance = permutation_importance(model.predict, Xs[sample], target[sample], n_repeats=n_repeats)
        table = pd.DataFrame({"importance": importance}, index=X.columns)
        table = table[~table.index.isin(data.assets)]                          # asset dummies are intercepts, not features
        table["share"] = table["importance"].clip(lower=0.0) / max(table["importance"].clip(lower=0.0).sum(), 1e-12)
        return table.sort_values("importance", ascending=False)

    def score(self, data):
        return self.forecast(data).mean


LEARNERS = ("hgb", "random_forest", "extra_trees", "lightgbm", "xgboost", "catboost")


def make_tree_learner(learner: str, n_estimators: int, max_depth: int, learning_rate: float, min_samples_leaf: int, seed: int):
    """A regressor for the walk-forward protocol. The scikit-learn learners are always available; ``lightgbm``, ``xgboost`` and ``catboost`` are optional dependencies (extra ``boosting``) and
    raise an ImportError that says so."""
    if learner == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_samples_leaf=min_samples_leaf, l2_regularization=10.0, random_state=seed)
    if learner == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=0.5, random_state=seed, n_jobs=1)
    if learner == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=0.5, random_state=seed, n_jobs=1)
    try:
        if learner == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_child_samples=min_samples_leaf, reg_lambda=10.0, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
        if learner == "xgboost":
            import xgboost as xgb

            return xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_samples_leaf, reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8,
                                    random_state=seed, n_jobs=1, tree_method="hist")
        if learner == "catboost":
            import catboost as cb

            return cb.CatBoostRegressor(iterations=n_estimators, depth=max_depth, learning_rate=learning_rate, l2_leaf_reg=10.0, random_seed=seed, verbose=False, thread_count=1)
    except ImportError as exc:
        raise ImportError(f"learner '{learner}' needs the optional package: pip install \"systematic-multi-asset-research[boosting]\" ({exc})") from exc
    raise ValueError(f"learner must be one of {LEARNERS}")


@register_model("ml_trees", "machine learning", "Walk-forward tree ensemble (gradient boosting, random forest, extra trees, LightGBM, XGBoost or CatBoost) on the same price features as the ridge")
class MLTrees(MLRidge):
    """Trees can capture interactions and non-linearities a ridge cannot (momentum matters more when volatility is low); with a signal-to-noise ratio this small they usually overfit unless the
    leaves are large and the learning rate small, which is what the defaults enforce."""

    name, family = "ml_trees", "machine learning"

    def __init__(self, learner: str = "hgb", n_estimators: int = 150, max_depth: int = 3, learning_rate: float = 0.05, min_samples_leaf: int = 100, seed: int = 0,
                 features: str = "price", min_train: int = 1260, refit_every: int = 252, embargo: int = 21):
        if learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}")
        if n_estimators < 1 or max_depth < 1 or not 0 < learning_rate <= 1 or min_samples_leaf < 1:
            raise ValueError("n_estimators >= 1, max_depth >= 1, 0 < learning_rate <= 1, min_samples_leaf >= 1")
        super().__init__(features, min_train, refit_every, embargo, alpha=0.0)
        self.learner, self.n_estimators, self.max_depth, self.learning_rate, self.min_samples_leaf, self.seed = learner, n_estimators, max_depth, learning_rate, min_samples_leaf, seed

    def _fit(self, X, y):
        return make_tree_learner(self.learner, self.n_estimators, self.max_depth, self.learning_rate, self.min_samples_leaf, self.seed).fit(X, y)
