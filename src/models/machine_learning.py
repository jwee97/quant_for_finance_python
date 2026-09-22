"""Machine learning extension (Ch. 23, spec §46-§48).

Deliberately last. The ladder ML0 -> ML3 exists so the question can be
answered honestly:

    Does nonlinear ML provide genuine out-of-sample economic improvement
    over the simpler econometric models already built?

"Economic" is the operative word. A classifier can improve accuracy and still
lose money, because accuracy weights a 0.1% day the same as a 5% day.
``evaluate_predictions`` therefore reports predictive metrics (accuracy, AUC),
alpha metrics (IC) and investment metrics (Sharpe, drawdown, turnover) side by
side (spec §48).

Three things here that most naive financial ML gets wrong:

*Purged, embargoed cross-validation.* Random k-fold on overlapping forward
returns leaks the answer into the training set.

*Features standardised on training data only.* Fitting a scaler on the whole
sample tells the model the future's mean and variance.

*A fixed, ex-ante feature set.* Features are declared before any model is fit,
not selected by what improves the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.mean_reversion import distance_from_high, price_zscore, rsi
from ..features.momentum import total_return_momentum
from ..features.returns import forward_returns
from ..features.volatility import ewma_volatility, rolling_volatility
from ..utils.logging import get_logger
from ..validation.walk_forward import purged_kfold_indices

LOGGER = get_logger(__name__)

# Declared before any model is fit (spec §46). Nothing is added later because
# it improved a score.
FEATURE_SPEC = {
    "mom_21": lambda m: total_return_momentum(m.prices, 21, 1),
    "mom_63": lambda m: total_return_momentum(m.prices, 63, 1),
    "mom_126": lambda m: total_return_momentum(m.prices, 126, 1),
    "mom_252": lambda m: total_return_momentum(m.prices, 252, 1),
    "zscore_5": lambda m: price_zscore(m.prices, 5),
    "zscore_21": lambda m: price_zscore(m.prices, 21),
    "zscore_63": lambda m: price_zscore(m.prices, 63),
    "vol_21": lambda m: rolling_volatility(m.returns(), 21),
    "vol_63": lambda m: rolling_volatility(m.returns(), 63),
    "vol_ratio": lambda m: rolling_volatility(m.returns(), 21) / rolling_volatility(m.returns(), 252),
    "ewma_vol": lambda m: ewma_volatility(m.returns(), 40),
    "drawdown_252": lambda m: distance_from_high(m.prices, 252),
    "rsi_14": lambda m: rsi(m.prices, 14),
}


@dataclass
class MLDataset:
    """Stacked (date, asset) design matrix with its target and metadata."""

    X: pd.DataFrame
    y: pd.Series
    y_continuous: pd.Series
    dates: pd.DatetimeIndex
    assets: pd.Index
    feature_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)

    def describe(self) -> dict:
        return {
            "n_samples": int(len(self.X)),
            "n_features": int(self.X.shape[1]),
            "n_dates": int(self.dates.nunique()),
            "n_assets": int(self.assets.nunique()),
            "positive_rate": float(self.y.mean()),
            "start": str(self.dates.min().date()),
            "end": str(self.dates.max().date()),
        }


def build_dataset(market, horizon: int = 21, feature_spec: dict | None = None,
                  cross_sectional_standardise: bool = True) -> MLDataset:
    """Build the stacked design matrix.

    Features are standardised **cross-sectionally, per date** before stacking.
    That is a deliberate choice, not a convenience: it removes the common
    market move from every feature, so the model is asked "which asset" rather
    than "is it a good day", and it makes the features comparable across the
    volatility regimes the EDA documented.
    """
    spec = feature_spec or FEATURE_SPEC
    target = forward_returns(market.returns(), horizon)

    frames = {}
    for name, builder in spec.items():
        frame = builder(market).where(market.investable)
        if cross_sectional_standardise:
            frame = frame.sub(frame.mean(axis=1), axis=0).div(
                frame.std(axis=1, ddof=1).replace(0.0, np.nan), axis=0
            )
        frames[name] = frame.stack().rename(name)

    # Target is demeaned cross-sectionally too, so the label is relative
    # performance rather than market direction.
    relative = target.sub(target.mean(axis=1), axis=0)
    design = pd.concat(list(frames.values()) + [relative.stack().rename("__y__")], axis=1).dropna()
    if design.empty:
        raise ValueError("no complete rows in the ML design matrix")

    dates = pd.DatetimeIndex(design.index.get_level_values(0))
    assets = design.index.get_level_values(1)
    y_continuous = design["__y__"]
    return MLDataset(
        X=design.drop(columns="__y__"),
        y=(y_continuous > 0).astype(int),
        y_continuous=y_continuous,
        dates=dates,
        assets=assets,
        feature_names=list(spec),
    )


def make_models(seed: int = 42) -> dict:
    """The ladder: simple first, so any gain is attributable (spec §47)."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        # L2 is scikit-learn's default penalty in every supported version, so
        # it is selected by C alone. Passing penalty="l2" explicitly is
        # deprecated from 1.8 and would emit a FutureWarning on every fit.
        "ML0_logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, C=1e6)),   # effectively unregularised
        ]),
        "ML1_logistic_l2": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, C=0.1)),   # meaningfully regularised
        ]),
        "ML2_random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=200,
            random_state=seed, n_jobs=-1,
        ),
        "ML3_gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=200, subsample=0.8, random_state=seed,
        ),
    }


def cross_validate(dataset: MLDataset, models: dict | None = None, n_splits: int = 5,
                   embargo_days: int = 21, horizon: int = 21, seed: int = 42) -> pd.DataFrame:
    """Purged, embargoed cross-validation over the stacked design matrix.

    Splits are made over **unique dates**, not over stacked rows: a random row
    split would put SPY and TLT from the same day on opposite sides of the
    fold, which leaks the day's cross-section into the training set.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    models = models or make_models(seed)
    unique_dates = pd.DatetimeIndex(sorted(dataset.dates.unique()))
    folds = purged_kfold_indices(unique_dates, n_splits, embargo_days, horizon)
    if not folds:
        raise ValueError("no cross-validation folds could be generated")

    rows = []
    for name, model in models.items():
        for i, (train_positions, test_positions) in enumerate(folds, start=1):
            train_dates = set(unique_dates[train_positions])
            test_dates = set(unique_dates[test_positions])
            train_mask = dataset.dates.isin(train_dates)
            test_mask = dataset.dates.isin(test_dates)
            if train_mask.sum() < 500 or test_mask.sum() < 100:
                continue

            X_train = dataset.X[train_mask]
            y_train = dataset.y[train_mask]
            X_test = dataset.X[test_mask]
            y_test = dataset.y[test_mask]
            if y_train.nunique() < 2:
                continue

            fitted = model.fit(X_train.to_numpy(), y_train.to_numpy())
            probability = fitted.predict_proba(X_test.to_numpy())[:, 1]
            prediction = (probability > 0.5).astype(int)

            signal = pd.Series(probability - 0.5, index=X_test.index)
            realised = dataset.y_continuous[test_mask]
            ic = float(signal.corr(realised, method="spearman"))

            rows.append(
                {
                    "model": name,
                    "fold": i,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "accuracy": float(accuracy_score(y_test, prediction)),
                    "auc": float(roc_auc_score(y_test, probability)) if y_test.nunique() > 1 else np.nan,
                    "ic": ic,
                    "mean_probability": float(probability.mean()),
                }
            )
            LOGGER.info("  %s fold %d: accuracy %.4f auc %.4f ic %+.4f",
                        name, i, rows[-1]["accuracy"], rows[-1]["auc"], ic)
    return pd.DataFrame(rows)


def walk_forward_predictions(dataset: MLDataset, model, train_years: float = 5.0,
                             test_months: int = 12, embargo_days: int = 21,
                             min_train: int = 2000) -> pd.Series:
    """Out-of-sample probabilities from an expanding-window walk forward.

    The only predictions this project is willing to trade. The model is refit
    once per test block on data ending ``embargo_days`` before the block
    starts, and never sees a single observation from it.
    """
    unique_dates = pd.DatetimeIndex(sorted(dataset.dates.unique()))
    start = unique_dates[0]
    test_start = start + pd.DateOffset(years=int(train_years))
    predictions = []

    while test_start < unique_dates[-1]:
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        train_end = test_start - pd.Timedelta(days=embargo_days + 1)

        train_mask = (dataset.dates >= start) & (dataset.dates <= train_end)
        test_mask = (dataset.dates >= test_start) & (dataset.dates <= test_end)
        if train_mask.sum() < min_train or test_mask.sum() < 50:
            test_start = test_end + pd.Timedelta(days=1)
            continue
        if dataset.y[train_mask].nunique() < 2:
            test_start = test_end + pd.Timedelta(days=1)
            continue

        fitted = model.fit(dataset.X[train_mask].to_numpy(), dataset.y[train_mask].to_numpy())
        probability = fitted.predict_proba(dataset.X[test_mask].to_numpy())[:, 1]
        predictions.append(pd.Series(probability, index=dataset.X[test_mask].index))
        test_start = test_end + pd.Timedelta(days=1)

    if not predictions:
        raise RuntimeError("walk-forward ML produced no predictions")
    return pd.concat(predictions).sort_index()


def predictions_to_signal(predictions: pd.Series, columns) -> pd.DataFrame:
    """Unstack stacked probabilities back into a (date x asset) signal frame."""
    frame = predictions.unstack()
    return frame.reindex(columns=columns)


def feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Importances, or absolute standardised coefficients for linear models."""
    estimator = model[-1] if hasattr(model, "__getitem__") and hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        return pd.Series(dtype=float)
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def evaluate_predictions(signal: pd.DataFrame, returns: pd.DataFrame, horizon: int = 21,
                         market=None, transform_config: dict | None = None,
                         cost_bps: float = 10.0) -> dict:
    """Predictive, alpha and investment metrics together (spec §48).

    The three blocks routinely disagree, and that disagreement is the finding:
    a model can be the most accurate and the least profitable in the same
    table.
    """
    from ..backtest.costs import LinearCostModel
    from ..backtest.engine import BacktestEngine
    from ..models.regression import cross_sectional_ic, ic_summary
    from ..signals.transform import signal_to_positions

    forward = forward_returns(returns, horizon)
    ic = cross_sectional_ic(signal, forward)
    out = {f"ic_{k}": v for k, v in ic_summary(ic, horizon, breadth=signal.shape[1]).items()}

    if market is not None:
        weights = signal_to_positions(
            signal, rolling_volatility(returns, 63), investable=market.investable,
            **(transform_config or {})
        )
        engine = BacktestEngine(signal_lag=1, rebalance="monthly", weight_drift=True,
                                cost_model=LinearCostModel(cost_bps))
        result = engine.run(weights, returns, "ml", market.investable, apply_vol_target=False)
        summary = result.summary()
        out.update({
            "cagr": summary.get("cagr"),
            "ann_vol": summary.get("ann_vol"),
            "sharpe": summary.get("sharpe"),
            "max_drawdown": summary.get("max_drawdown"),
            "ann_turnover": summary.get("ann_turnover"),
            "gross_sharpe": summary.get("gross_sharpe"),
            "net_return": summary.get("ann_return_arith"),
        })
    return out
