"""Monthly asset-class forecasts from price and macro features (Generation 2).

The design question is how to give macro information a fair chance without
handing it an unfair advantage. The choices that matter:

* **Non-overlapping monthly targets.** A 21-day return sampled monthly has no
  overlap, so the Clark-West test needs no HAC correction and there are no
  overlapping-observation illusions of the kind Stage 5 documented.
* **Strict walk-forward.** The forecast made at month-end k is trained only on
  (features at s, return over s+1) pairs whose target was already realised by
  k, so s <= k-1. Nothing from month k+1 is in the training set.
* **Nested models.** price vs history, macro vs history, price+macro vs price.
  Nested comparisons are what the Clark-West test is built for, and the last
  one is the question that matters: does macro add anything *beyond* price?
* **Heavy regularisation, chosen in-sample on time-ordered folds.** With ~200
  monthly observations and 13+ correlated macro features, an unregularised
  regression will fit noise. The ridge penalty is picked by time-series
  cross-validation inside each training window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.sleeves import monthly_compound, month_end_dates

MODEL_FEATURES = {
    "hist": (),
    "price": ("price",),
    "macro": ("macro",),
    "both": ("price", "macro"),
}


@dataclass
class MonthlyPanel:
    """Everything the walk-forward needs, indexed by month-end date."""

    dates: pd.DatetimeIndex
    target: pd.DataFrame                       # next month's excess return, by sleeve
    price: dict[str, pd.DataFrame]             # per sleeve: price features at month-end
    macro: pd.DataFrame                        # macro features at month-end (common to all sleeves)
    realised: pd.DataFrame                     # this month's excess return, by sleeve


def build_monthly_panel(sleeve_daily: pd.DataFrame, cash_daily: pd.Series,
                        macro_z: pd.DataFrame) -> MonthlyPanel:
    """Month-end features and next-month targets for every sleeve."""
    dates = month_end_dates(sleeve_daily.index)
    cash_m = monthly_compound(cash_daily)
    excess = monthly_compound(sleeve_daily).sub(cash_m, axis=0)
    excess = excess.reindex(dates)

    price: dict[str, pd.DataFrame] = {}
    for sleeve in sleeve_daily.columns:
        daily = sleeve_daily[sleeve]
        index = (1.0 + daily.fillna(0.0)).cumprod()
        vol = daily.rolling(63, min_periods=40).std(ddof=1) * np.sqrt(252)
        drawdown = index / index.rolling(252, min_periods=126).max() - 1.0
        compounded = excess[sleeve]
        price[sleeve] = pd.DataFrame(
            {
                "mom_3m": (1.0 + compounded).rolling(3).apply(np.prod, raw=True) - 1.0,
                "mom_12m": (1.0 + compounded).rolling(12).apply(np.prod, raw=True) - 1.0,
                "vol_3m": vol.reindex(dates),
                "drawdown_12m": drawdown.reindex(dates),
            },
            index=dates,
        )
    return MonthlyPanel(
        dates=dates,
        target=excess.shift(-1),               # the return over the month AFTER the features
        price=price,
        macro=macro_z.reindex(dates),
        realised=excess,
    )


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                 alphas: list[float], cv_splits: int) -> float:
    splits = min(cv_splits, max(2, len(train_y) // 12))
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=splits)),
    )
    model.fit(train_x, train_y)
    return float(model.predict(test_x)[0])


def walk_forward_forecasts(panel: MonthlyPanel, min_train_months: int = 60,
                           alphas: list[float] | None = None, cv_splits: int = 5,
                           first_test_year: int | None = None,
                           macro_override: pd.DataFrame | None = None,
                           models: tuple[str, ...] | None = None,
                           extra_blocks: dict[str, pd.DataFrame] | None = None,
                           extra_models: dict[str, tuple[str, ...]] | None = None) -> dict:
    """Expanding-window one-month-ahead forecasts for every sleeve and model.

    Returns ``{"actual": DataFrame, "forecasts": {model: DataFrame}}`` indexed by
    the month-end at which the forecast was made. ``macro_override`` swaps in a
    different macro panel (used to run the same study with publication lags
    deliberately ignored, to measure how much that flatters the result). ``models``
    restricts which learned models are fitted (the history benchmark is always
    produced); models left out come back as all-NaN frames. ``extra_blocks``
    adds named feature blocks (month-end frames common to every sleeve) and
    ``extra_models`` names models built from blocks, e.g.
    ``{"all": ("price", "macro", "nonprice")}``, so a larger nested model can be
    tested against ``both`` with exactly the same machinery.
    """
    model_features = {**MODEL_FEATURES, **(extra_models or {})}
    alphas = alphas or [1.0, 10.0, 100.0, 1000.0]
    macro = (macro_override.reindex(panel.dates) if macro_override is not None else panel.macro)
    sleeves = list(panel.target.columns)
    dates = panel.dates

    forecasts = {m: pd.DataFrame(np.nan, index=dates, columns=sleeves) for m in model_features}
    actual = panel.target.copy()

    for sleeve in sleeves:
        y = panel.target[sleeve]
        blocks = {"price": panel.price[sleeve], "macro": macro,
                  **{k: v.reindex(dates) for k, v in (extra_blocks or {}).items()}}
        for k in range(len(dates)):
            when = dates[k]
            if first_test_year is not None and when.year < first_test_year:
                continue
            # Targets realised by `when` belong to rows s <= k-1.
            train_rows = np.arange(0, k)
            if len(train_rows) < min_train_months:
                continue
            y_train = y.iloc[train_rows]
            if y_train.notna().sum() < min_train_months or np.isnan(y.iloc[k]):
                continue

            forecasts["hist"].iloc[k, forecasts["hist"].columns.get_loc(sleeve)] = float(y_train.mean())
            for model, parts in model_features.items():
                if not parts or (models is not None and model not in models):
                    continue
                frame = pd.concat([blocks[p] for p in parts], axis=1)
                train_x = frame.iloc[train_rows]
                valid = train_x.notna().all(axis=1) & y_train.notna()
                test_x = frame.iloc[[k]]
                if valid.sum() < min_train_months or test_x.isna().any().any():
                    continue
                forecasts[model].iloc[k, forecasts[model].columns.get_loc(sleeve)] = _fit_predict(
                    train_x[valid].to_numpy(), y_train[valid].to_numpy(), test_x.to_numpy(),
                    alphas, cv_splits)
    return {"actual": actual, "forecasts": forecasts}


def evaluate_nested(result: dict, comparisons: list[tuple[str, str]], fdr: float = 0.10) -> pd.DataFrame:
    """Clark-West and OOS R-squared for every (restricted, unrestricted) pair, per sleeve."""
    from ..validation.forecast_tests import benjamini_hochberg, clark_west

    rows = []
    actual = result["actual"]
    for sleeve in actual.columns:
        for restricted, unrestricted in comparisons:
            f1 = result["forecasts"][restricted][sleeve]
            f2 = result["forecasts"][unrestricted][sleeve]
            frame = pd.concat([actual[sleeve].rename("y"), f1.rename("f1"), f2.rename("f2")],
                              axis=1).dropna()
            test = clark_west(frame["y"], frame["f1"], frame["f2"])
            if test:
                rows.append({"sleeve": sleeve, "restricted": restricted, "unrestricted": unrestricted,
                             **test})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["comparison"] = table["unrestricted"] + "_vs_" + table["restricted"]
    table["bh_significant"] = False
    for name, group in table.groupby("comparison"):
        table.loc[group.index, "bh_significant"] = benjamini_hochberg(group["p_value"], fdr)
    table["significant_raw_5pct"] = table["p_value"] < 0.05
    return table
