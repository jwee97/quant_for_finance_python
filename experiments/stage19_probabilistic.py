"""Stage 19 - Probabilistic forecasting and confidence-aware sizing (Generation 2, Priority 2).

A point forecast discards how sure the model is. This stage forecasts a
distribution for each ETF's return over the next 21 trading days, scores it with
proper scoring rules against benchmarks that carry no feature information, and
asks whether sizing positions by confidence beats sizing them by direction alone.

1. CALIBRATION. Does Platt scaling make the classifier's probabilities honest?
2. SKILL. Do the direction classifier (log loss) and the Gaussian forecast
   (CRPS) beat a base rate and a historical-mean Gaussian?
3. INFORMATION. Do macro and regime features improve either?
4. SIZING. Net of costs, do probability-sized and fractional-Kelly books beat a
   direction-only book at the same signal?

The design and the five retention rules were committed to
``config/forecasting.yaml`` before any of this ran. Figures 38-39.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import performance_summary
from src.data.macro import ensure_macro_raw
from src.features.macro import macro_feature_panel
from src.features.sleeves import month_end_dates
from src.models.probabilistic import (
    auc,
    brier_terms,
    build_panel,
    calibration_slope_intercept,
    edge_book,
    expected_calibration_error,
    kelly_book,
    paired_ece_test,
    pit_gaussian,
    pit_uniformity,
    reliability_table,
    scores_by_origin,
    walk_forward_probabilistic,
)
from src.utils.dates import slice_dates
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg, diebold_mariano
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context

STAGE = "stage19_probabilistic"
VARIANTS = ("price_only", "price_macro")
STRATEGIES = ("direction_only", "probability_sized", "fractional_kelly")
STRATEGY_COLOURS = {"direction_only": "#7F7F7F", "probability_sized": "#0072B2", "fractional_kelly": "#D55E00",
                    "passive_equal_weight": "#009E73"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def common_features(context, cfg, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Macro (point-in-time, standardised) and the Stage 16 regime probability, on the trading calendar."""
    node = cfg.get("forecasting.features", {}) or {}
    specs, raw = ensure_macro_raw(cfg)
    z = macro_feature_panel(raw, specs, index, cfg)["z"]
    macro_names = list(node.get("macro", []))
    missing = [c for c in macro_names if c not in z.columns]
    if missing:
        raise KeyError(f"macro features not available: {missing}")
    common = z[macro_names].copy()
    regime_path = context.tables / "stage16_filtered_probabilities.csv"
    if not regime_path.exists():
        raise FileNotFoundError("Stage 16 must run first: it writes the filtered regime probabilities this stage uses")
    regime = pd.read_csv(regime_path, index_col=0, parse_dates=True)["hmm2_p_high"]
    common["p_high_vol"] = regime.reindex(index)
    return common


def walk_settings(cfg) -> dict:
    walk = cfg.get("forecasting.walk_forward", {}) or {}
    clf = cfg.get("forecasting.classifier", {}) or {}
    gauss = cfg.get("forecasting.gaussian", {}) or {}
    return {"min_train": int(walk.get("min_train_days", 1260)), "refit_every": int(walk.get("refit_every", 252)),
            "horizon": int(cfg.get("forecasting.horizon_days", 21)), "embargo": int(walk.get("embargo_days", 21)),
            "calibration_fraction": float(walk.get("calibration_fraction", 0.20)),
            "classifier_c": float(clf.get("C", 0.1)), "ridge_alpha": float(gauss.get("ridge_alpha", 100.0))}


# ---------------------------------------------------------------------------
# Scoring and tests
# ---------------------------------------------------------------------------
def skill_row(scores: pd.DataFrame, model: str, benchmark: str, lag: int = 0) -> dict:
    """Mean score, benchmark score, skill and the Diebold-Mariano test on the monthly differences."""
    test = diebold_mariano(scores[model], scores[benchmark], lag=lag, alternative="two-sided")
    return {"mean_model": float(scores[model].mean()), "mean_benchmark": float(scores[benchmark].mean()),
            "skill_score": float(1.0 - scores[model].mean() / scores[benchmark].mean()),
            "mean_difference": float(test["mean_loss_difference"]), "statistic": float(test["statistic"]),
            "p_value": float(test["p_value"]), "n_origins": int(len(scores))}


def classifier_summary(pred: pd.DataFrame, n_bins: int) -> dict:
    valid = pred.dropna(subset=["up", "p_cal", "p_raw"])
    y = valid["up"].to_numpy()
    return {
        "n_predictions": int(len(valid)), "base_rate_in_sample": float(y.mean()),
        "auc_calibrated": auc(valid["p_cal"].to_numpy(), y),
        "auc_by_asset_demeaned": auc((valid["p_cal"] - valid["base_rate"]).to_numpy(), y),
        "brier_model": float(brier_terms(valid["p_cal"].to_numpy(), y).mean()),
        "brier_benchmark": float(brier_terms(valid["base_rate"].to_numpy(), y).mean()),
        "ece_raw": expected_calibration_error(valid["p_raw"].to_numpy(), y, n_bins),
        "ece_calibrated": expected_calibration_error(valid["p_cal"].to_numpy(), y, n_bins),
        **{f"calibrated_{k}": v for k, v in calibration_slope_intercept(valid["p_cal"].to_numpy(), y).items()},
        **{f"raw_{k}": v for k, v in calibration_slope_intercept(valid["p_raw"].to_numpy(), y).items()},
        "share_of_forecasts_with_edge_above_no_trade": float(
            ((valid["p_cal"] - valid["base_rate"]).abs() >= 0.02).mean()),
        "mean_abs_edge": float((valid["p_cal"] - valid["base_rate"]).abs().mean()),
    }


def aligned_scores(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-origin scores of two variants on exactly the same (origin, asset) rows."""
    key = ["origin", "asset"]
    common = pred_a[key].merge(pred_b[key], on=key)
    a = pred_a.merge(common, on=key)
    b = pred_b.merge(common, on=key)
    return scores_by_origin(a), scores_by_origin(b)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def build_weight_books(pred: pd.DataFrame, columns: list[str], sizing: dict) -> dict[str, pd.DataFrame]:
    edge = float(sizing.get("no_trade_edge", 0.02))
    cap = float(sizing.get("max_weight", 0.25))
    kelly = float(sizing.get("kelly_fraction", 0.25))
    gross = float(sizing.get("max_gross", 1.0))
    origins = sorted(pred["origin"].unique())
    books = {name: pd.DataFrame(0.0, index=origins, columns=columns) for name in STRATEGIES}
    for origin, group in pred.groupby("origin"):
        group = group.dropna(subset=["p_cal", "base_rate", "sigma", "mu"]).set_index("asset")
        if group.empty:
            continue
        books["direction_only"].loc[origin, group.index] = edge_book(group, "direction_only", edge, cap).to_numpy()
        books["probability_sized"].loc[origin, group.index] = edge_book(group, "probability_sized", edge, cap).to_numpy()
        books["fractional_kelly"].loc[origin, group.index] = kelly_book(group, kelly, cap, gross).to_numpy()
    return books


def run_books(engine, books: dict[str, pd.DataFrame], returns: pd.DataFrame, investable: pd.DataFrame, label: str) -> dict:
    out = {}
    for name, book in books.items():
        full = book.reindex(returns.index).ffill().fillna(0.0)
        out[name] = engine.run(full, returns, f"{label}|{name}", investable, apply_vol_target=False)
    return out


def book_table(runs: dict, start: pd.Timestamp) -> pd.DataFrame:
    rows = {}
    for name, result in runs.items():
        net = slice_dates(result.net_returns, start, None)
        stats_ = performance_summary(net, turnover=slice_dates(result.turnover, start, None),
                                     costs=slice_dates(result.costs, start, None))
        weights = result.weights.reindex(net.index)
        rows[name] = {"sharpe": stats_["sharpe"], "cagr": stats_["cagr"], "ann_vol": stats_["ann_vol"],
                      "max_drawdown": stats_["max_drawdown"],
                      "ann_turnover": float(slice_dates(result.turnover, start, None).sum() / max(len(net) / 252.0, 1e-9)),
                      "mean_gross_exposure": float(weights.abs().sum(axis=1).mean()),
                      "mean_active_positions": float((weights.abs() > 1e-9).sum(axis=1).mean())}
    return pd.DataFrame(rows).T


def sizing_tests(runs: dict, start: pd.Timestamp, bootstrap: dict) -> pd.DataFrame:
    rows = []
    base = slice_dates(runs["direction_only"].net_returns, start, None)
    for name in ("probability_sized", "fractional_kelly"):
        test = paired_sharpe_test(slice_dates(runs[name].net_returns, start, None), base,
                                  n_samples=int(bootstrap.get("n_samples", 2000)),
                                  block_length=63, seed=int(bootstrap.get("seed", 7)))
        rows.append({"strategy": name, "benchmark": "direction_only", **test})
    return pd.DataFrame(rows)


def confidence_quintiles(pred: pd.DataFrame) -> pd.DataFrame:
    """Average SIGNED next-period return by quintile of |edge|: is more confidence worth more return?"""
    valid = pred.dropna(subset=["p_cal", "base_rate", "target"]).copy()
    valid["edge"] = valid["p_cal"] - valid["base_rate"]
    valid["abs_edge"] = valid["edge"].abs()
    valid["signed_return"] = np.sign(valid["edge"]) * valid["target"]
    valid["quintile"] = pd.qcut(valid["abs_edge"], 5, labels=False, duplicates="drop")
    grouped = valid.groupby("quintile")
    return pd.DataFrame({"mean_abs_edge": grouped["abs_edge"].mean(), "mean_signed_return": grouped["signed_return"].mean(),
                         "hit_rate": grouped["signed_return"].apply(lambda s: float((s > 0).mean())),
                         "standard_error": grouped["signed_return"].apply(lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))),
                         "n": grouped.size()})
