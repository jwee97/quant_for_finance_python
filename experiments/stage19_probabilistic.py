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

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import performance_summary
from src.data.macro import ensure_macro_raw
from src.features.macro import macro_feature_panel
from src.features.sleeves import month_end_dates
from src.models.probabilistic import (
    sign_book,
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
from src.utils.plotting import new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg, diebold_mariano
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context

STAGE = "stage19_probabilistic"
VARIANTS = ("price_only", "price_macro")
VARIANT_NAMES = {"price_only": "price only", "price_macro": "price + macro"}
STRATEGIES = ("direction_only", "probability_sized", "fractional_kelly")
CONTROLS = ("direction_only_raw", "probability_sized_raw", "gaussian_sign_only")      # post-hoc, never judged
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
    positive = bool((scores[model] > 0).all() and (scores[benchmark] > 0).all())      # a log density can be negative
    return {"mean_model": float(scores[model].mean()), "mean_benchmark": float(scores[benchmark].mean()),
            "skill_score": float(1.0 - scores[model].mean() / scores[benchmark].mean()) if positive else np.nan,
            "mean_difference": float(test["mean_loss_difference"]), "statistic": float(test["statistic"]),
            "p_value": float(test["p_value"]), "n_origins": int(len(scores))}


def classifier_summary(pred: pd.DataFrame, n_bins: int) -> dict:
    valid = pred.dropna(subset=["up", "p_cal", "p_raw"])
    y = valid["up"].to_numpy()
    return {
        "n_predictions": int(len(valid)), "base_rate_in_sample": float(y.mean()),
        "auc_calibrated": auc(valid["p_cal"].to_numpy(), y),
        "auc_raw": auc(valid["p_raw"].to_numpy(), y),
        "auc_raw_by_asset_demeaned": auc((valid["p_raw"] - valid["base_rate"]).to_numpy(), y),
        "refits": int(valid["refit_day"].nunique()),
        "refits_where_platt_flipped_the_sign": int((valid.groupby("refit_day")["platt_a"].first() < 0).sum()),
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
    names = list(STRATEGIES) + list(CONTROLS)
    books = {name: pd.DataFrame(0.0, index=origins, columns=columns) for name in names}
    for origin, group in pred.groupby("origin"):
        group = group.dropna(subset=["p_cal", "p_raw", "base_rate", "sigma", "mu"]).set_index("asset")
        if group.empty:
            continue
        books["direction_only"].loc[origin, group.index] = edge_book(group, "direction_only", edge, cap).to_numpy()
        books["probability_sized"].loc[origin, group.index] = edge_book(group, "probability_sized", edge, cap).to_numpy()
        books["fractional_kelly"].loc[origin, group.index] = kelly_book(group, kelly, cap, gross).to_numpy()
        # post-hoc controls: what the calibration step did, and sizing separated from the signal
        books["direction_only_raw"].loc[origin, group.index] = edge_book(group, "direction_only", edge, cap, "p_raw").to_numpy()
        books["probability_sized_raw"].loc[origin, group.index] = edge_book(group, "probability_sized", edge, cap, "p_raw").to_numpy()
        books["gaussian_sign_only"].loc[origin, group.index] = sign_book(group, cap).to_numpy()
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
                      "mean_net_exposure": float(weights.sum(axis=1).mean()),
                      "mean_active_positions": float((weights.abs() > 1e-9).sum(axis=1).mean())}
    return pd.DataFrame(rows).T


def sizing_tests(runs: dict, start: pd.Timestamp, bootstrap: dict) -> pd.DataFrame:
    rows = []
    base = slice_dates(runs["direction_only"].net_returns, start, None)
    for name in ("probability_sized", "fractional_kelly"):
        test = paired_sharpe_test(slice_dates(runs[name].net_returns, start, None), base,
                                  n_samples=int(bootstrap.get("n_samples", 2000)),
                                  block_length=21 * int(bootstrap.get("block_length", 3)),   # months -> trading days
                                  seed=int(bootstrap.get("seed", 7)))
        rows.append({"strategy": name, "benchmark": "direction_only", **test})
    return pd.DataFrame(rows)


def posthoc_tests(runs: dict, passive, start: pd.Timestamp, bootstrap: dict) -> pd.DataFrame:
    """Controls that separate signal from sizing and put the books next to passive ownership."""
    pairs = [("fractional_kelly", "gaussian_sign_only"), ("probability_sized_raw", "direction_only_raw"),
             ("fractional_kelly", "passive_equal_weight"), ("direction_only", "passive_equal_weight"),
             ("direction_only_raw", "direction_only"), ("probability_sized_raw", "probability_sized")]
    streams = {name: slice_dates(result.net_returns, start, None) for name, result in runs.items()}
    streams["passive_equal_weight"] = slice_dates(passive.net_returns, start, None)
    rows = []
    for a, b in pairs:
        test = paired_sharpe_test(streams[a], streams[b], n_samples=int(bootstrap.get("n_samples", 2000)),
                                  block_length=21 * int(bootstrap.get("block_length", 3)),
                                  seed=int(bootstrap.get("seed", 7)))
        rows.append({"strategy": a, "benchmark": b, **test})
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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_calibration(pred: dict, scores: dict, ece_test: dict, summary: dict, n_bins: int, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    main = pred["price_only"].dropna(subset=["up", "p_cal", "p_raw"])
    ax = axes[0, 0]
    for column, label, colour in (("p_raw", "raw classifier", "#CC79A7"), ("p_cal", "after Platt scaling", "#0072B2")):
        table = reliability_table(main[column].to_numpy(), main["up"].to_numpy(), n_bins)
        used = table[table["count"] > 0]
        ax.plot(used["mean_forecast"], used["observed_frequency"], "-", color=colour, linewidth=1.2, label=label)
        ax.scatter(used["mean_forecast"], used["observed_frequency"], color=colour,
                   s=np.clip(used["count"] / used["count"].max() * 160, 12, 160))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, label="perfect calibration")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("Forecast probability that the 21-day return is positive")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability: ECE {summary['ece_raw']:.3f} raw, {summary['ece_calibrated']:.3f} calibrated")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.98, 0.04, f"calibrated slope {summary['calibrated_slope']:.2f}, AUC {summary['auc_calibrated']:.3f}",
            transform=ax.transAxes, ha="right", fontsize=8.5)

    for ax, (model, bench, title, ylabel) in zip(
            (axes[0, 1], axes[1, 0]),
            (("logloss_model", "logloss_benchmark", "Direction: log loss against the base rate", "Cumulative log-loss difference"),
             ("crps_model", "crps_benchmark", "Return: CRPS against the historical-mean Gaussian", "Cumulative CRPS difference"))):
        for variant, colour in zip(VARIANTS, ("#0072B2", "#D55E00")):
            diff = (scores[variant][model] - scores[variant][bench]).cumsum()
            ax.plot(diff.index, diff.to_numpy(), color=colour, linewidth=1.5, label=VARIANT_NAMES[variant])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel + " (below zero: skill)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    ax = axes[1, 1]
    valid = pred["price_only"].dropna(subset=["target", "mu", "sigma"])
    pit = pit_gaussian(valid["target"].to_numpy(), valid["mu"].to_numpy(), valid["sigma"].to_numpy())
    ax.hist(pit, bins=10, range=(0, 1), color="#009E73", edgecolor="white", density=True)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    check = pit_uniformity(pit)
    ax.set_xlabel("Probability integral transform of the realised return")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT of the Gaussian forecast: KS {check['ks_statistic']:.3f}; "
                 f"{100 * check['share_below_10pct']:.0f}% below 0.1, {100 * check['share_above_90pct']:.0f}% above 0.9")
    fig.suptitle("Figure 38. Are the forecast distributions honest, and do they beat a base rate?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Are the probability forecasts calibrated, do they beat benchmarks that carry no "
                           "feature information, and is the Gaussian forecast's spread right?", 38)


def figure_sizing(runs: dict, passive, start, tests: pd.DataFrame, quintiles: pd.DataFrame, table: pd.DataFrame, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    ax = axes[0, 0]
    for name in STRATEGIES:
        net = slice_dates(runs["price_only"][name].net_returns, start, None)
        ax.plot(net.index, (1.0 + net).cumprod().to_numpy(), color=STRATEGY_COLOURS[name], linewidth=1.4,
                label=name.replace("_", " "))
    curve = (1.0 + slice_dates(passive.net_returns, start, None)).cumprod()
    ax.plot(curve.index, curve.to_numpy(), color=STRATEGY_COLOURS["passive_equal_weight"], linewidth=1.2,
            linestyle="--", label="passive equal weight (reference)")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_ylabel("Growth of 1 unit, net of costs")
    ax.set_title("Three ways to use the same price-only forecasts")
    ax.legend(fontsize=8, loc="best")

    ax = axes[0, 1]
    rows = tests.reset_index(drop=True)
    y = np.arange(len(rows))
    for i, row in rows.iterrows():
        colour = "#009E73" if bool(row["passes"]) else ("#0072B2" if row["variant"] == "price_only" else "#7F7F7F")
        ax.errorbar(row["difference"], i, xerr=[[row["difference"] - row["ci_lower_5pct"]], [row["ci_upper_95pct"] - row["difference"]]],
                    fmt="o", color=colour, capsize=3)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y, [f"{r['strategy'].replace('_', ' ')}\n({VARIANT_NAMES[r['variant']]})" for _, r in rows.iterrows()], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Net Sharpe difference against direction-only (90% bootstrap interval)")
    ax.set_title("Does confidence-aware sizing beat direction-only?  (blue: pre-declared)")

    ax = axes[1, 0]
    x = np.arange(len(quintiles))
    ax.bar(x, 100 * quintiles["mean_signed_return"], yerr=100 * quintiles["standard_error"], color="#0072B2", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"Q{int(q) + 1}\n|edge| {100 * e:.1f}pp" for q, e in zip(quintiles.index, quintiles["mean_abs_edge"])], fontsize=8.5)
    ax.set_ylabel("Mean SIGNED next-month return, % (+1 s.e.)")
    ax.set_title("Is a bigger edge worth a bigger bet?  Return by confidence quintile")

    ax = axes[1, 1]
    labels = list(table.index)
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, table["mean_gross_exposure"], width=0.4, color="#56B4E9", label="mean gross exposure")
    ax.bar(xs + 0.2, table["ann_turnover"], width=0.4, color="#E69F00", label="annual turnover (x)")
    ax.set_xticks(xs, [l.replace("_", "\n") for l in labels], fontsize=8.5)
    for xi, (g, t, n) in zip(xs, zip(table["mean_gross_exposure"], table["ann_turnover"], table["mean_active_positions"])):
        ax.text(xi, max(g, t), f"{n:.1f} names", ha="center", va="bottom", fontsize=8)
    ax.set_title("What each rule actually holds (price-only forecasts)")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Figure 39. Sizing positions by confidence", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Does sizing positions by calibrated confidence or fractional Kelly beat a "
                           "direction-only book, and is a bigger edge worth a bigger bet?", 39)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def _direction(value: float, lower: str, higher: str) -> str:
    return lower if value < 0 else higher


def _sizing_caveat(name: str, posthoc: pd.DataFrame, table: pd.DataFrame, passive_stats: dict) -> str:
    def row(a, b):
        return posthoc[(posthoc["strategy"] == a) & (posthoc["benchmark"] == b)].iloc[0]

    passive = float(passive_stats["sharpe"])
    if name == "fractional_kelly":
        vs_sign, vs_passive = row("fractional_kelly", "gaussian_sign_only"), row("fractional_kelly", "passive_equal_weight")
        return ("CAUTION, post-hoc controls: this comparison is confounded by design. Direction-only trades the classifier's "
                "signal long and short and loses; the Kelly book uses a different signal (the Gaussian mean) and is net long, "
                f"mean net exposure {float(table.loc['fractional_kelly', 'mean_net_exposure']):.2f}, through a long bull market. "
                f"Against passive equal weight (net Sharpe {passive:.2f}) the Kelly book's difference is {float(vs_passive['difference']):+.2f} "
                f"(p={float(vs_passive['p_value']):.3f}); against sign-only sizing of the SAME Gaussian signal it is "
                f"{float(vs_sign['difference']):+.2f} (p={float(vs_sign['p_value']):.3f}). Retained by the letter of the rule, "
                "not as evidence that Kelly sizing adds value.")
    vs = row("probability_sized_raw", "direction_only_raw")
    return ("Post-hoc control with the RAW probabilities (no Platt step): probability-sized minus direction-only "
            f"{float(vs['difference']):+.2f} (p={float(vs['p_value']):.3f}).")


def log_registry(context, cfg, preds, skill, macro_tests, ece_test, summary, sizing_family, sizing_all, table,
                 pit_check, settings, first_origin, quintiles, posthoc, passive_stats) -> None:
    registry = context.registry
    decisions = cfg.get("forecasting.decisions", {}) or {}
    fdr = float(decisions.get("fdr", 0.10))
    cost_bps = float(cfg.get("backtest.costs.cost_bps", 10.0))
    common = {"horizon_days": settings["horizon"], "refit_every": settings["refit_every"],
              "min_train_days": settings["min_train"], "embargo_days": settings["embargo"],
              "calibration_fraction": settings["calibration_fraction"], "classifier_C": settings["classifier_c"],
              "ridge_alpha": settings["ridge_alpha"]}
    train_period = f"expanding window from the start of data, refit every {settings['refit_every']} days"
    test_period = f"{first_origin.date()} onward, monthly origins"

    registry.log(
        "Platt scaling lowers the expected calibration error of the raw direction classifier.",
        stage=STAGE, parameters={**common, "test": "paired block bootstrap over months, one-sided 5%"},
        train_period=train_period, test_period=test_period,
        results={"ece_raw": ece_test["ece_raw"], "ece_calibrated": ece_test["ece_calibrated"],
                 "difference": ece_test["difference"], "p_value_not_improved": ece_test["p_value_not_improved"],
                 "calibrated_slope": summary["calibrated_slope"], "raw_slope": summary["raw_slope"]},
        decision="retain" if (ece_test["difference"] < 0 and ece_test["p_value_not_improved"] < 0.05) else "reject",
        notes=("Retained only if the calibrated ECE is lower and the paired bootstrap rejects 'no improvement' at 5%. "
               f"Calibration slope {summary['raw_slope']:.2f} raw and {summary['calibrated_slope']:.2f} calibrated (1 is perfect). "
               f"Why: the Platt slope is estimated on the last 20% of each training block, one to three years or "
               f"12 to 36 months that share the market, so it is very noisy; it came out NEGATIVE in "
               f"{summary['refits_where_platt_flipped_the_sign']} of {summary['refits']} refits, reversing the classifier's ordering "
               f"there (pooled AUC {summary['auc_raw']:.3f} raw, {summary['auc_calibrated']:.3f} after calibration). "
               "Calibration makes probabilities honest about the information they have; fitted on too little data it "
               "manufactures a confident, wrong signal. Raw-classifier results are reported as post-hoc controls."),
    )

    for variant in ("price_only",):
        row = skill[(skill["variant"] == variant) & (skill["forecast"] == "direction (log loss)")].iloc[0]
        registry.log(
            "The calibrated price-only direction classifier has a lower log loss than the per-asset base rate.",
            stage=STAGE, parameters={**common, "test": "Diebold-Mariano on monthly cross-asset mean loss differences, two-sided"},
            train_period=train_period, test_period=test_period,
            results={"mean_log_loss_model": row["mean_model"], "mean_log_loss_base_rate": row["mean_benchmark"],
                     "skill_score": row["skill_score"], "p_value": row["p_value"], "n_origins": row["n_origins"],
                     "auc": summary["auc_calibrated"], "auc_after_removing_asset_base_rates": summary["auc_by_asset_demeaned"]},
            decision="retain" if (row["mean_difference"] < 0 and row["p_value"] < 0.05) else "reject",
            notes=("The benchmark knows each asset's historical base rate, which is itself strong information (equities rise in "
                   "about 60% of months). Skill means doing better than that. AUC after removing each asset's base rate is "
                   f"{summary['auc_by_asset_demeaned']:.3f} for the calibrated probabilities and "
                   f"{summary['auc_raw_by_asset_demeaned']:.3f} for the raw ones (0.5 is no information). Post-hoc: the RAW "
                   "classifier's log loss is "
                   + (lambda r: f"{r['mean_model']:.4f} against the benchmark's {r['mean_benchmark']:.4f} (p={r['p_value']:.3f})")(
                       skill[(skill['variant'] == 'price_only') & (skill['forecast'].str.contains('raw classifier'))].iloc[0])
                   + ": the calibration step, not the features, is where most of the damage is done."),
        )
        row = skill[(skill["variant"] == variant) & (skill["forecast"] == "return (CRPS)")].iloc[0]
        registry.log(
            "The price-only Gaussian forecast has a lower CRPS than the same Gaussian centred on the asset's historical mean.",
            stage=STAGE, parameters={**common, "test": "Diebold-Mariano on monthly cross-asset mean loss differences, two-sided"},
            train_period=train_period, test_period=test_period,
            results={"mean_crps_model": row["mean_model"], "mean_crps_benchmark": row["mean_benchmark"],
                     "skill_score": row["skill_score"], "p_value": row["p_value"], "n_origins": row["n_origins"],
                     "pit_ks_statistic": pit_check["ks_statistic"], "pit_share_below_10pct": pit_check["share_below_10pct"],
                     "pit_share_above_90pct": pit_check["share_above_90pct"]},
            decision="retain" if (row["mean_difference"] < 0 and row["p_value"] < 0.05) else "reject",
            notes=("Both forecasts use the same EWMA volatility, so the comparison isolates the ridge mean. The PIT histogram "
                   f"checks the spread: {100 * pit_check['share_below_10pct']:.0f}% of realised returns fell below the forecast's "
                   f"10th percentile and {100 * pit_check['share_above_90pct']:.0f}% above its 90th (10% each if calibrated)."),
        )

    macro_table = pd.DataFrame(macro_tests)
    any_better = bool((macro_table["bh_significant"] & (macro_table["mean_difference"] < 0)).any())
    registry.log(
        "Adding macro and regime features improves the probabilistic forecasts (log loss or CRPS) over price features alone.",
        stage=STAGE, parameters={**common, "test": "Diebold-Mariano, two-sided, BH across two", "fdr": fdr,
                                 "macro_features": cfg.get("forecasting.features.macro"),
                                 "interactions": cfg.get("forecasting.macro_interactions")},
        train_period=train_period, test_period=test_period,
        results={f"{r['forecast']}_mean_difference": r["mean_difference"] for _, r in macro_table.iterrows()}
        | {f"{r['forecast']}_p_value": r["p_value"] for _, r in macro_table.iterrows()},
        decision="retain" if any_better else "reject",
        notes=("Retained only if at least one of the two tests is significant after BH control with the price+macro variant "
               "scoring better. Macro and regime features enter alone and interacted with asset-class dummies. Training rows "
               "before the regime model's first out-of-sample probability (December 2008) are dropped for this variant."),
    )

    for _, row in sizing_family.iterrows():
        name = row["strategy"]
        registry.log(
            {"probability_sized": "Sizing positions by the size of the calibrated edge beats direction-only sizing of the same signal, "
                                  "net of costs.",
             "fractional_kelly": "Fractional-Kelly sizing from the Gaussian forecast beats direction-only sizing of the classifier "
                                 "signal, net of costs."}[name],
            stage=STAGE,
            parameters={**common, "sizing": cfg.get("forecasting.sizing"), "rebalance": "monthly",
                        "test": "paired stationary bootstrap on net Sharpe, BH across two", "forecasts": "price-only"},
            train_period=train_period, test_period=f"{first_origin.date()} onward, net of costs", cost_bps=cost_bps,
            results={"net_sharpe": float(table.loc[name, "sharpe"]), "net_sharpe_direction_only": float(table.loc["direction_only", "sharpe"]),
                     "sharpe_difference": float(row["difference"]), "p_value": float(row["p_value"]),
                     "bh_significant": bool(row["bh_significant"]), "annual_turnover": float(table.loc[name, "ann_turnover"]),
                     "mean_gross_exposure": float(table.loc[name, "mean_gross_exposure"])},
            decision="retain" if bool(row["passes"]) else "reject",
            notes=("Retained only if the paired bootstrap is significant after BH control with a POSITIVE difference. "
                   "Sizing by confidence can only help if confidence carries information: realised signed return by "
                   f"|edge| quintile runs from {100 * float(quintiles['mean_signed_return'].iloc[0]):.2f}% (smallest) to "
                   f"{100 * float(quintiles['mean_signed_return'].iloc[-1]):.2f}% (largest). The price+macro versions are in "
                   "stage19_sizing_tests.csv and are not judged. "
                   + _sizing_caveat(name, posthoc, table, passive_stats)),
        )

    registry.log(
        "Descriptive: how much information do these features carry about monthly ETF direction?",
        stage=STAGE, parameters={"post_hoc": True}, train_period=train_period, test_period=test_period,
        results={"auc_pooled": summary["auc_calibrated"], "auc_asset_base_rates_removed": summary["auc_by_asset_demeaned"],
                 "mean_abs_edge": summary["mean_abs_edge"],
                 "share_of_forecasts_above_no_trade_edge": summary["share_of_forecasts_with_edge_above_no_trade"],
                 "brier_model": summary["brier_model"], "brier_base_rate": summary["brier_benchmark"]},
        decision="record",
        notes="Descriptive statistics of the price-only classifier; none feeds a decision.",
    )


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 19: probabilistic forecasting").parse_args(argv)
    context, logger = build_context(STAGE, generation=2)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 19 | probabilistic forecasting and sizing (Generation 2, Priority 2)")
    logger.info("=" * 72)
    market = context.market_data()
    prices = market.prices
    returns = market.returns()
    index = pd.DatetimeIndex(prices.index)
    settings = walk_settings(cfg)
    decisions = cfg.get("forecasting.decisions", {}) or {}
    fdr = float(decisions.get("fdr", 0.10))
    bootstrap = decisions.get("bootstrap", {}) or {}
    n_bins = int(cfg.get("forecasting.scoring.reliability_bins", 10))
    sleeve_of = {asset: sleeve for sleeve, members in (cfg.get("forecasting.sleeves", {}) or {}).items() for asset in members}

    common = common_features(context, cfg, index)
    origins = month_end_dates(index)
    panel = build_panel(prices, origins, settings["horizon"], float(cfg.get("forecasting.gaussian.vol_halflife", 40)),
                        common, sleeve_of)
    logger.info("panel: %d origins x %d assets, %d price features, %d macro/regime features",
                len(origins), prices.shape[1], len(panel.price_columns), len(panel.common_columns))
    preds = {}
    for variant in VARIANTS:
        preds[variant] = walk_forward_probabilistic(panel, variant, **settings, n_total_days=len(index))
        logger.info("%s: %d out-of-sample forecasts from %s", variant, len(preds[variant]),
                    preds[variant]["origin"].min().date())
        context.save_table(preds[variant].drop(columns=["sleeve"]), f"stage19_predictions_{variant}.csv", index=False)
    first_origin = preds["price_only"]["origin"].min()

    # ------------------------------------------------------------ scores and tests
    scores = {v: scores_by_origin(preds[v]) for v in VARIANTS}
    skill_rows = []
    for variant in VARIANTS:
        for label, model, bench in (("direction (log loss)", "logloss_model", "logloss_benchmark"),
                                    ("direction (log loss, raw classifier; post-hoc)", "logloss_raw", "logloss_benchmark"),
                                    ("direction (Brier)", "brier_model", "brier_benchmark"),
                                    ("return (CRPS)", "crps_model", "crps_benchmark"),
                                    ("return (negative log score)", "logscore_model", "logscore_benchmark")):
            skill_rows.append({"variant": variant, "forecast": label, **skill_row(scores[variant], model, bench)})
    skill = pd.DataFrame(skill_rows)
    context.save_table(skill, "stage19_skill_scores.csv", index=False)
    logger.info("skill against the benchmarks (negative difference = better):\n%s",
                skill[["variant", "forecast", "mean_model", "mean_benchmark", "skill_score", "p_value"]].round(5).to_string(index=False))
    for variant in VARIANTS:
        context.save_table(scores[variant], f"stage19_scores_by_origin_{variant}.csv")

    sa, sb = aligned_scores(preds["price_only"], preds["price_macro"])
    macro_tests = []
    for label, column in (("log_loss", "logloss_model"), ("crps", "crps_model")):
        test = diebold_mariano(sb[column], sa[column], lag=0, alternative="two-sided")
        macro_tests.append({"forecast": label, "n_origins": int(len(sa)), "mean_price_only": float(sa[column].mean()),
                            "mean_price_macro": float(sb[column].mean()), "mean_difference": float(test["mean_loss_difference"]),
                            "statistic": float(test["statistic"]), "p_value": float(test["p_value"])})
    macro_table = pd.DataFrame(macro_tests)
    macro_table["bh_significant"] = benjamini_hochberg(macro_table["p_value"], fdr)
    context.save_table(macro_table, "stage19_macro_vs_price.csv", index=False)
    logger.info("price+macro against price-only on identical rows:\n%s", macro_table.round(5).to_string(index=False))

    ece_test = paired_ece_test(preds["price_only"], int(bootstrap.get("n_samples", 2000)),
                               int(bootstrap.get("block_length", 3)), int(bootstrap.get("seed", 7)), n_bins)
    summary = classifier_summary(preds["price_only"], n_bins)
    context.save_table(pd.Series({**summary, **{f"ece_test_{k}": v for k, v in ece_test.items()}}, name="value").to_frame(),
                       "stage19_classifier_summary.csv")
    logger.info("calibration: ECE raw %.4f, calibrated %.4f (p not improved %.3f); slope raw %.2f calibrated %.2f; "
                "AUC %.3f (asset base rates removed %.3f)", ece_test["ece_raw"], ece_test["ece_calibrated"],
                ece_test["p_value_not_improved"], summary["raw_slope"], summary["calibrated_slope"],
                summary["auc_calibrated"], summary["auc_by_asset_demeaned"])
    for column in ("p_raw", "p_cal"):
        valid = preds["price_only"].dropna(subset=["up", column])
        context.save_table(reliability_table(valid[column].to_numpy(), valid["up"].to_numpy(), n_bins),
                           f"stage19_reliability_{column}.csv", index=False)
    valid = preds["price_only"].dropna(subset=["target", "mu", "sigma"])
    pit_check = pit_uniformity(pit_gaussian(valid["target"].to_numpy(), valid["mu"].to_numpy(), valid["sigma"].to_numpy()))
    logger.info("Gaussian PIT: %s", {k: round(v, 3) for k, v in pit_check.items()})
    quintiles = confidence_quintiles(preds["price_only"])
    context.save_table(quintiles, "stage19_confidence_quintiles.csv")
    logger.info("signed return by |edge| quintile:\n%s", quintiles.round(5).to_string())

    # ------------------------------------------------------------ sizing
    engine = BacktestEngine.from_config(cfg)
    sizing = cfg.get("forecasting.sizing", {}) or {}
    runs, tables, tests = {}, {}, []
    for variant in VARIANTS:
        books = build_weight_books(preds[variant], list(prices.columns), sizing)
        runs[variant] = run_books(engine, books, returns, market.investable, variant)
        tables[variant] = book_table(runs[variant], first_origin)
        family = sizing_tests(runs[variant], first_origin, bootstrap)
        family.insert(0, "variant", variant)
        tests.append(family)
        logger.info("%s books from %s (net):\n%s", variant, first_origin.date(), tables[variant].round(4).to_string())
    equal = pd.DataFrame(1.0 / prices.shape[1], index=sorted(preds["price_only"]["origin"].unique()), columns=prices.columns)
    passive = engine.run(equal.reindex(returns.index).ffill().fillna(0.0), returns, "passive_equal_weight",
                         market.investable, apply_vol_target=False)
    posthoc = posthoc_tests(runs["price_only"], passive, first_origin, bootstrap)
    context.save_table(posthoc, "stage19_posthoc_sizing_controls.csv", index=False)
    passive_stats = performance_summary(slice_dates(passive.net_returns, first_origin, None))
    logger.info("post-hoc controls (price-only forecasts; passive equal-weight net Sharpe %.3f):\n%s",
                passive_stats["sharpe"], posthoc[["strategy", "benchmark", "sharpe_a", "sharpe_b", "difference", "p_value"]].round(3).to_string(index=False))
    all_tests = pd.concat(tests, ignore_index=True)
    primary = all_tests["variant"] == "price_only"
    all_tests["bh_significant"] = False
    all_tests.loc[primary, "bh_significant"] = benjamini_hochberg(all_tests.loc[primary, "p_value"], fdr)
    all_tests["passes"] = all_tests["bh_significant"] & (all_tests["difference"] > 0)
    context.save_table(all_tests, "stage19_sizing_tests.csv", index=False)
    context.save_table(pd.concat(tables, names=["variant"]), "stage19_book_performance.csv")
    logger.info("sizing tests vs direction-only (BH across the two price-only tests):\n%s",
                all_tests[["variant", "strategy", "sharpe_a", "sharpe_b", "difference", "p_value", "bh_significant", "passes"]]
                .round(4).to_string(index=False))

    # ------------------------------------------------------------ figures and registry
    figure_calibration(preds, scores, ece_test, summary, n_bins, context.figure("fig38_probabilistic_calibration.png"))
    forest = all_tests.copy()
    figure_sizing(runs, passive, first_origin, forest, quintiles, tables["price_only"],
                  context.figure("fig39_confidence_sizing.png"))
    log_registry(context, cfg, preds, skill, macro_table.to_dict("records"), ece_test, summary, all_tests[primary], all_tests,
                 tables["price_only"], pit_check, settings, first_origin, quintiles, posthoc, passive_stats)
    logger.info("STAGE 19 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
