"""Stage 15 - Macro features (Generation 2, Priority 5).

Three questions, in order of how much they should be trusted:

1. DATA. Can macro series be used without look-ahead? Each carries an explicit
   publication lag; the panel is built from availability dates.
2. INFORMATION. Do macro features improve out-of-sample forecasts of monthly
   asset-class excess returns BEYOND what price features already give? Tested
   with nested-model Clark-West tests, per sleeve, with FDR control.
3. STRATEGY. Does a signal f(price, macro) beat f(price) after costs?

A control runs alongside: the same study with publication lags deliberately
ignored. It measures how much the classic macro-data mistake would have
flattered the result, which is the justification for all the lag machinery.

Figures 28-29.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.data.macro import (
    MacroDownloader,
    MacroSeriesSpec,
    load_macro_raw,
    load_specs,
    macro_data_version,
    staleness,
)
from src.features.macro import FEATURE_COLUMNS, macro_feature_panel, monthly_grid
from src.features.sleeves import sleeve_returns
from src.features.volatility import rolling_volatility
from src.models.macro_forecast import build_monthly_panel, evaluate_nested, walk_forward_forecasts
from src.signals.transform import signal_to_positions
from src.utils.dates import slice_dates
from src.utils.plotting import PALETTE, new_axes, plot_heatmap, save_figure
from src.validation.forecast_tests import benjamini_hochberg, diebold_mariano
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context
from experiments.strategies import transform_config

STAGE = "stage15_macro"
COMPARISONS = [("hist", "price"), ("hist", "macro"), ("price", "both")]


def feature_dimension(z: pd.DataFrame) -> dict:
    """How many independent dimensions do the standardised features carry?

    The participation ratio of the correlation eigenvalues, (sum l)^2 / sum l^2,
    is 1 when one factor drives everything and N when the features are
    orthogonal. Thirteen series that move together are not thirteen predictors.
    """
    sample = z.resample("ME").last().dropna()
    corr = sample.corr()
    eigenvalues = np.sort(np.linalg.eigvalsh(corr.to_numpy()))[::-1]
    share = np.cumsum(eigenvalues) / eigenvalues.sum()
    return {
        "corr": corr,
        "eigenvalues": eigenvalues,
        "effective_dimensions": float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()),
        "components_for_80pct": int(np.searchsorted(share, 0.80) + 1),
        "n_features": int(corr.shape[0]),
        "n_months": int(len(sample)),
    }


def lag_inflation_tests(result: dict, naive_result: dict, models=("macro", "both"),
                        fdr: float = 0.10) -> pd.DataFrame:
    """Does ignoring publication lags make the SAME model forecast better?

    Same regressors, different timing, so the models are not nested and a plain
    Diebold-Mariano test on squared errors is appropriate. A positive statistic
    means the lag-respecting forecast has the larger loss, i.e. ignoring lags
    helped. Reported per sleeve and model; one-sided, BH-controlled.
    """
    rows = []
    actual = result["actual"]
    for model in models:
        for sleeve in actual.columns:
            frame = pd.concat([actual[sleeve].rename("y"),
                               result["forecasts"][model][sleeve].rename("respected"),
                               naive_result["forecasts"][model][sleeve].rename("ignored")],
                              axis=1).dropna()
            if len(frame) < 24:
                continue
            loss_respected = (frame["y"] - frame["respected"]) ** 2
            loss_ignored = (frame["y"] - frame["ignored"]) ** 2
            test = diebold_mariano(loss_respected, loss_ignored, lag=0, alternative="greater")
            if test:
                rows.append({"model": model, "sleeve": sleeve, "n_obs": len(frame),
                             "mse_respected": float(loss_respected.mean()),
                             "mse_ignored": float(loss_ignored.mean()),
                             "statistic": test["statistic"], "p_value": test["p_value"]})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["bh_significant"] = benjamini_hochberg(table["p_value"], fdr)
    return table


def figure_panel(context, z, levels, raw, specs, stale, dimension, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.0))

    monthly = z.resample("QE").last().dropna(how="all")
    image = plot_heatmap(axes[0, 0], monthly.T, "Standardised macro state (expanding z-score)",
                         vmin=-3, vmax=3, annotate=False)
    ticks = np.linspace(0, len(monthly) - 1, 8).astype(int)
    axes[0, 0].set_xticks(ticks, [monthly.index[i].year for i in ticks], rotation=0)
    fig.colorbar(image, ax=axes[0, 0], shrink=0.7)

    cpi = raw["CPIAUCNS"]
    yoy_reference = cpi.pct_change(12).loc["2020-06-01":"2023-12-31"]
    spec = next(s for s in specs if s.id == "CPIAUCNS")
    known = levels["CPI_YOY"].loc["2020-06-01":"2023-12-31"]
    axes[0, 1].step(yoy_reference.index, 100 * yoy_reference.to_numpy(), where="post",
                    color="#D55E00", label="by reference month (what a naive join uses)")
    axes[0, 1].plot(known.index, 100 * known.to_numpy(), color="#0072B2", linewidth=1.6,
                    label=f"as known on the day ({spec.release_lag_days}-day publication lag)")
    axes[0, 1].set_ylabel("CPI inflation, % y/y")
    axes[0, 1].set_title("The same series, two dates: reference vs publication")
    axes[0, 1].legend(fontsize=8, loc="lower right")

    summary = stale.describe().T[["mean", "max"]].sort_values("mean")
    y = np.arange(len(summary))
    axes[1, 0].barh(y - 0.2, summary["mean"].to_numpy(), height=0.4, color="#0072B2", label="mean")
    axes[1, 0].barh(y + 0.2, summary["max"].to_numpy(), height=0.4, color="#CC79A7", label="max")
    axes[1, 0].set_yticks(y, summary.index)
    axes[1, 0].set_xlabel("Days since the value in use was published")
    axes[1, 0].set_title("How stale is each series on a typical day?")
    axes[1, 0].legend(fontsize=8, loc="center right")
    gaps = []
    for spec_ in specs:
        if spec_.frequency == "monthly":
            grid = monthly_grid(raw[spec_.id])
            missing = [d.strftime("%b %Y") for d in grid.index[grid.isna()]]
            if missing:
                gaps.append(f"{spec_.id}: no print for {', '.join(missing)}")
    if gaps:
        axes[1, 0].text(0.98, 0.04, "\n".join(gaps) + "\n(these gaps set the 60-day maxima)",
                        transform=axes[1, 0].transAxes, ha="right", va="bottom", fontsize=7.5,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999", alpha=0.9))

    image = plot_heatmap(axes[1, 1], dimension["corr"], "", vmin=-1, vmax=1, annotate=True)
    for label in axes[1, 1].get_xticklabels() + axes[1, 1].get_yticklabels():
        label.set_fontsize(8)
    fig.colorbar(image, ax=axes[1, 1], shrink=0.7)
    axes[1, 1].set_title(f"{dimension['n_features']} features, but about "
                         f"{dimension['effective_dimensions']:.1f} independent dimensions")

    fig.suptitle("Figure 28. The macro panel and the date that matters",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Can macro series be used without look-ahead, how stale is the "
                           "information on a typical trading day, and how many independent "
                           "dimensions do the features really carry?", 28)


def figure_predictability(context, table, naive_table, curves, test, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.0))
    sleeves = list(table["sleeve"].unique())

    x = np.arange(len(sleeves))
    any_significant = False
    for i, (restricted, unrestricted, label) in enumerate(
        [("hist", "price", "price vs history"), ("hist", "macro", "macro vs history"),
         ("price", "both", "price+macro vs price")]
    ):
        subset = table[(table["restricted"] == restricted) & (table["unrestricted"] == unrestricted)]
        subset = subset.set_index("sleeve").reindex(sleeves)
        bars = axes[0, 0].bar(x + (i - 1) * 0.27, 100 * subset["oos_r2"].to_numpy(), width=0.27,
                              color=PALETTE[i], label=label)
        for rect, flag in zip(bars, subset["bh_significant"].fillna(False)):
            if flag:
                any_significant = True
                axes[0, 0].text(rect.get_x() + rect.get_width() / 2, rect.get_height(), "*",
                                ha="center", va="bottom", fontsize=14, fontweight="bold")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 0].set_xticks(x, sleeves, rotation=15)
    axes[0, 0].set_ylabel("Out-of-sample R-squared (%)")
    axes[0, 0].set_title("Does the model beat its benchmark? (* = significant after FDR)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    low, high = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(low, high + 0.22 * (high - low))                   # headroom for the note
    if not any_significant:
        axes[0, 0].text(0.02, 0.97, "No comparison is significant after FDR control",
                        transform=axes[0, 0].transAxes, ha="left", va="top", fontsize=9,
                        fontstyle="italic", bbox=dict(boxstyle="round", facecolor="white",
                                                      edgecolor="#999999", alpha=0.9))

    subset = table[(table["restricted"] == "price") & (table["unrestricted"] == "both")].set_index("sleeve").reindex(sleeves)
    axes[0, 1].bar(x, subset["statistic"].to_numpy(), color="#0072B2")
    axes[0, 1].axhline(1.645, color="#CC0000", linestyle="--", linewidth=1.0, label="5% one-sided")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 1].set_xticks(x, sleeves, rotation=15)
    axes[0, 1].set_ylabel("Clark-West statistic")
    axes[0, 1].set_title("Does macro add information BEYOND price?")
    low, high = axes[0, 1].get_ylim()
    axes[0, 1].set_ylim(low, max(high, 1.645) + 0.35 * (max(high, 1.645) - low))
    axes[0, 1].legend(fontsize=8, loc="upper left")

    width = 0.2
    for j, (comparison, hatch, name) in enumerate([("macro_vs_hist", None, "macro vs history"),
                                                   ("both_vs_price", "//", "price+macro vs price")]):
        for k, (frame, color, label) in enumerate([(table, "#0072B2", "lags respected"),
                                                   (naive_table, "#D55E00", "lags ignored")]):
            subset = frame[frame["comparison"] == comparison].set_index("sleeve").reindex(sleeves)
            axes[1, 0].bar(x + (2 * j + k - 1.5) * width, 100 * subset["oos_r2"].to_numpy(), width=width,
                           color=color, hatch=hatch, edgecolor="white", label=f"{name}: {label}")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 0].set_xticks(x, sleeves, rotation=15)
    axes[1, 0].set_ylabel("Out-of-sample R-squared (%)")
    axes[1, 0].set_title("What ignoring publication lags does to apparent skill")
    axes[1, 0].legend(fontsize=7.5, loc="lower right", ncol=2)

    for i, (name, curve) in enumerate(curves.items()):
        axes[1, 1].plot(curve.index, curve.to_numpy(), color=PALETTE[i], linewidth=1.4, label=name)
    axes[1, 1].axhline(1.0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("Growth of 1 unit (net)")
    axes[1, 1].set_title("f(price, macro) vs f(price), after costs")
    axes[1, 1].legend(fontsize=8, loc="upper right")
    if test:
        axes[1, 1].text(0.02, 0.04,
                        f"Sharpe difference {test['difference']:+.2f}  "
                        f"[{test['ci_lower_5pct']:+.2f}, {test['ci_upper_95pct']:+.2f}]\n"
                        f"paired bootstrap p = {test['p_value']:.2f}",
                        transform=axes[1, 1].transAxes, ha="left", va="bottom", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999", alpha=0.9))

    fig.suptitle("Figure 29. Does macro information improve forecasts, and a strategy?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Do macro features forecast asset-class returns beyond what price "
                           "features already do, how much would ignoring publication lags have "
                           "flattered the answer, and does it survive costs?", 29)


def forecasts_to_signal(forecast: pd.DataFrame, sleeves: dict[str, list[str]], columns: list[str],
                        index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map sleeve forecasts onto ETFs; the cash proxy and unmapped assets stay NaN."""
    signal = pd.DataFrame(np.nan, index=forecast.index, columns=columns)
    for sleeve, members in sleeves.items():
        for member in members:
            if member in signal.columns:
                signal[member] = forecast[sleeve]
    return signal.reindex(index).ffill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 15: macro features")
    parser.add_argument("--download", action="store_true", help="refetch macro data")
    parser.add_argument("--force", action="store_true", help="overwrite immutable raw macro files")
    args = parser.parse_args(argv)

    context, logger = build_context(STAGE, generation=2)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 15 | macro features (Generation 2, Priority 5)")
    logger.info("=" * 72)

    specs = load_specs(cfg)
    raw_dir = cfg.root / "data" / "raw" / "macro"
    metadata_dir = cfg.path("metadata")
    downloader = MacroDownloader(raw_dir, metadata_dir)
    if args.download or args.force or not all((raw_dir / f"{s.id}.csv").exists() for s in specs):
        manifest = downloader.download_all(specs, force=args.force)
        failed = [k for k, v in manifest["series"].items() if v["status"] == "failed"]
        if failed:
            raise RuntimeError(f"macro download failed for: {failed}")
    raw = load_macro_raw(raw_dir, specs)
    version = macro_data_version(metadata_dir)
    logger.info("macro data version %s", version)

    market = context.market_data()
    returns = market.returns()
    index = pd.DatetimeIndex(market.index)

    coverage = pd.DataFrame(
        {s.id: {"source": s.source, "frequency": s.frequency, "release_lag_days": s.release_lag_days,
                "first": str(raw[s.id].index.min().date()), "last": str(raw[s.id].index.max().date()),
                "observations": int(len(raw[s.id]))} for s in specs}).T
    context.save_table(coverage, "stage15_macro_coverage.csv")
    logger.info("macro series:\n%s", coverage.to_string())

    panel = macro_feature_panel(raw, specs, index, cfg)
    z, levels = panel["z"], panel["levels"]
    stale = staleness(raw, specs, index)
    context.save_table(stale.describe().T, "stage15_staleness.csv")
    context.save_table(z.describe().T, "stage15_feature_summary.csv")
    logger.info("feature availability: first usable date per feature:\n%s",
                z.apply(lambda c: str(c.first_valid_index().date()) if c.first_valid_index() is not None else "never").to_string())
    dimension = feature_dimension(z)
    spectrum = pd.DataFrame({"eigenvalue": dimension["eigenvalues"],
                             "cumulative_share": np.cumsum(dimension["eigenvalues"]) / dimension["eigenvalues"].sum()},
                            index=np.arange(1, dimension["n_features"] + 1))
    spectrum.index.name = "component"
    context.save_table(spectrum, "stage15_feature_spectrum.csv")
    logger.info("feature dimensionality: %d features, %.1f effective independent dimensions "
                "(participation ratio), %d components for 80%% of variance, %d month-ends",
                dimension["n_features"], dimension["effective_dimensions"],
                dimension["components_for_80pct"], dimension["n_months"])
    figure_panel(context, z, levels, raw, specs, stale, dimension, context.figure("fig28_macro_panel.png"))

    # ------------------------------------------- the same panel, lags ignored
    naive_specs = [MacroSeriesSpec(s.id, s.source, s.frequency, 0, s.symbol, s.feature) for s in specs]
    naive_z = macro_feature_panel(raw, naive_specs, index, cfg)["z"]

    # ------------------------------------------------------- predictability
    pred = cfg.get("macro.predictability", {}) or {}
    sleeves = {k: list(v) for k, v in (pred.get("sleeves", {}) or {}).items()}
    cash = str(pred.get("cash", "SHY"))
    sleeve_daily = sleeve_returns(returns, sleeves, market.investable)
    monthly = build_monthly_panel(sleeve_daily, returns[cash], z)
    kwargs = dict(min_train_months=int(pred.get("min_train_months", 60)),
                  alphas=list(pred.get("ridge_alphas", [1, 10, 100, 1000])),
                  cv_splits=int(pred.get("cv_splits", 5)),
                  first_test_year=int(pred.get("first_test_year", 2010)))
    fdr = float(pred.get("fdr", 0.10))

    logger.info("walk-forward forecasts (publication lags respected)")
    result = walk_forward_forecasts(monthly, **kwargs)
    table = evaluate_nested(result, COMPARISONS, fdr)
    logger.info("walk-forward forecasts (publication lags IGNORED: the control)")
    naive_result = walk_forward_forecasts(monthly, macro_override=naive_z, **kwargs)
    naive_table = evaluate_nested(naive_result, COMPARISONS, fdr)

    context.save_table(table, "stage15_clark_west.csv", index=False)
    context.save_table(naive_table, "stage15_clark_west_lags_ignored.csv", index=False)
    cols = ["sleeve", "comparison", "oos_r2", "statistic", "p_value", "bh_significant", "n_obs"]
    logger.info("Clark-West tests, lags respected:\n%s", table[cols].round(4).to_string(index=False))
    logger.info("Clark-West tests, lags IGNORED:\n%s", naive_table[cols].round(4).to_string(index=False))

    n_obs_by_comparison = table.groupby("comparison")["n_obs"].agg(["min", "max"])
    n_oos = int(n_obs_by_comparison.loc["both_vs_price", "min"])
    logger.info("out-of-sample months per comparison:\n%s", n_obs_by_comparison.to_string())

    both_vs_price = table[table["comparison"] == "both_vs_price"]
    macro_vs_hist = table[table["comparison"] == "macro_vs_hist"]
    naive_macro_vs_hist = naive_table[naive_table["comparison"] == "macro_vs_hist"]
    adds_information = bool(both_vs_price["bh_significant"].any() and both_vs_price["oos_r2"].mean() > 0)
    inflation = float(naive_macro_vs_hist["oos_r2"].mean() - macro_vs_hist["oos_r2"].mean())
    flips = int(((naive_macro_vs_hist.set_index("sleeve")["p_value"] < 0.05)
                 & ~(macro_vs_hist.set_index("sleeve")["p_value"] < 0.05)).sum())

    # Post-hoc, labelled as such: the same comparison for the question that matters
    # for the decision (price+macro vs price), and a direct test of whether ignoring
    # lags makes the same model forecast better. Neither feeds a decision.
    naive_both_vs_price = naive_table[naive_table["comparison"] == "both_vs_price"]
    inflation_both = float(naive_both_vs_price["oos_r2"].mean() - both_vs_price["oos_r2"].mean())
    flips_both = int(((naive_both_vs_price.set_index("sleeve")["bh_significant"].astype(bool))
                      & ~(both_vs_price.set_index("sleeve")["bh_significant"].astype(bool))).sum())
    inflation_tests = lag_inflation_tests(result, naive_result, fdr=fdr)
    context.save_table(inflation_tests, "stage15_lag_inflation_tests.csv", index=False)
    logger.info("does ignoring lags make the same model forecast better? (DM, one-sided, post-hoc)\n%s",
                inflation_tests.round(5).to_string(index=False))
    logger.info("R-squared inflation from ignoring lags: macro-vs-history %+.4f (pre-declared), "
                "price+macro-vs-price %+.4f (post-hoc)", inflation, inflation_both)

    # -------------------------------------------------------------- strategy
    engine = BacktestEngine.from_config(cfg)
    transform = transform_config(cfg)
    volatility = rolling_volatility(returns, int(cfg.get("portfolio.volatility.lookback", 63)))
    first_forecast = result["forecasts"]["both"].dropna(how="all").index.min()

    streams, strategy_results = {}, {}
    for label, model in (("price only", "price"), ("price + macro", "both")):
        forecast = result["forecasts"][model]
        signal = forecasts_to_signal(forecast, sleeves, list(returns.columns), index)
        weights = signal_to_positions(signal.where(market.investable), volatility,
                                      investable=market.investable, **transform)
        run = engine.run(weights, returns, f"macro_{model}", market.investable, apply_vol_target=True)
        strategy_results[label] = run
        streams[label] = slice_dates(run.net_returns, first_forecast, None)
        logger.info("%-14s net Sharpe %+.3f gross %+.3f turnover %.1fx", label,
                    run.summary()["sharpe"], run.summary()["gross_sharpe"], run.summary()["ann_turnover"])

    test = paired_sharpe_test(streams["price + macro"], streams["price only"],
                              n_samples=2000, block_length=21, seed=7)
    logger.info("paired Sharpe test, (price+macro) minus (price only): %s",
                {k: round(v, 4) if isinstance(v, float) else v for k, v in test.items()})
    from src.backtest.metrics import performance_summary

    strat_table = pd.DataFrame({k: performance_summary(v) for k, v in streams.items()}).T
    context.save_table(strat_table, "stage15_macro_strategy.csv")
    context.save_table(pd.Series(test, name="value").to_frame(), "stage15_macro_strategy_paired_test.csv")
    curves = {k: (1.0 + v).cumprod() for k, v in streams.items()}
    figure_predictability(context, table, naive_table, curves, test,
                          context.figure("fig29_macro_predictability.png"))

    # -------------------------------------------------------------- registry
    n_min, n_max = int(n_obs_by_comparison["min"].min()), int(n_obs_by_comparison["max"].max())
    context.registry.log(
        "Macro features improve out-of-sample forecasts of monthly asset-class excess returns "
        "beyond what price features already provide.",
        stage=STAGE,
        parameters={"sleeves": list(sleeves), "n_macro_features": len(FEATURE_COLUMNS),
                    "min_train_months": kwargs["min_train_months"], "fdr": fdr,
                    "macro_data_version": version},
        train_period="expanding window from the start of data",
        test_period=f"{first_forecast.date()} onward, one-month-ahead",
        results={
            "oos_months_price_macro_vs_price": n_oos,
            "oos_months_range_over_comparisons": [n_min, n_max],
            "mean_oos_r2_both_vs_price": float(both_vs_price["oos_r2"].mean()),
            "sleeves_bh_significant_both_vs_price": int(both_vs_price["bh_significant"].sum()),
            "sleeves_raw_significant_both_vs_price": int(both_vs_price["significant_raw_5pct"].sum()),
            "mean_oos_r2_macro_vs_hist": float(macro_vs_hist["oos_r2"].mean()),
            "mean_oos_r2_price_vs_hist": float(table[table["comparison"] == "price_vs_hist"]["oos_r2"].mean()),
            "min_p_value_both_vs_price": float(both_vs_price["p_value"].min()),
            "effective_independent_feature_dimensions": dimension["effective_dimensions"],
        },
        decision="retain" if adds_information else "reject",
        notes=(
            "Rule fixed in advance: retain if at least one sleeve passes Clark-West at FDR 10% "
            "for price+macro vs price AND the mean OOS R-squared of that comparison is positive. "
            f"Only {n_oos} monthly out-of-sample observations per sleeve, and the "
            f"{dimension['n_features']} macro features carry about "
            f"{dimension['effective_dimensions']:.1f} independent dimensions, so there is little "
            "independent information to find and little power to find it. A null here is weak "
            "evidence of absence, not strong evidence of it. Nested comparisons use Clark-West "
            "because a plain Diebold-Mariano test is mechanically undersized on nested models. "
            "The test sample starts in 2011 because the standardised features, 60 training months "
            "and 12-month momentum all have to exist first."
        ),
    )
    context.registry.log(
        "Ignoring publication lags inflates the apparent out-of-sample skill of macro features.",
        stage=STAGE,
        parameters={"lags_respected": True, "lags_ignored": True},
        results={
            "mean_oos_r2_macro_vs_hist_lags_respected": float(macro_vs_hist["oos_r2"].mean()),
            "mean_oos_r2_macro_vs_hist_lags_ignored": float(naive_macro_vs_hist["oos_r2"].mean()),
            "r2_inflation_macro_vs_hist": inflation,
            "sleeves_where_significance_appears_only_without_lags": flips,
            "posthoc_r2_inflation_both_vs_price": inflation_both,
            "posthoc_sleeves_bh_significant_only_without_lags_both_vs_price": flips_both,
            "posthoc_dm_min_p_value_ignored_better": (float(inflation_tests["p_value"].min())
                                                      if not inflation_tests.empty else None),
            "posthoc_dm_pairs_bh_significant": (int(inflation_tests["bh_significant"].sum())
                                                if not inflation_tests.empty else None),
            "posthoc_dm_pairs_tested": int(len(inflation_tests)),
        },
        decision="record",
        notes=(
            "A methodology control, not an alpha hypothesis. This entry was first drafted with the "
            "literal rule 'retain if R-squared inflation > 0'; the measured inflation of "
            f"{100 * inflation:+.2f} percentage points of R-squared on the pre-declared comparison "
            "shows that rule could not tell an effect from noise, and the control was never meant to "
            "gate a decision, so it is recorded rather than retained or rejected (the original "
            "wording is in the git history). The post-hoc numbers ask the question that matters for "
            f"the decision: on price+macro vs price the inflation is {100 * inflation_both:+.2f} "
            "percentage points, and the direct Diebold-Mariano tests of whether ignoring lags makes "
            "the same model forecast better find "
            f"{int(inflation_tests['bh_significant'].sum()) if not inflation_tests.empty else 0} "
            f"of {len(inflation_tests)} sleeve-model pairs significant after BH control. Reading: "
            "ignoring publication lags flatters the numbers a little but does not manufacture "
            "significance here, because the lags are two to six weeks and the features barely "
            "forecast monthly returns even with the extra look; there is little to inflate. The "
            "availability machinery is kept because it is correct by construction, not because "
            "this sample demanded it."
        ),
    )
    strategy_better = bool(test and test["difference"] > 0 and test["p_value"] < 0.05)
    context.registry.log(
        "A signal built from price and macro, f(price, macro), beats the price-only signal "
        "after transaction costs.",
        stage=STAGE,
        parameters={"models": ["price", "both"], "rebalance": engine.rebalance,
                    "oos_start": first_forecast.date().isoformat()},
        cost_bps=float(cfg.get("backtest.costs.cost_bps", 10.0)),
        results={
            "sharpe_price_only": float(strat_table.loc["price only", "sharpe"]),
            "sharpe_price_macro": float(strat_table.loc["price + macro", "sharpe"]),
            "sharpe_difference": float(test.get("difference", np.nan)),
            "paired_p_value": float(test.get("p_value", np.nan)),
            "return_correlation": float(test.get("return_correlation", np.nan)),
        },
        decision="retain" if strategy_better else "reject",
        notes=("Judged by the paired bootstrap on the Sharpe difference; a higher point "
               "estimate without significance is not an improvement."),
    )
    logger.info("STAGE 15 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
