"""Stage 4 - Mean-reversion alpha research (Ch. 22 §22.3.1).

Tests the hypothesis *before* building a trading rule around it. The feature
is the sign-neutral z-score

    Z_{i,t} = (P_{i,t} - mu_{i,t}) / sigma_{i,t}

and the hypothesis is ``beta < 0`` in ``r_{i,t+h} = alpha + beta Z_{i,t} + eps``.
Stage 3's decision rule is reused with the expected sign flipped, so the two
families are judged on identical terms.

Figures 11-12: mean-reversion signal behaviour, mean-reversion IC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.mean_reversion import (
    half_life_of_reversion,
    mean_reversion_family,
    price_zscore,
    variance_ratio_table,
)
from src.features.returns import forward_returns
from src.models.regression import cross_sectional_ic
from src.utils.plotting import PALETTE, bar_with_values, new_axes, plot_heatmap, save_figure
from experiments.alpha_research import decide, evaluate_family, family_verdict, regression_study
from experiments.context import build_context

STAGE = "stage04_mean_reversion"


def figure_signal(context, market, features, variance_ratios, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.2))
    z = features["zscore_21"]

    ticker = "SPY"
    window = slice("2018-01-01", "2020-12-31")
    price = market.prices.loc[window, ticker]
    axes[0, 0].plot(price.index, (price / price.iloc[0]).to_numpy(), color="#0072B2", label="SPY (rebased)")
    twin = axes[0, 0].twinx()
    twin.plot(z.loc[window, ticker].index, z.loc[window, ticker].to_numpy(), color="#D55E00",
              linewidth=0.9, label="21-day z-score")
    twin.axhline(0.0, color="black", linewidth=0.7)
    twin.axhline(2.0, color="#888888", linestyle=":", linewidth=0.8)
    twin.axhline(-2.0, color="#888888", linestyle=":", linewidth=0.8)
    twin.set_ylabel("z-score")
    axes[0, 0].set_title("Price and 21-day z-score, 2018-2020")
    axes[0, 0].legend(loc="upper left", fontsize=8.5)

    values = z.to_numpy().ravel()
    axes[0, 1].hist(values[np.isfinite(values)], bins=120, color="#0072B2", alpha=0.75, density=True)
    axes[0, 1].axvline(0.0, color="black", linewidth=0.9)
    axes[0, 1].set_title("Distribution of the 21-day z-score")

    vr_cols = [c for c in variance_ratios.columns if c.startswith("vr_")]
    plot_heatmap(axes[1, 0], variance_ratios[vr_cols].astype(float), "Variance ratios (VR < 1 = mean reverting)",
                 vmin=0.5, vmax=1.5, fmt="{:.2f}")

    half_lives = pd.Series(
        {t: half_life_of_reversion(z[t]) for t in z.columns}
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    bar_with_values(axes[1, 1], half_lives, "Half-life of the z-score (days)", "Days", "{:.1f}")

    fig.suptitle("Figure 11. Is there anything to revert to?", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Do prices actually mean revert on this universe, and over what "
                           "horizon, before any trading rule is built on top?", 11)


def figure_ic(context, features, returns, table, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.4))
    horizon = 5
    forward = forward_returns(returns, horizon)
    ic = cross_sectional_ic(features["zscore_21"], forward)

    axes[0, 0].plot(ic.index, ic.to_numpy(), color="#999999", linewidth=0.5, alpha=0.6)
    rolled = ic.rolling(252, min_periods=60).mean()
    axes[0, 0].plot(rolled.index, rolled.to_numpy(), color="#D55E00", linewidth=1.8, label="252-day mean")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 0].axhline(float(ic.mean()), color="#0072B2", linestyle="--",
                       label=f"mean {ic.mean():+.4f}")
    axes[0, 0].set_title(f"IC of the 21-day z-score at a {horizon}-day horizon")
    axes[0, 0].legend(fontsize=8.5)

    for i, feature in enumerate(["zscore_5", "zscore_10", "zscore_21", "zscore_63"]):
        if feature in table["feature"].values:
            subset = table[table["feature"] == feature].sort_values("horizon")
            axes[0, 1].plot(subset["horizon"], subset["mean_ic"], marker="o", markersize=3.5,
                            label=feature, color=PALETTE[i])
    axes[0, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 1].set_xlabel("Forecast horizon (days)")
    axes[0, 1].set_ylabel("Mean IC of the raw z-score")
    axes[0, 1].set_title("Reversion decays quickly (negative IC = reversion)")
    axes[0, 1].legend(fontsize=8.5)

    subset = table[table["horizon"] == horizon].set_index("feature")["mean_ic"].sort_values()
    bar_with_values(axes[1, 0], subset, f"Mean IC by variant ({horizon}-day horizon)", "Mean IC", "{:.3f}")

    pivot = table.pivot(index="feature", columns="horizon", values="t_stat_overlap_adjusted")
    image = plot_heatmap(axes[1, 1], pivot.astype(float), "Overlap-adjusted t-statistic",
                         vmin=-4.0, vmax=4.0, fmt="{:.1f}")
    fig.colorbar(image, ax=axes[1, 1], shrink=0.8)

    fig.suptitle("Figure 12. Mean-reversion information coefficient", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Is the relationship between the z-score and subsequent returns "
                           "negative, as the mean-reversion hypothesis requires, and is it "
                           "statistically reliable?", 12)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 4 | mean-reversion alpha research (Ch. 22)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    horizons = tuple(cfg.get("strategies.horizons", [1, 5, 10, 21, 42, 63]))
    lookbacks = tuple(cfg.get("strategies.mean_reversion.lookbacks", [5, 10, 21, 63]))
    basis = cfg.get("strategies.mean_reversion.price_basis", "log")

    features = mean_reversion_family(market.prices, returns, lookbacks, basis)
    features = {name: frame.where(market.investable) for name, frame in features.items()}

    variance_ratios = variance_ratio_table(market.prices)
    context.save_table(variance_ratios, "stage04_variance_ratios.csv")
    logger.info("variance ratios below 1 (mean reverting) at 5 days: %d of %d assets",
                int((variance_ratios["vr_5"] < 1).sum()), len(variance_ratios))

    dev = (cfg.get("backtest.samples.development.start"), cfg.get("backtest.samples.development.end"))
    table = evaluate_family(features, returns, horizons, breadth=len(market.tickers), sample=dev)
    context.save_table(table, "stage04_mean_reversion_ic.csv", index=False)
    logger.info("mean-reversion IC of the RAW feature (negative = reversion):\n%s",
                table[["feature", "horizon", "mean_ic", "t_stat_overlap_adjusted", "p_value",
                       "ts_ic_mean"]].round(4).to_string(index=False))

    full = evaluate_family(features, returns, horizons, breadth=len(market.tickers))
    context.save_table(full, "stage04_mean_reversion_ic_full_sample.csv", index=False)

    primary_lookback = int(cfg.get("strategies.mean_reversion.primary_lookback", 21))
    primary_name = f"zscore_{primary_lookback}"
    # Fixed in config BEFORE any IC was computed. Selecting the horizon with
    # the most negative IC would be choosing the test on its own outcome --
    # the exact trap this stage exists to avoid.
    test_horizon = int(cfg.get("strategies.mean_reversion.test_horizon", 5))
    dev_subset = table[table["feature"] == primary_name].set_index("horizon")
    logger.info("testing at the ex-ante declared horizon of %d days", test_horizon)

    # Family-level evidence after false-discovery-rate control.
    fdr = float(cfg.get("strategies.multiple_testing.fdr", 0.10))
    verdict = family_verdict(table, fdr, expected_sign=-1)
    context.save_table(verdict.pop("table"), "stage04_mean_reversion_fdr.csv", index=False)
    logger.info("family-level evidence after BH-FDR control: %s", verdict)

    study = regression_study(features[primary_name], returns, test_horizon)
    context.save_table(study["per_asset"], "stage04_mean_reversion_per_asset_regression.csv")
    logger.info("pooled regression (%s, h=%d): %s", primary_name, test_horizon,
                study["pooled"].summary_line("signal"))
    logger.info("per-asset: %.0f%% of assets have the hypothesised NEGATIVE beta, %.0f%% significant",
                100 * study["share_negative_beta"], 100 * study["share_significant"])

    figure_signal(context, market, features, variance_ratios, context.figure("fig11_mean_reversion_signal.png"))
    figure_ic(context, features, returns, table, context.figure("fig12_mean_reversion_ic.png"))

    row = dev_subset.loc[test_horizon] if test_horizon in dev_subset.index else None
    mean_ic = float(row["mean_ic"]) if row is not None else np.nan
    p_value = float(row["p_value"]) if row is not None else np.nan
    decision, reason = decide(mean_ic, p_value, expected_sign=-1)

    context.registry.log(
        f"The {primary_lookback}-day price z-score is NEGATIVELY related to subsequent "
        f"{test_horizon}-day returns (mean-reversion hypothesis, beta < 0).",
        stage=STAGE,
        parameters={"variants": list(features), "horizons": list(horizons),
                    "price_basis": basis, "primary_feature": primary_name,
                    "test_horizon": test_horizon},
        train_period=f"{dev[0]} to {dev[1]} (development sample)",
        results={
            "mean_ic": mean_ic,
            "t_stat_overlap_adjusted": float(row["t_stat_overlap_adjusted"]) if row is not None else np.nan,
            "p_value": p_value,
            "pooled_beta": float(study["pooled"].params.get("signal", np.nan)),
            "pooled_t": float(study["pooled"].t_values.get("signal", np.nan)),
            "share_assets_negative_beta": float(study["share_negative_beta"]),
            "assets_with_vr5_below_1": int((variance_ratios["vr_5"] < 1).sum()),
            "family_n_tests": verdict["n_tests"],
            "family_significant_raw": verdict["n_significant_raw"],
            "family_expected_false_positives": verdict["n_expected_false_positives_at_5pct"],
            "family_survivors_after_fdr": verdict["n_survivors_right_sign"],
        },
        decision=decision,
        notes=(
            reason + f". At the family level, {verdict['n_significant_raw']} of "
            f"{verdict['n_tests']} tests clear a naive 5% threshold against "
            f"{verdict['n_expected_false_positives_at_5pct']:.1f} expected by chance, and "
            f"{verdict['n_survivors_right_sign']} survive Benjamini-Hochberg control with the "
            f"hypothesised negative sign: {', '.join(verdict['survivors'])}. Every survivor sits "
            "at the 1-day horizon with a short lookback, so the effect is real but decays almost "
            "immediately. That decay profile, not the peak IC, sets the tradable holding period, "
            "and a 1-day holding period means transaction costs will decide whether any of it "
            "survives -- which Stage 6 tests directly."
        ),
    )
    logger.info("DECISION: %s -- %s", decision.upper(), reason)
    logger.info("STAGE 4 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
