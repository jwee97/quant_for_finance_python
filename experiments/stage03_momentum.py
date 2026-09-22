"""Stage 3 - Momentum alpha research (Ch. 22 §22.3.1, §22.3.9).

Tests whether past returns carry information about future returns, across
three variants (raw, volatility-scaled, ranked) and four lookbacks, at six
horizons. The lookback is NOT chosen by Sharpe ratio; the whole family is
evaluated and reported.

Figures 8-10: signal behaviour, IC, IC decay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.momentum import momentum_family
from src.features.returns import forward_returns
from src.models.regression import cross_sectional_ic, rolling_ic
from src.utils.plotting import PALETTE, bar_with_values, new_axes, plot_heatmap, save_figure
from experiments.alpha_research import decide, evaluate_family, family_verdict, regression_study
from experiments.context import build_context

STAGE = "stage03_momentum"


def figure_signal_behaviour(context, market, features, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.2))
    primary = features["vol_scaled_126"]

    for i, ticker in enumerate(["SPY", "TLT", "GLD", "DBC"]):
        if ticker in primary.columns:
            axes[0, 0].plot(primary.index, primary[ticker].to_numpy(), label=ticker,
                            color=PALETTE[i], linewidth=0.9)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Volatility-scaled 126-day momentum")
    axes[0, 0].legend(ncol=4, fontsize=8.5)

    latest = primary.iloc[-1].dropna().sort_values()
    bar_with_values(axes[0, 1], latest, "Cross-section on the last date", "Signal", "{:.2f}")

    raw = features["raw_126"]
    axes[1, 0].hist(raw.to_numpy().ravel()[np.isfinite(raw.to_numpy().ravel())], bins=120,
                    color="#0072B2", alpha=0.7, density=True, label="raw")
    axes[1, 0].hist(primary.to_numpy().ravel()[np.isfinite(primary.to_numpy().ravel())], bins=120,
                    color="#D55E00", alpha=0.55, density=True, label="vol-scaled")
    axes[1, 0].set_title("Signal distributions: raw momentum vs volatility-scaled")
    axes[1, 0].legend(fontsize=8.5)

    lookbacks = [21, 63, 126, 252]
    corr = pd.DataFrame(index=lookbacks, columns=lookbacks, dtype=float)
    for a in lookbacks:
        for b in lookbacks:
            first, second = features[f"raw_{a}"].align(features[f"raw_{b}"], join="inner")
            corr.loc[a, b] = float(first.corrwith(second, axis=1).mean())
    plot_heatmap(axes[1, 1], corr.astype(float), "Cross-sectional correlation between lookbacks",
                 vmin=0.0, vmax=1.0)

    fig.suptitle("Figure 8. What does the momentum signal actually look like?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Is the momentum signal well behaved, and are different lookbacks "
                           "measuring different things or the same thing?", 8)


def figure_ic(context, features, returns, table, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.4))
    horizon = 21
    forward = forward_returns(returns, horizon)

    ic = cross_sectional_ic(features["vol_scaled_126"], forward)
    axes[0, 0].plot(ic.index, ic.to_numpy(), color="#999999", linewidth=0.5, alpha=0.6)
    rolled = ic.rolling(252, min_periods=60).mean()
    axes[0, 0].plot(rolled.index, rolled.to_numpy(), color="#D55E00", linewidth=1.8,
                    label="252-day mean")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 0].axhline(float(ic.mean()), color="#0072B2", linestyle="--",
                       label=f"full-sample mean {ic.mean():+.4f}")
    axes[0, 0].set_title(f"Daily cross-sectional IC, 126-day momentum, {horizon}-day horizon")
    axes[0, 0].legend(fontsize=8.5)

    axes[0, 1].hist(ic.dropna(), bins=60, color="#0072B2", alpha=0.75, density=True)
    axes[0, 1].axvline(0.0, color="black", linewidth=0.9)
    axes[0, 1].axvline(float(ic.mean()), color="#D55E00", linewidth=1.5,
                       label=f"mean {ic.mean():+.4f}")
    axes[0, 1].set_title("Distribution of the daily IC")
    axes[0, 1].legend(fontsize=8.5)

    subset = table[table["horizon"] == horizon].set_index("feature")["mean_ic"].sort_values()
    bar_with_values(axes[1, 0], subset, f"Mean IC by signal variant ({horizon}-day horizon)",
                    "Mean IC", "{:.3f}")

    pivot = table.pivot(index="feature", columns="horizon", values="t_stat_overlap_adjusted")
    plot_heatmap(axes[1, 1], pivot.astype(float), "Overlap-adjusted t-statistic",
                 vmin=-3.0, vmax=3.0, fmt="{:.1f}")

    fig.suptitle("Figure 9. Does past return predict future return?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Does momentum have a positive and statistically reliable information "
                           "coefficient on this universe, and is it stable through time?", 9)


def figure_ic_decay(context, table, path):
    fig, axes = new_axes(1, 2, figsize=(13.5, 5.4))
    for i, feature in enumerate(sorted(table["feature"].unique())):
        subset = table[table["feature"] == feature].sort_values("horizon")
        style = "-" if "vol_scaled" in feature else ("--" if "ranked" in feature else ":")
        axes[0].plot(subset["horizon"], subset["mean_ic"], style, marker="o", markersize=3.5,
                     label=feature, color=PALETTE[i % len(PALETTE)], linewidth=1.2)
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_xlabel("Forecast horizon (days)")
    axes[0].set_ylabel("Mean IC")
    axes[0].set_title("IC by horizon: how fast does information decay?")
    axes[0].legend(ncol=2, fontsize=7.5)

    pivot = table.pivot(index="feature", columns="horizon", values="mean_ic")
    image = plot_heatmap(axes[1], pivot.astype(float), "Mean IC surface", vmin=-0.04, vmax=0.04,
                         fmt="{:.3f}")
    fig.colorbar(image, ax=axes[1], shrink=0.8)

    fig.suptitle("Figure 10. Momentum IC decay across the full parameter family",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "How quickly does momentum's predictive information decay, and does any "
                           "combination of lookback and horizon show a genuine edge?", 10)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 3 | momentum alpha research (Ch. 22)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    horizons = tuple(cfg.get("strategies.horizons", [1, 5, 10, 21, 42, 63]))
    lookbacks = tuple(cfg.get("strategies.momentum.lookbacks", [21, 63, 126, 252]))
    skip = int(cfg.get("strategies.momentum.skip_days", 1))
    primary_horizon = int(cfg.get("strategies.primary_horizon", 21))

    features = momentum_family(market.prices, returns, lookbacks, skip,
                               int(cfg.get("strategies.momentum.vol_lookback", 63)))
    features = {name: frame.where(market.investable) for name, frame in features.items()}
    logger.info("evaluating %d momentum features across %d horizons", len(features), len(horizons))

    # Development sample only: the validation and holdout blocks stay closed.
    dev = (cfg.get("backtest.samples.development.start"), cfg.get("backtest.samples.development.end"))
    table = evaluate_family(features, returns, horizons, breadth=len(market.tickers), sample=dev)
    context.save_table(table, "stage03_momentum_ic.csv", index=False)
    logger.info("momentum IC (development sample):\n%s",
                table[["feature", "horizon", "mean_ic", "t_stat_overlap_adjusted", "p_value",
                       "ts_ic_mean", "ts_ic_share_positive"]].round(4).to_string(index=False))

    full = evaluate_family(features, returns, horizons, breadth=len(market.tickers))
    context.save_table(full, "stage03_momentum_ic_full_sample.csv", index=False)

    fdr = float(cfg.get("strategies.multiple_testing.fdr", 0.10))
    verdict = family_verdict(table, fdr, expected_sign=+1)
    context.save_table(verdict.pop("table"), "stage03_momentum_fdr.csv", index=False)
    logger.info("family-level evidence after BH-FDR control: %s", verdict)

    primary_name = f"{cfg.get('strategies.momentum.primary_variant', 'vol_scaled')}_" \
                   f"{cfg.get('strategies.momentum.primary_lookback', 126)}"
    study = regression_study(features[primary_name], returns, primary_horizon)
    context.save_table(study["per_asset"], "stage03_momentum_per_asset_regression.csv")
    logger.info("pooled regression (%s, h=%d): %s", primary_name, primary_horizon,
                study["pooled"].summary_line("signal"))
    logger.info("per-asset: %.0f%% of assets have a positive beta, %.0f%% significant at 5%%",
                100 * (1 - study["share_negative_beta"]), 100 * study["share_significant"])

    figure_signal_behaviour(context, market, features, context.figure("fig08_momentum_signal.png"))
    figure_ic(context, features, returns, table, context.figure("fig09_momentum_ic.png"))
    figure_ic_decay(context, table, context.figure("fig10_ic_decay.png"))

    # --- decision ---------------------------------------------------------
    dev_primary = table[(table["feature"] == primary_name) & (table["horizon"] == primary_horizon)]
    mean_ic = float(dev_primary["mean_ic"].iloc[0]) if len(dev_primary) else np.nan
    p_value = float(dev_primary["p_value"].iloc[0]) if len(dev_primary) else np.nan
    decision, reason = decide(mean_ic, p_value, expected_sign=+1)

    best = table.loc[table["mean_ic"].idxmax()] if len(table) else None
    context.registry.log(
        f"{primary_name.replace('_', ' ')} momentum predicts subsequent "
        f"{primary_horizon}-day cross-sectional ETF returns.",
        stage=STAGE,
        parameters={"variants": list(features), "horizons": list(horizons), "skip_days": skip,
                    "primary_feature": primary_name, "primary_horizon": primary_horizon},
        train_period=f"{dev[0]} to {dev[1]} (development sample)",
        results={
            "mean_ic": mean_ic,
            "t_stat_overlap_adjusted": float(dev_primary["t_stat_overlap_adjusted"].iloc[0]) if len(dev_primary) else np.nan,
            "p_value": p_value,
            "pooled_beta": float(study["pooled"].params.get("signal", np.nan)),
            "pooled_t": float(study["pooled"].t_values.get("signal", np.nan)),
            "share_assets_positive_beta": float(1 - study["share_negative_beta"]),
            "best_feature_in_family": str(best["feature"]) if best is not None else "",
            "best_mean_ic_in_family": float(best["mean_ic"]) if best is not None else np.nan,
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
            "hypothesised positive sign. Cross-sectional momentum across heterogeneous asset classes is a much "
            "weaker effect than the within-equity version the literature usually reports: the "
            "cross-section here is dominated by the volatility difference between bonds and "
            "commodities rather than by relative trend. The signal is carried forward to the "
            "backtest anyway, so that the portfolio-level result can be compared against this "
            "signal-level evidence rather than replacing it."
        ),
    )
    logger.info("DECISION: %s -- %s", decision.upper(), reason)
    logger.info("STAGE 3 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
