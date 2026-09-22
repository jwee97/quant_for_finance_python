"""Stage 5 - Expected returns, IC and signal validation (Ch. 20 §20.1).

Brings the two alpha families together and asks the questions that decide
what the backtest is allowed to trade:

1. How reliable is each signal's IC, once overlapping observations are
   accounted for? (§20.1.4, serial correlation)
2. How fast does the information decay, and what does that imply for the
   holding period? (§20.1.6-7)
3. What information ratio does the Fundamental Law imply, and does it survive
   contact with realistic breadth? (§20.1.8)
4. How should a signal be turned into an expected-return vector for the
   optimiser without importing enormous estimation error? (§20.1.1-3)

Figure 13: naive vs HAC inference and the implied information ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.momentum import total_return_momentum, volatility_scaled_momentum
from src.features.mean_reversion import price_zscore
from src.features.returns import forward_returns
from src.models.regression import (
    compare_standard_errors,
    cross_sectional_ic,
    historical_expected_returns,
    ic_decay,
    ic_summary,
    signal_to_expected_returns,
)
from src.signals.combine import combine_signals, signal_correlation
from src.utils.plotting import PALETTE, bar_with_values, new_axes, save_figure
from experiments.context import build_context

STAGE = "stage05_expected_returns"


def figure_inference(context, comparisons, decay_tables, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.4))

    frame = pd.DataFrame(comparisons).T
    x = np.arange(len(frame))
    axes[0, 0].bar(x - 0.2, frame["t_naive"].abs(), width=0.4, label="naive OLS", color="#D55E00")
    axes[0, 0].bar(x + 0.2, frame["t_hac"].abs(), width=0.4, label="Newey-West (HAC)", color="#0072B2")
    axes[0, 0].axhline(1.96, color="black", linestyle="--", linewidth=0.9, label="|t| = 1.96")
    axes[0, 0].set_xticks(x, frame.index, rotation=20, ha="right")
    axes[0, 0].set_ylabel("|t-statistic|")
    axes[0, 0].set_title("Overlapping returns inflate the naive t-statistic")
    axes[0, 0].legend(fontsize=8.5)

    bar_with_values(axes[0, 1], frame["se_inflation"], "Standard-error inflation factor (HAC / naive)",
                    "x", "{:.2f}", rotation=20)
    axes[0, 1].axhline(1.0, color="black", linewidth=0.9)

    for i, (name, table) in enumerate(decay_tables.items()):
        axes[1, 0].plot(table.index, table["mean_ic"], marker="o", markersize=4,
                        label=name, color=PALETTE[i])
    axes[1, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 0].set_xlabel("Forecast horizon (days)")
    axes[1, 0].set_ylabel("Mean IC")
    axes[1, 0].set_title("Signal decay: information is concentrated at short horizons")
    axes[1, 0].legend(fontsize=8.5)

    for i, (name, table) in enumerate(decay_tables.items()):
        if "implied_information_ratio" in table:
            axes[1, 1].plot(table.index, table["implied_information_ratio"], marker="s",
                            markersize=4, label=name, color=PALETTE[i])
    axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 1].set_xlabel("Forecast horizon (days)")
    axes[1, 1].set_ylabel("IR = IC x sqrt(breadth)")
    axes[1, 1].set_title("Fundamental Law: implied information ratio before costs")
    axes[1, 1].legend(fontsize=8.5)

    fig.suptitle("Figure 13. How much of the apparent signal survives honest inference?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Once overlapping observations are corrected for, how much statistical "
                           "evidence remains, and what information ratio does the Fundamental Law "
                           "imply at each horizon?", 13)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 5 | expected returns, IC and signal validation (Ch. 20)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    horizons = tuple(cfg.get("strategies.horizons", [1, 5, 10, 21, 42, 63]))
    breadth = len(market.tickers)

    signals = {
        "momentum_126_vol_scaled": volatility_scaled_momentum(
            market.prices, returns, 126, 1, cfg.get("strategies.momentum.vol_lookback", 63)
        ),
        "momentum_252_raw": total_return_momentum(market.prices, 252, 1),
        "reversion_5_zscore": -price_zscore(market.prices, 5),
        "reversion_21_zscore": -price_zscore(market.prices, 21),
    }
    signals = {name: frame.where(market.investable) for name, frame in signals.items()}

    # --- IC decay per signal (spec §18) ----------------------------------
    decay_tables = {}
    for name, signal in signals.items():
        table = ic_decay(signal, returns, horizons, breadth=breadth)
        decay_tables[name] = table
        context.save_table(table, f"stage05_ic_decay_{name}.csv")
    combined_decay = pd.concat(decay_tables, names=["signal", "horizon"])
    context.save_table(combined_decay, "stage05_ic_decay_all.csv")
    logger.info("IC decay (mean IC by horizon):\n%s",
                combined_decay["mean_ic"].unstack(level=0).round(4).to_string())
    logger.info("implied information ratio (Fundamental Law, pre-cost):\n%s",
                combined_decay["implied_information_ratio"].unstack(level=0).round(3).to_string())

    # --- naive vs HAC inference (spec §19) -------------------------------
    comparisons = {}
    for name, signal in signals.items():
        horizon = 1 if "reversion" in name else 21
        forward = forward_returns(returns, horizon)
        centred_signal = signal.sub(signal.mean(axis=1), axis=0)
        centred_forward = forward.sub(forward.mean(axis=1), axis=0)
        stacked = pd.concat(
            [centred_forward.stack().rename("y"), centred_signal.stack().rename("signal")], axis=1
        ).dropna()
        table = compare_standard_errors(stacked["y"], stacked[["signal"]], horizon)
        comparisons[f"{name}\n(h={horizon})"] = table.loc["signal"]
    comparison_frame = pd.DataFrame(comparisons).T
    context.save_table(comparison_frame, "stage05_standard_error_comparison.csv")
    logger.info("naive vs HAC standard errors:\n%s", comparison_frame.round(4).to_string())

    # --- signal correlation: is the second signal adding anything? -------
    correlation = signal_correlation(signals)
    context.save_table(correlation, "stage05_signal_correlation.csv")
    logger.info("average cross-sectional signal correlation:\n%s", correlation.round(3).to_string())

    combined = combine_signals(
        {"momentum": signals["momentum_252_raw"], "reversion": signals["reversion_5_zscore"]},
        {"momentum": 0.5, "reversion": 0.5},
    )
    combined_ic = {}
    for horizon in horizons:
        ic = cross_sectional_ic(combined, forward_returns(returns, horizon))
        combined_ic[horizon] = ic_summary(ic, horizon=horizon, breadth=breadth)
    combined_table = pd.DataFrame(combined_ic).T
    combined_table.index.name = "horizon"
    context.save_table(combined_table, "stage05_combined_signal_ic.csv")
    logger.info("combined-signal IC:\n%s",
                combined_table[["mean_ic", "t_stat_overlap_adjusted", "p_value",
                                "implied_information_ratio"]].round(4).to_string())

    # --- expected-return construction for the optimiser ------------------
    mu_signal = signal_to_expected_returns(
        signals["momentum_252_raw"],
        target_spread=float(cfg.get("portfolio.mean_variance.mu_scale", 0.05)),
        shrinkage=float(cfg.get("portfolio.mean_variance.mu_shrinkage", 0.5)),
    )
    mu_history = historical_expected_returns(returns, 252, 0.5)
    dispersion = pd.DataFrame(
        {
            "signal_implied_mu_ann": (mu_signal * 252).std(axis=1),
            "historical_mu_ann": (mu_history * 252).std(axis=1),
        }
    ).dropna()
    context.save_table(dispersion.describe(), "stage05_expected_return_dispersion.csv")
    logger.info("cross-sectional dispersion of annualised expected returns:\n%s",
                dispersion.describe().round(4).to_string())
    logger.info(
        "the signal-implied dispersion is constant by construction (%.3f = target_spread x "
        "(1 - shrinkage)): how far apart the optimiser is allowed to believe assets are is an "
        "explicit research assumption here, not an accident of signal scale. The historical "
        "estimator's dispersion instead wanders between %.3f and %.3f, and every one of those "
        "swings feeds straight into the weights -- which is the estimation-error problem "
        "quantified in Stage 7.",
        float(dispersion["signal_implied_mu_ann"].mean()),
        float(dispersion["historical_mu_ann"].min()), float(dispersion["historical_mu_ann"].max()),
    )

    figure_inference(context, comparisons, decay_tables, context.figure("fig13_signal_inference.png"))

    best_h = int(combined_decay["mean_ic"].abs().groupby(level="horizon").mean().idxmax())
    context.registry.log(
        "Correcting for overlapping observations materially changes the evidence for "
        "both signal families, and the Fundamental Law implies only a modest "
        "information ratio even before costs.",
        stage=STAGE,
        parameters={"signals": list(signals), "horizons": list(horizons), "breadth": breadth},
        results={
            "max_se_inflation": float(comparison_frame["se_inflation"].max()),
            "mean_se_inflation": float(comparison_frame["se_inflation"].mean()),
            "signals_significant_naive": int((comparison_frame["t_naive"].abs() > 1.96).sum()),
            "signals_significant_hac": int((comparison_frame["t_hac"].abs() > 1.96).sum()),
            "horizon_with_strongest_mean_ic": best_h,
            "momentum_reversion_signal_correlation": float(
                correlation.loc["momentum_252_raw", "reversion_5_zscore"]
            ),
            "best_implied_ir": float(combined_decay["implied_information_ratio"].abs().max()),
        },
        decision="investigate",
        notes=(
            "Naive standard errors understate uncertainty by up to "
            f"{comparison_frame['se_inflation'].max():.1f}x on overlapping forward returns. "
            "Both families' information is concentrated at 1-5 days, where the implied "
            "pre-cost information ratio is at its highest and the required turnover is also at "
            "its highest -- the central tension the backtest must resolve. Momentum and "
            "reversion signals are close to uncorrelated in the cross-section, which is the "
            "precondition for the strategy-level diversification tested in Stage 10."
        ),
    )
    logger.info("STAGE 5 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
