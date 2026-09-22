"""Stage 10 - Combining strategies (Ch. 22 §22.5, spec §38).

Two levels of diversification, measured separately:

    asset diversification     within one strategy
    strategy diversification  across return streams

The book's point is quantitative: a lower-Sharpe strategy can still improve
the total if its return stream is uncorrelated with the others. This stage
tests whether that holds here -- and it is a genuine test, because Stage 6
showed the alpha strategies are individually poor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, buy_and_hold
from src.backtest.metrics import comparison_table
from src.signals.combine import combine_strategy_returns, diversification_table
from src.utils.plotting import PALETTE, bar_with_values, new_axes, plot_heatmap, save_figure
from experiments.context import build_context
from experiments.strategies import cached_ladder

STAGE = "stage10_combination"
MEMBERS = ["M1_inverse_vol", "M2_risk_parity", "M3_momentum", "M4_mean_reversion", "M9_mean_cvar"]


def figure_combination(context, streams, combos, correlation, table, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.6))

    plot_heatmap(axes[0, 0], correlation, "Correlation between strategy return streams",
                 vmin=-1.0, vmax=1.0)

    for i, (name, series) in enumerate(combos.items()):
        curve = (1.0 + series.dropna()).cumprod()
        axes[0, 1].plot(curve.index, curve.to_numpy(), color=PALETTE[i], linewidth=1.4, label=name)
    for i, (name, series) in enumerate(streams.items()):
        curve = (1.0 + series.dropna()).cumprod()
        axes[0, 1].plot(curve.index, curve.to_numpy(), color="#BBBBBB", linewidth=0.8, alpha=0.8)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Growth of 1 unit (log)")
    axes[0, 1].set_title("Combined streams (coloured) vs components (grey)")
    axes[0, 1].legend(fontsize=8)

    sharpes = table["sharpe"].dropna().sort_values()
    bar_with_values(axes[1, 0], sharpes, "Sharpe ratio: components and combinations",
                    "Sharpe", "{:.2f}", rotation=35)

    if "avg_component_sharpe" in table.columns:
        gains = table[["sharpe", "avg_component_sharpe", "best_component_sharpe"]].dropna()
        x = np.arange(len(gains))
        for i, column in enumerate(gains.columns):
            axes[1, 1].bar(x + i * 0.27, gains[column].to_numpy(), width=0.27,
                           label=column.replace("_", " "), color=PALETTE[i])
        axes[1, 1].set_xticks(x + 0.27, gains.index, rotation=20, ha="right", fontsize=8)
        axes[1, 1].set_ylabel("Sharpe ratio")
        axes[1, 1].set_title("Did combining beat the average component?")
        axes[1, 1].legend(fontsize=8)

    fig.suptitle("Figure 21. Strategy-level diversification", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Does combining weakly performing but uncorrelated strategies "
                           "produce a better return stream than any of them alone?", 21)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 10 | combining quant strategies (Ch. 22 §22.5)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    engine = BacktestEngine.from_config(cfg)
    books = cached_ladder(market, cfg, context.processed, MEMBERS)
    results = {name: engine.run(weights, returns, name, market.investable,
                                apply_vol_target=name.startswith(("M3", "M4", "M5")))
               for name, weights in books.items()}
    streams = pd.DataFrame({name: r.net_returns for name, r in results.items()}).dropna(how="all")

    correlation = streams.corr()
    context.save_table(correlation, "stage10_strategy_correlation.csv")
    logger.info("strategy return-stream correlations:\n%s", correlation.round(3).to_string())

    combos = {}
    for method in ("equal", "inverse_vol"):
        combined, weights = combine_strategy_returns(
            streams, method, int(cfg.get("strategies.combination.vol_lookback", 252))
        )
        combos[f"combined_{method}"] = combined
        context.save_table(weights.dropna(how="all"), f"stage10_combination_weights_{method}.csv")

    alpha_only = streams[[c for c in streams.columns if c.startswith(("M3", "M4"))]]
    if alpha_only.shape[1] >= 2:
        combined_alpha, _ = combine_strategy_returns(alpha_only, "inverse_vol")
        combos["combined_alpha_only"] = combined_alpha

    table = diversification_table(streams, combos["combined_inverse_vol"])
    for name, series in combos.items():
        if name == "combined_inverse_vol":
            continue
        extra = diversification_table(streams, series).loc[["combined"]]
        extra.index = [name]
        table = pd.concat([table, extra])
    table = table.rename(index={"combined": "combined_inverse_vol"})
    context.save_table(table, "stage10_diversification.csv")
    logger.info("diversification table:\n%s", table.round(4).to_string())

    full = comparison_table({**{n: s for n, s in streams.items()}, **combos},
                            benchmark=buy_and_hold(returns, "SPY").net_returns)
    context.save_table(full, "stage10_combination_performance.csv")
    logger.info("performance of components and combinations:\n%s",
                full[["cagr", "ann_vol", "sharpe", "max_drawdown"]].astype(float).round(4).to_string())

    figure_combination(context, {n: streams[n] for n in streams.columns}, combos, correlation,
                       table, context.figure("fig21_strategy_combination.png"))

    best_component = float(table.loc[list(streams.columns), "sharpe"].max())
    best_combo_name = max(combos, key=lambda k: float(table.loc[k, "sharpe"]))
    best_combo = float(table.loc[best_combo_name, "sharpe"])
    mean_off_diagonal = float(
        correlation.to_numpy()[~np.eye(len(correlation), dtype=bool)].mean()
    )

    context.registry.log(
        "Combining strategy return streams improves risk-adjusted performance beyond "
        "the best individual strategy (Ch. 22 §22.5).",
        stage=STAGE,
        parameters={"members": list(streams.columns), "methods": ["equal", "inverse_vol"]},
        results={
            "mean_stream_correlation": mean_off_diagonal,
            "best_component_sharpe": best_component,
            "best_combination": best_combo_name,
            "best_combination_sharpe": best_combo,
            "combination_minus_best_component": best_combo - best_component,
            "combination_minus_average_component": best_combo - float(
                table.loc[list(streams.columns), "sharpe"].mean()
            ),
        },
        decision="retain" if best_combo > best_component else "reject",
        notes=(
            f"The return streams are weakly correlated on average ({mean_off_diagonal:+.2f}), so "
            "combining does raise the Sharpe ratio above the *average* component. Whether it "
            "beats the *best* component is the sterner test and the one reported here: "
            f"{best_combo:.2f} versus {best_component:.2f}. Combination is not alchemy -- it "
            "cannot turn two strategies with no edge into one that has an edge, and Stage 6 "
            "established that the alpha strategies have no edge net of costs. What it does is "
            "reduce the variance of the total, which is worth having when the components are "
            "genuinely positive."
        ),
    )
    logger.info("STAGE 10 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
