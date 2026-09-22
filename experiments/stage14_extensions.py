"""Stage 14 (optional) - Pairs trading and PCA statistical arbitrage.

Extensions E and F (spec §53-§54). Kept as a separate research branch, run
only on request, and deliberately *not* folded into the headline model
ladder: they are different strategy types with different assumptions, and
mixing them into the core comparison would muddy the question that comparison
answers.

Figures 26-27.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import LinearCostModel
from src.backtest.engine import BacktestEngine
from src.features.returns import cumulative_returns
from src.features.volatility import rolling_volatility
from src.models.regression import cross_sectional_ic, ic_summary
from src.features.returns import forward_returns
from src.signals.pairs import backtest_pair, build_spread, pairs_positions, screen_pairs, analyse_pair
from src.signals.pca_strategy import (
    explained_variance_by_component,
    factor_exposure_check,
    pca_stat_arb_signal,
)
from src.signals.transform import signal_to_positions
from src.utils.plotting import PALETTE, bar_with_values, new_axes, save_figure, write_figure_index
from experiments.context import build_context
from experiments.strategies import transform_config

STAGE = "stage14_extensions"

# Economically motivated candidates, declared before any test is run.
CANDIDATE_PAIRS = [
    ("GLD", "SLV"),    # the book's gold / gold-miners analogue
    ("SPY", "QQQ"),
    ("SPY", "IWM"),
    ("EFA", "EEM"),
    ("IEF", "TLT"),
    ("LQD", "HYG"),
    ("AGG", "IEF"),
    ("SPY", "VNQ"),
    ("GLD", "DBC"),
    ("SHY", "IEF"),
]


def figure_pairs(context, prices, screen, headline, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.4))

    subset = screen.head(10).set_index("pair")
    bar_with_values(axes[0, 0], subset["adf_pvalue"], "Engle-Granger ADF p-value (training window)",
                    "p", "{:.3f}", rotation=60)
    axes[0, 0].axhline(0.05, color="#CC0000", linestyle="--", linewidth=1.1, label="5% level")
    axes[0, 0].legend(fontsize=8)

    left, right = headline["pair"].split("/")
    spread = build_spread(prices[left], prices[right], headline["hedge_ratio"], headline["intercept"])
    axes[0, 1].plot(spread.index, spread.to_numpy(), color="#0072B2", linewidth=0.9)
    axes[0, 1].axhline(float(spread.mean()), color="black", linestyle="--", linewidth=0.9)
    axes[0, 1].set_title(f"{left}/{right} spread (log, hedge ratio {headline['hedge_ratio']:.2f})")

    signals = pairs_positions(prices[left], prices[right], headline["hedge_ratio"],
                              headline["intercept"])
    axes[1, 0].plot(signals.index, signals["zscore"].to_numpy(), color="#999999", linewidth=0.7)
    for level, style in ((2.0, "--"), (-2.0, "--"), (0.5, ":"), (-0.5, ":")):
        axes[1, 0].axhline(level, color="#D55E00", linestyle=style, linewidth=0.9)
    axes[1, 0].set_ylabel("Spread z-score")
    axes[1, 0].set_title("Entry at ±2σ, exit at ±0.5σ")

    half_lives = screen.head(10).set_index("pair")["half_life_days"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(half_lives):
        bar_with_values(axes[1, 1], half_lives, "Spread half-life (days)", "days", "{:.0f}",
                        rotation=60)

    fig.suptitle("Figure 26. Pairs trading: does the relationship exist before the rule is built?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Are any of these pairs genuinely cointegrated on training data, and "
                           "does the spread revert fast enough to be tradable?", 26)


def figure_pca_strategy(context, cumulative, exposures, ic_table, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.4))

    axes[0, 0].plot(cumulative.index, cumulative.to_numpy(), color="#0072B2", linewidth=1.3)
    axes[0, 0].axhline(1.0, color="black", linewidth=0.8)
    axes[0, 0].set_ylabel("Growth of 1 unit (net)")
    axes[0, 0].set_title("PCA statistical-arbitrage equity curve")

    bar_with_values(axes[0, 1], exposures, "Residual factor exposures of the book",
                    "exposure", "{:.3f}", rotation=0)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.9)

    axes[1, 0].plot(ic_table.index, ic_table["mean_ic"], marker="o", markersize=4,
                    color="#D55E00")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 0].set_xlabel("Forecast horizon (days)")
    axes[1, 0].set_ylabel("Mean IC")
    axes[1, 0].set_title("IC of the residual reversion signal")

    axes[1, 1].plot(range(1, len(ic_table) + 1), ic_table["t_stat_overlap_adjusted"],
                    marker="s", markersize=4, color="#009E73")
    axes[1, 1].axhline(1.96, color="#CC0000", linestyle="--", linewidth=1.0, label="|t| = 1.96")
    axes[1, 1].axhline(-1.96, color="#CC0000", linestyle="--", linewidth=1.0)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 1].set_xticks(range(1, len(ic_table) + 1), [str(h) for h in ic_table.index])
    axes[1, 1].set_xlabel("Forecast horizon (days)")
    axes[1, 1].set_ylabel("Overlap-adjusted t")
    axes[1, 1].set_title("Is the residual signal statistically reliable?")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Figure 27. PCA statistical arbitrage: from understanding risk to generating alpha",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Do returns unexplained by the first three principal components mean "
                           "revert, and is a factor-neutral book built on them profitable net of "
                           "costs?", 27)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 14 | extensions: pairs trading and PCA statistical arbitrage")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    prices = market.prices
    cost_bps = float(cfg.get("backtest.costs.cost_bps", 10.0))
    development_end = cfg.get("backtest.samples.development.end")

    # ---------------------------------------------------- Extension E: pairs
    logger.info("--- Extension E: pairs trading (Ch. 22 §22.3.3-4) ---")
    screen = screen_pairs(prices, CANDIDATE_PAIRS, sample_end=development_end)
    context.save_table(screen, "stage14_pairs_screen.csv", index=False)
    logger.info("pair screen on the development sample only:\n%s", screen.round(4).to_string(index=False))

    tradable = screen[screen["tradable"]]
    logger.info("%d of %d candidate pairs are cointegrated and revert fast enough to trade",
                int(len(tradable)), int(len(screen)))

    pair_results = []
    for _, row in screen.iterrows():
        left, right = row["pair"].split("/")
        result = analyse_pair(
            prices.loc[prices.index <= pd.Timestamp(development_end), left],
            prices.loc[prices.index <= pd.Timestamp(development_end), right], left, right,
        )
        if not np.isfinite(result.hedge_ratio):
            continue
        # Traded over the WHOLE sample using the hedge ratio estimated on the
        # development window only, so the out-of-sample portion is honest.
        summary = backtest_pair(prices[left], prices[right], result, cost_bps=cost_bps)
        summary["selected_on_development"] = bool(row["tradable"])
        pair_results.append(summary)

    pair_frame = pd.DataFrame(pair_results).set_index("pair")
    context.save_table(pair_frame, "stage14_pairs_backtests.csv")
    logger.info("pair backtests (hedge ratio fitted on development data only):\n%s",
                pair_frame[["sharpe", "gross_sharpe", "cagr", "max_drawdown", "ann_turnover",
                            "n_round_trips", "half_life_days", "adf_pvalue"]].astype(float).round(4).to_string())

    headline_row = screen.iloc[0]
    headline = {"pair": headline_row["pair"], "hedge_ratio": headline_row["hedge_ratio"],
                "intercept": 0.0}
    left, right = headline_row["pair"].split("/")
    refit = analyse_pair(prices.loc[prices.index <= pd.Timestamp(development_end), left],
                      prices.loc[prices.index <= pd.Timestamp(development_end), right], left, right)
    headline["hedge_ratio"], headline["intercept"] = refit.hedge_ratio, refit.intercept
    figure_pairs(context, prices, screen, headline, context.figure("fig26_pairs_trading.png"))

    # ------------------------------------------- Extension F: PCA stat arb
    logger.info("--- Extension F: PCA statistical arbitrage (Ch. 22 §22.3.5-7) ---")
    explained = explained_variance_by_component(returns)
    logger.info("cumulative explained variance: PC1 %.1f%%  PC1-3 %.1f%%  PC1-5 %.1f%%",
                100 * explained.iloc[0], 100 * explained.iloc[2], 100 * explained.iloc[4])

    signal = pca_stat_arb_signal(returns, n_components=3, window=252, zscore_window=21)
    signal = signal.where(market.investable)

    horizons = tuple(cfg.get("strategies.horizons", [1, 5, 10, 21, 42, 63]))
    ic_rows = {}
    for horizon in horizons:
        ic = cross_sectional_ic(signal, forward_returns(returns, horizon))
        summary = ic_summary(ic, horizon=horizon, breadth=len(market.tickers))
        if summary:
            ic_rows[horizon] = summary
    ic_table = pd.DataFrame(ic_rows).T
    ic_table.index.name = "horizon"
    context.save_table(ic_table, "stage14_pca_stat_arb_ic.csv")
    logger.info("PCA residual signal IC:\n%s",
                ic_table[["mean_ic", "t_stat_overlap_adjusted", "p_value"]].astype(float).round(4).to_string())

    weights = signal_to_positions(signal, rolling_volatility(returns, 63),
                                  investable=market.investable, **transform_config(cfg))
    engine = BacktestEngine(signal_lag=1, rebalance=str(cfg.get("backtest.engine.rebalance", "monthly")),
                            weight_drift=True,
                            cost_model=LinearCostModel(cost_bps,
                                                       dict(cfg.get("backtest.costs.per_asset_bps", {}) or {})))
    result = engine.run(weights, returns, "pca_stat_arb", market.investable, apply_vol_target=False)
    summary = result.summary()
    context.save_table(pd.Series(summary).to_frame("value"), "stage14_pca_stat_arb_backtest.csv")
    logger.info("PCA stat-arb backtest: net Sharpe %+.3f gross %+.3f turnover %.1fx/yr breakeven %.1f bps",
                summary["sharpe"], summary["gross_sharpe"], summary["ann_turnover"],
                result.breakeven_cost_bps())

    exposures = factor_exposure_check(result.weights.iloc[-1], returns, 3)
    context.save_table(exposures.to_frame(), "stage14_pca_factor_exposures.csv")
    logger.info("residual factor exposures of the final book: %s", exposures.round(4).to_dict())

    figure_pca_strategy(context, cumulative_returns(result.net_returns), exposures, ic_table,
                        context.figure("fig27_pca_stat_arb.png"))
    write_figure_index(cfg.reports_dir() / "figure_index.md", context.figures)

    # ---------------------------------------------------------- registry
    best_pair = pair_frame["sharpe"].astype(float).idxmax() if len(pair_frame) else "none"
    context.registry.log(
        "Cointegrated pairs within this ETF universe produce tradable relative-value "
        "returns after costs (Extension E, Ch. 22 §22.3.3-4).",
        stage=STAGE,
        parameters={"candidates": [f"{a}/{b}" for a, b in CANDIDATE_PAIRS],
                    "entry_sigma": 2.0, "exit_sigma": 0.5, "zscore_window": 63},
        train_period=f"hedge ratio and cointegration test on data to {development_end}",
        cost_bps=cost_bps,
        results={
            "pairs_tested": int(len(screen)),
            "pairs_cointegrated": int(screen["cointegrated"].sum()),
            "pairs_tradable": int(screen["tradable"].sum()),
            "best_pair": str(best_pair),
            "best_pair_net_sharpe": float(pair_frame.loc[best_pair, "sharpe"]) if len(pair_frame) else np.nan,
            "best_pair_gross_sharpe": float(pair_frame.loc[best_pair, "gross_sharpe"]) if len(pair_frame) else np.nan,
            "median_pair_net_sharpe": float(pair_frame["sharpe"].astype(float).median()) if len(pair_frame) else np.nan,
        },
        decision="investigate",
        notes=(
            "Pairs were selected by cointegration on the development sample only and then "
            "traded over the whole history with that hedge ratio, so the later portion is "
            "genuinely out of sample. The screen is the discipline: a pair that is 0.95 "
            "correlated but not cointegrated has a spread that can wander off forever, and "
            "trading it is a bet on nothing."
        ),
    )
    context.registry.log(
        "Returns unexplained by the first three principal components mean revert, and a "
        "factor-neutral book built on them is profitable net of costs "
        "(Extension F, Ch. 22 §22.3.5-7).",
        stage=STAGE,
        parameters={"n_components": 3, "window": 252, "zscore_window": 21, "refit_step": 21},
        cost_bps=cost_bps,
        results={
            "pc1_explained": float(explained.iloc[0]),
            "pc1_to_pc3_explained": float(explained.iloc[2]),
            "best_ic": float(ic_table["mean_ic"].abs().max()),
            "ic_at_h1": float(ic_table.loc[1, "mean_ic"]) if 1 in ic_table.index else np.nan,
            "t_at_h1": float(ic_table.loc[1, "t_stat_overlap_adjusted"]) if 1 in ic_table.index else np.nan,
            "net_sharpe": float(summary["sharpe"]),
            "gross_sharpe": float(summary["gross_sharpe"]),
            "ann_turnover": float(summary["ann_turnover"]),
            "breakeven_cost_bps": float(result.breakeven_cost_bps()),
            "max_residual_factor_exposure": float(exposures.abs().max()),
        },
        decision="retain" if summary["sharpe"] > 0.2 and result.breakeven_cost_bps() > 2 * cost_bps else "reject",
        notes=(
            "The construction does what it claims -- residual factor exposures of the final "
            f"book are at most {float(exposures.abs().max()):.3f} -- so any result is genuinely "
            "factor neutral rather than a disguised beta. Whether it survives costs is the "
            "same question that decided every other alpha strategy in this project, and the "
            "breakeven cost is the number that answers it."
        ),
    )
    logger.info("STAGE 14 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
