"""Stage 6 - Backtesting the signal strategies (Ch. 22 §22.2).

Runs the alpha models (M3-M5) through the engine with the full timing
convention and cost model, then asks the question the IC work left open: the
information was concentrated at 1-5 days, so does anything survive the
turnover that implies?

Figures 14-15: gross vs net returns, transaction-cost sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import breakeven_cost, cost_sensitivity, turnover_statistics
from src.backtest.engine import BacktestEngine, buy_and_hold
from src.backtest.metrics import annual_returns, rolling_sharpe
from src.features.returns import cumulative_returns
from src.utils.plotting import PALETTE, bar_with_values, new_axes, save_figure
from experiments.context import build_context
from experiments.strategies import build_ladder

STAGE = "stage06_backtest"
SIGNAL_MODELS = ["M3_momentum", "M4_mean_reversion", "M5_momentum_plus_mr"]


def figure_gross_vs_net(context, results, benchmark, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.6))

    for i, (name, result) in enumerate(results.items()):
        axes[0, 0].plot(result.gross_equity_curve.index, result.gross_equity_curve.to_numpy(),
                        color=PALETTE[i], linestyle="--", linewidth=1.0, alpha=0.75)
        axes[0, 0].plot(result.equity_curve.index, result.equity_curve.to_numpy(),
                        color=PALETTE[i], linewidth=1.5, label=f"{name} (net)")
    curve = cumulative_returns(benchmark.dropna())
    axes[0, 0].plot(curve.index, curve.to_numpy(), color="#555555", linewidth=1.2,
                    linestyle=":", label="SPY buy & hold")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Growth of 1 unit (log)")
    axes[0, 0].set_title("Equity curves: dashed = gross, solid = net")
    axes[0, 0].legend(fontsize=8)

    gap = pd.Series({name: r.summary()["gross_sharpe"] - r.summary()["sharpe"]
                     for name, r in results.items()})
    bar_with_values(axes[0, 1], gap, "Sharpe ratio destroyed by transaction costs",
                    "Gross Sharpe - net Sharpe", "{:.3f}", rotation=20)

    turnovers = pd.Series({name: turnover_statistics(r.weights, r.trades)["annualised_turnover"]
                           for name, r in results.items()})
    bar_with_values(axes[1, 0], turnovers, "Annualised turnover", "x per year", "{:.1f}", rotation=20)

    for i, (name, result) in enumerate(results.items()):
        rolled = rolling_sharpe(result.net_returns, 252)
        axes[1, 1].plot(rolled.index, rolled.to_numpy(), color=PALETTE[i], linewidth=1.0, label=name)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 1].set_ylabel("252-day rolling Sharpe (net)")
    axes[1, 1].set_title("Was the strategy ever working?")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Figure 14. Gross versus net: what transaction costs do to the alpha models",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "How much of each strategy's gross performance survives realistic "
                           "transaction costs, and how much turnover is it paying for?", 14)


def figure_cost_sensitivity(context, results, sensitivities, breakevens, path):
    fig, axes = new_axes(1, 2, figsize=(13.5, 5.6))

    for i, (name, table) in enumerate(sensitivities.items()):
        axes[0].plot(table.index, table["net_sharpe"], marker="o", markersize=4.5,
                     color=PALETTE[i], label=name)
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_xlabel("Transaction cost (bps of traded notional)")
    axes[0].set_ylabel("Net Sharpe ratio")
    axes[0].set_title("Net Sharpe against the cost assumption")
    axes[0].legend(fontsize=8.5)

    series = pd.Series(breakevens).sort_values()
    bar_with_values(axes[1], series, "Breakeven transaction cost", "bps", "{:.1f}", rotation=20)
    axes[1].axhline(10.0, color="#D55E00", linestyle="--", linewidth=1.2,
                    label="baseline assumption (10 bps)")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("Figure 15. Transaction-cost sensitivity and breakeven costs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "At what level of transaction costs does each strategy stop working, "
                           "and is that level plausible for these instruments?", 15)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 6 | backtesting the alpha models (Ch. 22)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    engine = BacktestEngine.from_config(cfg)
    logger.info("engine: lag=%d rebalance=%s drift=%s costs=%.1fbps target_vol=%s",
                engine.signal_lag, engine.rebalance, engine.weight_drift,
                engine.cost_model.cost_bps, engine.target_vol)

    books = build_ladder(market, cfg, SIGNAL_MODELS)
    results = {name: engine.run(weights, returns, name, market.investable, apply_vol_target=True)
               for name, weights in books.items()}
    benchmark = buy_and_hold(returns, "SPY").net_returns

    summary = pd.DataFrame({name: r.summary(benchmark=benchmark) for name, r in results.items()}).T
    context.save_table(summary, "stage06_signal_backtests.csv")
    logger.info("backtest summary:\n%s",
                summary[["cagr", "ann_vol", "sharpe", "gross_sharpe", "max_drawdown",
                         "ann_turnover", "ann_cost_drag"]].astype(float).round(4).to_string())

    levels = tuple(cfg.get("backtest.costs.sensitivity_bps", [0, 5, 10, 25, 50]))
    per_asset = dict(cfg.get("backtest.costs.per_asset_bps", {}) or {})
    sensitivities, breakevens = {}, {}
    for name, result in results.items():
        table = cost_sensitivity(result.gross_returns, result.trades, levels, per_asset)
        sensitivities[name] = table
        breakevens[name] = breakeven_cost(result.gross_returns, result.trades)
        context.save_table(table, f"stage06_cost_sensitivity_{name}.csv")
    context.save_table(pd.Series(breakevens, name="breakeven_bps").to_frame(),
                       "stage06_breakeven_costs.csv")
    logger.info("breakeven transaction costs (bps): %s",
                {k: round(v, 1) for k, v in breakevens.items()})
    for name, table in sensitivities.items():
        logger.info("%s cost sensitivity:\n%s", name, table[["net_ann_return", "net_sharpe",
                                                             "annual_cost_drag"]].round(4).to_string())

    turnover_table = pd.DataFrame(
        {name: turnover_statistics(r.weights, r.trades) for name, r in results.items()}
    ).T
    context.save_table(turnover_table, "stage06_turnover.csv")

    annual = pd.DataFrame({name: annual_returns(r.net_returns) for name, r in results.items()})
    context.save_table(annual, "stage06_annual_returns.csv")

    for name, result in results.items():
        result.write(context.processed / "backtests")

    figure_gross_vs_net(context, results, benchmark, context.figure("fig14_gross_vs_net.png"))
    figure_cost_sensitivity(context, results, sensitivities, breakevens,
                            context.figure("fig15_cost_sensitivity.png"))

    baseline_bps = float(cfg.get("backtest.costs.cost_bps", 10.0))
    for name, result in results.items():
        stats = result.summary()
        decision = ("retain" if stats["sharpe"] > 0.2 and breakevens[name] > baseline_bps * 2
                    else "reject")
        context.registry.log(
            f"{name} generates positive risk-adjusted returns after realistic transaction costs.",
            stage=STAGE,
            parameters={"rebalance": engine.rebalance, "signal_lag": engine.signal_lag,
                        "target_vol": engine.target_vol, "model": name},
            test_period=f"{stats['start']} to {stats['end']}",
            cost_bps=baseline_bps,
            results={
                "gross_sharpe": stats["gross_sharpe"],
                "net_sharpe": stats["sharpe"],
                "net_cagr": stats["cagr"],
                "max_drawdown": stats["max_drawdown"],
                "ann_turnover": stats["ann_turnover"],
                "ann_cost_drag": stats["ann_cost_drag"],
                "breakeven_cost_bps": breakevens[name],
            },
            decision=decision,
            notes=(
                f"Turnover of {stats['ann_turnover']:.1f}x per year costs "
                f"{1e4 * stats['ann_cost_drag']:.0f} bps annually at the {baseline_bps:.0f} bps "
                f"baseline, taking the Sharpe ratio from {stats['gross_sharpe']:.2f} gross to "
                f"{stats['sharpe']:.2f} net. The strategy breaks even at "
                f"{breakevens[name]:.1f} bps. This is the direct consequence of Stage 5's finding "
                "that the predictive information sits at the 1-5 day horizon: the only way to "
                "harvest it is to trade constantly, and the trading costs more than the signal "
                "is worth."
            ),
        )
    logger.info("STAGE 6 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
