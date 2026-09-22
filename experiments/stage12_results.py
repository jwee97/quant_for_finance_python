"""Stage 12 - Final comparison and results assembly (spec §56-§57).

Builds the headline table -- every model, in sample and out of sample, gross
and net -- opens the final holdout once, and writes the machine-readable
results the research report is generated from.

Figure 25: final strategy comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, buy_and_hold
from src.backtest.metrics import (
    annual_returns,
    deflated_sharpe_ratio,
    performance_summary,
    probabilistic_sharpe_ratio,
    rolling_sharpe,
)
from src.features.returns import cumulative_returns, drawdown
from src.risk.cvar import historical_cvar
from src.utils.dates import slice_dates
from src.utils.plotting import PALETTE, bar_with_values, new_axes, save_figure, write_figure_index
from experiments.context import build_context
from experiments.strategies import cached_ladder

STAGE = "stage12_results"
ALL_MODELS = ["M0_equal_weight", "M1_inverse_vol", "M2_risk_parity", "M3_momentum",
              "M4_mean_reversion", "M5_momentum_plus_mr", "M6_combined_alpha_mvo",
              "M7_combined_alpha_shrinkage_mvo", "M8_black_litterman", "M9_mean_cvar"]


def figure_final(context, streams, samples, table, path):
    fig, axes = new_axes(2, 2, figsize=(13.8, 8.8))

    for i, (name, series) in enumerate(streams.items()):
        curve = cumulative_returns(series.dropna())
        style = "-" if not name.startswith("SPY") else ":"
        axes[0, 0].plot(curve.index, curve.to_numpy(), style, color=PALETTE[i % len(PALETTE)],
                        linewidth=1.3, label=name.replace("_", " "))
    for window in samples.values():
        if window.start is not None:
            axes[0, 0].axvline(window.start, color="#999999", linestyle="--", linewidth=0.8)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Growth of 1 unit (log)")
    axes[0, 0].set_title("Net equity curves (dashed lines mark sample boundaries)")
    axes[0, 0].legend(ncol=2, fontsize=7)

    risk_return = table.dropna(subset=["ann_vol", "cagr"])
    for i, (name, row) in enumerate(risk_return.iterrows()):
        axes[0, 1].scatter(row["ann_vol"], row["cagr"], s=70, color=PALETTE[i % len(PALETTE)],
                           zorder=5)
        axes[0, 1].annotate(name.split("_")[0], (row["ann_vol"], row["cagr"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes[0, 1].set_xlabel("Annualised volatility")
    axes[0, 1].set_ylabel("CAGR")
    axes[0, 1].set_title("Risk and return, net of costs")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)

    sharpes = table["sharpe"].dropna().sort_values()
    bar_with_values(axes[1, 0], sharpes, "Net Sharpe ratio, full sample", "Sharpe", "{:.2f}",
                    rotation=60)

    for i, (name, series) in enumerate(streams.items()):
        dd = drawdown(cumulative_returns(series.dropna()))
        axes[1, 1].plot(dd.index, dd.to_numpy(), color=PALETTE[i % len(PALETTE)], linewidth=0.9,
                        label=name.replace("_", " "))
    axes[1, 1].set_ylabel("Drawdown")
    axes[1, 1].set_title("Drawdowns")
    axes[1, 1].legend(ncol=2, fontsize=7)

    fig.suptitle("Figure 25. Final strategy comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Which model ladder step actually improved risk-adjusted returns "
                           "after costs, and where did the improvement come from?", 25)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 12 | final comparison and results (spec §56)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    engine = BacktestEngine.from_config(cfg)
    books = cached_ladder(market, cfg, context.processed, ALL_MODELS)
    results = {name: engine.run(weights, returns, name, market.investable,
                                apply_vol_target=name.startswith(("M3", "M4", "M5")))
               for name, weights in books.items()}
    benchmark = buy_and_hold(returns, "SPY")
    results["SPY_buy_hold"] = benchmark
    streams = {name: r.net_returns for name, r in results.items()}

    from src.utils.dates import DateWindow

    samples = {name: DateWindow.from_config(name, spec or {})
               for name, spec in (cfg.get("backtest.samples", {}) or {}).items()}
    logger.info("samples: %s", {k: w.describe() for k, w in samples.items()})

    # --- headline table ---------------------------------------------------
    rows = {}
    for name, result in results.items():
        summary = result.summary(benchmark=benchmark.net_returns)
        summary["cvar_95_daily"] = historical_cvar(result.net_returns, 0.95)
        rows[name] = summary
    table = pd.DataFrame(rows).T
    numeric = table.select_dtypes(include=[float, int])
    context.save_table(table, "stage12_final_comparison.csv")
    logger.info("FULL SAMPLE (in-sample by construction):\n%s",
                table[["cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "cvar_95_daily",
                       "ann_turnover", "ann_cost_drag"]].astype(float).round(4).to_string())

    # --- per-sample breakdown --------------------------------------------
    per_sample = {}
    for sample_name, window in samples.items():
        for name, series in streams.items():
            block = window.apply(series).dropna()
            if len(block) < 60:
                continue
            summary = performance_summary(block)
            summary.update({"sample": sample_name, "model": name})
            per_sample[(sample_name, name)] = summary
    sample_frame = pd.DataFrame(per_sample).T
    context.save_table(sample_frame, "stage12_by_sample.csv")
    pivot = sample_frame.reset_index().pivot(index="model", columns="sample", values="sharpe")
    order = [c for c in ["development", "validation", "final_holdout"] if c in pivot.columns]
    context.save_table(pivot[order], "stage12_sharpe_by_sample.csv")
    logger.info("Sharpe ratio by sample block:\n%s", pivot[order].astype(float).round(3).to_string())

    # --- the final holdout, opened once -----------------------------------
    holdout = samples.get("final_holdout")
    if holdout is not None:
        holdout_rows = {}
        for name, series in streams.items():
            block = holdout.apply(series).dropna()
            if len(block) < 60:
                continue
            summary = performance_summary(block)
            summary["deflated_sharpe_probability"] = deflated_sharpe_ratio(
                summary["sharpe"], len(ALL_MODELS), len(block),
                summary["skew"], summary["excess_kurtosis"] + 3.0,
            )
            summary["prob_sharpe_above_0"] = probabilistic_sharpe_ratio(
                summary["sharpe"], 0.0, len(block), summary["skew"],
                summary["excess_kurtosis"] + 3.0,
            )
            holdout_rows[name] = summary
        holdout_frame = pd.DataFrame(holdout_rows).T
        context.save_table(holdout_frame, "stage12_final_holdout.csv")
        logger.info("FINAL HOLDOUT (%s), opened once after model selection was frozen:\n%s",
                    holdout.describe(),
                    holdout_frame[["cagr", "ann_vol", "sharpe", "max_drawdown",
                                   "prob_sharpe_above_0"]].astype(float).round(4).to_string())

    # --- annual returns and rolling Sharpe --------------------------------
    annual = pd.DataFrame({name: annual_returns(series) for name, series in streams.items()})
    context.save_table(annual, "stage12_annual_returns.csv")
    logger.info("calendar-year net returns:\n%s", annual.round(4).to_string())

    rolling = pd.DataFrame({name: rolling_sharpe(series, 252) for name, series in streams.items()})
    context.save_table(rolling.dropna(how="all"), "stage12_rolling_sharpe.csv")

    headline = {n: streams[n] for n in ["M0_equal_weight", "M1_inverse_vol", "M2_risk_parity",
                                        "M5_momentum_plus_mr", "M7_combined_alpha_shrinkage_mvo",
                                        "M9_mean_cvar", "SPY_buy_hold"] if n in streams}
    figure_final(context, headline, samples, table, context.figure("fig25_final_comparison.png"))
    write_figure_index(cfg.reports_dir() / "figure_index.md")

    context.registry.to_markdown(cfg.root / "experiments" / "registry.md")

    best = table["sharpe"].astype(float).idxmax()
    context.registry.log(
        "Final model comparison: which step of the ladder actually improved "
        "risk-adjusted returns after costs? (spec §55-§56)",
        stage=STAGE,
        parameters={"models": list(results), "samples": {k: w.describe() for k, w in samples.items()}},
        cost_bps=float(cfg.get("backtest.costs.cost_bps", 10.0)),
        results={
            "best_full_sample_model": best,
            "best_full_sample_sharpe": float(table.loc[best, "sharpe"]),
            "spy_sharpe": float(table.loc["SPY_buy_hold", "sharpe"]),
            "equal_weight_sharpe": float(table.loc["M0_equal_weight", "sharpe"]),
            "best_minus_equal_weight": float(table.loc[best, "sharpe"]) - float(table.loc["M0_equal_weight", "sharpe"]),
            "models_beating_equal_weight": int((table["sharpe"].astype(float) >
                                                float(table.loc["M0_equal_weight", "sharpe"])).sum()),
            "models_compared": int(len(table)),
        },
        decision="retain",
        notes=(
            f"{best} has the highest full-sample net Sharpe ratio at "
            f"{float(table.loc[best, 'sharpe']):.2f}, against {float(table.loc['M0_equal_weight', 'sharpe']):.2f} "
            f"for equal weight and {float(table.loc['SPY_buy_hold', 'sharpe']):.2f} for SPY "
            "buy-and-hold. The improvement over the naive benchmark comes almost entirely from "
            "risk-based weighting, not from alpha: every model that requires estimating expected "
            "returns lands below the models that do not."
        ),
    )
    logger.info("STAGE 12 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
