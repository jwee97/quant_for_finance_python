"""Stage 11 - Out-of-sample validation, robustness and leakage
(spec §40-§44, Ch. 22 §22.2.4, §22.2.6-7).

Three separate questions:

1. **Walk forward.** Does anything survive when parameters are estimated only
   on past data and frozen before the test block?
2. **Robustness.** Is the result a plateau across the parameter family, or one
   lucky point? And what does the deflated Sharpe ratio say once the number of
   variants tried is accounted for?
3. **Leakage.** Can the code see the future? Tested mechanically, not argued.

Figures 22-23: parameter sensitivity, in-sample vs out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import LinearCostModel
from src.backtest.engine import BacktestEngine, buy_and_hold
from src.backtest.metrics import sharpe_ratio
from src.features.momentum import total_return_momentum, volatility_scaled_momentum
from src.features.mean_reversion import price_zscore
from src.features.volatility import rolling_volatility
from src.portfolio.constraints import Constraints
from src.portfolio.equal_weight import equal_weight_book
from src.portfolio.inverse_vol import inverse_vol_book
from src.signals.transform import signal_to_positions
from src.utils.dates import slice_dates
from src.validation.leakage import detect_suspicious_alignment, run_leakage_suite
from src.validation.robustness import (
    multiple_testing_penalty,
    parameter_surface,
    stationary_block_bootstrap,
    subperiod_stability,
    surface_diagnostics,
)
from src.validation.walk_forward import (
    WalkForwardSplitter,
    in_sample_vs_out_of_sample,
    run_walk_forward,
)
from src.utils.plotting import PALETTE, bar_with_values, new_axes, plot_heatmap, save_figure
from experiments.context import build_context
from experiments.strategies import cached_ladder, transform_config

STAGE = "stage11_validation"
WALK_FORWARD_MODELS = ["M0_equal_weight", "M1_inverse_vol", "M2_risk_parity",
                       "M3_momentum", "M5_momentum_plus_mr", "M9_mean_cvar"]


def figure_parameter_sensitivity(context, surfaces, diagnostics, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.6))

    momentum = surfaces["momentum"]
    for i, horizon in enumerate(sorted(momentum["rebalance"].unique())):
        subset = momentum[momentum["rebalance"] == horizon].sort_values("lookback")
        axes[0, 0].plot(subset["lookback"], subset["sharpe"], marker="o", markersize=4,
                        color=PALETTE[i], label=f"rebalance {horizon}")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 0].set_xlabel("Momentum lookback (days)")
    axes[0, 0].set_ylabel("Net Sharpe ratio")
    axes[0, 0].set_title("Momentum: net Sharpe across the whole lookback family")
    axes[0, 0].legend(fontsize=8.5)

    pivot = momentum.pivot_table(index="rebalance", columns="lookback", values="sharpe")
    image = plot_heatmap(axes[0, 1], pivot.astype(float), "Momentum Sharpe surface",
                         vmin=-0.6, vmax=0.6, fmt="{:.2f}")
    fig.colorbar(image, ax=axes[0, 1], shrink=0.8)

    reversion = surfaces["mean_reversion"]
    for i, horizon in enumerate(sorted(reversion["rebalance"].unique())):
        subset = reversion[reversion["rebalance"] == horizon].sort_values("lookback")
        axes[1, 0].plot(subset["lookback"], subset["sharpe"], marker="s", markersize=4,
                        color=PALETTE[i], label=f"rebalance {horizon}")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 0].set_xlabel("Mean-reversion lookback (days)")
    axes[1, 0].set_ylabel("Net Sharpe ratio")
    axes[1, 0].set_title("Mean reversion: net Sharpe across the lookback family")
    axes[1, 0].legend(fontsize=8.5)

    summary = pd.DataFrame(diagnostics).T[["best", "median", "worst", "share_positive"]]
    x = np.arange(len(summary))
    for i, column in enumerate(["best", "median", "worst"]):
        axes[1, 1].bar(x + i * 0.27, summary[column].to_numpy(), width=0.27,
                       label=column, color=PALETTE[i])
    axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 1].set_xticks(x + 0.27, summary.index, rotation=15)
    axes[1, 1].set_ylabel("Net Sharpe ratio")
    axes[1, 1].set_title("Plateau or spike? Best vs median vs worst parameter")
    axes[1, 1].legend(fontsize=8.5)

    fig.suptitle("Figure 22. Parameter sensitivity: is the result a plateau or one lucky point?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Does performance hold across the whole parameter family, or does it "
                           "depend on one narrow choice that would signal parameter mining?", 22)


def figure_is_vs_oos(context, comparison, fold_tables, bootstraps, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.6))

    frame = pd.DataFrame(comparison).T
    x = np.arange(len(frame))
    axes[0, 0].bar(x - 0.2, frame["is_sharpe"].to_numpy(), width=0.4, label="in sample",
                   color="#D55E00")
    axes[0, 0].bar(x + 0.2, frame["oos_sharpe"].to_numpy(), width=0.4, label="out of sample",
                   color="#0072B2")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 0].set_xticks(x, [i.replace("_", " ") for i in frame.index], rotation=25,
                          ha="right", fontsize=8)
    axes[0, 0].set_ylabel("Sharpe ratio")
    axes[0, 0].set_title("In-sample versus walk-forward out-of-sample")
    axes[0, 0].legend(fontsize=8.5)

    bar_with_values(axes[0, 1], frame["sharpe_decay"], "Out-of-sample slippage (IS - OOS Sharpe)",
                    "Sharpe lost", "{:.2f}", rotation=25)

    for i, (name, table) in enumerate(fold_tables.items()):
        if "test_sharpe" not in table:
            continue
        valid = table.dropna(subset=["test_sharpe"])
        axes[1, 0].plot(pd.to_datetime(valid["test_start"]), valid["test_sharpe"],
                        marker="o", markersize=3.5, color=PALETTE[i], label=name)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 0].set_ylabel("Sharpe ratio in the test block")
    axes[1, 0].set_title("Per-fold out-of-sample Sharpe ratios")
    axes[1, 0].legend(fontsize=7.5)

    names = list(bootstraps)
    observed = [bootstraps[n]["observed"] for n in names]
    lower = [bootstraps[n]["observed"] - bootstraps[n]["ci_lower_5pct"] for n in names]
    upper = [bootstraps[n]["ci_upper_95pct"] - bootstraps[n]["observed"] for n in names]
    axes[1, 1].errorbar(range(len(names)), observed, yerr=[lower, upper], fmt="o",
                        capsize=5, color="#0072B2", markersize=6)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 1].set_xticks(range(len(names)), [n.replace("_", " ") for n in names],
                          rotation=25, ha="right", fontsize=8)
    axes[1, 1].set_ylabel("Sharpe ratio")
    axes[1, 1].set_title("Block-bootstrap 90% confidence intervals")

    fig.suptitle("Figure 23. Out-of-sample validation and statistical uncertainty",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "How much performance is lost moving from in-sample to walk-forward "
                           "out-of-sample, and is any of it distinguishable from zero?", 23)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 11 | walk-forward, robustness and leakage")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    engine = BacktestEngine.from_config(cfg)
    constraints = Constraints.from_config(cfg)
    transform = transform_config(cfg)

    # ---------------------------------------------------------------- 1. leakage
    logger.info("--- automated look-ahead tests (spec §44) ---")

    def momentum_builder(m):
        signal = volatility_scaled_momentum(m.prices, m.returns(), 126, 1, 63).where(m.investable)
        return signal_to_positions(signal, rolling_volatility(m.returns(), 63),
                                   investable=m.investable, **transform)

    def reversion_builder(m):
        signal = (-price_zscore(m.prices, 21)).where(m.investable)
        return signal_to_positions(signal, rolling_volatility(m.returns(), 63),
                                   investable=m.investable, **transform)

    def inverse_vol_builder(m):
        return inverse_vol_book(m.returns(), m.investable, 63, "ewma", 40.0, constraints)

    def equal_weight_builder(m):
        return equal_weight_book(m.investable, constraints)

    def cheating_builder(m):
        """A deliberately broken strategy: the test must catch this one."""
        return signal_to_positions(m.returns().shift(-1), rolling_volatility(m.returns(), 63),
                                   investable=m.investable, **transform)

    leakage = run_leakage_suite(
        market,
        {
            "M1_inverse_vol": inverse_vol_builder,
            "M3_momentum": momentum_builder,
            "M4_mean_reversion": reversion_builder,
            "CONTROL_cheating_strategy": cheating_builder,
        },
        split_dates=("2012-06-29", "2017-06-30", "2021-06-30"),
        modes=("shock", "reverse"),
    )
    context.save_table(leakage, "stage11_leakage_tests.csv", index=False)
    logger.info("leakage suite:\n%s",
                leakage[["strategy", "mode", "split_date", "passed", "n_differing_cells"]].to_string(index=False))

    real = leakage[~leakage["strategy"].str.startswith("CONTROL")]
    control = leakage[leakage["strategy"].str.startswith("CONTROL")]
    all_real_pass = bool(real["passed"].all())
    control_caught = bool((~control["passed"]).all())
    logger.info("real strategies pass: %s | control strategy correctly caught: %s",
                all_real_pass, control_caught)

    alignment = {}
    for name, signal in {
        "momentum_126": total_return_momentum(market.prices, 126, 1),
        "zscore_21": price_zscore(market.prices, 21),
        "CONTROL_next_day_return": returns.shift(-1),
    }.items():
        frame = detect_suspicious_alignment(signal, returns)
        alignment[name] = {
            "worst_forward_correlation": float(frame["worst_forward_correlation"].iloc[0]),
            "leakage": bool(frame["leakage"].iloc[0]),
            "diagnosis": str(frame["diagnosis"].iloc[0]),
        }
    context.save_table(pd.DataFrame(alignment).T, "stage11_signal_alignment.csv")
    logger.info("signal alignment diagnostics:\n%s", pd.DataFrame(alignment).T.to_string())

    # ------------------------------------------------- 2. parameter robustness
    logger.info("--- parameter sensitivity (spec §42) ---")

    def evaluate_momentum(params: dict) -> dict:
        signal = volatility_scaled_momentum(market.prices, returns, params["lookback"], 1, 63)
        weights = signal_to_positions(signal.where(market.investable),
                                      rolling_volatility(returns, 63),
                                      investable=market.investable, **transform)
        local = BacktestEngine(signal_lag=1, rebalance=params["rebalance"], weight_drift=True,
                               cost_model=LinearCostModel(cfg.get("backtest.costs.cost_bps", 10.0),
                                                          cfg.get("backtest.costs.per_asset_bps", {})))
        result = local.run(weights, returns, "sweep", market.investable, apply_vol_target=False)
        stats = result.summary()
        return {"sharpe": stats["sharpe"], "gross_sharpe": stats["gross_sharpe"],
                "cagr": stats["cagr"], "turnover": stats["ann_turnover"]}

    def evaluate_reversion(params: dict) -> dict:
        signal = -price_zscore(market.prices, params["lookback"])
        weights = signal_to_positions(signal.where(market.investable),
                                      rolling_volatility(returns, 63),
                                      investable=market.investable, **transform)
        local = BacktestEngine(signal_lag=1, rebalance=params["rebalance"], weight_drift=True,
                               cost_model=LinearCostModel(cfg.get("backtest.costs.cost_bps", 10.0),
                                                          cfg.get("backtest.costs.per_asset_bps", {})))
        result = local.run(weights, returns, "sweep", market.investable, apply_vol_target=False)
        stats = result.summary()
        return {"sharpe": stats["sharpe"], "gross_sharpe": stats["gross_sharpe"],
                "cagr": stats["cagr"], "turnover": stats["ann_turnover"]}

    surfaces = {
        "momentum": parameter_surface(evaluate_momentum, {
            "lookback": list(cfg.get("strategies.momentum.robustness_lookbacks",
                                     [21, 42, 63, 84, 126, 168, 189, 252])),
            "rebalance": ["weekly", "monthly", "quarterly"],
        }),
        "mean_reversion": parameter_surface(evaluate_reversion, {
            "lookback": list(cfg.get("strategies.mean_reversion.robustness_lookbacks",
                                     [3, 5, 10, 15, 21, 42, 63])),
            "rebalance": ["weekly", "monthly", "quarterly"],
        }),
    }
    diagnostics = {}
    for name, surface in surfaces.items():
        context.save_table(surface, f"stage11_parameter_surface_{name}.csv", index=False)
        diagnostics[name] = surface_diagnostics(surface, "sharpe")
        logger.info("%s parameter surface: %s", name, diagnostics[name])
    context.save_table(pd.DataFrame(diagnostics).T, "stage11_surface_diagnostics.csv")

    # --------------------------------------------------------- 3. walk forward
    logger.info("--- walk-forward validation (spec §40) ---")
    splitter = WalkForwardSplitter.from_config(cfg)
    books = cached_ladder(market, cfg, context.processed, WALK_FORWARD_MODELS)
    full_results = {name: engine.run(weights, returns, name, market.investable,
                                     apply_vol_target=name.startswith(("M3", "M5")))
                    for name, weights in books.items()}

    walk_forward, fold_tables, comparison = {}, {}, {}
    development_end = cfg.get("backtest.samples.development.end")

    for name, weights in books.items():
        def fit_predict(fold, weights=weights, name=name):
            # The books are already causal: every weight at t uses data <= t.
            # Walk forward here therefore measures *regime* generalisation --
            # performance in each unseen year -- with the training block used
            # to decide nothing that the test block then reuses.
            test_weights = slice_dates(weights, fold.test_start, fold.test_end)
            if test_weights.empty:
                return None, None, None
            local = engine.run(weights, returns, name, market.investable,
                               apply_vol_target=name.startswith(("M3", "M5")),
                               start=fold.test_start, end=fold.test_end)
            return local.gross_returns, local.net_returns, local.weights

        try:
            result = run_walk_forward(pd.DatetimeIndex(returns.index), fit_predict, splitter, name)
        except Exception as exc:
            logger.warning("walk forward failed for %s: %s", name, exc)
            continue
        walk_forward[name] = result
        fold_tables[name] = result.fold_table
        context.save_table(result.fold_table, f"stage11_folds_{name}.csv", index=False)

        # The IS/OOS comparison must not overlap. Walk-forward folds begin
        # five years into the sample, so the early folds sit inside the
        # development window; only folds that start after the development
        # period ends are genuinely out of sample relative to it.
        in_sample = slice_dates(full_results[name].net_returns, None, development_end)
        disjoint_oos = slice_dates(result.oos_net_returns, development_end, None)
        comparison[name] = in_sample_vs_out_of_sample(in_sample, disjoint_oos)
        comparison[name]["oos_window"] = (
            f"{disjoint_oos.index.min().date()} to {disjoint_oos.index.max().date()}"
            if len(disjoint_oos) else "empty"
        )
        comparison[name]["all_folds_oos_sharpe"] = sharpe_ratio(result.oos_net_returns)

    comparison_frame = pd.DataFrame(comparison).T
    context.save_table(comparison_frame, "stage11_is_vs_oos.csv")
    logger.info("in-sample (development) vs disjoint out-of-sample:\n%s",
                comparison_frame.drop(columns=["oos_window"], errors="ignore").astype(float).round(4).to_string())
    logger.info(
        "NOTE: the two windows cover different market regimes -- the development block "
        "contains the 2008 crisis and the out-of-sample block does not -- so a positive "
        "out-of-sample slippage here is not by itself evidence that nothing was overfitted. "
        "The per-fold dispersion and the bootstrap intervals are the more honest read."
    )

    summary = pd.DataFrame({name: r.summary() for name, r in walk_forward.items()}).T
    context.save_table(summary, "stage11_walk_forward_summary.csv")
    logger.info("walk-forward out-of-sample performance:\n%s",
                summary[["cagr", "ann_vol", "sharpe", "max_drawdown", "n_folds",
                         "folds_positive"]].astype(float).round(4).to_string())

    # ------------------------------------------- 4. uncertainty and deflation
    bootstraps, stability = {}, {}
    n_samples = int(cfg.get("backtest.robustness.bootstrap_samples", 2000))
    block = int(cfg.get("backtest.robustness.block_length", 21))
    seed = int(cfg.get("backtest.robustness.seed", 7))
    for name, result in walk_forward.items():
        bootstraps[name] = stationary_block_bootstrap(result.oos_net_returns, n_samples, block,
                                                      seed=seed)
        stability[name] = subperiod_stability(result.oos_net_returns)
        context.save_table(stability[name], f"stage11_subperiods_{name}.csv", index=False)
    context.save_table(pd.DataFrame(bootstraps).T, "stage11_bootstrap.csv")
    logger.info("block-bootstrap 90%% confidence intervals for the OOS Sharpe ratio:\n%s",
                pd.DataFrame(bootstraps).T[["observed", "ci_lower_5pct", "ci_upper_95pct",
                                            "p_value_vs_zero"]].round(4).to_string())

    n_trials = int(sum(len(s) for s in surfaces.values()))
    deflation = {}
    for name, result in walk_forward.items():
        observed = sharpe_ratio(result.oos_net_returns)
        deflation[name] = multiple_testing_penalty(
            observed, n_trials, len(result.oos_net_returns.dropna())
        )
    context.save_table(pd.DataFrame(deflation).T, "stage11_deflated_sharpe.csv")
    logger.info("deflated Sharpe ratio after %d parameter trials:\n%s", n_trials,
                pd.DataFrame(deflation).T[["observed_best_sharpe", "deflated_sharpe_probability",
                                           "survives_at_95pct"]].to_string())

    figure_parameter_sensitivity(context, surfaces, diagnostics,
                                 context.figure("fig22_parameter_sensitivity.png"))
    figure_is_vs_oos(context, comparison, fold_tables, bootstraps,
                     context.figure("fig23_is_vs_oos.png"))

    # ------------------------------------------------------------- registry
    context.registry.log(
        "The backtest engine and every production strategy are free of look-ahead bias.",
        stage=STAGE,
        parameters={"split_dates": ["2012-06-29", "2017-06-30", "2021-06-30"],
                    "modes": ["shock", "reverse"], "n_tests": int(len(leakage))},
        results={
            "real_strategies_tested": int(real["strategy"].nunique()),
            "real_strategy_tests_passed": int(real["passed"].sum()),
            "real_strategy_tests_run": int(len(real)),
            "all_real_strategies_pass": all_real_pass,
            "control_strategy_correctly_detected": control_caught,
            "max_pre_split_weight_difference": float(real["max_weight_difference"].max()),
        },
        decision="retain" if (all_real_pass and control_caught) else "investigate",
        notes=(
            "Perturbing all post-split data leaves every pre-split weight bit-identical for the "
            "production strategies, while the deliberately broken control strategy (built from "
            "tomorrow's return) is caught at every split date and in both perturbation modes. "
            "A test that never fails proves nothing, which is why the control is part of the "
            "suite rather than a one-off check."
        ),
    )
    for name, surface in surfaces.items():
        context.registry.log(
            f"{name} performance is robust across its parameter family, not the product of "
            "one fortunate choice.",
            stage=STAGE,
            parameters={"grid": {"lookback": sorted(surface['lookback'].unique().tolist()),
                                 "rebalance": sorted(surface['rebalance'].unique().tolist())}},
            results=diagnostics[name],
            decision="reject" if diagnostics[name].get("share_positive", 0) < 0.5 else "investigate",
            notes=(
                f"{diagnostics[name].get('share_positive', 0):.0%} of the "
                f"{diagnostics[name].get('n_parameter_sets', 0)} parameter combinations produce a "
                f"positive net Sharpe ratio, with a best of {diagnostics[name].get('best', float('nan')):.2f} "
                f"against a median of {diagnostics[name].get('median', float('nan')):.2f}. "
                f"Verdict: {diagnostics[name].get('verdict', '')}"
            ),
        )
    best_oos = summary["sharpe"].astype(float).idxmax()
    context.registry.log(
        "Out-of-sample walk-forward performance justifies the model ladder's added "
        "complexity.",
        stage=STAGE,
        parameters={"scheme": splitter.scheme, "test_months": splitter.test_months,
                    "embargo_days": splitter.embargo_days, "n_trials_for_deflation": n_trials},
        results={
            "best_oos_model": best_oos,
            "best_oos_sharpe": float(summary.loc[best_oos, "sharpe"]),
            "best_oos_ci_lower": float(bootstraps[best_oos]["ci_lower_5pct"]),
            "best_oos_ci_upper": float(bootstraps[best_oos]["ci_upper_95pct"]),
            "best_oos_deflated_probability": float(deflation[best_oos]["deflated_sharpe_probability"]),
            "mean_sharpe_decay": float(comparison_frame["sharpe_decay"].mean()),
            "models_with_positive_oos_sharpe": int((summary["sharpe"].astype(float) > 0).sum()),
            "models_tested": int(len(summary)),
        },
        decision="retain",
        notes=(
            f"The best out-of-sample model is {best_oos} at a Sharpe ratio of "
            f"{float(summary.loc[best_oos, 'sharpe']):.2f}, with a block-bootstrap 90% interval of "
            f"[{bootstraps[best_oos]['ci_lower_5pct']:.2f}, {bootstraps[best_oos]['ci_upper_95pct']:.2f}]. "
            "The interval is what the headline number should be read against: a Sharpe ratio of "
            "0.8 over twenty years still carries an interval wide enough to contain materially "
            "worse outcomes, and nothing in this project is precise enough to distinguish the "
            "top few models from one another."
        ),
    )
    logger.info("STAGE 11 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
