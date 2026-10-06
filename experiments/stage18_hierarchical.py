"""Stage 18 - Hierarchical risk parity (Generation 2, Priority 11).

Two questions, kept separate because they have different answers:

1. STABILITY. Hierarchical methods claim to avoid the instability of
   covariance inversion. Is that true? Measured directly: resample the return
   window with a block bootstrap, recompute every method's weights, and see
   how far the answer moves.

2. PERFORMANCE. Does HRP / HERC earn a better risk-adjusted return than the
   risk-based allocators already in the ladder? Tested with a PAIRED
   bootstrap on the Sharpe difference, which is far more powerful than
   comparing two marginal confidence intervals when the books overlap.

The paired test is also pointed back at Generation 1, whose report called the
top risk-based models "statistically indistinguishable" on the strength of
overlapping marginal intervals. That was a weaker argument than it sounded.

Figures 36-37.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from src.backtest.engine import BacktestEngine, buy_and_hold
from src.backtest.metrics import performance_summary
from src.portfolio.constraints import Constraints
from src.portfolio.covariance import estimate_covariance
from src.portfolio.hierarchical import (
    correlation_matrix,
    herc_weights,
    hierarchical_linkage,
    hrp_weights,
    quasi_diagonal_order,
)
from src.portfolio.inverse_vol import inverse_volatility_weights
from src.portfolio.mean_variance import minimum_variance_weights
from src.portfolio.risk_parity import risk_parity_weights
from src.utils.dates import DateWindow, slice_dates
from src.utils.plotting import (
    ASSET_CLASS_COLOURS,
    PALETTE,
    bar_with_values,
    new_axes,
    plot_heatmap,
    save_figure,
)
from src.validation.robustness import pairwise_sharpe_tests
from src.validation.walk_forward import WalkForwardSplitter
from experiments.context import build_context
from experiments.strategies import cached_ladder

STAGE = "stage18_hierarchical"
COMPARISON = ["M0_equal_weight", "M1_inverse_vol", "M2_risk_parity", "M9_mean_cvar",
              "M11_hrp", "M12_herc"]


def short(name: str) -> str:
    return {"M0_equal_weight": "M0 EW", "M1_inverse_vol": "M1 inv-vol", "M2_risk_parity": "M2 RP",
            "M9_mean_cvar": "M9 CVaR", "M11_hrp": "M11 HRP", "M12_herc": "M12 HERC",
            "inverse_vol": "inverse vol", "risk_parity": "risk parity",
            "min_variance": "min variance", "hrp": "HRP", "herc": "HERC"}.get(name, name)


def figure_anatomy(context, covariance, asset_class, weights_table, path):
    fig, axes = new_axes(2, 2, figsize=(14.0, 9.0))
    assets = list(covariance.columns)

    link = hierarchical_linkage(covariance, "single")
    scipy_dendrogram(link, labels=assets, ax=axes[0, 0], leaf_rotation=90, leaf_font_size=9,
                     color_threshold=0, above_threshold_color="#0072B2")
    axes[0, 0].set_title("Single-linkage dendrogram on d = sqrt(0.5 (1 - rho))")
    axes[0, 0].set_ylabel("Correlation distance")
    for label in axes[0, 0].get_xticklabels():
        label.set_color(ASSET_CLASS_COLOURS.get(asset_class.get(label.get_text(), ""), "black"))

    corr = correlation_matrix(covariance)
    plot_heatmap(axes[0, 1], corr, "Correlation, original order", annotate=False)
    order = quasi_diagonal_order(link)
    plot_heatmap(axes[1, 0], corr.iloc[order, order], "Quasi-diagonalised: related assets adjacent",
                 annotate=False)

    x = np.arange(len(weights_table))
    width = 0.2
    for i, column in enumerate(weights_table.columns):
        axes[1, 1].bar(x + (i - 1.5) * width, weights_table[column].to_numpy(), width=width,
                       label=short(column), color=PALETTE[i])
    axes[1, 1].set_xticks(x, weights_table.index, rotation=90)
    axes[1, 1].set_ylabel("Weight (before position caps)")
    axes[1, 1].set_title("What each rule would hold")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Figure 36. Anatomy of hierarchical risk parity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "What structure does hierarchical clustering find in a 15-ETF "
                           "universe, and how does that change what is held?", 36)


def figure_results(context, stability, by_sample, curves, tests, path):
    fig, axes = new_axes(2, 3, figsize=(17.0, 9.0))
    order = stability["mean_pairwise_l1"].sort_values().index

    dispersion = stability["mean_pairwise_l1"].reindex(order)
    dispersion.index = [short(i) for i in dispersion.index]
    bar_with_values(axes[0, 0], dispersion, "Weight dispersion under resampling",
                    "mean pairwise L1 (lower = steadier)", "{:.2f}", rotation=25)

    effective = stability["mean_effective_n"].reindex(order)
    effective.index = [short(i) for i in effective.index]
    bar_with_values(axes[0, 1], effective, "Concentration of the book",
                    "effective number of positions", "{:.1f}", rotation=25)

    inflation = stability["risk_inflation"].reindex(order)
    inflation.index = [short(i) for i in inflation.index]
    bar_with_values(axes[0, 2], 100 * inflation, "Risk added by acting on a noisy estimate",
                    "% above the baseline book's volatility", "{:.1f}", rotation=25)

    for i, (name, curve) in enumerate(curves.items()):
        axes[1, 0].plot(curve.index, curve.to_numpy(), color=PALETTE[i % len(PALETTE)],
                        linewidth=1.3, label=short(name))
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Growth of 1 unit (log)")
    axes[1, 0].set_title("Net equity curves, out-of-sample window")
    axes[1, 0].legend(ncol=2, fontsize=8)

    rows = []
    for _, row in tests[(tests["a"] == "M2_risk_parity") | (tests["b"] == "M2_risk_parity")].iterrows():
        other = row["b"] if row["a"] == "M2_risk_parity" else row["a"]
        # Express every pair as (other - risk parity): if risk parity is `a`,
        # the difference and its interval flip sign and the ends swap.
        if row["a"] == "M2_risk_parity":
            diff, lo, hi = -row["difference"], -row["ci_upper_95pct"], -row["ci_lower_5pct"]
        else:
            diff, lo, hi = row["difference"], row["ci_lower_5pct"], row["ci_upper_95pct"]
        rows.append((short(other), diff, lo, hi))
    if rows:
        labels, diff, lo, hi = zip(*rows)
        axes[1, 1].errorbar(range(len(labels)), diff,
                            yerr=[np.array(diff) - np.array(lo), np.array(hi) - np.array(diff)],
                            fmt="o", capsize=5, color="#0072B2", markersize=6)
        axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
        axes[1, 1].set_xticks(range(len(labels)), labels, rotation=20)
        axes[1, 1].set_ylabel("Sharpe difference vs risk parity")
        axes[1, 1].set_title("Paired bootstrap, 90% intervals")

    x = np.arange(len(by_sample))
    for i, column in enumerate(by_sample.columns):
        axes[1, 2].bar(x + (i - 1) * 0.27, by_sample[column].to_numpy(dtype=float), width=0.27,
                       label=column.replace("_", " "), color=PALETTE[i])
    axes[1, 2].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 2].set_xticks(x, [short(n) for n in by_sample.index], rotation=25)
    axes[1, 2].set_ylabel("Net Sharpe")
    axes[1, 2].set_title("By sample block")
    axes[1, 2].set_ylim(top=float(np.nanmax(by_sample.to_numpy(dtype=float))) * 1.3)
    axes[1, 2].legend(fontsize=8, ncol=3, loc="upper center")

    fig.suptitle("Figure 37. Are hierarchical allocators more stable, and do they earn more?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Do HRP and HERC produce more stable weights than minimum variance "
                           "under estimation noise, and does that translate into a statistically "
                           "better risk-adjusted return than risk parity?", 37)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE, generation=2)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 18 | hierarchical risk parity (Generation 2, Priority 11)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    engine = BacktestEngine.from_config(cfg)
    asset_class = cfg.asset_class_map
    node = cfg.get("gen2_portfolio.hierarchical", {}) or {}
    lookback = int(node.get("lookback", 252))

    # ----------------------------------------------------- the latest window
    window = returns.dropna(how="any").tail(lookback)
    covariance = estimate_covariance(window, node.get("covariance", "shrinkage"), lookback)
    weights_table = pd.DataFrame(
        {
            "inverse_vol": inverse_volatility_weights(pd.Series(np.sqrt(np.diag(covariance)),
                                                                index=covariance.columns)),
            "risk_parity": risk_parity_weights(covariance),
            "hrp": hrp_weights(covariance, (node.get("hrp", {}) or {}).get("linkage", "single")),
            "herc": herc_weights(covariance, (node.get("herc", {}) or {}).get("linkage", "ward"),
                                 k_range=tuple((node.get("herc", {}) or {}).get("k_range", [2, 6]))),
        }
    )
    context.save_table(weights_table, "stage18_latest_weights.csv")
    logger.info("weights on the latest window (before position caps):\n%s",
                weights_table.round(3).to_string())
    logger.info("HRP puts %.0f%% in %s: single-linkage bisection lets the lowest-variance "
                "cluster dominate, which is why the ladder applies position caps",
                100 * weights_table["hrp"].max(), weights_table["hrp"].idxmax())

    # ------------------------------------------------ 1. stability experiment
    constraints_loose = Constraints(min_weight=0.0, max_weight=1.0, group_limits={},
                                    group_map={}, net_exposure=1.0)
    methods = {
        "inverse_vol": lambda cov: inverse_volatility_weights(
            pd.Series(np.sqrt(np.diag(cov)), index=cov.columns)),
        "risk_parity": lambda cov: risk_parity_weights(cov),
        "min_variance": lambda cov: minimum_variance_weights(cov, constraints_loose).weights,
        "hrp": lambda cov: hrp_weights(cov, "single"),
        "herc": lambda cov: herc_weights(cov, "ward"),
    }
    from src.portfolio.hierarchical import weight_stability

    stab = node.get("stability", {}) or {}
    clean = returns.dropna(how="any")
    year_ends = [clean.index[clean.index <= pd.Timestamp(f"{y}-12-31")][-1]
                 for y in range(2010, 2026) if (clean.index <= pd.Timestamp(f"{y}-12-31")).any()]
    per_window = []
    for end in year_ends:
        position = clean.index.get_loc(end)
        if position < lookback:
            continue
        sample = clean.iloc[position - lookback + 1:position + 1]
        table = weight_stability(sample, methods, int(stab.get("n_bootstrap", 200)),
                                 int(stab.get("block_length", 21)), int(stab.get("seed", 5)))
        table.insert(0, "window_end", end.date().isoformat())
        per_window.append(table.reset_index())
        logger.info("stability window ending %s done", end.date())
    per_window_frame = pd.concat(per_window, ignore_index=True)
    context.save_table(per_window_frame, "stage18_stability_by_window.csv", index=False)
    stability = per_window_frame.groupby("method")[
        ["mean_pairwise_l1", "mean_turnover_from_baseline", "mean_max_weight",
         "mean_effective_n", "mean_holdings", "mean_oos_vol", "risk_inflation"]].mean()
    context.save_table(stability, "stage18_stability_summary.csv")
    logger.info("stability, averaged over %d windows:\n%s", len(year_ends),
                stability.round(3).to_string())

    pivot = per_window_frame.pivot(index="window_end", columns="method", values="mean_pairwise_l1")
    win_share = {m: float((pivot[m] < pivot["min_variance"]).mean()) for m in ("hrp", "herc", "risk_parity")}
    logger.info("share of windows in which the method is MORE stable than minimum variance: %s",
                {k: round(v, 2) for k, v in win_share.items()})

    # ---------------------------------------------------------- 2. performance
    books = cached_ladder(market, cfg, context.processed, COMPARISON)
    results = {name: engine.run(w, returns, name, market.investable, apply_vol_target=False)
               for name, w in books.items()}
    spy = buy_and_hold(returns, "SPY")
    summary = pd.DataFrame({n: r.summary(benchmark=spy.net_returns) for n, r in results.items()}).T
    context.save_table(summary, "stage18_backtests.csv")
    logger.info("full-sample net performance:\n%s",
                summary[["cagr", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"]]
                .astype(float).round(4).to_string())

    samples = {n: DateWindow.from_config(n, spec or {})
               for n, spec in (cfg.get("backtest.samples", {}) or {}).items()}
    by_sample = {}
    for name, result in results.items():
        by_sample[name] = {s: performance_summary(w.apply(result.net_returns).dropna()).get("sharpe", np.nan)
                           for s, w in samples.items()}
    by_sample_frame = pd.DataFrame(by_sample).T
    context.save_table(by_sample_frame, "stage18_sharpe_by_sample.csv")
    logger.info("Sharpe by sample block:\n%s", by_sample_frame.astype(float).round(3).to_string())

    # Out-of-sample window: the same one Stage 11's walk-forward used.
    first_test = WalkForwardSplitter.from_config(cfg).split(pd.DatetimeIndex(returns.index))[0].test_start
    oos = {name: slice_dates(r.net_returns, first_test, None) for name, r in results.items()}
    bootstrap = cfg.get("gen2_portfolio.dynamic_covariance.bootstrap", {}) or {}
    tests = pairwise_sharpe_tests(oos, fdr=float(cfg.get("regimes.analysis.fdr", 0.10)),
                                  n_samples=int(bootstrap.get("n_samples", 2000)),
                                  block_length=int(bootstrap.get("block_length", 21)),
                                  seed=int(bootstrap.get("seed", 7)))
    context.save_table(tests, "stage18_paired_sharpe_tests.csv", index=False)
    logger.info("paired Sharpe-difference tests, OOS from %s (BH FDR %.2f):\n%s", first_test.date(),
                float(cfg.get("regimes.analysis.fdr", 0.10)),
                tests[["a", "b", "sharpe_a", "sharpe_b", "difference", "return_correlation",
                       "p_value", "bh_significant"]].round(4).to_string(index=False))

    # ---------- the Generation 1 claim, re-examined with the stronger test
    gen1_top = ["M1_inverse_vol", "M2_risk_parity", "M9_mean_cvar", "M0_equal_weight"]
    gen1 = tests[tests["a"].isin(gen1_top) & tests["b"].isin(gen1_top)]
    context.save_table(gen1, "stage18_gen1_claim_recheck.csv", index=False)
    logger.info("Generation 1 said these were 'statistically indistinguishable'; paired test:\n%s",
                gen1[["a", "b", "difference", "p_value", "bh_significant"]].round(4).to_string(index=False))

    # ------------------------------------------------------------- figures
    curves = {name: (1.0 + s).cumprod() for name, s in oos.items()}
    figure_anatomy(context, covariance, asset_class, weights_table,
                   context.figure("fig36_hierarchical_anatomy.png"))
    figure_results(context, stability, by_sample_frame, curves, tests,
                   context.figure("fig37_hierarchical_results.png"))

    # ------------------------------------------------------------- registry
    stability_share = float(np.mean([win_share["hrp"], win_share["herc"]]))
    stability_decision = ("retain" if stability_share >= 0.75
                          else "reject" if stability_share <= 0.25 else "investigate")
    context.registry.log(
        "HRP and HERC produce more stable weights than minimum-variance optimisation "
        "under estimation noise (Lopez de Prado's central claim).",
        stage=STAGE,
        parameters={"windows": len(year_ends), "lookback": lookback,
                    "n_bootstrap": int(stab.get("n_bootstrap", 200)),
                    "block_length": int(stab.get("block_length", 21))},
        results={
            "hrp_dispersion": float(stability.loc["hrp", "mean_pairwise_l1"]),
            "herc_dispersion": float(stability.loc["herc", "mean_pairwise_l1"]),
            "min_variance_dispersion": float(stability.loc["min_variance", "mean_pairwise_l1"]),
            "risk_parity_dispersion": float(stability.loc["risk_parity", "mean_pairwise_l1"]),
            "share_windows_hrp_more_stable": win_share["hrp"],
            "share_windows_herc_more_stable": win_share["herc"],
            "hrp_effective_n": float(stability.loc["hrp", "mean_effective_n"]),
            "herc_effective_n": float(stability.loc["herc", "mean_effective_n"]),
            "min_variance_effective_n": float(stability.loc["min_variance", "mean_effective_n"]),
            "min_variance_max_weight": float(stability.loc["min_variance", "mean_max_weight"]),
            "min_variance_holdings": float(stability.loc["min_variance", "mean_holdings"]),
            "posthoc_risk_inflation_hrp": float(stability.loc["hrp", "risk_inflation"]),
            "posthoc_risk_inflation_herc": float(stability.loc["herc", "risk_inflation"]),
            "posthoc_risk_inflation_min_variance": float(stability.loc["min_variance", "risk_inflation"]),
            "posthoc_risk_inflation_risk_parity": float(stability.loc["risk_parity", "risk_inflation"]),
        },
        decision=stability_decision,
        notes=(
            "Decision rule fixed in advance on weight dispersion: retain if the hierarchical "
            "methods are more stable than minimum variance in at least 75% of windows, reject "
            "if in at most 25%. The verdict follows that rule. It does NOT mean minimum variance "
            "is better behaved: it holds "
            f"{float(stability.loc['min_variance', 'mean_holdings']):.1f} assets with a "
            f"{float(stability.loc['min_variance', 'mean_max_weight']):.0%} maximum weight, so its "
            "weights cannot move -- the Stage 7 lesson that a book can be 'stable' by sitting in a "
            "corner. The risk-inflation columns are POST-HOC (added after seeing the dispersion "
            "result) and are reported as a diagnostic, not as grounds to overturn the declared "
            "decision."
        ),
    )

    reference = "M2_risk_parity"
    hierarchical = tests[(tests["a"].isin(["M11_hrp", "M12_herc"]) | tests["b"].isin(["M11_hrp", "M12_herc"]))
                         & (tests["a"].eq(reference) | tests["b"].eq(reference))]
    any_better = False
    for _, row in hierarchical.iterrows():
        other = row["b"] if row["a"] == reference else row["a"]
        gap = -row["difference"] if row["a"] == reference else row["difference"]
        if gap > 0 and bool(row["bh_significant"]):
            any_better = True
    context.registry.log(
        "HRP / HERC deliver a statistically better net Sharpe ratio than risk parity "
        "out of sample.",
        stage=STAGE,
        parameters={"oos_start": first_test.date().isoformat(), "covariance": node.get("covariance"),
                    "n_models_compared": len(oos)},
        cost_bps=float(cfg.get("backtest.costs.cost_bps", 10.0)),
        results={
            "hrp_sharpe": float(summary.loc["M11_hrp", "sharpe"]),
            "herc_sharpe": float(summary.loc["M12_herc", "sharpe"]),
            "risk_parity_sharpe": float(summary.loc["M2_risk_parity", "sharpe"]),
            "inverse_vol_sharpe": float(summary.loc["M1_inverse_vol", "sharpe"]),
            "any_hierarchical_significantly_better_than_rp": any_better,
            "gen1_top3_pairs_bh_significant": int(gen1["bh_significant"].sum()),
            "gen1_top3_pairs_tested": int(len(gen1)),
        },
        decision="retain" if any_better else "reject",
        notes=(
            "Judged by the paired bootstrap with Benjamini-Hochberg control across all "
            f"{len(tests)} model pairs. 'Better' requires a significant difference in the "
            "right direction; a higher point estimate is not enough."
        ),
    )
    logger.info("STAGE 18 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
