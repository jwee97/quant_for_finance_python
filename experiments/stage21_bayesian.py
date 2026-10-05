"""Stage 21 - Bayesian portfolio construction (Generation 3, Priority 3).

Plug-in mean-variance optimisation maximises over its own estimation error.
This stage treats (mu, Sigma) as uncertain (a conjugate normal-inverse-Wishart
posterior, prior mean by empirical Bayes), makes the portfolio weights a
distribution, and asks three questions whose rules are in ``config/bayes.yaml``
and were committed before any result:

1. Does Bayesian construction beat the same inputs treated as certain, and beat
   the Generation 1 winner that estimates no mean at all (risk parity)?
2. Are the posterior-averaged weights more stable under estimation noise?
3. (Descriptive.) Is the posterior predictive distribution of next month's
   return calibrated?

Figures 42-43.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import performance_summary
from src.portfolio.bayesian import (
    bayesian_book,
    bayesian_weights,
    niw_posterior,
    predictive_quantiles,
    weight_distribution,
    weight_stability_bayes,
)
from src.portfolio.constraints import Constraints
from src.utils.dates import DateWindow, rebalance_dates
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context
from experiments.strategies import cached_ladder

STAGE = "stage21_bayesian"
KINDS = ["mvo_sample", "bayes_stein", "bayes_predictive"]
REFERENCES = ["M0_equal_weight", "M1_inverse_vol", "M2_risk_parity", "M7_combined_alpha_shrinkage_mvo"]
SHORT = {"mvo_sample": "plug-in MVO", "bayes_stein": "Bayes-Stein", "bayes_predictive": "Bayes predictive",
         "M0_equal_weight": "equal weight", "M1_inverse_vol": "inverse vol", "M2_risk_parity": "risk parity",
         "M7_combined_alpha_shrinkage_mvo": "signal MVO (M7)"}
COLOUR = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(list(SHORT))}


def bayes_fingerprint(cfg) -> str:
    """Identity of what the Bayesian books depend on: the core configuration and the ``bayes`` namespace only."""
    return cfg.fingerprint(["universe", "data", "strategies", "portfolio", "risk", "backtest", "bayes"])


def cached_bayes_books(market, cfg, directory: Path, node: dict, constraints: Constraints) -> dict[str, pd.DataFrame]:
    """The three Bayesian books, cached on the Generation 3 fingerprint and the data version."""
    directory = Path(directory) / "books_gen3"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = directory / "bayes_cache_key.txt"
    key = f"{bayes_fingerprint(cfg)}|{market.data_version}"
    files = {k: directory / f"bayes_{k}.csv" for k in KINDS}
    if stamp.exists() and stamp.read_text() == key and all(f.exists() for f in files.values()):
        return {k: pd.read_csv(f, index_col=0, parse_dates=True) for k, f in files.items()}
    est = node.get("estimation", {}) or {}
    marks = rebalance_dates(pd.DatetimeIndex(market.returns().index), str(est.get("rebalance", "monthly")))
    post = node.get("posterior", {}) or {}
    books = {}
    for kind in KINDS:
        books[kind] = bayesian_book(
            market.returns(), market.investable, kind, constraints, marks,
            lookback=int(est.get("lookback_days", 252)), min_assets=int(cfg.get("backtest.engine.min_assets", 5)),
            risk_aversion=float(est.get("risk_aversion", 5.0)), nu0=float((node.get("prior", {}) or {}).get("nu0_days", 126)),
            n_draws=int(post.get("draws", 100)), seed=int(post.get("seed", 11)))
        books[kind].to_csv(files[kind])
    stamp.write_text(key)
    return books


def stability_by_window(returns: pd.DataFrame, constraints: Constraints, node: dict, logger, draws_in_bootstrap: int = 20):
    stab = node.get("stability", {}) or {}
    est = node.get("estimation", {}) or {}
    lookback = int(est.get("lookback_days", 252))
    ra = float(est.get("risk_aversion", 5.0))
    nu0 = float((node.get("prior", {}) or {}).get("nu0_days", 126))
    methods = {
        "mvo_sample": lambda r: bayesian_weights(r, "mvo_sample", constraints, ra),
        "bayes_stein": lambda r: bayesian_weights(r, "bayes_stein", constraints, ra, nu0),
        "bayes_predictive": lambda r: bayesian_weights(r, "bayes_predictive", constraints, ra, nu0, draws_in_bootstrap, 3),
    }
    clean = returns.dropna(how="any")
    ends = [clean.index[clean.index <= pd.Timestamp(f"{y}-12-31")][-1] for y in range(2010, 2026)
            if (clean.index <= pd.Timestamp(f"{y}-12-31")).any()]
    rows = []
    for end in ends:
        position = clean.index.get_loc(end)
        if position < lookback:
            continue
        table = weight_stability_bayes(clean.iloc[position - lookback + 1:position + 1], methods, constraints,
                                       int(stab.get("n_bootstrap", 60)), int(stab.get("block_length", 21)),
                                       int(stab.get("seed", 5)), ra)
        table.insert(0, "window_end", end.date().isoformat())
        rows.append(table.reset_index().rename(columns={"index": "method"}))
        logger.info("stability window ending %s done", end.date())
    return pd.concat(rows, ignore_index=True)


def predictive_coverage(returns: pd.DataFrame, book: pd.DataFrame, node: dict, horizon: int = 21) -> pd.DataFrame:
    """At each month-end, the book's predictive interval against the return its weights then earned."""
    est = node.get("estimation", {}) or {}
    lookback = int(est.get("lookback_days", 252))
    levels = tuple((node.get("predictive_check", {}) or {}).get("levels", [0.5, 0.9]))
    nu0 = float((node.get("prior", {}) or {}).get("nu0_days", 126))
    index = pd.DatetimeIndex(returns.index)
    rows = []
    for stamp in rebalance_dates(index, "monthly"):
        position = index.get_loc(stamp)
        if position < lookback or position + horizon >= len(index) or stamp not in book.index:
            continue
        weights = book.loc[stamp]
        if weights.abs().sum() < 0.5:
            continue
        window = returns.iloc[position - lookback + 1:position + 1].dropna(axis=1, how="any")
        w = weights.reindex(window.columns).fillna(0.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        q = predictive_quantiles(window, w, horizon, levels, nu0, seed=position)
        future = returns.iloc[position + 1:position + 1 + horizon][window.columns].fillna(0.0)
        realised = float(((1.0 + future).prod() - 1.0) @ w)
        row = {"date": stamp, "realised": realised, "median": q["median"]}
        for level in levels:
            row[f"covered_{level}"] = bool(q[f"lo_{level}"] <= realised <= q[f"hi_{level}"])
            row[f"width_{level}"] = q[f"hi_{level}"] - q[f"lo_{level}"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def figure_distributions(dist: pd.DataFrame, stability: pd.DataFrame, win_share: float, path):
    fig, axes = new_axes(1, 3, figsize=(16.5, 5.2))
    ax = axes[0]
    order = dist.sort_values("mean", ascending=True)
    y = np.arange(len(order))
    ax.barh(y, order["mean"] * 100, color=PALETTE[0], alpha=0.8, label="posterior mean")
    ax.hlines(y, order["p05"] * 100, order["p95"] * 100, color="black", linewidth=1.4, label="5th to 95th percentile")
    ax.set_yticks(y, order.index)
    ax.set_xlabel("Weight, %")
    ax.set_title("Weights are a distribution\n(latest window)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    pivot = stability.pivot(index="window_end", columns="method", values="mean_pairwise_l1")
    for i, m in enumerate(KINDS):
        ax.plot(pd.to_datetime(pivot.index), pivot[m], marker="o", markersize=3, color=COLOUR[m], label=SHORT[m])
    ax.set_ylabel("Mean pairwise L1 distance between bootstrap weights")
    ax.set_title(f"Dispersion under estimation noise\n(Bayes predictive lower in {100 * win_share:.0f}% of windows)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    eff = stability.pivot(index="window_end", columns="method", values="mean_effective_n")
    ax.bar(np.arange(len(KINDS)), [eff[m].mean() for m in KINDS], color=[COLOUR[m] for m in KINDS])
    ax.set_xticks(np.arange(len(KINDS)), [SHORT[m] for m in KINDS], rotation=15)
    ax.set_ylabel("Effective number of positions")
    ax.set_title("Concentration\n(low dispersion can be a corner)", fontsize=10)
    fig.suptitle("Figure 42. Bayesian weights: how uncertain, and how stable", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "How uncertain is the optimal allocation once parameter uncertainty is acknowledged, and are "
                           "posterior-averaged weights more stable than plug-in ones under estimation noise?", 42)


def figure_results(curves: dict, summary: pd.DataFrame, tests: pd.DataFrame, coverage: pd.DataFrame, levels, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    ax = axes[0, 0]
    for name, s in curves.items():
        ax.plot(s.index, s.to_numpy(), color=COLOUR[name], linewidth=1.3, label=SHORT[name])
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (net, log scale)")
    ax.set_title("Net cumulative return, common window")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    names = list(summary.index)
    ax.bar(np.arange(len(names)), summary["sharpe"].astype(float), color=[COLOUR[n] for n in names])
    ax.set_xticks(np.arange(len(names)), [SHORT[n] for n in names], rotation=25, ha="right")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Net Sharpe ratio")
    ax.axhline(0, color="black", linewidth=0.8)

    ax = axes[1, 0]
    y = np.arange(len(tests))
    ax.errorbar(tests["difference"], y, xerr=[tests["difference"] - tests["ci_lower_5pct"], tests["ci_upper_95pct"] - tests["difference"]],
                fmt="o", color="black", capsize=3)
    for yi, (_, r) in zip(y, tests.iterrows()):
        ax.text(r["ci_upper_95pct"], yi, f"  p={r['p_value']:.2f}" + ("  *" if r["bh_significant"] else ""), va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [f"{SHORT[a]} vs {SHORT[b]}" for a, b in zip(tests["a"], tests["b"])], fontsize=8)
    ax.set_xlabel("Net Sharpe difference (90% interval, paired block bootstrap)")
    ax.set_title("The pre-declared family of four")

    ax = axes[1, 1]
    rates = [float(coverage[f"covered_{lv}"].mean()) for lv in levels]
    ax.bar(np.arange(len(levels)), rates, color=PALETTE[0], width=0.5, label="achieved")
    ax.plot(np.arange(len(levels)), levels, "k_", markersize=40, markeredgewidth=2, label="nominal")
    ax.set_xticks(np.arange(len(levels)), [f"{int(100 * lv)}% interval" for lv in levels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of months the realised return fell inside")
    ax.set_title("Posterior predictive calibration of the Bayes predictive book")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Figure 43. Does Bayesian construction earn more, and is its predictive distribution honest?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does treating mean and covariance as uncertain earn a better net Sharpe than treating them as "
                           "known or than risk parity, and does the posterior predictive distribution cover realised returns "
                           "at its nominal rates?", 43)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 21: Bayesian portfolio construction").parse_args(argv)
    context, logger = build_context(STAGE, generation=3)
    cfg = context.config
    node = cfg.get("bayes", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 21 | Bayesian portfolio construction (Generation 3, Priority 3)")
    logger.info("=" * 72)
    market = context.market_data()
    returns = market.returns()
    constraints = Constraints.from_config(cfg)
    engine = BacktestEngine.from_config(cfg)
    decisions = node.get("decisions", {}) or {}
    fdr = float(decisions.get("fdr", 0.10))
    boot = decisions.get("bootstrap", {}) or {}

    # ------------------------------------------------------------------ books
    bayes_books = cached_bayes_books(market, cfg, context.processed, node, constraints)
    ref_books = cached_ladder(market, cfg, context.processed, REFERENCES)
    books = {**bayes_books, **ref_books}
    runs = {n: engine.run(w, returns, n, market.investable, apply_vol_target=False) for n, w in books.items()}
    start = max(r.net_returns.dropna().index[np.flatnonzero(books[n].abs().sum(axis=1).to_numpy() > 0.5)[0]] for n, r in runs.items())
    common = {n: r.net_returns.loc[start:].dropna() for n, r in runs.items()}
    logger.info("common window from %s (%d days)", start.date(), len(next(iter(common.values()))))
    summary = pd.DataFrame({n: performance_summary(s) for n, s in common.items()}).T
    turnover = pd.Series({n: runs[n].summary().get("ann_turnover", np.nan) for n in runs})
    summary["ann_turnover"] = turnover
    context.save_table(summary, "stage21_performance.csv")
    logger.info("net performance, common window:\n%s",
                summary[["cagr", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"]].astype(float).round(4).to_string())
    samples = {n: DateWindow.from_config(n, spec or {}) for n, spec in (cfg.get("backtest.samples", {}) or {}).items()}
    by_sample = pd.DataFrame({n: {s: performance_summary(w.apply(r).dropna()).get("sharpe", np.nan) for s, w in samples.items()}
                              for n, r in common.items()}).T
    context.save_table(by_sample, "stage21_sharpe_by_sample.csv")

    # ------------------------------------------------- the pre-declared family
    pairs = [("bayes_predictive", "mvo_sample"), ("bayes_stein", "mvo_sample"),
             ("bayes_predictive", "M2_risk_parity"), ("bayes_stein", "M2_risk_parity")]
    rows = []
    for a, b in pairs:
        t = paired_sharpe_test(common[a], common[b], n_samples=int(boot.get("n_samples", 2000)),
                               block_length=int(boot.get("block_length", 21)), seed=int(boot.get("seed", 7)))
        rows.append({"a": a, "b": b, **t})
    tests = pd.DataFrame(rows)
    tests["bh_significant"] = benjamini_hochberg(tests["p_value"], fdr)
    tests["passes"] = tests["bh_significant"] & (tests["difference"] > 0)
    context.save_table(tests, "stage21_sharpe_tests.csv", index=False)
    logger.info("pre-declared family (BH FDR %.2f):\n%s", fdr,
                tests[["a", "b", "sharpe_a", "sharpe_b", "difference", "p_value", "bh_significant", "passes"]].round(4).to_string(index=False))
    extra = []
    for b in ("M0_equal_weight", "M1_inverse_vol", "M7_combined_alpha_shrinkage_mvo"):
        t = paired_sharpe_test(common["bayes_predictive"], common[b], n_samples=int(boot.get("n_samples", 2000)),
                               block_length=int(boot.get("block_length", 21)), seed=int(boot.get("seed", 7)))
        extra.append({"a": "bayes_predictive", "b": b, **t})
    context.save_table(pd.DataFrame(extra), "stage21_sharpe_tests_reported_not_judged.csv", index=False)

    # -------------------------------------------------------------- stability
    stability = stability_by_window(returns, constraints, node, logger)
    context.save_table(stability, "stage21_stability_by_window.csv", index=False)
    pivot = stability.pivot(index="window_end", columns="method", values="mean_pairwise_l1")
    win_share = float((pivot["bayes_predictive"] < pivot["mvo_sample"]).mean())
    summary_stab = stability.groupby("method")[["mean_pairwise_l1", "mean_max_weight", "mean_effective_n"]].mean()
    context.save_table(summary_stab, "stage21_stability_summary.csv")
    logger.info("stability averaged over windows:\n%s\nBayes predictive more stable than plug-in in %.0f%% of windows",
                summary_stab.round(3).to_string(), 100 * win_share)

    # --------------------------------------------- weights as a distribution
    latest = returns.dropna(how="any").tail(int((node.get("estimation", {}) or {}).get("lookback_days", 252)))
    dist = weight_distribution(latest, constraints, float((node.get("estimation", {}) or {}).get("risk_aversion", 5.0)),
                               float((node.get("prior", {}) or {}).get("nu0_days", 126)), 200,
                               int((node.get("posterior", {}) or {}).get("seed", 11)))
    context.save_table(dist, "stage21_latest_weight_distribution.csv")
    post = niw_posterior(latest, float((node.get("prior", {}) or {}).get("nu0_days", 126)))
    logger.info("latest window: Jorion prior weight %.3f (kappa0 %.0f days)", post.prior_weight, post.kappa0)

    # ------------------------------------------------------ predictive check
    coverage = predictive_coverage(returns, bayes_books["bayes_predictive"], node)
    context.save_table(coverage, "stage21_predictive_coverage.csv")
    levels = tuple((node.get("predictive_check", {}) or {}).get("levels", [0.5, 0.9]))
    achieved = {lv: float(coverage[f"covered_{lv}"].mean()) for lv in levels}
    logger.info("predictive coverage over %d months: %s", len(coverage), {k: round(v, 3) for k, v in achieved.items()})

    # --------------------------------------------------------------- figures
    curves = {n: (1.0 + s).cumprod() for n, s in common.items()}
    figure_distributions(dist, stability, win_share, context.figure("fig42_bayesian_weights.png"))
    figure_results(curves, summary[["sharpe"]], tests, coverage, levels, context.figure("fig43_bayesian_results.png"))

    # -------------------------------------------------------------- registry
    bp = tests[(tests.a == "bayes_predictive")]
    h_bayes = bool(bp["passes"].all())
    context.registry.log(
        "Bayesian portfolio construction improves the allocation: the posterior-predictive book beats the same "
        "inputs treated as certain AND the Generation 1 risk-parity book, after costs.",
        stage=STAGE, parameters={"prior": node.get("prior"), "draws": (node.get("posterior") or {}).get("draws"),
                                 "window_start": str(start.date()), "test": "paired stationary bootstrap on net Sharpe, BH across four"},
        results={"sharpe_bayes_predictive": float(summary.loc["bayes_predictive", "sharpe"]),
                 "sharpe_bayes_stein": float(summary.loc["bayes_stein", "sharpe"]),
                 "sharpe_mvo_sample": float(summary.loc["mvo_sample", "sharpe"]),
                 "sharpe_risk_parity": float(summary.loc["M2_risk_parity", "sharpe"]),
                 **{f"diff_{a}_vs_{b}": float(d) for a, b, d in zip(tests.a, tests.b, tests.difference)},
                 **{f"p_{a}_vs_{b}": float(p) for a, b, p in zip(tests.a, tests.b, tests.p_value)}},
        decision="retain" if h_bayes else "reject",
        test_period=f"{start.date()} onward, net of costs",
        notes=("Retained only if bayes_predictive beats BOTH comparators with a positive, BH-significant paired Sharpe "
               f"difference. Jorion's empirical-Bayes prior weight on the latest window is {post.prior_weight:.2f}. "
               "Reported, not judged: comparisons with equal weight, inverse volatility and the signal-driven M7."),
    )
    stab_decision = "retain" if win_share >= 0.75 else "reject"
    context.registry.log(
        "Posterior-averaged weights are more stable than plug-in mean-variance weights under estimation noise.",
        stage=STAGE, parameters={"windows": int(pivot.shape[0]), "n_bootstrap": (node.get("stability") or {}).get("n_bootstrap"),
                                 "draws_in_bootstrap": 20},
        results={"share_windows_bayes_more_stable": win_share,
                 "dispersion_bayes_predictive": float(summary_stab.loc["bayes_predictive", "mean_pairwise_l1"]),
                 "dispersion_mvo_sample": float(summary_stab.loc["mvo_sample", "mean_pairwise_l1"]),
                 "effective_n_bayes_predictive": float(summary_stab.loc["bayes_predictive", "mean_effective_n"]),
                 "effective_n_mvo_sample": float(summary_stab.loc["mvo_sample", "mean_effective_n"])},
        decision=stab_decision,
        notes=("Rule: retain if more stable in at least 75% of windows, reject otherwise (the 'inconclusive' band is "
               "recorded as reject). The bootstrap uses 20 posterior draws per resample (compute); the books use 100."),
    )
    context.registry.log(
        "Descriptive: does the posterior predictive distribution of next month's book return cover realised returns at its nominal levels?",
        stage=STAGE, parameters={"levels": list(levels)},
        results={**{f"coverage_{lv}": achieved[lv] for lv in levels}, "n_months": int(len(coverage))},
        decision="record",
        notes="Calibration check of the model, not a decision. Daily Gaussian returns scaled to 21 days understate fat tails and volatility clustering.",
    )
    logger.info("STAGE 21 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
