"""Stage 24 - Alpha-combination engine (Generation 3, Priority 13).

Momentum, mean reversion, PCA statistical arbitrage and the Stage 19 ridge forecast
are standardised onto one scale, described by the five things the roadmap says
every alpha should carry (expected return, confidence, turnover, decay,
correlation), and combined at the FORECAST level under four rules for how much
to trust each. The pre-declared question (``config/combination.yaml``): does
the cost-aware rule beat equal weighting and the best Generation 1 alpha after
costs?

Figures 48-49.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import deflated_sharpe_ratio, performance_summary
from src.models.online import hedge_aggregate
from src.signals.alpha_engine import (
    annual_turnover, combine_alphas, forward_returns, ic_decay, matured_ic, standardise_alpha, trailing_ir,
    trailing_sharpe, trust_weights)
from src.signals.pca_strategy import pca_stat_arb_signal
from src.signals.transform import signal_to_positions
from src.features.volatility import rolling_volatility
from src.utils.dates import rebalance_dates
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context
from experiments.strategies import mean_reversion_signal, momentum_signal, transform_config

STAGE = "stage24_combination"
ALPHAS = ["momentum", "mean_reversion", "pca_residual", "ridge_forecast"]
COMBOS = ["equal", "ic_weighted", "cost_aware", "hedge"]
LABEL = {"momentum": "momentum", "mean_reversion": "mean reversion", "pca_residual": "PCA residual", "ridge_forecast": "ridge forecast",
         "equal": "equal weight", "ic_weighted": "IC-weighted", "cost_aware": "cost-aware", "hedge": "Hedge"}
COLOUR = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(ALPHAS + COMBOS)}


def ridge_signal(context, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    path = context.tables / "stage19_predictions_price_only.csv"
    if not path.exists():
        raise FileNotFoundError("Stage 19 must run first: its ridge forecasts are one of the alphas")
    pred = pd.read_csv(path, parse_dates=["origin"])
    wide = pred.pivot(index="origin", columns="asset", values="mu").reindex(columns=columns)
    # a forecast made at a month-end is held until the next one; nothing before the first origin
    return wide.reindex(index.union(wide.index)).ffill().reindex(index)           # NaN before the first origin


def figure_descriptors(decay: pd.DataFrame, table: pd.DataFrame, signal_corr: pd.DataFrame, book_corr: pd.DataFrame, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    ax = axes[0, 0]
    for a in decay.columns:
        ax.plot([str(h) for h in decay.index], decay[a], marker="o", color=COLOUR[a], label=LABEL[a])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Horizon after the signal, days")
    ax.set_ylabel("Mean rank IC")
    ax.set_title("Decay: where each alpha's information lives")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for a, r in table.iterrows():
        ax.scatter(r["ann_turnover"], r["net_sharpe"], s=90, color=COLOUR[a])
        ax.annotate(LABEL[a], (r["ann_turnover"], r["net_sharpe"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Annual turnover of the standalone book (x)")
    ax.set_ylabel("Net Sharpe of the standalone book")
    ax.set_title("What each alpha pays for its trading")
    for ax, mat, title in ((axes[1, 0], signal_corr, "Correlation of the standardised signals"),
                           (axes[1, 1], book_corr, "Correlation of the standalone books' net returns")):
        im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(mat)), [LABEL[c] for c in mat.columns], rotation=25, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(mat)), [LABEL[c] for c in mat.index], fontsize=8)
        for i in range(len(mat)):
            for j in range(len(mat)):
                ax.text(j, i, f"{mat.iat[i, j]:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(mat.iat[i, j]) > 0.7 else "black")
        ax.set_title(title)
    fig.suptitle("Figure 48. What each alpha is worth: decay, cost, correlation", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "For each alpha, where does its information live (decay), what does its trading cost, and how "
                           "correlated are the alphas as forecasts and as books?", 48)


def figure_results(curves: dict, weights: dict, tests: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(16.5, 5.2))
    ax = axes[0]
    for name, s in curves.items():
        ax.plot(s.index, s.to_numpy(), color=COLOUR[name], linewidth=1.3 if name in COMBOS else 0.9, linestyle="-" if name in COMBOS else "--", label=LABEL[name])
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (net, log scale)")
    ax.set_title("Combinations (solid) and standalone alphas (dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    w = weights["cost_aware"].resample("ME").last()
    ax.stackplot(w.index, [w[c].to_numpy() for c in w.columns], labels=[LABEL[c] for c in w.columns], colors=[COLOUR[c] for c in w.columns])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Trust weight")
    ax.set_title("What the cost-aware rule trusts")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[2]
    y = np.arange(len(tests))
    ax.errorbar(tests["difference"], y, xerr=[tests["difference"] - tests["ci_lower_5pct"], tests["ci_upper_95pct"] - tests["difference"]], fmt="o", color="black", capsize=3)
    for yi, (_, r) in zip(y, tests.iterrows()):
        ax.text(r["ci_upper_95pct"], yi, f"  p={r['p_value']:.2f}" + ("  *" if r["bh_significant"] else ""), va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [f"{LABEL[a]} vs {LABEL[b]}" for a, b in zip(tests["a"], tests["b"])], fontsize=8)
    ax.set_xlabel("Net Sharpe difference (90% interval)")
    ax.set_title("The pre-declared family of two")
    fig.suptitle("Figure 49. Does optimising trust in each alpha beat equal weight and the best single alpha?", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does a cost-aware rule for how much to trust each alpha, applied to the combined forecast, beat equal "
                           "weighting and the best Generation 1 alpha net of costs?", 49)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 24: alpha combination").parse_args(argv)
    context, logger = build_context(STAGE, generation=3)
    cfg = context.config
    node = cfg.get("combination", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 24 | alpha-combination engine (Generation 3, Priority 13)")
    logger.info("=" * 72)
    decisions = node.get("decisions", {}) or {}
    fdr = float(decisions.get("fdr", 0.10))
    boot = decisions.get("bootstrap", {}) or {}
    market = context.market_data()
    returns, investable = market.returns(), market.investable
    index = pd.DatetimeIndex(returns.index)
    engine = BacktestEngine.from_config(cfg)
    volatility = rolling_volatility(returns, int(cfg.get("portfolio.volatility.lookback", 63)))
    transform = transform_config(cfg)
    comb = node.get("combinations", {}) or {}
    window = 504                                   # trailing days, as declared
    min_hist = int(comb.get("min_history_days", 252))

    raw = {"momentum": momentum_signal(market, cfg), "mean_reversion": mean_reversion_signal(market, cfg),
           "pca_residual": pca_stat_arb_signal(returns, 3, 252, 21).where(investable),
           "ridge_forecast": ridge_signal(context, index, returns.columns).where(investable)}
    z = {k: standardise_alpha(v, investable) for k, v in raw.items()}
    first_date = ridge_first = z["ridge_forecast"].dropna(how="all").index[0]
    logger.info("first date every alpha exists: %s", first_date.date())

    # ----------------------------------------------------- standalone books
    def book(signal):
        return signal_to_positions(signal, volatility, investable=investable, **transform)
    runs = {}
    for name in ALPHAS:
        runs[name] = engine.run(book(raw[name]), returns, name, investable, apply_vol_target=True)
    alpha_net = pd.DataFrame({n: r.net_returns for n, r in runs.items()})

    # --------------------------------------------------------- trust scores
    updates = rebalance_dates(index, "monthly")
    fwd21 = forward_returns(returns, 21)
    ic_ir = pd.DataFrame({n: trailing_ir(matured_ic(z[n], fwd21, 21), window, 126) for n in ALPHAS})
    net_sharpe = pd.DataFrame({n: trailing_sharpe(alpha_net[n], window, min_hist) for n in ALPHAS})
    hedge = hedge_aggregate(alpha_net.loc[first_date:].dropna(how="any"))
    hedge_scores = hedge.weights.reindex(index).ffill()
    weights = {
        "equal": pd.DataFrame(1.0 / len(ALPHAS), index, ALPHAS),
        "ic_weighted": trust_weights(ic_ir, updates, shrink=0.0),
        "cost_aware": trust_weights(net_sharpe, updates, shrink=0.5),
        "hedge": trust_weights(hedge_scores, updates, shrink=0.0),
    }
    for k, w in weights.items():
        context.save_table(w.resample("ME").last(), f"stage24_weights_{k}.csv")

    # ------------------------------------------------------ combined books
    combo_runs = {}
    for name, w in weights.items():
        signal = combine_alphas(z, w)
        combo_runs[name] = engine.run(book(signal), returns, name, investable, apply_vol_target=True)
    all_runs = {**runs, **combo_runs}
    saved = context.processed / "stage24_runs"                     # read by Stage 25 (descriptive re-costing); not used here
    saved.mkdir(parents=True, exist_ok=True)
    for name, run in combo_runs.items():
        run.gross_returns.to_frame("gross").to_csv(saved / f"{name}_gross.csv", index_label="date")
        run.trades.to_csv(saved / f"{name}_trades.csv", index_label="date")
    common = {n: r.net_returns.loc[first_date:].dropna() for n, r in all_runs.items()}
    perf = pd.DataFrame({n: performance_summary(s) for n, s in common.items()}).T
    perf["ann_turnover"] = pd.Series({n: all_runs[n].summary().get("ann_turnover", np.nan) for n in all_runs})
    n_obs = len(next(iter(common.values())))
    perf["deflated_sharpe_probability"] = [deflated_sharpe_ratio(float(perf.loc[n, "sharpe"]), len(all_runs), n_obs) for n in perf.index]
    context.save_table(perf, "stage24_performance.csv")
    logger.info("net performance from %s:\n%s", first_date.date(),
                perf[["cagr", "ann_vol", "sharpe", "max_drawdown", "ann_turnover", "deflated_sharpe_probability"]].astype(float).round(4).to_string())

    # ------------------------------------------------------------ the family
    pairs = [("cost_aware", "equal"), ("cost_aware", "momentum")]
    rows = [{"a": a, "b": b, **paired_sharpe_test(common[a], common[b], n_samples=int(boot.get("n_samples", 2000)),
                                                   block_length=int(boot.get("block_length", 21)), seed=int(boot.get("seed", 7)))}
            for a, b in pairs]
    tests = pd.DataFrame(rows)
    tests["bh_significant"] = benjamini_hochberg(tests["p_value"], fdr)
    tests["passes"] = tests["bh_significant"] & (tests["difference"] > 0)
    context.save_table(tests, "stage24_tests.csv", index=False)
    extra = [{"a": a, "b": b, **paired_sharpe_test(common[a], common[b], n_samples=int(boot.get("n_samples", 2000)),
                                                    block_length=int(boot.get("block_length", 21)), seed=int(boot.get("seed", 7)))}
             for a, b in (("ic_weighted", "equal"), ("hedge", "equal"), ("equal", "momentum"), ("cost_aware", "ic_weighted"))]
    context.save_table(pd.DataFrame(extra), "stage24_tests_reported_not_judged.csv", index=False)
    logger.info("pre-declared family (BH FDR %.2f):\n%s", fdr,
                tests[["a", "b", "sharpe_a", "sharpe_b", "difference", "p_value", "bh_significant", "passes"]].round(4).to_string(index=False))

    # --------------------------------------------------------- descriptors
    ev = slice(first_date, None)
    desc = {}
    for n in ALPHAS:
        ic = matured_ic(z[n], fwd21, 21).loc[ev].dropna()
        desc[n] = {"mean_ic_21d": float(ic.mean()), "ic_information_ratio": float(ic.mean() / ic.std(ddof=1)),
                   "implied_annual_ir": float(ic.mean() * np.sqrt(15 * 12)), "ann_turnover": float(perf.loc[n, "ann_turnover"]),
                   "net_sharpe": float(perf.loc[n, "sharpe"]), "signal_turnover_x": annual_turnover(runs[n].weights)}
    table = pd.DataFrame(desc).T
    decay = pd.DataFrame({n: ic_decay(z[n].loc[ev], returns) for n in ALPHAS})
    flat = {n: z[n].loc[ev].stack() for n in ALPHAS}
    signal_corr = pd.DataFrame(flat).corr()
    book_corr = pd.DataFrame({n: common[n] for n in ALPHAS}).corr()
    context.save_table(table, "stage24_alpha_descriptors.csv")
    context.save_table(decay, "stage24_ic_decay.csv")
    context.save_table(signal_corr, "stage24_signal_correlation.csv")
    context.save_table(book_corr, "stage24_book_correlation.csv")
    logger.info("alpha descriptors:\n%s\nIC decay:\n%s", table.round(4).to_string(), decay.round(4).to_string())

    # --------------------------------------------------------------- figures
    curves = {n: (1.0 + s).cumprod() for n, s in common.items()}
    figure_descriptors(decay, table, signal_corr, book_corr, context.figure("fig48_alpha_descriptors.png"))
    figure_results(curves, {k: w.loc[first_date:] for k, w in weights.items()}, tests, context.figure("fig49_alpha_combination.png"))

    # -------------------------------------------------------------- registry
    h_pass = bool(tests["passes"].all())
    context.registry.log(
        "A cost-aware rule for how much to trust each alpha, applied to the combined forecast, beats BOTH equal weighting of the "
        "alphas and the Generation 1 momentum book, net of costs.",
        stage=STAGE, parameters={"alphas": ALPHAS, "trust_window_days": window, "update": "monthly", "shrink_to_equal": 0.5,
                                 "window_start": str(first_date.date()), "test": "paired stationary bootstrap, BH across two"},
        results={"sharpe_cost_aware": float(perf.loc["cost_aware", "sharpe"]), "sharpe_equal": float(perf.loc["equal", "sharpe"]),
                 "sharpe_momentum": float(perf.loc["momentum", "sharpe"]), "sharpe_ic_weighted": float(perf.loc["ic_weighted", "sharpe"]),
                 "sharpe_hedge": float(perf.loc["hedge", "sharpe"]),
                 **{f"diff_{a}_vs_{b}": float(d) for a, b, d in zip(tests.a, tests.b, tests.difference)},
                 **{f"p_{a}_vs_{b}": float(p) for a, b, p in zip(tests.a, tests.b, tests.p_value)},
                 "deflated_sharpe_probability_cost_aware": float(perf.loc["cost_aware", "deflated_sharpe_probability"])},
        decision="retain" if h_pass else "reject", test_period=f"{first_date.date()} onward, net of costs",
        notes=("The first date on which every alpha exists is the first Stage 19 origin (2011-01-31); the declaration's '2011-01-03' was "
               "approximate. Retained only if BOTH paired tests are significant after BH control with a positive difference. The comparator "
               "momentum book is the Generation 1 alpha declared in advance, not the best alpha in hindsight. Costs are the linear Generation 1 "
               "model; Stage 25 re-costs every book with an impact model."),
    )
    context.registry.log(
        "Descriptive: what each alpha is worth (IC, its information ratio, decay, standalone turnover and net Sharpe, correlation).",
        stage=STAGE, parameters={"horizon_days": 21}, results={f"{k}_{c}": float(v) for k, r in table.iterrows() for c, v in r.items()},
        decision="record", notes="Descriptive table behind the engine's trust weights; no decision rests on it.",
    )
    logger.info("STAGE 24 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
