"""Stage 30 - The strategy library as a search (Generation 5).

42 specifications of 31 registered models run through the same pipeline and are judged the way Stage 26 judged a grid: after the size of
the search, does any specification have a positive expected net return (against cash) or beat passive equal weight? Then the
forecast-level combination of all 31 models is compared with equal weighting and with Generation 1 momentum. Rules: ``config/strategy_library.yaml``.

Figures 59-61.
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import numpy as np
import pandas as pd

from src.backtest.metrics import deflated_sharpe_ratio, performance_summary
from src.distributed.executor import run_tasks
from src.framework import MODELS, Pipeline, PipelineSpec, load_default_bundle, load_library
from src.framework.allocation import Context, book
from src.utils.dates import DateWindow
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg
from src.validation.multiple_testing import bootstrap_means, hansen_spa, pbo_cscv, white_reality_check
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context

STAGE = "stage30_library"
ANN = 252.0


def label(item: dict) -> str:
    params = item.get("params") or {}
    return item["model"] + ("" if not params else "[" + ",".join(f"{k}={v}" for k, v in params.items()) + "]")


def run_spec(item: dict, bundle, config) -> dict:
    import logging
    load_library()                                          # worker processes start with an empty registry
    logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)
    spec = PipelineSpec(name=label(item), models=[{"name": item["model"], "params": item.get("params") or {}}],
                        evaluation={"benchmarks": [], "causality": False})
    result = Pipeline(spec, config, bundle).run(validate=False)
    return {"label": spec.name, "model": item["model"], "family": MODELS.entries()[MODELS.names().index(item["model"])].family, "metrics": result.metrics,
            "net": result.net_returns, "start": result.start, "turnover": result.turnover,
            "weights_abs": result.weights.abs().sum(axis=1)}


def figure_search(table: pd.DataFrame, expected_best: float, passive_sharpe: float, verdict: dict, path):
    fig, axes = new_axes(1, 2, figsize=(17, 8.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ordered = table.sort_values("sharpe")
    families = sorted(ordered["family"].unique())
    colour = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(families)}
    ax.barh(range(len(ordered)), ordered["sharpe"], color=[colour[f] for f in ordered["family"]])
    ax.set_yticks(range(len(ordered)), ordered.index, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(passive_sharpe, color="#009E73", linestyle="--", linewidth=1.2)
    ax.axvline(expected_best, color="#CC0000", linestyle="--", linewidth=1.2)
    ax.text(passive_sharpe, len(ordered) - 0.5, " equal weight", color="#009E73", fontsize=8, va="top")
    ax.text(expected_best, len(ordered) - 3, " best of 42 noise rules", color="#CC0000", fontsize=8, va="top")
    ax.set_xlabel("Net Sharpe, common window")
    ax.set_title("The library as a search: 42 specifications")
    for f in families:
        ax.barh([], [], color=colour[f], label=f)
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    labels = ["RC vs cash", "SPA vs cash", "RC vs equal weight", "SPA vs equal weight"]
    values = [verdict["rc_cash"], verdict["spa_cash"], verdict["rc_passive"], verdict["spa_passive"]]
    ax.bar(range(4), values, color=[PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]])
    ax.axhline(0.10, color="#CC0000", linestyle="--", linewidth=1.0)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4), labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("p-value (null: no specification has an edge)")
    ax.set_title("After accounting for the search (dashed = 10%)")
    fig.suptitle("Figure 59. Forty-two strategies, one search", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Across 42 specifications of the strategy library, how are net Sharpe ratios distributed against what a search of that size finds in noise, and does any "
                           "specification have an edge over cash or over passive equal weight after the search?", 59)


def figure_correlation(corr: pd.DataFrame, path):
    fig, ax = new_axes(1, 1, figsize=(11, 9.5))
    from scipy.cluster.hierarchy import leaves_list, linkage
    order = leaves_list(linkage(1 - corr.fillna(0).to_numpy()[np.triu_indices(len(corr), 1)], "average"))
    c = corr.iloc[order, order]
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(c)), c.index, fontsize=6)
    ax.set_title("Correlation of daily net returns, clustered: how many independent ideas are there?")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Figure 60. Diversification inside the library", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "How correlated are the 42 specifications' net returns, and so how many independent bets does the library contain?", 60)


def figure_combination(curves: dict, perf: pd.DataFrame, tests: pd.DataFrame, path):
    fig, axes = new_axes(1, 2, figsize=(16, 5.6))
    ax = axes[0]
    for i, (name, s) in enumerate(curves.items()):
        ax.plot(s.index, s.to_numpy(), color=PALETTE[i % len(PALETTE)], linewidth=1.5 if "combination" in name else 1.1, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (net, log)")
    ax.set_title("Combining 31 models at the forecast level")
    ax.legend(fontsize=7)
    ax = axes[1]
    y = np.arange(len(tests))
    ax.errorbar(tests["difference"], y, xerr=[tests["difference"] - tests["ci_lower_5pct"], tests["ci_upper_95pct"] - tests["difference"]], fmt="o", color="black", capsize=3)
    for yi, (_, r) in zip(y, tests.iterrows()):
        ax.text(r["ci_upper_95pct"], yi, f"  p={r['p_value']:.2f}" + ("  *" if r["bh_significant"] else ""), va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [f"{a} vs {b}" for a, b in zip(tests["a"], tests["b"])], fontsize=8)
    ax.set_xlabel("Difference in net Sharpe (90% bootstrap interval)")
    ax.set_title("Pre-declared comparisons")
    fig.suptitle("Figure 61. Does combining the whole library help?", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does the confidence-weighted combination of all 31 library models beat the equal-weight combination and Generation 1 momentum, net of costs?", 61)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 30: the strategy library as a search").parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = cfg.get("strategy_library", {}) or {}
    load_library()
    logger.info("=" * 72)
    logger.info("STAGE 30 | the strategy library as a search (Generation 5)")
    logger.info("=" * 72)
    mt = node.get("multiple_testing", {}) or {}
    boot = mt.get("bootstrap", {}) or {}
    alpha = float((node.get("decisions", {}) or {}).get("alpha", 0.10))
    bundle = load_default_bundle(cfg)
    items = node["specs"]
    logger.info("%d specifications of %d models", len(items), len({i["model"] for i in items}))

    t0 = time.perf_counter()
    results = run_tasks(partial(run_spec, bundle=bundle, config=cfg), items, "joblib", -1, 1)
    logger.info("ran the specifications in %.0fs", time.perf_counter() - t0)

    start = max(pd.Timestamp(r["start"]) for r in results)
    net = pd.DataFrame({r["label"]: r["net"] for r in results})
    common = net.loc[start:].dropna(axis=1, how="any")
    assert common.shape[1] == len(results), f"specifications without full history on the common window: {set(net.columns) - set(common.columns)}"
    logger.info("common window from %s (%d days)", start.date(), len(common))
    perf = pd.DataFrame({k: performance_summary(common[k]) for k in common.columns}).T
    table = pd.DataFrame({"family": {r["label"]: r["family"] for r in results},
                          "sharpe": perf["sharpe"], "cagr": perf["cagr"], "ann_vol": perf["ann_vol"], "max_drawdown": perf["max_drawdown"],
                          "ann_turnover": pd.Series({r["label"]: float(r["turnover"].loc[start:].mean() * ANN) for r in results}),
                          "own_window_sharpe": pd.Series({r["label"]: r["metrics"]["sharpe"] for r in results}),
                          "gross_sharpe": pd.Series({r["label"]: r["metrics"]["gross_sharpe"] for r in results}),
                          "ann_cost_bps": pd.Series({r["label"]: r["metrics"]["ann_cost_bps"] for r in results}),
                          "own_start": pd.Series({r["label"]: str(pd.Timestamp(r["start"]).date()) for r in results}),
                          "share_days_flat": pd.Series({r["label"]: float((r["weights_abs"].loc[start:] < 1e-9).mean()) for r in results})})
    context.save_table(table, "stage30_specs.csv")
    context.processed.mkdir(parents=True, exist_ok=True)
    common.to_csv(context.processed / "stage30_returns.csv", index_label="date")
    logger.info("net Sharpe over the library: median %.3f, best %.3f (%s), worst %.3f; %.0f%% positive\n%s", table["sharpe"].median(), table["sharpe"].max(), table["sharpe"].idxmax(),
                table["sharpe"].min(), 100 * (table["sharpe"] > 0).mean(), table[["family", "sharpe", "gross_sharpe", "ann_turnover", "ann_cost_bps"]].sort_values("sharpe", ascending=False).round(3).to_string())

    # ------------------------------------------------------ the search-aware tests
    x = common.to_numpy()
    n_boot, block, seed = int(boot.get("n_samples", 2000)), float(boot.get("mean_block_length", 21)), int(boot.get("seed", 7))
    b_cash = bootstrap_means(x, n_boot, block, seed)
    rc, spa = white_reality_check(x, boot=b_cash), hansen_spa(x, boot=b_cash)
    ctx = Context(bundle, cfg)
    from src.backtest.engine import BacktestEngine
    engine = BacktestEngine.from_config(cfg)
    ew = engine.run(book("equal_weight", ctx), bundle.returns, "EW", bundle.investable, apply_vol_target=False).net_returns.reindex(common.index).fillna(0.0)
    relative = x - ew.to_numpy()[:, None]
    b_rel = bootstrap_means(relative, n_boot, block, seed)
    rc_p, spa_p = white_reality_check(relative, boot=b_rel), hansen_spa(relative, boot=b_rel)
    best = common.columns[rc["best_index"]]
    logger.info("best %s: net Sharpe %.3f; vs cash: RC p = %.3f, SPA p = %.3f; vs equal weight (Sharpe %.3f): RC p = %.3f, SPA p = %.3f", best, table.loc[best, "sharpe"], rc["p_value"],
                spa["p_consistent"], performance_summary(ew)["sharpe"], rc_p["p_value"], spa_p["p_consistent"])
    h_cash = bool(rc["p_value"] <= alpha and spa["p_consistent"] <= alpha)
    h_passive = bool(rc_p["p_value"] <= alpha and spa_p["p_consistent"] <= alpha)
    tests = pd.DataFrame([
        {"benchmark": "cash", "test": "reality_check", "p_value": rc["p_value"], "best": best}, {"benchmark": "cash", "test": "spa_lower", "p_value": spa["p_lower"], "best": best},
        {"benchmark": "cash", "test": "spa_consistent", "p_value": spa["p_consistent"], "best": best}, {"benchmark": "cash", "test": "spa_upper", "p_value": spa["p_upper"], "best": best},
        {"benchmark": "equal_weight", "test": "reality_check", "p_value": rc_p["p_value"], "best": common.columns[rc_p["best_index"]]},
        {"benchmark": "equal_weight", "test": "spa_consistent", "p_value": spa_p["p_consistent"], "best": common.columns[spa_p["best_index"]]}])
    context.save_table(tests, "stage30_search_tests.csv", index=False)
    pbo = pbo_cscv(x, int(mt.get("pbo_blocks", 16)))
    n_obs, n_trials = len(common), len(common.columns)
    dsr = deflated_sharpe_ratio(float(table.loc[best, "sharpe"]), n_trials, n_obs)
    from scipy import stats
    expected_best = float((1 - 0.5772156649015329) * stats.norm.ppf(1 - 1 / n_trials) + 0.5772156649015329 * stats.norm.ppf(1 - 1 / (n_trials * np.e))) / np.sqrt(n_obs) * np.sqrt(ANN)
    samples = {n: DateWindow.from_config(n, spec or {}) for n, spec in (cfg.get("backtest.samples", {}) or {}).items()}
    positive = {n: float((common.apply(lambda c: performance_summary(w.apply(c).dropna()).get("sharpe", np.nan)) > 0).mean()) for n, w in samples.items()}
    context.save_table(pd.DataFrame([{"specifications": n_trials, "common_start": str(start.date()), "n_days": n_obs, "pbo": pbo["pbo"], "mean_is_best_sharpe": pbo["mean_is_of_best"],
                                      "mean_oos_of_best_sharpe": pbo["mean_oos_of_best"], "degradation_slope": pbo["degradation_slope"], "best": best,
                                      "best_net_sharpe": float(table.loc[best, "sharpe"]), "deflated_sharpe_probability_best": dsr, "expected_best_sharpe_of_noise": expected_best,
                                      "equal_weight_sharpe": performance_summary(ew)["sharpe"]}]), "stage30_overfitting.csv", index=False)
    context.save_table(pd.DataFrame({"sample": list(positive), "share_positive_net_sharpe": list(positive.values())}), "stage30_positive_share_by_sample.csv", index=False)
    by_family = table.groupby("family")["sharpe"].agg(["count", "median", "max", lambda s: (s > 0).mean()])
    by_family.columns = ["specifications", "median_sharpe", "best_sharpe", "share_positive"]
    context.save_table(by_family, "stage30_by_family.csv")
    corr = common.corr().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    context.save_table(corr, "stage30_correlation.csv")
    off_diagonal = corr.to_numpy()[np.triu_indices(len(corr), 1)]
    ev = np.linalg.eigvalsh(corr.to_numpy())[::-1]
    effective_n = float(ev.sum() ** 2 / (ev ** 2).sum())
    logger.info("PBO %.3f; IS winner %.2f vs OOS %.2f; DSR of the best %.3f; expected best of noise %.2f; mean pairwise correlation %.3f; effective number of independent specifications %.1f\n%s\npositive share by sample %s",
                pbo["pbo"], pbo["mean_is_of_best"], pbo["mean_oos_of_best"], dsr, expected_best, off_diagonal.mean(), effective_n, by_family.round(3).to_string(), {k: round(v, 3) for k, v in positive.items()})

    # ---------------------------------------------------------------- combination
    comb = node["combination"]
    names = [n for n in MODELS.names() if MODELS.entries()[MODELS.names().index(n)].family != "crypto" and not n.startswith("test_")]
    base = PipelineSpec(name="library", models=[{"name": n} for n in names], allocation=comb["allocation"], evaluation={"benchmarks": [], "causality": False})
    shared = Pipeline(base, cfg, bundle)
    panels = shared._forecasts(shared._models())
    logger.info("forecasts for %d models computed once; combining under %s", len(panels), comb["rules"])
    combos = {}
    for rule in comb["rules"]:
        spec = PipelineSpec(name=f"combination:{rule}", models=base.models, combination={"rule": rule}, allocation=comb["allocation"], evaluation={"benchmarks": [], "causality": False})
        combos[f"combination:{rule}"] = Pipeline(spec, cfg, bundle).run(validate=False, precomputed=panels)
    cnet = {k: r.net_returns for k, r in combos.items()}
    window = max([start] + [r.start for r in combos.values()])
    series = {**{k: v.loc[window:].dropna() for k, v in cnet.items()}, "momentum (G1)": net["momentum"].loc[window:].dropna(), "equal weight (M0)": ew.loc[window:]}
    cperf = pd.DataFrame({k: performance_summary(v) for k, v in series.items()}).T
    cperf["ann_turnover"] = pd.Series({k: float(combos[k].turnover.loc[window:].mean() * ANN) for k in combos})
    context.save_table(cperf, "stage30_combination.csv")
    pairs = [("combination:confidence", "combination:equal"), ("combination:confidence", "momentum (G1)")]
    rows = [{"a": a, "b": b, **paired_sharpe_test(series[a], series[b].reindex(series[a].index).dropna(), n_samples=n_boot, block_length=int(block), seed=seed)} for a, b in pairs]
    ctests = pd.DataFrame(rows)
    ctests["bh_significant"] = benjamini_hochberg(ctests["p_value"], alpha)
    ctests["passes"] = ctests["bh_significant"] & (ctests["difference"] > 0)
    context.save_table(ctests, "stage30_combination_tests.csv", index=False)
    extra = [{"a": a, "b": b, **paired_sharpe_test(series[a], series[b].reindex(series[a].index).dropna(), n_samples=n_boot, block_length=int(block), seed=seed)}
             for a, b in (("combination:ic_weighted", "combination:equal"), ("combination:cost_aware", "combination:equal"), ("combination:confidence", "equal weight (M0)"))]
    context.save_table(pd.DataFrame(extra), "stage30_combination_tests_reported_not_judged.csv", index=False)
    h_comb = bool(ctests["passes"].all())
    logger.info("combinations from %s:\n%s\n%s", window.date(), cperf[["cagr", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"]].astype(float).round(3).to_string(),
                ctests[["a", "b", "difference", "p_value", "bh_significant", "passes"]].round(4).to_string(index=False))

    figure_search(table, expected_best, float(performance_summary(ew)["sharpe"]), {"rc_cash": rc["p_value"], "spa_cash": spa["p_consistent"], "rc_passive": rc_p["p_value"],
                                                                                   "spa_passive": spa_p["p_consistent"]}, context.figure("fig59_library_search.png"))
    figure_correlation(corr, context.figure("fig60_library_correlation.png"))
    figure_combination({k: (1 + v).cumprod() for k, v in series.items()}, cperf, ctests, context.figure("fig61_library_combination.png"))

    reg = context.registry
    reg.log("Across the 42 specifications of the strategy library, at least one has a positive expected net return after accounting for the search: both White's Reality Check and "
            "Hansen's SPA reject the null of none, at 10%.", stage=STAGE, parameters={"specifications": n_trials, "benchmark": "cash", "window_start": str(start.date()), "alpha": alpha},
            results={"best_net_sharpe": float(table.loc[best, "sharpe"]), "reality_check_p": rc["p_value"], "spa_consistent_p": spa["p_consistent"], "spa_lower_p": spa["p_lower"],
                     "spa_upper_p": spa["p_upper"], "share_positive_net_sharpe": float((table["sharpe"] > 0).mean()), "median_net_sharpe": float(table["sharpe"].median()),
                     "pbo": pbo["pbo"], "deflated_sharpe_probability_best": dsr, "expected_best_sharpe_of_noise": expected_best, "effective_number_of_specifications": effective_n},
            decision="retain" if h_cash else "reject", test_period=f"{start.date()} onward, net of costs", notes="Retained only if BOTH tests reject. The common window starts when the slowest specification has a live book.")
    reg.log("Across the 42 specifications, at least one beats passive equal weight after accounting for the search (Reality Check and SPA on differences of daily net returns, both at 10%).",
            stage=STAGE, parameters={"specifications": n_trials, "benchmark": "equal weight", "alpha": alpha},
            results={"reality_check_p": rc_p["p_value"], "spa_consistent_p": spa_p["p_consistent"], "equal_weight_sharpe": float(performance_summary(ew)["sharpe"])},
            decision="retain" if h_passive else "reject", test_period=f"{start.date()} onward, net of costs", notes="Equal weight is net of its own linear costs.")
    reg.log("The confidence-weighted combination of all 31 library models has a higher net Sharpe than both the equal-weight combination and the Generation 1 momentum specification.",
            stage=STAGE, parameters={"models": len(names), "rule": "confidence", "test": "paired stationary bootstrap, BH across two", "window_start": str(window.date())},
            results={**{f"sharpe_{k.replace(':', '_').replace(' ', '_').replace('(', '').replace(')', '')}": float(cperf.loc[k, "sharpe"]) for k in cperf.index},
                     **{f"diff_{i}": float(d) for i, d in enumerate(ctests["difference"])}, **{f"p_{i}": float(p) for i, p in enumerate(ctests["p_value"])}},
            decision="retain" if h_comb else "reject", test_period=f"{window.date()} onward, net of costs", notes="Retained only if both paired tests pass after BH control with positive differences.")
    reg.log("Descriptive: how many independent ideas the library holds, and the IC-weighted and cost-aware combinations.", stage=STAGE, parameters={"specifications": n_trials},
            results={"mean_pairwise_correlation": float(off_diagonal.mean()), "effective_number_of_specifications": effective_n,
                     "sharpe_ic_weighted": float(cperf.loc["combination:ic_weighted", "sharpe"]), "sharpe_cost_aware": float(cperf.loc["combination:cost_aware", "sharpe"])},
            decision="record", notes="Reported, not judged.")
    logger.info("STAGE 30 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
