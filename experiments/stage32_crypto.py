"""Stage 32 - The optional crypto branch (Generation 5): funding carry, basis reversion and a stablecoin flow factor on Deribit and DefiLlama data.

Rules in ``config/strategy_library.yaml`` (section ``crypto``), committed before any result. Figures 65.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import performance_summary
from src.data.crypto import crypto_version
from src.framework import Pipeline, PipelineSpec, load_library
from src.framework.crypto_bundle import load_crypto_bundle
from src.utils.dates import DateWindow
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context

STAGE = "stage32_crypto"
ANN = 252.0


def figure_crypto(bundle, curves: dict, tests: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    macro = bundle.macro
    ax.plot(macro.index, macro["FUNDING_BTC"] * 100, color=PALETTE[0], linewidth=0.9, label="BTC funding (annualised %, 7-day mean)")
    ax.plot(macro.index, macro["FUNDING_ETH"] * 100, color=PALETTE[1], linewidth=0.9, label="ETH funding")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("What a short perpetual collects")
    ax.legend(fontsize=7)
    ax = axes[1]
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c.index, c, color=PALETTE[i % len(PALETTE)], linewidth=1.2, label=name)
    ax.set_yscale("log")
    ax.set_title("Net growth of 1 (log)")
    ax.legend(fontsize=7)
    ax = axes[2]
    y = np.arange(len(tests))
    ax.errorbar(tests["difference"], y, xerr=[tests["difference"] - tests["ci_lower_5pct"], tests["ci_upper_95pct"] - tests["difference"]], fmt="o", color="black", capsize=3)
    for yi, (_, r) in zip(y, tests.iterrows()):
        ax.text(r["ci_upper_95pct"], yi, f"  p={r['p_value']:.2f}" + ("  *" if r["bh_significant"] else ""), va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [f"{a}\nvs {b}" for a, b in zip(tests["a"], tests["b"])], fontsize=7)
    ax.set_title("Pre-declared comparisons")
    fig.suptitle("Figure 65. The crypto branch", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does timing the funding carry, trading the basis, or following stablecoin supply growth beat the unconditioned alternative (always-on carry, buy and hold) "
                           "on Deribit perpetual data?", 65)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 32: crypto branch").parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = (cfg.get("strategy_library", {}) or {}).get("crypto", {})
    load_library()
    logger.info("=" * 72)
    logger.info("STAGE 32 | the optional crypto branch (Generation 5)")
    logger.info("=" * 72)
    fdr = float(node["decisions"]["fdr"])
    boot = node["bootstrap"]
    bundle = load_crypto_bundle(cfg)
    logger.info("crypto data %s: %d weekdays from %s to %s", crypto_version(cfg), len(bundle.index), bundle.index[0].date(), bundle.index[-1].date())
    import logging
    logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)
    results = {}
    for name in node["specs"]:
        spec = PipelineSpec(name=name, models=[{"name": name}], execution=node["execution"], allocation=node["allocation"], risk=node["risk"],
                            evaluation={"benchmarks": [], "causality": True})
        results[name] = Pipeline(spec, cfg, bundle).run()
        logger.info("%s: causality %s", name, {k: v["ok"] for k, v in results[name].validation["causality"].items()})
    engine = BacktestEngine.from_config(cfg)
    from dataclasses import replace
    engine = replace(engine, rebalance=node["execution"]["rebalance"], signal_lag=int(node["execution"]["signal_lag"]), min_assets=int(node["execution"]["min_assets"]))

    def static(weights: dict, label: str) -> pd.Series:
        w = pd.DataFrame(0.0, index=bundle.index, columns=bundle.assets)
        for a, v in weights.items():
            w[a] = v
        return engine.run(w, bundle.returns, label, bundle.investable, apply_vol_target=False).net_returns

    benches = {"always-on carry": static({"BTC_CARRY": 0.5, "ETH_CARRY": 0.5}, "carry"), "buy and hold": static({"BTC": 0.5, "ETH": 0.5}, "hold")}
    start = max(r.start for r in results.values())
    series = {k: r.net_returns.loc[start:].dropna() for k, r in results.items()}
    series.update({k: v.loc[start:].dropna() for k, v in benches.items()})
    perf = pd.DataFrame({k: performance_summary(v) for k, v in series.items()}).T
    perf["ann_turnover"] = pd.Series({k: float(results[k].turnover.loc[start:].mean() * ANN) for k in results})
    perf["ann_cost_bps"] = pd.Series({k: results[k].metrics["ann_cost_bps"] for k in results})
    context.save_table(perf, "stage32_performance.csv")
    logger.info("net, from %s:\n%s", start.date(), perf[["cagr", "ann_vol", "sharpe", "max_drawdown", "ann_turnover", "ann_cost_bps"]].astype(float).round(3).to_string())
    pairs = [("funding_carry", "always-on carry"), ("basis_reversion", "always-on carry"), ("stablecoin_flow", "buy and hold")]
    tests = pd.DataFrame([{"a": a, "b": b, **paired_sharpe_test(series[a], series[b].reindex(series[a].index).dropna(), n_samples=int(boot["n_samples"]), block_length=int(boot["block_length"]), seed=int(boot["seed"]))}
                          for a, b in pairs])
    tests["bh_significant"] = benjamini_hochberg(tests["p_value"], fdr)
    tests["passes"] = tests["bh_significant"] & (tests["difference"] > 0)
    context.save_table(tests, "stage32_tests.csv", index=False)
    logger.info("pre-declared family:\n%s", tests[["a", "b", "sharpe_a", "sharpe_b", "difference", "p_value", "bh_significant", "passes"]].round(4).to_string(index=False))
    samples = {n: DateWindow.from_config(n, s or {}) for n, s in (cfg.get("backtest.samples", {}) or {}).items()}
    by_sample = pd.DataFrame({k: {n: performance_summary(w.apply(v).dropna()).get("sharpe", np.nan) for n, w in samples.items()} for k, v in series.items()}).T
    context.save_table(by_sample, "stage32_sharpe_by_sample.csv")
    cash = pd.Series(0.0, index=series["always-on carry"].index)
    h = bool(tests["passes"].any())
    figure_crypto(bundle, {k: (1 + v).cumprod() for k, v in series.items()}, tests, context.figure("fig65_crypto.png"))
    context.registry.log("Timing the funding carry, trading the basis, or following stablecoin supply growth beats the unconditioned alternative (always-on carry, buy and hold) after "
                         "Benjamini-Hochberg control at 10% across three paired tests.", stage=STAGE,
                         parameters={"data": crypto_version(cfg), "specs": node["specs"], "rebalance": node["execution"]["rebalance"], "window_start": str(start.date()), "fdr": fdr},
                         results={**{f"sharpe_{k.replace(' ', '_').replace('-', '_')}": float(perf.loc[k, "sharpe"]) for k in perf.index},
                                  **{f"diff_{i}": float(d) for i, d in enumerate(tests["difference"])}, **{f"p_{i}": float(p) for i, p in enumerate(tests["p_value"])}},
                         decision="retain" if h else "reject", test_period=f"{start.date()} onward, net of costs",
                         notes="Retained if at least one of the three pre-declared comparisons passes. The carry asset treats the inverse perpetual as linear; one venue (Deribit).")
    logger.info("STAGE 32 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
