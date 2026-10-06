"""Stage 31 - The adaptive pipeline (Generation 5): Market -> Regime -> Adaptive signal -> Adaptive portfolio -> Adaptive risk.

Does it help when everything downstream reacts to the regime? Four pre-declared decisions in ``config/adaptive.yaml``, each against a control:
regime-switching allocation (and a placebo that shifts the regime path), regime-dependent volatility targets, and regime-conditional trust in
the signals.

Figures 62-64.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import performance_summary
from src.framework import ALLOCATORS, DETECTORS, Pipeline, PipelineSpec, RegimeSeries, load_default_bundle, load_library
from src.framework.allocation import Context, book
from src.framework.risk import RegimeRiskPolicy, vol_target_series
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context

STAGE = "stage31_adaptive"
ANN = 252.0
REGIME_COLOUR = {"LowVol": "#BBBBBB", "HighVol": "#E69F00", "Crisis": "#D55E00", "Inflation": "#CC79A7", "Deflation": "#56B4E9"}


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(ANN ** 0.5 * r.mean() / r.std(ddof=1)) if len(r) > 30 and r.std(ddof=1) > 0 else float("nan")


def cvar95(r: pd.Series) -> float:
    r = r.dropna()
    return float(-r[r <= r.quantile(0.05)].mean())


def figure_regimes(regimes: RegimeSeries, proxy_curve: pd.Series, summary: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [2, 1, 1]})
    ax = axes[0]
    p = regimes.probabilities.dropna().resample("ME").mean()
    ax.stackplot(p.index, [p[c] for c in p.columns], labels=list(p.columns), colors=[REGIME_COLOUR.get(c, "#888888") for c in p.columns], alpha=0.9)
    ax2 = ax.twinx()
    ax2.plot(proxy_curve.index, proxy_curve, color="black", linewidth=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("Equal-weight market proxy (log)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Regime probability (monthly mean)")
    ax.set_title("Where the composite detector thinks the market has been")
    ax.legend(fontsize=7, loc="upper left", ncol=5)
    ax = axes[1]
    ax.bar(range(len(summary)), summary["share_of_days"], color=[REGIME_COLOUR.get(i, "#888888") for i in summary.index])
    ax.set_xticks(range(len(summary)), summary.index, rotation=20, fontsize=8)
    ax.set_title("Share of days")
    ax = axes[2]
    ax.bar(range(len(summary)), summary["mean_spell_days"], color=[REGIME_COLOUR.get(i, "#888888") for i in summary.index])
    ax.set_xticks(range(len(summary)), summary.index, rotation=20, fontsize=8)
    ax.set_title("Mean spell length (days)")
    fig.suptitle("Figure 62. The regimes everything downstream reacts to", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "What regimes does the composite detector find, how much of the sample does each occupy, and how long does a spell last?", 62)


def figure_allocation(curves: dict, placebo: np.ndarray, actual: float, tests: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c.index, c, color=PALETTE[i % len(PALETTE)], linewidth=1.6 if "switch" in name else 1.0, label=name)
    ax.set_yscale("log")
    ax.set_title("Net growth of 1 (log)")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.hist(placebo, bins=30, color=PALETTE[0], alpha=0.8)
    ax.axvline(actual, color="#CC0000", linewidth=1.6)
    ax.axvline(np.quantile(placebo, 0.9), color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Net Sharpe")
    ax.set_title(f"Real path (red) vs {len(placebo)} shifted (dashed: 90th pct)")
    ax = axes[2]
    y = np.arange(len(tests))
    ax.errorbar(tests["difference"], y, xerr=[tests["difference"] - tests["ci_lower_5pct"], tests["ci_upper_95pct"] - tests["difference"]], fmt="o", color="black", capsize=3)
    for yi, (_, r) in zip(y, tests.iterrows()):
        ax.text(r["ci_upper_95pct"], yi, f"  p={r['p_value']:.2f}" + ("  *" if r["bh_significant"] else ""), va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [f"{a} vs {b}" for a, b in zip(tests["a"], tests["b"])], fontsize=8)
    ax.set_xlabel("Difference in net Sharpe")
    ax.set_title("Pre-declared comparisons")
    fig.suptitle("Figure 63. Regime-switching allocation and its placebo", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does choosing the allocator by regime beat static risk parity and equal weight, and is the timing of the regime path worth anything compared with the same "
                           "path shifted by a random number of years?", 63)


def figure_risk_signal(target: pd.Series, scalar: pd.Series, dd: dict, trust: pd.DataFrame, tests: pd.DataFrame, path):
    fig, axes = new_axes(1, 4, figsize=(19, 5.2))
    ax = axes[0]
    ax.plot(target.index, target, color=PALETTE[1])
    ax.axhline(0.10, color="black", linestyle=":")
    ax.set_title("Regime-dependent volatility target")
    ax = axes[1]
    for i, (name, d) in enumerate(dd.items()):
        ax.plot(d.index, d, color=PALETTE[i], linewidth=1.0, label=name)
    ax.set_title("Drawdown of risk parity")
    ax.legend(fontsize=7)
    ax = axes[2]
    m = trust.resample("ME").last()
    ax.stackplot(m.index, [m[c] for c in m.columns], labels=list(m.columns), colors=(PALETTE * 2)[: m.shape[1]])
    ax.set_ylim(0, 1)
    ax.set_title("Regime-conditional trust in the twelve models")
    ax.legend(fontsize=5, ncol=2, loc="upper left")
    ax = axes[3]
    y = np.arange(len(tests))
    ax.errorbar(tests["difference"], y, xerr=[tests["difference"] - tests["ci_lower_5pct"], tests["ci_upper_95pct"] - tests["difference"]], fmt="o", color="black", capsize=3)
    for yi, (_, r) in zip(y, tests.iterrows()):
        ax.text(r["ci_upper_95pct"], yi, f"  p={r['p_value']:.2f}", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [f"{a}\nvs {b}" for a, b in zip(tests["a"], tests["b"])], fontsize=7)
    ax.set_title("Adaptive risk and signal, pre-declared")
    fig.suptitle("Figure 64. Adaptive risk and the adaptive signal", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does a regime-dependent volatility target beat a constant one, and does trusting each model according to its record in the current regime beat trusting them equally?", 64)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 31: the adaptive pipeline").parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = cfg.get("adaptive", {}) or {}
    load_library()
    logger.info("=" * 72)
    logger.info("STAGE 31 | the adaptive pipeline (Generation 5)")
    logger.info("=" * 72)
    fdr = float((node.get("decisions", {}) or {}).get("fdr", 0.10))
    boot = node.get("bootstrap", {}) or {}
    n_boot, block, seed = int(boot.get("n_samples", 2000)), int(boot.get("block_length", 21)), int(boot.get("seed", 7))
    bundle = load_default_bundle(cfg)
    returns = bundle.returns
    engine = BacktestEngine.from_config(cfg)
    det = node["detector"]
    regimes = DETECTORS.create(det["name"], **(det.get("params") or {})).detect(bundle)
    ctx = Context(bundle, cfg, regimes=regimes)
    labels = regimes.hard_labels()
    logger.info("regime shares: %s", regimes.share().round(3).to_dict())

    def net(weights: pd.DataFrame, name: str, vol_target: bool = False) -> pd.Series:
        import logging
        logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)
        return engine.run(weights, returns, name, bundle.investable, apply_vol_target=vol_target).net_returns

    # ---------------------------------------------------------------- 1. adaptive portfolio
    alloc = node["allocation"]
    names = ["risk_parity", "equal_weight", "mean_cvar", "mvo"]
    books = {n: book(n, ctx) for n in names}
    soft = ALLOCATORS.create("regime_switch", rules=alloc["rules"], default=alloc["default"], mode="soft").build(ctx)
    hard = ALLOCATORS.create("regime_switch", rules=alloc["rules"], default=alloc["default"], mode="hard").build(ctx)
    shares = regimes.probabilities.mean()
    constant = RegimeSeries(pd.DataFrame({r: float(shares[r]) for r in shares.index}, index=bundle.index))
    static_blend = ALLOCATORS.create("regime_switch", rules=alloc["rules"], default=alloc["default"], mode="soft").build(Context(bundle, cfg, regimes=constant, cache=ctx.cache))
    series = {"regime switch (soft)": net(soft, "soft"), "regime switch (hard)": net(hard, "hard"), "static blend (same books, fixed weights)": net(static_blend, "blend"),
              "risk parity (M2)": net(books["risk_parity"], "m2"), "equal weight (M0)": net(books["equal_weight"], "m0"), "mean-CVaR (M9)": net(books["mean_cvar"], "m9"),
              "plug-in MVO": net(books["mvo"], "mvo")}
    start = max(regimes.probabilities.dropna().index[0], *[b.abs().sum(axis=1).gt(1e-9).idxmax() for b in books.values()])
    window = {k: v.loc[start:].dropna() for k, v in series.items()}
    perf = pd.DataFrame({k: performance_summary(v) for k, v in window.items()}).T
    perf["ann_turnover"] = [float(x) for x in [engine.run(w, returns, "t", bundle.investable, apply_vol_target=False).turnover.loc[start:].mean() * ANN
                                               for w in (soft, hard, static_blend, books["risk_parity"], books["equal_weight"], books["mean_cvar"], books["mvo"])]]
    context.save_table(perf, "stage31_allocation_performance.csv")
    logger.info("allocation, net, from %s:\n%s", start.date(), perf[["cagr", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"]].astype(float).round(3).to_string())
    rows = [{"a": "regime switch (soft)", "b": b, **paired_sharpe_test(window["regime switch (soft)"], window[b], n_samples=n_boot, block_length=block, seed=seed)} for b in ("risk parity (M2)", "equal weight (M0)")]
    tests = pd.DataFrame(rows)
    tests["bh_significant"] = benjamini_hochberg(tests["p_value"], fdr)
    tests["passes"] = tests["bh_significant"] & (tests["difference"] > 0)
    context.save_table(tests, "stage31_allocation_tests.csv", index=False)
    extra = [{"a": a, "b": b, **paired_sharpe_test(window[a], window[b], n_samples=n_boot, block_length=block, seed=seed)} for a, b in
             (("regime switch (soft)", "static blend (same books, fixed weights)"), ("regime switch (hard)", "risk parity (M2)"), ("regime switch (soft)", "regime switch (hard)"))]
    context.save_table(pd.DataFrame(extra), "stage31_allocation_tests_reported_not_judged.csv", index=False)
    h_alloc = bool(tests["passes"].all())
    logger.info("h_adaptive_allocation:\n%s", tests[["a", "b", "sharpe_a", "sharpe_b", "difference", "p_value", "bh_significant", "passes"]].round(4).to_string(index=False))

    # placebo: the same books with a regime path shifted circularly by a random number of years
    pl = alloc["placebo"]
    rng = np.random.default_rng(int(pl.get("seed", 7)))
    probs = regimes.probabilities
    T, k = len(probs), int(pl.get("min_shift_days", 252))
    placebo = []
    t0 = time.perf_counter()
    for _ in range(int(pl.get("draws", 200))):
        shift = int(rng.integers(k, T - k))
        shifted = RegimeSeries(pd.DataFrame(np.roll(probs.to_numpy(), shift, axis=0), index=probs.index, columns=probs.columns))
        w = ALLOCATORS.create("regime_switch", rules=alloc["rules"], default=alloc["default"], mode="soft").build(Context(bundle, cfg, regimes=shifted, cache=ctx.cache))
        placebo.append(sharpe(net(w, "placebo").loc[start:]))
    placebo = np.array(placebo)
    actual = float(perf.loc["regime switch (soft)", "sharpe"])
    q90 = float(np.quantile(placebo, 0.90))
    placebo_p = float((1 + (placebo >= actual).sum()) / (1 + len(placebo)))
    h_timing = bool(actual > q90)
    context.save_table(pd.DataFrame({"placebo_sharpe": placebo}), "stage31_placebo.csv", index=False)
    logger.info("placebo (%.0fs): actual %.3f; placebo mean %.3f, 90th percentile %.3f; one-sided p = %.3f -> %s", time.perf_counter() - t0, actual, placebo.mean(), q90, placebo_p,
                "RETAIN" if h_timing else "REJECT")

    # regimes: summary and performance of the switch by regime
    spells = (labels != labels.shift()).cumsum()
    summary = pd.DataFrame({"share_of_days": labels.value_counts(normalize=True),
                            "mean_spell_days": labels.groupby(spells).agg(["first", "size"]).groupby("first")["size"].mean()})
    by_regime = {}
    for r in summary.index:
        mask = labels.reindex(window["regime switch (soft)"].index) == r
        for name in ("regime switch (soft)", "risk parity (M2)", "equal weight (M0)"):
            s = window[name][mask.to_numpy()]
            by_regime.setdefault(r, {})[f"{name} ann return"] = float(s.mean() * ANN) if len(s) > 5 else np.nan
        by_regime[r]["days"] = int(mask.sum())
    summary = summary.join(pd.DataFrame(by_regime).T)
    context.save_table(summary, "stage31_regime_summary.csv")
    logger.info("regime summary:\n%s", summary.round(3).to_string())

    # ------------------------------------------------------------------ 2. adaptive risk
    rk = node["risk"]
    policy = RegimeRiskPolicy(targets=rk["regime_targets"], default_target=float(rk["constant_target"]), lookback=int(rk["lookback"]), max_leverage=float(rk["max_leverage"]))
    risk_rows, risk_series, rtests = [], {}, []
    for base in rk["base_books"]:
        w = books[base]
        constant_w, _ = vol_target_series(w, returns, float(rk["constant_target"]), int(rk["lookback"]), float(rk["max_leverage"]))
        regime_w, scalar, target = policy.apply(w, returns, regimes)
        a, b = net(regime_w, f"{base}-regime-risk").loc[start:].dropna(), net(constant_w, f"{base}-constant-risk").loc[start:].dropna()
        common = a.index.intersection(b.index)
        risk_series[base] = (a.loc[common], b.loc[common], scalar, target)
        for label, s in (("regime target", a.loc[common]), ("constant 10%", b.loc[common])):
            curve = (1 + s).cumprod()
            risk_rows.append({"base": base, "risk": label, "sharpe": sharpe(s), "ann_vol": float(s.std(ddof=1) * ANN ** 0.5), "max_drawdown": float((curve / curve.cummax() - 1).min()),
                              "cvar_95_daily": cvar95(s), "vol_of_vol": float(s.rolling(63).std().std() * ANN ** 0.5), "cagr": float(curve.iloc[-1] ** (ANN / len(s)) - 1)})
        t = paired_sharpe_test(a.loc[common], b.loc[common], n_samples=n_boot, block_length=block, seed=seed)
        rtests.append({"a": f"{base}, regime target", "b": f"{base}, constant 10%", **t})
    context.save_table(pd.DataFrame(risk_rows), "stage31_risk_performance.csv", index=False)
    risk_tests = pd.DataFrame(rtests)
    risk_tests["bh_significant"] = benjamini_hochberg(risk_tests["p_value"], fdr)
    risk_tests["passes"] = risk_tests["bh_significant"] & (risk_tests["difference"] > 0)
    context.save_table(risk_tests, "stage31_risk_tests.csv", index=False)
    h_risk = bool(risk_tests["passes"].all())
    logger.info("adaptive risk:\n%s\n%s", pd.DataFrame(risk_rows).round(4).to_string(index=False), risk_tests[["a", "b", "difference", "p_value", "bh_significant", "passes"]].round(4).to_string(index=False))

    # ------------------------------------------------------------------ 3. adaptive signal
    sg = node["signal"]
    models = [{"name": n} for n in sg["models"]]
    base_spec = PipelineSpec(name="adaptive-signal", models=models, regime={"detector": det["name"], "params": det.get("params") or {}}, allocation=sg["allocation"],
                             evaluation={"benchmarks": [], "causality": False})
    pipe = Pipeline(base_spec, cfg, bundle)
    panels = pipe._forecasts(pipe._models())
    trust_cfg = sg["regime_trust"]
    results = {}
    for label, rule in (("regime-conditional trust", "regime_conditional"), ("unconditional equal", sg["comparator"])):
        spec = PipelineSpec(name=label, models=models, regime=base_spec.regime, allocation=sg["allocation"], evaluation={"benchmarks": [], "causality": False},
                            combination={"rule": rule, "min_regime_days": trust_cfg["min_regime_days"], "shrink": trust_cfg["shrink"]})
        results[label] = Pipeline(spec, cfg, bundle).run(validate=False, precomputed=panels)
    sig_start = max(r.start for r in results.values())
    sa, sb = (results[k].net_returns.loc[sig_start:].dropna() for k in ("regime-conditional trust", "unconditional equal"))
    common = sa.index.intersection(sb.index)
    stest = paired_sharpe_test(sa.loc[common], sb.loc[common], n_samples=n_boot, block_length=block, seed=seed)
    signal_tests = pd.DataFrame([{"a": "regime-conditional trust", "b": "unconditional equal", **stest}])
    context.save_table(signal_tests, "stage31_signal_tests.csv", index=False)
    sig_perf = pd.DataFrame({k: performance_summary(results[k].net_returns.loc[sig_start:].dropna()) for k in results}).T
    context.save_table(sig_perf, "stage31_signal_performance.csv")
    h_signal = bool(stest["p_value"] <= fdr and stest["difference"] > 0)
    logger.info("adaptive signal from %s:\n%s\ndifference %.3f, p = %.3f -> %s", sig_start.date(), sig_perf[["cagr", "ann_vol", "sharpe", "max_drawdown"]].astype(float).round(3).to_string(),
                stest["difference"], stest["p_value"], "RETAIN" if h_signal else "REJECT")
    trust = results["regime-conditional trust"].tables.get("trust_weights", pd.DataFrame())
    # post-hoc (labelled, never overturns the decision): how fragile is the result to the two trust parameters?
    sens = []
    for shrink in (0.25, 0.5, 0.75):
        for days in (63, 126, 252):
            spec = PipelineSpec(name="sens", models=models, regime=base_spec.regime, allocation=sg["allocation"], evaluation={"benchmarks": [], "causality": False},
                                combination={"rule": "regime_conditional", "min_regime_days": days, "shrink": shrink})
            r = Pipeline(spec, cfg, bundle).run(validate=False, precomputed=panels).net_returns.loc[sig_start:].dropna()
            c = r.index.intersection(sb.index)
            t = paired_sharpe_test(r.loc[c], sb.loc[c], n_samples=500, block_length=block, seed=seed)
            sens.append({"shrink": shrink, "min_regime_days": days, "sharpe": t["sharpe_a"], "sharpe_equal": t["sharpe_b"], "difference": t["difference"], "p_value": t["p_value"]})
    sens = pd.DataFrame(sens)
    context.save_table(sens, "stage31_signal_sensitivity_posthoc.csv", index=False)
    logger.info("post-hoc sensitivity of the adaptive signal to its two parameters:\n%s", sens.round(3).to_string(index=False))

    # --------------------------------------------------------------------------- figures
    proxy = (1 + __import__("src.framework.regimes", fromlist=["market_proxy"]).market_proxy(bundle)).cumprod()
    figure_regimes(regimes, proxy, summary, context.figure("fig62_regimes.png"))
    figure_allocation({k: (1 + v).cumprod() for k, v in window.items() if k in ("regime switch (soft)", "risk parity (M2)", "equal weight (M0)", "static blend (same books, fixed weights)")},
                      placebo, actual, tests, context.figure("fig63_adaptive_allocation.png"))
    a_, b_, scalar_, target_ = risk_series["risk_parity"]
    dd = {"regime target": (1 + a_).cumprod() / (1 + a_).cumprod().cummax() - 1, "constant 10%": (1 + b_).cumprod() / (1 + b_).cumprod().cummax() - 1}
    figure_risk_signal(target_.loc[start:], scalar_.loc[start:], dd, trust if len(trust) else pd.DataFrame({"none": [1.0]}, index=[start]), pd.concat([risk_tests[["a", "b", "difference", "ci_lower_5pct", "ci_upper_95pct", "p_value"]], signal_tests[["a", "b", "difference", "ci_lower_5pct", "ci_upper_95pct", "p_value"]]]),
                       context.figure("fig64_adaptive_risk_signal.png"))

    reg = context.registry
    reg.log("A regime-switching allocator (Crisis: mean-CVaR, LowVol: MVO, Inflation: commodity tilt, otherwise risk parity) has a higher net Sharpe than both static risk parity and equal weight.",
            stage=STAGE, parameters={"detector": det["name"], "mode": "soft", "rules": alloc["rules"], "window_start": str(start.date()), "fdr": fdr},
            results={"sharpe_switch": actual, "sharpe_risk_parity": float(perf.loc["risk parity (M2)", "sharpe"]), "sharpe_equal_weight": float(perf.loc["equal weight (M0)", "sharpe"]),
                     "sharpe_static_blend": float(perf.loc["static blend (same books, fixed weights)", "sharpe"]), "sharpe_hard_switch": float(perf.loc["regime switch (hard)", "sharpe"]),
                     **{f"diff_{i}": float(d) for i, d in enumerate(tests["difference"])}, **{f"p_{i}": float(p) for i, p in enumerate(tests["p_value"])}},
            decision="retain" if h_alloc else "reject", test_period=f"{start.date()} onward, net of costs", notes="Retained only if both paired tests pass after BH control with positive differences.")
    reg.log("The regime path adds value: the switching allocator's net Sharpe exceeds the 90th percentile of 200 circular-shift placebos.", stage=STAGE,
            parameters={"draws": int(len(placebo)), "min_shift_days": k, "seed": int(pl.get("seed", 7))},
            results={"actual_sharpe": actual, "placebo_mean": float(placebo.mean()), "placebo_p90": q90, "placebo_max": float(placebo.max()), "one_sided_p": placebo_p},
            decision="retain" if h_timing else "reject", test_period=f"{start.date()} onward, net of costs", notes="Separates the average mix of books from the timing of the switches.")
    reg.log("Regime-dependent volatility targets give a higher net Sharpe than a constant 10% target for both risk parity and equal weight.", stage=STAGE,
            parameters={"targets": rk["regime_targets"], "constant": rk["constant_target"], "fdr": fdr},
            results={**{f"diff_{i}": float(d) for i, d in enumerate(risk_tests["difference"])}, **{f"p_{i}": float(p) for i, p in enumerate(risk_tests["p_value"])}},
            decision="retain" if h_risk else "reject", test_period=f"{start.date()} onward, net of costs", notes="Drawdown, CVaR and volatility of volatility are in the table, not judged.")
    reg.log("Combining twelve models with regime-conditional trust weights has a higher net Sharpe than their unconditional equal combination.", stage=STAGE,
            parameters={"models": sg["models"], "min_regime_days": trust_cfg["min_regime_days"], "shrink": trust_cfg["shrink"]},
            results={"sharpe_regime_conditional": float(sig_perf.loc["regime-conditional trust", "sharpe"]), "sharpe_equal": float(sig_perf.loc["unconditional equal", "sharpe"]),
                     "difference": float(stest["difference"]), "p_value": float(stest["p_value"]), "posthoc_sensitivity_min_difference": float(sens["difference"].min()),
                     "posthoc_sensitivity_max_difference": float(sens["difference"].max()), "posthoc_sensitivity_share_p_below_10pct": float((sens["p_value"] <= 0.10).mean())},
            decision="retain" if h_signal else "reject", test_period=f"{sig_start.date()} onward, net of costs", notes="One paired test, p <= 0.10 and a positive difference. Post-hoc: a 3 x 3 sensitivity to the two trust parameters is in stage31_signal_sensitivity_posthoc.csv. This is one of four decisions declared in the stage.")
    logger.info("STAGE 31 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
