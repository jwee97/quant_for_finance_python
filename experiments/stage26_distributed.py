"""Stage 26 - Distributed experimentation and honest search (Generation 4, Priorities 15 and 19).

A grid of 1,584 simple rules (momentum and mean reversion, with execution choices) is run in parallel and
every candidate's net return is recorded. The pre-declared question (``config/distributed.yaml``): after
accounting for the size of the search, is there any rule with a positive expected net return? Answered by
White's Reality Check and Hansen's SPA together. Probability of backtest overfitting and the deflated Sharpe
ratio of the best rule are reported alongside.

Figures 52-53.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import deflated_sharpe_ratio, performance_summary
from src.distributed.executor import available_backends, run_tasks
from src.features.mean_reversion import price_zscore
from src.features.momentum import ranked_momentum, total_return_momentum, volatility_scaled_momentum
from src.features.volatility import rolling_volatility
from src.models.regression import cross_sectional_ic
from src.signals.alpha_engine import forward_returns
from src.signals.transform import signal_to_positions
from src.utils.dates import DateWindow
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.multiple_testing import bootstrap_means, hansen_spa, pbo_cscv, white_reality_check
from experiments.context import build_context
from experiments.strategies import cached_ladder, transform_config

STAGE = "stage26_distributed"
ANN = 252.0


def expand_grid(node: dict) -> list[dict]:
    """Every candidate rule as a plain dict, in a fixed order."""
    grid = node.get("grid", {}) or {}
    mom, rev, execution = grid.get("momentum", {}), grid.get("mean_reversion", {}), grid.get("execution", {})
    rules = []
    for variant in mom.get("variant", []):
        vols = mom.get("vol_lookback_days", [63]) if variant == "vol_scaled" else [None]
        for lookback, skip, vol in itertools.product(mom.get("lookback_days", []), mom.get("skip_days", []), vols):
            rules.append({"family": "momentum", "variant": variant, "lookback": lookback, "skip": skip, "vol_lookback": vol})
    for lookback, basis, sign in itertools.product(rev.get("lookback_days", []), rev.get("price_basis", []), rev.get("sign", [-1])):
        rules.append({"family": "mean_reversion", "variant": basis, "lookback": lookback, "skip": None, "vol_lookback": None, "sign": sign})
    tasks = []
    for rule, (rebalance, lag, cross) in itertools.product(rules, itertools.product(
            execution.get("rebalance", ["monthly"]), execution.get("signal_lag_days", [1]), execution.get("cross_sectional", [True]))):
        tasks.append({**rule, "rebalance": rebalance, "signal_lag": lag, "cross_sectional": bool(cross)})
    for k, task in enumerate(tasks):
        task["id"] = k
    return tasks


def build_signal(task: dict, prices: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame) -> pd.DataFrame:
    if task["family"] == "momentum":
        if task["variant"] == "vol_scaled":
            signal = volatility_scaled_momentum(prices, returns, task["lookback"], task["skip"], task["vol_lookback"])
        elif task["variant"] == "raw":
            signal = total_return_momentum(prices, task["lookback"], task["skip"])
        else:
            signal = ranked_momentum(prices, task["lookback"], task["skip"])
    else:
        signal = task.get("sign", -1) * price_zscore(prices, task["lookback"], task["variant"])
    return signal.where(investable)


def evaluate_rule(task: dict, shared: SimpleNamespace) -> dict:
    """One candidate, end to end: signal, positions, backtest with the Generation 1 costs. Pure in its inputs."""
    signal = build_signal(task, shared.prices, shared.returns, shared.investable)
    transform = {**shared.transform, "cross_sectional": task["cross_sectional"]}
    positions = signal_to_positions(signal, shared.volatility, investable=shared.investable, **transform)
    engine = replace(shared.engine, rebalance=task["rebalance"], signal_lag=task["signal_lag"])
    result = engine.run(positions, shared.returns, f"rule{task['id']}", shared.investable, apply_vol_target=True)
    net = result.net_returns
    ic = cross_sectional_ic(signal, shared.fwd21).mean()
    live = result.weights.abs().sum(axis=1) > 1e-9
    return {"id": task["id"], "net": net.to_numpy(), "dates": net.index.to_numpy(), "first_live": live.idxmax().to_datetime64(),
            "gross_sharpe": float(ANN ** 0.5 * result.gross_returns.mean() / result.gross_returns.std(ddof=1)),
            "ann_turnover": float(result.turnover.mean() * ANN), "mean_ic_21d": float(ic)}


def _evaluate(task: dict, shared: SimpleNamespace | None = None) -> dict:   # shared is bound with functools.partial
    logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)       # one INFO line per rule would bury the log
    return evaluate_rule(task, shared)


def figure_grid(table: pd.DataFrame, expected_max_sharpe: float, grid_heat: pd.DataFrame, verdict: dict, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for i, (family, frame) in enumerate(table.groupby("family")):
        ax.hist(frame["sharpe"], bins=40, alpha=0.65, color=PALETTE[i], label=f"{family} ({len(frame)})")
    ax.axvline(0, color="black", linewidth=0.8)
    best = table["sharpe"].max()
    ax.axvline(best, color="#CC0000", linewidth=1.2)
    ax.axvline(expected_max_sharpe, color="#CC0000", linewidth=1.0, linestyle="--")
    ax.text(best, ax.get_ylim()[1] * 0.92, " best", color="#CC0000", fontsize=8, ha="right")
    ax.text(expected_max_sharpe, ax.get_ylim()[1] * 0.70, "expected best of\nN noise rules ", color="#CC0000", fontsize=8, ha="right")
    ax.set_xlabel("Net Sharpe, common window")
    ax.set_ylabel("Candidates")
    ax.set_title(f"Net Sharpe of {len(table):,} rules")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    im = ax.imshow(grid_heat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(grid_heat.shape[1]), [str(int(c)) for c in grid_heat.columns])
    ax.set_yticks(range(grid_heat.shape[0]), [str(int(r)) for r in grid_heat.index])
    for i in range(grid_heat.shape[0]):
        for j in range(grid_heat.shape[1]):
            v = grid_heat.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.5 else "black")
    ax.set_xlabel("Skip, days")
    ax.set_ylabel("Lookback, days")
    ax.set_title("Vol-scaled momentum, monthly, cross-sectional")
    fig.colorbar(im, ax=ax, fraction=0.046, label="net Sharpe")
    ax = axes[2]
    labels = ["Reality Check", "SPA (consistent)", "SPA (lower)", "SPA (upper)"]
    values = [verdict["rc"], verdict["spa_c"], verdict["spa_l"], verdict["spa_u"]]
    ax.bar(range(4), values, color=[PALETTE[0], PALETTE[1], PALETTE[5], PALETTE[5]])
    ax.axhline(0.10, color="#CC0000", linewidth=1.0, linestyle="--")
    ax.set_xticks(range(4), labels, rotation=15, ha="right")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("p-value that no rule has a positive expected return")
    ax.set_title("After accounting for the search (dashed = 10%)")
    fig.suptitle("Figure 52. The search: what 1,584 rules look like and what survives", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Across a grid of 1,584 simple rules, how does net Sharpe distribute, how much of the best is what a search of that size "
                           "produces from noise, and does any rule have a positive expected net return after White's Reality Check and Hansen's SPA?", 52)


def figure_overfit(pbo: dict, timing: pd.DataFrame, by_group: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.hist(pbo["logit"], bins=40, color=PALETTE[0], alpha=0.8)
    ax.axvline(0, color="#CC0000", linewidth=1.2)
    ax.set_xlabel("logit of the in-sample winner's out-of-sample rank")
    ax.set_ylabel(f"Splits (of {pbo['n_splits']:,})")
    ax.set_title(f"Probability of backtest overfitting = {pbo['pbo']:.2f}")
    ax = axes[1]
    ax.scatter(pbo["is_best_sharpe"], pbo["oos_of_best_sharpe"], s=6, alpha=0.25, color=PALETTE[1])
    lim = [min(pbo["is_best_sharpe"].min(), pbo["oos_of_best_sharpe"].min()), max(pbo["is_best_sharpe"].max(), pbo["oos_of_best_sharpe"].max())]
    ax.plot(lim, lim, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("In-sample Sharpe of the in-sample winner")
    ax.set_ylabel("Its out-of-sample Sharpe")
    ax.set_title(f"Selection degradation (slope {pbo['degradation_slope']:+.2f})")
    ax = axes[2]
    ax.barh(range(len(by_group)), by_group["median_sharpe"], color=PALETTE[2])
    ax.set_yticks(range(len(by_group)), by_group.index, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, v in enumerate(by_group["median_sharpe"]):
        ax.text(v, i, f" {v:+.2f}", va="center", fontsize=8)
    ax.set_xlabel("Median net Sharpe")
    ax.set_title("Where in the grid: median by group")
    fig.suptitle("Figure 53. Overfitting and where the grid works", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "How often would the best-in-sample rule from this grid land in the bottom half out of sample, how much does its Sharpe "
                           "degrade, and which parts of the grid are on average above or below zero?", 53)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 26: distributed experimentation")
    parser.add_argument("--backend", default=None, help="override the configured backend (serial, joblib, dask, ray)")
    args = parser.parse_args(argv)
    context, logger = build_context(STAGE, generation=4)
    cfg = context.config
    node = cfg.get("distributed", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 26 | distributed experimentation and honest search (Generation 4, Priorities 15 and 19)")
    logger.info("=" * 72)
    par = node.get("parallel", {}) or {}
    mt = node.get("multiple_testing", {}) or {}
    boot_cfg = mt.get("bootstrap", {}) or {}
    alpha = float((node.get("decisions", {}) or {}).get("alpha", 0.10))
    backend = args.backend or str(par.get("backend", "joblib"))
    logger.info("backends available: %s; using %s on %s", available_backends(), backend, par.get("n_jobs", -1))

    market = context.market_data()
    returns = market.returns()
    shared = SimpleNamespace(
        prices=market.prices, returns=returns, investable=market.investable,
        volatility=rolling_volatility(returns, int(cfg.get("portfolio.volatility.lookback", 63))),
        transform=transform_config(cfg), engine=BacktestEngine.from_config(cfg), fwd21=forward_returns(returns, 21))
    tasks = expand_grid(node)
    logger.info("%d candidate rules", len(tasks))
    from functools import partial
    fn = partial(_evaluate, shared=shared)

    # ------------------------------------------- the identity: parallel == serial
    rng = np.random.default_rng(int(boot_cfg.get("seed", 7)))
    check = [tasks[i] for i in sorted(rng.choice(len(tasks), size=int(par.get("identity_check_tasks", 24)), replace=False))]
    t0 = time.perf_counter()
    serial = run_tasks(fn, check, "serial")
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    parallel = run_tasks(fn, check, backend, int(par.get("n_jobs", -1)), int(par.get("chunk_size", 8)))
    t_parallel = time.perf_counter() - t0
    worst = max(float(np.max(np.abs(a["net"] - b["net"]))) if len(a["net"]) == len(b["net"]) else np.inf for a, b in zip(serial, parallel))
    logger.info("identity check on %d rules: max |serial - %s| = %.1e (serial %.1fs, %s %.1fs)", len(check), backend, worst, t_serial, backend, t_parallel)
    assert worst < 1e-12, "parallel execution must reproduce the serial results"

    # ----------------------------------------------------------------- the grid
    t0 = time.perf_counter()
    results = run_tasks(fn, tasks, backend, int(par.get("n_jobs", -1)), int(par.get("chunk_size", 8)))
    wall = time.perf_counter() - t0
    logger.info("grid of %d rules took %.0fs (%.2fs per rule per worker-equivalent: serial extrapolation %.0fs)", len(tasks), wall, wall / len(tasks),
                t_serial / len(check) * len(tasks))
    frame = pd.DataFrame(tasks).set_index("id")
    frame["gross_sharpe"] = [r["gross_sharpe"] for r in results]
    frame["ann_turnover"] = [r["ann_turnover"] for r in results]
    frame["mean_ic_21d"] = [r["mean_ic_21d"] for r in results]
    series = pd.DataFrame({r["id"]: pd.Series(r["net"], index=pd.DatetimeIndex(r["dates"])) for r in results})
    # The declared window: it begins when the slowest rule has a LIVE book (non-zero weights), so no rule is padded with idle cash days.
    start = max(pd.Timestamp(r["first_live"]) for r in results) + pd.Timedelta(days=1)
    start = series.index[series.index >= start][0]
    common = series.loc[start:].dropna(axis=1, how="any")
    dropped = series.shape[1] - common.shape[1]
    logger.info("common window from %s (%d days); %d rules without a full history dropped", start.date(), len(common), dropped)
    assert dropped == 0, "every candidate must have returns on the common window"
    summary = pd.DataFrame({k: performance_summary(common[k]) for k in common.columns}).T
    frame["sharpe"] = summary["sharpe"]
    frame["cagr"] = summary["cagr"]
    frame["ann_vol"] = summary["ann_vol"]
    frame["max_drawdown"] = summary["max_drawdown"]
    context.save_table(frame, "stage26_grid_results.csv", float_format="%.6f")
    context.processed.mkdir(parents=True, exist_ok=True)
    np.save(context.processed / "stage26_grid_returns.npy", common.to_numpy())
    logger.info("net Sharpe over the grid: median %.3f, best %.3f, worst %.3f; %.0f%% positive", frame["sharpe"].median(), frame["sharpe"].max(),
                frame["sharpe"].min(), 100 * (frame["sharpe"] > 0).mean())

    # --------------------------------------------- the search-aware tests (h_search)
    x = common.to_numpy()
    bootstrap = bootstrap_means(x, int(boot_cfg.get("n_samples", 2000)), float(boot_cfg.get("mean_block_length", 21)), int(boot_cfg.get("seed", 7)))
    rc = white_reality_check(x, boot=bootstrap)
    spa = hansen_spa(x, boot=bootstrap)
    best_id = common.columns[rc["best_index"]]
    logger.info("best rule #%s: net Sharpe %.3f; Reality Check p = %.3f; SPA p (lower/consistent/upper) = %.3f / %.3f / %.3f",
                best_id, frame.loc[best_id, "sharpe"], rc["p_value"], spa["p_lower"], spa["p_consistent"], spa["p_upper"])
    h_search = bool(rc["p_value"] <= alpha and spa["p_consistent"] <= alpha)
    # the same two tests with equal weight (M0) as the benchmark: reported, not judged
    m0 = cached_ladder(market, cfg, context.processed, ["M0_equal_weight"])["M0_equal_weight"]
    m0_net = shared.engine.run(m0, returns, "M0", market.investable, apply_vol_target=False).net_returns.reindex(common.index).fillna(0.0)
    relative = x - m0_net.to_numpy()[:, None]
    boot_rel = bootstrap_means(relative, int(boot_cfg.get("n_samples", 2000)), float(boot_cfg.get("mean_block_length", 21)), int(boot_cfg.get("seed", 7)))
    rc_rel, spa_rel = white_reality_check(relative, boot=boot_rel), hansen_spa(relative, boot=boot_rel)
    tests = pd.DataFrame([
        {"benchmark": "cash", "test": "reality_check", "statistic": rc["statistic"], "p_value": rc["p_value"], "best_rule": int(best_id)},
        {"benchmark": "cash", "test": "spa_lower", "statistic": spa["statistic"], "p_value": spa["p_lower"], "best_rule": int(best_id)},
        {"benchmark": "cash", "test": "spa_consistent", "statistic": spa["statistic"], "p_value": spa["p_consistent"], "best_rule": int(best_id)},
        {"benchmark": "cash", "test": "spa_upper", "statistic": spa["statistic"], "p_value": spa["p_upper"], "best_rule": int(best_id)},
        {"benchmark": "M0_equal_weight", "test": "reality_check", "statistic": rc_rel["statistic"], "p_value": rc_rel["p_value"], "best_rule": int(common.columns[rc_rel["best_index"]])},
        {"benchmark": "M0_equal_weight", "test": "spa_consistent", "statistic": spa_rel["statistic"], "p_value": spa_rel["p_consistent"], "best_rule": int(common.columns[spa_rel["best_index"]])},
    ])
    context.save_table(tests, "stage26_search_tests.csv", index=False)
    logger.info("vs equal weight (reported, not judged): Reality Check p = %.3f, SPA p = %.3f", rc_rel["p_value"], spa_rel["p_consistent"])

    # ----------------------------------------------------- descriptive diagnostics
    pbo = pbo_cscv(x, int((mt.get("pbo", {}) or {}).get("blocks", 16)))
    n_obs = len(common)
    best_sharpe = float(frame.loc[best_id, "sharpe"])
    dsr = deflated_sharpe_ratio(best_sharpe, len(tasks), n_obs)
    from scipy import stats
    expected_max = float((1 - 0.5772156649015329) * stats.norm.ppf(1 - 1 / len(tasks)) + 0.5772156649015329 * stats.norm.ppf(1 - 1 / (len(tasks) * np.e))) / np.sqrt(n_obs) * np.sqrt(ANN)
    logger.info("PBO = %.3f over %d splits; in-sample winner mean Sharpe %.2f, out of sample %.2f (slope %.2f); deflated Sharpe probability of the best %.3f; "
                "expected best of %d noise rules %.2f", pbo["pbo"], pbo["n_splits"], pbo["mean_is_of_best"], pbo["mean_oos_of_best"],
                pbo["degradation_slope"], dsr, len(tasks), expected_max)
    samples = {n: DateWindow.from_config(n, spec or {}) for n, spec in (cfg.get("backtest.samples", {}) or {}).items()}
    positive = {n: float((common.apply(lambda col: performance_summary(w.apply(col).dropna()).get("sharpe", np.nan)) > 0).mean()) for n, w in samples.items()}
    frame["group"] = frame.apply(lambda r: f"{r['family']}-{r['variant']}", axis=1)
    by_family = frame.groupby("family")["sharpe"].agg(["count", "median", "max", lambda s: (s > 0).mean()])
    by_family.columns = ["rules", "median_sharpe", "best_sharpe", "share_positive"]
    by_exec = pd.concat([frame.groupby(c)["sharpe"].agg(["median", "max", lambda s: (s > 0).mean()]).assign(choice=f"{c}={{}}") for c in ("rebalance", "signal_lag", "cross_sectional")])
    by_exec["choice"] = [c.format(i) for c, i in zip(by_exec["choice"], by_exec.index)]
    by_exec = by_exec.set_index("choice")
    by_exec.columns = ["median_sharpe", "best_sharpe", "share_positive"]
    by_group = pd.concat([frame.groupby("group")["sharpe"].median().rename("median_sharpe").to_frame(), by_exec[["median_sharpe"]]]).sort_values("median_sharpe")
    context.save_table(by_family, "stage26_by_family.csv")
    context.save_table(by_exec, "stage26_by_execution_choice.csv")
    context.save_table(pd.DataFrame({"sample": list(positive), "share_positive_net_sharpe": list(positive.values())}), "stage26_positive_share_by_sample.csv", index=False)
    context.save_table(pd.DataFrame([{"n_rules": len(tasks), "common_start": str(start.date()), "n_days": n_obs, "pbo": pbo["pbo"], "pbo_splits": pbo["n_splits"],
                                      "mean_is_best_sharpe": pbo["mean_is_of_best"], "mean_oos_of_best_sharpe": pbo["mean_oos_of_best"],
                                      "share_best_oos_negative": pbo["share_oos_negative"], "degradation_slope": pbo["degradation_slope"],
                                      "best_net_sharpe": best_sharpe, "deflated_sharpe_probability_best": dsr, "expected_best_sharpe_of_noise": expected_max}]),
                       "stage26_overfitting.csv", index=False)
    timing = pd.DataFrame([{"backend": backend, "tasks_checked": len(check), "serial_s": t_serial, "parallel_s": t_parallel, "speedup_on_check": t_serial / t_parallel,
                            "grid_rules": len(tasks), "grid_wall_s": wall, "cpu_count": __import__("os").cpu_count(), "max_abs_serial_vs_parallel": worst}])
    context.save_table(timing, "stage26_timing.csv", index=False)
    best_row = frame.loc[[best_id]].T
    context.save_table(best_row, "stage26_best_rule.csv")
    logger.info("share of rules with positive net Sharpe by sample: %s\nby family:\n%s", {k: round(v, 3) for k, v in positive.items()}, by_family.round(3).to_string())

    # --------------------------------------------------------------------- figures
    heat = frame[(frame["family"] == "momentum") & (frame["variant"] == "vol_scaled") & (frame["rebalance"] == "monthly") & (frame["signal_lag"] == 1)
                 & (frame["cross_sectional"]) & (frame["vol_lookback"] == 63)].pivot(index="lookback", columns="skip", values="sharpe")
    figure_grid(frame, expected_max, heat, {"rc": rc["p_value"], "spa_c": spa["p_consistent"], "spa_l": spa["p_lower"], "spa_u": spa["p_upper"]},
                context.figure("fig52_search.png"))
    figure_overfit(pbo, timing, by_group.tail(12) if len(by_group) > 12 else by_group, context.figure("fig53_overfitting.png"))

    # -------------------------------------------------------------------- registry
    context.registry.log(
        "Across a grid of 1,584 simple rules, at least one has a positive expected net return after accounting for the search: both White's Reality "
        "Check and Hansen's SPA reject the null of none at 10%.",
        stage=STAGE, parameters={"rules": len(tasks), "benchmark": "cash", "bootstrap": boot_cfg, "alpha": alpha, "backend": backend,
                                 "window_start": str(start.date())},
        results={"best_net_sharpe": best_sharpe, "reality_check_p": rc["p_value"], "spa_lower_p": spa["p_lower"], "spa_consistent_p": spa["p_consistent"],
                 "spa_upper_p": spa["p_upper"], "share_positive_net_sharpe": float((frame["sharpe"] > 0).mean()), "median_net_sharpe": float(frame["sharpe"].median()),
                 "pbo": pbo["pbo"], "deflated_sharpe_probability_best": dsr, "expected_best_sharpe_of_noise": expected_max,
                 "reality_check_p_vs_equal_weight": rc_rel["p_value"], "spa_consistent_p_vs_equal_weight": spa_rel["p_consistent"]},
        decision="retain" if h_search else "reject", test_period=f"{start.date()} onward, net of costs (common window)",
        notes="Retained only if BOTH tests reject at 10%. The grid is the pre-declared set of rules; the only data snooping is the search itself. "
              "The common window starts when the slowest rule (252-day lookback plus skip) has a full history. The comparison with equal weight and "
              "all other diagnostics are reported, not judged.",
    )
    context.registry.log(
        "Descriptive: parallel and serial execution of the grid give identical results, and the grid's wall-clock time.",
        stage=STAGE, parameters={"backend": backend, "identity_tasks": len(check)},
        results={"max_abs_serial_vs_parallel": worst, "grid_wall_seconds": wall, "identity_check_speedup": t_serial / t_parallel, "cpu_count": float(__import__("os").cpu_count())},
        decision="record", notes="The equality is an assertion in the run (1e-12), not a hypothesis. Speedup is hardware-dependent and reported for the record.",
    )
    logger.info("STAGE 26 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
