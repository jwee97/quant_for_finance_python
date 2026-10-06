"""Stage 38 - Reinforcement-learning allocation by evolution strategies (Generation 5).

A linear softmax policy over the twelve Stage 19 price features allocates across the 15 ETFs each month. It is trained by evolution strategies on a
reward of mean-variance utility net of costs, refitted annually on the Stage 19 schedule using only matured months, and judged through the Generation 1
backtest engine against equal weight. Rules: ``config/frontier.yaml`` (section ``reinforcement``), committed before this stage ran. Figures 75-76.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.context import build_context
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import performance_summary
from src.framework import load_default_bundle
from src.framework.allocation import Context, book
from src.models.probabilistic import price_features
from src.models.rl_allocation import evolution_strategies, softmax_weights, utility
from src.utils.dates import rebalance_dates
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.robustness import paired_sharpe_test

STAGE = "stage38_rl"
ANN = 252.0
SEEDS = (1, 2, 3)
RISK_AVERSION, PAIRS, SIGMA, LR, ITERATIONS = 5.0, 32, 0.1, 0.05, 60


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(ANN ** 0.5 * r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else float("nan")


def train_refit(start: int, seed: int, Xs: np.ndarray, R: np.ndarray, mask: np.ndarray, pos_next: np.ndarray, n_features: int, cost: float) -> dict:
    ok = (pos_next <= start) & mask.any(axis=1)
    X, Rm, M = Xs[ok], R[ok], mask[ok]
    fitness = lambda p: utility(p, X, Rm, n_features, RISK_AVERSION, cost, M)
    params, history = evolution_strategies(fitness, n_features + Xs.shape[1], seed, PAIRS, SIGMA, LR, ITERATIONS)
    return {"start": start, "seed": seed, "params": params, "train_utility": float(fitness(params)), "initial_utility": history[0], "n_months": int(ok.sum())}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 38: RL allocation").parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = (cfg.get("frontier", {}) or {})["reinforcement"]
    fdr = float(node["decisions"]["fdr"])
    logger.info("=" * 72)
    logger.info("STAGE 38 | reinforcement-learning allocation (Generation 5)")
    logger.info("=" * 72)
    bundle = load_default_bundle(cfg)
    prices, returns = bundle.prices, bundle.returns
    index, assets = bundle.index, bundle.assets
    cost = float(cfg.get("backtest.costs.cost_bps", 10.0)) / 1e4
    frozen = pd.read_csv(context.tables / "stage19_predictions_price_only.csv", parse_dates=["origin"])
    boot = (node.get("bootstrap") or {})
    n_boot, block, seed_b = int(boot.get("n_samples", 2000)), int(boot.get("block_length", 21)), int(boot.get("seed", 7))

    me = rebalance_dates(index, "monthly")
    pos = np.array([index.get_loc(d) for d in me])
    pos = pos[pos >= 252]
    cube = np.stack([f.to_numpy(dtype=float) for f in price_features(prices).values()], axis=2)
    Xm = cube[pos]                                             # (months, assets, 12)
    nxt = np.append(pos[1:], len(index) - 1)
    px = prices.to_numpy(dtype=float)
    R = px[nxt] / px[pos] - 1.0
    mask = np.isfinite(px[pos]) & np.isfinite(R) & (np.arange(len(pos))[:, None] < len(pos) - 1)
    R = np.nan_to_num(R)
    n_features = Xm.shape[2]
    logger.info("%d monthly decision dates, %d assets, %d features", len(pos), len(assets), n_features)

    refits = sorted(frozen["refit_day"].unique())
    # standardise with the statistics of the months available at each refit
    jobs = []
    stats = {}
    for start in refits:
        ok = (nxt <= start) & mask.any(axis=1)
        flat = Xm[ok].reshape(-1, n_features)
        mu, sd = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0, ddof=1)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        stats[start] = (mu, sd)
        Xs = (Xm - mu) / sd
        for seed in SEEDS:
            jobs.append((start, seed, Xs))
    t0 = time.perf_counter()
    fits = Parallel(n_jobs=-1)(delayed(train_refit)(s, seed, Xs, R, mask, nxt, n_features, cost) for s, seed, Xs in jobs)
    logger.info("%d ES runs in %.0fs", len(fits), time.perf_counter() - t0)

    # weights for each Stage 19 origin from its refit's policies
    origin_dates = sorted(frozen["origin"].unique())
    pos_of_origin = {d: index.get_loc(d) for d in origin_dates}
    refit_of = frozen.drop_duplicates("origin").set_index("origin")["refit_day"].to_dict()
    month_of_pos = {int(p): i for i, p in enumerate(pos)}
    weight_rows = {s: {} for s in SEEDS}
    oos_rows = []
    for d in origin_dates:
        p = pos_of_origin[d]
        if p not in month_of_pos:
            continue
        i, start = month_of_pos[p], refit_of[d]
        mu, sd = stats[start]
        x = ((Xm[i] - mu) / sd)[None]
        for seed in SEEDS:
            f = next(f for f in fits if f["start"] == start and f["seed"] == seed)
            weight_rows[seed][d] = softmax_weights(x, f["params"][:n_features], f["params"][n_features:], np.isfinite(px[p])[None])[0]
    def weights_frame(rows: dict) -> pd.DataFrame:
        w = pd.DataFrame(rows, index=assets).T.sort_index()
        return w.reindex(index).ffill().fillna(0.0)
    frames = {f"seed {s}": weights_frame(weight_rows[s]) for s in SEEDS}
    averaged = sum(frames.values()) / len(frames)
    start_date = pd.Timestamp(min(weight_rows[SEEDS[0]]))

    engine = BacktestEngine.from_config(cfg)
    import logging
    logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)
    ctx = Context(bundle, cfg)
    ew = book("equal_weight", ctx)
    def run(w, name):
        return engine.run(w, returns, name, bundle.investable, apply_vol_target=False)
    runs = {"RL policy (3-seed average)": run(averaged, "rl"), "equal weight (M0)": run(ew, "m0"), **{f"RL {k}": run(v, k) for k, v in frames.items()}}
    series = {k: r.net_returns.loc[start_date:].dropna() for k, r in runs.items()}
    perf = pd.DataFrame({k: performance_summary(v) for k, v in series.items()}).T
    perf["ann_turnover"] = [float(r.turnover.loc[start_date:].mean() * ANN) for r in runs.values()]
    context.save_table(perf, "stage38_performance.csv")
    logger.info("performance, net, from %s:\n%s", start_date.date(), perf[["cagr", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"]].astype(float).round(3).to_string())

    test = paired_sharpe_test(series["RL policy (3-seed average)"], series["equal weight (M0)"], n_samples=n_boot, block_length=block, seed=seed_b)
    h_rl = bool(test["p_value"] <= fdr and test["difference"] > 0)
    logger.info("paired Sharpe test RL vs equal weight: %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test.items()})
    seed_tests = [{"a": k, "b": "equal weight (M0)", **paired_sharpe_test(series[k], series["equal weight (M0)"], n_samples=n_boot, block_length=block, seed=seed_b)} for k in (f"RL seed {s}" for s in SEEDS)]
    context.save_table(pd.DataFrame([{"a": "RL policy (3-seed average)", "b": "equal weight (M0)", **test}] + seed_tests), "stage38_tests.csv", index=False)

    # overfitting gap: utility on the training months against the utility of the following year's months
    gap_rows = []
    for f in fits:
        start = f["start"]
        mu, sd = stats[start]
        later = np.flatnonzero((pos > start - 1) & (pos <= start + 252) & (nxt <= len(index) - 1) & mask.any(axis=1))
        if len(later) < 6:
            continue
        Xs = (Xm - mu) / sd
        gap_rows.append({"refit_day": start, "seed": f["seed"], "train_utility": f["train_utility"], "equal_weight_train_utility": utility(np.zeros(n_features + len(assets)), Xs[(nxt <= start) & mask.any(axis=1)], R[(nxt <= start) & mask.any(axis=1)], n_features, RISK_AVERSION, cost, mask[(nxt <= start) & mask.any(axis=1)]),
                         "oos_utility": utility(f["params"], Xs[later], R[later], n_features, RISK_AVERSION, cost, mask[later]),
                         "equal_weight_oos_utility": utility(np.zeros(n_features + len(assets)), Xs[later], R[later], n_features, RISK_AVERSION, cost, mask[later])})
    gap = pd.DataFrame(gap_rows)
    gap["train_gain"] = gap["train_utility"] - gap["equal_weight_train_utility"]
    gap["oos_gain"] = gap["oos_utility"] - gap["equal_weight_oos_utility"]
    context.save_table(gap, "stage38_overfitting_gap.csv", index=False)
    logger.info("mean utility gain over equal weight: training %.5f, out of sample %.5f (%d refit-seeds)", gap["train_gain"].mean(), gap["oos_gain"].mean(), len(gap))

    # null: random tilts from the ES perturbation distribution, evaluated on the same decision months with the first refit's standardisation
    mu0, sd0 = stats[refits[0]]
    Xs0 = (Xm - mu0) / sd0
    sel = np.flatnonzero((pos >= pos_of_origin[origin_dates[0]]) & mask.any(axis=1))
    rng = np.random.default_rng(7)
    def monthly_sharpe(params):
        w = softmax_weights(Xs0[sel], params[:n_features], params[n_features:], mask[sel])
        prev = np.vstack([np.where(mask[sel][:1], 1.0 / mask[sel][:1].sum(), 0.0), w[:-1]])
        net = (w * R[sel]).sum(axis=1) - cost * np.abs(w - prev).sum(axis=1)
        return float(np.sqrt(12) * net.mean() / net.std(ddof=1))
    null = np.array([monthly_sharpe(SIGMA * rng.normal(size=n_features + len(assets))) for _ in range(200)])
    zero_sharpe = monthly_sharpe(np.zeros(n_features + len(assets)))
    context.save_table(pd.DataFrame({"null_monthly_sharpe": null}), "stage38_random_policy_null.csv", index=False)
    logger.info("random-tilt null Sharpe (monthly, ann.): mean %.3f sd %.3f; equal weight %.3f", null.mean(), null.std(), zero_sharpe)
    avg_w = averaged.loc[start_date:].mean()
    context.save_table(pd.DataFrame({"mean_weight": avg_w}), "stage38_mean_weights.csv")

    # ------------------------------------------------------------- figures
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for k, c in zip(["RL policy (3-seed average)", "equal weight (M0)"], [PALETTE[0], "#555555"]):
        ax.plot(series[k].index, (1 + series[k]).cumprod().to_numpy(), color=c, label=k, linewidth=1.4)
    for s_ in SEEDS:
        v = series[f"RL seed {s_}"]
        ax.plot(v.index, (1 + v).cumprod().to_numpy(), color=PALETTE[0], alpha=0.25, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1, net of costs")
    ax.set_title("Out of sample (thin lines: single ES seeds)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(null, bins=25, color="#BBBBBB", label="200 random tilts")
    ax.axvline(zero_sharpe, color="black", linestyle=":", label="equal weight")
    ax.axvline(float(perf.loc["RL policy (3-seed average)", "sharpe"]), color=PALETTE[0], label="RL policy (engine Sharpe)")
    ax.set_xlabel("Annualised Sharpe")
    ax.set_title(f"Paired test vs equal weight: diff {test['difference']:+.3f}, p = {test['p_value']:.3f}")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.bar(np.arange(len(avg_w)), (avg_w - 1 / len(assets)).to_numpy() * 100, color=PALETTE[0])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(avg_w)), list(avg_w.index), rotation=60, fontsize=8)
    ax.set_ylabel("Mean weight minus 1/15, percentage points")
    ax.set_title("Where does the policy tilt?")
    fig.suptitle("Figure 75. A reinforcement-learned (evolution-strategies) allocator against equal weight", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, context.figure("fig75_rl_allocator.png"), "Does a small policy trained by evolution strategies on a net-of-cost utility beat equal weight out of sample, and is it distinguishable from a random tilt?", 75)

    fig, axes = new_axes(1, 2, figsize=(12, 5))
    ax = axes[0]
    g = gap.groupby("refit_day")[["train_gain", "oos_gain"]].mean()
    ax.plot(g.index, g["train_gain"].to_numpy(), marker="o", color=PALETTE[0], label="training months")
    ax.plot(g.index, g["oos_gain"].to_numpy(), marker="o", color=PALETTE[3], label="following year")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Refit (day index)")
    ax.set_ylabel("Utility gain over equal weight")
    ax.set_title("The overfitting gap, refit by refit")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(range(len(SEEDS) + 1), [float(perf.loc[k, "sharpe"]) for k in ["RL policy (3-seed average)"] + [f"RL seed {s}" for s in SEEDS]], color=PALETTE[0])
    ax.axhline(float(perf.loc["equal weight (M0)", "sharpe"]), color="black", linestyle=":", label="equal weight")
    ax.set_xticks(range(len(SEEDS) + 1), ["average"] + [f"seed {s}" for s in SEEDS])
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Seed dispersion")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 76. Training gain, out-of-sample gain and seed dispersion", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, context.figure("fig76_rl_overfitting.png"), "How much of the policy's training-period gain survives the next year, and how much does the answer depend on the ES seed?", 76)

    context.registry.log(
        "A policy learned by evolution strategies (linear softmax over price features, mean-variance utility net of costs, annual refits) has a higher net Sharpe than equal weight (paired test, p <= 0.10).",
        stage=STAGE, parameters={"pairs": PAIRS, "sigma": SIGMA, "lr": LR, "iterations": ITERATIONS, "risk_aversion": RISK_AVERSION, "seeds": list(SEEDS), "fdr": fdr},
        results={"sharpe_rl": float(perf.loc["RL policy (3-seed average)", "sharpe"]), "sharpe_equal_weight": float(perf.loc["equal weight (M0)", "sharpe"]),
                 "sharpe_difference": float(test["difference"]), "p_value": float(test["p_value"]), "mean_train_gain": float(gap["train_gain"].mean()), "mean_oos_gain": float(gap["oos_gain"].mean())},
        decision="retain" if h_rl else "reject", test_period=f"{start_date.date()} onward, net of costs",
        notes="One paired test. Single seeds, turnover, the overfitting gap and the random-tilt null are reported, not judged.")
    logger.info("STAGE 38 complete: h_rl=%s", h_rl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
