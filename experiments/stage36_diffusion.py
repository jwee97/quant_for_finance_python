"""Stage 36 - Diffusion-model scenarios for tail risk (Generation 5).

A denoising diffusion model is trained on the joint distribution of the 15 ETFs' standardised 21-day forward returns (z-vectors), sampled at each
monthly origin, scaled by the origin's volatility forecast, and read off as the 5% and 1% value-at-risk of the equal-weight portfolio. It is held to
two baselines trained on the SAME rows: a multivariate normal and the empirical (bootstrap) distribution. Rules: ``config/frontier.yaml`` (section
``diffusion``), committed before this stage ran. Figures 71-72.
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from experiments.context import build_context
from src.distributed.executor import run_tasks
from src.models.diffusion import Diffusion, kupiec_pof, pinball_loss
from src.models.probabilistic import ewma_sigma
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg, diebold_mariano

STAGE = "stage36_diffusion"
METHODS = ["diffusion", "gaussian", "bootstrap"]
LEVELS = (0.05, 0.01)
COLOUR = {"diffusion": PALETTE[0], "gaussian": PALETTE[3], "bootstrap": PALETTE[2]}


def refit_task(task: dict, shared: SimpleNamespace) -> dict:
    start = task["start"]
    cfg = shared.cfg
    usable = (shared.day + shared.horizon + shared.embargo <= start) & shared.good
    Z = shared.z[usable]
    m = int(cfg["n_scenarios"])
    model = Diffusion(steps=int(cfg["steps"]), hidden=int(cfg["hidden"]), layers=int(cfg["layers"]), epochs=int(cfg["epochs"]), lr=float(cfg["lr"]), batch=int(cfg["batch"]), seed=int(cfg["seed"])).fit(Z)
    mean, cov = Z.mean(axis=0), np.cov(Z.T)
    out = {"start": start, "n_train": len(Z), "final_loss": model.losses[-1], "rows": []}
    for pos, sigma in zip(shared.origins[start], shared.sigma[start]):
        rng = np.random.default_rng(int(pos))
        draws = {"diffusion": model.sample(m, int(pos)), "gaussian": rng.multivariate_normal(mean, cov, size=m), "bootstrap": Z[rng.integers(0, len(Z), size=m)]}
        w_inv = (1.0 / sigma) / (1.0 / sigma).sum()
        for name, d in draws.items():
            for book, w in (("equal", np.full(len(sigma), 1.0 / len(sigma))), ("inverse_vol", w_inv)):
                port = (d * sigma) @ w
                out["rows"].append({"pos": int(pos), "method": name, "book": book, **{f"q{int(l * 100)}": float(np.quantile(port, l)) for l in LEVELS}})
    # realism diagnostics on the last origin's draws
    ref = shared.realism[start]
    real = {}
    train_corr = np.corrcoef(Z.T)
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    last = {"diffusion": model.sample(5000, 12345), "gaussian": np.random.default_rng(1).multivariate_normal(mean, cov, size=5000), "bootstrap": Z[np.random.default_rng(2).integers(0, len(Z), size=5000)]}
    for name, d in last.items():
        real[name] = {"excess_kurtosis_mean": float(np.mean(kurtosis(d, axis=0))), "corr_distance": float(np.sqrt(np.mean((np.corrcoef(d.T) - train_corr) ** 2))),
                      "share_outside_training_range": float(np.mean(((d < lo) | (d > hi)).any(axis=1)))}
    real["training"] = {"excess_kurtosis_mean": float(np.mean(kurtosis(Z, axis=0))), "corr_distance": 0.0, "share_outside_training_range": 0.0}
    out["realism"] = real
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 36: diffusion scenarios")
    parser.add_argument("--backend", default=None)
    args = parser.parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = (cfg.get("frontier", {}) or {})["diffusion"]
    run_cfg = {**{"steps": 100, "hidden": 128, "layers": 3, "epochs": 600, "lr": 1e-3, "batch": 128, "seed": 11, "n_scenarios": 2000}, **(node.get("run") or {})}
    logger.info("=" * 72)
    logger.info("STAGE 36 | diffusion scenarios for tail risk (Generation 5)")
    logger.info("=" * 72)
    fdr = float(node["decisions"]["fdr"])
    horizon = int(cfg.get("forecasting.horizon_days", 21))
    embargo = int(((cfg.get("deeplearning", {}) or {})["task"]["walk_forward"])["embargo_days"])
    backend = args.backend or str((cfg.get("distributed.parallel", {}) or {}).get("backend", "joblib"))
    frozen = pd.read_csv(context.tables / "stage19_predictions_price_only.csv", parse_dates=["origin"]).dropna(subset=["target", "sigma"])
    prices = context.market_data().prices
    returns = prices.pct_change()
    index = pd.DatetimeIndex(prices.index)
    assets = list(prices.columns)
    sigma_h = ewma_sigma(returns, float(cfg.get("forecasting.gaussian.vol_halflife", 40)), horizon)
    forward = prices.shift(-horizon) / prices - 1.0
    z_cube = np.clip(forward.to_numpy() / sigma_h.to_numpy(), -5.0, 5.0)
    days = np.arange(252, len(index) - horizon, 5)
    Zall = z_cube[days]
    good = np.isfinite(Zall).all(axis=1)
    origins, sigmas, realism = {}, {}, {}
    wide = frozen.pivot(index="origin", columns="asset", values="sigma")[assets]
    for start, g in frozen.groupby("refit_day"):
        os_ = sorted(g["origin"].unique())
        origins[int(start)] = np.array([index.get_loc(o) for o in os_])
        sigmas[int(start)] = wide.loc[os_].to_numpy()
        realism[int(start)] = None
    shared = SimpleNamespace(cfg=run_cfg, z=np.nan_to_num(Zall), day=days, good=good, horizon=horizon, embargo=embargo, origins=origins, sigma=sigmas, realism=realism)
    tasks = [{"start": s} for s in sorted(origins)]
    t0 = time.perf_counter()
    results = run_tasks(partial(refit_task, shared=shared), tasks, backend, int((cfg.get("distributed.parallel", {}) or {}).get("n_jobs", -1)), 1)
    logger.info("%d refits trained and sampled in %.0fs", len(results), time.perf_counter() - t0)

    q = pd.DataFrame([r for res in results for r in res["rows"]])
    q["origin"] = index[q["pos"].to_numpy()]
    # realised portfolio returns
    fwd = forward.to_numpy()
    realised = {}
    for book in ("equal", "inverse_vol"):
        vals = {}
        for pos in q["pos"].unique():
            sig = wide.loc[index[pos]].to_numpy()
            w = np.full(len(assets), 1.0 / len(assets)) if book == "equal" else (1.0 / sig) / (1.0 / sig).sum()
            vals[pos] = float(fwd[pos] @ w)
        realised[book] = vals
    q["realised"] = [realised[b][p] for b, p in zip(q["book"], q["pos"])]
    context.save_table(q.drop(columns=["pos"]), "stage36_var_forecasts.csv", index=False)

    rows, loss_by = [], {}
    for book in ("equal", "inverse_vol"):
        for level in LEVELS:
            col = f"q{int(level * 100)}"
            for method in METHODS:
                s = q[(q["book"] == book) & (q["method"] == method)].sort_values("origin")
                exceed = int((s["realised"] < s[col]).sum())
                k = kupiec_pof(exceed, len(s), level)
                loss = pd.Series(pinball_loss(s["realised"].to_numpy(), s[col].to_numpy(), level), index=s["origin"])
                loss_by[(book, level, method)] = loss
                rows.append({"book": book, "level": level, "method": method, "n_origins": len(s), "exceedances": exceed, "exceedance_rate": exceed / len(s), "kupiec_p": k["p_value"], "mean_pinball": float(loss.mean())})
    summary = pd.DataFrame(rows)
    context.save_table(summary, "stage36_var_summary.csv", index=False)
    logger.info("VaR summary:\n%s", summary.round(5).to_string(index=False))

    tests = []
    for book in ("equal", "inverse_vol"):
        for level in LEVELS:
            for other in ("gaussian", "bootstrap"):
                a, b = loss_by[(book, level, "diffusion")], loss_by[(book, level, other)]
                t = diebold_mariano(a, b, lag=0, alternative="two-sided")
                tests.append({"book": book, "level": level, "a": "diffusion", "b": other, "mean_pinball_difference": float(t["mean_loss_difference"]), "p_value": float(t["p_value"])})
    tests = pd.DataFrame(tests)
    decl = tests[(tests["book"] == "equal") & (tests["level"] == 0.05)].copy()
    decl["bh_significant"] = benjamini_hochberg(decl["p_value"], fdr)
    decl["passes"] = decl["bh_significant"] & (decl["mean_pinball_difference"] < 0)
    tests["status"] = np.where((tests["book"] == "equal") & (tests["level"] == 0.05), "declared", "reported")
    tests = tests.merge(decl[["b", "bh_significant", "passes"]].assign(book="equal", level=0.05), on=["book", "level", "b"], how="left")
    context.save_table(tests, "stage36_var_tests.csv", index=False)
    logger.info("pinball tests:\n%s", tests.round(5).to_string(index=False))
    h_diffusion = bool(decl["passes"].all())

    real = pd.DataFrame({m: np.mean([[r["realism"][m][k] for k in ("excess_kurtosis_mean", "corr_distance", "share_outside_training_range")] for r in results], axis=0) for m in METHODS + ["training"]},
                        index=["excess_kurtosis_mean", "corr_distance", "share_outside_training_range"]).T
    context.save_table(real, "stage36_scenario_realism.csv")
    logger.info("scenario realism (mean over refits):\n%s", real.round(4).to_string())
    train_log = pd.DataFrame([{"refit_day": r["start"], "n_train": r["n_train"], "final_loss": r["final_loss"]} for r in results])
    context.save_table(train_log, "stage36_training_log.csv", index=False)

    # ---------------------------------------------------------------- figures
    eq5 = q[(q["book"] == "equal")].pivot_table(index="origin", columns="method", values="q5")
    realised_eq = q[(q["book"] == "equal") & (q["method"] == "diffusion")].set_index("origin")["realised"]
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.plot(realised_eq.index, realised_eq.to_numpy(), color="#999999", linewidth=0.8, label="realised 21-day return")
    for m in METHODS:
        ax.plot(eq5.index, eq5[m].to_numpy(), color=COLOUR[m], linewidth=1.2, label=f"{m} 5% VaR")
    ax.set_ylabel("Equal-weight 21-day return")
    ax.set_title("5% value-at-risk forecasts")
    ax.legend(fontsize=8)
    ax = axes[1]
    sub = summary[summary["book"] == "equal"]
    x = np.arange(2)
    for j, m in enumerate(METHODS):
        vals = [float(sub[(sub["method"] == m) & (sub["level"] == l)]["exceedance_rate"].iloc[0]) for l in LEVELS]
        ax.bar(x + (j - 1) * 0.25, vals, 0.25, color=COLOUR[m], label=m)
    for xi, l in zip(x, LEVELS):
        ax.hlines(l, xi - 0.4, xi + 0.4, color="black", linestyle=":")
    ax.set_xticks(x, ["5% VaR", "1% VaR"])
    ax.set_ylabel("Exceedance rate (dotted = promised)")
    ax.set_title("Kupiec coverage, equal-weight book")
    ax.legend(fontsize=8)
    ax = axes[2]
    diff = decl.set_index("b")["mean_pinball_difference"]
    ax.bar(range(2), -diff.to_numpy() * 1e4, color=[COLOUR[b] for b in diff.index])
    ax.axhline(0, color="black", linewidth=0.8)
    for i, (b, row) in enumerate(decl.set_index("b").iterrows()):
        ax.text(i, -row["mean_pinball_difference"] * 1e4, f"p={row['p_value']:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(2), [f"vs {b}" for b in diff.index])
    ax.set_ylabel("Pinball-loss advantage of diffusion, 1e-4 (above zero = better)")
    ax.set_title("Declared test, 5% VaR, equal-weight")
    fig.suptitle("Figure 71. Diffusion-model scenarios against Gaussian and bootstrap value-at-risk", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, context.figure("fig71_diffusion_var.png"), "Do scenarios drawn from a trained diffusion model give better tail-risk forecasts for the equal-weight portfolio than a normal or the empirical distribution?", 71)

    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    for ax, col, title in zip(axes, real.columns, ("Mean marginal excess kurtosis", "RMS correlation-matrix error", "Share of scenarios outside the training range")):
        ax.bar(range(len(real)), real[col].to_numpy(), color=[COLOUR.get(i, "#777777") for i in real.index])
        ax.set_xticks(range(len(real)), real.index, rotation=15, fontsize=8)
        ax.set_title(title)
    fig.suptitle("Figure 72. Do the generated scenarios look like the data they were trained on?", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, context.figure("fig72_diffusion_realism.png"), "Does the diffusion model reproduce the fat tails and correlations of the training z-vectors, and does it invent extremes outside them?", 72)

    context.registry.log(
        "The diffusion-model 5% value-at-risk of the equal-weight 21-day return has a lower mean pinball loss than both the Gaussian and the bootstrap quantile (Benjamini-Hochberg across the two).",
        stage=STAGE, parameters={**run_cfg, "levels": list(LEVELS), "fdr": fdr},
        results={f"pinball_difference_vs_{r['b']}": float(r["mean_pinball_difference"]) for _, r in decl.iterrows()} | {f"p_value_vs_{r['b']}": float(r["p_value"]) for _, r in decl.iterrows()}
        | {f"kupiec_p_{m}_5pct": float(summary[(summary['book'] == 'equal') & (summary['level'] == 0.05) & (summary['method'] == m)]['kupiec_p'].iloc[0]) for m in METHODS},
        decision="retain" if h_diffusion else "reject", test_period="2011-01 onward, monthly origins, 15 ETFs",
        notes="Both differences must be negative and BH-significant. Kupiec tests, the 1% VaR, the inverse-volatility book and realism diagnostics are reported, not judged.")
    logger.info("STAGE 36 complete: h_diffusion=%s", h_diffusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
