"""Stage 39 - Statistical power of the platform's own tests (Generation 5, a ground-truth harness).

Every earlier 'reject' is only as informative as the test's power. Here the platform's paired Sharpe-difference test and its Diebold-Mariano CRPS test are
run on simulated data with a KNOWN effect, over the grid declared in ``config/diagnostics.yaml`` (section ``power``), to say how large an effect each test
could have found. The declared grid is not changed after seeing results. Figures 77-78.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.context import build_context
from experiments.strategies import cached_ladder
from src.backtest.engine import BacktestEngine
from src.models.probabilistic import crps_gaussian
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import diebold_mariano
from src.validation.robustness import paired_sharpe_test

STAGE = "stage39_power"
ANN = 252.0
ALPHA = 0.10


def alpha_for(delta: float, mean_b: float, sd_b: float, te_daily: float) -> float:
    """Constant daily alpha giving an exact population Sharpe difference ``delta`` when tracking noise of daily sd ``te_daily`` is added."""
    sd_a = np.sqrt(sd_b ** 2 + te_daily ** 2)
    sharpe_a = delta / np.sqrt(ANN) + mean_b / sd_b
    return float(sharpe_a * sd_a - mean_b)


def sharpe_cell(base: np.ndarray, delta: float, te: float, years: float, reps: int, seed: int, n_boot: int) -> dict:
    n = int(round(ANN * years))
    mean_b, sd_b = float(base.mean()), float(base.std(ddof=1))
    te_d = te / np.sqrt(ANN)
    alpha = alpha_for(delta, mean_b, sd_b, te_d)
    rng = np.random.default_rng(seed)
    rejects, diffs = 0, []
    index = pd.bdate_range("2000-01-03", periods=n)
    for r in range(reps):
        starts = rng.integers(0, len(base) - 21, size=int(np.ceil(n / 21)))
        b = np.concatenate([base[s:s + 21] for s in starts])[:n]
        eps = rng.standard_t(5, size=n) / np.sqrt(5 / 3) * te_d
        a = b + alpha + eps
        t = paired_sharpe_test(pd.Series(a, index=index), pd.Series(b, index=index), n_samples=n_boot, block_length=21, seed=int(rng.integers(1 << 30)))
        rejects += int(t["p_value"] <= ALPHA)
        diffs.append(t["difference"])
    return {"sharpe_difference": delta, "tracking_error": te, "years": years, "power": rejects / reps, "mean_estimated_difference": float(np.mean(diffs)), "reps": reps}


def dm_cell(r2: float, origins: int, reps: int, seed: int, assets: int = 15, rho: float = 0.4) -> dict:
    if r2 == 0.0:      # the skilled and benchmark forecasts are then identical, the loss differential is exactly zero and the test is undefined, not "never rejecting"
        return {"r2": r2, "origins": origins, "power": float("nan"), "reps": reps}
    rng = np.random.default_rng(seed)
    rejects = 0
    for _ in range(reps):
        z = rng.normal(size=(origins, assets))
        eps = np.sqrt(rho) * rng.normal(size=(origins, 1)) + np.sqrt(1 - rho) * rng.normal(size=(origins, assets))
        y = np.sqrt(r2) * z + np.sqrt(1 - r2) * eps
        skilled = crps_gaussian(y, np.sqrt(r2) * z, np.ones_like(y)).mean(axis=1)
        bench = crps_gaussian(y, np.zeros_like(y), np.ones_like(y)).mean(axis=1)
        t = diebold_mariano(skilled, bench, lag=0, alternative="two-sided")
        rejects += int(bool(t) and t["p_value"] <= ALPHA and t["mean_loss_difference"] < 0)
    return {"r2": r2, "origins": origins, "power": rejects / reps, "reps": reps}


def harm_cell(scale: float, origins: int, reps: int, seed: int, assets: int = 15, rho: float = 0.4) -> dict:
    """POST-HOC: a forecaster whose mean is pure noise of sd ``scale`` (an overfit model) against the zero-mean benchmark; how often is it found significantly WORSE?"""
    rng = np.random.default_rng(seed)
    worse = better = 0
    for _ in range(reps):
        eps = np.sqrt(rho) * rng.normal(size=(origins, 1)) + np.sqrt(1 - rho) * rng.normal(size=(origins, assets))
        noise = scale * rng.normal(size=(origins, assets))
        model = crps_gaussian(eps, noise, np.ones_like(eps)).mean(axis=1)
        bench = crps_gaussian(eps, np.zeros_like(eps), np.ones_like(eps)).mean(axis=1)
        t = diebold_mariano(model, bench, lag=0, alternative="two-sided")
        worse += int(t["p_value"] <= ALPHA and t["mean_loss_difference"] > 0)
        better += int(t["p_value"] <= ALPHA and t["mean_loss_difference"] < 0)
    return {"noise_scale": scale, "origins": origins, "rate_significantly_worse": worse / reps, "rate_significantly_better": better / reps, "reps": reps}


def minimum_detectable(frame: pd.DataFrame, x: str, target: float = 0.80) -> float:
    """Smallest x with power >= target, linearly interpolated between grid points (NaN if the grid never reaches it)."""
    f = frame.sort_values(x)
    xs, ps = f[x].to_numpy(), f["power"].to_numpy()
    for i in range(len(xs)):
        if ps[i] >= target:
            if i == 0:
                return float(xs[0])
            return float(xs[i - 1] + (target - ps[i - 1]) / (ps[i] - ps[i - 1]) * (xs[i] - xs[i - 1]))
    return float("nan")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 39: power analysis").parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = (cfg.get("diagnostics", {}) or {})["power"]
    sg, fg = node["sharpe_test"], node["forecast_test"]
    logger.info("=" * 72)
    logger.info("STAGE 39 | statistical power of the platform's tests (Generation 5)")
    logger.info("=" * 72)
    market = context.market_data()
    engine = BacktestEngine.from_config(cfg)
    books = cached_ladder(market, cfg, context.processed, ["M0_equal_weight"])
    base = engine.run(books["M0_equal_weight"], market.returns(), "M0", market.investable, apply_vol_target=False).net_returns.dropna().to_numpy()
    logger.info("benchmark: M0 equal-weight net daily returns, %d days, annualised Sharpe %.3f", len(base), np.sqrt(ANN) * base.mean() / base.std(ddof=1))

    cells = [(d, te, y) for te in sg["grid"]["tracking_error_annual"] for y in sg["grid"]["years"] for d in sg["grid"]["sharpe_difference"]]
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=-1)(delayed(sharpe_cell)(base, d, te, y, int(sg["replications"]), int(sg["seed"]) + k, 400) for k, (d, te, y) in enumerate(cells))
    sharpe_power = pd.DataFrame(rows)
    logger.info("Sharpe-test grid (%d cells x %d reps) in %.0fs", len(cells), int(sg["replications"]), time.perf_counter() - t0)
    context.save_table(sharpe_power, "stage39_sharpe_power.csv", index=False)
    mde = sharpe_power[sharpe_power["sharpe_difference"] > 0].groupby(["tracking_error", "years"]).apply(lambda f: minimum_detectable(f.assign(power=f["power"]), "sharpe_difference"), include_groups=False).rename("min_detectable_sharpe_difference").reset_index()
    context.save_table(mde, "stage39_sharpe_mde.csv", index=False)
    size = sharpe_power[sharpe_power["sharpe_difference"] == 0].set_index(["tracking_error", "years"])["power"]
    logger.info("false-positive rate at difference 0 (nominal %.2f):\n%s", ALPHA, size.round(3).to_string())
    logger.info("minimum detectable Sharpe difference (power 0.80):\n%s", mde.round(3).to_string(index=False))

    dm_rows = [dm_cell(r2, o, int(fg["replications"]), int(fg["seed"]) + k) for k, (r2, o) in enumerate((r2, o) for o in fg["grid"]["origins"] for r2 in fg["grid"]["r2"])]
    dm = pd.DataFrame(dm_rows)
    context.save_table(dm, "stage39_forecast_power.csv", index=False)
    harm = pd.DataFrame([harm_cell(sc, o, 400, 5000 + k) for k, (sc, o) in enumerate((sc, o) for o in (100, 187, 400) for sc in (0.05, 0.1, 0.2))])
    context.save_table(harm, "stage39_forecast_harm_posthoc.csv", index=False)
    logger.info("POST-HOC: pure-noise forecaster vs the zero-mean benchmark:\n%s", harm.round(3).to_string(index=False))
    dm_mde = dm[dm["r2"] > 0].groupby("origins").apply(lambda f: minimum_detectable(f, "r2"), include_groups=False).rename("min_detectable_r2").reset_index()
    context.save_table(dm_mde, "stage39_forecast_mde.csv", index=False)
    logger.info("DM-CRPS power:\n%s", dm.pivot(index="r2", columns="origins", values="power").round(3).to_string())
    logger.info("minimum detectable r2:\n%s", dm_mde.round(4).to_string(index=False))

    # earlier results read against the grid (approximate: nearest declared tracking error and years)
    ctx_rows = []
    sigma_b = float(base.std(ddof=1) * np.sqrt(ANN))
    for name, path in (("stage31_allocation", "stage31_allocation_tests.csv"), ("stage38_rl", "stage38_tests.csv")):
        t = pd.read_csv(context.tables / path)
        for _, r in t.iterrows():
            te = sigma_b * np.sqrt(2 * (1 - r["return_correlation"]))
            te_bin = min(sg["grid"]["tracking_error_annual"], key=lambda x: abs(x - te))
            yrs = r["n_obs"] / ANN
            y_bin = min(sg["grid"]["years"], key=lambda x: abs(x - yrs))
            m = mde[(mde["tracking_error"] == te_bin) & (mde["years"] == y_bin)]["min_detectable_sharpe_difference"]
            ctx_rows.append({"source": name, "a": r["a"], "b": r["b"], "observed_difference": r["difference"], "p_value": r["p_value"], "years": yrs, "implied_tracking_error": te,
                             "grid_tracking_error": te_bin, "grid_years": y_bin, "min_detectable_difference_at_grid": float(m.iloc[0]) if len(m) else float("nan")})
    context.save_table(pd.DataFrame(ctx_rows), "stage39_earlier_results_in_context.csv", index=False)
    logger.info("earlier results in context:\n%s", pd.DataFrame(ctx_rows).round(3).to_string(index=False))

    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    for ax, te in zip(axes[:2], sg["grid"]["tracking_error_annual"]):
        for y, c in zip(sg["grid"]["years"], PALETTE):
            f = sharpe_power[(sharpe_power["tracking_error"] == te) & (sharpe_power["years"] == y)].sort_values("sharpe_difference")
            ax.plot(f["sharpe_difference"], f["power"], marker="o", color=c, label=f"{y:g} years")
        ax.axhline(0.8, color="black", linestyle=":")
        ax.axhline(ALPHA, color="black", linestyle="--", linewidth=0.7)
        ax.set_xlabel("True Sharpe difference")
        ax.set_ylabel("Rejection rate at p <= 0.10")
        ax.set_title(f"Paired Sharpe test, tracking error {te:.0%}")
        ax.legend(fontsize=8)
    ax = axes[2]
    for o, c in zip(fg["grid"]["origins"], PALETTE):
        f = dm[dm["origins"] == o].sort_values("r2")
        ax.plot(f["r2"] * 100, f["power"], marker="o", color=c, label=f"{o} monthly origins")
    ax.axhline(0.8, color="black", linestyle=":")
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel("True out-of-sample R-squared of the mean forecast, %")
    ax.set_ylabel("Rejection rate (negative CRPS difference)")
    ax.set_title("Diebold-Mariano on CRPS")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 77. What the platform's tests can and cannot detect (dotted 80% power, dashed nominal size)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, context.figure("fig77_power_curves.png"), "How large does a true Sharpe edge or forecast R-squared have to be before the platform's tests would notice it?", 77)

    fig, axes = new_axes(1, 2, figsize=(12, 5))
    ax = axes[0]
    piv = mde.pivot(index="tracking_error", columns="years", values="min_detectable_sharpe_difference")
    im = ax.imshow(piv.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{c:g}y" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{i:.0%}" for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iloc[i, j]:.2f}" if np.isfinite(piv.iloc[i, j]) else ">0.60", ha="center", va="center", color="white" if np.isfinite(piv.iloc[i, j]) else "black")
    ax.set_xlabel("Sample length")
    ax.set_ylabel("Tracking error vs benchmark")
    ax.set_title("Minimum detectable Sharpe diff. (80%)", fontsize=11)
    ax = axes[1]
    c = pd.DataFrame(ctx_rows)
    ax.scatter(c["min_detectable_difference_at_grid"], c["observed_difference"].abs(), color=[PALETTE[0] if s.startswith("stage31") else PALETTE[3] for s in c["source"]])
    lim = max(c["min_detectable_difference_at_grid"].max(), c["observed_difference"].abs().max()) * 1.1
    ax.plot([0, lim], [0, lim], color="black", linewidth=0.8)
    ax.set_xlabel("Minimum detectable difference for that test's sample")
    ax.set_ylabel("Observed absolute difference")
    ax.set_title("Earlier comparisons (below line: undetectable)", fontsize=11)
    fig.suptitle("Figure 78. Earlier results read against the power grid", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, context.figure("fig78_power_context.png"), "Which of the platform's earlier null Sharpe comparisons were simply too small to detect?", 78)

    context.registry.log(
        "Descriptive: the platform's paired Sharpe test holds its size and has the stated minimum detectable differences; the CRPS Diebold-Mariano test has the stated minimum detectable R-squared.",
        stage=STAGE, parameters={"alpha": ALPHA, "sharpe_reps": int(sg["replications"]), "dm_reps": int(fg["replications"])},
        results={"size_sharpe_test_mean": float(size.mean()), "mde_sharpe_6pct_te_15y": float(mde[(mde["tracking_error"] == 0.06) & (mde["years"] == 15.7)]["min_detectable_sharpe_difference"].iloc[0]),
                 "mde_sharpe_3pct_te_15y": float(mde[(mde["tracking_error"] == 0.03) & (mde["years"] == 15.7)]["min_detectable_sharpe_difference"].iloc[0]),
                 "mde_r2_187_origins": float(dm_mde[dm_mde["origins"] == 187]["min_detectable_r2"].iloc[0]), "harm_detection_noise_0.1_187": float(harm[(harm["noise_scale"] == 0.1) & (harm["origins"] == 187)]["rate_significantly_worse"].iloc[0])},
        decision="record", notes="A diagnostic. It does not change any earlier decision; it says what a rejection could and could not mean.")
    logger.info("STAGE 39 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
