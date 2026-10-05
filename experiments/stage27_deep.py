"""Stage 27 - Modern deep learning (Generation 4, Priority 17).

A patch transformer (PatchTST-style) and an MLP-mixer (TSMixer-style) forecast the 21-day return of each of the 15
ETFs from the 252 normalised daily returns before the origin. They are held to the Stage 19 rule: do they have a lower
mean CRPS than the annually refitted price-only ridge, with the same volatility forecast, on the same
(origin, asset) rows? A linear ridge on the same window is run as a control (Zeng et al.: transformers are often no
better than a linear model on the raw window). Rules: ``config/deeplearning.yaml``.

Refits are independent, so they run as tasks in the Stage 26 executor.

Figures 54-55.
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.distributed.executor import run_tasks
from src.models.deep_forecast import (TrainSettings, daily_sigma, normalised_windows, parameter_count, ridge_window,
                                      train_and_predict)
from src.models.probabilistic import crps_gaussian, ewma_sigma
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg, diebold_mariano
from experiments.context import build_context

STAGE = "stage27_deep"
DEEP = ["patchtst", "tsmixer"]
CONTROLS = ["linear_window"]
LABEL = {"patchtst": "patch transformer", "tsmixer": "MLP-mixer", "linear_window": "linear on window (control)", "frozen_ridge": "Stage 19 ridge",
         "historical_mean": "historical mean"}
COLOUR = {"patchtst": PALETTE[0], "tsmixer": PALETTE[1], "linear_window": PALETTE[2], "frozen_ridge": "#555555", "historical_mean": PALETTE[3]}


def refit_task(task: dict, shared: SimpleNamespace) -> dict:
    """One walk-forward refit for one model: train rows, validation block, test rows -> predictions per seed."""
    start, kind = task["start"], task["model"]
    cfg = shared.cfg
    tr = shared.rows
    horizon, embargo, step = shared.horizon, shared.embargo, shared.step
    usable = (tr["day"] + horizon + embargo <= start) & tr["valid"] & np.isfinite(tr["z"])
    days = np.unique(tr["day"][usable])
    cut_day = days[int(np.floor(0.8 * len(days)))]
    train_mask = usable & (tr["day"] <= cut_day - 42)
    val_mask = usable & (tr["day"] > cut_day)
    test = shared.test[start]
    out = {"start": start, "model": kind, "origin": test["origin"], "asset": test["asset"], "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum())}
    if kind == "linear_window":
        fit_mask = usable
        out["mu_z"] = ridge_window(tr["X"][fit_mask], tr["z"][fit_mask], test["X"], float(cfg["models"]["linear_window"]["ridge_alpha"]))
        return out
    net_cfg = cfg["models"][kind]
    kwargs = {"d_model": net_cfg["d_model"], "layers": net_cfg["layers"], "ff": net_cfg["ff"], "dropout": net_cfg["dropout"]}
    if kind == "patchtst":
        kwargs["heads"] = net_cfg["heads"]
    t = cfg["training"]
    settings = TrainSettings(float(t["learning_rate"]), float(t["weight_decay"]), int(t["batch_size"]), int(t["max_epochs"]), int(t["patience"]))
    preds, curves, best = [], [], []
    for seed in t["ensemble_seeds"]:
        res = train_and_predict(kind, tr["X"][train_mask], tr["asset"][train_mask], tr["z"][train_mask], tr["X"][val_mask], tr["asset"][val_mask],
                                tr["z"][val_mask], test["X"], test["asset_idx"], int(seed), shared.n_assets, kwargs, settings)
        preds.append(res["pred"])
        curves.append(res["val_loss"])
        best.append(res["best_epoch"])
    out.update({"mu_z": np.mean(preds, axis=0), "seed_preds": np.stack(preds), "val_curves": curves, "best_epochs": best})
    return out


def score(pred: pd.DataFrame, frozen: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pred.merge(frozen[["origin", "asset", "mu", "sigma", "target", "mu_benchmark"]].rename(columns={"mu": "mu_frozen"}),
                        on=["origin", "asset"]).dropna(subset=[column, "mu_frozen", "sigma", "target"])
    y, s = merged["target"].to_numpy(), merged["sigma"].to_numpy()
    scored = pd.DataFrame({"origin": merged["origin"], "crps_model": crps_gaussian(y, merged[column].to_numpy(), s),
                           "crps_frozen": crps_gaussian(y, merged["mu_frozen"].to_numpy(), s),
                           "se_model": (y - merged[column].to_numpy()) ** 2, "se_frozen": (y - merged["mu_frozen"].to_numpy()) ** 2})
    return scored.groupby("origin").mean(), merged


def mean_ic(merged: pd.DataFrame, column: str) -> float:
    ics = [spearmanr(g[column], g["target"])[0] for _, g in merged.groupby("origin") if len(g) >= 5 and g[column].nunique() > 1]
    return float(np.nanmean(ics))


def figure_scores(cumulative: pd.DataFrame, table: pd.DataFrame, by_year: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for name in cumulative.columns:
        ax.plot(cumulative.index, cumulative[name].to_numpy(), color=COLOUR[name], linewidth=1.5, label=LABEL[name])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative CRPS advantage over the Stage 19 ridge\n(above zero = better)")
    ax.set_title("Out of sample, month by month")
    ax.legend(fontsize=8)
    ax = axes[1]
    names = list(table.index)
    diff = -table["mean_crps_difference"].to_numpy() * 1e4
    ax.bar(range(len(names)), diff, color=[COLOUR[n] for n in names])
    ax.axhline(0, color="black", linewidth=0.8)
    for i, (n, row) in enumerate(table.iterrows()):
        ax.text(i, -row["mean_crps_difference"] * 1e4, f"p={row['p_value']:.3f}" + (" *" if row["bh_significant"] else ""), ha="center",
                va="bottom" if diff[i] >= 0 else "top", fontsize=8)
    ax.set_xticks(range(len(names)), [LABEL[n] for n in names], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("CRPS advantage, units of 1e-4 (above zero = better)")
    ax.set_title("Mean advantage over the Stage 19 ridge")
    ax = axes[2]
    for name in by_year.columns:
        ax.plot(by_year.index, by_year[name].to_numpy() * 1e4, marker="o", color=COLOUR[name], linewidth=1.2, label=LABEL[name])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("CRPS advantage by calendar year, 1e-4")
    ax.set_title("Is any year doing the work?")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 54. Deep forecasters against the Stage 19 ridge", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Do a patch transformer and an MLP-mixer forecast 21-day ETF returns better than the annually refitted price-only ridge, "
                           "on the same rows and with the same volatility forecast?", 54)


def figure_training(curves: dict, seed_ic: pd.DataFrame, params: dict, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for kind, runs in curves.items():
        for j, c in enumerate(runs):
            ax.plot(range(1, len(c) + 1), c, color=COLOUR[kind], alpha=0.8, linewidth=1.2, label=LABEL[kind] if j == 0 else None)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE of the z-scored return")
    ax.set_title("Validation loss, latest refit (dotted = predicting zero ~ 1)")
    ax.legend(fontsize=8)
    ax = axes[1]
    positions = np.arange(len(seed_ic.columns))
    for j, name in enumerate(seed_ic.columns):
        ax.scatter(np.full(len(seed_ic), j) + np.linspace(-0.12, 0.12, len(seed_ic)), seed_ic[name], color=COLOUR[name.split(":")[0]], s=30)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, [c.replace(":", "\n") for c in seed_ic.columns], fontsize=8)
    ax.set_ylabel("Mean rank IC out of sample")
    ax.set_title("Single seeds against the 3-seed ensemble")
    ax = axes[2]
    ax.bar(range(len(params)), list(params.values()), color=[COLOUR[k] for k in params])
    ax.set_xticks(range(len(params)), [LABEL[k] for k in params], rotation=15, ha="right", fontsize=8)
    for i, v in enumerate(params.values()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Parameters")
    ax.set_title("Model size")
    fig.suptitle("Figure 55. Training behaviour, seed variance and size", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "How well do the deep models fit, how much of their skill is seed noise, and how big are they next to the 252-feature linear control?", 55)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 27: deep learning")
    parser.add_argument("--backend", default=None)
    args = parser.parse_args(argv)
    context, logger = build_context(STAGE, generation=4)
    cfg = context.config
    node = cfg.get("deeplearning", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 27 | modern deep learning (Generation 4, Priority 17)")
    logger.info("=" * 72)
    fdr = float((node.get("decisions", {}) or {}).get("fdr", 0.10))
    task_cfg = node["task"]
    horizon = int(cfg.get("forecasting.horizon_days", 21))
    embargo = int(task_cfg["walk_forward"]["embargo_days"])
    window = int(task_cfg["window"])
    backend = args.backend or str((cfg.get("distributed.parallel", {}) or {}).get("backend", "joblib"))
    frozen_path = context.tables / "stage19_predictions_price_only.csv"
    if not frozen_path.exists():
        raise FileNotFoundError("Stage 19 must run first: its price-only predictions are this stage's benchmark")
    frozen = pd.read_csv(frozen_path, parse_dates=["origin"])

    market = context.market_data()
    prices = market.prices
    returns = prices.pct_change()
    index = pd.DatetimeIndex(prices.index)
    assets = list(prices.columns)
    half = float(cfg.get("forecasting.gaussian.vol_halflife", 40))
    sigma_d = daily_sigma(returns, half)
    sigma_h = ewma_sigma(returns, half, horizon)
    forward = prices.shift(-horizon) / prices - 1.0

    # training rows: every 5th day, all assets pooled
    step = 5
    days = np.arange(window, len(index) - horizon, step)
    day_grid = np.repeat(days, len(assets))
    asset_grid = np.tile(np.arange(len(assets)), len(days))
    X, valid = normalised_windows(returns, sigma_d, day_grid, asset_grid, window)
    target = forward.to_numpy()[day_grid, asset_grid]
    sig = sigma_h.to_numpy()[day_grid, asset_grid]
    z = np.clip(target / sig, -5.0, 5.0)
    rows = {"X": X, "valid": valid, "z": z, "day": day_grid, "asset": asset_grid}

    # test rows are exactly the Stage 19 (origin, asset) rows of each refit
    test = {}
    for start, g in frozen.groupby("refit_day"):
        pos = np.array([index.get_loc(o) for o in g["origin"]])
        a = np.array([assets.index(x) for x in g["asset"]])
        Xt, ok = normalised_windows(returns, sigma_d, pos, a, window)
        keep = ok
        test[int(start)] = {"X": Xt[keep], "asset_idx": a[keep], "origin": g["origin"].to_numpy()[keep], "asset": g["asset"].to_numpy()[keep],
                            "sigma": g["sigma"].to_numpy()[keep]}
    shared = SimpleNamespace(cfg=node, rows=rows, test=test, horizon=horizon, embargo=embargo, step=step, n_assets=len(assets))
    models = DEEP + CONTROLS
    tasks = [{"start": s, "model": m} for s in sorted(test) for m in models]
    logger.info("%d refits x %d models = %d tasks; %d training rows (every %dth day), %d assets", len(test), len(models), len(tasks), int(valid.sum()), step, len(assets))
    t0 = time.perf_counter()
    results = run_tasks(partial(refit_task, shared=shared), tasks, backend, int((cfg.get("distributed.parallel", {}) or {}).get("n_jobs", -1)), 1)
    logger.info("trained in %.0fs on backend %s", time.perf_counter() - t0, backend)

    # --------------------------------------------------------------- assemble
    preds = {}
    seed_frames = {}
    for kind in models:
        parts = []
        for r in (x for x in results if x["model"] == kind):
            t = test[r["start"]]
            df = pd.DataFrame({"origin": r["origin"], "asset": r["asset"], "mu_z": r["mu_z"], "sigma_model": t["sigma"]})
            df["mu"] = df["mu_z"] * df["sigma_model"]
            parts.append(df)
            if "seed_preds" in r:
                for k, seed_pred in enumerate(r["seed_preds"]):
                    seed_frames.setdefault((kind, k), []).append(pd.DataFrame({"origin": r["origin"], "asset": r["asset"], "mu": seed_pred * t["sigma"]}))
        preds[kind] = pd.concat(parts, ignore_index=True)
        context.save_table(preds[kind], f"stage27_predictions_{kind}.csv", index=False)

    rows_out, scores, merged_all = [], {}, {}
    for kind in models:
        s, merged = score(preds[kind], frozen, "mu")
        scores[kind], merged_all[kind] = s, merged
        test_r = diebold_mariano(s["crps_model"], s["crps_frozen"], lag=0, alternative="two-sided")
        rows_out.append({"model": kind, "n_origins": len(s), "n_rows": len(merged), "mean_crps_model": float(s["crps_model"].mean()), "mean_crps_frozen": float(s["crps_frozen"].mean()),
                         "mean_crps_difference": float(test_r["mean_loss_difference"]), "statistic": float(test_r["statistic"]), "p_value": float(test_r["p_value"]),
                         "mse_ratio": float(s["se_model"].mean() / s["se_frozen"].mean()), "mean_ic_model": mean_ic(merged, "mu"), "mean_ic_frozen": mean_ic(merged, "mu_frozen")})
    table = pd.DataFrame(rows_out).set_index("model")
    deep = table.loc[DEEP]
    table["bh_significant"] = False
    table.loc[DEEP, "bh_significant"] = benjamini_hochberg(deep["p_value"], fdr)
    table["passes"] = table["bh_significant"] & (table["mean_crps_difference"] < 0) & table.index.isin(DEEP)
    context.save_table(table, "stage27_forecast_tests.csv")
    logger.info("against the annually refitted ridge (CRPS, lower is better):\n%s", table[["n_origins", "mean_crps_model", "mean_crps_frozen", "mean_crps_difference", "p_value",
                                                                                           "bh_significant", "mse_ratio", "mean_ic_model", "mean_ic_frozen"]].round(6).to_string())
    h_deep = bool(table.loc[DEEP, "passes"].any())

    # historical-mean benchmark (reported), and the control's relation to the deep models
    bench = frozen.rename(columns={"mu_benchmark": "mu_hist"})[["origin", "asset", "mu_hist"]]
    bench_scored, _ = score(bench.assign(mu=bench["mu_hist"]), frozen, "mu")
    t_hist = diebold_mariano(bench_scored["crps_model"], bench_scored["crps_frozen"], lag=0, alternative="two-sided")
    context.save_table(pd.DataFrame([{"model": "historical_mean", "mean_crps_difference": float(t_hist["mean_loss_difference"]), "p_value": float(t_hist["p_value"])}]),
                       "stage27_historical_mean_vs_ridge.csv", index=False)
    rel = []
    for kind in DEEP:
        m = preds[kind].merge(preds["linear_window"][["origin", "asset", "mu"]].rename(columns={"mu": "mu_lin"}), on=["origin", "asset"])
        m = m.merge(frozen[["origin", "asset", "sigma", "target"]], on=["origin", "asset"]).dropna()
        a = pd.DataFrame({"origin": m["origin"], "d": crps_gaussian(m["target"].to_numpy(), m["mu"].to_numpy(), m["sigma"].to_numpy())
                          - crps_gaussian(m["target"].to_numpy(), m["mu_lin"].to_numpy(), m["sigma"].to_numpy())}).groupby("origin")["d"].mean()
        t_rel = diebold_mariano(a, pd.Series(0.0, index=a.index), lag=0, alternative="two-sided")
        rel.append({"model": kind, "mean_crps_difference_vs_linear_window": float(t_rel["mean_loss_difference"]), "p_value": float(t_rel["p_value"])})
    context.save_table(pd.DataFrame(rel), "stage27_deep_vs_linear_window.csv", index=False)
    logger.info("deep against the linear control:\n%s", pd.DataFrame(rel).round(6).to_string(index=False))

    # post-hoc (labelled, never overturns the decision): is the gain skill, or simply a forecast that is closer to zero?
    base = frozen[["origin", "asset", "mu", "sigma", "target", "mu_benchmark"]].dropna()
    candidates = {"frozen_ridge": base.set_index(["origin", "asset"])["mu"], "historical_mean": base.set_index(["origin", "asset"])["mu_benchmark"],
                  "zero": base.set_index(["origin", "asset"])["mu"] * 0.0}
    for kind in models:
        candidates[kind] = preds[kind].set_index(["origin", "asset"])["mu"]
    ref = base.set_index(["origin", "asset"])
    post_rows = []
    for a, b in [("patchtst", "zero"), ("tsmixer", "zero"), ("patchtst", "historical_mean"), ("tsmixer", "historical_mean"), ("linear_window", "zero"),
                 ("frozen_ridge", "zero"), ("frozen_ridge", "historical_mean"), ("tsmixer", "frozen_ridge")]:
        idx = candidates[a].index.intersection(candidates[b].index).intersection(ref.index)
        y, sg = ref.loc[idx, "target"].to_numpy(), ref.loc[idx, "sigma"].to_numpy()
        d = pd.Series(crps_gaussian(y, candidates[a].loc[idx].to_numpy(), sg) - crps_gaussian(y, candidates[b].loc[idx].to_numpy(), sg), index=idx).groupby(level=0).mean()
        t_p = diebold_mariano(d, pd.Series(0.0, index=d.index), lag=0, alternative="two-sided")
        post_rows.append({"a": a, "b": b, "n_origins": len(d), "mean_crps_difference_a_minus_b": float(t_p["mean_loss_difference"]), "p_value": float(t_p["p_value"]),
                          "mean_abs_forecast_a": float(candidates[a].loc[idx].abs().mean()), "mean_abs_forecast_b": float(candidates[b].loc[idx].abs().mean())})
    post = pd.DataFrame(post_rows)
    context.save_table(post, "stage27_posthoc_controls.csv", index=False)
    logger.info("post-hoc controls (CRPS a minus b, negative favours a):\n%s", post.round(6).to_string(index=False))

    # seed diagnostics, size, curves
    seed_ic = {}
    for kind in DEEP:
        seed_ic[f"{kind}:ensemble"] = [table.loc[kind, "mean_ic_model"]]
        for k in range(len(node["training"]["ensemble_seeds"])):
            sp = pd.concat(seed_frames[(kind, k)], ignore_index=True)
            _, merged = score(sp, frozen, "mu")
            seed_ic.setdefault(f"{kind}:seed {k + 1}", []).append(mean_ic(merged, "mu"))
    seed_table = pd.DataFrame({k: [v[0]] for k, v in seed_ic.items()}).T.rename(columns={0: "mean_ic"})
    context.save_table(seed_table, "stage27_seed_ic.csv")
    last_start = max(test)
    curves = {kind: next(r["val_curves"] for r in results if r["model"] == kind and r["start"] == last_start) for kind in DEEP}
    net = node["models"]
    params = {k: parameter_count(k, len(assets), d_model=net[k]["d_model"], layers=net[k]["layers"], ff=net[k]["ff"], dropout=net[k]["dropout"],
                                 **({"heads": net[k]["heads"]} if k == "patchtst" else {})) for k in DEEP}
    params["linear_window"] = window + len(assets) + 1
    context.save_table(pd.DataFrame({"parameters": params}), "stage27_parameter_counts.csv")
    best_epochs = pd.DataFrame([{"model": r["model"], "refit_day": r["start"], "n_train": r["n_train"], "n_val": r["n_val"], "best_epochs": str(r["best_epochs"])} for r in results if "best_epochs" in r])
    context.save_table(best_epochs, "stage27_training_log.csv", index=False)

    # figures
    advantage = pd.DataFrame({k: (scores[k]["crps_frozen"] - scores[k]["crps_model"]) for k in models})
    cumulative = advantage.cumsum()
    by_year = advantage.groupby(advantage.index.year).mean()
    figure_scores(cumulative, table, by_year, context.figure("fig54_deep_scores.png"))
    figure_training(curves, seed_table.T.rename(index={"mean_ic": 0}) if False else pd.DataFrame({k: [v] for k, v in seed_table["mean_ic"].items()}), params, context.figure("fig55_deep_training.png"))

    context.registry.log(
        "A patch transformer or an MLP-mixer forecasts 21-day ETF returns with a significantly lower mean CRPS than the Stage 19 annually refitted "
        "price-only ridge (same volatility forecast, same rows).",
        stage=STAGE, parameters={"models": DEEP, "window": window, "seeds": node["training"]["ensemble_seeds"], "refits": len(test), "fdr": fdr},
        results={**{f"crps_difference_{k}": float(table.loc[k, "mean_crps_difference"]) for k in models}, **{f"p_value_{k}": float(table.loc[k, "p_value"]) for k in models},
                 "mean_crps_ridge": float(table.loc["patchtst", "mean_crps_frozen"]), "n_origins": int(table.loc["patchtst", "n_origins"]),
                 **{f"mean_ic_{k}": float(table.loc[k, "mean_ic_model"]) for k in models}, "mean_ic_ridge": float(table.loc["patchtst", "mean_ic_frozen"])},
        decision="retain" if h_deep else "reject", test_period="2011-01 onward, monthly origins, 15 ETFs",
        notes="Retained only if a deep model has a significantly lower CRPS after BH control across the two, with a negative loss difference. The linear-window "
              "control is outside the family. Zero-shot foundation models are not run: their pre-training overlaps the sample (config/deeplearning.yaml).",
    )
    logger.info("STAGE 27 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
