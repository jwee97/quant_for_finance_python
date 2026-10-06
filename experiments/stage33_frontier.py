"""Stage 33 - The deep-forecasting family, graph networks and Bayesian deep learning (Generation 5).

Stage 27 found that the "win" of a deep model over the Stage 19 ridge was the ridge's weakness: the mixer was no better than an asset's
own historical mean. This stage therefore holds N-BEATS, N-HiTS, a TimeMixer-style multiscale mixer and a graph attention network to the
HISTORICAL-MEAN forecast (same volatility forecast, same rows), declares one family-wise rule (BH across the four), and then asks whether a
deep ensemble plus Monte Carlo dropout gives a better *spread* than the EWMA volatility alone. Rules: ``config/frontier.yaml`` (section
``deep_family``), committed before this stage was run.

Figures 66-67.
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from experiments.context import build_context
from src.distributed.executor import run_tasks
from src.models.deep_forecast import TrainSettings, daily_sigma, normalised_windows, parameter_count, train_and_predict
from src.models.deep_family import FAMILY
from src.models.graph_forecast import build_graph_network, correlation_neighbours, train_and_predict_graph
from src.models.probabilistic import crps_gaussian, ewma_sigma, price_features
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg, diebold_mariano

STAGE = "stage33_frontier"
MODELS = ["nbeats", "nhits", "timemixer", "gat"]
LABEL = {"nbeats": "N-BEATS", "nhits": "N-HiTS", "timemixer": "TimeMixer-style", "gat": "graph attention", "historical_mean": "historical mean",
         "frozen_ridge": "Stage 19 ridge", "patchtst": "patch transformer (S27)", "tsmixer": "MLP-mixer (S27)"}
COLOUR = {"nbeats": PALETTE[0], "nhits": PALETTE[1], "timemixer": PALETTE[2], "gat": PALETTE[4], "historical_mean": "#555555", "frozen_ridge": PALETTE[3]}
KEYS = ["origin", "asset"]


def settings_from(cfg: dict) -> TrainSettings:
    t = cfg["training"]
    return TrainSettings(float(t["learning_rate"]), float(t["weight_decay"]), int(t["batch_size"]), int(t["max_epochs"]), int(t["patience"]))


def split_masks(day: np.ndarray, ok: np.ndarray, start: int, horizon: int, embargo: int) -> tuple[np.ndarray, np.ndarray]:
    """The Stage 27 split: rows matured before ``start`` (with embargo); the last 20% of days validate, with a 42-day gap before them."""
    usable = (day + horizon + embargo <= start) & ok
    days = np.unique(day[usable])
    cut = days[int(np.floor(0.8 * len(days)))]
    return usable & (day <= cut - 42), usable & (day > cut)


def refit_task(task: dict, shared: SimpleNamespace) -> dict:
    start, kind, cfg = task["start"], task["model"], shared.cfg
    t = cfg["training"]
    settings = settings_from(cfg)
    mc = int(cfg["bayesian"]["mc_samples"])
    out = {"start": start, "model": kind}
    preds, mcs, curves, best = [], [], [], []
    if kind == "gat":
        d = shared.days
        train_m, val_m = split_masks(d["day"], np.isfinite(d["z"]).any(axis=1), start, shared.horizon, shared.embargo)
        feats = d["F"][train_m].reshape(-1, d["F"].shape[2])
        mean, std = np.nanmean(feats, axis=0), np.nanstd(feats, axis=0, ddof=1)
        std[~np.isfinite(std) | (std == 0)] = 1.0
        prep = lambda F: np.nan_to_num((F - mean) / std, nan=0.0)
        test = shared.gat_test[start]
        node = cfg["models"]["gat"]
        kwargs = {"hidden": node["hidden"], "heads": node["heads"], "dropout": node["dropout"]}
        for seed in t["ensemble_seeds"]:
            r = train_and_predict_graph(prep(d["F"][train_m]), d["N"][train_m], d["z"][train_m], prep(d["F"][val_m]), d["N"][val_m], d["z"][val_m],
                                        prep(test["F"]), test["N"], int(seed), shared.n_assets, settings, kwargs, mc)
            preds.append(r["pred"][test["row_day"], test["row_asset"]])
            mcs.append(r["mc_pred"][:, test["row_day"], test["row_asset"]])
            curves.append(r["val_loss"])
            best.append(r["best_epoch"])
        out.update({"origin": test["origin"], "asset": test["asset"], "n_train": int(train_m.sum()), "n_val": int(val_m.sum())})
    else:
        rows = shared.rows
        train_m, val_m = split_masks(rows["day"], rows["valid"] & np.isfinite(rows["z"]), start, shared.horizon, shared.embargo)
        test = shared.test[start]
        kwargs = {"family": {k: v for k, v in cfg["models"][kind].items() if k != "description"}}
        for seed in t["ensemble_seeds"]:
            r = train_and_predict(kind, rows["X"][train_m], rows["asset"][train_m], rows["z"][train_m], rows["X"][val_m], rows["asset"][val_m],
                                  rows["z"][val_m], test["X"], test["asset_idx"], int(seed), shared.n_assets, kwargs, settings, mc_samples=mc)
            preds.append(r["pred"])
            mcs.append(r["mc_pred"])
            curves.append(r["val_loss"])
            best.append(r["best_epoch"])
        out.update({"origin": test["origin"], "asset": test["asset"], "n_train": int(train_m.sum()), "n_val": int(val_m.sum())})
    out["mu_z"] = np.mean(preds, axis=0)
    out["seed_preds"] = np.stack(preds)
    out["epistemic_var"] = np.concatenate(mcs, axis=0).var(axis=0)          # variance of z over 3 seeds x mc passes
    out["val_curves"], out["best_epochs"] = curves, best
    return out


def gaussian_scores(frame: pd.DataFrame, mu: str, sigma: str) -> pd.Series:
    return pd.Series(crps_gaussian(frame["target"].to_numpy(), frame[mu].to_numpy(), frame[sigma].to_numpy()), index=frame.index)


def by_origin(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.groupby(frame["origin"]).mean()


def mean_ic(frame: pd.DataFrame, column: str) -> float:
    ics = [spearmanr(g[column], g["target"])[0] for _, g in frame.groupby("origin") if len(g) >= 5 and g[column].nunique() > 1]
    return float(np.nanmean(ics))


def figure_scores(cumulative: pd.DataFrame, table: pd.DataFrame, by_year: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for name in cumulative.columns:
        ax.plot(cumulative.index, cumulative[name].to_numpy(), color=COLOUR[name], linewidth=1.5, label=LABEL[name])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative CRPS advantage over the historical mean\n(above zero = better)")
    ax.set_title("Out of sample, month by month")
    ax.legend(fontsize=8)
    ax = axes[1]
    names = list(table.index)
    adv = -table["mean_crps_difference"].to_numpy() * 1e4
    ax.bar(range(len(names)), adv, color=[COLOUR[n] for n in names])
    ax.axhline(0, color="black", linewidth=0.8)
    for i, (n, row) in enumerate(table.iterrows()):
        ax.text(i, adv[i], f"p={row['p_value']:.3f}" + (" *" if row["bh_significant"] else ""), ha="center", va="bottom" if adv[i] >= 0 else "top", fontsize=8)
    ax.set_xticks(range(len(names)), [LABEL[n] for n in names], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("CRPS advantage, 1e-4 (above zero = better)")
    ax.set_title("Mean advantage over the historical mean (* = BH-significant)")
    ax = axes[2]
    for name in by_year.columns:
        ax.plot(by_year.index, by_year[name].to_numpy() * 1e4, marker="o", color=COLOUR[name], linewidth=1.2, label=LABEL[name])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("CRPS advantage by calendar year, 1e-4")
    ax.set_title("Is any year doing the work?")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 66. N-BEATS, N-HiTS, TimeMixer-style and graph attention against the historical mean", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does any modern deep forecaster beat an asset's own historical mean for the 21-day return, once the volatility forecast is held fixed?", 66)


def figure_bayes(coverage: pd.DataFrame, widening: pd.DataFrame, seed_ic: pd.DataFrame, params: dict, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    x = np.arange(len(coverage))
    w = 0.38
    ax.bar(x - w / 2, coverage["without"], w, color="#999999", label="EWMA sigma")
    ax.bar(x + w / 2, coverage["with"], w, color=PALETTE[0], label="widened by epistemic term")
    ax.set_xticks(x, coverage.index, fontsize=8)
    for level, ls in ((0.5, ":"), (0.9, "--")):
        ax.axhline(level, color="black", linestyle=ls, linewidth=0.9)
    ax.set_ylabel("Share of realised returns inside the interval")
    ax.set_title("Interval coverage, best model")
    ax.set_ylim(0, 1.2)
    ax.text(0.5, 0.97, "dotted = 50% target, dashed = 90% target", ha="center", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    for name in widening.columns:
        ax.plot(widening.index, widening[name].to_numpy(), color=COLOUR[name], linewidth=1.2, label=LABEL[name])
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_ylabel("Forecast std ratio: widened / EWMA, monthly mean")
    ax.set_title("Widening from the epistemic term")
    ax.legend(fontsize=8)
    ax = axes[2]
    for j, name in enumerate(seed_ic.columns):
        ax.scatter(np.full(len(seed_ic), j) + np.linspace(-0.12, 0.12, len(seed_ic)), seed_ic[name], color=COLOUR[name], s=30)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(seed_ic.columns)), [f"{LABEL[c]}\n{params[c]:,} params" for c in seed_ic.columns], fontsize=8)
    ax.set_ylabel("Mean rank IC (first dot = ensemble, then seeds)")
    ax.set_title("Seed dispersion and size")
    fig.suptitle("Figure 67. Bayesian deep learning: ensembles, MC dropout and the width of the forecast", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does the disagreement between seeds and dropout masks tell us anything about the forecast's spread that the EWMA volatility does not?", 67)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 33: the deep-forecasting family")
    parser.add_argument("--backend", default=None)
    args = parser.parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = (cfg.get("frontier", {}) or {})["deep_family"]
    s27 = cfg.get("deeplearning", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 33 | deep-forecasting family, graph attention, Bayesian deep learning (Generation 5)")
    logger.info("=" * 72)
    fdr = float(node["decisions"]["fdr"])
    horizon = int(cfg.get("forecasting.horizon_days", 21))
    embargo = int(s27["task"]["walk_forward"]["embargo_days"])
    window = int(s27["task"]["window"])
    backend = args.backend or str((cfg.get("distributed.parallel", {}) or {}).get("backend", "joblib"))
    frozen = pd.read_csv(context.tables / "stage19_predictions_price_only.csv", parse_dates=["origin"])

    market = context.market_data()
    prices = market.prices
    returns = prices.pct_change()
    index = pd.DatetimeIndex(prices.index)
    assets = list(prices.columns)
    half = float(cfg.get("forecasting.gaussian.vol_halflife", 40))
    sigma_d = daily_sigma(returns, half)
    sigma_h = ewma_sigma(returns, half, horizon)
    forward = prices.shift(-horizon) / prices - 1.0
    k_nb = int(node["models"]["gat"]["k_neighbours"])

    step = 5
    days = np.arange(window, len(index) - horizon, step)
    day_grid, asset_grid = np.repeat(days, len(assets)), np.tile(np.arange(len(assets)), len(days))
    X, valid = normalised_windows(returns, sigma_d, day_grid, asset_grid, window)
    z = np.clip(forward.to_numpy()[day_grid, asset_grid] / sigma_h.to_numpy()[day_grid, asset_grid], -5.0, 5.0)
    rows = {"X": X, "valid": valid, "z": z, "day": day_grid, "asset": asset_grid}

    feats = price_features(prices)
    feat_cube = np.stack([f.to_numpy(dtype=float) for f in feats.values()], axis=2)           # (days, assets, 12)
    z_cube = np.clip(forward.to_numpy() / sigma_h.to_numpy(), -5.0, 5.0)
    graph_days = days
    day_data = {"day": graph_days, "F": feat_cube[graph_days], "z": z_cube[graph_days], "N": correlation_neighbours(returns, graph_days, window, k_nb)}

    test, gat_test = {}, {}
    for start, g in frozen.groupby("refit_day"):
        pos = np.array([index.get_loc(o) for o in g["origin"]])
        a = np.array([assets.index(x) for x in g["asset"]])
        Xt, ok = normalised_windows(returns, sigma_d, pos, a, window)
        test[int(start)] = {"X": Xt[ok], "asset_idx": a[ok], "origin": g["origin"].to_numpy()[ok], "asset": g["asset"].to_numpy()[ok]}
        uniq = np.unique(pos)
        slot = {p: i for i, p in enumerate(uniq)}
        gat_test[int(start)] = {"F": feat_cube[uniq], "N": correlation_neighbours(returns, uniq, window, k_nb), "row_day": np.array([slot[p] for p in pos]),
                                "row_asset": a, "origin": g["origin"].to_numpy(), "asset": g["asset"].to_numpy()}
    shared = SimpleNamespace(cfg=node, rows=rows, days=day_data, test=test, gat_test=gat_test, horizon=horizon, embargo=embargo, n_assets=len(assets))
    tasks = [{"start": s, "model": m} for s in sorted(test) for m in MODELS]
    logger.info("%d refits x %d models = %d tasks; %d training rows, %d graph days", len(test), len(MODELS), len(tasks), int(valid.sum()), len(graph_days))
    t0 = time.perf_counter()
    results = run_tasks(partial(refit_task, shared=shared), tasks, backend, int((cfg.get("distributed.parallel", {}) or {}).get("n_jobs", -1)), 1)
    logger.info("trained in %.0fs on backend %s", time.perf_counter() - t0, backend)

    # ------------------------------------------------------------ assemble: one frame per model on the Stage 19 rows
    base = frozen[KEYS + ["sigma", "target", "mu", "mu_benchmark"]].rename(columns={"mu": "mu_ridge"})
    frames = {}
    for kind in MODELS:
        parts, seed_parts = [], {}
        for r in (x for x in results if x["model"] == kind):
            df = pd.DataFrame({"origin": r["origin"], "asset": r["asset"], "mu_z": r["mu_z"], "epi_var": r["epistemic_var"]})
            parts.append(df)
            for k, sp in enumerate(r["seed_preds"]):
                seed_parts.setdefault(k, []).append(pd.DataFrame({"origin": r["origin"], "asset": r["asset"], f"mu_z_s{k}": sp}))
        f = pd.concat(parts, ignore_index=True).merge(base, on=KEYS)
        for k, ps in seed_parts.items():
            f = f.merge(pd.concat(ps, ignore_index=True), on=KEYS)
        f["mu"] = f["mu_z"] * f["sigma"]
        f["sigma_wide"] = f["sigma"] * np.sqrt(1.0 + f["epi_var"])
        frames[kind] = f.dropna(subset=["mu", "sigma", "target", "mu_benchmark", "mu_ridge"])
        context.save_table(frames[kind], f"stage33_predictions_{kind}.csv", index=False)
    common = frames[MODELS[0]][KEYS]
    for kind in MODELS[1:]:
        common = common.merge(frames[kind][KEYS], on=KEYS)
    logger.info("common scoring rows: %d (origins %d)", len(common), common["origin"].nunique())
    frames = {k: f.merge(common, on=KEYS).sort_values(KEYS).reset_index(drop=True) for k, f in frames.items()}

    rows_out, adv_by_origin = [], {}
    for kind in MODELS:
        f = frames[kind]
        c_model, c_hist, c_ridge = (by_origin(f, gaussian_scores(f.assign(m=f[col]), "m", "sigma")) for col in ("mu", "mu_benchmark", "mu_ridge"))
        t_hist = diebold_mariano(c_model, c_hist, lag=0, alternative="two-sided")
        t_ridge = diebold_mariano(c_model, c_ridge, lag=0, alternative="two-sided")
        adv_by_origin[kind] = c_hist - c_model
        rows_out.append({"model": kind, "n_origins": len(c_model), "n_rows": len(f), "mean_crps_model": float(c_model.mean()), "mean_crps_hist_mean": float(c_hist.mean()),
                         "mean_crps_difference": float(t_hist["mean_loss_difference"]), "statistic": float(t_hist["statistic"]), "p_value": float(t_hist["p_value"]),
                         "mean_crps_ridge": float(c_ridge.mean()), "diff_vs_ridge": float(t_ridge["mean_loss_difference"]), "p_vs_ridge": float(t_ridge["p_value"]),
                         "mse_ratio_vs_hist": float(((f["target"] - f["mu"]) ** 2).mean() / ((f["target"] - f["mu_benchmark"]) ** 2).mean()),
                         "mean_ic": mean_ic(f, "mu"), "mean_ic_hist_mean": mean_ic(f, "mu_benchmark"), "mean_abs_mu": float(f["mu"].abs().mean())})
    table = pd.DataFrame(rows_out).set_index("model")
    table["bh_significant"] = benjamini_hochberg(table["p_value"], fdr)
    table["passes"] = table["bh_significant"] & (table["mean_crps_difference"] < 0)
    context.save_table(table, "stage33_forecast_tests.csv")
    logger.info("against the historical mean (CRPS, lower is better):\n%s",
                table[["n_origins", "mean_crps_model", "mean_crps_hist_mean", "mean_crps_difference", "p_value", "bh_significant", "mean_ic", "diff_vs_ridge", "p_vs_ridge"]].round(6).to_string())
    h_frontier = bool(table["passes"].any())

    # ------------------------------------------------------------ reported-not-judged: the Stage 27 models on the same rows
    s27_rows = []
    for kind in ("patchtst", "tsmixer"):
        p = pd.read_csv(context.tables / f"stage27_predictions_{kind}.csv", parse_dates=["origin"]).rename(columns={"mu": "mu_s27"})[KEYS + ["mu_s27"]]
        for m in MODELS:
            f = frames[m].merge(p, on=KEYS)
            d = by_origin(f, gaussian_scores(f.assign(a=f["mu"]), "a", "sigma") - gaussian_scores(f.assign(a=f["mu_s27"]), "a", "sigma"))
            tt = diebold_mariano(d, pd.Series(0.0, index=d.index), lag=0, alternative="two-sided")
            s27_rows.append({"model": m, "against": kind, "n_origins": len(d), "mean_crps_difference_model_minus_other": float(tt["mean_loss_difference"]), "p_value": float(tt["p_value"])})
    context.save_table(pd.DataFrame(s27_rows), "stage33_vs_stage27.csv", index=False)

    # ------------------------------------------------------------ Bayesian: does the epistemic widening help the best model?
    best = str(table["mean_crps_difference"].idxmin())
    f = frames[best]
    d_plain = by_origin(f, gaussian_scores(f, "mu", "sigma"))
    d_wide = by_origin(f, gaussian_scores(f, "mu", "sigma_wide"))
    t_bayes = diebold_mariano(d_wide, d_plain, lag=0, alternative="two-sided")
    h_bayesian = bool(t_bayes["p_value"] <= fdr and t_bayes["mean_loss_difference"] < 0)
    logger.info("best model by primary-benchmark CRPS difference: %s; widened - plain CRPS %.6f, p = %.4f -> %s", best, t_bayes["mean_loss_difference"], t_bayes["p_value"], h_bayesian)
    widening_rows, coverage_rows, ratio = [], {}, {}
    for kind in MODELS:
        g = frames[kind]
        dw, dp = by_origin(g, gaussian_scores(g, "mu", "sigma_wide")), by_origin(g, gaussian_scores(g, "mu", "sigma"))
        tt = diebold_mariano(dw, dp, lag=0, alternative="two-sided")
        row = {"model": kind, "mean_crps_plain": float(dp.mean()), "mean_crps_widened": float(dw.mean()), "mean_crps_difference": float(tt["mean_loss_difference"]), "p_value": float(tt["p_value"]),
               "mean_sigma_ratio": float((g["sigma_wide"] / g["sigma"]).mean()), "mean_epistemic_var_z": float(g["epi_var"].mean()),
               "corr_epistemic_vs_abs_error": float(spearmanr(g["epi_var"], ((g["target"] - g["mu"]).abs() / g["sigma"]))[0])}
        for level in (0.5, 0.9):
            q = norm.ppf(0.5 + level / 2)
            for col, tag in (("sigma", "ewma"), ("sigma_wide", "widened")):
                row[f"coverage_{int(level * 100)}_{tag}"] = float(((g["target"] - g["mu"]).abs() <= q * g[col]).mean())
        widening_rows.append(row)
        ratio[kind] = (g["sigma_wide"] / g["sigma"]).groupby(g["origin"]).mean()
    wide_table = pd.DataFrame(widening_rows).set_index("model")
    context.save_table(wide_table, "stage33_bayesian.csv")
    logger.info("epistemic widening by model:\n%s", wide_table.round(4).to_string())
    cov_best = wide_table.loc[best]
    coverage = pd.DataFrame({"without": [cov_best["coverage_50_ewma"], cov_best["coverage_90_ewma"]], "with": [cov_best["coverage_50_widened"], cov_best["coverage_90_widened"]]},
                            index=["50% interval", "90% interval"])

    # ------------------------------------------------------------ seed dispersion, size, training log
    seed_ic = {}
    for kind in MODELS:
        g = frames[kind]
        seed_ic[kind] = [table.loc[kind, "mean_ic"]] + [mean_ic(g.assign(mu=g[c] * g["sigma"]), "mu") for c in sorted(c for c in g.columns if c.startswith("mu_z_s"))]
    seed_frame = pd.DataFrame(seed_ic)
    context.save_table(seed_frame, "stage33_seed_ic.csv", index=False)
    params = {k: parameter_count(k, len(assets), family={a: b for a, b in node["models"][k].items() if a != "description"}) for k in MODELS if k in FAMILY}
    g_node = node["models"]["gat"]
    params["gat"] = sum(p.numel() for p in build_graph_network(len(assets), feat_cube.shape[2], hidden=g_node["hidden"], heads=g_node["heads"], dropout=g_node["dropout"]).parameters())
    context.save_table(pd.DataFrame({"parameters": params}), "stage33_parameter_counts.csv")
    log = pd.DataFrame([{"model": r["model"], "refit_day": r["start"], "n_train": r["n_train"], "n_val": r["n_val"], "best_epochs": str(r["best_epochs"])} for r in results])
    context.save_table(log, "stage33_training_log.csv", index=False)

    # ------------------------------------------------------------ figures
    adv = pd.DataFrame(adv_by_origin)
    figure_scores(adv.cumsum(), table, adv.groupby(adv.index.year).mean(), context.figure("fig66_frontier_scores.png"))
    figure_bayes(coverage, pd.DataFrame(ratio), seed_frame, params, context.figure("fig67_frontier_bayesian.png"))

    context.registry.log(
        "At least one of N-BEATS, N-HiTS, a TimeMixer-style mixer or a graph attention network forecasts 21-day ETF returns with a significantly lower mean CRPS "
        "than the asset's own historical mean (same volatility forecast, same rows; Benjamini-Hochberg across the four).",
        stage=STAGE, parameters={"models": MODELS, "seeds": node["training"]["ensemble_seeds"], "refits": len(test), "fdr": fdr},
        results={**{f"crps_difference_{k}": float(table.loc[k, "mean_crps_difference"]) for k in MODELS}, **{f"p_value_{k}": float(table.loc[k, "p_value"]) for k in MODELS},
                 **{f"mean_ic_{k}": float(table.loc[k, "mean_ic"]) for k in MODELS}, "n_origins": int(table["n_origins"].iloc[0]), "n_rows": int(table["n_rows"].iloc[0])},
        decision="retain" if h_frontier else "reject", test_period="2011-01 onward, monthly origins, 15 ETFs",
        notes="Primary benchmark is the historical mean, declared after Stage 27 showed the ridge was the weak benchmark. The Stage 19 ridge and the Stage 27 models are reported, not judged.",
    )
    context.registry.log(
        f"For the best model ({best}), widening the forecast standard deviation by the deep-ensemble and MC-dropout epistemic variance lowers the mean CRPS.",
        stage=STAGE, parameters={"model": best, "mc_samples": int(node["bayesian"]["mc_samples"]), "seeds": node["training"]["ensemble_seeds"]},
        results={"mean_crps_plain": float(d_plain.mean()), "mean_crps_widened": float(d_wide.mean()), "difference": float(t_bayes["mean_loss_difference"]), "p_value": float(t_bayes["p_value"]),
                 "mean_sigma_ratio": float(cov_best["mean_sigma_ratio"])},
        decision="retain" if h_bayesian else "reject", test_period="2011-01 onward, monthly origins, 15 ETFs",
        notes="The model is the one with the most negative CRPS difference against the historical mean, chosen whether or not that difference was significant. The mean forecast is unchanged.",
    )
    logger.info("STAGE 33 complete: h_frontier=%s h_bayesian=%s", h_frontier, h_bayesian)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
